from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from risk_aware_girpo.schema import read_jsonl


def _pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    for attempt in read_jsonl(path):
        for step in attempt.get("steps", []):
            pairs.append((str(step["prompt"]), str(step["model_response"])))
    if not pairs:
        raise ValueError(f"no prompt-response pairs in {path}")
    return pairs


class PairDataset(Dataset):
    def __init__(self, forget_path: str, retain_path: str, tokenizer, max_length: int):
        forget = _pairs(forget_path)
        retain = _pairs(retain_path)
        retain_cycle = itertools.cycle(retain)
        self.rows = [(item, next(retain_cycle)) for item in forget]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, pair: tuple[str, str]) -> dict[str, list[int]]:
        prompt, response = pair
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            response_ids = response_ids + [eos]
        input_ids = (prompt_ids + response_ids)[-self.max_length :]
        prompt_kept = max(0, len(input_ids) - len(response_ids))
        labels = [-100] * prompt_kept + input_ids[prompt_kept:]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def __getitem__(self, index: int) -> dict[str, Any]:
        forget, retain = self.rows[index]
        return {"forget": self._encode(forget), "retain": self._encode(retain)}


@dataclass
class PairCollator:
    tokenizer: Any

    def _pad(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        pad_id = self.tokenizer.pad_token_id
        values = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            missing = width - len(row["input_ids"])
            values["input_ids"].append(row["input_ids"] + [pad_id] * missing)
            values["attention_mask"].append(row["attention_mask"] + [0] * missing)
            values["labels"].append(row["labels"] + [-100] * missing)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in values.items()}

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {domain: self._pad([row[domain] for row in rows]) for domain in ("forget", "retain")}


def _token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), safe_labels.view(-1), reduction="none"
    ).view_as(safe_labels)
    return nll, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--forget", required=True)
    parser.add_argument("--retain", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    dataset = PairDataset(args.forget, args.retain, tokenizer, args.max_length)

    class SimNPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            forget_outputs = model(**inputs["forget"])
            forget_nll, forget_mask = _token_nll(forget_outputs.logits, inputs["forget"]["labels"])
            lengths = forget_mask.sum(-1).clamp_min(1)
            normalized = (forget_nll * forget_mask).sum(-1) / lengths - args.delta
            forget_loss = -F.logsigmoid(args.beta * normalized).mean() * 2.0 / args.beta

            retain_outputs = model(**inputs["retain"])
            retain_nll, retain_mask = _token_nll(retain_outputs.logits, inputs["retain"]["labels"])
            retain_loss = (retain_nll * retain_mask).sum() / retain_mask.sum().clamp_min(1)
            loss = args.gamma * forget_loss + args.alpha * retain_loss
            return (loss, forget_outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
    )
    trainer = SimNPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PairCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
