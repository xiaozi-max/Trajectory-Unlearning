"""Offline trajectory-level DPO for reviewed MT-AgentRisk rollouts."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from risk_aware_girpo.schema import read_jsonl
from risk_aware_girpo.trajectory_dpo import build_pairs, encode_trajectory


class TrajectoryPairDataset(Dataset):
    def __init__(self, attempts, specs, tokenizer, max_length):
        self.pairs = build_pairs(attempts, specs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Fail before starting a costly model load.
        for pair in self.pairs:
            for side in ("chosen", "rejected"):
                encode_trajectory(pair[side], tokenizer, max_length)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        return {side: encode_trajectory(pair[side], self.tokenizer, self.max_length)
                for side in ("chosen", "rejected")}


class TrajectoryCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, pairs):
        rows, owners, sides = [], [], []
        for pair_index, pair in enumerate(pairs):
            for side_index, side in enumerate(("chosen", "rejected")):
                for step in pair[side]:
                    rows.append(step)
                    owners.append(pair_index)
                    sides.append(side_index)
        width = max(len(row["input_ids"]) for row in rows)
        def padded(field, fill):
            return torch.tensor([row[field] + [fill] * (width - len(row[field])) for row in rows])
        return {"input_ids": padded("input_ids", self.pad_token_id),
                "attention_mask": padded("attention_mask", 0),
                "labels": padded("labels", -100),
                "owner": torch.tensor(owners), "side": torch.tensor(sides),
                "pair_count": len(pairs)}


def trajectory_logps(model, batch, normalize=True):
    """Aggregate conditional response log probabilities over all steps of each rollout."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    token_logps = F.log_softmax(logits, -1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    step_logps = (token_logps * mask).sum(-1)
    target = batch["owner"] * 2 + batch["side"]
    totals = step_logps.new_zeros(batch["pair_count"] * 2).scatter_add(0, target, step_logps)
    if normalize:
        lengths = totals.new_zeros(batch["pair_count"] * 2).scatter_add(
            0, target, mask.sum(-1).to(totals.dtype))
        totals = totals / lengths.clamp_min(1)
    return totals.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", required=True, help="Train attempt JSONL with exact prompts and observations")
    parser.add_argument("--pairs", required=True, help="Reviewed pair manifest JSONL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.beta <= 0:
        parser.error("--beta must be positive")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TrajectoryPairDataset(list(read_jsonl(args.attempts)), list(read_jsonl(args.pairs)),
                                    tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))

    class TrajectoryDPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            policy = trajectory_logps(model, inputs)
            adapter_model = model.module if hasattr(model, "module") else model
            with torch.no_grad(), adapter_model.disable_adapter():
                reference = trajectory_logps(model, inputs)
            margins = (policy[:, 0] - policy[:, 1]) - (reference[:, 0] - reference[:, 1])
            loss = -F.logsigmoid(args.beta * margins).mean()
            return (loss, {"margin": margins.detach()}) if return_outputs else loss

    trainer = TrajectoryDPOTrainer(
        model=model,
        args=TrainingArguments(output_dir=args.output_dir, learning_rate=args.learning_rate,
                               num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
                               gradient_accumulation_steps=args.grad_accum, bf16=True,
                               gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                               logging_steps=1, save_strategy="epoch", report_to="none",
                               remove_unused_columns=False, seed=args.seed),
        train_dataset=dataset, data_collator=TrajectoryCollator(tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
