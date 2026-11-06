"""Reviewed, same-task MT-AgentRisk trajectory preferences and tokenization."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def build_pairs(attempts: Iterable[Mapping[str, Any]], specs: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Pair complete harmful-task rollouts using explicit human-reviewed IDs."""
    index = {}
    for row in attempts:
        key = (row.get("task_id"), row.get("attempt_id"))
        if key in index:
            raise ValueError(f"duplicate attempt: {key}")
        index[key] = row
    result = []
    seen = set()
    for spec in specs:
        task_id = spec["task_id"]
        chosen_id, rejected_id = spec["chosen_attempt_id"], spec["rejected_attempt_id"]
        key = (task_id, chosen_id, rejected_id)
        if not spec.get("reviewed") or key in seen or chosen_id == rejected_id:
            raise ValueError(f"pair must be unique and reviewed: {key}")
        seen.add(key)
        try:
            chosen, rejected = index[(task_id, chosen_id)], index[(task_id, rejected_id)]
        except KeyError as exc:
            raise ValueError(f"unknown attempt in pair: {key}") from exc
        if chosen.get("split") != "train" or rejected.get("split") != "train":
            raise ValueError(f"pair must use train attempts: {key}")
        if (chosen.get("judge_output") or {}).get("outcome") != "REJECT":
            raise ValueError(f"chosen must be verified safe REJECT: {key}")
        if (rejected.get("judge_output") or {}).get("outcome") != "COMPLETE":
            raise ValueError(f"rejected must be verified unsafe COMPLETE: {key}")
        for row in (chosen, rejected):
            if not row.get("evaluator_evidence") or not row.get("steps"):
                raise ValueError(f"missing evaluator evidence or steps: {key}")
            if any(not all(field in step for field in ("prompt", "model_response", "action", "observation"))
                   for step in row["steps"]):
                raise ValueError(f"incomplete trajectory steps: {key}")
        result.append({"task_id": task_id, "chosen": chosen, "rejected": rejected})
    if not result:
        raise ValueError("no reviewed trajectory pairs")
    return result


def encode_trajectory(attempt: Mapping[str, Any], tokenizer: Any, max_length: int) -> list[dict]:
    """Keep every response intact; truncate only the left side of its exact prompt."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    encoded = []
    for step in attempt["steps"]:
        prompt = tokenizer(step["prompt"], add_special_tokens=False)["input_ids"]
        response = tokenizer(step["model_response"], add_special_tokens=False)["input_ids"]
        if not prompt or not response or len(response) >= max_length:
            raise ValueError("empty prompt/response or response exceeds max_length")
        prompt = prompt[-(max_length - len(response)):]
        encoded.append({"input_ids": prompt + response,
                        "attention_mask": [1] * (len(prompt) + len(response)),
                        "labels": [-100] * len(prompt) + response,
                        "response_tokens": len(response)})
    return encoded
