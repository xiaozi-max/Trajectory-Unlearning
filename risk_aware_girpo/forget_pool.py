from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from risk_aware_girpo.schema import read_jsonl


def load_forget_pool(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load verified train COMPLETE attempts and preserve multiple attempts per task."""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in read_jsonl(path):
        if row.get("split") != "train":
            raise ValueError(f"forget pool may only contain train rows: {row.get('attempt_id')}")
        outcome = (row.get("judge_output") or {}).get("outcome")
        unsafe = float((row.get("judge_output") or {}).get("unsafe_intermediate", 0.0))
        if outcome != "COMPLETE" and unsafe <= 0.5:
            continue
        trajectory_id = str(row.get("trajectory_id") or row.get("attempt_id") or "")
        if not trajectory_id or trajectory_id in seen_ids:
            raise ValueError(f"missing or duplicate trajectory_id: {trajectory_id!r}")
        seen_ids.add(trajectory_id)
        steps = []
        for index, step in enumerate(row.get("steps", [])):
            weight = float(step.get("jev_weight", step.get("step_weight", 1.0)))
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"invalid weight in {trajectory_id} step {index}: {weight}")
            steps.append(
                {
                    "step_id": int(step.get("step_id", index)),
                    "turn_id": str(step.get("turn_id", "")),
                    "prompt": str(step["prompt"]),
                    "model_response": str(step["model_response"]),
                    "action": step.get("action"),
                    "observation": str(step.get("observation", "")),
                    "step_weight": weight,
                }
            )
        if not steps:
            raise ValueError(f"trajectory {trajectory_id} contains no usable steps")
        pool[str(row["task_id"])].append(
            {
                "task_id": str(row["task_id"]),
                "trajectory_id": trajectory_id,
                "steps": steps,
            }
        )
    return dict(pool)


class RoundRobinForgetPool:
    def __init__(self, trajectories: dict[str, list[dict[str, Any]]]):
        self.trajectories = trajectories
        self._offsets: dict[str, int] = defaultdict(int)

    def next(self, task_id: str) -> dict[str, Any] | None:
        choices = self.trajectories.get(task_id)
        if not choices:
            return None
        offset = self._offsets[task_id]
        self._offsets[task_id] = offset + 1
        return choices[offset % len(choices)]
