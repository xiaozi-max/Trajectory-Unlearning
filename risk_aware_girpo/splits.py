from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Iterable


def grouped_stratified_split(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = 20261101,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[dict[str, Any]]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    rows = [dict(row) for row in records]
    by_goal: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        goal_id = str(row.get("goal_id") or row.get("original_goal_id") or row["task_id"])
        by_goal[goal_id].append(index)

    by_stratum: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    for goal_id, indices in by_goal.items():
        tools = "+".join(sorted({str(rows[index].get("tool", "unknown")) for index in indices}))
        attacks = "+".join(sorted({str(rows[index].get("attack_type", "unknown")) for index in indices}))
        benign = any(bool(rows[index].get("is_benign", False)) for index in indices)
        by_stratum[(tools, attacks, benign)].append(goal_id)

    assignments: dict[str, str] = {}
    labels = ("train", "validation", "test")
    for stratum, goal_ids_for_stratum in sorted(by_stratum.items()):
        stratum_seed = int(hashlib.sha256(repr((seed, stratum)).encode()).hexdigest()[:16], 16)
        goal_ids = sorted(goal_ids_for_stratum)
        random.Random(stratum_seed).shuffle(goal_ids)
        counts = [int(len(goal_ids) * ratio) for ratio in ratios]
        while sum(counts) < len(goal_ids):
            counts[sum(counts) % 3] += 1
        cursor = 0
        for label, count in zip(labels, counts, strict=True):
            for goal_id in goal_ids[cursor : cursor + count]:
                assignments[goal_id] = label
            cursor += count

    for row in rows:
        goal_id = str(row.get("goal_id") or row.get("original_goal_id") or row["task_id"])
        row["split"] = assignments[goal_id]
    return rows
