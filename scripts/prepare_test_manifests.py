#!/usr/bin/env python3
"""Build held-out harmful + benign test manifests for terminal/filesystem only.

No official train/validation/test split exists in MT-AgentRisk (the released
CSVs have no `split` column), so we hold out harmful tasks that never appear
in the current train/validation manifests, and use the dataset's own
`workspaces/benign_tasks/` as the over-refusal test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HARMFUL = {
    "terminal": ("terminal/multi_turn_tasks", "multi-turn_", "multi-turn-"),
    "filesystem": ("filesystem/multi-turn-tasks", "multi-turn-",),
}
BENIGN = {
    "terminal": "benign_tasks/benign_tasks_terminal",
    "filesystem": "benign_tasks/benign_tasks_filesystem",
}


def read_used(manifest_dir: Path) -> set[str]:
    used: set[str] = set()
    for path in manifest_dir.glob("*.jsonl"):
        if path.name.endswith("_test.jsonl") or path.name.startswith("benign_"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                used.add(str(row["task_path"]))
    return used


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"[manifest] {path.name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=10)
    args = parser.parse_args()

    used = read_used(args.manifest_dir)
    harmful_rows: list[dict] = []
    benign_rows: list[dict] = []

    for domain, spec in HARMFUL.items():
        relative = spec[0]
        prefixes = spec[1:]
        root = args.tasks_root / relative
        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and any(p.name.startswith(prefix) for prefix in prefixes)
        )
        held_out = [p for p in candidates if f"{relative}/{p.name}" not in used]
        chosen = held_out[: args.per_domain]
        rows = [
            {
                "task_id": p.name,
                "task_path": f"{relative}/{p.name}",
                "is_harmful": True,
                "is_benign": False,
                "split": "test",
                "tool_domain": domain,
            }
            for p in chosen
        ]
        write_jsonl(args.manifest_dir / f"{domain}_test.jsonl", rows)
        harmful_rows.extend(rows)

    for domain, relative in BENIGN.items():
        root = args.tasks_root / relative
        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p.name.startswith("benign_task.") or p.name.startswith("task."))
        )
        rows = [
            {
                "task_id": f"benign_{domain}_{index:03d}",
                "task_path": f"{relative}/{p.name}",
                "is_harmful": False,
                "is_benign": True,
                "split": "test",
                "tool_domain": domain,
            }
            for index, p in enumerate(candidates, 1)
        ]
        write_jsonl(args.manifest_dir / f"benign_{domain}_test.jsonl", rows)
        benign_rows.extend(rows)

    write_jsonl(args.manifest_dir / "test.jsonl", harmful_rows)
    write_jsonl(args.manifest_dir / "benign_test.jsonl", benign_rows)
    print(
        f"[manifest] harmful test={len(harmful_rows)} benign test={len(benign_rows)} "
        f"excluded_used={len(used)}"
    )


if __name__ == "__main__":
    main()
