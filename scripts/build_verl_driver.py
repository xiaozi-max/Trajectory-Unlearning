#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            value = json.load(handle)
            rows = value if isinstance(value, list) else value["tasks"]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return [row if isinstance(row, dict) else {"task_id": str(row)} for row in rows]


def driver_rows(rows: list[dict], seed: int) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        task_id = str(row["task_id"])
        output.append(
            {
                "data_source": "mt_agentrisk",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"index": index},
                "env_kwargs": {"task_id": task_id, "seed": seed + index},
            }
        )
    return output


def write_driver(manifest: Path, output: Path, seed: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(driver_rows(load_manifest(manifest), seed)), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    write_driver(args.train_manifest, args.output_dir / "train.parquet", args.seed)
    write_driver(args.val_manifest, args.output_dir / "validation.parquet", args.seed + 1_000_000)


if __name__ == "__main__":
    main()
