#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from risk_aware_girpo.schema import write_jsonl
from risk_aware_girpo.splits import grouped_stratified_split


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    return value if isinstance(value, list) else value["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--official-split-field", default="split")
    args = parser.parse_args()
    rows = load_rows(args.input)
    official = {str(row.get(args.official_split_field, "")).lower() for row in rows}
    official.discard("")
    if official and official <= {"train", "validation", "test"}:
        split_rows = [{**row, "split": str(row[args.official_split_field]).lower()} for row in rows]
    else:
        split_rows = grouped_stratified_split(rows, seed=args.seed)
    for split in ("train", "validation", "test"):
        write_jsonl(args.output_dir / f"{split}.jsonl", (row for row in split_rows if row["split"] == split))


if __name__ == "__main__":
    main()
