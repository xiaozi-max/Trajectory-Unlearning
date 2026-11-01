#!/usr/bin/env python3
from __future__ import annotations

import argparse

from risk_aware_girpo.schema import write_jsonl
from risk_aware_girpo.toolshield import convert_attempt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    row = convert_attempt(
        task_dir=args.task_dir,
        output_dir=args.output_dir,
        split=args.split,
        attempt_id=args.attempt_id,
        seed=args.seed,
        model_id=args.model_id,
        adapter_id=args.adapter_id,
    )
    write_jsonl(args.output, [row])


if __name__ == "__main__":
    main()
