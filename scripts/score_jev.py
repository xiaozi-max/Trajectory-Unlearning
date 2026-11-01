#!/usr/bin/env python3
from __future__ import annotations

import argparse

from risk_aware_girpo import jev
from risk_aware_girpo.schema import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--steps", action="store_true")
    parser.add_argument("--attempts", action="store_true")
    args = parser.parse_args()
    if not args.steps and not args.attempts:
        parser.error("select --steps and/or --attempts")
    client = jev.JEVClient.from_env(args.cache_dir)
    output = []
    for row in read_jsonl(args.input):
        if args.steps:
            history = []
            for step in row["steps"]:
                payload = jev.step_request(
                    model=args.model,
                    task_goal={"instructions": row.get("task_instructions", []), "boundaries": row.get("boundaries")},
                    history_before_action=history,
                    current_model_response=step["model_response"],
                    current_action=step["action"],
                )
                step["jev_weight"] = jev.parse_step_weight(client.score(payload))
                history.extend(
                    [
                        {"role": "agent", "response": step["model_response"], "action": step["action"]},
                        {"role": "tool", "observation": step.get("observation", "")},
                    ]
                )
        if args.attempts:
            payload = jev.attempt_request(
                model=args.model,
                task_instructions=row.get("task_instructions", []),
                trajectory=row["steps"],
                final_state=row.get("terminal_state"),
                evaluator_evidence=row.get("evaluator_evidence"),
            )
            row["judge_output"] = jev.parse_attempt(client.score(payload))
        output.append(row)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
