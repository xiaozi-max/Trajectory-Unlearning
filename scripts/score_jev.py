#!/usr/bin/env python3
"""Offline JEV scoring of unified attempt JSONL.

Pipeline (two stages, in this order):

1. ``--attempts`` asks JEV once per attempt for the final outcome
   (COMPLETE / REJECT / FAILED) and ``unsafe_intermediate``. This *replaces*
   any heuristic ``judge_output`` and is the authoritative dangerousness label.
2. ``--steps`` asks JEV once per step **only for dangerous attempts**
   (``COMPLETE`` or ``unsafe_intermediate > 0.5``), writing ``jev_weight``.
   Safe refusals / technical failures are never step-scored and keep
   ``jev_weight = 1.0``; they are filtered out by ``load_forget_pool`` anyway.

Use ``--score-all-steps`` to restore the old "score every step of every
attempt" behaviour (diagnostics only; not recommended for training data).
"""
from __future__ import annotations

import argparse

from risk_aware_girpo import jev
from risk_aware_girpo.schema import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--steps", action="store_true")
    parser.add_argument("--attempts", action="store_true")
    parser.add_argument(
        "--score-all-steps",
        action="store_true",
        help="score every step of every attempt (default: only dangerous attempts)",
    )
    args = parser.parse_args()
    if not args.steps and not args.attempts:
        parser.error("select --steps and/or --attempts")

    client = jev.JEVClient.from_env(args.cache_dir)
    output = []
    stats = {"attempts": 0, "dangerous": 0, "steps_scored": 0, "steps_skipped": 0}

    for row in read_jsonl(args.input):
        stats["attempts"] += 1

        # Stage 1: JEV attempt-level judgement must run BEFORE the step loop,
        # otherwise the dangerous filter would have to rely on a heuristic label.
        if args.attempts:
            payload = jev.attempt_request(
                model=args.model,
                task_instructions=row.get("task_instructions", []),
                trajectory=row["steps"],
                final_state=row.get("terminal_state"),
                evaluator_evidence=row.get("evaluator_evidence"),
            )
            row["judge_output"] = jev.parse_attempt(client.score(payload))
            row.setdefault("metadata", {})["jev_attempt_scored"] = True

        dangerous = jev.is_dangerous_attempt(row)
        if dangerous:
            stats["dangerous"] += 1

        if args.steps:
            history = []
            step_enabled = dangerous or args.score_all_steps
            for step in row["steps"]:
                if step_enabled:
                    payload = jev.step_request(
                        model=args.model,
                        task_goal={
                            "instructions": row.get("task_instructions", []),
                            "boundaries": row.get("boundaries"),
                        },
                        history_before_action=history,
                        current_model_response=step["model_response"],
                        current_action=step["action"],
                    )
                    step["jev_weight"] = jev.parse_step_weight(client.score(payload))
                    stats["steps_scored"] += 1
                else:
                    # No JEV call for safe attempts: keep the neutral default.
                    step["jev_weight"] = 1.0
                    stats["steps_skipped"] += 1
                history.extend(
                    [
                        {"role": "agent", "response": step["model_response"], "action": step["action"]},
                        {"role": "tool", "observation": step.get("observation", "")},
                    ]
                )
            row.setdefault("metadata", {})["jev_step_scored"] = bool(step_enabled)
            row.setdefault("metadata", {})["jev_dangerous"] = bool(dangerous)

        output.append(row)

    write_jsonl(args.output, output)
    print(
        f"[score_jev] attempts={stats['attempts']} dangerous={stats['dangerous']} "
        f"steps_scored={stats['steps_scored']} steps_skipped={stats['steps_skipped']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
