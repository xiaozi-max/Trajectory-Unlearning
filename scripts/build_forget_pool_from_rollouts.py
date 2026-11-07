#!/usr/bin/env python3
"""Convert veRL rollout dumps into unified attempts + a verified forget pool.

Dangerousness is decided from the environment's *explicit* outcome evidence
(``terminal_state`` / ``outcome`` written by the gateway), NOT from a guessed
command keyword and NOT from JEV. JEV only runs later, offline, via
``scripts/score_jev.py`` to (a) confirm the attempt-level judgement and (b)
assign per-step weights to dangerous attempts.

Outputs:
  * ``--output``     : dangerous attempts only (forget pool for GiRPO)
  * ``--all-output`` : every attempt, labelled, for JEV scoring / auditing
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from risk_aware_girpo.gateway import parse_actor_action
from risk_aware_girpo.schema import write_jsonl

OUTCOMES = {"COMPLETE", "REJECT", "FAILED"}


def _decode(value):
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _outcome_of(rows: list[dict]) -> tuple[str, dict, bool]:
    """Return (outcome, terminal_state, is_harmful) from explicit env fields."""
    terminal = {}
    for row in reversed(rows):
        candidate = _decode(row.get("terminal_state"))
        if isinstance(candidate, dict) and candidate.get("outcome"):
            terminal = candidate
            break
    outcome = str(terminal.get("outcome") or "").upper()
    if outcome not in OUTCOMES:
        for row in reversed(rows):
            candidate = str(row.get("outcome") or "").upper()
            if candidate in OUTCOMES:
                outcome = candidate
                break
    is_harmful = terminal.get("is_harmful")
    if is_harmful is None:
        is_harmful = _decode(rows[-1].get("is_harmful")) or True
    is_harmful = bool(is_harmful) if not isinstance(is_harmful, str) else is_harmful.lower() != "false"
    if outcome not in OUTCOMES:
        # Backward compatibility with older dumps that lack provenance fields.
        reward = float(rows[0].get("episode_rewards", 0.0))
        outcome = "COMPLETE" if reward < 0 else ("REJECT" if reward > 0 else "FAILED")
    return outcome, terminal, is_harmful


def _load_instructions(tasks_root: Path | None, task_id: str, task_path: str | None) -> list[str]:
    if not tasks_root:
        return []
    candidates = []
    if task_path:
        candidates.append(tasks_root / task_path)
    candidates.append(tasks_root / "terminal" / "multi_turn_tasks" / task_id)
    candidate = next((p for p in candidates if p.is_dir()), None)
    if candidate is None:
        return []
    instructions = []
    turns_file = candidate / "turns.yml"
    if turns_file.exists():
        try:
            import yaml

            manifest = yaml.safe_load(turns_file.read_text(encoding="utf-8")) or {}
            for turn in manifest.get("turns", []):
                path = candidate / str(turn["instruction_file"])
                if path.exists():
                    instructions.append(path.read_text(encoding="utf-8").strip())
        except (ImportError, KeyError, OSError):
            instructions.extend(p.read_text(encoding="utf-8").strip()
                                for p in sorted(candidate.glob("task-turn-*.md")))
    elif (candidate / "task.md").exists():
        instructions.append((candidate / "task.md").read_text(encoding="utf-8").strip())
    return instructions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, default=None)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-Coder-3B-Instruct"))
    parser.add_argument("--tasks-root", type=Path, default=os.environ.get("TASKS_ROOT"))
    parser.add_argument("--train-manifest", type=Path, default=os.environ.get("TRAIN_MANIFEST"))
    args = parser.parse_args()

    task_paths: dict[str, str] = {}
    if args.train_manifest and args.train_manifest.exists():
        for line in args.train_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                task_paths[str(row["task_id"])] = str(row.get("task_path", ""))

    groups: dict[str, list[dict]] = defaultdict(list)
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[str(row["traj_uid"])].append(row)

    attempts = []
    for uid, rows in groups.items():
        rows.sort(key=lambda item: int(item.get("step_num", 0)))
        reward = float(rows[0].get("episode_rewards", 0.0))
        task_id = str(rows[0]["task_id"])
        outcome, terminal, is_harmful = _outcome_of(rows)
        evidence = _decode(rows[-1].get("evaluator_evidence")) or terminal.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {"raw": str(evidence)}

        # A dangerous trajectory is a harmful task that was actually completed,
        # or a trajectory with a negative terminal reward (legacy dumps).
        dangerous = bool(is_harmful) and (outcome == "COMPLETE" or reward < 0)

        steps = []
        for index, row in enumerate(rows):
            try:
                action = parse_actor_action(str(row["output"]))
            except ValueError:
                action = None
            observation = rows[index + 1].get("obs_text", "") if index + 1 < len(rows) else ""
            steps.append(
                {
                    "step_id": index,
                    "turn_id": str(row.get("step_id", index)),
                    "prompt": str(row["input"]),
                    "model_response": str(row["output"]),
                    "action": action,
                    "observation": str(observation),
                    "jev_weight": 1.0,
                }
            )

        attempts.append(
            {
                "task_id": task_id,
                "split": "train",
                "attempt_id": uid,
                "trajectory_id": uid,
                "seed": int(rows[0].get("sample_id", 0)),
                "model_id": args.model_id,
                "adapter_id": None,
                "task_instructions": _load_instructions(args.tasks_root, task_id, task_paths.get(task_id)),
                "steps": steps,
                "judge_output": {
                    "outcome": outcome,
                    "unsafe_intermediate": 1.0 if dangerous else 0.0,
                },
                "terminal_state": terminal or {"outcome": outcome, "is_harmful": bool(is_harmful)},
                "evaluator_evidence": {
                    **evidence,
                    "source": "gateway_success_patterns",
                    "outcome": outcome,
                    "is_harmful": bool(is_harmful),
                },
                "metadata": {
                    "episode_reward": reward,
                    "source": "online_rollout",
                    "dangerous": dangerous,
                    "judge_source": "gateway_code",  # JEV overwrites this later
                    "jev_status": "not_scored",
                },
            }
        )

    write_jsonl(args.output, [row for row in attempts if row["metadata"]["dangerous"]])
    if args.all_output:
        write_jsonl(args.all_output, attempts)
    dangerous_count = sum(1 for row in attempts if row["metadata"]["dangerous"])
    print(
        f"[build_forget_pool] trajectories={len(attempts)} dangerous={dangerous_count} "
        f"pool={args.output} all={args.all_output or '-'}"
    )
    if not dangerous_count:
        raise SystemExit(
            "no dangerous (harmful COMPLETE) trajectories found; collect broader base-model "
            "rollouts first. REJECT/FAILED attempts are intentionally NOT injected."
        )


if __name__ == "__main__":
    main()
