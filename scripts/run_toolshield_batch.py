#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    return value if isinstance(value, list) else value["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolshield-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--agent-config-file", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--server-hostname", default="localhost")
    args = parser.parse_args()

    run_eval = args.toolshield_root / "agentrisk" / "run_eval.py"
    if not run_eval.exists():
        raise FileNotFoundError(run_eval)
    for task in load_tasks(args.manifest):
        task_id = str(task["task_id"] if isinstance(task, dict) else task)
        relative_path = str(task.get("task_path", task_id)) if isinstance(task, dict) else task_id
        task_dir = args.tasks_root / relative_path
        for seed in args.seeds:
            output = args.outputs_root / args.model_id / f"seed-{seed}" / task_id
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(run_eval),
                "--task-path",
                str(task_dir),
                "--outputs-path",
                str(output),
                "--server-hostname",
                args.server_hostname,
                "--agent-llm-config",
                "agent",
                "--env-llm-config",
                "env",
                "--agent-llm-config-file",
                str(args.agent_config_file),
                "--env-llm-config-file",
                str(args.agent_config_file),
            ]
            attempt_env = os.environ.copy()
            attempt_env["PYTHONHASHSEED"] = str(seed)
            attempt_env["MT_AGENTRISK_SEED"] = str(seed)
            subprocess.run(command, cwd=args.toolshield_root, env=attempt_env, check=True)


if __name__ == "__main__":
    main()
