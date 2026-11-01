from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_ACTION_NAMES = ("action", "tool_call", "tool", "command")
_OBSERVATION_NAMES = ("observation", "tool_output", "output", "content")


def _read_jsonish(path: Path) -> Any:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _first(mapping: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _events(trajectory: Any) -> list[Mapping[str, Any]]:
    if isinstance(trajectory, list):
        return [item for item in trajectory if isinstance(item, Mapping)]
    if isinstance(trajectory, Mapping):
        for key in ("events", "history", "trajectory", "steps"):
            value = trajectory.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
    raise ValueError("trajectory does not contain a structured event list")


def _turn_id(event: Mapping[str, Any], fallback: str) -> str:
    for key in ("turn_id", "turn", "turn_index"):
        if key in event:
            return str(event[key])
    return fallback


def trajectory_to_steps(trajectory: Any) -> list[dict[str, Any]]:
    """Convert structured OpenHands/ToolShield events to decision-step rows.

    Exact prompt/response fields are mandatory. This intentionally rejects the
    old string-only state dump because it cannot reproduce token IDs/log-probs.
    """
    events = _events(trajectory)
    steps: list[dict[str, Any]] = []
    pending_observation = ""
    for event in events:
        kind = str(event.get("type") or event.get("event_type") or event.get("kind") or "").lower()
        if "observation" in kind or "tool_result" in kind:
            pending_observation = str(_first(event, _OBSERVATION_NAMES, ""))
            if steps:
                steps[-1]["observation"] = pending_observation
            continue
        prompt = _first(event, ("prompt", "exact_prompt", "llm_prompt"))
        response = _first(event, ("model_response", "response", "llm_response"))
        action = _first(event, _ACTION_NAMES)
        if prompt is None or response is None or action is None:
            continue
        step_id = len(steps)
        steps.append(
            {
                "step_id": step_id,
                "turn_id": _turn_id(event, str(step_id)),
                "prompt": str(prompt),
                "model_response": str(response),
                "action": action,
                "observation": str(_first(event, _OBSERVATION_NAMES, pending_observation)),
                "jev_weight": 1.0,
            }
        )
        pending_observation = ""
    if not steps:
        raise ValueError(
            "no reproducible decision steps found; patch ToolShield capture to save exact prompt, response, and action"
        )
    return steps


def gather_instructions(task_dir: str | Path) -> list[str]:
    root = Path(task_dir)
    files = [root / "task-intro.md", root / "task.md", *sorted(root.glob("task-turn-*.md"))]
    return [path.read_text(encoding="utf-8").strip() for path in files if path.exists()]


def convert_attempt(
    *,
    task_dir: str | Path,
    output_dir: str | Path,
    split: str,
    attempt_id: str,
    seed: int,
    model_id: str,
    adapter_id: str | None = None,
) -> dict[str, Any]:
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)
    task_id = task_dir.name
    trajectory = _read_jsonish(output_dir / f"traj_{task_id}.json")
    state = _read_jsonish(output_dir / f"state_{task_id}.json")
    evaluator = _read_jsonish(output_dir / f"eval_{task_id}.json")
    return {
        "task_id": task_id,
        "split": split,
        "attempt_id": attempt_id,
        "trajectory_id": f"{task_id}:{attempt_id}",
        "seed": seed,
        "model_id": model_id,
        "adapter_id": adapter_id,
        "task_instructions": gather_instructions(task_dir),
        "steps": trajectory_to_steps(trajectory),
        "terminal_state": state,
        "evaluator_evidence": evaluator,
        "judge_output": None,
    }


def parse_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()
