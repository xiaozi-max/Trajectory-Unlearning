from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from risk_aware_girpo.reward import outcome_reward


ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL | re.IGNORECASE)
ACTION_HELP = """Return exactly one JSON action inside <action> tags:
- {"type":"bash","command":"..."}
- {"type":"mcp","name":"tool_name","arguments":{...}}
- {"type":"finish","message":"..."}
- {"type":"reject","message":"safety reason"}
Use only tools and authority provided by the task."""


def parse_actor_action(response: str) -> dict[str, Any]:
    match = ACTION_RE.search(response)
    if not match:
        raise ValueError("response must contain one <action>{...}</action> block")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"action is not valid JSON: {exc.msg}") from exc
    action_type = value.get("type")
    if action_type == "bash" and isinstance(value.get("command"), str):
        return {"action": "run", "args": {"command": value["command"]}}
    if action_type == "mcp" and isinstance(value.get("name"), str) and isinstance(value.get("arguments"), dict):
        return {
            "action": "call_tool_mcp",
            "args": {"name": value["name"], "arguments": value["arguments"]},
        }
    if action_type in {"finish", "reject"}:
        return {"action": action_type, "args": {"thought": str(value.get("message", ""))}}
    raise ValueError("action must be bash, mcp, finish, or reject with the required fields")


class Runner(Protocol):
    def execute(self, action: dict[str, Any]) -> str: ...

    def close(self) -> None: ...


class RunnerFactory(Protocol):
    def __call__(self, task_path: Path, session_dir: Path, seed: int) -> Runner: ...


@dataclass
class Turn:
    instruction: str
    max_steps: int = 10
    success_patterns: list[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    turns: list[Turn]
    intro: str = ""
    index: int = 0
    steps: int = 0

    @classmethod
    def from_task_dir(cls, task_path: Path) -> "TaskPlan":
        turns_path = task_path / "turns.yml"
        if not turns_path.exists():
            instruction = (task_path / "task.md").read_text(encoding="utf-8").strip()
            return cls([Turn(instruction=instruction, max_steps=30)])
        manifest = yaml.safe_load(turns_path.read_text(encoding="utf-8")) or {}
        turns = []
        for row in manifest.get("turns", []):
            instruction_path = task_path / row["instruction_file"]
            turns.append(
                Turn(
                    instruction=instruction_path.read_text(encoding="utf-8").strip(),
                    max_steps=int(row.get("max_steps", 10)),
                    success_patterns=[str(value) for value in row.get("success_patterns", [])],
                )
            )
        if not turns:
            raise ValueError(f"no valid turns in {turns_path}")
        intro_path = task_path / "task-intro.md"
        return cls(turns, intro_path.read_text(encoding="utf-8").strip() if intro_path.exists() else "")

    def initial_observation(self) -> str:
        parts = [ACTION_HELP]
        if len(self.turns) > 1:
            parts.append("This task has multiple turns. Do not finish before the final turn.")
        if self.intro:
            parts.append(self.intro)
        parts.append(self.turns[0].instruction)
        return "\n\n".join(parts)

    def transition(self, observation: str, action_type: str) -> tuple[str, str | None, dict[str, Any]]:
        self.steps += 1
        if action_type == "reject":
            return observation, "REJECT", {"reason": "agent_reject"}
        turn = self.turns[self.index]
        matched = next((pattern for pattern in turn.success_patterns if pattern.lower() in observation.lower()), None)
        if matched:
            if self.index + 1 == len(self.turns):
                return observation, "COMPLETE", {"matched_success_pattern": matched}
            self.index += 1
            self.steps = 0
            return f"{observation}\n\nNext user instruction:\n{self.turns[self.index].instruction}", None, {
                "matched_success_pattern": matched
            }
        if action_type == "finish":
            return observation, "FAILED", {"reason": "finished_before_verified_completion"}
        if self.steps >= turn.max_steps:
            return observation, "FAILED", {"reason": "turn_step_limit"}
        return observation, None, {}


@dataclass
class Session:
    task_id: str
    is_harmful: bool
    runner: Runner
    plan: TaskPlan
    session_dir: Path
    terminal_state: dict[str, Any] | None = None


class TaskCatalog:
    def __init__(self, tasks_root: Path, rows: list[dict[str, Any]]):
        self.tasks_root = tasks_root.resolve()
        self.rows = {str(row["task_id"]): row for row in rows}

    def resolve(self, task_id: str) -> tuple[Path, bool]:
        row = self.rows[task_id]
        task_path = (self.tasks_root / str(row.get("task_path", task_id))).resolve()
        if not task_path.is_relative_to(self.tasks_root):
            raise ValueError(f"task path escapes tasks root: {task_path}")
        if not task_path.is_dir():
            raise FileNotFoundError(task_path)
        return task_path, bool(row.get("is_harmful", not row.get("is_benign", False)))


def load_catalog_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                rows.extend(json.loads(line) for line in handle if line.strip())
            else:
                value = json.load(handle)
                rows.extend(value if isinstance(value, list) else value["tasks"])
    return rows


class GatewayService:
    def __init__(self, catalog: TaskCatalog, work_root: Path, runner_factory: RunnerFactory):
        self.catalog = catalog
        self.work_root = work_root
        self.runner_factory = runner_factory
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        task_path, is_harmful = self.catalog.resolve(task_id)
        session_id = uuid.uuid4().hex
        session_dir = self.work_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        session = Session(
            task_id=task_id,
            is_harmful=is_harmful,
            runner=self.runner_factory(task_path, session_dir, seed),
            plan=TaskPlan.from_task_dir(task_path),
            session_dir=session_dir,
        )
        with self.lock:
            self.sessions[session_id] = session
        return {"session_id": session_id, "observation": session.plan.initial_observation(), "info": {}}

    def step(self, session_id: str, response: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        if session.terminal_state is not None:
            raise ValueError(f"session already finished: {session_id}")
        try:
            action = parse_actor_action(response)
            action_type = action["action"]
            observation = "" if action_type in {"finish", "reject"} else session.runner.execute(action)
            observation, outcome, evidence = session.plan.transition(observation, action_type)
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "observation": f"Invalid action: {exc}\n\n{ACTION_HELP}",
                "reward": 0.0,
                "done": False,
                "won": False,
                "info": {"is_action_valid": False},
            }
        done = outcome is not None
        reward = 0.0
        if done:
            decision = outcome_reward(is_harmful=session.is_harmful, outcome=outcome)
            reward = decision.reward
            session.terminal_state = {"outcome": outcome, "evidence": evidence}
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "won": reward > 0,
            "terminal_state": session.terminal_state,
            "evaluator_evidence": evidence,
            "info": {"is_action_valid": True, "outcome": outcome},
        }

    def close(self, session_id: str) -> None:
        with self.lock:
            session = self.sessions.pop(session_id)
        session.runner.close()
