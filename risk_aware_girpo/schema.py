from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


OUTCOMES = frozenset({"COMPLETE", "REJECT", "FAILED"})


@dataclass(frozen=True)
class TrajectoryStep:
    step_id: int
    turn_id: str
    prompt: str
    model_response: str
    action: Any
    observation: str = ""
    jev_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not 0.0 <= float(self.jev_weight) <= 1.0:
            raise ValueError("jev_weight must be in [0, 1]")
        if not self.prompt or not self.model_response:
            raise ValueError("prompt and model_response are required")


@dataclass(frozen=True)
class Attempt:
    task_id: str
    split: str
    attempt_id: str
    seed: int
    model_id: str
    adapter_id: str | None
    steps: tuple[TrajectoryStep, ...]
    task_instructions: tuple[str, ...] = ()
    terminal_state: Any = None
    evaluator_evidence: Any = None
    judge_output: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.attempt_id:
            raise ValueError("task_id and attempt_id are required")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {self.split}")
        if not self.steps:
            raise ValueError("attempt must contain at least one step")
        expected = list(range(len(self.steps)))
        actual = [step.step_id for step in self.steps]
        if actual != expected:
            raise ValueError(f"step_id values must be contiguous: {actual}")

    @property
    def outcome(self) -> str | None:
        if not self.judge_output:
            return None
        value = self.judge_output.get("outcome")
        return value if value in OUTCOMES else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Attempt":
        steps = tuple(TrajectoryStep(**step) for step in value["steps"])
        return cls(**{**value, "steps": steps})


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
