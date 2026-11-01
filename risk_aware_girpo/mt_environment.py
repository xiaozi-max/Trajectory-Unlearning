from __future__ import annotations

import json
import itertools
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Backend(Protocol):
    def reset(self, task_id: str, seed: int) -> dict[str, Any]: ...

    def step(self, session_id: str, action: str) -> dict[str, Any]: ...

    def close(self, session_id: str) -> None: ...


@dataclass
class OpenHandsHTTPBackend:
    """Thin client for a stateful OpenHands gateway.

    The gateway is deliberately separate from veRL workers: it owns task
    sandboxes/MCP services, while this client exposes reset/step semantics.
    """

    base_url: str
    timeout_seconds: float = 120.0

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        return self._request("/reset", {"task_id": task_id, "seed": seed})

    def step(self, session_id: str, action: str) -> dict[str, Any]:
        return self._request("/step", {"session_id": session_id, "action": action})

    def close(self, session_id: str) -> None:
        self._request("/close", {"session_id": session_id})


class MTAgentRiskVectorEnv:
    def __init__(self, backend: Backend, task_ids: list[str], seeds: list[int]):
        if len(task_ids) != len(seeds):
            raise ValueError("task_ids and seeds must have equal length")
        self.backend = backend
        self.task_ids = task_ids
        self.seeds = seeds
        self.sessions: list[str] = []
        self.current_task_ids: list[str] = []

    @staticmethod
    def _overrides(kwargs: Any) -> tuple[list[str] | None, list[int] | None]:
        if kwargs is None:
            return None, None
        if isinstance(kwargs, dict):
            return kwargs.get("task_ids"), kwargs.get("seeds")
        if isinstance(kwargs, (list, tuple, np.ndarray)):
            rows = list(kwargs)
            if rows and all(isinstance(row, dict) for row in rows):
                return [str(row["task_id"]) for row in rows], [int(row["seed"]) for row in rows]
        raise ValueError("env_kwargs must be a mapping or a sequence of {task_id, seed} mappings")

    def reset(self, kwargs: dict[str, Any] | None = None):
        override_task_ids, override_seeds = self._overrides(kwargs)
        task_ids = list(override_task_ids or self.task_ids)
        seeds = list(override_seeds or self.seeds)
        results = [self.backend.reset(task_id, int(seed)) for task_id, seed in zip(task_ids, seeds, strict=True)]
        self.sessions = [str(result["session_id"]) for result in results]
        self.current_task_ids = task_ids
        observations = [str(result["observation"]) for result in results]
        infos = [
            {
                **result.get("info", {}),
                "task_id": task_id,
                "seed": seed,
                "won": False,
            }
            for result, task_id, seed in zip(results, task_ids, seeds, strict=True)
        ]
        return observations, infos

    def step(self, actions: list[str]):
        if len(actions) != len(self.sessions):
            raise ValueError("one action is required for each active session")
        results = [self.backend.step(session, action) for session, action in zip(self.sessions, actions, strict=True)]
        observations = [str(result.get("observation", "")) for result in results]
        rewards = np.asarray([float(result.get("reward", 0.0)) for result in results], dtype=np.float32)
        dones = np.asarray([bool(result.get("done", False)) for result in results], dtype=np.bool_)
        infos = [
            {
                **result.get("info", {}),
                "task_id": task_id,
                "won": bool(result.get("won", result.get("reward", 0.0) > 0.0)),
                "terminal_state": result.get("terminal_state"),
                "evaluator_evidence": result.get("evaluator_evidence"),
            }
            for result, task_id in zip(results, self.current_task_ids, strict=True)
        ]
        return observations, rewards, dones, infos

    def close(self) -> None:
        for session in self.sessions:
            self.backend.close(session)
        self.sessions = []
        self.current_task_ids = []


class MTAgentRiskEnvironmentManager:
    def __init__(self, envs: MTAgentRiskVectorEnv):
        self.envs = envs

    def reset(self, kwargs=None):
        observations, infos = self.envs.reset(kwargs=kwargs)
        task_ids = [info["task_id"] for info in infos]
        return {
            "text": observations,
            "image": None,
            "anchor": observations.copy(),
            "task_id": task_ids,
        }, infos

    def step(self, text_actions: list[str]):
        observations, rewards, dones, infos = self.envs.step(text_actions)
        task_ids = [info.get("task_id", "") for info in infos]
        for info in infos:
            info["is_action_valid"] = True
        return {
            "text": observations,
            "image": None,
            "anchor": observations.copy(),
            "task_id": task_ids,
        }, rewards, dones, infos

    def success_evaluator(self, *, total_infos, total_batch_list):
        values = []
        for infos in total_infos:
            terminal = next((info for info in reversed(infos) if info), {})
            values.append(float(terminal.get("won", False)))
        return {"success_rate": np.asarray(values, dtype=np.float32)}

    def close(self) -> None:
        self.envs.close()


def _load_task_ids(path: str) -> list[str]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            value = json.load(handle)
            rows = value if isinstance(value, list) else value.get("tasks", [])
    task_ids = [str(row["task_id"] if isinstance(row, dict) else row) for row in rows]
    if not task_ids:
        raise ValueError(f"no task IDs in manifest: {path}")
    return task_ids


def _repeat_to_size(values: list[str], size: int) -> list[str]:
    return list(itertools.islice(itertools.cycle(values), size))


def make_mt_agentrisk_envs(config):
    endpoint = str(config.env.mt_agentrisk.gateway_url)
    group_n = max(int(config.env.rollout.n), 1)
    train_size = int(config.data.train_batch_size) * group_n
    val_size = int(config.data.val_batch_size)
    train_ids = _repeat_to_size(_load_task_ids(str(config.env.mt_agentrisk.train_manifest)), train_size)
    val_ids = _repeat_to_size(_load_task_ids(str(config.env.mt_agentrisk.val_manifest)), val_size)
    train_seed = int(config.env.seed)
    backend = OpenHandsHTTPBackend(endpoint)
    train = MTAgentRiskVectorEnv(backend, train_ids, [train_seed + i for i in range(train_size)])
    validation = MTAgentRiskVectorEnv(backend, val_ids, [train_seed + 1000 + i for i in range(val_size)])
    return MTAgentRiskEnvironmentManager(train), MTAgentRiskEnvironmentManager(validation)
