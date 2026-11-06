import json

import pytest

from risk_aware_girpo.gateway import GatewayService, TaskCatalog, parse_actor_action


class FakeRunner:
    def __init__(self):
        self.closed = False

    def execute(self, action):
        return f"executed:{action['action']}"

    def close(self):
        self.closed = True


def test_action_parser_accepts_only_small_protocol():
    assert parse_actor_action('<action>{"type":"bash","command":"pwd"}</action>')["action"] == "run"
    assert parse_actor_action('```json\n{"type":"bash","command":"pwd"}\n```')["action"] == "run"
    assert parse_actor_action(
        '<action>{"type":"mcp","name":"read_file","arguments":{"path":"x"}}</action>'
    )["action"] == "call_tool_mcp"
    with pytest.raises(ValueError):
        parse_actor_action('<action>{"action":"run","args":{"command":"pwd"}}</action>')


def test_gateway_advances_turns_and_rewards_harmful_completion(tmp_path):
    task = tmp_path / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "one.md").write_text("first", encoding="utf-8")
    (task / "two.md").write_text("second", encoding="utf-8")
    (task / "turns.yml").write_text(
        "turns:\n"
        "  - instruction_file: one.md\n"
        "    success_patterns: [executed]\n"
        "  - instruction_file: two.md\n"
        "    success_patterns: [executed]\n",
        encoding="utf-8",
    )
    runner = FakeRunner()
    service = GatewayService(
        TaskCatalog(tmp_path / "tasks", [{"task_id": "task-a", "is_harmful": True}]),
        tmp_path / "sessions",
        lambda *_: runner,
    )
    reset = service.reset("task-a", 7)
    action = '<action>{"type":"bash","command":"pwd"}</action>'
    first = service.step(reset["session_id"], action)
    assert not first["done"] and "Next user instruction" in first["observation"]
    second = service.step(reset["session_id"], action)
    assert second["done"] and second["reward"] == -1.0
    repeated = service.step(reset["session_id"], action)
    assert repeated["done"] and repeated["reward"] == 0.0
    assert repeated["info"]["already_done"]
    service.close(reset["session_id"])
    assert runner.closed


def test_invalid_action_is_retryable(tmp_path):
    task = tmp_path / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.md").write_text("task", encoding="utf-8")
    service = GatewayService(
        TaskCatalog(tmp_path / "tasks", [{"task_id": "task-a"}]),
        tmp_path / "sessions",
        lambda *_: FakeRunner(),
    )
    session_id = service.reset("task-a", 0)["session_id"]
    result = service.step(session_id, "not an action")
    assert not result["done"] and not result["info"]["is_action_valid"]
