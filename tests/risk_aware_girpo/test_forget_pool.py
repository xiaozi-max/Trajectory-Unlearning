import json

from risk_aware_girpo.forget_pool import RoundRobinForgetPool, load_forget_pool


def test_pool_filters_and_rotates(tmp_path):
    rows = []
    for index in range(2):
        rows.append(
            {
                "task_id": "task-a",
                "split": "train",
                "attempt_id": f"a{index}",
                "judge_output": {"outcome": "COMPLETE", "unsafe_intermediate": 0.9},
                "steps": [
                    {
                        "step_id": 0,
                        "turn_id": "t0",
                        "prompt": "p",
                        "model_response": "r",
                        "action": "a",
                        "jev_weight": index / 2,
                    }
                ],
            }
        )
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    pool = RoundRobinForgetPool(load_forget_pool(path))
    assert pool.next("task-a")["trajectory_id"] == "a0"
    assert pool.next("task-a")["trajectory_id"] == "a1"
    assert pool.next("task-a")["trajectory_id"] == "a0"
    assert pool.next("missing") is None
