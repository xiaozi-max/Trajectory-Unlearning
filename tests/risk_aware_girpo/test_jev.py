from risk_aware_girpo.jev import (
    canonical_hash,
    is_dangerous_attempt,
    parse_attempt,
    parse_step_weight,
    step_request,
)


def test_request_hash_is_order_invariant():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_step_template_and_parsing():
    payload = step_request(
        model="jev-fixed",
        task_goal="goal",
        history_before_action=[],
        current_model_response="response",
        current_action="action",
    )
    assert set(payload) == {"model", "state", "questions"}
    assert parse_step_weight({"answers": {"suppress_this_step": {"noul": 0.25}}}) == 0.25


def test_attempt_parsing():
    parsed = parse_attempt(
        {
            "answers": {
                "outcome": {"choice": "REJECT", "probabilities": {"REJECT": 0.9}, "confidence": 0.9},
                "unsafe_intermediate": {"noul": 0.1},
            }
        }
    )
    assert parsed["outcome"] == "REJECT"
    assert parsed["unsafe_intermediate"] == 0.1


def test_only_dangerous_attempts_qualify_for_step_weights():
    assert is_dangerous_attempt({"judge_output": {"outcome": "COMPLETE", "unsafe_intermediate": 0.0}})
    assert is_dangerous_attempt({"judge_output": {"outcome": "FAILED", "unsafe_intermediate": 0.9}})
    assert not is_dangerous_attempt({"judge_output": {"outcome": "REJECT", "unsafe_intermediate": 0.1}})
    assert not is_dangerous_attempt({"judge_output": {"outcome": "FAILED", "unsafe_intermediate": 0.5}})
    # Raw rollout conversions fall back to terminal_state.
    assert is_dangerous_attempt({"terminal_state": {"outcome": "COMPLETE"}})
    assert not is_dangerous_attempt({"terminal_state": {"outcome": "REJECT"}})
    assert not is_dangerous_attempt({})
