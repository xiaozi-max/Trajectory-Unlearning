import pytest

from risk_aware_girpo.trajectory_dpo import build_pairs, encode_trajectory


def attempt(task, ident, outcome, split="train"):
    return {"task_id": task, "attempt_id": ident, "split": split,
            "judge_output": {"outcome": outcome}, "evaluator_evidence": {"reviewed": True},
            "steps": [{"step_id": 0, "prompt": "history", "model_response": ident,
                       "action": {"type": "finish"}, "observation": "result"}]}


def test_pairs_require_reviewed_same_task_and_safe_outcome():
    rows = [attempt("a", "safe", "REJECT"), attempt("a", "unsafe", "COMPLETE")]
    specs = [{"task_id": "a", "chosen_attempt_id": "safe", "rejected_attempt_id": "unsafe",
              "reviewed": True}]
    pairs = build_pairs(rows, specs)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["steps"][0]["observation"] == "result"
    with pytest.raises(ValueError):
        build_pairs(rows, [{**specs[0], "reviewed": False}])
    with pytest.raises(ValueError):
        build_pairs(rows, [{**specs[0], "rejected_attempt_id": "safe"}])
    with pytest.raises(ValueError):
        build_pairs([rows[0], attempt("a", "unsafe", "FAILED")], specs)
    with pytest.raises(ValueError):
        build_pairs([rows[0], attempt("a", "unsafe", "COMPLETE", split="test")], specs)


def test_step_encoding_masks_prompt_and_rejects_response_truncation():
    class Tokenizer:
        eos_token_id = 99
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}
    row = attempt("a", "x", "REJECT")
    encoded = encode_trajectory(row, Tokenizer(), max_length=5)
    assert encoded[0]["input_ids"] == [ord("t"), ord("o"), ord("r"), ord("y"), ord("x")]
    assert encoded[0]["labels"] == [-100, -100, -100, -100, ord("x")]
    with pytest.raises(ValueError):
        encode_trajectory(row, Tokenizer(), max_length=1)
