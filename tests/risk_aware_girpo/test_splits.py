from risk_aware_girpo.splits import grouped_stratified_split


def test_goal_variants_never_cross_splits():
    rows = [
        {"task_id": "a-single", "goal_id": "a", "tool": "terminal", "attack_type": "method-1"},
        {"task_id": "a-multi", "goal_id": "a", "tool": "filesystem", "attack_type": "method-2"},
        {"task_id": "b", "goal_id": "b", "tool": "terminal", "attack_type": "method-1"},
        {"task_id": "c", "goal_id": "c", "tool": "terminal", "attack_type": "method-1"},
    ]
    split = grouped_stratified_split(rows, seed=7)
    a_splits = {row["split"] for row in split if row["goal_id"] == "a"}
    assert len(a_splits) == 1
