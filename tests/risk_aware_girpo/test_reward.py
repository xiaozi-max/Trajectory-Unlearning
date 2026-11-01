from risk_aware_girpo.reward import outcome_reward


def test_harmful_reward_table():
    assert outcome_reward(is_harmful=True, outcome="REJECT").reward == 1.0
    assert outcome_reward(is_harmful=True, outcome="COMPLETE").reward == -1.0
    failed = outcome_reward(is_harmful=True, outcome="FAILED")
    assert failed.reward == 0.0 and failed.zero_actor_advantage
    assert outcome_reward(is_harmful=True, outcome="REJECT", unsafe_intermediate=0.9).reward == -1.0


def test_benign_reward_table():
    assert outcome_reward(is_harmful=False, outcome="COMPLETE").reward == 1.0
    assert outcome_reward(is_harmful=False, outcome="REJECT").reward == -1.0
