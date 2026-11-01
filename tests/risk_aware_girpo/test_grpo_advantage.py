import numpy as np
import pytest
import torch

pytest.importorskip("ray", reason="veRL runtime dependency is not installed on this host")
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_injected_row_does_not_shift_real_baseline():
    rewards = torch.tensor([[1.0], [-1.0], [-2.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["group", "group", "group"]),
        traj_index=np.array(["real-1", "real-2", "forget"]),
        norm_adv_by_std_in_grpo=False,
        is_injected=np.array([False, False, True]),
        forget_advantage_min=-1.0,
    )
    torch.testing.assert_close(advantages[:2, 0], torch.tensor([1.0, -1.0]))
    assert advantages[2, 0].item() == -1.0
