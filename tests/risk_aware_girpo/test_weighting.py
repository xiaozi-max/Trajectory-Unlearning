import numpy as np
import pytest
import torch

from risk_aware_girpo.weighting import apply_policy_weights, attach_policy_weights, validate_step_weights


class FakeData:
    def __init__(self):
        self.batch = {"advantages": torch.tensor([[-0.4, -0.4], [0.3, 0.3]])}
        self.non_tensor_batch = {"forget_step_weight": np.array([0.25, 1.0])}


def test_all_one_weights_reproduce_girpo():
    advantages = torch.tensor([[-1.0, -1.0], [0.5, 0.5]])
    weighted = apply_policy_weights(advantages, torch.ones_like(advantages))
    torch.testing.assert_close(weighted, advantages)


def test_zero_weight_removes_only_that_step():
    advantages = torch.tensor([[-0.2, -0.2], [0.4, 0.4]])
    weights = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    weighted = apply_policy_weights(advantages, weights)
    torch.testing.assert_close(weighted[0], torch.zeros(2))
    torch.testing.assert_close(weighted[1], advantages[1])


def test_invalid_weights_fail_closed():
    with pytest.raises(ValueError):
        validate_step_weights(np.array([1.2], dtype=np.float32), 1)


def test_attach_expands_one_step_weight_to_response_tokens():
    data = attach_policy_weights(FakeData())
    torch.testing.assert_close(data.batch["policy_weights"], torch.tensor([[0.25, 0.25], [1.0, 1.0]]))
