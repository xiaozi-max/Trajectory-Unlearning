from __future__ import annotations

from typing import Any

import numpy as np
import torch


WEIGHT_FIELD = "forget_step_weight"
POLICY_WEIGHT_KEY = "policy_weights"


def validate_step_weights(values: Any, expected_rows: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(weights) != expected_rows:
        raise ValueError(f"expected {expected_rows} step weights, got {len(weights)}")
    if not np.isfinite(weights).all():
        raise ValueError("step weights must be finite")
    if ((weights < 0.0) | (weights > 1.0)).any():
        raise ValueError("step weights must be in [0, 1]")
    return weights


def attach_policy_weights(data):
    """Attach token-shaped weights without changing advantages or loss masks."""
    raw = data.non_tensor_batch.get(WEIGHT_FIELD)
    if raw is None:
        return data
    advantages = data.batch["advantages"]
    weights = validate_step_weights(raw, advantages.shape[0])
    scalar = torch.as_tensor(weights, device=advantages.device, dtype=advantages.dtype).unsqueeze(-1)
    data.batch[POLICY_WEIGHT_KEY] = scalar.expand_as(advantages)
    return data


def apply_policy_weights(advantages: torch.Tensor, policy_weights: torch.Tensor | None) -> torch.Tensor:
    if policy_weights is None:
        return advantages
    if policy_weights.shape != advantages.shape:
        raise ValueError(
            f"policy weight shape {tuple(policy_weights.shape)} does not match advantages {tuple(advantages.shape)}"
        )
    return advantages * policy_weights.to(device=advantages.device, dtype=advantages.dtype)
