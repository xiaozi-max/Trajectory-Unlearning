from __future__ import annotations

import torch
import torch.nn.functional as F


def simnpo_forget_loss(
    token_nll: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    beta: float = 1.0,
    delta: float = 0.0,
) -> torch.Tensor:
    """Official open-unlearning SimNPO forget term, factored for testing."""
    if token_nll.shape != loss_mask.shape:
        raise ValueError("token_nll and loss_mask must have the same shape")
    lengths = loss_mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("each forget response must contain at least one loss token")
    per_example_nll = (token_nll * loss_mask).sum(-1) / lengths - delta
    return -F.logsigmoid(beta * per_example_nll).mean() * 2.0 / beta


def simnpo_total_loss(
    token_nll: torch.Tensor,
    loss_mask: torch.Tensor,
    retain_loss: torch.Tensor,
    *,
    beta: float = 1.0,
    delta: float = 0.0,
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    forget = simnpo_forget_loss(token_nll, loss_mask, beta=beta, delta=delta)
    return gamma * forget + alpha * retain_loss
