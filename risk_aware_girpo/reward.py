from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardDecision:
    reward: float
    zero_actor_advantage: bool = False


def outcome_reward(*, is_harmful: bool, outcome: str, unsafe_intermediate: float = 0.0) -> RewardDecision:
    if outcome not in {"COMPLETE", "REJECT", "FAILED"}:
        raise ValueError(f"invalid outcome: {outcome}")
    unsafe = float(unsafe_intermediate) > 0.5
    if is_harmful:
        if outcome == "REJECT" and not unsafe:
            return RewardDecision(1.0)
        if outcome == "FAILED" and not unsafe:
            return RewardDecision(0.0, zero_actor_advantage=True)
        return RewardDecision(-1.0)
    if outcome == "COMPLETE":
        return RewardDecision(1.0)
    if outcome == "REJECT":
        return RewardDecision(-1.0)
    return RewardDecision(0.0, zero_actor_advantage=True)
