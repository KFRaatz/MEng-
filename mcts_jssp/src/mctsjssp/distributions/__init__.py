"""Rollout distributions (Section 3.2.2), their entropy and the temperatures of the entropy levels."""
from .base import RolloutPolicy, StepStats
from .entropy import entropy, greedy, kl_divergence, normalised_entropy, same_top, softmax
from .fixed import (FIFORollout, FIXED_RULES, MORRollout, MWKRRollout, RULE_ROLLOUTS, SPTRollout,
                    UniformRollout, make_rule_rollout)
from .learned import FastRollout, NetworkRollout, make_learned_rollout

__all__ = ["RolloutPolicy", "StepStats", "UniformRollout", "MWKRRollout", "MORRollout", "SPTRollout",
           "FIFORollout", "RULE_ROLLOUTS", "FIXED_RULES", "make_rule_rollout", "NetworkRollout", "FastRollout",
           "make_learned_rollout",
           "entropy", "greedy", "kl_divergence", "normalised_entropy", "same_top", "softmax"]
