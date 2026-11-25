"""Temperatures of the entropy levels (Sections 3.2.2 and 3.6.2).

MWKR, LR and AZ are compared at the same normalised entropy.  For the levels 0.25, 0.5 and 0.75 the
temperature of each distribution is set so that the normalised entropy, averaged over the states that its
rollouts visit on the validation instances, equals the level.  The temperature is found by bisection, which
works because the normalised entropy rises with the temperature.

In the search a rollout starts below every decision, so states late in the schedule are visited by more
rollouts than early ones.  The rollouts used here do the same: on every validation instance one schedule is
built by the distribution itself, and rollouts are started from states at evenly spaced depths of that
schedule (``start_depths``; a single depth of 0 gives rollouts from the empty schedule only).  The same seeds
are used for every temperature (common random numbers), which makes the mean entropy a deterministic function
of the temperature.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from ..env.model import State
from ..instances.instance import Instance
from .base import RolloutPolicy, StepStats

#: Entropy levels of the study (Section 3.2.2); level 1 is uniform sampling (U).
ENTROPY_LEVELS = (0.0, 0.25, 0.5, 0.75)


def mean_rollout_entropy(policy: RolloutPolicy, instances: Sequence[Instance], rollouts_per_instance: int = 1,
                         seed: int = 0, start_depths: int = 10) -> float:
    """Mean normalised entropy over the states visited by rollouts of ``policy`` on ``instances``.

    For every instance (and repetition) the policy first builds one complete schedule; rollouts are then
    started from ``start_depths`` evenly spaced depths of it (0, 1/k, 2/k, ... of the operations).
    """
    stats = StepStats(rate=0.0)
    for i, inst in enumerate(instances):
        for r in range(rollouts_per_instance):
            base_seed = seed * 1_000_003 + i * 1_009 + r
            if start_depths <= 1:
                policy.rollout_recorded(State(inst), random.Random(base_seed), [], stats)
                continue
            base: list[int] = []
            policy.rollout(State(inst), random.Random(base_seed), base)
            for d in range(start_depths):
                state = State(inst)
                for job in base[: (d * inst.n_ops) // start_depths]:
                    state.step(job)
                policy.rollout_recorded(state, random.Random(base_seed * 31 + d + 1), [], stats)
    return stats.mean_entropy


@dataclass
class Calibration:
    target: float
    tau: float
    realised: float
    evaluations: int
    converged: bool
    tolerance: float

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_tau(make_policy: Callable[[float], RolloutPolicy], instances: Sequence[Instance], target: float,
                  tolerance: float = 0.01, rollouts_per_instance: int = 1, seed: int = 0,
                  tau_low: float = 1e-4, tau_high: float = 1e3, max_evaluations: int = 60,
                  start_depths: int = 10) -> Calibration:
    """Find the temperature at which the mean normalised entropy of the rollout states equals ``target``.

    ``make_policy(tau)`` builds the rollout distribution at a temperature.  The bisection is carried out on
    ``log(tau)`` and stops when the realised entropy is within ``tolerance`` of the target.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("bisection is only needed for levels strictly between 0 and 1")

    evaluations = 0

    def f(tau: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return mean_rollout_entropy(make_policy(tau), instances, rollouts_per_instance, seed, start_depths)

    lo, hi = tau_low, tau_high
    f_lo, f_hi = f(lo), f(hi)
    while f_lo > target and lo > 1e-12:                 # widen the bracket if necessary
        lo /= 100.0
        f_lo = f(lo)
    while f_hi < target and hi < 1e12:
        hi *= 100.0
        f_hi = f(hi)
    best_tau, best_h = (lo, f_lo) if abs(f_lo - target) <= abs(f_hi - target) else (hi, f_hi)
    if not (f_lo <= target <= f_hi):
        # The target cannot be bracketed (for example exact ties keep the entropy above the target).
        return Calibration(target, best_tau, best_h, evaluations, abs(best_h - target) <= tolerance, tolerance)
    while evaluations < max_evaluations and abs(best_h - target) > tolerance and hi / lo > 1.0 + 1e-9:
        mid = math.sqrt(lo * hi)
        h = f(mid)
        if abs(h - target) < abs(best_h - target):
            best_tau, best_h = mid, h
        if h < target:
            lo = mid
        else:
            hi = mid
    return Calibration(target, best_tau, best_h, evaluations, abs(best_h - target) <= tolerance, tolerance)
