"""Offline assessment of the trained networks before they are used in the search (Section 3.6.2).

The networks are assessed on the validation instances by

* the makespans of the policy network dispatching on its own (greedy), next to uniform random dispatching and
  the MWKR rule;
* its agreement with MWKR: the share of states in which its most probable operation is the one with the most
  work remaining;
* its entropy (Equation 2.21, normalised) at temperature 1;
* the prediction error of the value network against the ranked reward of the episode;
* the time per network evaluation.

This separates weaknesses of a network from effects of the search.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional, Sequence

import numpy as np

from ..distributions.fixed import MWKRRollout
from ..env.model import State
from ..env.observation import FeatureTracker
from ..instances.instance import Instance
from .ranked_reward import RankedReward
from .trainer import uniform_ratio


def assess_offline(infer, instances: Sequence[Instance], ranked_state: Optional[dict] = None,
                   timing_calls: int = 200) -> dict:
    """Assess one trained network (inference object) on ``instances``."""
    ranked = RankedReward(**ranked_state) if ranked_state else None
    threshold = ranked.threshold() if ranked is not None else None
    rnd = random.Random(0)
    mwkr = MWKRRollout(0.0)
    ratios, agree, entropies, values, labels = [], [], [], [], []
    timing_state = None
    for inst in instances:
        infer.check(inst)
        state = State(inst)
        tracker = FeatureTracker(inst).reset(state)
        episode_values = []
        while state.active:
            active = state.active
            if len(active) == 1:
                job = active[0]
            else:
                logits, value = infer.evaluate(tracker, state)
                z = np.asarray(logits, dtype=np.float64)
                p = np.exp(z - z.max())
                p /= p.sum()
                nz = p[p > 0]
                entropies.append(float(-(nz * np.log(nz)).sum()) / math.log(len(active)))
                work = [inst.rem[j][state.next_op[j]] for j in active]
                agree.append(int(np.argmax(z)) == work.index(max(work)))
                episode_values.append(value)
                job = active[int(np.argmax(z))]
                if timing_state is None and state.n_done >= inst.n_ops // 2:
                    timing_state = (state.copy(), FeatureTracker(inst).reset(state))
            state.step(job)
            tracker.update(state)
        ratio = state.makespan / inst.lb0
        ratios.append(ratio)
        if threshold is not None:
            values.extend(episode_values)
            labels.extend([RankedReward.label(ratio, threshold)] * len(episode_values))
    out = {
        "instances": len(instances),
        "greedy_ratio": float(np.mean(ratios)),
        "uniform_ratio": uniform_ratio(instances, seed=0),
        "mwkr_ratio": float(np.mean([mwkr.rollout(State(inst), rnd, []) / inst.lb0 for inst in instances])),
        "agreement_with_mwkr": float(np.mean(agree)) if agree else float("nan"),
        "policy_entropy_tau1": float(np.mean(entropies)) if entropies else float("nan"),
    }
    if values:
        v, z = np.asarray(values), np.asarray(labels)
        out["value_mse"] = float(np.mean((v - z) ** 2))
        out["value_sign_accuracy"] = float(np.mean(np.sign(v) == z))
        out["ranked_threshold"] = threshold
    if timing_state is not None:
        st, tr = timing_state
        t0 = time.perf_counter()
        for _ in range(timing_calls):
            infer.logits(tr, st)
        out["seconds_per_evaluation"] = (time.perf_counter() - t0) / timing_calls
    return out
