"""Learned rollout distributions: LR (GIN policy) and AZ (fully connected policy network) (Section 3.2.2).

The score of a candidate is the logit that the policy network gives it.  ``tau = 1`` samples from the learned
policy itself, ``tau -> 0`` follows it greedily, and for an entropy level between 0 and 1 the temperature is
set by bisection.  Every rollout step with more than one candidate needs one network evaluation.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_right
from itertools import accumulate
from typing import Optional

import numpy as np

from ..env.model import State
from ..env.observation import FeatureTracker
from .base import RolloutPolicy


class NetworkRollout(RolloutPolicy):
    """Rollouts sampled from the softmax of the logits of a policy network.

    ``infer`` is an inference object of :mod:`mctsjssp.networks.inference` (GIN for LR, fully connected for
    AZ); the rollout itself is identical for both, so the two learned configurations differ only in their
    network.
    """

    def __init__(self, name: str, infer, level: Optional[float] = 0.0, tau: Optional[float] = None):
        """``level`` is the target normalised entropy (0 greedy, 1 uniform, otherwise ``tau`` is required);
        ``level=None`` samples at the given temperature ``tau`` without an entropy level (self-play)."""
        self.name = name
        self.infer = infer
        if level is None:
            if tau is None or not tau > 0:
                raise ValueError("sampling without an entropy level needs a positive temperature")
            self.level, self.tau = None, float(tau)
            self._draw_when_forced = math.isinf(self.tau)
            self._trackers: dict[int, tuple] = {}
            self._tr: Optional[FeatureTracker] = None
            return
        self.level = float(level)
        if self.level <= 0.0:
            self.tau = None
        elif self.level >= 1.0:
            self.tau = math.inf
        else:
            if tau is None or not tau > 0:
                raise ValueError("an entropy level between 0 and 1 needs a positive temperature")
            self.tau = float(tau)
        self._draw_when_forced = self.tau is not None and math.isinf(self.tau)
        self._trackers: dict[int, tuple] = {}
        self._tr: Optional[FeatureTracker] = None

    def _tracker(self, state: State) -> FeatureTracker:
        inst = state.inst
        entry = self._trackers.get(id(inst))
        if entry is None:
            self.infer.check(inst)
            entry = (inst, FeatureTracker(inst))       # keep the instance alive with its tracker
            self._trackers[id(inst)] = entry
        return entry[1].reset(state)

    # ------------------------------------------------------------------------------------ fast rollout
    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        active = state.active
        step = state.step
        append = trace.append
        tau = self.tau
        if tau is not None and math.isinf(tau):                # uniform sampling, no network needed
            rand = rnd.random
            while active:
                j = active[int(rand() * len(active))]
                step(j)
                append(j)
            return state.makespan
        if len(active) > 1:
            infer = self.infer
            tracker = self._tracker(state)
            infer.begin(tracker, state)
            update = tracker.update
            advance = infer.advance
            logits = infer.rollout_logits
            inv_tau = None if tau is None else 1.0 / tau
            rand = rnd.random
            while len(active) > 1:
                z = logits(tracker, state)
                if inv_tau is None:
                    j = active[int(z.argmax())]
                else:
                    c = np.exp((z - z.max()) * inv_tau).cumsum()
                    index = int(np.searchsorted(c, rand() * c[-1], side="right"))
                    j = active[index if index < len(active) else len(active) - 1]
                step(j)
                append(j)
                update(state)
                advance(tracker, state)
        while active:                                          # one job left: no decision to make
            j = active[0]
            step(j)
            append(j)
        return state.makespan

    # ------------------------------------------------------------------------- recorded rollout step
    def _begin(self, state: State) -> None:
        self._tr = self._tracker(state) if len(state.active) > 1 else None
        if self._tr is not None:
            self.infer.begin(self._tr, state)

    def _after_step(self, state: State) -> None:
        if self._tr is not None and state.active:
            self._tr.update(state)
            self.infer.advance(self._tr, state)

    def _choose(self, state: State, rnd: random.Random):
        k = len(state.active)
        tau = self.tau
        if tau is not None and math.isinf(tau):
            return int(rnd.random() * k), np.full(k, 1.0 / k)
        z = self.infer.rollout_logits(self._tr, state)
        if tau is None:
            index = int(z.argmax())
            p = np.zeros(k)
            p[index] = 1.0
            return index, p
        e = np.exp((z - z.max()) * (1.0 / tau))      # the same arithmetic as in rollout()
        c = e.cumsum()
        index = int(np.searchsorted(c, rnd.random() * c[-1], side="right"))
        return (index if index < k else k - 1), e / c[-1]

    # --------------------------------------------------------------------------------- distribution
    def scores(self, state: State) -> np.ndarray:
        if len(state.active) < 2:
            return np.zeros(len(state.active))
        return np.asarray(self.infer.logits(self._tracker(state), state), dtype=np.float64)

    def distribution(self, state: State) -> np.ndarray:
        k = len(state.active)
        if k == 1 or (self.tau is not None and math.isinf(self.tau)):
            return np.full(k, 1.0 / k)
        z = self.scores(state)
        if self.tau is None:
            p = np.zeros(k)
            p[int(np.argmax(z))] = 1.0
            return p
        e = np.exp((z - z.max()) / self.tau)
        return e / e.sum()

    def describe(self) -> dict:
        d = super().describe()
        d["network"] = getattr(self.infer, "kind", "?")
        return d


class FastRollout(NetworkRollout):
    """Rollouts sampled from a fast policy network (:mod:`mctsjssp.networks.fast`).

    The fast networks compute their features from the partial schedule itself, so no graph observation is
    kept, and the candidates are sampled in pure Python, which is cheaper than NumPy for a few candidates.
    The fast and the recorded rollout consume the random numbers identically.
    """

    def _sample(self, z: list, rnd: random.Random) -> tuple[int, list]:
        tau = self.tau
        if tau is None:
            best = max(z)
            return z.index(best), None
        inv_tau = 1.0 / tau
        zmax = max(z)
        exp = math.exp
        weights = [exp((v - zmax) * inv_tau) for v in z]
        cum = list(accumulate(weights))
        index = bisect_right(cum, rnd.random() * cum[-1])
        return (index if index < len(z) else len(z) - 1), weights

    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        active = state.active
        step = state.step
        append = trace.append
        tau = self.tau
        if tau is not None and math.isinf(tau):                # uniform sampling, no network needed
            rand = rnd.random
            while active:
                j = active[int(rand() * len(active))]
                step(j)
                append(j)
            return state.makespan
        self.infer.check(state.inst)
        scores = self.infer.score_list
        sample = self._sample
        while len(active) > 1:
            index, _ = sample(scores(state), rnd)
            j = active[index]
            step(j)
            append(j)
        while active:                                          # one job left: no decision to make
            j = active[0]
            step(j)
            append(j)
        return state.makespan

    def _begin(self, state: State) -> None:
        self.infer.check(state.inst)

    def _after_step(self, state: State) -> None:
        """No observation to update."""

    def _choose(self, state: State, rnd: random.Random):
        k = len(state.active)
        tau = self.tau
        if tau is not None and math.isinf(tau):
            return int(rnd.random() * k), np.full(k, 1.0 / k)
        index, weights = self._sample(self.infer.score_list(state), rnd)
        if weights is None:
            p = np.zeros(k)
            p[index] = 1.0
            return index, p
        p = np.asarray(weights, dtype=np.float64)
        return index, p / p.sum()

    def scores(self, state: State) -> np.ndarray:
        if len(state.active) < 2:
            return np.zeros(len(state.active))
        self.infer.check(state.inst)
        return np.asarray(self.infer.score_list(state), dtype=np.float64)


def make_learned_rollout(name: str, infer, level: float = 0.0, tau: Optional[float] = None) -> NetworkRollout:
    """The rollout distribution of a trained network: :class:`FastRollout` for a fast policy network,
    :class:`NetworkRollout` for the GIN of LR and the networks of AZ."""
    if getattr(infer, "needs_tracker", True):
        return NetworkRollout(name, infer, level, tau)
    return FastRollout(name, infer, level, tau)
