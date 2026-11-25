"""Fixed rollout distributions: uniform sampling (U) and the dispatching-rule distributions (Section 3.2.2).

* U chooses uniformly at random among the candidates (``tau -> inf``, normalised entropy 1) and uses no
  knowledge of the problem.
* MWKR scores a candidate by the work remaining for its job divided by the largest work remaining among the
  candidates.  ``tau -> 0`` gives the MWKR dispatching rule; a larger ``tau`` spreads the probability over
  the other candidates.
* MOR, SPT and FIFO (the other rules compared by Ho et al., added as an extension of RQ1 and RQ2) are built in
  the same way from their own criterion: the operations remaining for the job (MOR, more is better), the
  processing time of the candidate operation (SPT, shorter is better) and the time at which the operation
  became available, the end of its job predecessor (FIFO, earlier is better).  As for MWKR, the difference to
  the best candidate is divided by the largest value among the candidates, so the probability of a candidate is
  ``exp((v - v_max) / (tau v_max))`` for MOR and ``exp((v_min - v) / (tau v_max))`` for SPT and FIFO.  Level 0
  is the dispatching rule itself, with ties broken by the lower job index as in :mod:`mctsjssp.env.rules`.
  The criterion of MOR, a count of operations, is often equal for several candidates, and MOR keeps the counts
  equal: as ``tau -> 0`` its distribution becomes uniform over the tied jobs (MOR with random tie-breaking),
  whose normalised entropy is about 0.7 on the validation instances.  Lower levels cannot be reached without
  an arbitrary tie-break; the calibration then keeps the smallest temperature and reports the entropy reached.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from ..env.model import State
from .base import RolloutPolicy


class UniformRollout(RolloutPolicy):
    """Experiment 1: rollouts that choose uniformly at random."""

    name = "U"
    level = 1.0
    tau = math.inf
    _draw_when_forced = True

    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        active = state.active
        step = state.step
        rand = rnd.random
        append = trace.append
        while active:
            j = active[int(rand() * len(active))]
            step(j)
            append(j)
        return state.makespan

    def _choose(self, state: State, rnd: random.Random):
        k = len(state.active)
        return int(rnd.random() * k), np.full(k, 1.0 / k)

    def distribution(self, state: State) -> np.ndarray:
        k = len(state.active)
        return np.full(k, 1.0 / k)

    def scores(self, state: State) -> np.ndarray:
        return np.zeros(len(state.active))


class MWKRRollout(RolloutPolicy):
    """Experiment 2: rollouts sampled from a softmax over the remaining work.

    ``level = 0`` is the MWKR rule itself (most work remaining, lowest job index on ties), ``level = 1`` is
    uniform sampling, and a level in between needs the temperature ``tau`` found by bisection.
    """

    name = "MWKR"

    def __init__(self, level: float = 0.0, tau: Optional[float] = None):
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

    # ------------------------------------------------------------------------------------ fast rollout
    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        rem = state.inst.rem
        nxt = state.next_op
        active = state.active
        step = state.step
        append = trace.append
        if self.tau is None:                                   # the MWKR rule
            while active:
                work = [rem[j][nxt[j]] for j in active]
                j = active[work.index(max(work))]
                step(j)
                append(j)
            return state.makespan
        if math.isinf(self.tau):                               # uniform sampling
            rand = rnd.random
            while active:
                j = active[int(rand() * len(active))]
                step(j)
                append(j)
            return state.makespan
        inv_tau = 1.0 / self.tau
        exp = math.exp
        choices = rnd.choices
        while active:
            if len(active) == 1:
                j = active[0]
            else:
                work = [rem[j][nxt[j]] for j in active]
                mx = max(work)
                if mx > 0:
                    scale = inv_tau / mx
                    j = choices(active, [exp((w - mx) * scale) for w in work])[0]
                else:
                    j = choices(active)[0]
            step(j)
            append(j)
        return state.makespan

    # ------------------------------------------------------------------------- recorded rollout step
    def _choose(self, state: State, rnd: random.Random):
        inst = state.inst
        active = state.active
        k = len(active)
        work = [inst.rem[j][state.next_op[j]] for j in active]
        mx = max(work)
        if self.tau is None:
            index = work.index(mx)
            p = np.zeros(k)
            p[index] = 1.0
            return index, p
        if math.isinf(self.tau):
            return int(rnd.random() * k), np.full(k, 1.0 / k)
        if mx > 0:
            scale = (1.0 / self.tau) / mx            # the same arithmetic as in rollout()
            weights = [math.exp((w - mx) * scale) for w in work]
            index = rnd.choices(range(k), weights)[0]
            p = np.array(weights)
            return index, p / p.sum()
        return rnd.choices(range(k))[0], np.full(k, 1.0 / k)

    # --------------------------------------------------------------------------------- distribution
    def scores(self, state: State) -> np.ndarray:
        inst = state.inst
        work = np.array([inst.rem[j][state.next_op[j]] for j in state.active], dtype=np.float64)
        mx = work.max()
        return work / mx if mx > 0 else np.zeros_like(work)

    def distribution(self, state: State) -> np.ndarray:
        s = self.scores(state)
        k = len(s)
        if self.tau is None:
            p = np.zeros(k)
            p[int(np.argmax(s))] = 1.0
            return p
        if math.isinf(self.tau):
            return np.full(k, 1.0 / k)
        e = np.exp((s - s.max()) / self.tau)
        return e / e.sum()


# --------------------------------------------------------------------------- MOR, SPT and FIFO (extension)
class _RuleRollout(RolloutPolicy):
    """Softmax over the criterion of a dispatching rule, built exactly like :class:`MWKRRollout`.

    Subclasses give the criterion of the candidates (``_values``) and whether a larger value is better
    (``maximise``).  The fast rollouts of the subclasses repeat the arithmetic of :meth:`_weights` inline, so
    that a rule costs no more per step than MWKR; :meth:`_choose` uses :meth:`_weights` and therefore makes the
    same decisions for the same random numbers.
    """

    maximise = True

    def __init__(self, level: float = 0.0, tau: Optional[float] = None):
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

    def _values(self, state: State) -> list:
        raise NotImplementedError

    def _best(self, values: list) -> int:
        """Index of the rule's choice: the first best value (the lower job index on ties)."""
        return values.index(max(values) if self.maximise else min(values))

    def _weights(self, values: list) -> Optional[list]:
        """Unnormalised probabilities at ``tau``; None if every candidate gets the same weight (``choices``)."""
        mx = max(values)
        if mx <= 0:
            return None
        scale = (1.0 / self.tau) / mx
        if self.maximise:
            return [math.exp((v - mx) * scale) for v in values]
        mn = min(values)
        return [math.exp((mn - v) * scale) for v in values]

    def _choose(self, state: State, rnd: random.Random):
        values = self._values(state)
        k = len(values)
        if self.tau is None:
            index = self._best(values)
            p = np.zeros(k)
            p[index] = 1.0
            return index, p
        if math.isinf(self.tau):
            return int(rnd.random() * k), np.full(k, 1.0 / k)
        weights = self._weights(values)
        if weights is None:
            return rnd.choices(range(k))[0], np.full(k, 1.0 / k)
        index = rnd.choices(range(k), weights)[0]
        p = np.array(weights)
        return index, p / p.sum()

    def scores(self, state: State) -> np.ndarray:
        """``v / v_max`` (MOR) or ``-v / v_max`` (SPT, FIFO): the larger, the more the rule prefers it."""
        v = np.array(self._values(state), dtype=np.float64)
        mx = v.max()
        if mx <= 0:
            return np.zeros_like(v)
        return v / mx if self.maximise else -v / mx

    def distribution(self, state: State) -> np.ndarray:
        values = self._values(state)
        k = len(values)
        if self.tau is None:
            p = np.zeros(k)
            p[self._best(values)] = 1.0
            return p
        if math.isinf(self.tau):
            return np.full(k, 1.0 / k)
        weights = self._weights(values)
        if weights is None:
            return np.full(k, 1.0 / k)
        p = np.array(weights)
        return p / p.sum()

    def _uniform(self, state: State, rnd: random.Random, trace: list) -> int:
        active, step, append, rand = state.active, state.step, trace.append, rnd.random
        while active:
            j = active[int(rand() * len(active))]
            step(j)
            append(j)
        return state.makespan


class MORRollout(_RuleRollout):
    """Most operations remaining for the job (more is better)."""

    name = "MOR"
    maximise = True

    def _values(self, state: State) -> list:
        m = state.inst.m
        nxt = state.next_op
        return [m - nxt[j] for j in state.active]

    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        m = state.inst.m
        nxt = state.next_op
        active = state.active
        step = state.step
        append = trace.append
        if self.tau is None:                                   # the MOR rule
            while active:
                v = [m - nxt[j] for j in active]
                j = active[v.index(max(v))]
                step(j)
                append(j)
            return state.makespan
        if math.isinf(self.tau):
            return self._uniform(state, rnd, trace)
        inv_tau = 1.0 / self.tau
        exp = math.exp
        choices = rnd.choices
        while active:
            if len(active) == 1:
                j = active[0]
            else:
                v = [m - nxt[j] for j in active]
                mx = max(v)
                if mx > 0:
                    scale = inv_tau / mx
                    j = choices(active, [exp((x - mx) * scale) for x in v])[0]
                else:
                    j = choices(active)[0]
            step(j)
            append(j)
        return state.makespan


class SPTRollout(_RuleRollout):
    """Shortest processing time of the candidate operation (shorter is better)."""

    name = "SPT"
    maximise = False

    def _values(self, state: State) -> list:
        dur = state.inst.dur
        nxt = state.next_op
        return [dur[j][nxt[j]] for j in state.active]

    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        dur = state.inst.dur
        nxt = state.next_op
        active = state.active
        step = state.step
        append = trace.append
        if self.tau is None:                                   # the SPT rule
            while active:
                v = [dur[j][nxt[j]] for j in active]
                j = active[v.index(min(v))]
                step(j)
                append(j)
            return state.makespan
        if math.isinf(self.tau):
            return self._uniform(state, rnd, trace)
        inv_tau = 1.0 / self.tau
        exp = math.exp
        choices = rnd.choices
        while active:
            if len(active) == 1:
                j = active[0]
            else:
                v = [dur[j][nxt[j]] for j in active]
                mx = max(v)
                if mx > 0:
                    mn = min(v)
                    scale = inv_tau / mx
                    j = choices(active, [exp((mn - x) * scale) for x in v])[0]
                else:
                    j = choices(active)[0]
            step(j)
            append(j)
        return state.makespan


class FIFORollout(_RuleRollout):
    """First in, first out: the operation that became available first, i.e. whose job predecessor ended
    earliest (earlier is better)."""

    name = "FIFO"
    maximise = False

    def _values(self, state: State) -> list:
        ready = state.job_ready
        return [ready[j] for j in state.active]

    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        ready = state.job_ready
        active = state.active
        step = state.step
        append = trace.append
        if self.tau is None:                                   # the FIFO rule
            while active:
                v = [ready[j] for j in active]
                j = active[v.index(min(v))]
                step(j)
                append(j)
            return state.makespan
        if math.isinf(self.tau):
            return self._uniform(state, rnd, trace)
        inv_tau = 1.0 / self.tau
        exp = math.exp
        choices = rnd.choices
        while active:
            if len(active) == 1:
                j = active[0]
            else:
                v = [ready[j] for j in active]
                mx = max(v)
                if mx > 0:
                    mn = min(v)
                    scale = inv_tau / mx
                    j = choices(active, [exp((mn - x) * scale) for x in v])[0]
                else:
                    j = choices(active)[0]
            step(j)
            append(j)
        return state.makespan


#: the dispatching-rule distributions; MWKR is the one of the main design, the others extend RQ1 and RQ2
RULE_ROLLOUTS = {"MWKR": MWKRRollout, "MOR": MORRollout, "SPT": SPTRollout, "FIFO": FIFORollout}
FIXED_RULES = tuple(RULE_ROLLOUTS)


def make_rule_rollout(name: str, level: float = 0.0, tau: Optional[float] = None) -> RolloutPolicy:
    """The rollout distribution of a dispatching rule at an entropy level (with its temperature)."""
    try:
        cls = RULE_ROLLOUTS[name]
    except KeyError:
        raise ValueError(f"unknown dispatching rule {name!r}; expected one of {FIXED_RULES}") from None
    return cls(level, tau)
