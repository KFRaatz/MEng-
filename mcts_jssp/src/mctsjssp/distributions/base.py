"""Common interface of the rollout distributions (Section 3.2.2).

A rollout distribution completes a partial schedule by sampling one job after another.  Every distribution
offers two rollouts that make exactly the same decisions for the same random numbers:

* :meth:`RolloutPolicy.rollout` - the fast path used by the search;
* :meth:`RolloutPolicy.rollout_recorded` - additionally accumulates the normalised entropy over all steps
  and stores a sample of the steps (Section 3.6.4).

``level`` is the target normalised entropy: 0 is the deterministic rule (highest score), 1 is uniform
sampling, and for a level in between the temperature ``tau`` of the softmax is set by bisection
(:mod:`mctsjssp.distributions.calibration`).  Steps with a single candidate need no decision; they are
neither sampled nor counted in the entropy.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np

from ..env.model import State
from .entropy import kl_divergence


class StepStats:
    """Entropy accumulated over all rollout steps and a random sample of the steps.

    One step in ``1 / rate`` is stored, drawn with its own seeded generator so that recording never changes
    the random numbers of the search.
    """

    __slots__ = ("h_sum", "h_cnt", "rate", "skip", "samples", "context", "_rnd", "_log1m")

    def __init__(self, rate: float = 0.001, seed: int = 0):
        self.h_sum = 0.0
        self.h_cnt = 0
        self.rate = float(rate)
        self._rnd = random.Random(seed)
        self._log1m = math.log1p(-self.rate) if 0.0 < self.rate < 1.0 else None
        self.skip = self._gap()
        self.samples: list[dict] = []
        self.context: dict = {}            # identifiers added to every sampled step (decision, rollout)

    def _gap(self) -> int:
        """Number of steps to skip before the next stored step (geometric distribution)."""
        if self.rate <= 0.0:
            return 1 << 62
        if self.rate >= 1.0:
            return 0
        return int(math.log(1.0 - self._rnd.random()) / self._log1m)

    @property
    def mean_entropy(self) -> float:
        return self.h_sum / self.h_cnt if self.h_cnt else float("nan")

    def store(self, policy: "RolloutPolicy", state: State, p: np.ndarray, index: int, hbar: float) -> None:
        """Store one rollout step: the distribution, the operation chosen and its relation to MWKR."""
        self.skip = self._gap()
        inst = state.inst
        active = state.active
        work = [inst.rem[j][state.next_op[j]] for j in active]
        chosen_work = work[index]
        record = dict(self.context)
        record.update(
            depth=state.n_done,
            n_candidates=len(active),
            probs=[round(float(v), 6) for v in p],
            chosen_job=active[index],
            chosen_prob=float(p[index]),
            # rank of the chosen operation under MWKR: 1 = most work remaining
            mwkr_rank=1 + sum(1 for w in work if w > chosen_work),
            norm_entropy=hbar,
            agree_mwkr=bool(int(np.argmax(p)) == work.index(max(work))),
        )
        ref = policy.reference
        if ref is not None and policy.level is not None and policy.level > 0.0:
            # Equation 2.22 with the MWKR distribution of the same entropy level as pi'
            record["kl_mwkr"] = kl_divergence(p, ref(state))
        self.samples.append(record)


class RolloutPolicy:
    """Base class of U, MWKR, LR and AZ."""

    name: str = "?"
    level: Optional[float] = None          # target normalised entropy; None is not used
    tau: Optional[float] = None            # temperature of the softmax (None for the greedy rule and U)
    #: MWKR distribution of the same entropy level, used as pi' of Equation 2.22 when steps are recorded
    reference: Optional[Callable[[State], np.ndarray]] = None
    #: uniform sampling draws a random number even when a single job is left (as its fast rollout does)
    _draw_when_forced: bool = False

    # --------------------------------------------------------------------------- to be implemented
    def rollout(self, state: State, rnd: random.Random, trace: list) -> int:
        """Complete ``state`` in place, append the jobs chosen to ``trace`` and return the makespan."""
        raise NotImplementedError

    def distribution(self, state: State) -> np.ndarray:
        """Probabilities over ``state.active`` (in that order) at the entropy level of the policy."""
        raise NotImplementedError

    def scores(self, state: State) -> np.ndarray:
        """Scores ``f(s, a)`` of Equation 2.20 over ``state.active``, before the temperature is applied."""
        raise NotImplementedError

    # ----------------------------------------------------------------------------- recorded rollout
    def _begin(self, state: State) -> None:
        """Hook called before a recorded rollout (the learned distributions attach their observation)."""

    def _after_step(self, state: State) -> None:
        """Hook called after every step of a recorded rollout."""

    def _choose(self, state: State, rnd: random.Random) -> tuple[int, np.ndarray]:
        """Index of the job chosen among ``state.active`` and the distribution it was drawn from.

        Must consume the random numbers exactly as :meth:`rollout` does.
        """
        raise NotImplementedError

    def rollout_recorded(self, state: State, rnd: random.Random, trace: list, stats: StepStats) -> int:
        active = state.active
        self._begin(state)
        while active:
            k = len(active)
            if k == 1:
                if self._draw_when_forced:
                    rnd.random()
                j = active[0]
            else:
                index, p = self._choose(state, rnd)
                nz = p[p > 0]
                hbar = float(-(nz * np.log(nz)).sum()) / math.log(k)
                stats.h_sum += hbar
                stats.h_cnt += 1
                stats.skip -= 1
                if stats.skip < 0:
                    stats.store(self, state, p, index, hbar)
                j = active[index]
            state.step(j)
            trace.append(j)
            self._after_step(state)
        return state.makespan

    # ------------------------------------------------------------------------------------- helpers
    @property
    def label(self) -> str:
        return self.name if self.level is None else f"{self.name}@H{self.level:g}"

    def describe(self) -> dict:
        return {"name": self.name, "level": self.level, "tau": self.tau}
