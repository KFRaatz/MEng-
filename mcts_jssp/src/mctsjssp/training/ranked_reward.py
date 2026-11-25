"""Ranked reward for single-agent self-play (Section 3.6.1, after Laterre et al.).

An episode receives ``z = +1`` if its outcome ``Cmax / LB0`` is better than a percentile of the recent
outcomes of the agent and ``z = -1`` otherwise.  With the 75th percentile an episode must beat 75 % of the
last 250 results to be rewarded.

The threshold is frozen for one generation of self-play: it is computed from the results before the
generation and used for the complete schedules reached inside the search trees and for the labels of the
episodes; the results of the generation are added afterwards.  This keeps a generation independent of the
order in which its episodes finish, so they can be played in parallel with the same result.  Generation 0
(searches of U) has no earlier results and is ranked against its own outcomes; so is, by default, the first
generation that searches with the networks, after which the window holds only results of the networks (see
:meth:`mctsjssp.training.trainer.Trainer._store`).
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np


class RankedReward:
    def __init__(self, percentile: float = 75.0, window: int = 250, history: Optional[Iterable[float]] = None):
        if not 0.0 < percentile < 100.0:
            raise ValueError("the percentile must lie strictly between 0 and 100")
        self.percentile = float(percentile)
        self.window = int(window)
        self.history: deque = deque(history or [], maxlen=self.window)   # recent outcomes Cmax / LB0

    def threshold(self, outcomes: Optional[Iterable[float]] = None) -> Optional[float]:
        """The outcome ``Cmax / LB0`` that an episode must beat (lower is better); None without results.

        The ``percentile`` refers to the returns ``-Cmax / LB0``: beating the 75th percentile of the returns
        means a ratio below the 25th percentile of the ratios.
        """
        values = list(self.history) if outcomes is None else list(outcomes)
        if not values:
            return None
        return float(np.percentile(values, 100.0 - self.percentile))

    @staticmethod
    def label(outcome: float, threshold: float) -> float:
        """``+1`` if the outcome is better (smaller) than the threshold, ``-1`` otherwise."""
        return 1.0 if outcome < threshold else -1.0

    def update(self, outcomes: Iterable[float]) -> None:
        self.history.extend(float(v) for v in outcomes)

    def state(self) -> dict:
        return {"percentile": self.percentile, "window": self.window, "history": list(self.history)}
