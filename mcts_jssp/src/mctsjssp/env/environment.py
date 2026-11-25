"""The job shop environment (Section 3.4).

The environment keeps the current partial schedule and exposes it to the agent as observations, legal
actions and a reward.  It does not compute any dynamics itself: every transition is delegated to the
dynamics model (:mod:`mctsjssp.env.model`), so that the search, the execution of decisions and the training
of the networks use the same scheduling logic.

Following L2D, scheduling is an MDP in which one operation is dispatched at each step.  An action is a job;
finished jobs are masked.  The only reward is given when the schedule is complete: ``R = -Cmax / LB0``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..instances.instance import Instance
from .checker import assert_valid_schedule
from .model import State, execute
from .observation import FeatureTracker


class JobShopEnv:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.state = State(inst)
        self.sequence: list[int] = []
        self._tracker: Optional[FeatureTracker] = None

    # ----------------------------------------------------------------------------------------- episode
    def reset(self) -> State:
        self.state = State(self.inst)
        self.sequence = []
        if self._tracker is not None:
            self._tracker.reset(self.state)
        return self.state

    def step(self, job: int) -> tuple[float, bool]:
        """Dispatch the next operation of ``job``; returns ``(reward, done)``."""
        if job not in self.state.active:
            raise ValueError(f"job {job} is finished or unknown; legal actions are {self.state.active}")
        self.state.step(job)
        self.sequence.append(job)
        if self._tracker is not None:
            self._tracker.update(self.state)
        done = self.state.complete
        return (self.reward() if done else 0.0), done

    # ------------------------------------------------------------------------------------ MDP elements
    @property
    def done(self) -> bool:
        return self.state.complete

    def legal_actions(self) -> list[int]:
        """Jobs with an unscheduled operation, A(s); at most ``n`` actions."""
        return list(self.state.active)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.inst.n, dtype=bool)
        mask[self.state.active] = True
        return mask

    def observation(self) -> FeatureTracker:
        """Features and arcs of the current state (created on first use, then updated at every step)."""
        if self._tracker is None:
            self._tracker = FeatureTracker(self.inst).reset(self.state)
        return self._tracker

    def reward(self) -> float:
        """Return of the complete schedule, ``R = -Cmax / LB0`` (Section 3.4.3)."""
        if not self.state.complete:
            return 0.0
        return -self.state.makespan / self.inst.lb0

    # ---------------------------------------------------------------------------------------- results
    @property
    def makespan(self) -> int:
        return self.state.makespan

    def schedule(self) -> np.ndarray:
        """Start times of the complete schedule as an ``n x m`` array, checked independently."""
        start = self.state.start_times()
        assert_valid_schedule(self.inst, start, self.state.makespan)
        return start

    def execute(self, durations: Sequence[int]) -> int:
        """Realised makespan of the complete schedule under the given scenario (Section 3.3)."""
        return execute(self.inst, self.state.m_op, durations)
