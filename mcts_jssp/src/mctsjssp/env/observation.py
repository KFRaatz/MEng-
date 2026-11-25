"""Observations of the partial schedule (Section 3.4.1).

Every operation carries two features: a binary indicator of whether it is scheduled and the lower bound of
its completion time, normalised by the initial lower bound LB0.  For an unscheduled operation the bound is
the completion time of the last scheduled operation of its job plus the processing times up to and including
the operation; for a scheduled operation it is the actual completion time.

The GIN reads the features on the disjunctive graph: conjunctive arcs from every operation to its job
successor and, for scheduled operations, arcs from the machine predecessor.  The fully connected networks of
AZ read the same features as one vector of length ``2 * n * m`` in the fixed order of the operations.

:class:`FeatureTracker` keeps these arrays for one state and updates them in place after every step, so a
rollout does not rebuild the observation from scratch.
"""
from __future__ import annotations

import numpy as np

from ..instances.instance import Instance
from .model import State


class FeatureTracker:
    """Features and arcs of one state, updated incrementally.

    Attributes
    ----------
    x : ``(n_ops, 2)`` float32 array; column 0 is the scheduled flag, column 1 is LB / LB0.
    xp : the same features with one extra row of zeros (``xp[n_ops]``), so that "no predecessor" can be
        gathered like any other node; ``x`` is a view of its first ``n_ops`` rows.
    mpred : ``(n_ops,)`` int64 array; machine predecessor of every operation, ``n_ops`` if it has none.
    msucc : ``(n_ops,)`` int64 array; machine successor, ``n_ops`` if it has none.
    jpred : ``(n_ops,)`` int64 array; job predecessor, ``n_ops`` if it has none (constant).
    """

    __slots__ = ("inst", "x", "xp", "mpred", "msucc", "jpred", "_cumn", "_rows")

    def __init__(self, inst: Instance):
        self.inst = inst
        self.xp = np.zeros((inst.n_ops + 1, 2), dtype=np.float32)
        self.x = self.xp[: inst.n_ops]
        self.mpred = np.full(inst.n_ops, inst.n_ops, dtype=np.int64)
        self.msucc = np.full(inst.n_ops, inst.n_ops, dtype=np.int64)
        self.jpred = inst.job_pred
        self._cumn = (inst.cum / inst.lb0).astype(np.float32)   # cumulative durations of each job / LB0
        self._rows = np.arange(inst.n)

    # ------------------------------------------------------------------------------------------- reset
    def reset(self, state: State) -> "FeatureTracker":
        """Compute the observation of ``state`` from scratch."""
        inst = self.inst
        n, m = inst.n, inst.m
        lb0 = inst.lb0
        nxt = np.array(state.next_op, dtype=np.int64)
        ready = np.array(state.job_ready, dtype=np.float32) / lb0
        ends = np.array(state.op_end, dtype=np.float32).reshape(n, m) / lb0
        scheduled = np.arange(m)[None, :] < nxt[:, None]
        # cumulative duration of the operations of each job that are already scheduled
        before = np.where(nxt > 0, self._cumn[self._rows, np.maximum(nxt - 1, 0)], 0.0).astype(np.float32)
        lb = np.where(scheduled, ends, (ready - before)[:, None] + self._cumn)
        self.x[:, 0] = scheduled.ravel()
        self.x[:, 1] = lb.ravel()
        self.mpred[:] = inst.n_ops
        self.msucc[:] = inst.n_ops
        for ops in state.m_op:
            if len(ops) > 1:
                self.mpred[ops[1:]] = ops[:-1]
                self.msucc[ops[:-1]] = ops[1:]
        return self

    # ------------------------------------------------------------------------------------------ update
    def update(self, state: State) -> None:
        """Update the observation after ``state.step``; uses ``state.last``."""
        op, mc, pos = state.last
        m = self.inst.m
        j, k = divmod(op, m)
        e = state.op_end[op] / self.inst.lb0
        x = self.x
        x[op, 0] = 1.0
        cum = self._cumn[j]
        x[op: (j + 1) * m, 1] = e + (cum[k:] - cum[k])
        ops = state.m_op[mc]
        if pos > 0:
            self.mpred[op] = ops[pos - 1]
            self.msucc[ops[pos - 1]] = op
        if pos + 1 < len(ops):
            self.mpred[ops[pos + 1]] = op
            self.msucc[op] = ops[pos + 1]

    # ------------------------------------------------------------------------------------------- views
    def flat(self) -> np.ndarray:
        """The observation as one vector of length ``2 * n * m`` (input of the networks of AZ)."""
        return self.x.reshape(-1)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Copies of the features and machine predecessors, for the replay buffer."""
        return self.x.copy(), self.mpred.astype(np.int32)


def candidate_ops(state: State) -> list[int]:
    """Operation dispatched by every legal action (the next operation of every unfinished job)."""
    m = state.inst.m
    nxt = state.next_op
    return [j * m + nxt[j] for j in state.active]
