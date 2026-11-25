"""Explicit dynamics model of the job shop (Section 3.3).

The state is the partial schedule, kept like a Gantt chart: for every machine the operations scheduled on it
with their start and end times, and for every job its next operation.  A step receives a job and schedules
its next operation at the earliest time allowed by its job predecessor and its machine, inserting it into an
idle interval if one is long enough (a left shift), so that every schedule built is active.  A second
function executes a complete schedule with realised processing times for the stochastic setting (RQ5).

The model is written for speed in pure Python: rollouts call :meth:`State.step` millions of times.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Sequence

import numpy as np

from ..instances.instance import Instance


class State:
    """A partial schedule.  ``step`` mutates the state; use :meth:`copy` to branch."""

    __slots__ = ("inst", "next_op", "job_ready", "m_start", "m_end", "m_op", "op_start", "op_end",
                 "makespan", "n_done", "active", "last")

    def __init__(self, inst: Instance):
        self.inst = inst
        n, m = inst.n, inst.m
        self.next_op = [0] * n                      # index of the next unscheduled operation of each job
        self.job_ready = [0] * n                    # completion time of the last scheduled operation
        self.m_start = [[] for _ in range(m)]       # per machine: start times in processing order
        self.m_end = [[] for _ in range(m)]         # per machine: end times in processing order
        self.m_op = [[] for _ in range(m)]          # per machine: operations in processing order
        self.op_start = [0] * (n * m)
        self.op_end = [0] * (n * m)
        self.makespan = 0
        self.n_done = 0
        self.active = list(range(n))                # unfinished jobs = the legal actions A(s), ascending
        self.last = None                            # (operation, machine, position) of the last step

    # --------------------------------------------------------------------------------------------- copy
    def copy(self) -> "State":
        c = State.__new__(State)
        c.inst = self.inst
        c.next_op = self.next_op[:]
        c.job_ready = self.job_ready[:]
        c.m_start = [x[:] for x in self.m_start]
        c.m_end = [x[:] for x in self.m_end]
        c.m_op = [x[:] for x in self.m_op]
        c.op_start = self.op_start[:]
        c.op_end = self.op_end[:]
        c.makespan = self.makespan
        c.n_done = self.n_done
        c.active = self.active[:]
        c.last = self.last
        return c

    # --------------------------------------------------------------------------------------------- step
    def step(self, j: int) -> None:
        """Dispatch the next operation of job ``j`` at its earliest feasible time (with insertion)."""
        inst = self.inst
        k = self.next_op[j]
        mc = inst.mach[j][k]
        p = inst.dur[j][k]
        s = self.job_ready[j]
        starts = self.m_start[mc]
        ends = self.m_end[mc]
        # Intervals that end at or before the job is ready cannot delay the operation: skip them.
        i = bisect_right(ends, s)
        n_iv = len(starts)
        while i < n_iv:
            if s + p <= starts[i]:          # fits into the idle interval before operation i
                break
            e = ends[i]
            if e > s:
                s = e
            i += 1
        e = s + p
        op = j * inst.m + k
        if i == n_iv:
            starts.append(s)
            ends.append(e)
            self.m_op[mc].append(op)
        else:
            starts.insert(i, s)
            ends.insert(i, e)
            self.m_op[mc].insert(i, op)
        self.op_start[op] = s
        self.op_end[op] = e
        self.job_ready[j] = e
        k += 1
        self.next_op[j] = k
        if k == inst.m:
            self.active.remove(j)
        if e > self.makespan:
            self.makespan = e
        self.n_done += 1
        self.last = (op, mc, i)

    # ----------------------------------------------------------------------------------------- queries
    @property
    def complete(self) -> bool:
        return self.n_done == self.inst.n_ops

    def start_times(self) -> np.ndarray:
        """Start times as an ``n x m`` array (only meaningful for scheduled operations)."""
        return np.array(self.op_start, dtype=np.int64).reshape(self.inst.n, self.inst.m)

    def machine_orders(self) -> list[list[int]]:
        """For every machine the operations in processing order (the solution of a complete schedule)."""
        return [ops[:] for ops in self.m_op]


def replay(inst: Instance, sequence: Iterable[int]) -> State:
    """Build the schedule obtained by dispatching the jobs in ``sequence``."""
    state = State(inst)
    for j in sequence:
        state.step(j)
    return state


def execute(inst: Instance, machine_orders: Sequence[Sequence[int]], durations: Sequence[int],
            ends_out: list | None = None) -> int:
    """Realised makespan of a complete schedule under the processing times ``durations`` (Section 3.3).

    Building a schedule fixes the order of the operations on every machine.  During execution every
    operation starts as soon as its job predecessor and its machine predecessor have completed, so the
    makespan follows the realised processing times.  ``durations`` is indexed by operation (``j * m + k``).
    With the nominal durations the function reproduces the nominal makespan (validation, Section 3.7).
    If ``ends_out`` is a list, it receives the realised completion time of every operation.
    """
    m = inst.m
    n_ops = inst.n_ops
    mpred = [-1] * n_ops
    msucc = [-1] * n_ops
    scheduled = 0
    for ops in machine_orders:
        prev = -1
        for op in ops:
            if prev >= 0:
                mpred[op] = prev
                msucc[prev] = op
            prev = op
        scheduled += len(ops)
    if scheduled != n_ops:
        raise ValueError("execute() needs a complete schedule")
    indeg = [0] * n_ops
    stack = []
    for op in range(n_ops):
        d = (1 if op % m else 0) + (1 if mpred[op] >= 0 else 0)
        indeg[op] = d
        if d == 0:
            stack.append(op)
    end = [0] * n_ops
    cmax = 0
    done = 0
    while stack:
        op = stack.pop()
        s = end[op - 1] if op % m else 0
        mp = mpred[op]
        if mp >= 0 and end[mp] > s:
            s = end[mp]
        e = s + durations[op]
        end[op] = e
        if e > cmax:
            cmax = e
        done += 1
        nxt = op + 1
        if nxt % m:                         # the job successor exists
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                stack.append(nxt)
        ms = msucc[op]
        if ms >= 0:
            indeg[ms] -= 1
            if indeg[ms] == 0:
                stack.append(ms)
    if done != n_ops:
        raise ValueError("the machine orders are not consistent with the job orders (cycle)")
    if ends_out is not None:
        ends_out[:] = end
    return cmax
