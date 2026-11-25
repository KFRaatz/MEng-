"""Observations (Section 3.4.1): scheduled flag, normalised lower bound and the arcs of the disjunctive graph."""
from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import random_instance
from mctsjssp.env import FeatureTracker, State, candidate_ops


def test_initial_features_are_cumulative_durations_over_lb0(tiny):
    tracker = FeatureTracker(tiny).reset(State(tiny))
    assert tracker.x.shape == (12, 2)
    assert not tracker.x[:, 0].any()                                   # nothing is scheduled
    expected = np.cumsum(tiny.durations, axis=1).ravel() / tiny.lb0     # LB(O_jk) = sum of p_j1..p_jk
    assert np.allclose(tracker.x[:, 1], expected)
    assert (tracker.mpred == 12).all() and (tracker.msucc == 12).all()  # no machine arcs yet
    assert list(tracker.jpred) == [12, 0, 1, 12, 3, 4, 12, 6, 7, 12, 9, 10]
    assert tracker.flat().shape == (24,)


def test_lower_bound_becomes_the_completion_time_and_propagates(tiny):
    state = State(tiny)
    tracker = FeatureTracker(tiny).reset(state)
    for job in (0, 3):                       # job 3 waits for machine 0 until time 3
        state.step(job)
        tracker.update(state)
    lb = tracker.x[:, 1] * tiny.lb0
    assert tracker.x[9, 0] == 1.0 and lb[9] == pytest.approx(4)          # scheduled: actual completion time
    assert lb[10] == pytest.approx(4 + 6) and lb[11] == pytest.approx(4 + 6 + 2)
    assert lb[3] == pytest.approx(2)                                      # other jobs are unchanged
    assert tracker.mpred[9] == 0 and tracker.msucc[0] == 9                # machine arc J0-op0 -> J3-op0


@pytest.mark.parametrize("n, m, seed", [(4, 3, 0), (6, 6, 1), (10, 5, 2), (5, 10, 3)])
def test_incremental_update_equals_recomputation(n, m, seed):
    inst = random_instance(n, m, seed)
    rnd = random.Random(seed)
    state = State(inst)
    live = FeatureTracker(inst).reset(state)
    fresh = FeatureTracker(inst)
    while state.active:
        state.step(rnd.choice(state.active))
        live.update(state)
        fresh.reset(state)
        assert np.allclose(live.x, fresh.x, atol=1e-5)
        assert np.array_equal(live.mpred, fresh.mpred) and np.array_equal(live.msucc, fresh.msucc)
    assert live.x[:, 0].all()
    assert np.allclose(live.x[:, 1] * inst.lb0, state.op_end, atol=1e-3)   # all bounds are completion times


def test_machine_arcs_follow_the_machine_orders(ft06):
    rnd = random.Random(5)
    state = State(ft06)
    tracker = FeatureTracker(ft06).reset(state)
    for _ in range(25):
        state.step(rnd.choice(state.active))
        tracker.update(state)
    for ops in state.m_op:
        for a, b in zip(ops, ops[1:]):
            assert tracker.mpred[b] == a and tracker.msucc[a] == b
        if ops:
            assert tracker.mpred[ops[0]] == ft06.n_ops and tracker.msucc[ops[-1]] == ft06.n_ops
    scheduled = {op for ops in state.m_op for op in ops}
    assert all(tracker.mpred[op] == ft06.n_ops for op in range(ft06.n_ops) if op not in scheduled)
    assert candidate_ops(state) == [j * ft06.m + state.next_op[j] for j in state.active]


def test_snapshot_is_a_copy(ft06):
    state = State(ft06)
    tracker = FeatureTracker(ft06).reset(state)
    x, mpred = tracker.snapshot()
    state.step(0)
    tracker.update(state)
    assert x[0, 0] == 0.0 and tracker.x[0, 0] == 1.0
    assert mpred.dtype == np.int32
