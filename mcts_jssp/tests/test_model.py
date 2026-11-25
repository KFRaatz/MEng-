"""Dynamics model (Section 3.3): insertion, active schedules, execution, and the independent checker."""
from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import random_instance, random_state
from mctsjssp.env import JobShopEnv, assert_valid_schedule, execute, replay, schedule_violations
from mctsjssp.instances import scenario


def reference_schedule(inst, sequence):
    """Slow reference: schedule every operation at the earliest feasible time by trying all gaps."""
    busy = [[] for _ in range(inst.m)]                 # per machine: (start, end)
    ready = [0] * inst.n
    nxt = [0] * inst.n
    start = np.zeros((inst.n, inst.m), dtype=np.int64)
    for j in sequence:
        k = nxt[j]
        mc, p = inst.mach[j][k], inst.dur[j][k]
        intervals = sorted(busy[mc])
        candidates = [ready[j]] + [e for _, e in intervals if e >= ready[j]]
        for s in sorted(candidates):
            # half-open intervals: [s, s + p) must not intersect any busy interval of positive length
            if all(s + p <= a or s >= b for a, b in intervals if b > a) and \
               all(not (a < s < b) for a, b in intervals):
                break
        start[j, k] = s
        busy[mc].append((s, s + p))
        ready[j] = s + p
        nxt[j] += 1
    return start


@pytest.mark.parametrize("n, m, seed", [(3, 3, 0), (5, 4, 1), (8, 3, 2), (6, 6, 3), (10, 5, 4), (4, 9, 5)])
def test_step_matches_the_reference_insertion(n, m, seed):
    inst = random_instance(n, m, seed, low=1, high=15)
    rnd = random.Random(seed)
    for _ in range(20):
        sequence = [j for j in range(n) for _ in range(m)]
        rnd.shuffle(sequence)
        state = replay(inst, sequence)
        assert np.array_equal(state.start_times(), reference_schedule(inst, sequence))
        assert state.makespan == int((state.start_times() + inst.durations).max())


def test_hand_checked_schedule(tiny):
    # jobs dispatched in the order 0,1,2,3,0,1,2,3,0,1,2,3
    state = replay(tiny, [0, 1, 2, 3] * 3)
    start = state.start_times()
    # machine 0: J0 op0 [0,3), J3 op0 [3,4); machine 1: J1 op0 [0,2); machine 2: J2 op0 [0,4)
    assert list(start[:, 0]) == [0, 0, 0, 3]
    # second operations: J0 on m1 after J1 -> 3; J1 on m0 (zero time) at 2 fits before J3? machine 0 is busy
    # [0,3) so it starts at 3; J2 on m1 after J0 [3,5) -> 5; J3 on m2 after J2 op0 -> 4
    assert list(start[:, 1]) == [3, 3, 5, 4]
    assert_valid_schedule(tiny, start, state.makespan)
    assert state.complete and state.active == []


def test_insertion_uses_idle_intervals_left_shift():
    from mctsjssp.instances import Instance

    # machine 0 is needed late by job 0 and early by job 1
    inst = Instance("gap", [[1, 0], [0, 1]], [[5, 2], [3, 1]])
    state = replay(inst, [0, 0, 1, 1])
    start = state.start_times()
    assert start[0, 1] == 5                      # job 0 reaches machine 0 at time 5
    assert start[1, 0] == 0                      # job 1 is inserted into the idle interval [0, 5) before it
    assert state.m_op[0] == [2, 1]               # machine order differs from the dispatching order
    assert state.makespan == 7


@pytest.mark.parametrize("seed", range(6))
def test_schedules_are_active(seed):
    """No operation can start earlier without delaying another one (local left shift impossible).

    In the disjunctive formulation two operations i and j of a machine satisfy ``s_j >= s_i + p_i`` or
    ``s_i >= s_j + p_j``.  An operation with zero processing time is therefore still a point that no other
    operation may contain, which is how the model and the checker treat it.
    """
    inst = random_instance(6, 5, seed, low=0, high=12)
    state = random_state(inst, inst.n_ops, seed)
    start = state.start_times()
    end = start + inst.durations
    for j in range(inst.n):
        for k in range(inst.m):
            ready = end[j, k - 1] if k else 0
            p = inst.durations[j, k]
            others = sorted((start[a, b], end[a, b]) for a in range(inst.n) for b in range(inst.m)
                            if inst.machines[a, b] == inst.machines[j, k] and (a, b) != (j, k))
            earliest = ready
            for s, e in others:
                if earliest + p <= s:
                    break
                if e > earliest:
                    earliest = e
            assert earliest == start[j, k], (j, k)
            assert start[j, k] >= ready


def test_copy_is_independent(ft06):
    state = random_state(ft06, 10, 1)
    clone = state.copy()
    snapshot = (state.next_op[:], [x[:] for x in state.m_op], state.makespan, state.active[:])
    while clone.active:
        clone.step(clone.active[0])
    assert (state.next_op, state.m_op, state.makespan, state.active) == \
           (snapshot[0], snapshot[1], snapshot[2], snapshot[3])
    assert clone.complete and not state.complete


@pytest.mark.parametrize("name", ["ft06", "la01", "orb07", "la36", "swv11"])
def test_random_schedules_pass_the_checker_and_execution_reproduces_them(orlib, name):
    inst = orlib[name]
    nominal = inst.durations.ravel().tolist()
    for seed in range(5):
        state = random_state(inst, inst.n_ops, seed)
        assert schedule_violations(inst, state.start_times(), state.makespan) == []
        assert execute(inst, state.m_op, nominal) == state.makespan
        ends: list = []
        execute(inst, state.m_op, nominal, ends_out=ends)
        assert ends == state.op_end


def test_execution_under_disturbances_is_feasible_and_monotone(orlib):
    inst = orlib["la16"]
    state = random_state(inst, inst.n_ops, 3)
    realised = scenario(inst, 0.2, 0)
    ends: list = []
    cmax = execute(inst, state.m_op, realised, ends_out=ends)
    dur = np.array(realised).reshape(inst.n, inst.m)
    start = np.array(ends).reshape(inst.n, inst.m) - dur
    assert schedule_violations(inst, start, cmax, durations=dur) == []
    assert cmax >= dur.sum(axis=1).max()
    # the order of the operations on every machine is the one of the nominal schedule
    for mc, ops in enumerate(state.m_op):
        assert sorted(ops, key=lambda op: (start.ravel()[op], np.array(ends)[op])) == ops or \
            min(dur.ravel()[op] for op in ops) == 0
    longer = [d + 1 for d in realised]
    assert execute(inst, state.m_op, longer) >= cmax


def test_execute_needs_a_complete_schedule(ft06):
    partial = random_state(ft06, 5, 0)
    with pytest.raises(ValueError, match="complete schedule"):
        execute(ft06, partial.m_op, ft06.durations.ravel().tolist())


def test_checker_detects_violations(ft06):
    state = random_state(ft06, ft06.n_ops, 0)
    start = state.start_times()
    assert schedule_violations(ft06, start, state.makespan) == []
    assert any("makespan" in v for v in schedule_violations(ft06, start, state.makespan + 1))
    early = start.copy()
    early[0, 1] = max(0, early[0, 0] + ft06.dur[0][0] - 1)          # starts before its job predecessor ends
    assert any("before operation" in v for v in schedule_violations(ft06, early))
    clash = start.copy()
    j, k = np.argwhere(ft06.machines == 0)[0]
    j2, k2 = np.argwhere(ft06.machines == 0)[1]
    clash[j2, k2] = clash[j, k]                                      # two operations at once on machine 0
    assert any("overlap" in v for v in schedule_violations(ft06, clash))
    with pytest.raises(AssertionError):
        assert_valid_schedule(ft06, clash)


def test_environment_wraps_the_model(ft06):
    env = JobShopEnv(ft06)
    state = env.reset()
    assert env.legal_actions() == list(range(6)) and env.action_mask().all()
    reward, done = env.step(2)
    assert (reward, done) == (0.0, False) and env.sequence == [2]
    with pytest.raises(ValueError):
        env.step(17)
    rnd = random.Random(0)
    while not env.done:
        reward, done = env.step(rnd.choice(env.legal_actions()))
    assert done and reward == pytest.approx(-env.makespan / ft06.lb0)
    assert env.legal_actions() == [] and not env.action_mask().any()
    assert env.schedule().shape == (6, 6)
    assert env.execute(ft06.durations.ravel().tolist()) == env.makespan
    assert env.state is state                    # the environment keeps one state; searches copy it
    with pytest.raises(ValueError):
        env.step(0)
