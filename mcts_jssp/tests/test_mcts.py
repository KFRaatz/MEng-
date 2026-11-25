"""The MCTS agent (Section 3.2.1, Table 3.1): UCT with rollouts, PUCT for self-play, budgets and recording."""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

from conftest import random_instance, random_state
from mctsjssp.distributions import MWKRRollout, NetworkRollout, UniformRollout
from mctsjssp.env import State, assert_valid_schedule, replay
from mctsjssp.mcts import Budget, Node, PUCTSearch, RunRecorder, UCTSearch, make_return, solve
from mctsjssp.networks import make_inference


def searched(inst, iterations, seed=0, policy=None, c=1.0, state=None):
    state = state or State(inst)
    search = UCTSearch(policy or UniformRollout(), make_return(inst), c=c, seed=seed)
    search.start(state)
    search.run_iterations(state, iterations)
    return search, state


def all_nodes(root):
    stack, out = [root], []
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


# -------------------------------------------------------------------------------------------- the tree
def test_untried_operations_have_priority(ft06):
    """All children of a node are added before any of them is visited a second time."""
    search, _ = searched(ft06, 6)
    assert sorted(c.action for c in search.root.children) == list(range(6))
    assert all(c.N == 1 for c in search.root.children) and search.root.untried == []
    search, _ = searched(ft06, 4)
    assert len(search.root.children) == 4 and len(search.root.untried) == 2


def test_one_node_per_iteration_and_consistent_statistics(ft06):
    search, _ = searched(ft06, 150, seed=3)
    nodes = all_nodes(search.root)
    assert len(nodes) == 151                                   # the root and one new node per iteration
    assert search.root.N == 150 and search.iterations == 150
    for node in nodes:
        if node.children:
            # every visit of a node went on to one child, except the visit that created the node
            assert sum(c.N for c in node.children) == node.N - (0 if node is search.root else 1)
        assert node.Q == pytest.approx(node.W / node.N)
    # returns are R = -Cmax / LB0, at most -1 because LB0 is a lower bound
    assert all(-5.0 < n.Q <= -1.0 for n in nodes)


def test_backup_adds_the_same_value_on_the_whole_path(ft06):
    search, state = searched(ft06, 6, seed=1)
    total = sum(c.W for c in search.root.children)
    before = {c.action: c.W for c in search.root.children}
    assert search.root.W == pytest.approx(total)               # depth-1 paths: root and child get the same v
    search.run_iterations(state, 1)                            # the seventh iteration goes two levels deep
    deep = [c for c in search.root.children if c.children]
    assert len(deep) == 1 and deep[0].N == 2
    grandchild = deep[0].children[0]
    assert grandchild.N == 1
    assert search.root.W == pytest.approx(total + grandchild.W)
    assert deep[0].W == pytest.approx(before[deep[0].action] + grandchild.W)


def test_uct_formula_selects_the_expected_child(ft06):
    search, state = searched(ft06, 6, seed=2)
    root = search.root
    scores = {c.action: c.Q + 1.0 * math.sqrt(math.log(root.N) / c.N) for c in root.children}
    expected = max(scores, key=scores.get)
    before = {c.action: c.N for c in root.children}
    search.run_iterations(state, 1)
    grown = [c.action for c in root.children if c.N > before[c.action]]
    assert grown == [expected]


def test_exploration_constant_changes_the_visit_distribution(ft06):
    low, _ = searched(ft06, 300, seed=5, c=0.01)
    high, _ = searched(ft06, 300, seed=5, c=10.0)
    share = lambda s: max(c.N for c in s.root.children) / s.root.N      # noqa: E731
    assert share(low) > share(high)                           # a small c concentrates the visits
    assert share(high) < 0.25


def test_search_does_not_change_the_environment_state(ft06):
    state = random_state(ft06, 9, 0)
    snapshot = (state.next_op[:], state.makespan, state.n_done, [x[:] for x in state.m_op])
    searched(ft06, 80, state=state)
    assert (state.next_op, state.makespan, state.n_done, state.m_op) == snapshot


def test_complete_schedules_in_the_tree_need_no_rollout():
    inst = random_instance(2, 2, 0, low=1, high=9)             # 4 operations: the tree reaches the leaves
    search, _ = searched(inst, 60)
    terminal = [n for n in all_nodes(search.root) if n.terminal]
    assert terminal and all(not n.children and not n.untried for n in terminal)
    assert max(n.N for n in terminal) > 1                      # visited again, valued by the makespan
    best = min(replay(inst, seq).makespan for seq in ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1],
                                                     [1, 0, 1, 0], [1, 1, 0, 0]))
    assert search.best_makespan == best


def test_most_visited_action_is_committed_with_ties_broken_by_value():
    root = Node()
    for action, (n, w) in enumerate([(5, -6.0), (7, -9.1), (7, -8.4), (3, -3.0)]):
        child = Node(action)
        child.N, child.W = n, w
        root.children.append(child)
    assert root.most_visited().action == 2                     # 7 visits and the higher mean value
    root.children[2].W = root.children[1].W
    assert root.most_visited().action == 1                     # full tie: the lower job index


def test_subtree_is_kept_for_the_next_decision(ft06):
    search, state = searched(ft06, 200, seed=4)
    action = search.choose()
    child = search.root.child(action)
    visits, size = child.N, child.subtree_size()
    state.step(action)
    search.advance(action, state)
    assert search.root is child and search.root.N == visits and search.root.subtree_size() == size
    assert search.prefix == [action]
    search.run_iterations(state, 50)
    assert search.root.N == visits + 50


# --------------------------------------------------------------------------------------------- episodes
def test_solve_returns_a_valid_schedule_and_exact_budget(ft06):
    res = solve(ft06, UniformRollout(), Budget.of_iterations(30), seed=1)
    final = replay(ft06, res.sequence)
    assert final.makespan == res.makespan
    assert_valid_schedule(ft06, final.start_times(), res.makespan)
    assert res.iterations == 30 * res.decisions and set(res.decision_iterations) == {30}
    assert res.decisions <= ft06.n_ops and len(res.decision_times) == res.decisions
    assert res.decision_depths == sorted(res.decision_depths)
    assert res.best_rollout_makespan <= res.makespan or res.best_rollout_makespan >= ft06.bks
    assert replay(ft06, res.best_rollout_sequence).makespan == res.best_rollout_makespan
    assert res.best_rollout_makespan >= ft06.bks and res.makespan >= ft06.bks


def test_solve_is_deterministic_for_a_seed(ft06):
    a = solve(ft06, MWKRRollout(0.5, 0.05), Budget.of_iterations(25), seed=7)
    b = solve(ft06, MWKRRollout(0.5, 0.05), Budget.of_iterations(25), seed=7)
    c = solve(ft06, MWKRRollout(0.5, 0.05), Budget.of_iterations(25), seed=8)
    assert a.sequence == b.sequence and a.makespan == b.makespan
    assert a.sequence != c.sequence


def test_forced_decisions_are_not_searched():
    from mctsjssp.instances import Instance

    inst = Instance("one_long_job", [[0, 1, 2, 3], [0, 1, 2, 3]], [[1, 1, 1, 1], [9, 9, 9, 9]])
    res = solve(inst, UniformRollout(), Budget.of_iterations(10), seed=0)
    assert res.decisions < inst.n_ops                          # once a job is finished nothing is decided
    full = solve(inst, UniformRollout(), Budget.of_iterations(10), seed=0, skip_forced=False)
    assert full.decisions == inst.n_ops


def test_search_improves_on_single_rollouts(orlib):
    """With more iterations U finds better schedules than random dispatching does on average."""
    inst = orlib["la01"]
    rnd = random.Random(0)
    random_mean = np.mean([UniformRollout().rollout(State(inst), rnd, []) for _ in range(200)])
    res = solve(inst, UniformRollout(), Budget.of_iterations(200), seed=0)
    assert res.makespan < random_mean
    assert inst.gap(res.makespan) < 15.0
    assert solve(orlib["ft06"], UniformRollout(), Budget.of_iterations(200), seed=1).makespan <= 60


def test_time_budget(ft06):
    budget = Budget.of_time(0.004, label="T")
    t0 = time.perf_counter()
    res = solve(ft06, UniformRollout(), budget, seed=0)
    elapsed = time.perf_counter() - t0
    assert all(i >= 1 for i in res.decision_iterations)        # an iteration that has started is completed
    assert all(t >= 0.004 for t in res.decision_times)
    assert res.time_search <= elapsed
    assert res.time_per_decision < 0.05
    profile = Budget("time", seconds=1.0, profile=(0.001, 0.0005))
    assert profile.seconds_for(10) == pytest.approx(0.006) and budget.seconds_for(10) == 0.004


def test_learned_rollouts_run_in_the_search(ft06, gin, az66):
    for name, model in (("LR", gin), ("AZ", az66)):
        infer = make_inference(model)
        before = infer.calls
        res = solve(ft06, NetworkRollout(name, infer, 0.5, 0.5), Budget.of_iterations(5), seed=0)
        assert res.makespan >= ft06.bks and infer.calls > before


def test_stochastic_search_samples_scenarios(orlib):
    inst = orlib["la01"]
    static = make_return(inst)
    noisy = make_return(inst, sigma=0.2, seed=1)
    state = random_state(inst, inst.n_ops, 0)
    assert static(state) == pytest.approx(-state.makespan / inst.lb0) and not static.stochastic
    values = {noisy(state) for _ in range(5)}
    assert len(values) > 1 and noisy.stochastic               # a new scenario for every evaluation
    assert make_return(inst, sigma=0.0).stochastic is False
    res = solve(inst, UniformRollout(), Budget.of_iterations(10), seed=2, sigma=0.2)
    assert_valid_schedule(inst, replay(inst, res.sequence).start_times(), res.makespan)


# ------------------------------------------------------------------------------------------- recording
def test_recording_levels_and_contents(ft06):
    plain = solve(ft06, UniformRollout(), Budget.of_iterations(20), seed=3)
    rec = RunRecorder("steps", step_rate=0.1, seed=1)
    res = solve(ft06, UniformRollout(), Budget.of_iterations(20), seed=3, recorder=rec)
    assert res.sequence == plain.sequence                      # recording does not change the search
    assert len(rec.decisions) == res.decisions
    first = rec.decisions[0]
    assert first["depth"] == 0 and first["n_candidates"] == 6 and first["iterations"] == 20
    assert sum(first["visits"]) == 20 and first["rollouts"] == 20
    assert first["action"] in first["actions"] and 0 < first["top_visit_share"] <= 1
    assert 0 <= first["visit_entropy_norm"] <= 1 + 1e-9 and first["return_var"] >= 0
    tables = rec.tables()
    assert len(tables["rollouts"]["return"]) == res.iterations == len(tables["rollouts"]["length"])
    assert (tables["rollouts"]["return"] <= -1.0).all()
    assert tables["steps"] and {"decision", "rollout", "depth", "probs", "mwkr_rank"} <= set(tables["steps"][0])
    summary = res.summary
    assert summary["rollout_entropy"] == pytest.approx(1.0) and summary["sampled_steps"] == len(tables["steps"])
    assert {"visit_entropy_norm", "top_visit_share", "return_var", "return_bias", "value_var"} <= set(summary)
    # level "decisions" stores neither rollouts nor steps
    rec = RunRecorder("decisions")
    solve(ft06, UniformRollout(), Budget.of_iterations(5), seed=3, recorder=rec)
    assert set(rec.tables()) == {"decisions"} and rec.steps is None
    with pytest.raises(ValueError):
        RunRecorder("everything")


# ------------------------------------------------------------------------------------------------- PUCT
def test_puct_expands_with_priors_and_backs_up_values(ft06, gin):
    infer = make_inference(gin)
    search = PUCTSearch(infer, lambda st: 1.0, c=1.0, noise_weight=0.0, seed=0)
    state = State(ft06)
    search.start(state)
    search.run_iterations(state, 40)
    root = search.root
    assert root.expanded and len(root.children) == 6 and root.N == 40
    assert sum(c.P0 for c in root.children) == pytest.approx(1.0)
    assert all(c.P == c.P0 for c in root.children)             # no noise requested
    assert sum(c.N for c in root.children) == 39               # the first iteration evaluated the root itself
    assert search.evaluations <= 40 and all(-1.0 <= n.Q <= 1.0 for n in all_nodes(root))
    pi = search.visit_distribution(ft06.n)
    assert pi.sum() == pytest.approx(1.0) and pi.shape == (6,)
    # the first selection follows the priors: Q = 0 for every child, so the largest prior wins
    fresh = PUCTSearch(infer, lambda st: 1.0, noise_weight=0.0, seed=0)
    fresh.start(state)
    fresh.run_iterations(state, 2)
    visited = [c for c in fresh.root.children if c.N]
    assert len(visited) == 1 and visited[0].P0 == max(c.P0 for c in fresh.root.children)


def test_dirichlet_noise_only_at_the_root(ft06, gin):
    search = PUCTSearch(make_inference(gin), lambda st: 0.0, noise_weight=0.25, noise_alpha_scale=10.0, seed=1)
    state = State(ft06)
    search.start(state)
    search.run_iterations(state, 60)
    root = search.root
    assert any(abs(c.P - c.P0) > 1e-6 for c in root.children)
    assert sum(c.P for c in root.children) == pytest.approx(1.0)
    assert all(c.P >= 0.75 * c.P0 - 1e-12 for c in root.children)        # weight 0.25
    below = [g for c in root.children for g in c.children]
    assert below and all(g.P == g.P0 for g in below)
    # after the move the new root receives fresh noise from its own priors
    action = search.choose()
    state.step(action)
    search.advance(action, state)
    search.run_iterations(state, 10)
    assert any(abs(c.P - c.P0) > 1e-6 for c in search.root.children)


def test_puct_values_complete_schedules_with_the_terminal_function(gin):
    inst = random_instance(2, 2, 3, low=1, high=9)
    calls = []

    def terminal(st):
        calls.append(st.makespan)
        return -1.0

    search = PUCTSearch(make_inference(gin), terminal, noise_weight=0.0, seed=0)
    state = State(inst)
    search.start(state)
    search.run_iterations(state, 80)
    assert calls and all(st_makespan > 0 for st_makespan in calls)
    assert any(n.terminal for n in all_nodes(search.root))
