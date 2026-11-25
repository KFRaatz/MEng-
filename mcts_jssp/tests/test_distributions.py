"""Rollout distributions, entropy, Kullback-Leibler divergence and the temperatures of the entropy levels."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from conftest import random_instance, random_state
from mctsjssp.distributions import (MWKRRollout, NetworkRollout, StepStats, UniformRollout, entropy, greedy,
                                    kl_divergence, normalised_entropy, same_top, softmax)
from mctsjssp.distributions.calibration import calibrate_tau, mean_rollout_entropy
from mctsjssp.env import State
from mctsjssp.instances import validation_set
from mctsjssp.networks import make_inference


# ------------------------------------------------------------------------------------------------ maths
def test_softmax_limits():
    scores = np.array([1.0, 2.0, 4.0])
    assert np.allclose(softmax(scores, math.inf), 1 / 3)
    assert softmax(scores, 1e-3)[2] == pytest.approx(1.0)
    p = softmax(scores, 1.0)
    assert p.sum() == pytest.approx(1.0) and np.allclose(p, np.exp(scores) / np.exp(scores).sum())
    assert np.allclose(softmax(scores + 100.0, 0.7), softmax(scores, 0.7))       # shift invariant
    with pytest.raises(ValueError):
        softmax(scores, 0.0)
    assert list(greedy(np.array([0.2, 0.9, 0.9]))) == [0.0, 1.0, 0.0]            # first maximum


def test_entropy_of_uniform_and_deterministic_distributions():
    for k in (2, 15):
        assert entropy(np.full(k, 1 / k)) == pytest.approx(math.log(k))
        assert normalised_entropy(np.full(k, 1 / k)) == pytest.approx(1.0)
    # the example of Section 3.2.2: ln 2 = 0.69 is fully random for 2 candidates but focused for 15
    assert math.log(2) == pytest.approx(0.69, abs=0.01) and math.log(15) == pytest.approx(2.71, abs=0.01)
    assert entropy(np.array([0.0, 1.0, 0.0])) == 0.0 and normalised_entropy(np.array([0.0, 1.0, 0.0])) == 0.0
    assert math.isnan(normalised_entropy(np.array([1.0])))


def test_entropy_rises_with_temperature():
    scores = np.array([0.3, 1.0, 0.8, 0.1, 0.55])
    values = [normalised_entropy(softmax(scores, tau)) for tau in (0.01, 0.05, 0.2, 1.0, 5.0, 100.0)]
    assert values == sorted(values) and 0.0 <= values[0] < values[-1] <= 1.0


def test_kl_divergence_properties():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    assert kl_divergence(p, p) == pytest.approx(0.0)
    assert kl_divergence(p, q) > 0 and kl_divergence(q, p) > 0
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))             # not symmetric
    cross_entropy = float(-(p * np.log(q)).sum())
    assert cross_entropy == pytest.approx(entropy(p) + kl_divergence(p, q))      # H(P,Q) = H(P) + D_KL(P||Q)
    assert kl_divergence(p, np.array([0.5, 0.5, 0.0])) == math.inf               # q = 0 where p > 0
    assert kl_divergence(np.array([0.5, 0.5, 0.0]), p) < math.inf                # p = 0 terms are dropped
    assert same_top(p, np.array([0.5, 0.3, 0.2])) and not same_top(p, q)


def test_kl_of_greedy_rule_is_zero_or_infinite():
    """Why the divergence is only recorded above entropy level 0 (Section 3.6.4)."""
    one_hot = greedy(np.array([0.1, 0.9, 0.3]))
    assert kl_divergence(one_hot, greedy(np.array([0.2, 0.8, 0.1]))) == 0.0
    assert kl_divergence(one_hot, greedy(np.array([0.9, 0.8, 0.1]))) == math.inf


# ------------------------------------------------------------------------------------------------- MWKR
def test_mwkr_rule_dispatches_most_work_remaining(tiny):
    state = State(tiny)
    policy = MWKRRollout(0.0)
    assert policy.tau is None
    # remaining work: J0 9, J1 7, J2 8, J3 9 -> the tie between J0 and J3 goes to the lower index
    assert list(policy.distribution(state)) == [1.0, 0.0, 0.0, 0.0]
    trace: list = []
    policy.rollout(State(tiny), random.Random(0), trace)
    check = State(tiny)
    for job in trace:
        work = [tiny.rem[j][check.next_op[j]] for j in check.active]
        assert job == check.active[work.index(max(work))]
        check.step(job)


def test_mwkr_scores_and_distribution(tiny):
    state = State(tiny)
    policy = MWKRRollout(0.5, tau=0.1)
    scores = policy.scores(state)
    assert np.allclose(scores, np.array([9, 7, 8, 9]) / 9)             # work remaining / largest work remaining
    p = policy.distribution(state)
    assert p.sum() == pytest.approx(1.0) and np.allclose(p, softmax(scores, 0.1))
    assert p[0] == pytest.approx(p[3]) and p[0] > p[2] > p[1]
    assert np.allclose(MWKRRollout(1.0).distribution(state), 0.25)
    with pytest.raises(ValueError):
        MWKRRollout(0.5)                                               # a level in (0, 1) needs a temperature


def test_all_rollouts_complete_the_schedule(ft06, gin, az66):
    policies = [UniformRollout(), MWKRRollout(0.0), MWKRRollout(0.5, 0.05), MWKRRollout(1.0),
                NetworkRollout("LR", make_inference(gin), 0.0), NetworkRollout("LR", make_inference(gin), 0.5, 0.3),
                NetworkRollout("AZ", make_inference(az66), 0.5, 0.3), NetworkRollout("AZ", make_inference(az66), 1.0)]
    for policy in policies:
        state = random_state(ft06, 7, 1)
        before = state.n_done
        trace: list = []
        makespan = policy.rollout(state, random.Random(3), trace)
        assert state.complete and makespan == state.makespan
        assert len(trace) == ft06.n_ops - before
        assert all(trace.count(j) <= ft06.m for j in range(ft06.n))


def test_recorded_rollout_makes_the_same_decisions(ft06, gin, az66):
    """Recording must not change the search: both rollouts consume the random numbers identically."""
    policies = [UniformRollout(), MWKRRollout(0.0), MWKRRollout(0.25, 0.02), MWKRRollout(1.0),
                NetworkRollout("LR", make_inference(gin), 0.0), NetworkRollout("LR", make_inference(gin), 0.75, 0.5),
                NetworkRollout("AZ", make_inference(az66), 0.5, 0.3), NetworkRollout("AZ", make_inference(az66), 1.0)]
    for policy in policies:
        for seed in range(4):
            fast, recorded = [], []
            a = policy.rollout(random_state(ft06, 5, seed), random.Random(seed), fast)
            stats = StepStats(rate=0.2, seed=1)
            b = policy.rollout_recorded(random_state(ft06, 5, seed), random.Random(seed), recorded, stats)
            assert fast == recorded and a == b, policy.label
            assert stats.h_cnt > 0


def test_recorded_entropy_matches_the_policy(ft06):
    stats = StepStats(rate=1.0, seed=0)
    UniformRollout().rollout_recorded(State(ft06), random.Random(0), [], stats)
    assert stats.mean_entropy == pytest.approx(1.0)
    assert len(stats.samples) == stats.h_cnt                      # rate 1: every step with a choice is stored
    assert all(s["n_candidates"] >= 2 and 1 <= s["mwkr_rank"] <= s["n_candidates"] for s in stats.samples)
    stats = StepStats(rate=1.0, seed=0)
    MWKRRollout(0.0).rollout_recorded(State(ft06), random.Random(0), [], stats)
    assert stats.mean_entropy == 0.0
    assert all(s["mwkr_rank"] == 1 and s["agree_mwkr"] for s in stats.samples)   # MWKR agrees with itself


def test_step_sampling_rate_and_kl_reference(ft06, gin):
    stats = StepStats(rate=0.05, seed=3)
    policy = MWKRRollout(0.5, 0.05)
    for seed in range(40):
        policy.rollout_recorded(State(ft06), random.Random(seed), [], stats)
    share = len(stats.samples) / stats.h_cnt
    assert 0.03 < share < 0.07
    assert "kl_mwkr" not in stats.samples[0]                      # only learned distributions have a reference
    learned = NetworkRollout("LR", make_inference(gin), 0.5, 0.5)
    learned.reference = MWKRRollout(0.5, 0.05).distribution
    stats = StepStats(rate=1.0, seed=0)
    learned.rollout_recorded(State(ft06), random.Random(0), [], stats)
    assert all(s["kl_mwkr"] >= 0 and math.isfinite(s["kl_mwkr"]) for s in stats.samples)
    greedy_learned = NetworkRollout("LR", make_inference(gin), 0.0)
    greedy_learned.reference = MWKRRollout(0.0).distribution
    stats = StepStats(rate=1.0, seed=0)
    greedy_learned.rollout_recorded(State(ft06), random.Random(0), [], stats)
    assert all("kl_mwkr" not in s for s in stats.samples)        # not recorded at level 0


def test_zero_rate_stores_nothing(ft06):
    stats = StepStats(rate=0.0)
    UniformRollout().rollout_recorded(State(ft06), random.Random(0), [], stats)
    assert stats.samples == [] and stats.h_cnt > 0


# ------------------------------------------------------------------------------------------ calibration
@pytest.mark.parametrize("target", [0.25, 0.5, 0.75])
def test_bisection_reaches_the_entropy_level(target):
    instances = validation_set(6, 6)[:15]
    cal = calibrate_tau(lambda tau: MWKRRollout(target, tau), instances, target, tolerance=0.01, seed=0)
    assert cal.converged and abs(cal.realised - target) <= 0.01
    realised = mean_rollout_entropy(MWKRRollout(target, cal.tau), instances, seed=0)
    assert realised == pytest.approx(cal.realised)                # deterministic for a temperature


def test_temperatures_rise_with_the_level():
    instances = validation_set(6, 6)[:10]
    taus = [calibrate_tau(lambda tau, lv=lv: MWKRRollout(lv, tau), instances, lv).tau for lv in (0.25, 0.5, 0.75)]
    assert taus == sorted(taus)


def test_calibration_for_a_network(gin):
    instances = validation_set(6, 6)[:6]
    infer = make_inference(gin)
    cal = calibrate_tau(lambda tau: NetworkRollout("LR", infer, 0.5, tau), instances, 0.5, tolerance=0.02)
    assert abs(cal.realised - 0.5) <= 0.02


def test_mean_entropy_start_depths():
    instances = [random_instance(6, 4, s, low=1, high=30) for s in range(4)]
    policy = MWKRRollout(0.5, 0.05)
    from_empty = mean_rollout_entropy(policy, instances, start_depths=1)
    spread = mean_rollout_entropy(policy, instances, start_depths=8)
    assert 0.0 < from_empty < 1.0 and 0.0 < spread < 1.0
    with pytest.raises(ValueError):
        calibrate_tau(lambda tau: policy, instances, 1.0)
