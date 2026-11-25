"""MOR, SPT and FIFO as fixed rollout distributions (extension of RQ1 and RQ2)."""
from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import pytest

from conftest import random_state
from mctsjssp.analysis.rq import analyse_rules, validation_ranking
from mctsjssp.distributions import (FIXED_RULES, FIFORollout, MORRollout, MWKRRollout, SPTRollout, StepStats,
                                    make_rule_rollout, softmax)
from mctsjssp.distributions.calibration import calibrate_tau, mean_rollout_entropy
from mctsjssp.env import State
from mctsjssp.env.rules import RULES
from mctsjssp.experiments.config import config_hash, load_config
from mctsjssp.experiments.design import KNOWN, plan
from mctsjssp.instances import validation_set

NEW = ("MOR", "SPT", "FIFO")


def rule_sequence(inst, rule) -> list:
    state, sequence = State(inst), []
    while state.active:
        job = rule(state)
        sequence.append(job)
        state.step(job)
    return sequence


# --------------------------------------------------------------------------------------------- the rules
def test_level_zero_is_the_dispatching_rule_of_the_second_check(tiny, orlib):
    """At level 0 each distribution is the rule of Table 4.1 in the MDP (ties: the lower job index)."""
    for inst in (tiny, orlib["ft06"], orlib["la01"], orlib["swv01"]):
        for name in FIXED_RULES:
            trace: list = []
            makespan = make_rule_rollout(name, 0.0).rollout(State(inst), random.Random(0), trace)
            assert trace == rule_sequence(inst, RULES[name]), (inst.name, name)
            assert makespan == State(inst).makespan or makespan > 0


def test_scores_and_distributions_by_hand(tiny):
    state = State(tiny)                                   # durations of the first operations: 3, 2, 4, 1
    spt = SPTRollout(0.5, tau=0.1)
    assert np.allclose(spt.scores(state), -np.array([3, 2, 4, 1]) / 4)      # -p / largest p
    p = spt.distribution(state)
    assert p.sum() == pytest.approx(1.0) and np.allclose(p, softmax(spt.scores(state), 0.1))
    assert p[3] > p[1] > p[0] > p[2]                                         # shorter is better
    assert list(SPTRollout(0.0).distribution(state)) == [0.0, 0.0, 0.0, 1.0]
    mor = MORRollout(0.5, tau=0.1)                        # every job has three operations left
    assert np.allclose(mor.distribution(state), 0.25) and list(MORRollout(0.0).distribution(state)) == [1, 0, 0, 0]
    fifo = FIFORollout(0.5, tau=0.1)                      # every job is available at time 0
    assert np.allclose(fifo.distribution(state), 0.25) and list(FIFORollout(0.0).distribution(state)) == [1, 0, 0, 0]
    state.step(3)                                         # job 3 ends its first operation at 1
    q = fifo.distribution(state)
    assert np.allclose(fifo.scores(state), [0, 0, 0, -1]) and q[3] < q[0] == pytest.approx(q[1])
    state.step(0)                                         # job 0: 3 operations -> 2 left
    assert MORRollout(0.0).distribution(state)[0] == 0.0
    assert np.allclose(MORRollout(1.0).distribution(state), 0.25)
    for cls in (MORRollout, SPTRollout, FIFORollout):
        with pytest.raises(ValueError):
            cls(0.5)                                      # a level in (0, 1) needs a temperature
    with pytest.raises(ValueError):
        make_rule_rollout("LPT", 0.0)


def test_rollouts_complete_and_recording_makes_the_same_decisions(ft06):
    """Recording must not change the search: both rollouts consume the random numbers identically."""
    for name in NEW:
        for level, tau in ((0.0, None), (0.25, 0.02), (0.5, 0.3), (1.0, None)):
            policy = make_rule_rollout(name, level, tau)
            for seed in range(4):
                fast, recorded = [], []
                state = random_state(ft06, 5, seed)
                a = policy.rollout(state, random.Random(seed), fast)
                assert state.complete and a == state.makespan and len(fast) == ft06.n_ops - 5
                stats = StepStats(rate=0.2, seed=1)
                b = policy.rollout_recorded(random_state(ft06, 5, seed), random.Random(seed), recorded, stats)
                assert fast == recorded and a == b, policy.label


def test_temperatures_of_the_levels_can_be_calibrated():
    instances = list(validation_set(6, 6)[:6])
    for name in NEW:
        low = mean_rollout_entropy(make_rule_rollout(name, 0.5, 0.01), instances, start_depths=3)
        high = mean_rollout_entropy(make_rule_rollout(name, 0.5, 10.0), instances, start_depths=3)
        assert 0.0 <= low < high <= 1.0, name                                # entropy rises with the temperature
    for name in ("SPT", "FIFO"):
        cal = calibrate_tau(lambda tau: make_rule_rollout(name, 0.5, tau), instances, 0.5, tolerance=0.03,
                            start_depths=3)
        assert cal.converged and abs(cal.realised - 0.5) <= 0.03, (name, cal)


def test_mor_cannot_go_below_the_entropy_of_its_ties():
    """MOR's counts of remaining operations tie: as tau -> 0 it samples uniformly among the tied jobs."""
    instances = list(validation_set(6, 6)[:6])
    floor = mean_rollout_entropy(make_rule_rollout("MOR", 0.5, 1e-12), instances, start_depths=3)
    assert 0.6 < floor < 0.8
    cal = calibrate_tau(lambda tau: make_rule_rollout("MOR", 0.5, tau), instances, 0.5, tolerance=0.03,
                        start_depths=3)
    assert not cal.converged and cal.realised == pytest.approx(floor, abs=0.02) and cal.tau <= 1e-10
    high = calibrate_tau(lambda tau: make_rule_rollout("MOR", 0.75, tau), instances, max(0.75, floor + 0.05),
                         tolerance=0.03, start_depths=3)
    assert high.converged                                                    # above the floor: reachable
    state = State(validation_set(6, 6)[0])                                   # all jobs tie at the start
    assert np.allclose(MORRollout(0.25, 1e-12).distribution(state), 1 / 6)
    assert MORRollout(0.0).distribution(state)[0] == 1.0                     # the rule: the lower job index


def test_mwkr_is_unchanged():
    assert make_rule_rollout("MWKR", 0.5, 0.1).__class__ is MWKRRollout


# -------------------------------------------------------------------------------------------- the design
def test_design_of_the_extension(root):
    cfg = load_config(root / "configs" / "experiments" / "rules.yaml")
    rq2 = load_config(root / "configs" / "experiments" / "rq2.yaml")
    planned = plan(cfg)
    assert len(planned) == 82 * 3 * 6 * 5 and len({p.key for p in planned}) == len(planned)
    assert {p.name for p in planned} == set(NEW) and set(KNOWN) >= set(FIXED_RULES)
    assert cfg["budgets"] == rq2["budgets"] and cfg["instances"] == rq2["instances"]
    assert {p.calibration for p in planned} == {(15, 15)} and {p.label for p in planned} == {"MOR@H0.5", "SPT@H0.5",
                                                                                              "FIFO@H0.5"}
    assert [c["name"] for c in cfg["entropy"]["selection"]["configurations"]] == ["U", "MOR", "SPT", "FIFO"]
    assert cfg["entropy"]["levels"] == [0.0, 0.25, 0.5, 0.75] and cfg["entropy"]["calibration"]["instances"] == 100
    assert config_hash(cfg) != config_hash(rq2)


def test_validation_ranking_names_the_best_fixed_distribution():
    levels = {"MWKR": {"best": 0.5, "scores": {"0.0": 1.42, "0.5": 1.408}, "instances": 20, "iterations": 100},
              "MOR": {"best": 0.25, "scores": {"0.0": 1.43, "0.25": 1.401}, "instances": 20, "iterations": 100},
              "U": {"best": 1.0, "scores": {"1.0": 1.47}, "instances": 20, "iterations": 100},
              "LR": {"best": 0.25, "scores": {"0.25": 1.38}}}
    temperatures = {"MOR|-|15x15|H0.25": {"realised": 0.71}, "MWKR|-|15x15|H0.5": {"realised": 0.5}}
    for entry in levels.values():
        entry["size"] = "15x15"
    table = validation_ranking(levels, temperatures)
    assert list(table["distribution"]) == ["MOR", "MWKR", "U"]                # LR is not a fixed distribution
    assert list(table["rank"]) == [1, 2, 3] and table.loc[0, "best"] and table.loc[0, "level"] == 0.25
    assert list(table["realised_entropy"]) == [0.71, 0.5, 1.0]               # MOR: the floor of its ties
    assert validation_ranking({}).empty


def test_analysis_compares_lr_with_every_fixed_distribution():
    rng = np.random.default_rng(0)
    rows = []
    shift = {"U": 6.0, "MWKR@H0.5": 4.0, "MOR@H0.25": 3.5, "SPT@H0.5": 8.0, "FIFO@H0.5": 5.0, "LR@H0.25": 2.0}
    config = {"U": "U", "MWKR@H0.5": "MWKR", "MOR@H0.25": "MOR", "SPT@H0.5": "SPT", "FIFO@H0.5": "FIFO",
              "LR@H0.25": "LR"}
    for i in range(12):
        for label, base in shift.items():
            for budget in ("I25", "T50"):
                for run in range(2):
                    gap = base + rng.normal(0, 0.3) + i * 0.1
                    rows.append({"instance": f"i{i}", "family": "la", "size_class": "small", "n": 10, "m": 5,
                                 "n_ops": 50, "label": label, "config": config[label], "variant": config[label],
                                 "level": 0.5, "budget": budget, "budget_kind": "iterations" if budget[0] == "I"
                                 else "time", "run": run, "gap": gap, "makespan": 100 + gap, "bks": 100,
                                 "best_rollout_makespan": 100 + gap, "best_rollout_gap": gap,
                                 "iterations_per_decision": 25.0, "time_per_decision": 0.01, "train_size": None})
    import pandas as pd

    levels = {"MWKR": {"best": 0.5, "scores": {"0.5": 1.408}}, "MOR": {"best": 0.25, "scores": {"0.25": 1.401}},
              "SPT": {"best": 0.5, "scores": {"0.5": 1.52}}, "FIFO": {"best": 0.5, "scores": {"0.5": 1.43}},
              "U": {"best": 1.0, "scores": {"1.0": 1.47}}}
    out = analyse_rules(pd.DataFrame(rows), entropy_levels=levels)
    assert list(out["validation"]["distribution"]) == ["MOR", "MWKR", "FIFO", "U", "SPT"]
    best = out["tests_best_fixed"]
    assert set(best["best_fixed"]) == {"MOR"} and set(best["second"]) == {"MOR@H0.25"}
    assert best[best["group"] == "all"]["significant"].all()                 # LR is clearly better here
    assert set(out["tests"]["second"]) == set(shift) - {"LR@H0.25"}          # LR against all five
    assert set(out["tests_rules"]["first"]) == {"MOR@H0.25", "SPT@H0.5", "FIFO@H0.5"}
    assert set(out["tests_rules"]["second"]) == {"U", "MWKR@H0.5"}
    assert not out["friedman"].empty


def test_pipeline_runs_and_analyses_the_extension(root):
    out = subprocess.run([sys.executable, "run_pipeline.py", "--dry-run", "--steps", "rules,analyse"],
                         cwd=root, capture_output=True, text=True, check=True).stdout
    assert "select-entropy --config configs/experiments/rules.yaml" in out and "--missing" in out
    assert "run --config configs/experiments/rules.yaml" in out
    line = next(l for l in out.splitlines() if "analyse --config configs/experiments/rules.yaml" in l)
    assert "--with" in line and "results of rq2" in line                  # analysed with RQ2
