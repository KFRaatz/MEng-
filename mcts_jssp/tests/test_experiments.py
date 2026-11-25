"""Experimental design (Table 3.6), configuration files, time budgets, validation and an end-to-end run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mctsjssp.analysis import load_frame, write_report
from mctsjssp.experiments import config_hash, load_config, plan, run_directory, run_experiment
from mctsjssp.experiments.config import apply_overrides, deep_merge, get
from mctsjssp.experiments.design import job_seed, run_identifier, sigma_tag
from mctsjssp.experiments.fixed_states import evaluate_fixed_states, sample_states
from mctsjssp.experiments.pilot import pilot_study, run_seconds, time_per_iteration
from mctsjssp.experiments.rules import RULES, run_rule
from mctsjssp.experiments.runner import read_results
from mctsjssp.experiments.time_budget import budget_key, measure_time_budget
from mctsjssp.experiments.validation import check_schedules, compare_with_published, learning_check, rule_gaps
from mctsjssp.distributions import UniformRollout
from mctsjssp.env import replay
from mctsjssp.training import TrainConfig, Trainer


# ------------------------------------------------------------------------------------ configuration
def test_deep_merge_overrides_and_replace():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "models": {"m1": {"n": 6}, "m2": {"n": 10}}}
    merged = deep_merge(base, {"b": {"y": 3}, "models": {"_replace": True, "only": {"n": 4}}, "c": [1]})
    assert merged == {"a": 1, "b": {"x": 1, "y": 3}, "models": {"only": {"n": 4}}, "c": [1]}
    assert base["b"]["y"] == 2                                       # the parent is not modified
    out = apply_overrides(base, ["b.y=7", "new.key=[1, 2]", "a=null", "s=text"])
    assert out["b"]["y"] == 7 and out["new"]["key"] == [1, 2] and out["a"] is None and out["s"] == "text"
    with pytest.raises(ValueError):
        apply_overrides(base, ["novalue"])
    assert get(base, "b.x") == 1 and get(base, "b.missing.deeper", "d") == "d"


def test_configuration_files_inherit_from_base(root):
    cfg = load_config(root / "configs" / "experiments" / "rq3.yaml")
    assert cfg["experiment"] == "rq3" and cfg["search"]["c"] == 1.0          # inherited from base.yaml
    assert cfg["recording"]["level"] == "steps" and cfg["recording"]["step_rate"] == 0.001
    assert Path(cfg["paths"]["data"]) == root / "data" / "orlib"
    assert cfg["training"]["buffer_size"] == 36000 and cfg["training"]["episodes_per_generation"] == 40
    smoke = load_config(root / "configs" / "smoke" / "base.yaml")
    assert set(smoke["models"]) == {"lr_6x6", "lr_10x10", "az_6x6", "az_10x10", "lr_6x6_s010", "az_6x6_s010",
                                    "az_10x10_s010"}                          # replaced, not merged


def test_config_hash_identifies_the_settings(root):
    path = root / "configs" / "experiments" / "rq1.yaml"
    a, b = load_config(path), load_config(path)
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 8
    assert config_hash(load_config(path, ["runs=3"])) != config_hash(a)
    moved = load_config(path, ["paths.results=/somewhere/else"])
    assert config_hash(moved) == config_hash(a)                              # where results go is not a setting
    assert run_directory(a).name == f"rq1-{config_hash(a)}"


def test_table_3_4_defaults_in_base_configuration(root):
    cfg = load_config(root / "configs" / "base.yaml")
    t = cfg["training"]
    assert (t["iterations"], t["batch_size"], t["learning_rate"], t["weight_decay"]) == (100, 256, 5e-5, 5e-4)
    assert (t["percentile"], t["window"], t["patience"]) == (75, 250, 5)
    assert (t["noise_weight"], t["noise_alpha_scale"]) == (0.25, 10.0)
    assert t["networks"]["gin"] == {"hidden": 64, "gin_layers": 2, "mlp_hidden_layers": 2, "head_hidden": 32,
                                    "head_hidden_layers": 2}
    assert t["networks"]["ffnn"] == {"hidden": 256, "hidden_layers": 2}
    assert cfg["entropy"]["levels"] == [0.0, 0.25, 0.5, 0.75] and cfg["runs"] == 5
    assert cfg["tuning"]["trials"] == 20 and cfg["entropy"]["selection"] == {
        "instances": 20, "iterations": 100, "configurations": [{"name": "MWKR"}, {"name": "LR", "model": "lr_15x15"},
                                                               {"name": "AZ", "model": "az_15x15"}]}
    defaults = TrainConfig()
    assert (defaults.buffer_size, defaults.batch_size, defaults.iterations, defaults.episodes_per_generation,
            defaults.learning_rate, defaults.percentile, defaults.patience) == (36000, 256, 100, 40, 5e-5, 75.0, 5)


# ------------------------------------------------------------------------------------------ the design
@pytest.mark.parametrize("name, runs", [
    ("rq1", 82 * 2 * 4 * 5),                    # U and MWKR, four iteration budgets
    ("rq2", 82 * 3 * 6 * 5),                    # U, MWKR, LR; four iteration budgets, T50 and T200
    ("rq3", 24 * (1 + 3 * 4) * 2 * 5),          # U and three configurations at four levels; I100 and T200
    ("rq4", 82 * 5 * 1 * 5),                    # U, MWKR and LR of three training sizes; T200
    ("rq5", 2 * (18 * 5 + 6 * 3) * 1 * 5),      # two sigmas; U, MWKR, 3 x LR; 3 x AZ on six instances; T200
    ("rq6", 24 * 5 * 1 * 5),                    # U, LR and AZ by self-labelling and by self-play; I100
])
def test_design_of_table_3_6(root, name, runs):
    cfg = load_config(root / "configs" / "experiments" / f"{name}.yaml")
    planned = plan(cfg)
    assert len(planned) == runs
    assert len({p.key for p in planned}) == runs                             # every run has its own key
    assert {p.run for p in planned} == set(range(5))


def test_design_details(root):
    rq3 = plan(load_config(root / "configs" / "experiments" / "rq3.yaml"))
    assert {p.instance for p in rq3} == set(__import__("mctsjssp.instances", fromlist=["rq3_names"]).rq3_names())
    lr = [p for p in rq3 if p.name == "LR"]
    assert {p.model for p in lr} == {"lr_6x6", "lr_10x10", "lr_15x15"}       # the network of the instance size
    assert all(p.calibration == tuple(int(v) for v in p.model.split("_")[1].split("x")) for p in lr)
    assert {p.level for p in lr} == {0.0, 0.25, 0.5, 0.75}
    mwkr = [p for p in rq3 if p.name == "MWKR" and p.instance == "ft06"]
    assert {p.calibration for p in mwkr} == {(6, 6)}                         # own size in RQ3
    assert {p.budget_label for p in rq3} == {"I100", "T200"}
    rq2 = plan(load_config(root / "configs" / "experiments" / "rq2.yaml"))
    assert {p.calibration for p in rq2 if p.name == "MWKR"} == {(15, 15)}    # default validation size
    assert {p.label for p in rq2} == {"U", "MWKR@H0.5", "LR[lr_15x15]@H0.5"}  # tuned level: placeholder 0.5
    rq5 = plan(load_config(root / "configs" / "experiments" / "rq5.yaml"))
    assert {p.sigma for p in rq5} == {0.1, 0.2}
    az = [p for p in rq5 if p.name == "AZ"]
    assert {p.instance for p in az} == {"ft06", "ft10", "la16", "la36", "abz5", "orb01"}
    assert {p.model for p in az if p.sigma == 0.2 and p.variant == "AZ-FT"} == {"az_6x6_s020", "az_10x10_s020",
                                                                               "az_15x15_s020"}
    assert {p.model for p in az if p.sigma == 0.2 and p.variant == "AZ"} == {"az_6x6", "az_10x10", "az_15x15"}
    assert {p.model for p in rq5 if p.name == "LR"} == {"lr_15x15", "lr_15x15_s010", "lr_15x15_s020",
                                                        "../scratch/models/lr_15x15_s010",
                                                        "../scratch/models/lr_15x15_s020"}
    assert {p.variant for p in rq5} == {"U", "MWKR", "LR", "LR-FT", "LR-SC", "AZ", "AZ-FT", "AZ-SC"}
    assert sigma_tag(0.1) == "010" and sigma_tag(None) == "000"


def test_design_of_rq6_pairs_each_network_with_its_self_play_twin(root):
    rq6 = plan(load_config(root / "configs" / "experiments" / "rq6.yaml"))
    assert {p.label for p in rq6} == {"U", "LR@H0.5", "LR-SP@H0.5", "AZ@H0.5", "AZ-SP@H0.5"}   # tuned: placeholder
    sp = [p for p in rq6 if p.variant == "LR-SP" and p.instance == "ft06"]
    assert {p.model for p in sp} == {"../self_play/models/lr_6x6"}
    assert {p.calibration for p in sp} == {(6, 6)}                            # the size of the network
    assert {p.budget_label for p in rq6} == {"I100"}
    assert {p.instance for p in rq6} == set(__import__("mctsjssp.instances", fromlist=["rq3_names"]).rq3_names())


def test_seeds_and_identifiers_are_stable():
    key = "ft06|LR[lr_6x6]@H0.5|I100|s0|r3"
    assert job_seed(1, key) == job_seed(1, key) != job_seed(2, key)
    assert job_seed(1, key) != job_seed(1, key.replace("|r3", "|r4"))
    assert run_identifier("rq2", key) == run_identifier("rq2", key) and run_identifier("rq2", key).startswith("rq2-")


def test_unknown_configuration_and_duplicate_labels_are_rejected(root):
    cfg = load_config(root / "configs" / "experiments" / "rq1.yaml")
    cfg["configurations"] = [{"name": "XYZ"}]
    with pytest.raises(ValueError, match="unknown configuration"):
        plan(cfg)
    cfg["configurations"] = [{"name": "U"}, {"name": "U"}]
    with pytest.raises(ValueError, match="same label"):
        plan(cfg)
    cfg["configurations"] = [{"name": "LR"}]
    with pytest.raises(ValueError, match="needs a 'model'"):
        plan(cfg)
    cfg["configurations"] = [{"name": "U", "c": 0.5}, {"name": "U", "c": 2.0}]
    assert {p.label for p in plan(cfg)} == {"U(c=0.5)", "U(c=2)"}


# ------------------------------------------------------------------------------------------ validation
def test_first_validation_check_passes(orlib):
    subset = [orlib[n] for n in ("ft06", "la01", "orb07", "la36", "swv06")]
    result = check_schedules(subset, random_schedules=3, seed=0)
    assert result["passed"] and result["schedules_checked"] == 5 * (2 * 4 + 2 * 3)


def test_dispatching_rules(tiny, orlib, tmp_path):
    assert set(RULES) == {"SPT", "MWKR", "MOR", "FIFO"}
    from mctsjssp.env import State

    state = State(tiny)
    # durations of the first operations: 3, 2, 4, 1 -> SPT takes job 3; work remaining 9, 7, 8, 9 -> MWKR job 0
    assert RULES["SPT"](state) == 3 and RULES["MWKR"](state) == 0
    assert RULES["MOR"](state) == 0 and RULES["FIFO"](state) == 0             # ties: the lower job index
    state.step(0)
    assert RULES["MOR"](state) == 1                                           # job 0 has fewer operations left
    for rule in RULES.values():
        final = run_rule(tiny, rule)
        assert final.complete
    gaps = rule_gaps([orlib["ft06"], orlib["la01"], orlib["la02"]])
    assert len(gaps) == 3 * 4 * 2 and (gaps["gap"] >= 0).all() and set(gaps["mode"]) == {"non_delay", "free"}
    table = compare_with_published(gaps, None)
    assert set(table.columns) == {"family", "rule", "gap_non_delay", "gap_free_dispatching"}
    published = tmp_path / "published.csv"
    published.write_text("# source: test\nfamily,rule,gap\nla,SPT,30.0\nla,MWKR,\n")
    merged = compare_with_published(gaps, published).set_index(["family", "rule"])
    assert merged.loc[("la", "SPT"), "difference"] == pytest.approx(merged.loc[("la", "SPT"), "gap_non_delay"] - 30.0)
    assert np.isnan(merged.loc[("la", "MWKR"), "gap_published"])


def test_non_delay_rules_only_choose_operations_that_can_start_now(tiny):
    from mctsjssp.env import State
    from mctsjssp.experiments.rules import startable_now

    state = State(tiny)
    assert startable_now(state) == [0, 1, 2, 3]                 # everything can start at time 0
    state.step(0)                                               # J0 occupies machine 0 until time 3
    assert startable_now(state) == [1, 2]                       # J0 (ready at 3) and J3 (machine 0) must wait
    final = run_rule(tiny, RULES["MWKR"], non_delay=True)
    # in a non-delay schedule no machine is idle while an operation is waiting for it
    start, end = final.start_times(), final.start_times() + tiny.durations
    for j in range(tiny.n):
        for k in range(tiny.m):
            ready = end[j, k - 1] if k else 0
            mc = tiny.machines[j, k]
            busy = [(start[a, b], end[a, b]) for a in range(tiny.n) for b in range(tiny.m)
                    if tiny.machines[a, b] == mc and (a, b) != (j, k)]
            idle_from = max([e for s, e in busy if e <= start[j, k]] + [0])
            assert start[j, k] == max(ready, idle_from)


def test_non_delay_rules_reproduce_the_published_gaps(orlib, root):
    """Second check of Section 3.7: the four rules of Ho et al. on all 82 instances."""
    table = compare_with_published(rule_gaps(list(orlib.values())), root / "data" / "published_rule_gaps.csv")
    assert len(table) == 6 * 4 and table["gap_published"].notna().all()
    assert table["difference"].abs().max() <= 0.15              # percentage points
    # in the MDP of the thesis (every unfinished job is a legal action) only FIFO is unchanged
    fifo = table[table.rule == "FIFO"]
    assert np.allclose(fifo["gap_non_delay"], fifo["gap_free_dispatching"])
    others = table[table.rule != "FIFO"]
    assert (others["gap_free_dispatching"] > others["gap_non_delay"]).all()


# ------------------------------------------------------------------------------- pilot and time budgets
def test_time_per_iteration_grows_with_the_remaining_operations(orlib):
    fit = time_per_iteration(orlib["la21"], UniformRollout(), depths=(0.0, 0.5, 0.8), iterations=40, repeats=3)
    assert fit["b"] > 0 and len(fit["points"]) == 3
    assert fit["points"][0][1] > fit["points"][-1][1]                         # shorter rollouts are cheaper


def test_run_seconds_for_iteration_and_time_budgets():
    from mctsjssp.experiments.design import Planned

    coeff = {("U", (6, 6), False): (0.0, 1e-6), ("LR", (6, 6), False): (0.0, 1e-4)}
    iter_run = Planned("ft06", "LR", "LR[m]", 0.5, "m", (6, 6), "iterations", 100, 0, None, 1.0)
    time_run = Planned("ft06", "LR", "LR[m]", 0.5, "m", (6, 6), "time", 200, 0, None, 1.0)
    u_time = Planned("ft06", "U", "U", 1.0, None, None, "time", 200, 0, None, 1.0)
    triangle = 36 * 37 / 2
    assert run_seconds(iter_run, coeff, 36) == pytest.approx(100 * 1e-4 * triangle)
    t200 = 200 * 1e-6 * 37 / 2
    assert run_seconds(u_time, coeff, 36) == pytest.approx(36 * t200)        # U needs exactly T200 per decision
    # LR: one iteration can take longer than T200; such decisions take the time of the iteration
    expected = sum(max(t200, 1e-4 * r) for r in range(1, 37))
    assert run_seconds(time_run, coeff, 36) == pytest.approx(expected)


def test_time_budget_measurement(root):
    entry = measure_time_budget({"instance": "ft06", "iterations": 20, "sigma": None, "repeats": 2, "seed": 0,
                                 "c": 1.0, "skip_forced": True, "data_dir": str(root / "data" / "orlib")})
    assert entry["key"] == budget_key("ft06", 20, None) == "ft06|I20|s0"
    assert entry["mean"] > 0 and len(entry["per_decision"]) == 2 and entry["per_decision"][1] >= 0
    a, b = entry["per_decision"]
    assert a + b * 36 > a + b * 2                                             # early decisions take longer


# ------------------------------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def workspace(tmp_path_factory, root):
    """A small project with trained tiny networks, shared by the end-to-end tests of this module."""
    base = tmp_path_factory.mktemp("project")
    (base / "configs").mkdir()
    cfg = {
        "inherits": str(root / "configs" / "base.yaml"),
        "paths": {"data": str(root / "data" / "orlib"), "artifacts": str(base / "artifacts"),
                  "results": str(base / "results")},
        "runs": 2,
        "entropy": {"default_size": [6, 6], "calibration": {"instances": 8, "start_depths": 3, "tolerance": 0.03}},
        "time_budget": {"repeats": 1},
        "stochastic": {"scenarios": 4},
    }
    (base / "configs" / "base.yaml").write_text(yaml.safe_dump(cfg))
    artifacts = base / "artifacts" / "models"
    for name, kind, sigma in (("lr_6x6", "gin", None), ("az_6x6", "ffnn", None), ("lr_6x6_s010", "gin", 0.1)):
        tc = TrainConfig(name=name, kind=kind, n=6, m=6, episodes_per_generation=3, iterations=8, max_generations=2,
                         validation_instances=6, learning_rate=1e-3, sigma=sigma, seed=1)
        Trainer(tc, artifacts / name, log=lambda *_: None).run()
    return base


def experiment(workspace, name, body):
    path = workspace / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"inherits": "base.yaml", "experiment": name, **body}))
    return load_config(path)


@pytest.mark.slow
def test_experiment_runs_resumes_and_is_analysed(workspace):
    cfg = experiment(workspace, "rq2", {
        "instances": ["ft06", "la01", "la02"],
        "budgets": [{"iterations": 4}, {"iterations": 8}, {"time": 4}],
        "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": 0.5},
                           {"name": "LR", "model": "lr_6x6", "entropy": 0.5}],
        "recording": {"level": "steps", "timed_level": "decisions", "step_rate": 0.05},
    })
    logs: list = []
    run_dir = run_experiment(cfg, workers=1, limit=10, log=logs.append)
    assert len(read_results(run_dir)) == 10                                   # interrupted after ten runs
    run_dir = run_experiment(cfg, workers=1, log=logs.append)
    rows = read_results(run_dir)
    assert len(rows) == 3 * 3 * 3 * 2 and len({r["key"] for r in rows}) == len(rows)
    assert any("10 already done" in line for line in logs)
    run_experiment(cfg, workers=1, log=logs.append)                           # nothing left to do
    assert len(read_results(run_dir)) == 54
    for name in ("config.yaml", "meta.json", "resolved.json"):
        assert (run_dir / name).exists()
    frame = load_frame(run_dir)
    # every reported schedule can be rebuilt from its job sequence and is at least as long as the BKS
    for _, row in frame.sample(8, random_state=0).iterrows():
        inst = __import__("mctsjssp.instances", fromlist=["load_orlib"]).load_orlib([row["instance"]])[row["instance"]]
        assert replay(inst, row["sequence"]).makespan == row["makespan"] >= inst.bks
        assert row["gap"] == pytest.approx((row["makespan"] - inst.bks) / inst.bks * 100)
    iter_rows = frame[frame.budget_kind == "iterations"]
    assert (iter_rows["iterations"] == iter_rows["budget_iterations"] * iter_rows["decisions"]).all()
    timed = frame[frame.budget_kind == "time"]
    assert (timed["budget_seconds"] > 0).all() and (timed["time_per_decision"] >= timed["budget_seconds"]).all()
    lr = frame[frame.config == "LR"]
    assert (lr["network_calls"] > 0).all() and (lr["train_size"] == "6x6").all() and lr["tau"].notna().all()
    assert np.allclose(frame[frame.config == "U"]["rollout_entropy"].dropna(), 1.0)
    assert abs(iter_rows[iter_rows.config == "MWKR"]["rollout_entropy"].mean() - 0.5) < 0.25
    assert "kl_mwkr" in lr.columns and lr[lr.budget_kind == "iterations"]["kl_mwkr"].notna().any()
    records = sorted(p.name for p in (run_dir / "records").iterdir())
    one = iter_rows.iloc[0]["run_id"]
    assert {f"{one}.decisions.jsonl.gz", f"{one}.rollouts.npz"} <= set(records)
    tables = write_report(run_dir, "rq2", log=logs.append)
    assert not tables["tests"].empty and (run_dir / "analysis" / "summary.csv").exists()
    stored = json.loads((workspace / "artifacts" / "temperatures.json").read_text())
    assert {"MWKR|-|6x6|H0.5", "LR|lr_6x6|6x6|H0.5"} <= set(stored)
    budgets = json.loads((workspace / "artifacts" / "time_budgets.json").read_text())
    assert set(budgets) == {"ft06|I4|s0", "la01|I4|s0", "la02|I4|s0"}


@pytest.mark.slow
def test_results_are_reproducible_across_processes(workspace):
    body = {"instances": ["ft06"], "budgets": [{"iterations": 6}], "runs": 3,
            "configurations": [{"name": "U"}, {"name": "LR", "model": "lr_6x6", "entropy": 0.0}]}
    first = read_results(run_experiment(experiment(workspace, "repro_a", body), workers=1, log=lambda *_: None))
    second = read_results(run_experiment(experiment(workspace, "repro_b", {**body, "description": "again"}),
                                         workers=2, log=lambda *_: None))
    a = {r["key"]: r["makespan"] for r in first}
    b = {r["key"]: r["makespan"] for r in second}
    assert a == b and len(a) == 6
    assert len({r["seed"] for r in first}) == 6                               # every run has its own seed


@pytest.mark.slow
def test_stochastic_experiment_evaluates_fixed_scenarios(workspace):
    cfg = experiment(workspace, "rq5", {
        "instances": ["ft06", "la01"], "budgets": [{"iterations": 5}],
        "stochastic": {"sigmas": [0.1], "scenarios": 4},
        "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": 0.0},
                           {"name": "LR", "model": "lr_6x6_s{sigma}", "entropy": 0.0, "instances": ["ft06"]}],
    })
    frame = load_frame(run_experiment(cfg, log=lambda *_: None))
    assert len(frame) == (2 * 2 + 1) * 2 and (frame["sigma"] == 0.1).all()
    assert (frame["realised"].map(len) == 4).all()
    row = frame.iloc[0]
    assert row["realised_mean"] == pytest.approx(np.mean(row["realised"]))
    assert row["realised_increase"] == pytest.approx((row["realised_mean"] - row["makespan"]) / row["makespan"] * 100)
    assert row["realised_gap"] == pytest.approx((row["realised_mean"] - row["bks"]) / row["bks"] * 100)
    # all configurations see the same scenarios: the same schedule gives the same realisations
    from mctsjssp.env import execute
    from mctsjssp.instances import load_orlib, scenarios

    inst = load_orlib([row["instance"]])[row["instance"]]
    final = replay(inst, row["sequence"])
    assert [execute(inst, final.m_op, d) for d in scenarios(inst, 0.1, 4)] == row["realised"]


@pytest.mark.slow
def test_artifacts_cannot_be_mixed(workspace):
    body = {"instances": ["ft06"], "budgets": [{"iterations": 3}], "runs": 1,
            "configurations": [{"name": "MWKR", "entropy": 0.5}]}
    cfg = experiment(workspace, "mixing", body)
    run_dir = run_experiment(cfg, log=lambda *_: None)
    resolved = json.loads((run_dir / "resolved.json").read_text())
    key = next(iter(resolved))
    resolved[key]["tau"] = 123.0                                              # as if the temperature had changed
    (run_dir / "resolved.json").write_text(json.dumps(resolved))
    cfg = experiment(workspace, "mixing", body)
    with pytest.raises(RuntimeError, match="other temperatures or checkpoints"):
        run_experiment(cfg, log=lambda *_: None)


@pytest.mark.slow
def test_fixed_states_and_learning_check(workspace):
    assert all(len(p) < 36 for p in sample_states(
        __import__("mctsjssp.instances", fromlist=["load_orlib"]).load_orlib(["ft06"])["ft06"], 6, seed=0))
    cfg = experiment(workspace, "rq3", {
        "instances": ["ft06"], "budgets": [{"iterations": 3}],
        "entropy": {"levels": [0.0, 0.5]},
        "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": [0, 0.5], "calibration": "own_size"},
                           {"name": "LR", "model": "lr_{size}", "entropy": [0, 0.5]},
                           {"name": "AZ", "model": "az_{size}", "entropy": [0, 0.5]}],
        "fixed_states": {"per_instance": 6, "search_iterations": 12, "search_per_instance": 3},
    })
    table = evaluate_fixed_states(cfg, log=lambda *_: None)
    assert len(table) == 6 * (1 + 3 * 2)                                      # states x distributions
    uniform = table[table.configuration == "U"]
    assert np.allclose(uniform["norm_entropy"], 1.0)
    # distance from U: zero for U itself, ln k for a greedy distribution, and consistent with the entropy
    assert np.allclose(uniform["kl_u"], 0.0) and np.allclose(uniform["tv_u"], 0.0)
    greedy_lr = table[(table.level == 0.0) & (table.configuration == "LR")]
    assert np.allclose(greedy_lr["kl_u"], np.log(greedy_lr["n_candidates"]))
    assert np.allclose(table["kl_u_norm"], 1.0 - table["norm_entropy"], atol=1e-9)
    assert ((table["tv_u"] >= 0) & (table["tv_u"] <= 1 - 1 / table["n_candidates"] + 1e-12)).all()
    # agreement with U in the search: on the first 3 states only, U against U is the baseline
    searched = table[table["state"] < 3]
    assert searched["agree_u_search"].notna().all() and table[table["state"] >= 3]["agree_u_search"].isna().all()
    assert ((searched["visit_overlap_u"] >= 0) & (searched["visit_overlap_u"] <= 1 + 1e-9)).all()
    from mctsjssp.analysis.plots import fixed_state_summary, plot_fixed_states
    summary = fixed_state_summary(table)
    assert set(summary["configuration"]) == {"U", "MWKR", "LR", "AZ"} and "agree_u_search_mean" in summary
    assert any(p.name == "fixed_states_vs_u.png" for p in plot_fixed_states(summary, workspace / "fig"))
    greedy = table[(table.level == 0.0) & (table.configuration != "U")]
    assert (greedy["norm_entropy"] == 0.0).all() and greedy["kl_mwkr"].isna().all()
    sampled = table[(table.level == 0.5) & (table.configuration == "LR")]
    assert sampled["kl_mwkr"].notna().all() and (sampled["kl_mwkr"] >= 0).all()
    assert sampled["kl_other_learned"].notna().all() and set(sampled["agree_other_learned"]) <= {True, False}
    check = learning_check(workspace / "artifacts" / "models" / "lr_6x6" / "best.pt", instances=10)
    assert check["size"] == "6x6" and check["passed"] == (check["greedy_ratio"] < check["uniform_ratio"])


def test_has_model_needs_a_finished_training(tmp_path):
    from mctsjssp.experiments.artifacts import Artifacts

    artifacts = Artifacts({"paths": {"artifacts": str(tmp_path)}}, log=lambda *_: None)
    folder = tmp_path / "models" / "lr_6x6"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"")
    assert not artifacts.has_model("lr_6x6")                  # interrupted (no summary.json): trained again
    (folder / "summary.json").write_text("{}")
    assert artifacts.has_model("lr_6x6")


@pytest.mark.slow
def test_temperature_requests_are_the_temperatures_that_resolve_needs(workspace):
    from mctsjssp.experiments.artifacts import Artifacts
    from mctsjssp.experiments.design import temperature_requests

    cfg = experiment(workspace, "requests", {
        "instances": ["ft06"], "budgets": [{"iterations": 3}], "runs": 1,
        "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": [0, 0.5]},
                           {"name": "LR", "model": "lr_6x6", "entropy": [0.25]}]})
    requests = set(temperature_requests(plan(cfg, Artifacts(cfg, log=lambda *_: None))))
    assert requests == {("MWKR", 0.5, (6, 6), None), ("LR", 0.25, (6, 6), "lr_6x6"), ("MWKR", 0.25, (6, 6), None)}


@pytest.mark.slow
def test_entropy_selection_and_calibration_in_workers_match_one_process(workspace, tmp_path):
    import multiprocessing as mp
    import shutil

    from mctsjssp.distributions import MWKRRollout
    from mctsjssp.experiments.artifacts import Artifacts
    from mctsjssp.experiments.runner import worker_init
    from mctsjssp.instances.generator import validation_set
    from mctsjssp.training.tuning import select_entropy_level, select_entropy_levels

    paths = yaml.safe_load((workspace / "configs" / "base.yaml").read_text())["paths"]
    instances = list(validation_set(6, 6)[:2])
    outcomes = []
    for workers in (1, 2):
        root = tmp_path / f"workers{workers}"
        shutil.copytree(workspace / "artifacts" / "models", root / "models")      # no stored temperatures
        cfg = experiment(workspace, f"selection{workers}", {"paths": {**paths, "artifacts": str(root)},
                                                             "entropy": {"levels": [0.0, 0.5]}})
        artifacts = Artifacts(cfg, log=lambda *_: None)
        specs = [("MWKR", None, (6, 6), instances), ("LR", "lr_6x6", (6, 6), instances),
                 ("AZ", "az_6x6", (6, 6), instances)]
        pool = mp.get_context("spawn").Pool(workers, initializer=worker_init) if workers > 1 else None
        try:
            results = select_entropy_levels(cfg, artifacts, specs, [0.0, 0.5], iterations=6, seed=5, pool=pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        taus = {key: entry["tau"] for key, entry in artifacts.temperatures.data.items()}
        outcomes.append((results, taus))
    assert outcomes[0] == outcomes[1] and len(outcomes[0][1]) == 3       # H0.5 of MWKR, LR and AZ
    mwkr = select_entropy_level(lambda level: MWKRRollout(level, artifacts.temperature("MWKR", level, (6, 6))),
                                instances, [0.0, 0.5], iterations=6, seed=5)
    assert mwkr == outcomes[1][0][0]                                         # as the one-process function


@pytest.mark.slow
def test_experiment_in_worker_processes_matches_one_process(workspace, tmp_path):
    import shutil

    paths = yaml.safe_load((workspace / "configs" / "base.yaml").read_text())["paths"]
    body = {"instances": ["ft06"], "budgets": [{"iterations": 4}], "runs": 2,
            "configurations": [{"name": "MWKR", "entropy": 0.5}, {"name": "LR", "model": "lr_6x6", "entropy": 0.25}]}
    outcomes = []
    for workers in (1, 2):
        root = tmp_path / f"workers{workers}"
        shutil.copytree(workspace / "artifacts" / "models", root / "artifacts" / "models")   # no temperatures
        cfg = experiment(workspace, "parallel", {**body, "paths": {**paths, "artifacts": str(root / "artifacts"),
                                                                   "results": str(root / "results")}})
        rows = read_results(run_experiment(cfg, workers=workers, log=lambda *_: None))
        taus = json.loads((root / "artifacts" / "temperatures.json").read_text())
        outcomes.append((sorted((r["key"], r["makespan"]) for r in rows), {k: v["tau"] for k, v in taus.items()}))
    assert len(outcomes[0][0]) == 4 and len(outcomes[0][1]) == 3      # MWKR H0.5, LR H0.25 and its MWKR H0.25
    assert outcomes[0] == outcomes[1]


@pytest.mark.slow
def test_fixed_states_are_the_same_in_worker_processes(workspace):
    cfg = experiment(workspace, "rq3_workers", {
        "instances": ["ft06", "la01"], "budgets": [{"iterations": 3}],
        "entropy": {"levels": [0.0, 0.5]},
        "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": [0, 0.5], "calibration": "own_size"},
                           {"name": "LR", "model": "lr_{size}", "entropy": [0, 0.5]}],
        "fixed_states": {"per_instance": 4, "search_iterations": 8, "search_per_instance": 2},
    })
    one = evaluate_fixed_states(cfg, log=lambda *_: None)
    two = evaluate_fixed_states(cfg, log=lambda *_: None, workers=2)
    assert set(one["instance"]) == {"ft06", "la01"}
    pd.testing.assert_frame_equal(one, two)


@pytest.mark.slow
def test_pilot_study_costs_the_design(workspace):
    experiment(workspace, "rq1", {"instances": ["ft06", "la01"], "budgets": [{"iterations": 25}, {"iterations": 50}],
                                  "configurations": [{"name": "U"}, {"name": "MWKR", "entropy": "tuned"}]})
    experiment(workspace, "rq4", {"instances": ["ft06", "la01"], "budgets": [{"time": 50}],
                                  "configurations": [{"name": "U"}, {"name": "LR", "model": "lr_6x6"},
                                                     {"name": "LR", "model": "lr_10x10"}]})
    path = workspace / "configs" / "pilot.yaml"
    path.write_text(yaml.safe_dump({"inherits": "base.yaml", "models": {"_replace": True,
                                                                        "lr_6x6": {"kind": "gin", "n": 6, "m": 6}},
                                    "pilot": {"designs": ["rq1.yaml", "rq4.yaml"], "iterations": 5, "repeats": 1,
                                              "depths": [0.0, 0.5], "available_hours": 100, "workers": 2}}))
    result = pilot_study(load_config(path), log=lambda *_: None)
    timing = result["timing"]
    assert {"U", "MWKR", "LR"} <= set(timing["configuration"]) and (timing["ms_per_iteration_first_decision"] > 0).all()
    lr = timing[(timing.configuration == "LR") & (timing["size"] == "6x6")]["ms_per_iteration_first_decision"].iloc[0]
    u = timing[(timing.configuration == "U") & (timing["size"] == "6x6")]["ms_per_iteration_first_decision"].iloc[0]
    assert lr > u                                                             # a network call per rollout step
    design = result["design"]
    assert design["total_core_hours"].iloc[0] > 0 and design["fits"].all()
    assert design["rq4_runs"].iloc[1] < design["rq4_runs"].iloc[0]            # RQ4 without the 10 x 10 network
    assert (design["wall_hours"] == design["total_core_hours"] / 2).all()
    assert not result["training"].empty and (result["training"]["hours_per_later_generation"] > 0).all()
    assert isinstance(result["by_configuration"], pd.DataFrame)


@pytest.mark.slow
def test_rules_extension_selects_runs_and_is_analysed_with_rq2(workspace, capsys):
    """MOR, SPT and FIFO: entropy selection with U (MWKR's level is kept), runs, analysis with the RQ2 runs."""
    from mctsjssp.cli import main

    paths = yaml.safe_load((workspace / "configs" / "base.yaml").read_text())["paths"]
    shared = {"instances": ["ft06", "la01"], "budgets": [{"iterations": 4}, {"time": 4}], "runs": 2}
    rq2 = experiment(workspace, "rq2ext", {**shared, "configurations": [
        {"name": "U"}, {"name": "MWKR", "entropy": "tuned"}, {"name": "LR", "model": "lr_6x6", "entropy": 0.5}]})
    selection = {"levels": [0.0, 0.5], "selection": {"instances": 3, "iterations": 4}}
    first = experiment(workspace, "select_mwkr", {"entropy": {**selection, "selection": {
        **selection["selection"], "configurations": [{"name": "MWKR"}]}}})
    assert main(["select-entropy", "--config", first["_file"], "--workers", "1"]) == 0
    store = workspace / "artifacts" / "entropy_levels.json"
    mwkr_entry = json.loads(store.read_text())["MWKR"]
    run_dir_rq2 = run_experiment(rq2, workers=1, log=lambda *_: None)

    rules = experiment(workspace, "rules", {**shared, "configurations": [
        {"name": "MOR", "entropy": "tuned"}, {"name": "SPT", "entropy": "tuned"}, {"name": "FIFO", "entropy": "tuned"}],
        "entropy": {**selection, "selection": {**selection["selection"], "configurations": [
            {"name": "U"}, {"name": "MWKR"}, {"name": "MOR"}, {"name": "SPT"}, {"name": "FIFO"}]}}})
    assert main(["select-entropy", "--config", rules["_file"], "--workers", "1", "--missing"]) == 0
    printed = capsys.readouterr().out
    assert "[MWKR] level chosen earlier" in printed and "Fixed rollout distributions on the validation" in printed
    stored = json.loads(store.read_text())
    assert stored["MWKR"] == mwkr_entry                                       # kept, not selected again
    assert set(stored) >= {"U", "MOR", "SPT", "FIFO"} and list(stored["U"]["scores"]) == ["1.0"]

    run_dir = run_experiment(rules, workers=1, log=lambda *_: None)
    rows = read_results(run_dir)
    assert len(rows) == 2 * 3 * 2 * 2 and {r["config"] for r in rows} == {"MOR", "SPT", "FIFO"}
    budgets = json.loads((workspace / "artifacts" / "time_budgets.json").read_text())
    timed = [r for r in rows if r["budget_kind"] == "time"]
    assert all(r["budget_seconds"] == budgets[f"{r['instance']}|I4|s0"]["mean"] for r in timed)   # RQ2's budgets
    taus = json.loads((workspace / "artifacts" / "temperatures.json").read_text())
    assert {f"{n}|-|6x6|H0.5" for n in ("MOR", "SPT", "FIFO")} <= set(taus)

    assert main(["analyse", "--config", rules["_file"], "--with", str(run_dir_rq2), "--no-figures"]) == 0
    out = run_dir / "analysis"
    validation = pd.read_csv(out / "validation.csv")
    assert set(validation["distribution"]) == {"U", "MWKR", "MOR", "SPT", "FIFO"}
    assert validation["validation_ratio"].is_monotonic_increasing and "realised_entropy" in validation
    best = pd.read_csv(out / "tests_best_fixed.csv")
    assert set(best["best_fixed"]) == {validation.loc[0, "distribution"]}
    tests = pd.read_csv(out / "tests.csv")
    assert len(set(tests["second"])) == 5 and set(tests["first"]) == {"LR[lr_6x6]@H0.5"}
    summary = pd.read_csv(out / "summary.csv")
    assert summary["label"].nunique() == 6                                     # three rules with U, MWKR and LR
    assert not (run_dir_rq2 / "analysis").exists()                           # RQ2's own analysis is untouched
