"""Performance measures and the analysis of the research questions, on synthetic result rows."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mctsjssp.analysis import (bks_hits, convergence, instance_means, load_frame, summarise, time_to_target,
                               value_to_reach, write_report)
from mctsjssp.analysis.metrics import budget_order
from mctsjssp.analysis.rq import analyse, labels_of, paired_tests
from mctsjssp.instances import EXPECTED_SIZE, FAMILY


def synthetic_rows(experiment="rq2", instances=("ft06", "la01", "la06", "la21", "la26", "la31", "abz7", "swv11"),
                   budgets=(("I25", "iterations", 25), ("I100", "iterations", 100), ("T50", "time", 0)),
                   configs=(("U", "U", 1.0, None, 12.0), ("MWKR", "MWKR@H0.25", 0.25, None, 9.0),
                            ("LR", "LR[lr_15x15]@H0.5", 0.5, "15x15", 5.0)),
                   runs=3, seed=0):
    """Result rows in the format of ``results.jsonl`` with a known ordering of the configurations."""
    rng = np.random.default_rng(seed)
    rows = []
    for name in instances:
        n, m = EXPECTED_SIZE[name]
        bks = 1000
        ops = n * m
        size_class = "small" if ops <= 100 else "medium" if ops <= 250 else "large"
        for config, label, level, train, base_gap in configs:
            for budget, kind, iterations in budgets:
                for run in range(runs):
                    gap = max(0.0, base_gap * (30 / (iterations or 40)) ** 0.3 + rng.normal(0, 0.4))
                    makespan = int(round(bks * (1 + gap / 100)))
                    rows.append({
                        "experiment": experiment, "run_id": f"{name}-{label}-{budget}-{run}", "instance": name,
                        "family": FAMILY[name], "n": n, "m": m, "n_ops": ops, "size_class": size_class, "bks": bks,
                        "lb0": 900, "config": config, "variant": label.split("@")[0], "label": label, "level": level,
                        "tau": None, "model": None, "train_size": train, "budget": budget, "budget_kind": kind,
                        "budget_iterations": iterations, "run": run, "sigma": None, "makespan": makespan,
                        "gap": (makespan - bks) / bks * 100, "ratio": makespan / 900,
                        "best_rollout_makespan": makespan - 5, "best_rollout_gap": (makespan - 5 - bks) / bks * 100,
                        "decisions": ops, "iterations": (iterations or 40) * ops,
                        "iterations_per_decision": float(iterations or 40),
                        "time_per_decision": 0.001 * (iterations or 40) * (5 if config == "LR" else 1),
                    })
    return rows


def write_rows(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "results.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------------------------- metrics
def test_instance_means_average_the_runs():
    frame = pd.DataFrame(synthetic_rows())
    means = instance_means(frame, "gap")
    assert len(means) == 8 * 3 * 3                                 # instances x configurations x budgets
    sel = frame[(frame.instance == "ft06") & (frame.label == "U") & (frame.budget == "I25")]
    got = means[(means.instance == "ft06") & (means.label == "U") & (means.budget == "I25")]["gap"].iloc[0]
    assert got == pytest.approx(sel["gap"].mean())
    assert {"family", "size_class", "n_ops", "config"} <= set(means.columns)


def test_summary_groups_by_size_class_and_family():
    frame = pd.DataFrame(synthetic_rows())
    table = summarise(frame)
    groups = set(table["group"])
    assert {"all", "size:small", "size:medium", "size:large", "family:la", "family:swv"} <= groups
    row = table[(table.group == "all") & (table.label == "U") & (table.budget == "I25")].iloc[0]
    assert row["instances"] == 8
    small = table[(table.group == "size:small") & (table.label == "U") & (table.budget == "I25")].iloc[0]
    assert small["instances"] == 3                                 # ft06, la01, la06


def test_bks_hits_counts_instances():
    rows = synthetic_rows(runs=2)
    for row in rows:
        if row["instance"] == "ft06" and row["label"] == "U" and row["budget"] == "I25":
            row["makespan"] = row["bks"] if row["run"] == 0 else row["bks"] + 7
    table = bks_hits(pd.DataFrame(rows))
    hit = table[(table.label == "U") & (table.budget == "I25")].iloc[0]
    assert hit["bks_reached_any_run"] == 1 and hit["bks_reached_every_run"] == 0 and hit["instances"] == 8


def test_convergence_and_budget_order():
    assert budget_order(["T200", "I100", "I25", "T50", "I200"]) == ["I25", "I100", "I200", "T50", "T200"]
    conv = convergence(pd.DataFrame(synthetic_rows()))
    assert list(conv[conv.label == "U"]["budget"]) == ["I25", "I100", "T50"]
    u = conv[conv.label == "U"].set_index("budget")
    assert u.loc["I100", "gap"] < u.loc["I25", "gap"]                # more iterations, smaller gap
    assert (conv["instances"] == 8).all()


def test_value_to_reach_interpolates():
    assert value_to_reach([25, 50, 100], [10.0, 8.0, 4.0], 6.0) == pytest.approx(75.0)
    assert value_to_reach([25, 50, 100], [10.0, 8.0, 4.0], 12.0) == 25.0     # reached at the smallest budget
    assert np.isnan(value_to_reach([25, 50, 100], [10.0, 8.0, 4.0], 1.0))    # never reached
    assert np.isnan(value_to_reach([], [], 1.0))


def test_time_to_target_uses_the_better_baseline_at_the_largest_budget():
    frame = pd.DataFrame(synthetic_rows())
    conv = convergence(frame)
    table = time_to_target(conv, ["U", "MWKR@H0.25"]).set_index("label")
    mwkr_at_largest = conv[(conv.label == "MWKR@H0.25") & (conv.budget == "I100")]["gap"].iloc[0]
    assert table.loc["U", "target_gap"] == pytest.approx(mwkr_at_largest)
    assert table.loc["MWKR@H0.25", "iterations_per_decision"] == pytest.approx(100.0)
    assert np.isnan(table.loc["U", "iterations_per_decision"])     # U does not reach the target
    assert table.loc["LR[lr_15x15]@H0.5", "iterations_per_decision"] == pytest.approx(25.0)


# ----------------------------------------------------------------------------------------- RQ analyses
def test_rq1_and_rq2_tables(tmp_path):
    frame = pd.DataFrame(synthetic_rows())
    assert labels_of(frame, "LR") == ["LR[lr_15x15]@H0.5"] and labels_of(frame, "MWKR", level=0.25) == ["MWKR@H0.25"]
    rq1 = analyse(frame[frame.config != "LR"], "rq1")
    tests = rq1["tests"]
    assert set(tests["group"]) == {"all", "small", "medium", "large"}
    whole = tests[tests.group == "all"]
    assert (whole["first"] == "MWKR@H0.25").all() and (whole["rank_biserial"] > 0).all()   # MWKR is better here
    assert (whole["p_holm"] >= whole["p_value"]).all() and "budgets" not in whole["holm_family"].iloc[0]
    rq2 = analyse(frame, "rq2")
    friedman = rq2["friedman"]
    assert (friedman[friedman.group == "all"]["rejects"]).all()
    assert friedman["rank[LR[lr_15x15]@H0.5]"].max() <= friedman["rank[U]"].min()
    post = rq2["tests"]
    assert set(post["second"]) == {"U", "MWKR@H0.25"} and post["friedman_rejects"].dtype == bool
    assert not rq2["time_to_target"].empty and not rq2["bks_reached"].empty


def test_holm_family_follows_the_design():
    frame = pd.DataFrame(synthetic_rows())
    pairs = [("LR[lr_15x15]@H0.5", "U"), ("LR[lr_15x15]@H0.5", "MWKR@H0.25")]
    per_budget = paired_tests(frame, pairs, groups=("all",), holm_over="pairs")
    across = paired_tests(frame, pairs, groups=("all",), holm_over="budgets")
    assert (across["p_holm"] >= per_budget["p_holm"] - 1e-12).all()   # a larger family corrects more
    assert per_budget["holm_family"].iloc[0].endswith("group and budget")
    assert across["holm_family"].iloc[0].endswith("group")


def test_rq3_tables():
    levels = (0.0, 0.25, 0.5, 0.75)
    configs = [("U", "U", 1.0, None, 12.0)]
    for config, template, base in (("MWKR", "MWKR", 9.0), ("LR", "LR[lr_{size}]", 7.0), ("AZ", "AZ[az_{size}]", 8.0)):
        configs += [(config, f"{template}@H{lv:g}", lv, None if config == "MWKR" else "6x6", base + 2 * lv)
                    for lv in levels]
    rows = synthetic_rows("rq3", instances=("ft06", "ft10", "la16", "la17", "la18", "orb01", "orb02", "la36"),
                          budgets=(("I100", "iterations", 100), ("T200", "time", 0)), configs=tuple(configs))
    rng = np.random.default_rng(1)
    for row in rows:                                                # recorded behaviour of the runs
        row.update(rollout_entropy=row["level"] + rng.normal(0, 0.01), return_var=0.01 + 0.05 * row["level"],
                   return_bias=-0.05 * row["level"], visit_entropy_norm=0.9, top_visit_share=0.3, value_var=0.001,
                   kl_mwkr=0.3 if row["config"] in ("LR", "AZ") else None, agree_mwkr=0.5, mwkr_rank=2.0)
    tables = analyse(pd.DataFrame(rows), "rq3")
    same = tables["tests_same_entropy"]
    assert len(same) == 2 * 4 * 3                                   # budgets x levels x (LR-MWKR, AZ-MWKR, LR-AZ)
    assert same["holm_family"].iloc[0].endswith("group and budget")
    levels_table = tables["friedman_levels"]
    assert set(levels_table["comparison"]) == {"entropy levels of MWKR", "entropy levels of LR", "entropy levels of AZ"}
    assert (levels_table["configurations"] == 5).all()             # four levels and U as level 1
    behaviour = tables["behaviour"]
    assert (behaviour["entropy_error"].abs() < 0.05).all()         # realised entropy against its target
    relations = tables["relations"].set_index(["property", "outcome"])
    assert relations.loc[("level", "return_var"), "spearman_rho"] > 0.9


def test_rq6_tables():
    configs = (("U", "U", 1.0, None, 12.0), ("LR", "LR@H0.5", 0.5, "6x6", 6.0), ("LR", "LR-SP@H0.5", 0.5, "6x6", 11.0),
               ("AZ", "AZ@H0.5", 0.5, "6x6", 8.0), ("AZ", "AZ-SP@H0.5", 0.5, "6x6", 12.5))
    rows = synthetic_rows("rq6", instances=("ft06", "ft10", "la16", "la17", "la18", "orb01", "orb02", "la36"),
                          budgets=(("I100", "iterations", 100),), configs=configs)
    tables = analyse(pd.DataFrame(rows), "rq6")
    tests = tables["tests"]
    every = tests[tests.group == "all"]
    assert set(zip(every["first"], every["second"])) == {
        ("LR@H0.5", "LR-SP@H0.5"), ("AZ@H0.5", "AZ-SP@H0.5"),
        ("LR@H0.5", "U"), ("LR-SP@H0.5", "U"), ("AZ@H0.5", "U"), ("AZ-SP@H0.5", "U")}
    assert (every["n"] == 8).all() and every["holm_family"].iloc[0].endswith("group and budget")
    lr = every[(every["first"] == "LR@H0.5") & (every["second"] == "LR-SP@H0.5")].iloc[0]
    assert lr["mean_first"] < lr["mean_second"] and lr["rank_biserial"] != 0


def test_rq4_tables():
    configs = (("U", "U", 1.0, None, 12.0), ("MWKR", "MWKR@H0.25", 0.25, None, 9.0),
               ("LR", "LR[lr_6x6]@H0.5", 0.5, "6x6", 7.0), ("LR", "LR[lr_15x15]@H0.5", 0.5, "15x15", 6.0))
    rows = synthetic_rows("rq4", instances=("ft06", "la01", "la16", "la21", "la36", "la37", "la31", "swv11"),
                          budgets=(("T200", "time", 0),), configs=configs)
    tables = analyse(pd.DataFrame(rows), "rq4")
    ratio = tables["by_size_ratio"]
    small_net = ratio[(ratio.train_size == "6x6") & (ratio.instances_of == "training size")]
    assert small_net["instances"].iloc[0] == 1                      # ft06 is the only 6 x 6 instance
    big_net = ratio[(ratio.train_size == "15x15") & (ratio.instances_of == "training size")]
    assert big_net["instances"].iloc[0] == 2                        # la36 and la37
    numeric = ratio[ratio.ratio != "all"]
    assert float(numeric[numeric.train_size == "6x6"]["ratio"].astype(float).max()) == pytest.approx(500 / 36, abs=0.01)
    tests = tables["tests_other_sizes"]
    assert set(tests["train_size"]) == {"6x6", "15x15"} and (tests[tests.train_size == "6x6"]["n"] == 7).all()
    assert len(tables["trend"]) == 4


def _rq5_rows(configs):
    rows = []
    for sigma in (0.1, 0.2):
        part = synthetic_rows("rq5", instances=("ft06", "ft10", "la16", "la36", "abz5", "orb01", "la01", "la21"),
                              budgets=(("T200", "time", 0),), configs=configs, seed=int(sigma * 10))
        for row in part:
            row.update(sigma=sigma, realised_gap=row["gap"] + 30 * sigma, realised_std_pct=10 * sigma,
                       realised_increase=25 * sigma)
        rows += [r for r in part if not (r["config"] == "AZ" and r["instance"] in ("la01", "la21"))]
    return pd.DataFrame(rows)


def test_rq5_tables_answer_both_questions():
    configs = (("U", "U", 1.0, None, 12.0), ("MWKR", "MWKR@H0.25", 0.25, None, 9.0),
               ("LR", "LR@H0.5", 0.5, "15x15", 7.0), ("LR", "LR-FT@H0.5", 0.5, "15x15", 6.5),
               ("LR", "LR-SC@H0.5", 0.5, "15x15", 6.8), ("AZ", "AZ@H0.5", 0.5, "6x6", 8.0),
               ("AZ", "AZ-FT@H0.5", 0.5, "6x6", 7.9), ("AZ", "AZ-SC@H0.5", 0.5, "6x6", 8.2))
    tables = analyse(_rq5_rows(configs), "rq5")
    summary = tables["summary"]
    assert set(summary["sigma"]) == {0.1, 0.2} and "realised_gap_mean" in summary.columns
    az = summary[summary.label.str.startswith("AZ")]
    assert (az["instances"] == 6).all() and (summary[summary.label == "U"]["instances"] == 8).all()
    tests = tables["tests"]
    assert set(tests["question"]) == {"training with disturbances", "robustness"}
    assert set(tests["measure"]) == {"realised_gap", "realised_std_pct", "realised_increase"}
    training = tables["tests_training"]
    pairs = set(zip(training["first"], training["second"]))
    assert pairs == {("LR-FT@H0.5", "LR@H0.5"), ("LR-SC@H0.5", "LR@H0.5"), ("LR-FT@H0.5", "LR-SC@H0.5"),
                     ("AZ-FT@H0.5", "AZ@H0.5"), ("AZ-SC@H0.5", "AZ@H0.5"), ("AZ-FT@H0.5", "AZ-SC@H0.5")}
    robust = tables["tests_robustness"]
    assert {("LR@H0.5", "U"), ("LR@H0.5", "MWKR@H0.25"), ("AZ@H0.5", "U")} <= set(zip(robust["first"], robust["second"]))
    assert (robust[robust["first"].str.startswith("AZ")]["n"] == 6).all()   # AZ only on instances of a training size
    assert set(tables["friedman"]["sigma"]) == {0.1, 0.2}


def test_rq5_tables_of_results_without_variants():
    configs = (("U", "U", 1.0, None, 12.0), ("MWKR", "MWKR@H0.25", 0.25, None, 9.0),
               ("LR", "LR[lr_15x15_s{sigma}]@H0.5", 0.5, "15x15", 7.0), ("AZ", "AZ[az_{size}_s{sigma}]@H0.5", 0.5, "6x6", 8.0))
    tables = analyse(_rq5_rows(configs), "rq5")
    assert set(tables["tests"]["question"]) == {"robustness"}
    assert set(tables["tests"]["second"]) == {"U", "MWKR@H0.25"}


def test_report_writes_tables_and_figures(tmp_path):
    run_dir = write_rows(tmp_path / "rq2-test", synthetic_rows())
    frame = load_frame(run_dir)
    assert len(frame) == 8 * 3 * 3 * 3 and (frame["sigma"] == 0.0).all()
    tables = write_report(run_dir, "rq2", log=lambda *_: None)
    out = run_dir / "analysis"
    for name in ("summary", "convergence", "bks_reached", "friedman", "tests", "time_to_target"):
        assert (out / f"{name}.csv").exists() and not tables[name].empty
    assert (out / "convergence_iterations.png").exists() and (out / "convergence_time.pdf").exists()
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path / "nothing")
