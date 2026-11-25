"""Results of the rollout policies of several experiments, instance by instance.

Every configuration of every experiment gets the name ``<experiment>:<label>`` (``LR@H0.5`` in ``rq3`` becomes
``rq3:LR@H0.5``).  Runs are compared only on the same instance and budget; disturbed runs (RQ5) are compared
only with runs of the same disturbance level ``sigma``.

Tables:

* ``per_instance`` - one row per instance, budget and configuration: runs, mean, best and standard deviation of
  the gap to the BKS, mean time, iterations, and the rank of the configuration on that instance and budget
  (1 = smallest mean gap; ties share the mean rank);
* ``gap_by_instance_<budget>`` - instances as rows (smallest first) and configurations as columns;
* ``vs_reference`` - every configuration against the reference (U by default) over the instances they share:
  instances better, equal and worse, the Wilcoxon signed-rank test with Holm's correction, the median
  difference and the rank-biserial correlation (Section 3.6.4);
* ``mean_rank`` - the mean rank of every configuration per budget over the instances that all configurations
  share.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import budget_order, load_frame
from .stats import compare_pairs


def load_experiments(run_dirs: dict) -> pd.DataFrame:
    frames = []
    for experiment, run_dir in run_dirs.items():
        f = load_frame(run_dir)
        f["experiment"] = experiment
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    if "sigma" not in frame.columns:
        frame["sigma"] = 0.0
    frame["sigma"] = frame["sigma"].fillna(0.0)
    single = frame["experiment"].nunique() == 1
    frame["name"] = frame["label"] if single else frame["experiment"] + ":" + frame["label"]
    return frame


def per_instance(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["sigma", "budget", "instance", "name"]
    g = frame.groupby(keys, sort=False)
    out = g.agg(runs=("gap", "size"), mean_gap=("gap", "mean"), best_gap=("gap", "min"), sd_gap=("gap", "std"),
                mean_time=("time_total", "mean") if "time_total" in frame else ("gap", "size"),
                iterations=("iterations", "mean") if "iterations" in frame else ("gap", "size")).reset_index()
    extra = [c for c in ("experiment", "label", "config", "level", "size_class", "n", "m", "n_ops", "bks")
             if c in frame.columns]
    out = out.merge(g[extra].first().reset_index(), on=keys)
    out["rank"] = out.groupby(["sigma", "budget", "instance"])["mean_gap"].rank(method="average")
    out["configurations_on_instance"] = out.groupby(["sigma", "budget", "instance"])["name"].transform("count")
    return out.sort_values(["sigma", "budget", "n_ops" if "n_ops" in out else "instance", "instance", "rank"])


def wide_by_budget(table: pd.DataFrame) -> dict:
    out = {}
    for (sigma, budget), g in table.groupby(["sigma", "budget"], sort=False):
        order = g.drop_duplicates("instance").sort_values(["n_ops", "instance"] if "n_ops" in g else ["instance"])
        w = g.pivot(index="instance", columns="name", values="mean_gap").reindex(order["instance"])
        tag = budget if not sigma else f"{budget}_s{sigma:g}"
        out[tag] = w
    return out


def choose_reference(names, reference: Optional[str]) -> Optional[str]:
    names = list(names)
    if reference and reference in names:
        return reference
    if reference:
        hits = [n for n in names if n.endswith(":" + reference) or n == reference]
        if hits:
            return hits[0]
    return next((n for n in names if n == "U" or n.endswith(":U")), None)


def vs_reference(wides: dict, reference: Optional[str] = None, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for tag, w in wides.items():
        ref = choose_reference(w.columns, reference)
        if ref is None:
            continue
        others = [c for c in w.columns if c != ref]
        res = compare_pairs(w, [(c, ref) for c in others], alpha=alpha)
        if res.empty:
            continue
        for c in others:
            both = w[[c, ref]].dropna()
            d = both[c] - both[ref]
            sel = res["first"] == c
            res.loc[sel, "better"] = int((d < -1e-12).sum())
            res.loc[sel, "equal"] = int((d.abs() <= 1e-12).sum())
            res.loc[sel, "worse"] = int((d > 1e-12).sum())
        res.insert(0, "budget", tag)
        res = res.rename(columns={"first": "configuration", "second": "reference"})
        rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mean_rank(wides: dict) -> pd.DataFrame:
    rows = []
    for tag, w in wides.items():
        common = w.dropna()
        if common.empty:
            continue
        ranks = common.rank(axis=1, method="average").mean()
        for name, r in ranks.items():
            rows.append({"budget": tag, "configuration": name, "mean_rank": float(r), "instances": len(common),
                         "mean_gap": float(common[name].mean())})
    out = pd.DataFrame(rows)
    return out.sort_values(["budget", "mean_rank"]) if not out.empty else out


def compare_experiments(run_dirs: dict, reference: Optional[str] = None, alpha: float = 0.05) -> dict:
    frame = load_experiments(run_dirs)
    table = per_instance(frame)
    wides = wide_by_budget(table)
    tables = {"per_instance": table, "vs_reference": vs_reference(wides, reference, alpha),
              "mean_rank": mean_rank(wides)}
    for tag, w in wides.items():
        tables[f"gap_by_instance_{tag}"] = w.reset_index()
    return {"tables": tables, "wides": wides}


def budget_tags(wides: dict) -> list:
    plain = [t for t in wides if "_s" not in t]
    return budget_order(plain) + [t for t in wides if "_s" in t]
