"""Performance measures of Section 3.6.4.

Solution quality is the optimality gap of the reported schedule, ``(Cmax - BKS) / BKS * 100``, averaged over
the runs of each instance and summarised over all instances, per family and per size class.  The central
measure is convergence to the BKS: the gap as a function of the number of iterations and of the computation
time per decision, the time that a configuration needs to reach a target gap, and the number of instances on
which the BKS itself is reached.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SIZE_CLASSES = ("small", "medium", "large")


def load_frame(run_dir: str | Path) -> pd.DataFrame:
    """All result rows of an experiment as a DataFrame."""
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no results at {path}; run the experiment first")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    frame = pd.DataFrame(rows)
    if "sigma" in frame.columns:
        frame["sigma"] = frame["sigma"].fillna(0.0)
    return frame


def budget_order(labels: Iterable[str]) -> list[str]:
    """Budgets in a natural order: iteration budgets by size, then time budgets by size."""
    def key(label: str):
        return (label[0] != "I", int("".join(ch for ch in label if ch.isdigit()) or 0))
    return sorted(set(labels), key=key)


def instance_means(frame: pd.DataFrame, value: str = "gap",
                   by: Sequence[str] = ("label", "budget")) -> pd.DataFrame:
    """Mean of ``value`` over the runs of every instance: the observation of the statistical tests."""
    keys = ["instance", *by]
    extra = [c for c in ("family", "size_class", "n", "m", "n_ops", "config", "variant", "level", "train_size")
             if c in frame.columns and c not in keys]
    grouped = frame.groupby(keys, dropna=False, sort=False)
    out = grouped[value].mean().reset_index()
    if extra:
        out = out.merge(grouped[extra].first().reset_index(), on=keys)
    return out


def wide(means: pd.DataFrame, value: str = "gap", columns: str = "label") -> pd.DataFrame:
    """Table with one row per instance and one column per configuration."""
    return means.pivot(index="instance", columns=columns, values=value)


def summarise(frame: pd.DataFrame, value: str = "gap", by: Sequence[str] = ("label", "budget")) -> pd.DataFrame:
    """Mean, median and spread of the instance means over all instances, per size class and per family."""
    means = instance_means(frame, value, by)
    parts = []
    groupings = [("all", None)] + [(f"size:{c}", ("size_class", c)) for c in SIZE_CLASSES
                                   if c in set(means["size_class"])]
    groupings += [(f"family:{f}", ("family", f)) for f in sorted(set(means["family"]))]
    for group, selector in groupings:
        sel = means if selector is None else means[means[selector[0]] == selector[1]]
        if sel.empty:
            continue
        agg = sel.groupby(list(by), sort=False)[value].agg(["mean", "median", "std", "min", "max", "count"])
        agg = agg.reset_index().rename(columns={"count": "instances"})
        agg.insert(0, "group", group)
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def bks_hits(frame: pd.DataFrame, by: Sequence[str] = ("label", "budget")) -> pd.DataFrame:
    """Number of instances on which the BKS is reached by the reported schedule (in any run, in every run)
    and by the best schedule of a rollout."""
    data = frame.assign(hit=frame["makespan"] <= frame["bks"],
                        hit_rollout=frame["best_rollout_makespan"] <= frame["bks"])
    per_instance = data.groupby(["instance", *by], sort=False).agg(any_run=("hit", "any"), every_run=("hit", "all"),
                                                                   rollout=("hit_rollout", "any")).reset_index()
    out = per_instance.groupby(list(by), sort=False).agg(
        instances=("instance", "count"), bks_reached_any_run=("any_run", "sum"),
        bks_reached_every_run=("every_run", "sum"), bks_reached_by_a_rollout=("rollout", "sum")).reset_index()
    return out


def convergence(frame: pd.DataFrame, by: Sequence[str] = ("label",)) -> pd.DataFrame:
    """Gap as a function of the iterations and of the computation time per decision.

    One row per configuration and budget: the mean over the instances of the instance means.
    """
    cols = ["gap", "best_rollout_gap", "iterations_per_decision", "time_per_decision"]
    keys = ["instance", *by, "budget", "budget_kind"]
    per_instance = frame.groupby(keys, sort=False)[cols].mean().reset_index()
    out = per_instance.groupby([*by, "budget", "budget_kind"], sort=False)[cols].mean().reset_index()
    out["instances"] = per_instance.groupby([*by, "budget", "budget_kind"], sort=False)["instance"].count().values
    order = {b: i for i, b in enumerate(budget_order(out["budget"]))}
    return out.sort_values([*by, "budget"], key=lambda s: s.map(order) if s.name == "budget" else s) \
        .reset_index(drop=True)


def value_to_reach(x: Sequence[float], gap: Sequence[float], target: float) -> float:
    """Smallest ``x`` (iterations or seconds) at which the gap curve reaches ``target``.

    The points are sorted by ``x`` and the curve is interpolated linearly between them; NaN if the target is
    not reached within the budgets that were run.
    """
    pts = sorted(zip(x, gap))
    if not pts:
        return float("nan")
    if pts[0][1] <= target:
        return float(pts[0][0])
    for (x0, g0), (x1, g1) in zip(pts, pts[1:]):
        if g1 <= target < g0:
            return float(x0 + (g0 - target) / (g0 - g1) * (x1 - x0))
    return float("nan")


def time_to_target(conv: pd.DataFrame, baseline_labels: Sequence[str], by: str = "label") -> pd.DataFrame:
    """Time (and iterations) that each configuration needs to reach the gap that the better of the baseline
    configurations (U and MWKR) attains with the largest budget.

    ``conv`` is the table of :func:`convergence`.  The target is taken from the largest iteration budget.
    """
    iters = conv[conv["budget_kind"] == "iterations"]
    if iters.empty:
        return pd.DataFrame()
    largest = budget_order(iters["budget"])[-1]
    base = iters[(iters["budget"] == largest) & (iters[by].isin(list(baseline_labels)))]
    if base.empty:
        return pd.DataFrame()
    target = float(base["gap"].min())
    target_label = base.loc[base["gap"].idxmin(), by]
    rows = []
    for label, grp in conv.groupby(by, sort=False):
        it = grp[grp["budget_kind"] == "iterations"]
        rows.append({
            by: label, "target_gap": target, "target_from": f"{target_label} at {largest}",
            "iterations_per_decision": value_to_reach(it["iterations_per_decision"], it["gap"], target),
            "seconds_per_decision": value_to_reach(grp["time_per_decision"], grp["gap"], target),
            "best_gap": float(grp["gap"].min()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- detailed records
def load_records(run_dir: str | Path, run_ids: Iterable[str], kind: str = "steps",
                 columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Detailed records (``decisions`` or ``steps``) of the given runs, with the run identifier attached."""
    rec_dir = Path(run_dir) / "records"
    rows = []
    for run_id in run_ids:
        path = rec_dir / f"{run_id}.{kind}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if columns is not None:
                    rec = {k: rec.get(k) for k in columns}
                rec["run_id"] = run_id
                rows.append(rec)
    return pd.DataFrame(rows)


def load_rollouts(run_dir: str | Path, run_id: str) -> Optional[dict]:
    """Return and length of every rollout of one run, if they were recorded."""
    path = Path(run_dir) / "records" / f"{run_id}.rollouts.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
