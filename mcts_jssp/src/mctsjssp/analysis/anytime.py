"""Anytime curves: the best makespan found so far against the wall time of a run (convergence to the BKS).

Every run records ``anytime``, the pairs (seconds since the start of the episode, best makespan of any
complete schedule found so far), one pair at every improvement; the committed schedule is included.  Its gap
to the BKS is a step function of time.  The curves are evaluated on a common log-spaced grid of times:

* before a run's first complete schedule its gap is undefined; a point of a curve is shown only once every
  run of that configuration has a schedule (``runs_defined == runs``);
* after a run has finished its gap stays at its final value (the run has nothing better to offer later).

Per configuration, budget and group (all instances, and every size class) the table gives the mean gap to the
BKS over all runs, the share of runs that have reached the BKS, and the number of runs.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import SIZE_CLASSES, budget_order

GRID_POINTS = 60


def _grid(frame: pd.DataFrame, points: int = GRID_POINTS) -> np.ndarray:
    firsts = [tr[0][0] for tr in frame["anytime"] if isinstance(tr, list) and tr]
    ends = pd.to_numeric(frame["time_total"], errors="coerce").dropna()
    lo = max(min(firsts) if firsts else 1e-4, 1e-5)
    hi = max(float(ends.max()) if len(ends) else lo * 10, lo * 1.01)
    return np.geomspace(lo, hi, points)


def _gaps_on_grid(trace: list, bks: float, grid: np.ndarray) -> np.ndarray:
    """Gap to the BKS (%) of one run at every time of ``grid``; NaN before its first complete schedule."""
    out = np.full(len(grid), np.nan)
    if not trace:
        return out
    times = np.array([t for t, _ in trace], dtype=float)
    best = np.minimum.accumulate(np.array([m for _, m in trace], dtype=float))
    idx = np.searchsorted(times, grid, side="right") - 1
    ok = idx >= 0
    out[ok] = (best[idx[ok]] - bks) / bks * 100.0
    return out


def anytime_table(frame: pd.DataFrame, groups: Iterable[str] = ("all", *SIZE_CLASSES),
                  points: int = GRID_POINTS) -> pd.DataFrame:
    """Mean gap to the BKS of the best schedule found so far, against time (one row per grid point)."""
    if "anytime" not in frame.columns or frame["anytime"].isna().all():
        return pd.DataFrame()
    frame = frame[frame["anytime"].apply(lambda v: isinstance(v, list))]
    rows = []
    for budget in budget_order(frame["budget"].unique()):
        fb = frame[frame["budget"] == budget]
        grid = _grid(fb, points)
        for group in groups:
            fg = fb if group == "all" else fb[fb["size_class"] == group]
            if fg.empty:
                continue
            for (label, config), runs in fg.groupby(["label", "config"], sort=False):
                gaps = np.vstack([_gaps_on_grid(tr, bks, grid) for tr, bks in zip(runs["anytime"], runs["bks"])])
                defined = np.isfinite(gaps).sum(axis=0)
                sums = np.nansum(gaps, axis=0)
                mean = np.where(defined > 0, sums / np.maximum(defined, 1), np.nan)
                at_bks = (np.nan_to_num(gaps, nan=np.inf) <= 1e-9).sum(axis=0) / len(runs)
                level = runs["level"].iloc[0] if "level" in runs.columns else None
                for t, g, d, share in zip(grid, mean, defined, at_bks):
                    rows.append({"budget": budget, "group": group, "label": label, "config": config,
                                 "level": level, "time": float(t), "mean_gap": float(g), "runs": len(runs),
                                 "runs_defined": int(d), "share_at_bks": float(share)})
    return pd.DataFrame(rows)


def anytime_summary(table: pd.DataFrame, fractions=(0.01, 0.1, 1.0)) -> pd.DataFrame:
    """Mean gap at a few moments of the run (as fractions of the longest run of each budget), for reports."""
    if table.empty:
        return table
    out = []
    for (budget, group), sub in table.groupby(["budget", "group"], sort=False):
        t_max = sub["time"].max()
        for (label, config), s in sub.groupby(["label", "config"], sort=False):
            row = {"budget": budget, "group": group, "label": label, "config": config}
            for f in fractions:
                k = int(np.searchsorted(s["time"].to_numpy(), f * t_max, side="right")) - 1
                ok = k >= 0 and s["runs_defined"].iloc[k] == s["runs"].iloc[k]
                row[f"gap_at_{f:g}_of_time"] = float(s["mean_gap"].iloc[k]) if ok else np.nan
                row[f"bks_share_at_{f:g}_of_time"] = float(s["share_at_bks"].iloc[k]) if k >= 0 else 0.0
            out.append(row)
    return pd.DataFrame(out)
