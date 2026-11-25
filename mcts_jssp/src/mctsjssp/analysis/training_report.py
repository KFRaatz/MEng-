"""Training curves of every network from its history.json: a summary table and one figure of small multiples.

Per network the summary gives the generations played, why training stopped, the best validation Cmax/LB0 and
its generation, the score of uniform random dispatching on the same validation instances, the improvement over
it, and the last entropy of the visit targets and of the network's policy.  ``learned`` is False when the best
validation score does not beat uniform random dispatching: such a network has not learned to schedule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_histories(models_dir: Path) -> dict:
    out = {}
    for hist in sorted(Path(models_dir).glob("*/history.json")):
        rows = json.loads(hist.read_text(encoding="utf-8"))
        if rows:
            out[hist.parent.name] = pd.DataFrame(rows)
    return out


def summary(histories: dict, models_dir: Path) -> pd.DataFrame:
    rows = []
    for name, h in histories.items():
        best = h.loc[h["validation_ratio"].idxmin()]
        uniform = float(h["uniform_ratio"].iloc[-1]) if "uniform_ratio" in h else np.nan
        meta = {}
        summ = Path(models_dir) / name / "summary.json"
        if summ.exists():
            meta = json.loads(summ.read_text(encoding="utf-8"))
        best_gen, best_val = int(best["generation"]), float(best["validation_ratio"])
        if meta.get("best_validation_ratio") is not None and float(meta["best_validation_ratio"]) < best_val:
            best_gen, best_val = int(meta.get("best_generation", -1)), float(meta["best_validation_ratio"])  # warm start
        rows.append({
            "model": name, "generations": len(h), "stopped_by": meta.get("stopped_by"),
            "best_generation": best_gen, "best_validation": best_val,
            "uniform": uniform, "improvement_over_uniform_pct": (uniform - best_val) / uniform * 100
            if uniform == uniform else np.nan,
            "learned": bool(best_val < uniform) if uniform == uniform else None,
            "last_target_entropy": float(h["target_entropy"].iloc[-1]) if "target_entropy" in h else np.nan,
            "last_policy_entropy": float(h["policy_entropy"].iloc[-1]) if "policy_entropy" in h else np.nan,
            "minutes": float(h["seconds"].sum() / 60) if "seconds" in h else np.nan,
            "model_steps": int(h["model_steps"].iloc[-1]) if "model_steps" in h else None,
        })
    return pd.DataFrame(rows)
