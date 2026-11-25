"""How the networks for disturbed processing times develop during training (RQ5): one table with every
iteration of every such network, fine-tuned (``FT``, in ``artifacts/models``) and trained from scratch (``SC``, in
``artifacts/scratch/models``), and two figures over the iterations:

* ``disturbed_over_time.png``: the realised Cmax / LB0 of the greedy schedules in the fixed scenarios (line: mean,
  band: 10th to 90th percentile over the instances and scenarios), with the network for nominal processing times
  (N), uniform random dispatching and greedy MWKR under the same disturbances;
* ``disturbed_distribution.png``: the rollout distribution on the probe states: entropy at temperature 1 and the
  agreement of its most probable job with MWKR and with N.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..training.disturbed import base_name

ROWS = (("lr", "15x15"), ("az", "6x6"), ("az", "10x10"), ("az", "15x15"))
COLOURS = {"FT": "#1aa874", "SC": "#7b5ea7"}
U_COL, MWKR_COL, N_COL = "#2c78d6", "#ea6a35", "#555555"


def histories(roots: dict) -> pd.DataFrame:
    """Every iteration (and the start, iteration -1) of every network for disturbances under ``roots``
    ({variant: models directory})."""
    rows = []
    for variant, models in roots.items():
        models = Path(models)
        if not models.is_dir():
            continue
        for d in sorted(models.iterdir()):
            hist, summ = d / "history.json", d / "summary.json"
            if not (hist.exists() and summ.exists()):
                continue
            with open(summ, encoding="utf-8") as fh:
                s = json.load(fh)
            if not s.get("sigma") or s.get("validation") != "disturbed":
                continue
            with open(hist, encoding="utf-8") as fh:
                h = json.load(fh)
            family, size = base_name(d.name).split("_")
            common = dict(network=d.name, variant=variant, family=family.upper(), size=size, sigma=float(s["sigma"]),
                          uniform=s.get("baseline", {}).get("uniform_realised"),
                          mwkr=s.get("baseline", {}).get("mwkr_realised"),
                          best_iteration=s.get("best_generation"), stopped_by=s.get("stopped_by"))
            if s.get("initial"):
                ini = s["initial"]
                rows.append(dict(common, iteration=-1, greedy=ini.get("greedy"), sampled=ini.get("sampled"),
                                 score=ini.get("score"), **{k: ini.get(k) for k in _EXTRA}))
            for r in h:
                rows.append(dict(common, iteration=r["generation"], greedy=r["greedy_ratio"], sampled=r["sampled_ratio"],
                                 score=r["validation_ratio"], policy_loss=r.get("policy_loss"),
                                 label=r.get("label_ratio"), **{k: r.get(k) for k in _EXTRA}))
    return pd.DataFrame(rows)


_EXTRA = ("greedy_nominal", "sampled_nominal", "greedy_std", "greedy_p10", "greedy_p50", "greedy_p90",
          "probe_entropy", "agree_mwkr", "agree_reference")


def _style(ax):
    ax.grid(axis="y", color="#e3e3e3", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
    ax.tick_params(colors="#333333", length=2.5, labelsize=7.5)


def plot(frame: pd.DataFrame, robustness: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = sorted(frame["sigma"].unique()) if not frame.empty else []
    paths = []
    if not sigmas:
        return paths
    rows = [r for r in ROWS if ((frame["family"] == r[0].upper()) & (frame["size"] == r[1])).any()]
    # ---------------------------------------------------------------- realised makespan over the iterations
    fig, axes = plt.subplots(len(rows), len(sigmas), figsize=(6.3, 2.0 * len(rows) + 0.5), dpi=200, squeeze=False)
    for i, (fam, size) in enumerate(rows):
        for j, sigma in enumerate(sigmas):
            ax = axes[i, j]
            part = frame[(frame["family"] == fam.upper()) & (frame["size"] == size) & np.isclose(frame["sigma"], sigma)]
            for variant, sub in part.groupby("variant"):
                sub = sub.sort_values("iteration")
                col = COLOURS.get(variant, "#333333")
                if sub["greedy_p10"].notna().any():
                    ax.fill_between(sub["iteration"], sub["greedy_p10"], sub["greedy_p90"], color=col, alpha=0.15, lw=0)
                ax.plot(sub["iteration"], sub["greedy"], color=col, lw=1.4)
                best = sub[sub["iteration"] == sub["best_iteration"].iloc[0]]
                if len(best):
                    ax.plot(best["iteration"], best["greedy"], marker="*", ms=7, mfc=col, mec="#222222", mew=0.5)
            ref = part.iloc[0] if len(part) else None
            if ref is not None:
                ax.axhline(ref["uniform"], color=U_COL, lw=1.0, ls=(0, (5, 3)))
                ax.axhline(ref["mwkr"], color=MWKR_COL, lw=1.1, ls=(0, (1, 1.6)))
            if robustness is not None and not robustness.empty:
                n = robustness[(robustness["family"] == fam.upper()) & (robustness["size"] == size)
                               & np.isclose(robustness["sigma"], sigma) & (robustness["variant"] == "N")]
                if len(n):
                    ax.axhline(float(n["greedy_realised"].iloc[0]), color=N_COL, lw=1.0, ls=(0, (6, 2, 1, 2)))
            ax.set_title(f"{fam.upper()} {size.replace('x', ' × ')}, σ = {sigma:g}", fontsize=8.5, loc="left", pad=3)
            _style(ax)
            if j == 0:
                ax.set_ylabel("Realised C$_{max}$ / LB$_0$", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("Iteration (−1: start)", fontsize=8)
    handles = [Line2D([], [], color=COLOURS["FT"], lw=1.4, label="fine-tuned (greedy)"),
               Line2D([], [], color=COLOURS["SC"], lw=1.4, label="from scratch (greedy)"),
               Patch(color="#999999", alpha=0.3, label="10–90 %"),
               Line2D([], [], color=N_COL, lw=1.0, ls=(0, (6, 2, 1, 2)), label="nominal network"),
               Line2D([], [], color=U_COL, lw=1.0, ls=(0, (5, 3)), label="U"),
               Line2D([], [], color=MWKR_COL, lw=1.1, ls=(0, (1, 1.6)), label="MWKR")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out_dir / "disturbed_over_time.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    paths.append(path)
    # ------------------------------------------------------------------- the distribution over the iterations
    metrics = (("probe_entropy", "Entropy at τ = 1"), ("agree_mwkr", "Same job as MWKR"),
               ("agree_reference", "Same job as the nominal network"))
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.3, 1.8 * len(rows) + 0.5), dpi=200, squeeze=False)
    styles = {s: ("-" if k == 0 else (0, (4, 2))) for k, s in enumerate(sigmas)}
    for i, (fam, size) in enumerate(rows):
        part = frame[(frame["family"] == fam.upper()) & (frame["size"] == size)]
        for k, (col_name, title) in enumerate(metrics):
            ax = axes[i, k]
            for (variant, sigma), sub in part.groupby(["variant", "sigma"]):
                sub = sub.sort_values("iteration")
                ax.plot(sub["iteration"], sub[col_name], color=COLOURS.get(variant, "#333"), lw=1.2,
                        ls=styles[sigma])
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{fam.upper()} {size.replace('x', ' × ')}: {title}" if k == 0 else title, fontsize=8,
                         loc="left", pad=3)
            _style(ax)
            if i == len(rows) - 1:
                ax.set_xlabel("Iteration (−1: start)", fontsize=8)
    handles = [Line2D([], [], color=COLOURS["FT"], lw=1.2, label="fine-tuned"),
               Line2D([], [], color=COLOURS["SC"], lw=1.2, label="from scratch")]
    handles += [Line2D([], [], color="#444444", lw=1.2, ls=styles[s], label=f"σ = {s:g}") for s in sigmas]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out_dir / "disturbed_distribution.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    paths.append(path)
    return paths
