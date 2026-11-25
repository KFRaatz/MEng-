"""Figures of Chapter 4, drawn from the result tables.

Conventions (one system for all figures): every configuration keeps the same colour and marker in every
figure (U blue circle, MWKR orange square, LR green triangle, AZ yellow diamond); the palette was checked for
colour-vision deficiencies, and because two of its colours are light on a white page every series is also
identified by its marker, a legend and a label at the end of its line.  Lines are thin, grids are faint and
there is never a second y-axis.  Every figure is written as PNG and PDF next to the CSV table it is drawn from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import SIZE_CLASSES, budget_order, instance_means  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
STYLE = {                     # configuration -> (colour, marker); fixed, never reassigned by rank
    "U": ("#2a78d6", "o"),
    "MWKR": ("#eb6834", "s"),
    "LR": ("#1baf7a", "^"),
    "AZ": ("#eda100", "D"),
    "FastLR": ("#e87ba4", "v"),       # fast rollout networks: next slots of the same validated palette
    "FastAZ": ("#008300", "P"),
    "FIFO": ("#6250d6", "h"),         # dispatching rules of the extension of RQ1 and RQ2: slots 7 and 8,
    "MOR": ("#e34948", "X"),          #   SPT in the neutral ink (the palette has eight slots)
    "SPT": ("#52514e", "d"),
}
FALLBACK = ("#52514e", "o")


def _style(config: str) -> tuple:
    return STYLE.get(config, FALLBACK)


def _axes(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")


def _line(ax, x, y, config: str, label: str) -> None:
    colour, marker = _style(config)
    ax.plot(x, y, color=colour, linewidth=2, marker=marker, markersize=7.5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, solid_capstyle="round", solid_joinstyle="round")


def _point(ax, x, y, config: str, label: str) -> None:
    colour, marker = _style(config)
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=8, color=colour, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label)


def _end_labels(ax, min_gap: float = 13.0) -> None:
    """Label every series at its last point, unless labels would collide (then the legend identifies them)."""
    ax.figure.canvas.draw()
    ends = []
    for line in ax.get_lines():
        label = line.get_label()
        xs, ys = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if label.startswith("_") or not ok.any():
            continue
        x, y = xs[ok][-1], ys[ok][-1]
        ends.append((label, x, y, ax.transData.transform((x, y))))
    for label, x, y, pixel in ends:
        crowded = any(other is not pixel and abs(other[1] - pixel[1]) < min_gap and abs(other[0] - pixel[0]) < 60
                      for _, _, _, other in ends)
        if not crowded:
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", color=TEXT, fontsize=8.5,
                        va="center")


def _legend(fig, axes) -> None:
    """One legend for the figure, below the plots, so that it never covers data."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if len(handles) >= 2:
        legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False,
                            fontsize=9, ncol=min(len(handles), 4))
        for text in legend.get_texts():
            text.set_color(TEXT)


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        paths.append(path)
    plt.close(fig)
    return paths


def _series(frame: pd.DataFrame) -> list[tuple]:
    """(label, config) of every configuration, in order of appearance."""
    return list(dict.fromkeys(zip(frame["label"], frame["config"])))


# ------------------------------------------------------------------------------------- anytime curves
LEVEL_DASH = {0.0: "-", 0.25: (0, (6, 2)), 0.5: (0, (3, 2)), 0.75: (0, (1, 2))}
GROUP_TITLE = {"all": "All instances", "small": "Small instances", "medium": "Medium instances",
               "large": "Large instances"}


def _label_style(label: str, config: str) -> tuple:
    """Colour and marker of a series: by its own name if it has one (FastLR, FastAZ), else by configuration."""
    base = label.split("@")[0].split("[")[0]
    return STYLE.get(base, _style(config))


def plot_anytime(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean gap to the BKS of the best schedule found so far against wall time: one figure per budget,
    one panel per group of instances.  Line style marks the entropy level (solid 0, dashed 0.5, ...)."""
    paths: list[Path] = []
    if table.empty:
        return paths
    for budget in budget_order(table["budget"].unique()):
        tb = table[table["budget"] == budget]
        groups = [g for g in ("all", *SIZE_CLASSES) if g in set(tb["group"])]
        if len(groups) == 2:                         # one size class only: it equals "all"
            groups = ["all"]
        cols = min(len(groups), 2)
        rows = int(np.ceil(len(groups) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.0 * rows), squeeze=False)
        for ax, group in zip(axes.ravel(), groups):
            tg = tb[tb["group"] == group]
            for (label, config), sub in tg.groupby(["label", "config"], sort=False):
                sub = sub[sub["runs_defined"] == sub["runs"]]
                if sub.empty:
                    continue
                colour, marker = _label_style(label, config)
                level = sub["level"].iloc[0]
                dash = "-" if config == "U" or level is None or pd.isna(level) else LEVEL_DASH.get(float(level), "-")
                ax.plot(sub["time"], sub["mean_gap"], color=colour, linewidth=2, linestyle=dash,
                        marker=marker, markevery=max(1, len(sub) // 6), markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=label, drawstyle="steps-post", solid_capstyle="round")
            ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), label="_bks")
            ax.annotate("BKS (gap 0%)", (0.01, 0.0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", color=TEXT_SECONDARY, fontsize=8)
            _axes(ax, "Wall time since the start of the run (s)", "Mean gap to the BKS (%)",
                  GROUP_TITLE.get(group, group))
            ax.set_xscale("log")
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]) - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        for ax in axes.ravel()[len(groups):]:
            ax.set_visible(False)
        fig.suptitle(f"Best makespan found so far against time, budget {budget}", color=TEXT, fontsize=12,
                     x=0.01, ha="left")
        fig.tight_layout()
        _legend(fig, axes.ravel()[:len(groups)])
        paths += _save(fig, out_dir, f"anytime_{budget}")
    return paths


# ------------------------------------------------------------------------------- fixed states (against U)
FIXED_MEASURES = [("kl_u", "KL divergence from U (nats)", "Divergence from U"),
                  ("tv_u", "Total variation distance from U", "Probability moved away from U"),
                  ("agree_u_search", "Same action as the U search (%)", "Search agreement with U"),
                  ("visit_overlap_u", "Overlap of root visits with U (%)", "Visit overlap with U")]


def fixed_state_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error of every measure against U, per distribution and entropy level."""
    rows = []
    for (conf, level), g in table.groupby(["configuration", "level"], sort=False):
        row = {"configuration": conf, "level": level, "states": len(g)}
        for col, _, _ in FIXED_MEASURES + [("norm_entropy", "", ""), ("kl_mwkr", "", ""),
                                           ("agree_mwkr", "", "")]:
            if col in g.columns:
                v = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{col}_mean"] = float(v.mean()) if len(v) else np.nan
                row[f"{col}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
                row[f"{col}_n"] = int(len(v))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fixed_states(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """How far each distribution is from U on the fixed states, against the entropy level: divergence and
    total variation of the distributions, and agreement of the search with a search that uses U."""
    measures = [m for m in FIXED_MEASURES if f"{m[0]}_mean" in summary.columns
                and summary[f"{m[0]}_n"].fillna(0).sum() > 0]
    if summary.empty or not measures:
        return []
    cols = 2
    rows = int(np.ceil(len(measures) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows), squeeze=False)
    baseline = summary[summary["configuration"] == "U"]
    for ax, (col, ylabel, title) in zip(axes.ravel(), measures):
        pct = col in ("agree_u_search", "visit_overlap_u")
        scale = 100.0 if pct else 1.0
        for conf, g in summary[summary["configuration"] != "U"].groupby("configuration", sort=False):
            g = g.dropna(subset=[f"{col}_mean"]).sort_values("level")
            if g.empty:
                continue
            colour, marker = _style(conf)
            ax.errorbar(g["level"], g[f"{col}_mean"] * scale, yerr=g[f"{col}_se"].fillna(0) * scale,
                        color=colour, marker=marker, markersize=7, linewidth=2, capsize=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=conf)
        ref = baseline[f"{col}_mean"].dropna()
        if len(ref):
            ref_value = float(ref.iloc[0]) * scale
            what = "U against U (baseline)" if pct else "U"
            ax.axhline(ref_value, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), label="_u")
            ax.annotate(what, (0.01, ref_value), xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", color=TEXT_SECONDARY, fontsize=8)
        _axes(ax, "Entropy level (normalised)", ylabel, title)
        ax.set_xticks(sorted(summary["level"][summary["configuration"] != "U"].unique()))
    for ax in axes.ravel()[len(measures):]:
        ax.set_visible(False)
    fig.suptitle("Rollout distributions compared with U on the fixed states", color=TEXT, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    _legend(fig, axes.ravel()[:len(measures)])
    return _save(fig, out_dir, "fixed_states_vs_u")


# ---------------------------------------------------------------------------- states visited by rollouts
EXPERIMENT_DASH = ["-", (0, (5, 2)), (0, (1, 2)), (0, (6, 2, 1, 2))]


def _label_level(label: str) -> float:
    """Entropy level written in a label such as ``LR@H0.5`` (0 when there is none)."""
    try:
        return float(label.split("@H")[1].split("[")[0]) if "@H" in label else 0.0
    except ValueError:
        return 0.0


def _level_shade(colour: str, level: float) -> str:
    """The configuration's colour, lighter for higher entropy levels (0 = full colour, 0.75 = light)."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    mixed = rgb + (1.0 - rgb) * min(max(level, 0.0), 1.0) * 0.75
    return matplotlib.colors.to_hex(mixed)


def plot_visited(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """States visited by the rollouts against the depth of the schedule: how diverse they are, how good they
    are and how much they overlap with those of U.  One figure per budget; line style = experiment."""
    paths: list[Path] = []
    if summary.empty:
        return paths
    experiments = list(dict.fromkeys(summary["experiment"]))
    panels = [("distinct_share", "Distinct states / states visited", "Diversity of the rollout states"),
              ("partial_ratio", "Partial makespan / LB0", "Quality of the rollout states"),
              ("overlap_u", "Jaccard overlap with U (%)", "Overlap with the states of U")]
    for budget in budget_order(summary["budget"].unique()):
        sb = summary[summary["budget"] == budget]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), squeeze=False)
        for ax, (col, ylabel, title) in zip(axes.ravel(), panels):
            scale = 100.0 if col == "overlap_u" else 1.0
            for (experiment, label, config), g in sb.groupby(["experiment", "label", "config"], sort=False):
                if col == "overlap_u" and label == "U":
                    continue
                g = g.dropna(subset=[col]).sort_values("depth")
                if g.empty:
                    continue
                colour, marker = _label_style(label, config)
                colour = _level_shade(colour, _label_level(label))
                name = label if len(experiments) == 1 else f"{experiment}: {label}"
                ax.plot(g["depth"] * 100, g[col] * scale, color=colour, marker=marker, markersize=5,
                        linewidth=2, markevery=max(1, len(g) // 5), markeredgecolor=SURFACE,
                        linestyle=EXPERIMENT_DASH[experiments.index(experiment) % len(EXPERIMENT_DASH)],
                        label=name)
            _axes(ax, "Operations scheduled (%)", ylabel, title)
        fig.suptitle(f"States visited by the rollouts, budget {budget}", color=TEXT, fontsize=12, x=0.01,
                     ha="left")
        fig.tight_layout()
        _legend(fig, axes.ravel())
        paths += _save(fig, out_dir, f"visited_states_{budget}")
    return paths


# ------------------------------------------------------------------- experiments compared by instance
def plot_compare_instances(wides: dict, reference_of, out_dir: Path) -> list[Path]:
    """Heatmap per budget: instances (rows, smallest first) by configuration (columns); each cell is the
    configuration's mean gap minus that of the reference (U) on the instance, in percentage points.  Blue =
    better than U, orange = worse, grey = equal; the reference's own gap is printed beside each instance."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("vs_u", ["#2a78d6", "#f2f1ee", "#eb6834"])
    paths: list[Path] = []
    for tag, w in wides.items():
        ref = reference_of(w.columns)
        if ref is None or w.shape[1] < 2:
            continue
        others = [c for c in w.columns if c != ref]
        diff = w[others].sub(w[ref], axis=0)
        if diff.dropna(how="all").empty:
            continue
        lim = float(np.nanpercentile(np.abs(diff.to_numpy(dtype=float)), 95)) or 1.0
        rows, cols = diff.shape
        fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * cols + 3.2), max(3.0, 0.28 * rows + 1.8)))
        im = ax.imshow(diff.to_numpy(dtype=float), aspect="auto", cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), interpolation="nearest")
        ax.set_xticks(range(cols))
        ax.set_xticklabels(others, rotation=35, ha="right", fontsize=8, color=TEXT_SECONDARY)
        ax.set_yticks(range(rows))
        ax.set_yticklabels([f"{i}  ({ref.split(':')[-1]} {g:.1f}%)" if pd.notna(g) else str(i)
                            for i, g in zip(diff.index, w[ref])], fontsize=7, color=TEXT_SECONDARY)
        if rows * cols <= 400:
            for r in range(rows):
                for c in range(cols):
                    v = diff.iat[r, c]
                    if pd.notna(v):
                        ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                                color="white" if abs(v) > 0.6 * lim else TEXT)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)
        ax.tick_params(which="both", length=0)
        bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label(f"Mean gap minus that of {ref} (percentage points)", color=TEXT_SECONDARY, fontsize=8)
        bar.outline.set_visible(False)
        ax.set_title(f"Gap to the BKS against {ref} per instance, budget {tag} (blue = better than {ref})",
                     loc="left", color=TEXT, fontsize=10)
        fig.tight_layout()
        paths += _save(fig, out_dir, f"compare_instances_{tag}")
    return paths


# --------------------------------------------------------------------------------------- training curves
def plot_training(histories: dict, out_dir: Path) -> list[Path]:
    """One row per network: validation Cmax/LB0 against uniform random dispatching, the losses, and the
    normalised entropy of the visit targets and of the network's policy (1 = uniform).  Networks trained by
    self-labelling also show their greedy and sampled validation ratios and the best of the training samples."""
    if not histories:
        return []
    names = list(histories)
    fig, axes = plt.subplots(len(names), 3, figsize=(13.5, 2.5 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        h = histories[name]
        g = h["generation"]
        sl = "greedy_ratio" in h                       # self-labelling
        unit = "Iteration" if sl else "Generation"
        colour = STYLE["LR"][0] if name.startswith(("lr", "flr")) else STYLE["AZ"][0]
        ax = row[0]
        ax.plot(g, h["validation_ratio"], color=colour, marker="o", markersize=3.5, linewidth=1.8,
                label="validation (mean of greedy and sampled)" if sl else "validation (greedy)")
        if sl:
            ax.plot(g, h["greedy_ratio"], color=colour, linewidth=1.2, linestyle=(0, (1, 1.5)), label="greedy")
            ax.plot(g, h["sampled_ratio"], color=colour, linewidth=1.2, linestyle=(0, (5, 2)), label="sampled")
            ax.plot(g, h["label_ratio"], color=STYLE["MWKR"][0], linewidth=1.2, label="labels (best sample)")
        best = h["validation_ratio"].idxmin()
        ax.plot(g.loc[best], h["validation_ratio"].loc[best], marker="*", markersize=11, color=TEXT,
                linestyle="none", label="best")
        if "uniform_ratio" in h:
            ax.axhline(float(h["uniform_ratio"].iloc[-1]), color=TEXT_SECONDARY, linewidth=1,
                       linestyle=(0, (4, 3)), label="uniform random")
        _axes(ax, unit, "Cmax / LB0", name)
        ax = row[1]
        ax.plot(g, h["policy_loss"], color=STYLE["U"][0], linewidth=1.8, label="policy loss")
        if h["value_loss"].notna().any():
            ax.plot(g, h["value_loss"], color=STYLE["MWKR"][0], linewidth=1.8, label="value loss")
        _axes(ax, unit, "Loss", "Losses")
        ax = row[2]
        if "target_entropy" in h:
            ax.plot(g, h["target_entropy"], color=STYLE["U"][0], linewidth=1.8, label="visit targets")
        if "policy_entropy" in h:
            ax.plot(g, h["policy_entropy"], color=STYLE["MWKR"][0], linewidth=1.8,
                    label="network policy (temperature 1)" if sl else "network policy")
        ax.set_ylim(0, 1.05)
        _axes(ax, unit, "Normalised entropy", "Entropy (1 = uniform)")
        for a in row:
            a.legend(frameon=False, fontsize=7, loc="best")
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, out_dir, "training_curves")


# ------------------------------------------------------------------------------------------ convergence
def plot_convergence(conv: pd.DataFrame, frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean gap against the iterations per decision and against the computation time per decision."""
    paths: list[Path] = []
    series = _series(frame)
    if len({c for _, c in series}) != len(series):
        return paths                                   # several variants of one configuration: see the tables
    iters = conv[conv["budget_kind"] == "iterations"]
    if iters["budget"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for label, config in series:
            sub = iters[iters["label"] == label].sort_values("iterations_per_decision")
            _line(ax, sub["iterations_per_decision"], sub["gap"], config, label)
        _axes(ax, "Iterations per decision", "Mean optimality gap (%)", "Gap against the number of iterations")
        ax.set_xscale("log", base=2)
        ticks = sorted(iters["iterations_per_decision"].round().unique())
        ax.set_xticks(ticks, [f"{int(t)}" for t in ticks])
        ax.margins(x=0.18)
        _end_labels(ax)
        _legend(fig, ax)
        paths += _save(fig, out_dir, "convergence_iterations")
    if conv["budget"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        timed = conv[conv["budget_kind"] == "time"]
        for label, config in series:
            sub = iters[iters["label"] == label].sort_values("time_per_decision")
            _line(ax, sub["time_per_decision"] * 1e3, sub["gap"], config, label)
        _end_labels(ax)
        for label, config in series:                   # runs with a time budget: open markers, not connected
            colour, marker = _style(config)
            sub = timed[timed["label"] == label]
            ax.plot(sub["time_per_decision"] * 1e3, sub["gap"], linestyle="none", marker=marker, markersize=8,
                    markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=2, label="_timed")
            for _, row in sub.iterrows():
                ax.annotate(row["budget"], (row["time_per_decision"] * 1e3, row["gap"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=7.5)
        subtitle = " (open markers: time budgets)" if len(timed) else ""
        _axes(ax, "Computation time per decision (ms)", "Mean optimality gap (%)",
              "Gap against the computation time per decision" + subtitle)
        ax.set_xscale("log")
        ax.margins(x=0.18)
        _legend(fig, ax)
        paths += _save(fig, out_dir, "convergence_time")
    return paths


def plot_size_classes(frame: pd.DataFrame, out_dir: Path, budget: Optional[str] = None) -> list[Path]:
    """Mean gap of every configuration per size class at one budget (the largest by default)."""
    budget = budget or budget_order(frame["budget"])[-1]
    means = instance_means(frame[frame["budget"] == budget], "gap", ("label", "budget"))
    series = _series(frame)
    classes = [c for c in SIZE_CLASSES if c in set(means["size_class"])]
    if not classes or len(series) > 6:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    width = 0.8 / len(series)
    for i, (label, config) in enumerate(series):
        colour, marker = _style(config)
        values = [means[(means["label"] == label) & (means["size_class"] == c)]["gap"].mean() for c in classes]
        xs = np.arange(len(classes)) + (i - (len(series) - 1) / 2) * width
        ax.vlines(xs, 0, values, color=colour, linewidth=2)
        ax.plot(xs, values, linestyle="none", marker=marker, markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        for x, v in zip(xs, values):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 7), textcoords="offset points", ha="center",
                            color=TEXT, fontsize=8)
    _axes(ax, "Size class", "Mean optimality gap (%)", f"Gap per size class at budget {budget}")
    ax.set_xticks(np.arange(len(classes)), [c.capitalize() for c in classes])
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    _legend(fig, ax)
    return _save(fig, out_dir, f"gap_by_size_class_{budget}")


# -------------------------------------------------------------------------------------------------- RQ3
def plot_entropy_levels(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean gap against the entropy level of the rollout distribution, one panel per budget (RQ3)."""
    budgets = budget_order(frame["budget"])
    means = instance_means(frame, "gap", ("label", "budget"))
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.6 * len(budgets), 3.9), squeeze=False, sharey=True)
    for ax, budget in zip(axes[0], budgets):
        sub = means[means["budget"] == budget]
        for config in ("MWKR", "LR", "AZ"):
            part = sub[sub["config"] == config]
            if part.empty:
                continue
            curve = part.groupby(part["level"].astype(float))["gap"].mean().sort_index()
            _line(ax, curve.index, curve.values, config, config)
        uniform = sub[sub["config"] == "U"]["gap"].mean()
        if np.isfinite(uniform):
            _point(ax, 1.0, uniform, "U", "U")
        _axes(ax, "Normalised entropy of the rollout distribution", "Mean optimality gap (%)" if ax is axes[0][0] else "",
              f"Budget {budget}")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.margins(x=0.12)
        _end_labels(ax)
    _legend(fig, axes)
    return _save(fig, out_dir, "gap_by_entropy_level")


def plot_realised_entropy(behaviour: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Realised normalised entropy of the rollouts against its target (first step of the analysis)."""
    if "rollout_entropy" not in behaviour.columns:
        return []
    data = behaviour.dropna(subset=["rollout_entropy"])
    if data.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, zorder=1)
    for config, part in data.groupby("config", sort=False):
        colour, marker = _style(config)
        ax.plot(part["level"].astype(float), part["rollout_entropy"], linestyle="none", marker=marker,
                markersize=8, color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, label=config)
    _axes(ax, "Target normalised entropy", "Realised normalised entropy", "Realised entropy against its target")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _legend(fig, ax)
    return _save(fig, out_dir, "entropy_realised_vs_target")


# -------------------------------------------------------------------------------------------------- RQ4
def plot_size_ratio(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Gap against the size ratio (operations of the instance / operations of the training size),
    one panel per training size of LR (RQ4)."""
    means = instance_means(frame, "gap", ("label", "budget"))
    budget = budget_order(means["budget"])[-1]
    means = means[means["budget"] == budget]
    lr_labels = list(dict.fromkeys(means.loc[means["config"] == "LR", "label"]))
    if not lr_labels:
        return []
    fig, axes = plt.subplots(1, len(lr_labels), figsize=(4.8 * len(lr_labels), 3.9), squeeze=False, sharey=True)
    for ax, lr in zip(axes[0], lr_labels):
        size = means.loc[means["label"] == lr, "train_size"].dropna().iloc[0]
        tn, tm = (int(v) for v in size.split("x"))
        for config, labels in (("U", means.loc[means["config"] == "U", "label"].unique()),
                               ("MWKR", means.loc[means["config"] == "MWKR", "label"].unique()), ("LR", [lr])):
            part = means[means["label"].isin(list(labels))]
            if part.empty:
                continue
            curve = part.groupby((part["n_ops"] / (tn * tm)).round(2))["gap"].mean().sort_index()
            _line(ax, curve.index, curve.values, config, config)
        _axes(ax, "Size ratio (operations / training operations)", "Mean optimality gap (%)" if ax is axes[0][0] else "",
              f"LR trained on {size}, budget {budget}")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.margins(x=0.15)
        _end_labels(ax)
    _legend(fig, axes)
    return _save(fig, out_dir, "gap_by_size_ratio")


# -------------------------------------------------------------------------------------------------- RQ5
def plot_robustness(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean realised gap with its standard deviation over the scenarios, per configuration and sigma (RQ5)."""
    if "realised_gap" not in frame.columns:
        return []
    per = frame.groupby(["sigma", "label", "config", "instance"], sort=False)[["realised_gap", "realised_std_pct"]] \
        .mean().reset_index()
    table = per.groupby(["sigma", "label", "config"], sort=False)[["realised_gap", "realised_std_pct"]].mean().reset_index()
    sigmas = sorted(table["sigma"].unique())
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.8 * len(sigmas), 3.8), squeeze=False, sharey=True)
    for ax, sigma in zip(axes[0], sigmas):
        sub = table[table["sigma"] == sigma].reset_index(drop=True)
        for i, row in sub.iterrows():
            colour, marker = _style(row["config"])
            ax.errorbar([i], [row["realised_gap"]], yerr=[row["realised_std_pct"]], color=colour, linewidth=2,
                        marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0,
                        label=row["label"])
            ax.annotate(f"{row['realised_gap']:.1f}", (i, row["realised_gap"]), xytext=(9, 0),
                        textcoords="offset points", color=TEXT, fontsize=8, va="center")
        _axes(ax, "", "Mean realised gap to the BKS (%)" if ax is axes[0][0] else "",
              f"sigma = {sigma:g} (bars: standard deviation over the scenarios)")
        names = list(sub["config"]) if sub["config"].is_unique else list(sub["label"])
        ax.set_xticks(range(len(sub)), names)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.2)
    return _save(fig, out_dir, "robustness")


def make_figures(experiment: str, frame: pd.DataFrame, tables: dict, out_dir: Path) -> list[Path]:
    """All figures that apply to an experiment; a figure that fails is skipped with a message."""
    jobs = []
    key = experiment.lower()
    if "anytime" in tables and not tables["anytime"].empty:
        jobs.append(lambda: plot_anytime(tables["anytime"], out_dir))
    if "convergence" in tables:
        jobs.append(lambda: plot_convergence(tables["convergence"], frame, out_dir))
    if not key.startswith(("rq3", "rq4", "rq5")):
        jobs.append(lambda: plot_size_classes(frame, out_dir))
    if key.startswith("rq3"):
        jobs.append(lambda: plot_entropy_levels(frame, out_dir))
        if "behaviour" in tables:
            jobs.append(lambda: plot_realised_entropy(tables["behaviour"], out_dir))
    if key.startswith("rq4"):
        jobs.append(lambda: plot_size_ratio(frame, out_dir))
    if key.startswith("rq5"):
        jobs.append(lambda: plot_robustness(frame, out_dir))
    paths: list[Path] = []
    for job in jobs:
        try:
            paths += job()
        except Exception as exc:  # a figure must never stop the tables from being written
            print(f"  figure skipped: {type(exc).__name__}: {exc}")
    return paths
