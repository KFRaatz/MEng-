"""Comparison of the two training styles of the rollout networks (Section 3.6.1): self-labelling, whose
networks are used in the experiments (``artifacts/models``), against AlphaZero-style self-play (by default
``artifacts/self_play/models``, trained by the pipeline step ``self_play``).

Both are judged in the same way, whatever their own validation used: the best checkpoint of each network
dispatches the same validation instances greedily (entropy 0) and with four samples at temperature 1 per
instance (the same random seeds for both), next to uniform random dispatching, MWKR and FIFO.  The table also
holds how far each training got (iterations or generations, early stopping, minutes, steps of the dynamics
model) and the normalised entropy of the policy at temperature 1 on the states of its greedy schedules, which
shows how sharp the network has become (1 = uniform).  For every network trained in both styles, two-sided
Wilcoxon signed-rank tests on the same validation instances compare the greedy Cmax / LB0 and the mean of the
sampled Cmax / LB0 of the two styles (as in Section 3.6.4); the rank-biserial correlation is positive when
self-labelling gives the shorter schedules.

Outputs in ``results/training``: compare_training.csv, compare_training.md and compare_training.png/.pdf
(validation Cmax / LB0 of the greedy schedules during training, against the minutes of training).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..env.model import State
from ..env.rules import RULES, run_rule
from ..instances.generator import validation_set
from ..networks.checkpoint import load_checkpoint
from ..training.self_label import Sampler, label_records
from ..training.trainer import uniform_ratio
from .stats import wilcoxon_signed_rank

STYLES = ("self-labelling", "self-play")
SAMPLES = 4                      # sampled rollouts per validation instance
ENTROPY_INSTANCES = 20           # instances whose greedy states measure the entropy of the policy


def _entropy(model, records: list) -> float:
    """Mean normalised entropy of the policy at temperature 1 over states with two or more candidates."""
    if not records:
        return float("nan")
    with torch.no_grad():
        if getattr(model, "uses_features", False):
            masks = torch.from_numpy(np.stack([r[1] for r in records]))
            logits = model.forward_features(torch.from_numpy(np.stack([r[0] for r in records])), masks)
        else:
            x = torch.from_numpy(np.stack([r.x for r in records]))
            mp_ = torch.from_numpy(np.stack([r.mpred for r in records]).astype(np.int64))
            nx = torch.from_numpy(np.stack([r.next_op for r in records]).astype(np.int64))
            masks = torch.from_numpy(np.stack([r.mask for r in records]))
            logits, _ = model.forward_batch(x, mp_, nx, masks)
    keep = masks.sum(dim=1) > 1
    logits, masks = logits[keep].masked_fill(~masks[keep], -1e9), masks[keep]
    logp = torch.log_softmax(logits, dim=1)
    h = -(logp.exp() * torch.where(masks, logp, torch.zeros_like(logp))).sum(dim=1)
    return float((h / torch.log(masks.sum(dim=1).double()).float()).mean())


def _training(model_dir: Path) -> dict:
    """What the training of one network recorded: method, length, stopping, time and steps."""
    summary = json.loads((model_dir / "summary.json").read_text())
    history = json.loads((model_dir / "history.json").read_text()) if (model_dir / "history.json").exists() else []
    initial = summary.get("initial") or {}
    start = Path(initial["from"]).parent.name if initial.get("from") else \
        ("copy of " + str((summary.get("warm_start") or {}).get("rule")) if summary.get("warm_start") else "random")
    last = history[-1] if history else {}
    return {
        "method": summary.get("method", "alphazero"),
        "start": start,
        "iterations": summary.get("generations"),
        "best_iteration": summary.get("best_generation"),
        "stopped_by": summary.get("stopped_by"),
        "minutes": last.get("elapsed_minutes"),
        "model_steps": last.get("model_steps", summary.get("model_steps")),
        "history": history,
    }


def compare_training(cfg: dict, models: list[str], roots: dict, workers: int, out_dir: Path,
                     log=print) -> list[dict]:
    """``roots`` maps a style (``STYLES``) to the models folder of its artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = int(cfg["training"].get("validation_instances", 100))
    rows, curves = [], {}
    zero_method = str((cfg.get("analysis") or {}).get("zero_method", "zsplit"))
    sampler = Sampler(workers)
    try:
        for name in models:
            spec = cfg["models"][name]
            n, m = int(spec["n"]), int(spec["m"])
            val = list(validation_set(n, m)[:count])
            seeds = [10007 * (i + 1) for i in range(len(val))]
            refs = {"uniform": uniform_ratio(val, seed=0, repeats=10)}
            for rule in ("MWKR", "FIFO"):
                refs[rule.lower()] = float(np.mean([run_rule(inst, RULES[rule]).makespan / inst.lb0 for inst in val]))
            per_instance = {}
            for style, root in roots.items():
                model_dir = Path(root) / name
                row = {"model": name, "kind": spec.get("kind"), "size": f"{n}x{m}", "style": style, **refs}
                if not (model_dir / "best.pt").exists() or not (model_dir / "summary.json").exists():
                    rows.append({**row, "status": "not trained"})
                    log(f"[compare-training] {name} {style}: not trained ({model_dir})")
                    continue
                info = _training(model_dir)
                curves[(name, style)] = info.pop("history")
                net, _ = load_checkpoint(model_dir / "best.pt")
                net.eval()
                key = ("compare", name, style)
                greedy = sampler.run(net, key, val, seeds, 1, 0.0, None, None, True)
                sampled = sampler.run(net, key, val, seeds, SAMPLES, None, 1.0, None, False)
                fast = bool(getattr(net, "uses_features", False))
                records = [r for inst, g in list(zip(val, greedy))[:ENTROPY_INSTANCES]
                           for r in label_records(inst, g[1], fast)]
                g_ratio = float(np.mean([r[2] for r in greedy]))
                s_ratio = float(np.mean([r[2] for r in sampled]))
                per_instance[style] = ([r[2] for r in greedy], [r[2] for r in sampled])
                row.update(info)
                row.update({"status": "trained", "greedy": g_ratio, "sampled": s_ratio,
                            "greedy_vs_uniform_pct": 100 * (refs["uniform"] - g_ratio) / refs["uniform"],
                            "sampled_vs_uniform_pct": 100 * (refs["uniform"] - s_ratio) / refs["uniform"],
                            "greedy_vs_mwkr_pct": 100 * (refs["mwkr"] - g_ratio) / refs["mwkr"],
                            "entropy": _entropy(net, records)})
                rows.append(row)
                log(f"[compare-training] {name:9s} {style:15s} greedy {g_ratio:.4f}  sampled {s_ratio:.4f}  "
                    f"(uniform {refs['uniform']:.4f}, MWKR {refs['mwkr']:.4f})  entropy {row['entropy']:.3f}  "
                    f"{row['iterations']} it., {row['minutes'] or float('nan'):.1f} min, stopped by {row['stopped_by']}")
            if len(per_instance) == 2:                           # both styles trained: paired tests
                (g1, s1), (g2, s2) = per_instance[STYLES[0]], per_instance[STYLES[1]]
                tg = wilcoxon_signed_rank(g1, g2, zero_method)
                ts = wilcoxon_signed_rank(s1, s2, zero_method)
                for r in rows[-2:]:
                    r.update({"p_greedy": tg.p_value, "rank_biserial_greedy": tg.rank_biserial,
                              "p_sampled": ts.p_value, "rank_biserial_sampled": ts.rank_biserial})
                log(f"[compare-training] {name:9s} self-labelling vs self-play: greedy p = {tg.p_value:.3g} "
                    f"(r = {tg.rank_biserial:+.2f}), sampled p = {ts.p_value:.3g} (r = {ts.rank_biserial:+.2f})")
    finally:
        sampler.close()
    _write(rows, curves, out_dir)
    return rows


COLUMNS = ["model", "kind", "size", "style", "status", "method", "start", "iterations", "best_iteration",
           "stopped_by", "minutes", "model_steps", "greedy", "sampled", "uniform", "mwkr", "fifo",
           "greedy_vs_uniform_pct", "sampled_vs_uniform_pct", "greedy_vs_mwkr_pct", "entropy",
           "p_greedy", "rank_biserial_greedy", "p_sampled", "rank_biserial_sampled"]


DIGITS = {"minutes": 1, "greedy": 4, "sampled": 4, "uniform": 4, "mwkr": 4, "fifo": 4, "entropy": 3,
          "greedy_vs_uniform_pct": 1, "sampled_vs_uniform_pct": 1, "greedy_vs_mwkr_pct": 1,
          "p_greedy": 4, "p_sampled": 4, "rank_biserial_greedy": 2, "rank_biserial_sampled": 2}


def _fmt(v, column: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{DIGITS.get(column, 4)}f}"
    return str(v)


def _write(rows: list[dict], curves: dict, out_dir: Path) -> None:
    import pandas as pd

    table = pd.DataFrame([{c: r.get(c) for c in COLUMNS} for r in rows], columns=COLUMNS)
    table.to_csv(out_dir / "compare_training.csv", index=False)
    shown = ["model", "style", "status", "start", "iterations", "stopped_by", "minutes", "greedy", "sampled",
             "uniform", "mwkr", "greedy_vs_uniform_pct", "entropy", "p_greedy", "p_sampled"]
    lines = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    lines += ["| " + " | ".join(_fmt(r.get(c), c) for c in shown) + " |" for r in rows]
    (out_dir / "compare_training.md").write_text(
        "Training styles on the validation instances (Cmax / LB0; greedy = entropy 0, sampled = temperature 1, "
        f"{SAMPLES} samples per instance; entropy of the policy at temperature 1, 1 = uniform; p: two-sided "
        f"Wilcoxon signed-rank tests of the two styles on the same instances)\n\n"
        + "\n".join(lines) + "\n")
    _figure(rows, curves, out_dir)


def _figure(rows: list[dict], curves: dict, out_dir: Path) -> None:
    from .plots import GRID, STYLE, TEXT, TEXT_SECONDARY, _axes, plt

    names = list(dict.fromkeys(r["model"] for r in rows))
    if not names or not curves:
        return
    cols = min(3, len(names))
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.4 * cols, 3.1 * nrows), squeeze=False)
    colour = {"self-labelling": STYLE["LR"][0], "self-play": STYLE["U"][0]}
    for ax, name in zip(axes.flat, names):
        ref = next((r for r in rows if r["model"] == name), {})
        for style in STYLES:
            h = curves.get((name, style))
            if not h:
                continue
            t = [r.get("elapsed_minutes", np.nan) for r in h]
            y = [r.get("greedy_ratio", r.get("validation_ratio")) for r in h]
            ax.plot(t, y, color=colour[style], linewidth=1.6, marker="o", markersize=2.8,
                    label=f"{style} (greedy)")
            if style == "self-labelling" and "sampled_ratio" in h[0]:
                ax.plot(t, [r["sampled_ratio"] for r in h], color=colour[style], linewidth=1.0,
                        linestyle=(0, (4, 2)), label=f"{style} (sampled)")
        for key, label, dash in (("uniform", "uniform random", (0, (4, 3))), ("mwkr", "MWKR", (0, (1, 1.5)))):
            if ref.get(key):
                ax.axhline(ref[key], color=TEXT_SECONDARY, linewidth=1, linestyle=dash, label=label)
        _axes(ax, "Minutes of training", "Validation Cmax / LB0", name)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=9,
               labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"compare_training.{ext}", dpi=200)
    plt.close(fig)
