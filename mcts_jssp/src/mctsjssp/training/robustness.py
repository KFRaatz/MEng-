"""Robustness of the rollout networks to disturbed processing times, without search (RQ5, network level).

For every disturbance level and training size the validation instances are dispatched greedily and with four
samples at temperature 1 by the networks of LR and AZ of three kinds, and every schedule is executed in the same
fixed scenarios (:mod:`mctsjssp.training.disturbed`):

* ``N``: the network for nominal processing times (never trained with disturbances),
* ``FT``: the network fine-tuned with disturbances from ``N``,
* ``SC``: the network trained with disturbances from scratch (``paths.artifacts``/../scratch).

Uniform random dispatching (U) and greedy MWKR are executed in the same scenarios.  The two questions of RQ5:

1. Does training with disturbances help?  ``FT`` and ``SC`` against ``N`` (and against each other).
2. Is a network trained without disturbances robust to them?  ``N`` against U and MWKR under disturbances,
   and the increase of its realised makespan over the makespan of the same schedule without disturbances.

Paired Wilcoxon signed-rank tests on the 100 instances, Holm-corrected over the comparisons of one disturbance
level, size, family and measure.  The probe metrics describe the rollout distribution of every network: mean
normalised entropy at temperature 1 and the agreement of its most probable job with MWKR and with ``N``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..analysis.stats import holm, wilcoxon_signed_rank
from ..experiments.artifacts import Artifacts
from ..instances.generator import validation_set
from ..networks.checkpoint import load_checkpoint
from ..networks.inference import make_inference
from . import disturbed as dist
from .self_label import Sampler

FAMILIES = ("lr", "az")
SIZES = ((6, 6), (10, 10), (15, 15))
VARIANTS = ("N", "FT", "SC")
PAIRS = (("FT", "N"), ("SC", "N"), ("FT", "SC"), ("N", "U"), ("N", "MWKR"), ("FT", "U"), ("FT", "MWKR"),
         ("SC", "U"), ("SC", "MWKR"))
MEASURES = ("greedy_realised", "sampled_realised")


def network_names(family: str, size: tuple, sigma: float, scratch: str = "../scratch/models") -> dict:
    s = f"{size[0]}x{size[1]}"
    suffix = f"_s{int(round(sigma * 100)):03d}"
    return {"N": f"{family}_{s}", "FT": f"{family}_{s}{suffix}", "SC": f"{scratch}/{family}_{s}{suffix}"}


def evaluate(cfg: dict, sigmas: Sequence[float], workers: int = 1, samples: int = 4,
             scenarios: int = dist.VALIDATION_SCENARIOS, log: Callable[[str], None] = print) -> dict:
    """Per-instance results, the summary and the tests (DataFrames)."""
    artifacts = Artifacts(cfg, log=lambda *_: None)
    seed = int(cfg.get("seed", 0))
    count = int(cfg.get("training", {}).get("validation_instances", 100))
    rows, probes_rows = [], []
    sampler = Sampler(workers)
    try:
        for sigma in sigmas:
            for size in SIZES:
                validation = list(validation_set(*size)[:count])
                label = f"{size[0]}x{size[1]}"
                base_seed = dist.network_seed(seed, f"lr_{label}")
                vseeds = np.random.SeedSequence([base_seed, 7, 0]).generate_state(len(validation), dtype=np.uint32)
                base = sampler.baselines(validation, np.random.SeedSequence([base_seed, 11, 0]).generate_state(
                    len(validation), dtype=np.uint32), sigma, scenarios)
                for inst, (u_r, m_r, u_n, m_n) in zip(validation, base):
                    rows.append(dict(sigma=sigma, size=label, family="U", variant="-", network="U",
                                     instance=inst.name, greedy_realised=u_r, sampled_realised=u_r, greedy_nominal=u_n,
                                     greedy_std=np.nan))
                    rows.append(dict(sigma=sigma, size=label, family="MWKR", variant="-", network="MWKR",
                                     instance=inst.name, greedy_realised=m_r, sampled_realised=m_r,
                                     greedy_nominal=m_n, greedy_std=np.nan))
                for family in FAMILIES:
                    names = network_names(family, size, sigma)
                    if not artifacts.has_model(names["N"]):
                        continue
                    fam_seed = dist.network_seed(seed, names["N"]) % (2 ** 31)
                    probes = dist.probe_states(validation, fam_seed)
                    mwkr = dist.mwkr_choices(probes)
                    nominal, _ = load_checkpoint(artifacts.model_path(names["N"]))
                    reference = dist.choices(dist.network_logits(make_inference(nominal), probes))
                    for variant in VARIANTS:
                        name = names[variant]
                        if not artifacts.has_model(name):
                            continue
                        model, _ = load_checkpoint(artifacts.model_path(name))
                        key = ("robustness", name, sigma)
                        greedy = sampler.run(model, key, validation, vseeds, 1, 0.0, None, sigma, False, scenarios)
                        sampled = sampler.run(model, key, validation, vseeds, samples, None, 1.0, sigma, False,
                                              scenarios)
                        for inst, g, s in zip(validation, greedy, sampled):
                            rows.append(dict(sigma=sigma, size=label, family=family.upper(), variant=variant,
                                             network=name, instance=inst.name, greedy_realised=g[2],
                                             sampled_realised=s[2], greedy_nominal=g[4]["nominal"],
                                             greedy_std=g[4]["std"]))
                        pm = dist.probe_metrics(dist.network_logits(make_inference(model), probes), mwkr, reference)
                        summ = _summary_json(artifacts.model_path(name).parent / "summary.json")
                        probes_rows.append(dict(sigma=sigma, size=label, family=family.upper(), variant=variant,
                                                network=name, **pm, **summ))
                        log(f"[robustness] sigma {sigma:g} {label} {family.upper()}-{variant}: greedy "
                            f"{np.mean([g[2] for g in greedy]):.4f} sampled {np.mean([s[2] for s in sampled]):.4f}"
                            f" (U {np.mean([b[0] for b in base]):.4f}, MWKR {np.mean([b[1] for b in base]):.4f})")
    finally:
        sampler.close()
    per_instance = pd.DataFrame(rows)
    return {"instances": per_instance, "summary": summary(per_instance, pd.DataFrame(probes_rows)),
            "tests": tests(per_instance)}


def _summary_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {"iterations": s.get("generations"), "best_iteration": s.get("best_generation"),
            "stopped_by": s.get("stopped_by"), "validation": s.get("validation", "nominal")}


def summary(per_instance: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    if per_instance.empty:
        return per_instance
    keys = ["sigma", "size", "family", "variant", "network"]
    out = per_instance.groupby(keys, sort=False)[["greedy_realised", "sampled_realised", "greedy_nominal",
                                                  "greedy_std"]].mean().reset_index()
    out["score"] = (out["greedy_realised"] + out["sampled_realised"]) / 2
    out["increase_pct"] = (out["greedy_realised"] / out["greedy_nominal"] - 1) * 100   # disturbance cost
    ref = out[out["family"].isin(["U", "MWKR"])].set_index(["sigma", "size", "family"])
    u = out.apply(lambda r: ref.loc[(r["sigma"], r["size"], "U"), "greedy_realised"], axis=1)
    u_nom = out.apply(lambda r: ref.loc[(r["sigma"], r["size"], "U"), "greedy_nominal"], axis=1)
    mw = out.apply(lambda r: ref.loc[(r["sigma"], r["size"], "MWKR"), "greedy_realised"], axis=1)
    out["vs_u_pct"] = (u - out["score"]) / u * 100                       # validation score against U, disturbed
    out["greedy_vs_u_nominal_pct"] = (u_nom - out["greedy_nominal"]) / u_nom * 100     # without disturbances
    out["greedy_vs_u_pct"] = (u - out["greedy_realised"]) / u * 100      # with disturbances
    out["greedy_vs_mwkr_pct"] = (mw - out["greedy_realised"]) / mw * 100
    if not probes.empty:
        out = out.merge(probes, on=keys, how="left")
    return out


def tests(per_instance: pd.DataFrame, zero_method: str = "zsplit") -> pd.DataFrame:
    """Paired tests of the variants against each other and against U and MWKR (Holm per family of pairs)."""
    out = []
    if per_instance.empty:
        return pd.DataFrame()
    for (sigma, size), part in per_instance.groupby(["sigma", "size"], sort=False):
        for family in ("LR", "AZ"):
            for measure in MEASURES:
                values = {}
                for variant in VARIANTS:
                    sel = part[(part["family"] == family) & (part["variant"] == variant)]
                    if len(sel):
                        values[variant] = sel.set_index("instance")[measure]
                for ref in ("U", "MWKR"):
                    sel = part[part["family"] == ref]
                    values[ref] = sel.set_index("instance")["greedy_realised"]
                rows = []
                for a, b in PAIRS:
                    if a not in values or b not in values:
                        continue
                    common = values[a].index.intersection(values[b].index)
                    res = wilcoxon_signed_rank(values[a][common].to_numpy(), values[b][common].to_numpy(),
                                               zero_method)
                    rows.append(dict(sigma=sigma, size=size, family=family, measure=measure, first=a, second=b,
                                     n=res.n, mean_first=float(values[a][common].mean()),
                                     mean_second=float(values[b][common].mean()), p_value=res.p_value,
                                     rank_biserial=res.rank_biserial))
                if rows:
                    adjusted = holm([r["p_value"] for r in rows])
                    for r, p in zip(rows, adjusted):
                        r["p_holm"] = float(p)
                    out += rows
    return pd.DataFrame(out)


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """A markdown table without extra packages."""
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else (f"{v:.{digits}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{digits}f}")
        return str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, name in (("instances", "robustness_instances.csv"), ("summary", "robustness.csv"),
                      ("tests", "robustness_tests.csv")):
        path = out_dir / name
        result[key].to_csv(path, index=False)
        paths.append(path)
    md = out_dir / "robustness.md"
    s = result["summary"]
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("Networks under disturbed processing times on the validation instances (mean realised Cmax / LB0 "
                 "in the fixed scenarios; N: trained without disturbances, FT: fine-tuned with disturbances from N, "
                 "SC: trained with disturbances from scratch)\n\n")
        cols = [c for c in ("sigma", "size", "family", "variant", "greedy_realised", "sampled_realised", "score",
                            "greedy_nominal", "increase_pct", "vs_u_pct", "greedy_vs_u_nominal_pct", "greedy_vs_u_pct",
                            "greedy_vs_mwkr_pct", "probe_entropy", "agree_mwkr", "agree_reference", "iterations",
                            "best_iteration") if c in s.columns]
        fh.write(_markdown(s[cols]) + "\n\nTests (Wilcoxon, Holm):\n\n")
        t = result["tests"]
        if not t.empty:
            fh.write(_markdown(t) + "\n")
    paths.append(md)
    return paths
