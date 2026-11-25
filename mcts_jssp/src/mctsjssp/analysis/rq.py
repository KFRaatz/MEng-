"""Analysis of every research question (Table 3.6 and Section 3.6.4).

Each ``analyse_*`` function receives the result rows of one experiment and returns named tables.  The tests
are two-sided at ``alpha`` after Holm's correction; the column ``holm_family`` of every test table states
which comparisons were corrected together.

========  ==========================================================================================
RQ1       MWKR against U at every budget, over all instances and within each size class
RQ2       Friedman test over U, MWKR and LR at every budget, then LR against U and against MWKR
RQ3       LR and AZ against MWKR and LR against AZ at every entropy level; the levels of each
          configuration against one another; search behaviour (entropy, divergence, variance, bias)
RQ4       LR against U and MWKR on instances of other sizes than its training size, by size ratio
RQ5       mean realised makespan and its variability under disturbances, per level of sigma
RQ6       each network trained by self-labelling against the same network trained by self-play, and
          both against U, as rollout distributions at every budget (variant ``<name>-SP``)
rules     extension of RQ1 and RQ2: MOR, SPT and FIFO as fixed rollout distributions, analysed with U, MWKR
          and LR of RQ2; LR against every fixed distribution and against the best of them on the
          validation instances (chosen before the test runs)
========  ==========================================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .metrics import (SIZE_CLASSES, bks_hits, budget_order, convergence, instance_means, load_records, summarise,
                      time_to_target)
from .stats import compare_pairs, friedman_test, holm

GROUPS = ("all",) + SIZE_CLASSES


# ---------------------------------------------------------------------------------------------- helpers
def labels_of(frame: pd.DataFrame, config: str, level: Optional[float] = None,
              variant: Optional[str] = None) -> list[str]:
    """Labels of a configuration (optionally at one entropy level or of one variant), in order of appearance."""
    sel = frame[frame["config"] == config]
    if level is not None:
        sel = sel[np.isclose(sel["level"].astype(float), level)]
    if variant is not None:
        sel = sel[sel["variant"] == variant]
    return list(dict.fromkeys(sel["label"]))


def _group(means: pd.DataFrame, group: str) -> pd.DataFrame:
    return means if group == "all" else means[means["size_class"] == group]


def paired_tests(frame: pd.DataFrame, pairs: Sequence[tuple], value: str = "gap", groups: Sequence[str] = GROUPS,
                 alpha: float = 0.05, zero_method: str = "zsplit", holm_over: str = "pairs",
                 extra_key: Optional[str] = None) -> pd.DataFrame:
    """Wilcoxon tests of ``pairs`` at every budget, for every group of instances.

    ``holm_over = "pairs"`` corrects the comparisons of one budget together (the post-hoc comparisons of one
    Friedman test); ``"budgets"`` corrects all comparisons of a group together, across the budgets.
    ``extra_key`` (for example ``sigma``) splits the analysis by a further column.
    """
    by = ["label", "budget"] + ([extra_key] if extra_key else [])
    means = instance_means(frame, value, by)
    parts = []
    extras = sorted(means[extra_key].unique()) if extra_key else [None]
    for extra in extras:
        sub_extra = means if extra is None else means[means[extra_key] == extra]
        for group in groups:
            sub = _group(sub_extra, group)
            if sub.empty:
                continue
            for budget in budget_order(sub["budget"]):
                table = sub[sub["budget"] == budget].pivot(index="instance", columns="label", values=value)
                res = compare_pairs(table, pairs, alpha, zero_method, correct=False)
                if res.empty:
                    continue
                res.insert(0, "budget", budget)
                res.insert(0, "group", group)
                if extra_key:
                    res.insert(0, extra_key, extra)
                parts.append(res)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    family_cols = [c for c in ([extra_key] if extra_key else [])] + ["group"] + (["budget"] if holm_over == "pairs" else [])
    out["p_holm"] = np.nan
    for _, idx in out.groupby(family_cols, sort=False).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_value"].to_numpy())
    out["significant"] = out["p_holm"] <= alpha
    out["holm_family"] = "comparisons of one " + " and ".join(family_cols)
    out["measure"] = value
    return out


def friedman_tests(frame: pd.DataFrame, columns: Sequence[str], value: str = "gap",
                   groups: Sequence[str] = GROUPS, alpha: float = 0.05, extra_key: Optional[str] = None,
                   name: str = "") -> pd.DataFrame:
    """Friedman test over ``columns`` (labels) at every budget, for every group of instances."""
    by = ["label", "budget"] + ([extra_key] if extra_key else [])
    means = instance_means(frame, value, by)
    rows = []
    extras = sorted(means[extra_key].unique()) if extra_key else [None]
    for extra in extras:
        sub_extra = means if extra is None else means[means[extra_key] == extra]
        for group in groups:
            sub = _group(sub_extra, group)
            for budget in budget_order(sub["budget"]) if not sub.empty else []:
                table = sub[sub["budget"] == budget].pivot(index="instance", columns="label", values=value)
                cols = [c for c in columns if c in table.columns]
                data = table[cols].dropna()
                if len(cols) < 3 or len(data) < 2:
                    continue
                res = friedman_test(data)
                row = {"comparison": name, "group": group, "budget": budget, "measure": value,
                       "instances": res.n, "configurations": res.k, "chi_square": res.statistic,
                       "p_value": res.p_value, "iman_davenport_F": res.iman_davenport,
                       "p_iman_davenport": res.p_iman_davenport, "rejects": res.p_value <= alpha}
                if extra_key:
                    row[extra_key] = extra
                row.update({f"rank[{c}]": r for c, r in res.average_ranks.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def _common(frame: pd.DataFrame) -> dict:
    conv = convergence(frame)
    return {"summary": summarise(frame, "gap"), "convergence": conv, "bks_reached": bks_hits(frame)}


# -------------------------------------------------------------------------------------------------- RQ1
def analyse_rq1(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """MWKR against U, over all instances and within each size class."""
    out = _common(frame)
    u, mwkr = labels_of(frame, "U"), labels_of(frame, "MWKR")
    pairs = [(m, b) for m in mwkr for b in u]
    out["tests"] = paired_tests(frame, pairs, "gap", GROUPS, alpha, zero_method, holm_over="budgets")
    out["time_to_target"] = time_to_target(out["convergence"], u + mwkr)
    return out


# -------------------------------------------------------------------------------------------------- RQ2
def analyse_rq2(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """LR against U and MWKR, at equal iterations and at equal time."""
    out = _common(frame)
    u, mwkr, lr = labels_of(frame, "U"), labels_of(frame, "MWKR"), labels_of(frame, "LR")
    out["friedman"] = friedman_tests(frame, u + mwkr + lr, "gap", GROUPS, alpha, name="U, MWKR, LR")
    pairs = [(learned, base) for learned in lr for base in u + mwkr]
    out["tests"] = paired_tests(frame, pairs, "gap", GROUPS, alpha, zero_method, holm_over="pairs")
    if not out["tests"].empty and not out["friedman"].empty:
        rejects = out["friedman"].set_index(["group", "budget"])["rejects"]
        out["tests"]["friedman_rejects"] = [bool(rejects.get((g, b), False))
                                            for g, b in zip(out["tests"]["group"], out["tests"]["budget"])]
    out["time_to_target"] = time_to_target(out["convergence"], u + mwkr)
    return out


# -------------------------------------------------------------------------------------------------- RQ3
BEHAVIOUR_COLUMNS = ("rollout_entropy", "kl_mwkr", "agree_mwkr", "mwkr_rank", "return_var", "return_bias",
                     "visit_entropy_norm", "top_visit_share", "value_var", "iterations_per_decision",
                     "time_per_decision", "gap")


def behaviour_table(frame: pd.DataFrame, by: Sequence[str] = ("config", "level", "label", "budget")) -> pd.DataFrame:
    """Search behaviour per configuration: realised entropy against its target, divergence from MWKR,
    agreement, variance and bias of the rollout returns, concentration of the visits at the root."""
    cols = [c for c in BEHAVIOUR_COLUMNS if c in frame.columns]
    frame = frame.assign(**{c: pd.to_numeric(frame[c], errors="coerce") for c in cols})
    per_instance = frame.groupby(["instance", *by], sort=False)[cols].mean().reset_index()
    table = per_instance.groupby(list(by), sort=False)[cols].mean().reset_index()
    if "rollout_entropy" in table.columns:
        table["entropy_error"] = table["rollout_entropy"] - table["level"].astype(float)
    return table


def depth_profile(frame: pd.DataFrame, run_dir: Optional[str | Path], kind: str = "steps", bins: int = 4) -> pd.DataFrame:
    """Recorded quantities by depth in the schedule (quarters of the operations) and by family."""
    if run_dir is None:
        return pd.DataFrame()
    wanted = {"steps": ["depth", "n_candidates", "norm_entropy", "kl_mwkr", "agree_mwkr", "mwkr_rank"],
              "decisions": ["depth", "n_candidates", "iterations", "time_s", "visit_entropy_norm",
                            "top_visit_share", "return_var", "value_var"]}[kind]
    records = load_records(run_dir, frame["run_id"], kind, wanted)
    if records.empty:
        return pd.DataFrame()
    meta = frame[["run_id", "label", "config", "level", "budget", "family", "n_ops"]]
    records = records.merge(meta, on="run_id")
    fraction = records["depth"] / records["n_ops"]
    records["depth_bin"] = np.minimum((fraction * bins).astype(int), bins - 1)
    values = [c for c in wanted if c not in ("depth",)]
    for col in values:
        records[col] = pd.to_numeric(records[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    by_depth = records.groupby(["config", "level", "label", "budget", "depth_bin"], sort=True)[values].mean().reset_index()
    by_depth.insert(0, "by", "depth")
    by_family = records.groupby(["config", "level", "label", "budget", "family"], sort=True)[values].mean().reset_index()
    by_family.insert(0, "by", "family")
    counts = records.groupby(["config", "level", "label", "budget", "depth_bin"], sort=True).size().values
    by_depth["records"] = counts
    by_family["records"] = records.groupby(["config", "level", "label", "budget", "family"], sort=True).size().values
    return pd.concat([by_depth, by_family], ignore_index=True)


def relations(frame: pd.DataFrame) -> pd.DataFrame:
    """Relation of the variance and bias of the rollout returns (and of the gap) to the properties of the
    rollout distribution: Spearman correlations over the instance means, after the mean of every instance
    has been subtracted, so that differences between instances do not drive the correlation."""
    properties = [c for c in ("level", "rollout_entropy", "kl_mwkr", "agree_mwkr") if c in frame.columns]
    outcomes = [c for c in ("return_var", "return_bias", "gap") if c in frame.columns]
    if not properties or not outcomes:
        return pd.DataFrame()
    data = frame.assign(**{c: pd.to_numeric(frame[c], errors="coerce") for c in properties + outcomes})
    per = data.groupby(["instance", "label"], sort=False)[properties + outcomes].mean().reset_index()
    centred = per.copy()
    centred[properties + outcomes] = per[properties + outcomes] - per.groupby("instance")[properties + outcomes].transform("mean")
    rows = []
    for prop in properties:
        for outcome in outcomes:
            pair = centred[[prop, outcome]].dropna()
            if len(pair) < 4 or pair[prop].nunique() < 2 or pair[outcome].nunique() < 2:
                continue
            rho, p = sps.spearmanr(pair[prop], pair[outcome])
            rows.append({"property": prop, "outcome": outcome, "spearman_rho": float(rho), "p_value": float(p),
                         "observations": len(pair)})
    return pd.DataFrame(rows)


def fixed_state_summary(fixed: pd.DataFrame) -> pd.DataFrame:
    """Summary of the comparison on the fixed states: per configuration and level, by depth and by family."""
    if fixed is None or fixed.empty:
        return pd.DataFrame()
    values = [c for c in ("norm_entropy", "top_probability", "kl_mwkr", "agree_mwkr", "kl_other_learned",
                          "agree_other_learned") if c in fixed.columns]
    data = fixed.copy()
    for col in values:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    data["depth_bin"] = np.minimum((data["depth_fraction"] * 3).astype(int), 2)
    parts = []
    for by, keys in (("all", []), ("depth", ["depth_bin"]), ("family", ["family"]), ("size", ["size"])):
        agg = data.groupby(["configuration", "level", *keys], sort=True)[values].mean().reset_index()
        agg["states"] = data.groupby(["configuration", "level", *keys], sort=True).size().values
        agg.insert(0, "by", by)
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def analyse_rq3(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit",
                run_dir: Optional[str | Path] = None, include_uniform: bool = True) -> dict:
    """Distribution form, network family and entropy."""
    out = _common(frame)
    groups = ("all",)
    levels = sorted(frame.loc[frame["config"] != "U", "level"].astype(float).unique())
    # LR and AZ against MWKR, and LR against AZ, at every entropy level: Holm over levels and comparisons
    pairs = []
    for level in levels:
        mwkr, lr, az = (labels_of(frame, c, level) for c in ("MWKR", "LR", "AZ"))
        pairs += [(a, b) for a in lr for b in mwkr] + [(a, b) for a in az for b in mwkr]
        pairs += [(a, b) for a in lr for b in az]
    out["tests_same_entropy"] = paired_tests(frame, pairs, "gap", groups, alpha, zero_method, holm_over="pairs")
    # the entropy levels of each configuration against one another (with U as level 1)
    parts = []
    uniform = labels_of(frame, "U") if include_uniform else []
    for config in ("MWKR", "LR", "AZ"):
        cols = [lab for level in levels for lab in labels_of(frame, config, level)] + uniform
        if len(cols) >= 3:
            parts.append(friedman_tests(frame, cols, "gap", groups, alpha, name=f"entropy levels of {config}"))
    out["friedman_levels"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out["behaviour"] = behaviour_table(frame)
    out["relations"] = relations(frame[frame["budget_kind"] == "iterations"])
    out["steps_profile"] = depth_profile(frame, run_dir, "steps")
    out["decisions_profile"] = depth_profile(frame, run_dir, "decisions")
    if run_dir is not None and (Path(run_dir) / "fixed_states.csv").exists():
        out["fixed_states"] = fixed_state_summary(pd.read_csv(Path(run_dir) / "fixed_states.csv"))
    return out


# -------------------------------------------------------------------------------------------------- RQ4
def analyse_rq4(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """Generalisation: LR trained on one size against U and MWKR on instances of other sizes."""
    out = _common(frame)
    means = instance_means(frame, "gap", ("label", "budget"))
    u, mwkr = labels_of(frame, "U"), labels_of(frame, "MWKR")
    ratio_rows, test_parts, trend_rows = [], [], []
    for lr in labels_of(frame, "LR"):
        size = frame.loc[frame["label"] == lr, "train_size"].iloc[0]
        tn, tm = (int(v) for v in size.split("x"))
        for budget in budget_order(means["budget"]):
            sub = means[means["budget"] == budget]
            table = sub.pivot(index="instance", columns="label", values="gap")
            info = sub.drop_duplicates("instance").set_index("instance")
            if lr not in table.columns:
                continue
            table = table.assign(ratio=info["n_ops"] / (tn * tm),
                                 same_size=(info["n"] == tn) & (info["m"] == tm))
            table = table.dropna(subset=[lr])
            seen = table[table["same_size"]]
            unseen = table[~table["same_size"]]
            for name, part in (("training size", seen), ("other sizes", unseen)):
                if not part.empty:
                    row = {"lr": lr, "train_size": size, "budget": budget, "instances_of": name,
                           "instances": len(part), "gap_lr": part[lr].mean()}
                    row.update({f"gap[{b}]": part[b].mean() for b in u + mwkr if b in part.columns})
                    ratio_rows.append({**row, "ratio": "all"})
            for ratio, part in unseen.groupby(unseen["ratio"].round(2)):
                row = {"lr": lr, "train_size": size, "budget": budget, "instances_of": "other sizes",
                       "ratio": float(ratio), "instances": len(part), "gap_lr": part[lr].mean()}
                row.update({f"gap[{b}]": part[b].mean() for b in u + mwkr if b in part.columns})
                ratio_rows.append(row)
            res = compare_pairs(unseen, [(lr, b) for b in u + mwkr], alpha, zero_method, correct=False)
            if not res.empty:
                res.insert(0, "budget", budget)
                res.insert(0, "train_size", size)
                test_parts.append(res)
            for base in u + mwkr:
                if base in unseen.columns and len(unseen) >= 4:
                    rho, p = sps.spearmanr(unseen["ratio"], unseen[lr] - unseen[base])
                    trend_rows.append({"lr": lr, "train_size": size, "budget": budget, "against": base,
                                       "spearman_rho_ratio_vs_gap_difference": float(rho), "p_value": float(p),
                                       "instances": len(unseen)})
    out["by_size_ratio"] = pd.DataFrame(ratio_rows)
    if test_parts:
        tests = pd.concat(test_parts, ignore_index=True)
        tests["p_holm"] = np.nan
        for _, idx in tests.groupby("budget", sort=False).groups.items():
            tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p_value"].to_numpy())
        tests["significant"] = tests["p_holm"] <= alpha
        tests["holm_family"] = "all comparisons of one budget (training sizes x baselines)"
        out["tests_other_sizes"] = tests
    else:
        out["tests_other_sizes"] = pd.DataFrame()
    out["trend"] = pd.DataFrame(trend_rows)
    return out


# -------------------------------------------------------------------------------------------------- RQ5
RQ5_MEASURES = ("realised_gap", "realised_std_pct", "realised_increase", "gap")


def analyse_rq5(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """Robustness: mean realised makespan relative to the BKS, its standard deviation over the scenarios and its
    increase over the undisturbed makespan of the same schedule, per level of sigma.  Two questions:

    1. ``tests_training``: does training with disturbances help?  For LR and AZ the network fine-tuned with
       disturbances (``-FT``) and the one trained with them from scratch (``-SC``) against the network trained
       without them (``LR``, ``AZ``) and against each other.
    2. ``tests_robustness``: is a network robust to disturbances?  Every network, above all the one trained without
       them, against U and MWKR under the same disturbances.

    ``tests`` holds both (column ``question``); Holm corrects the comparisons of each question, budget and sigma.
    The measures of every comparison are realised_gap, realised_std_pct and realised_increase."""
    out: dict = {}
    measures = [c for c in RQ5_MEASURES if c in frame.columns]
    per_instance = frame.groupby(["sigma", "label", "budget", "instance"], sort=False)[measures].mean().reset_index()
    summary = per_instance.groupby(["sigma", "label", "budget"], sort=False)[measures].agg(["mean", "median"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary["instances"] = per_instance.groupby(["sigma", "label", "budget"], sort=False)["instance"].count()
    out["summary"] = summary.reset_index()
    u, mwkr = labels_of(frame, "U"), labels_of(frame, "MWKR")
    lr = {v: labels_of(frame, "LR", variant=v) for v in ("LR", "LR-FT", "LR-SC")}
    az = {v: labels_of(frame, "AZ", variant=v) for v in ("AZ", "AZ-FT", "AZ-SC")}
    if not any(lr.values()):                                   # older results: the LR of the file, any variant
        lr = {"LR": labels_of(frame, "LR"), "LR-FT": [], "LR-SC": []}
    if not any(az.values()):
        az = {"AZ": labels_of(frame, "AZ"), "AZ-FT": [], "AZ-SC": []}
    training, robustness = [], []
    for fam in (lr, az):
        n, ft, sc = (fam[k] for k in fam)
        training += [(a, b) for a in ft + sc for b in n] + [(a, b) for a in ft for b in sc]
        robustness += [(a, b) for a in n + ft + sc for b in u + mwkr]
    friedman_parts, test_parts = [], []
    learned = [x for v in lr.values() for x in v]
    for measure in ("realised_gap", "realised_std_pct", "realised_increase"):
        if measure not in frame.columns:
            continue
        friedman_parts.append(friedman_tests(frame, u + mwkr + learned, measure, ("all",), alpha, extra_key="sigma",
                                             name="U, MWKR, " + ", ".join(k for k, v in lr.items() if v)))
        for question, pairs in (("training with disturbances", training), ("robustness", robustness)):
            if pairs:
                t = paired_tests(frame, pairs, measure, ("all",), alpha, zero_method, holm_over="pairs",
                                 extra_key="sigma")
                t.insert(0, "question", question)
                test_parts.append(t)
    out["friedman"] = pd.concat(friedman_parts, ignore_index=True) if friedman_parts else pd.DataFrame()
    tests = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    out["tests"] = tests
    if not tests.empty:
        out["tests_training"] = tests[tests["question"] == "training with disturbances"].reset_index(drop=True)
        out["tests_robustness"] = tests[tests["question"] == "robustness"].reset_index(drop=True)
    return out


# -------------------------------------------------------------------------------------------------- RQ6
def analyse_rq6(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """The training methods in MCTS: every network trained by self-labelling (variant ``LR`` or ``AZ``) against
    the same network trained by self-play (variant ``LR-SP`` or ``AZ-SP``) at the same entropy level, and every
    one of them against U, over all instances and within each size class (Holm over the comparisons of a budget)."""
    out = _common(frame)
    u = labels_of(frame, "U")
    pairs, learned = [], []
    for name in ("LR", "AZ"):
        for label in labels_of(frame, name, variant=name):
            level = float(frame.loc[frame["label"] == label, "level"].iloc[0])
            other = labels_of(frame, name, level=level, variant=f"{name}-SP")
            pairs += [(label, sp) for sp in other]
            learned += [label] + other
    pairs += [(lab, b) for lab in learned for b in u]
    out["tests"] = paired_tests(frame, pairs, "gap", GROUPS, alpha, zero_method, holm_over="pairs")
    return out


# ------------------------------------------------------------------------------- rules (RQ1, RQ2 extension)
FIXED_DISTRIBUTIONS = ("U", "MWKR", "MOR", "SPT", "FIFO")


def validation_ranking(entropy_levels: Optional[dict], temperatures: Optional[dict] = None) -> pd.DataFrame:
    """The fixed distributions ranked by their validation score at the entropy level chosen for them.

    The scores come from the entropy selection (``select-entropy``): mean ``Cmax / LB0`` of MCTS with the same
    iterations, validation instances and seeds for every distribution (U at level 1).  The first row is the
    best fixed distribution on the validation instances, named before any test run.  With ``temperatures``
    (artifacts/temperatures.json) the entropy reached at the chosen level is added (``realised_entropy``): a
    level below the lowest entropy of a distribution (MOR, whose criterion has ties) cannot be reached.
    """
    rows = []
    for name in FIXED_DISTRIBUTIONS:
        entry = (entropy_levels or {}).get(name)
        if not entry:
            continue
        best = float(entry["best"])
        scores = {float(k): float(v) for k, v in entry["scores"].items()}
        realised = 1.0 if name == "U" or best >= 1.0 else (0.0 if best <= 0.0 else float("nan"))
        cal = (temperatures or {}).get(f"{name}|-|{entry.get('size')}|H{best:g}")
        if cal is not None and 0.0 < best < 1.0:
            realised = float(cal["realised"])
        rows.append({"distribution": name, "level": best, "realised_entropy": realised,
                     "validation_ratio": scores[best], "instances": entry.get("instances"),
                     "iterations": entry.get("iterations"), "size": entry.get("size")})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("validation_ratio", kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    table["best"] = table["rank"] == 1
    return table


def analyse_rules(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit",
                  entropy_levels: Optional[dict] = None, temperatures: Optional[dict] = None) -> dict:
    """MOR, SPT and FIFO as fixed rollout distributions, with U, MWKR and LR of RQ2 (same instances, budgets
    and runs).

    * ``friedman``: all fixed distributions and LR at every budget;
    * ``tests``: LR against every fixed distribution (Holm over the pairs of one budget, as in RQ2);
    * ``tests_best_fixed``: LR against the best fixed distribution on the validation instances
      (``validation``), the strictest comparison of the aim (Holm over the budgets);
    * ``tests_rules``: MOR, SPT and FIFO against U and against MWKR (Holm over the pairs of one budget).
    """
    out = _common(frame)
    fixed = {name: labels_of(frame, name) for name in FIXED_DISTRIBUTIONS}
    fixed_labels = [label for name in FIXED_DISTRIBUTIONS for label in fixed[name]]
    lr = labels_of(frame, "LR")
    out["friedman"] = friedman_tests(frame, fixed_labels + lr, "gap", GROUPS, alpha,
                                     name="U, MWKR, MOR, SPT, FIFO, LR")
    out["tests"] = paired_tests(frame, [(learned, base) for learned in lr for base in fixed_labels], "gap",
                                GROUPS, alpha, zero_method, holm_over="pairs")
    if not out["tests"].empty and not out["friedman"].empty:
        rejects = out["friedman"].set_index(["group", "budget"])["rejects"]
        out["tests"]["friedman_rejects"] = [bool(rejects.get((g, b), False))
                                            for g, b in zip(out["tests"]["group"], out["tests"]["budget"])]
    ranking = validation_ranking(entropy_levels, temperatures)
    if not ranking.empty:
        ranking = ranking[ranking["distribution"].map(lambda n: bool(fixed.get(n)))].reset_index(drop=True)
    out["validation"] = ranking
    if not ranking.empty and lr:
        best = str(ranking.iloc[0]["distribution"])
        headline = paired_tests(frame, [(learned, base) for learned in lr for base in fixed[best]], "gap",
                                GROUPS, alpha, zero_method, holm_over="budgets")
        if not headline.empty:
            headline.insert(0, "best_fixed", best)
        out["tests_best_fixed"] = headline
    rules = [label for name in ("MOR", "SPT", "FIFO") for label in fixed[name]]
    out["tests_rules"] = paired_tests(frame, [(r, b) for r in rules for b in fixed["U"] + fixed["MWKR"]], "gap",
                                      GROUPS, alpha, zero_method, holm_over="pairs")
    out["time_to_target"] = time_to_target(out["convergence"], fixed_labels)
    return out


# ----------------------------------------------------------------------------------------------- generic
def analyse_generic(frame: pd.DataFrame, alpha: float = 0.05, zero_method: str = "zsplit") -> dict:
    """Any other experiment: summary, convergence and every configuration against the first one."""
    out = _common(frame)
    labels = list(dict.fromkeys(frame["label"]))
    pairs = [(lab, labels[0]) for lab in labels[1:]]
    out["tests"] = paired_tests(frame, pairs, "gap", ("all",), alpha, zero_method, holm_over="pairs")
    if len(labels) >= 3:
        out["friedman"] = friedman_tests(frame, labels, "gap", ("all",), alpha, name="all configurations")
    return out


ANALYSES = {"rq1": analyse_rq1, "rq2": analyse_rq2, "rq3": analyse_rq3, "rq4": analyse_rq4, "rq5": analyse_rq5,
            "rq6": analyse_rq6}


def analyse(frame: pd.DataFrame, experiment: str, alpha: float = 0.05, zero_method: str = "zsplit",
            run_dir: Optional[str | Path] = None, entropy_levels: Optional[dict] = None,
            temperatures: Optional[dict] = None) -> dict:
    """Run the analysis that belongs to ``experiment`` (``rq1`` .. ``rq6``, ``rules``; anything else is
    generic).  ``entropy_levels`` (artifacts/entropy_levels.json) is used by ``rules``."""
    if experiment.lower().startswith("rules"):
        return analyse_rules(frame, alpha, zero_method, entropy_levels, temperatures)
    key = next((k for k in ANALYSES if experiment.lower().startswith(k)), None)
    if key == "rq3":
        return analyse_rq3(frame, alpha, zero_method, run_dir)
    if key is None:
        return analyse_generic(frame, alpha, zero_method)
    return ANALYSES[key](frame, alpha, zero_method)
