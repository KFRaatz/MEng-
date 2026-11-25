"""Command line interface: one command per step of the workflow of Figure 3.6.

    python -m mctsjssp validate-env    --config configs/base.yaml            # Section 3.7, checks 1 and 2
    python -m mctsjssp pilot           --config configs/pilot.yaml           # time per iteration -> hours
    python -m mctsjssp tune            --config configs/base.yaml --configuration LR
    python -m mctsjssp train           --config configs/base.yaml --model lr_6x6
    python -m mctsjssp train-fast      --config configs/base.yaml --model flr_15x15   # fast rollout policy
    python -m mctsjssp validate-env    --config configs/base.yaml --learning # Section 3.7, check 3
    python -m mctsjssp assess          --config configs/base.yaml --model lr_15x15
    python -m mctsjssp select-entropy  --config configs/base.yaml            # level used outside RQ3
    python -m mctsjssp run             --config configs/experiments/rq1.yaml --workers 8
    python -m mctsjssp fixed-states    --config configs/experiments/rq3.yaml
    python -m mctsjssp analyse         --config configs/experiments/rq1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def _load(args):
    from .experiments.config import load_config

    return load_config(args.config, args.set or [])


def _print_table(table: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    if table is None or table.empty:
        print("  (empty)")
        return
    with pd.option_context("display.max_rows", 200, "display.max_columns", 40, "display.width", 220,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(table.to_string(index=False))


# -------------------------------------------------------------------------------------------- commands
def cmd_validate_env(args) -> int:
    from .experiments.artifacts import Artifacts
    from .experiments.config import get
    from .experiments.validation import check_schedules, compare_with_published, learning_check, rule_gaps
    from .instances.orlib import load_orlib

    cfg = _load(args)
    out_dir = Path(cfg["paths"]["results"]) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = list(load_orlib(data_dir=cfg["paths"]["data"]).values())
    ok = True
    if not args.learning:
        first = check_schedules(instances, int(get(cfg, "validation.random_schedules", 20)), int(cfg.get("seed", 0)))
        print(f"Check 1 - schedules: {first['schedules_checked']} schedules of {len(instances)} instances checked, "
              f"{len(first['failures'])} failures -> {'PASSED' if first['passed'] else 'FAILED'}")
        ok &= first["passed"]
        gaps = rule_gaps(instances)
        gaps.to_csv(out_dir / "rule_gaps_per_instance.csv", index=False)
        published = Path(cfg["paths"]["data"]).parent / "published_rule_gaps.csv"
        table = compare_with_published(gaps, published)
        table.to_csv(out_dir / "rule_gaps_per_family.csv", index=False)
        _print_table(table.pivot(index="family", columns="rule", values="gap_free_dispatching").reset_index(),
                     "Mean gap (%) of the dispatching rules in the MDP of the thesis (every unfinished job is a "
                     "legal action)")
        _print_table(table.pivot(index="family", columns="rule", values="gap_non_delay").reset_index(),
                     "Mean gap (%) of the same rules as non-delay rules (as implemented in the literature)")
        if "gap_published" in table.columns and table["gap_published"].notna().any():
            tolerance = float(get(cfg, "validation.rule_tolerance", 0.5))
            worst = float(table["difference"].abs().max())
            second = worst <= tolerance
            _print_table(table.pivot(index="family", columns="rule", values="gap_published").reset_index(),
                         "Published mean gap (%)")
            print(f"\nCheck 2 - dispatching rules: largest difference between the non-delay rules and the "
                  f"published gaps is {worst:.2f} percentage points (tolerance {tolerance:g}) -> "
                  f"{'PASSED' if second else 'FAILED'}")
            ok &= second
        else:
            print(f"\nCheck 2 - dispatching rules: no published values in {published}; fill the file in to compare")
        with open(out_dir / "check_schedules.json", "w", encoding="utf-8") as fh:
            json.dump(first, fh, indent=1)
    else:
        artifacts = Artifacts(cfg)
        model = args.model or get(cfg, "validation.learning_check.model", "lr_6x6")
        res = learning_check(artifacts.model_path(model), cfg.get("inference_backend", "numpy"))
        print(f"Check 3 - learning pipeline ({res['model']}, {res['size']}): greedy policy Cmax/LB0 = "
              f"{res['greedy_ratio']:.4f}, uniform random dispatching = {res['uniform_ratio']:.4f} -> "
              f"{'PASSED' if res['passed'] else 'FAILED'}")
        with open(out_dir / f"learning_check_{model}.json", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        ok &= res["passed"]
    return 0 if ok else 1


def cmd_pilot(args) -> int:
    from .experiments.pilot import pilot_study

    cfg = _load(args)
    out_dir = Path(cfg["paths"]["results"]) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Pilot study: time per iteration of every configuration")
    res = pilot_study(cfg)
    for name in ("timing", "by_configuration", "design", "training"):
        res[name].to_csv(out_dir / f"{name}.csv", index=False)
    _print_table(res["by_configuration"], "Core hours of the full design, by experiment and configuration")
    cols = ["design", "total_core_hours", "wall_hours", "fits"]
    _print_table(res["design"][cols], f"Design and reductions (wall hours with {res['workers']} workers; "
                                      f"{res['available_hours']:g} hours available)")
    _print_table(res["training"], "Self-play training: hours per generation in one process")
    print(f"\nTables written to {out_dir}")
    return 0


def _train_config(cfg: dict, name: str, workers=None):
    from .experiments.config import get
    from .training.trainer import TrainConfig

    if name not in cfg.get("models", {}):
        raise SystemExit(f"model {name!r} is not listed under 'models' in the configuration")
    spec = cfg["models"][name]
    t = dict(cfg.get("training", {}))
    kind = spec["kind"]
    learning_rate, percentile = float(t.get("learning_rate", 5e-5)), float(t.get("percentile", 75))
    fast = str(kind).startswith("fast")
    if fast:                                   # fast networks: their own learning rate (training.fast)
        learning_rate = float(get(cfg, "training.fast.learning_rate", learning_rate))
    tuned_path = Path(cfg["paths"]["artifacts"]) / "tuning" / ("LR" if kind == "gin" else "AZ") / "tuning.json"
    if t.get("use_tuned", True) and tuned_path.exists() and not fast:
        with open(tuned_path, encoding="utf-8") as fh:
            best = json.load(fh)["best"]
        learning_rate, percentile = float(best["learning_rate"]), float(best["percentile"])
        print(f"[{name}] using the tuned learning rate {learning_rate:.3g} and percentile {percentile:g}")
    from .training.disturbed import network_seed

    seed = network_seed(int(cfg.get("seed", 0)), name) if spec.get("sigma") else \
        int(np.random.SeedSequence([int(cfg.get("seed", 0)), zlib.crc32(name.encode())]).generate_state(1)[0])
    reference = t.get("reference_models")                   # networks for disturbances: the nominal networks
    if reference and not Path(reference).is_absolute():
        from .experiments.config import project_root

        reference = str(project_root(Path(cfg["_file"])) / reference) if cfg.get("_file") else str(Path(reference))
    return TrainConfig(
        name=name, kind=kind, n=int(spec["n"]), m=int(spec["m"]),
        hparams={**dict(get(cfg, f"training.networks.{kind}", {}) or {}), **dict(spec.get("hparams") or {})},
        episodes_per_generation=int(t.get("episodes_per_generation", 40)), iterations=int(t.get("iterations", 100)),
        c=float(get(cfg, "search.c", 1.0)), noise_weight=float(t.get("noise_weight", 0.25)),
        noise_alpha_scale=float(t.get("noise_alpha_scale", 10.0)), q_init=str(t.get("q_init", "zero")),
        percentile=percentile,
        window=int(t.get("window", 250)), rank_against_generation0=bool(t.get("rank_against_generation0", False)),
        value_target=str(t.get("value_target", "outcome")), leaf_rollouts=bool(t.get("leaf_rollouts", False)),
        rollout_mix=float(t.get("rollout_mix", 0.5)), normalize_q=bool(t.get("normalize_q", False)),
        method=str(t.get("method", "alphazero")), self_label=dict(t.get("self_label") or {}) or None,
        min_generations=int(t.get("min_generations", 0)), warm_start=t.get("warm_start"),
        sigma=spec.get("sigma"), buffer_size=int(t.get("buffer_size", 36000)),
        batch_size=int(t.get("batch_size", 256)), learning_rate=learning_rate,
        weight_decay=float(t.get("weight_decay", 5e-4)), policy_weight=float(t.get("policy_weight", 1.0)),
        value_weight=float(t.get("value_weight", 1.0)), patience=int(t["patience"]) if t.get("patience") else None,
        max_hours=float(t["max_hours"]) if t.get("max_hours") else None,
        verbose=int(t.get("verbose", 1)),
        max_generations=int(t.get("max_generations", 100)), budget_model_steps=t.get("budget_model_steps"),
        validation_instances=int(t.get("validation_instances", 100)), seed=seed,
        workers=int(workers if workers is not None else t.get("workers", 1)),
        fit_threads=t.get("fit_threads", "auto"), reference_models=reference,
        inference_backend=cfg.get("inference_backend", "numpy"),
    )


def _trainer(tc, out_dir):
    """The trainer of the configured method: self-labelling (the study) or PUCT self-play (run 1)."""
    if tc.method == "self_label":
        from .training.self_label import SelfLabelTrainer

        return SelfLabelTrainer(tc, out_dir)
    if tc.method != "alphazero":
        raise SystemExit(f"unknown training.method {tc.method!r}; expected self_label or alphazero")
    from .training.trainer import Trainer

    return Trainer(tc, out_dir)


def cmd_train(args) -> int:
    import torch

    from .experiments.artifacts import Artifacts

    cfg = _load(args)
    torch.set_num_threads(1 if (args.workers or 1) > 1 else max(1, torch.get_num_threads()))
    models = cfg.get("models", {})
    names = [n for n, spec in models.items() if not spec.get("teacher")] if args.all else (args.model or [])
    if not names:
        raise SystemExit("give --model NAME (repeatable) or --all")
    artifacts = Artifacts(cfg)
    for name in names:
        if models.get(name, {}).get("teacher"):
            raise SystemExit(f"{name} copies a teacher; train it with 'python -m mctsjssp train-fast'")
        if artifacts.has_model(name) and not args.force:
            print(f"[{name}] already trained ({artifacts.model_path(name)}); use --force to train again")
            continue
        summary = _trainer(_train_config(cfg, name, args.workers), artifacts.model_dir(name)).run()
        unit = "iterations" if summary.get("method") == "self_label" else "generations"
        print(f"[{name}] stopped by {summary['stopped_by']} after {summary['generations']} {unit}; best "
              f"validation Cmax/LB0 {summary['best_validation_ratio']:.4f} (uniform {summary['uniform_ratio']:.4f})")
    return 0


def cmd_train_fast(args) -> int:
    """Train the fast networks: by the same PUCT self-play as LR and AZ, from scratch (default), or by copying a
    dispatching rule or a trained network when the model lists a ``teacher``.  ``train`` trains them too."""
    import time

    import torch

    from .experiments.artifacts import Artifacts
    from .experiments.config import get
    from .instances.generator import validation_set
    from .networks.checkpoint import build_model, save_checkpoint
    from .networks.inference import make_inference
    from .training.imitation import imitate_fast, network_feature_records, rule_feature_records
    from .training.trainer import greedy_ratio, uniform_ratio

    cfg = _load(args)
    artifacts = Artifacts(cfg)
    models = cfg.get("models", {})
    names = [n for n, spec in models.items() if str(spec.get("kind", "")).startswith("fast")] if args.all \
        else (args.model or [])
    if not names:
        raise SystemExit("give --model NAME (repeatable) or --all")
    fcfg = get(cfg, "training.fast", {}) or {}
    rows = []
    for name in names:
        spec = models.get(name)
        if spec is None or not str(spec.get("kind", "")).startswith("fast"):
            raise SystemExit(f"{name} is not a fast policy network under 'models'")
        if artifacts.has_model(name) and not args.force:
            print(f"[{name}] already trained ({artifacts.model_path(name)}); use --force to train again")
            continue
        kind, n, m = spec["kind"], int(spec["n"]), int(spec["m"])
        hparams = dict(get(cfg, f"training.networks.{kind}", {}) or {})
        hparams.update(spec.get("hparams") or {})          # per-model architecture, e.g. {hidden: 0} = linear
        if kind == "fast_az":
            hparams.update(n=n, m=m)
        seed = int(np.random.SeedSequence([int(cfg.get("seed", 0)), zlib.crc32(name.encode())]).generate_state(1)[0])
        torch.manual_seed(seed)
        model = build_model(kind, hparams)
        if not spec.get("teacher"):     # trained from scratch by the training method of LR and AZ
            tc = _train_config(cfg, name, getattr(args, "workers", None))
            summary = _trainer(tc, artifacts.model_dir(name)).run()
            rows.append({"model": name, "kind": kind, "size": f"{n}x{m}",
                         "teacher": "self-labelling" if tc.method == "self_label" else "self-play (PUCT)",
                         "generations": summary["generations"], "greedy_ratio": summary["best_validation_ratio"],
                         "uniform_ratio": summary["uniform_ratio"]})
            continue
        teacher = dict(spec["teacher"])
        copy = dict(fcfg.get("copy", {}) or {})
        count = int(copy.get("instances", 300))
        t0 = time.perf_counter()
        if "model" in teacher:
            feats, masks, targets, _ = network_feature_records(artifacts.inference(teacher["model"]), n, m, count,
                                                               seed, tau=float(teacher.get("tau", 1.0)))
        else:
            feats, masks, targets, _ = rule_feature_records(str(teacher.get("rule", "FIFO")), n, m, count, seed,
                                                            bool(teacher.get("non_delay", False)))
        losses = imitate_fast(model, feats, masks, targets, epochs=int(copy.get("epochs", 8)),
                              learning_rate=float(copy.get("learning_rate", 1e-3)),
                              batch_size=int(get(cfg, "training.batch_size", 256)),
                              weight_decay=float(get(cfg, "training.weight_decay", 5e-4)), seed=seed)
        val = list(validation_set(n, m)[: int(get(cfg, "training.validation_instances", 100))])
        score = greedy_ratio(make_inference(model), val)
        meta = {"name": name, "config": {"n": n, "m": m, "kind": kind, "hparams": hparams}, "teacher": teacher,
                "records": int(len(feats)), "losses": losses, "validation_score": score,
                "uniform_score": uniform_ratio(val, seed=seed), "seconds": time.perf_counter() - t0}
        save_checkpoint(artifacts.model_path(name), model, meta)
        with open(artifacts.model_dir(name) / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        rows.append({"model": name, "kind": kind, "size": f"{n}x{m}", "teacher": teacher.get("model") or teacher.get("rule"),
                     "decisions": int(len(feats)), "loss": round(losses[-1], 4), "greedy_ratio": score,
                     "uniform_ratio": meta["uniform_score"]})
    _print_table(pd.DataFrame(rows), "Fast policy networks (mean Cmax/LB0 of the greedy policy on the validation instances)")
    return 0


def cmd_tune(args) -> int:
    from dataclasses import replace

    from .experiments.config import get
    from .training.tuning import random_search

    cfg = _load(args)
    if str(get(cfg, "training.method", "alphazero")) == "self_label":
        print(f"[{args.configuration}] tuning tunes PUCT self-play (training.method: alphazero); "
              f"self-labelling uses training.self_label - skipped")
        return 0
    kind = {"LR": "gin", "AZ": "ffnn"}[args.configuration]
    n, m = (int(v) for v in get(cfg, "tuning.size", [15, 15]))
    name = f"{'lr' if kind == 'gin' else 'az'}_{n}x{m}"
    cfg.setdefault("models", {}).setdefault(name, {"kind": kind, "n": n, "m": m})
    cfg.setdefault("training", {})["use_tuned"] = False
    base = _train_config(cfg, name, args.workers)
    base = replace(base, name=f"tune_{args.configuration}", n=n, m=m, kind=kind,
                   max_generations=int(get(cfg, "tuning.max_generations", base.max_generations)),
                   budget_model_steps=get(cfg, "tuning.budget_model_steps", base.budget_model_steps),
                   max_hours=get(cfg, "tuning.max_hours", None))      # training.max_hours is not used here
    out_dir = Path(cfg["paths"]["artifacts"]) / "tuning" / args.configuration
    if (out_dir / "tuning.json").exists() and not getattr(args, "force", False):
        best = json.loads((out_dir / "tuning.json").read_text(encoding="utf-8"))["best"]
        print(f"[{args.configuration}] already tuned: learning rate {best['learning_rate']:.3g}, percentile "
              f"{best['percentile']:g}; use --force to tune again")
        return 0
    result = random_search(base, out_dir, trials=int(get(cfg, "tuning.trials", 20)), seed=int(cfg.get("seed", 0)),
                           parallel_trials=args.parallel)
    _print_table(pd.DataFrame(result["trials"]), f"Random search for {args.configuration} on {n}x{m}")
    print(f"\nbest: learning rate {result['best']['learning_rate']:.3g}, percentile "
          f"{result['best']['percentile']:g} (validation Cmax/LB0 {result['best']['validation_ratio']:.4f})")
    return 0


def cmd_assess(args) -> int:
    from .experiments.artifacts import Artifacts
    from .instances.generator import validation_set
    from .training.offline import assess_offline

    cfg = _load(args)
    artifacts = Artifacts(cfg)
    names = list(cfg.get("models", {})) if args.all else (args.model or [])
    rows = []
    for name in names:
        if not artifacts.has_model(name):
            print(f"[{name}] not trained yet, skipped")
            continue
        infer = artifacts.inference(name)
        n, m = artifacts.model_size(name)
        res = assess_offline(infer, list(validation_set(n, m)), infer.meta.get("ranked_reward"))
        res = {"model": name, "size": f"{n}x{m}", **res}
        with open(artifacts.model_dir(name) / "offline.json", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        rows.append(res)
    _print_table(pd.DataFrame(rows), "Offline assessment on the validation instances")
    return 0


def cmd_select_entropy(args) -> int:
    import multiprocessing as mp
    import os
    import time

    from .experiments.artifacts import Artifacts
    from .experiments.config import get
    from .experiments.runner import worker_init
    from .instances.generator import validation_set
    from .training.tuning import select_entropy_levels

    cfg = _load(args)
    artifacts = Artifacts(cfg)
    levels = [float(v) for v in get(cfg, "entropy.levels", [0.0, 0.25, 0.5, 0.75])]
    count = int(get(cfg, "entropy.selection.instances", 20))
    iterations = int(get(cfg, "entropy.selection.iterations", 100))
    specs = []
    for spec in get(cfg, "entropy.selection.configurations", []):
        name, model = spec["name"], spec.get("model")
        if getattr(args, "missing", False) and name in artifacts.entropy_levels:
            print(f"[{name}] level chosen earlier ({artifacts.tuned_level(name):g}), kept")
            continue
        if model and not artifacts.has_model(model):
            print(f"[{name}] network {model} is not trained yet, skipped")
            continue
        size = artifacts.model_size(model) if model else tuple(get(cfg, "entropy.default_size", [15, 15]))
        specs.append((name, model, size, list(validation_set(*size)[:count])))
    # the episodes are independent (every one has its own seed): one per worker process, same result as in one
    # process; default: one worker per physical core (half of the logical CPUs)
    workers = int(args.workers) if args.workers else max(1, (os.cpu_count() or 2) // 2)
    episodes = len(specs) * len(levels) * count
    print(f"entropy selection: {episodes} episodes of MCTS with {iterations} iterations on {workers} worker(s)",
          flush=True)
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(workers, initializer=worker_init) if workers > 1 and specs else None
    try:
        results = select_entropy_levels(cfg, artifacts, specs, levels, iterations, seed=int(cfg.get("seed", 0)),
                                        c=float(get(cfg, "search.c", 1.0)), pool=pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"entropy selection: {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for (name, model, size, _), res in zip(specs, results):
        res.update(model=model, size=f"{size[0]}x{size[1]}")
        artifacts.entropy_levels.set(name, res)
        rows.append({"configuration": name, "model": model, "best_level": res["best"],
                     **{f"H{lv:g}": score for lv, score in res["scores"].items()}})
    if rows:
        _print_table(pd.DataFrame(rows), f"Entropy level by MCTS with {iterations} iterations on {count} "
                                         f"validation instances (mean Cmax/LB0)")
    from .analysis.rq import validation_ranking
    ranking = validation_ranking(artifacts.entropy_levels.data, artifacts.temperatures.data)
    if len(ranking) > 1:
        _print_table(ranking, "Fixed rollout distributions on the validation instances (at the chosen level)")
    return 0


def cmd_calibrate(args) -> int:
    from .experiments.artifacts import Artifacts
    from .experiments.design import plan, resolve

    cfg = _load(args)
    artifacts = Artifacts(cfg)
    planned = [p for p in plan(cfg, artifacts) if p.budget_kind == "iterations" and p.run == 0] or \
              [p for p in plan(cfg, artifacts) if p.run == 0]
    seen = set()
    unique = [p for p in planned if not ((p.name, p.model, p.calibration, p.level) in seen
                                         or seen.add((p.name, p.model, p.calibration, p.level)))]
    fake_times = {(p.instance, p.budget_value, p.sigma): {"mean": 0.0, "per_decision": []} for p in unique}
    jobs = resolve(cfg, unique, artifacts, fake_times)
    rows = [{"configuration": j["label"], "model": j["planned"]["model"],
             "validation_size": "x".join(str(v) for v in j["planned"]["calibration"] or ()),
             "level": j["planned"]["level"], "tau": j["tau"]} for j in jobs if j["planned"]["name"] != "U"]
    _print_table(pd.DataFrame(rows).drop_duplicates(), "Temperatures of the entropy levels")
    return 0


def cmd_run(args) -> int:
    from .experiments.runner import run_experiment

    cfg = _load(args)
    run_dir = run_experiment(cfg, workers=args.workers, limit=args.limit, hours=args.hours)
    print(f"results: {run_dir}")
    return 0


def cmd_fixed_states(args) -> int:
    import os

    from .experiments.fixed_states import evaluate_fixed_states
    from .experiments.runner import run_directory

    cfg = _load(args)
    run_dir = Path(args.run_dir) if args.run_dir else run_directory(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers) if args.workers else max(1, (os.cpu_count() or 2) // 2)
    table = evaluate_fixed_states(cfg, workers=workers)
    table.to_csv(run_dir / "fixed_states.csv", index=False)
    print(f"{len(table)} rows -> {run_dir / 'fixed_states.csv'}")
    from .analysis.plots import fixed_state_summary, plot_fixed_states

    summary = fixed_state_summary(table)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "fixed_states_summary.csv", index=False)
    print(f"summary -> {out / 'fixed_states_summary.csv'}")
    if not getattr(args, "no_figures", False):
        for path in plot_fixed_states(summary, out):
            print(f"figure -> {path}")
    return 0


SELF_PLAY_MODELS = ["lr_6x6", "lr_10x10", "lr_15x15", "az_6x6", "az_10x10", "az_15x15"]


def cmd_robustness(args) -> int:
    """RQ5 without search: the networks for nominal processing times, fine-tuned with disturbances and trained with
    disturbances from scratch, under disturbances on the validation instances (results/training/robustness*)."""
    import os

    import torch

    from .experiments.config import get, load_config
    from .training.robustness import evaluate, write

    cfg = _load(args)
    torch.set_num_threads(1)
    sigmas = [float(v) for v in (args.sigma or [])]
    if not sigmas:
        rq5 = Path(cfg["_file"]).resolve().parent / "experiments" / "rq5.yaml"
        sigmas = [float(v) for v in (get(load_config(rq5), "stochastic.sigmas", None) if rq5.exists() else None)
                  or [0.1, 0.2]]
    workers = int(args.workers) if args.workers else max(1, (os.cpu_count() or 2) // 2)
    result = evaluate(cfg, sigmas, workers)
    for path in write(result, Path(cfg["paths"]["results"]) / "training"):
        print(f"-> {path}")
    return 0


def cmd_disturbed_report(args) -> int:
    """The networks for disturbances over their training: table and figures in results/training."""
    import pandas as pd

    from .analysis.disturbed_report import histories, plot

    cfg = _load(args)
    root = Path(cfg["paths"]["artifacts"])
    scratch = Path(args.scratch) if args.scratch else root / "scratch"
    frame = histories({"FT": root / "models", "SC": scratch / "models"})
    out = Path(cfg["paths"]["results"]) / "training"
    out.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print("no network for disturbances validated under disturbances yet")
        return 0
    frame.to_csv(out / "disturbed_history.csv", index=False)
    rob = out / "robustness.csv"
    for path in [out / "disturbed_history.csv", *plot(frame, pd.read_csv(rob) if rob.exists() else None, out)]:
        print(f"-> {path}")
    last = frame.sort_values("iteration").groupby(["network", "variant"]).tail(1)
    print(last[["network", "variant", "sigma", "iteration", "best_iteration", "stopped_by"]].to_string(index=False))
    return 0


def cmd_compare_training(args) -> int:
    """Self-labelling (the networks of the experiments) against AlphaZero-style self-play (Section 3.6.1)."""
    from .analysis.compare_training import compare_training

    cfg = _load(args)
    artifacts = Path(cfg["paths"]["artifacts"])
    other = Path(args.other) if args.other else artifacts / "self_play"
    roots = {"self-labelling": artifacts / "models", "self-play": other / "models"}
    out = Path(args.out) if args.out else Path(cfg["paths"]["results"]) / "training"
    rows = compare_training(cfg, args.model or SELF_PLAY_MODELS, roots, int(args.workers), out)
    if not any(r.get("status") == "trained" for r in rows):
        print("compare-training: no trained networks found")
    print(f"table -> {out / 'compare_training.csv'} and .md; figure -> {out / 'compare_training.png'}")
    return 0


def cmd_training_report(args) -> int:
    """Training curves and a summary of every trained network."""
    from .analysis.plots import plot_training
    from .analysis.training_report import load_histories, summary

    cfg = _load(args)
    models_dir = Path(cfg["paths"]["artifacts"]) / "models"
    histories = load_histories(models_dir)
    if args.model:
        histories = {k: v for k, v in histories.items() if k in set(args.model)}
    if not histories:
        raise SystemExit(f"no history.json under {models_dir}")
    out = Path(cfg["paths"]["results"]) / "training"
    out.mkdir(parents=True, exist_ok=True)
    table = summary(histories, models_dir)
    table.to_csv(out / "training_summary.csv", index=False)
    print(table.round(4).to_string(index=False))
    need = float(args.min_improvement)
    bad = table[~(table["improvement_over_uniform_pct"] > need)]
    for path in plot_training(histories, out):
        print(f"figure -> {path}")
    if len(bad):
        print(f"\nWARNING: {len(bad)} network(s) do not beat uniform random dispatching by more than {need:g} %: "
              f"{', '.join(bad['model'])}")
        required = [m for m in bad["model"] if not args.require or any(m.startswith(p) for p in args.require)]
        if args.strict and required:
            print(f"STOP: the experiments need networks that have learned ({', '.join(required)}; see "
                  f"results/training/).")
            return 3
        if args.strict:
            print(f"(not a reason to stop: only networks starting with {', '.join(args.require)} must have learned)")
    else:
        print(f"\nall {len(table)} networks beat uniform random dispatching by more than {need:g} %")
    return 0


def cmd_compare(args) -> int:
    """Compare the rollout policies of several experiments instance by instance."""
    from .analysis.compare import choose_reference, compare_experiments
    from .analysis.plots import plot_compare_instances
    from .experiments.config import load_config
    from .experiments.runner import run_directory

    run_dirs = {}
    for path in args.config or []:
        cfg = load_config(path)
        run_dirs[cfg["experiment"]] = run_directory(cfg)
    for path in args.run_dir or []:
        run_dirs[Path(path).name] = Path(path)
    if not run_dirs:
        raise SystemExit("give --config FILE or --run-dir DIR (both repeatable)")
    result = compare_experiments(run_dirs, reference=args.reference, alpha=args.alpha)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in result["tables"].items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows -> {out / (name + '.csv')}")
    if not result["tables"]["mean_rank"].empty:
        print(result["tables"]["mean_rank"].round(3).to_string(index=False))
    if not args.no_figures:
        for path in plot_compare_instances(result["wides"], lambda cols: choose_reference(cols, args.reference),
                                           out):
            print(f"figure -> {path}")
    return 0


def cmd_compare_states(args) -> int:
    """Compare the states visited by the rollouts across configurations and experiments."""
    from .analysis.plots import plot_visited
    from .analysis.visited import collect, tables
    from .experiments.config import load_config
    from .experiments.runner import run_directory

    run_dirs = {}
    for path in args.config or []:
        cfg = load_config(path)
        run_dirs[cfg["experiment"]] = run_directory(cfg)
    for path in args.run_dir or []:
        run_dirs[Path(path).name] = Path(path)
    if not run_dirs:
        raise SystemExit("give --config FILE or --run-dir DIR (both repeatable)")
    collected = collect(run_dirs, part=args.part, bins=args.bins,
                        instances=set(args.instance) if args.instance else None,
                        max_sequences=args.max_sequences)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = tables(collected)
    for name, table in result.items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows -> {out / (name + '.csv')}")
    if not args.no_figures:
        for path in plot_visited(result["visited_summary"], out):
            print(f"figure -> {path}")
    return 0


def cmd_analyse(args) -> int:
    from .analysis.report import write_report
    from .experiments.config import get
    from .experiments.runner import run_directory

    from .experiments.artifacts import JsonStore

    cfg = _load(args)
    run_dir = Path(args.run_dir) if args.run_dir else run_directory(cfg)
    levels = JsonStore(Path(cfg["paths"]["artifacts"]) / "entropy_levels.json").data
    temperatures = JsonStore(Path(cfg["paths"]["artifacts"]) / "temperatures.json").data
    tables = write_report(run_dir, cfg["experiment"], alpha=float(get(cfg, "analysis.alpha", 0.05)),
                          zero_method=get(cfg, "analysis.zero_method", "zsplit"), figures=not args.no_figures,
                          extra_dirs=args.with_runs or (), entropy_levels=levels, temperatures=temperatures)
    for name in args.show or []:
        _print_table(tables.get(name), name)
    return 0


# ------------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mctsjssp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="configuration file (YAML)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting, e.g. runs=1")
        p.set_defaults(func=func)
        return p

    p = add("validate-env", cmd_validate_env, "validate the dynamics model and the environment (Section 3.7)")
    p.add_argument("--learning", action="store_true", help="third check: trained GIN against uniform dispatching")
    p.add_argument("--model", help="network of the third check (default: validation.learning_check.model)")
    add("pilot", cmd_pilot, "pilot study: time per iteration and hours of computation")
    p = add("train", cmd_train, "train networks (training.method: self-labelling by default)")
    p.add_argument("--model", action="append", help="name under 'models' (repeatable)")
    p.add_argument("--all", action="store_true", help="train every network listed under 'models'")
    p.add_argument("--workers", type=int, help="processes that play the self-play episodes")
    p.add_argument("--force", action="store_true", help="train again even if a checkpoint exists")
    p = add("train-fast", cmd_train_fast, "train the fast networks (PUCT self-play as LR and AZ, or copying a teacher)")
    p.add_argument("--model", action="append", help="name of a fast network under 'models' (repeatable)")
    p.add_argument("--all", action="store_true", help="train every fast network listed under 'models'")
    p.add_argument("--force", action="store_true", help="train again even if a checkpoint exists")
    p.add_argument("--workers", type=int, help="processes that play the self-play episodes")
    p = add("tune", cmd_tune, "random search of the learning rate and the ranked-reward percentile")
    p.add_argument("--configuration", required=True, choices=["LR", "AZ"])
    p.add_argument("--parallel", type=int, default=1, help="trials trained at the same time")
    p.add_argument("--workers", type=int, help="self-play processes of one trial")
    p.add_argument("--force", action="store_true", help="tune again even if tuning.json exists")
    p = add("assess", cmd_assess, "offline assessment of trained networks")
    p.add_argument("--model", action="append")
    p.add_argument("--all", action="store_true")
    p = add("select-entropy", cmd_select_entropy, "choose the entropy level used outside RQ3")
    p.add_argument("--workers", type=int, help="worker processes (default: half of the logical CPUs)")
    p.add_argument("--missing", action="store_true",
                   help="only the configurations without a stored level (the others are kept as they are)")
    add("calibrate", cmd_calibrate, "set the temperatures that an experiment needs (bisection)")
    p = add("run", cmd_run, "run or resume an experiment")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, help="run at most this many of the remaining runs")
    p.add_argument("--hours", type=float, help="stop after this many hours; run again to continue")
    p = add("fixed-states", cmd_fixed_states, "evaluate every distribution on a fixed set of states (RQ3)")
    p.add_argument("--run-dir")
    p.add_argument("--workers", type=int, help="worker processes (default: half of the logical CPUs)")
    p.add_argument("--no-figures", action="store_true")
    p = add("robustness", cmd_robustness, "RQ5 without search: nominal, fine-tuned and from-scratch networks under "
            "disturbances on the validation instances")
    p.add_argument("--sigma", action="append", type=float, help="disturbance level (repeatable; default: RQ5's)")
    p.add_argument("--workers", type=int, help="worker processes (default: half of the logical CPUs)")
    p = add("disturbed-report", cmd_disturbed_report, "the networks for disturbances over their training")
    p.add_argument("--scratch", help="artifacts folder of the networks trained from scratch (default <artifacts>/scratch)")
    p = add("compare-training", cmd_compare_training,
            "compare the training styles: self-labelling against AlphaZero-style self-play (Section 3.6.1)")
    p.add_argument("--model", action="append", help=f"networks (repeatable; default {' '.join(SELF_PLAY_MODELS)})")
    p.add_argument("--other", help="artifacts folder of the self-play networks (default <artifacts>/self_play)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", help="output folder (default <results>/training)")
    p = add("training-report", cmd_training_report, "training curves and a summary of every trained network")
    p.add_argument("--model", action="append", help="only these networks (repeatable)")
    p.add_argument("--strict", action="store_true", help="exit with code 3 if a network has not learned")
    p.add_argument("--require", action="append", help="with --strict: only networks whose name starts with this "
                   "must have learned (repeatable)")
    p.add_argument("--min-improvement", type=float, default=0.0,
                   help="improvement over uniform random (%%) a network needs to count as learned")
    p = sub.add_parser("compare", help="compare the rollout policies of several experiments instance by instance")
    p.set_defaults(func=cmd_compare)
    p.add_argument("--config", action="append", help="experiment configuration (repeatable)")
    p.add_argument("--run-dir", action="append", help="directory of results (repeatable)")
    p.add_argument("--reference", default="U", help="configuration to compare against (default U)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default="results/compare", help="folder for the tables and figures")
    p.add_argument("--no-figures", action="store_true")
    p = sub.add_parser("compare-states",
                       help="compare the states visited by the rollouts across configurations and experiments")
    p.set_defaults(func=cmd_compare_states)
    p.add_argument("--config", action="append", help="experiment configuration (repeatable)")
    p.add_argument("--run-dir", action="append", help="directory of results (repeatable)")
    p.add_argument("--out", default="results/visited_states", help="folder for the tables and figures")
    p.add_argument("--part", choices=["rollout", "all"], default="rollout",
                   help="states of the rollouts only (default) or also of the tree path")
    p.add_argument("--bins", type=int, default=10, help="depth bins (share of the operations scheduled)")
    p.add_argument("--instance", action="append", help="only these instances (repeatable)")
    p.add_argument("--max-sequences", type=int, help="at most this many stored rollouts per run")
    p.add_argument("--no-figures", action="store_true")
    p = add("analyse", cmd_analyse, "tables, statistical tests and figures of an experiment")
    p.add_argument("--run-dir", help="directory of the results (default: derived from the configuration)")
    p.add_argument("--with", dest="with_runs", action="append", metavar="RUN_DIR",
                   help="results of another experiment analysed together with these (the rules extension: RQ2)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--show", action="append", metavar="TABLE", help="print a table, e.g. --show tests")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
