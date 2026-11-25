"""Tuning on the validation instances (Section 3.6.2, Table 3.5).

Two things are tuned, in two stages.

1. The training settings of LR and AZ - the learning rate (log-uniform on 1e-5..1e-3) and the ranked-reward
   percentile (50, 75 or 90) - are chosen by random search, 20 trials per configuration.  A trial is scored
   by the mean ``Cmax / LB0`` that its policy network achieves when it dispatches on its own, choosing
   greedily, on the validation instances.
2. After the temperatures of the entropy levels have been set by bisection, the entropy level used outside
   RQ3 is chosen for MWKR, LR and AZ by running MCTS with 100 iterations per decision on 20 validation
   instances at every level.  The same procedure chooses the levels of MOR, SPT and FIFO (extension of RQ1
   and RQ2); U, which has no level, gets its score at level 1, so that the fixed distributions can be
   ranked on the validation instances.

The exploration constant, the size of the replay buffer and the numbers of hidden units are not tuned.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from ..distributions.base import RolloutPolicy
from ..instances.instance import Instance
from ..mcts.agent import Budget, solve
from .trainer import TrainConfig, Trainer

LEARNING_RATE_RANGE = (1e-5, 1e-3)
PERCENTILES = (50.0, 75.0, 90.0)


def sample_trials(trials: int, seed: int, lr_range=LEARNING_RATE_RANGE, percentiles=PERCENTILES) -> list[dict]:
    """The random-search trials: learning rate log-uniform on ``lr_range``, percentile from ``percentiles``."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2_020]))
    lo, hi = np.log10(lr_range[0]), np.log10(lr_range[1])
    return [{"trial": t, "learning_rate": float(10 ** rng.uniform(lo, hi)),
             "percentile": float(percentiles[int(rng.integers(len(percentiles)))])} for t in range(trials)]


def _run_trial(args) -> dict:
    cfg, out_dir, quiet = args
    import torch

    torch.set_num_threads(1)
    log = (lambda *_: None) if quiet else print
    done = Path(out_dir) / "summary.json"
    if done.exists():                      # finished in an earlier session: reuse it
        summary = json.loads(done.read_text(encoding="utf-8"))
    else:
        summary = Trainer(cfg, out_dir, log=log).run()
    return {"name": cfg.name, "learning_rate": cfg.learning_rate, "percentile": cfg.percentile,
            "validation_ratio": summary["best_validation_ratio"], "generations": summary["generations"],
            "model_steps": summary["model_steps"], "stopped_by": summary["stopped_by"]}


def random_search(base: TrainConfig, out_dir: str | Path, trials: int = 20, seed: int = 0,
                  parallel_trials: int = 1, log: Callable[[str], None] = print) -> dict:
    """Random search over the learning rate and the ranked-reward percentile for one configuration.

    Every trial trains with the settings of ``base`` (which carry the tuning budget, for example
    ``max_generations`` and ``budget_model_steps``) and its own learning rate and percentile.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    finished = sum((out_dir / f"trial{t['trial']:02d}" / "summary.json").exists() for t in sample_trials(trials, seed))
    if finished:
        log(f"{finished} of {trials} trials finished earlier; they are reused")
    for t in sample_trials(trials, seed):
        cfg = replace(base, name=f"{base.name}_trial{t['trial']:02d}", learning_rate=t["learning_rate"],
                      percentile=t["percentile"], workers=1 if parallel_trials > 1 else base.workers)
        jobs.append((cfg, out_dir / f"trial{t['trial']:02d}", parallel_trials > 1))
    if parallel_trials > 1:
        with mp.get_context("spawn").Pool(parallel_trials) as pool:
            rows = pool.map(_run_trial, jobs, chunksize=1)
    else:
        rows = []
        for job in jobs:
            rows.append(_run_trial(job))
            log(f"trial {len(rows):2d}/{trials}: lr {rows[-1]['learning_rate']:.2e} percentile "
                f"{rows[-1]['percentile']:.0f} -> validation Cmax/LB0 {rows[-1]['validation_ratio']:.4f}")
    best = min(rows, key=lambda r: r["validation_ratio"])
    result = {"configuration": base.name, "kind": base.kind, "size": f"{base.n}x{base.m}", "trials": rows,
              "best": {"learning_rate": best["learning_rate"], "percentile": best["percentile"],
                       "validation_ratio": best["validation_ratio"], "trial": best["name"]},
              "base_config": asdict(base)}
    with open(out_dir / "tuning.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def select_entropy_level(make_policy: Callable[[float], RolloutPolicy], instances: Sequence[Instance],
                         levels: Sequence[float], iterations: int = 100, seed: int = 0, c: float = 1.0,
                         sigma: Optional[float] = None) -> dict:
    """Choose the entropy level of one configuration with MCTS on validation instances (Table 3.5).

    ``make_policy(level)`` builds the rollout distribution at a level (with its calibrated temperature).
    Returns the mean ``Cmax / LB0`` of every level and the best level.
    """
    scores = {}
    for level in levels:
        policy = make_policy(level)
        ratios = [solve(inst, policy, Budget.of_iterations(iterations), seed=seed + i, c=c, sigma=sigma).makespan
                  / inst.lb0 for i, inst in enumerate(instances)]
        scores[float(level)] = float(np.mean(ratios))
    return _selection_result(scores, iterations, len(instances))


def _selection_result(scores: dict, iterations: int, instances: int) -> dict:
    best = min(scores, key=scores.get)
    return {"scores": scores, "best": best, "iterations": iterations, "instances": instances}


def _selection_episode(task: tuple) -> float:
    """``Cmax / LB0`` of one MCTS episode of the entropy selection (top-level: it runs in a worker process)."""
    cfg, name, model, level, tau, inst, iterations, seed, c = task
    from ..distributions.fixed import FIXED_RULES, UniformRollout, make_rule_rollout
    from ..distributions.learned import make_learned_rollout
    from ..experiments.artifacts import Artifacts

    if name == "U":
        policy = UniformRollout()
    elif name in FIXED_RULES:
        policy = make_rule_rollout(name, level, tau)
    else:
        policy = make_learned_rollout(name, Artifacts(cfg, log=lambda *_: None).inference(model), level, tau)
    return solve(inst, policy, Budget.of_iterations(iterations), seed=seed, c=c, sigma=None).makespan / inst.lb0


def select_entropy_levels(cfg: dict, artifacts, specs: Sequence[tuple], levels: Sequence[float], iterations: int,
                          seed: int, c: float = 1.0, pool=None) -> list[dict]:
    """:func:`select_entropy_level` for several configurations at once, with the episodes in the worker
    processes of ``pool`` (all configurations together, the most expensive first).

    ``specs`` holds (name, model, validation size, instances) per configuration (model None for U and the
    dispatching rules; U is scored at level 1 only, the level of uniform sampling); the
    results are in the same order.  The temperatures are calibrated first (in parallel too).  Every episode
    has its own seed, as in :func:`select_entropy_level`, so the result is the same as one episode after the
    other in one process.
    """
    levels = [float(v) for v in levels]
    def levels_of(name: str) -> list:
        return [1.0] if name == "U" else levels

    artifacts.calibrate_missing([(name, level, size, model) for name, model, size, _ in specs
                                 for level in levels_of(name)], pool)
    tasks, where = [], []
    for k, (name, model, size, instances) in enumerate(specs):
        for level in levels_of(name):
            tau = artifacts.temperature(name, level, size, model=model)
            for i, inst in enumerate(instances):
                tasks.append((cfg, name, model, level, tau, inst, iterations, seed + i, c))
                where.append((k, level))
    cost = [inst.n_ops ** 2 * (1.0 if model is None else 30.0)
            for _, name, model, _, _, inst, *_ in tasks]
    order = sorted(range(len(tasks)), key=lambda t: -cost[t])
    if pool is None:
        done = [_selection_episode(tasks[t]) for t in order]
    else:
        done = pool.map(_selection_episode, [tasks[t] for t in order], chunksize=1)
    ratios = [0.0] * len(tasks)
    for t, ratio in zip(order, done):
        ratios[t] = ratio
    out = []
    for k, (name, model, size, instances) in enumerate(specs):
        scores = {level: float(np.mean([r for r, w in zip(ratios, where) if w == (k, level)]))
                  for level in levels_of(name)}
        out.append(_selection_result(scores, iterations, len(instances)))
    return out
