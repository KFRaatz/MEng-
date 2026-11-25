"""Pilot study (Section 3.6.3): the time per iteration of every configuration, converted into hours.

The rollouts dominate the cost of the search, and a rollout from a state with ``r`` unscheduled operations
takes about ``a + b * r`` seconds.  The pilot study measures the time per iteration of every configuration
on one instance of every size at several depths of the schedule, fits ``a`` and ``b``, and adds up the time
of every run of the design of Table 3.6:

* iteration budget of ``B`` iterations: ``sum over the decisions of B * (a + b * r)``;
* time budget T_N: every decision takes the time that U needs for N iterations, or the time of one iteration
  of the configuration if that is longer (an iteration that has started is completed).

If the design does not fit the time available, the reductions of Section 3.6.3 are costed in their order:
RQ4 with the 6 x 6 and 15 x 15 networks only, RQ3 with the entropy levels 0 and 0.5 only, and the largest
iteration budget of LR on a subset of the large instances.

The cost of a rollout does not depend on the weights of a network, so the pilot study can run before
anything is trained: untrained networks are timed where no checkpoint exists.
"""
from __future__ import annotations

import random
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..distributions.fixed import FIXED_RULES, MWKRRollout, UniformRollout, make_rule_rollout
from ..distributions.learned import make_learned_rollout
from ..env.model import State
from ..instances.instance import Instance
from ..instances.orlib import EXPECTED_SIZE, load_orlib
from ..mcts.evaluation import make_return
from ..mcts.uct import UCTSearch
from ..networks.checkpoint import build_model
from ..networks.inference import make_inference
from .artifacts import Artifacts
from .config import get, load_config
from .design import Planned, plan


def _state_at_depth(inst: Instance, fraction: float, seed: int) -> State:
    state = State(inst)
    rnd = random.Random(seed)
    for _ in range(int(fraction * inst.n_ops)):
        state.step(rnd.choice(state.active))
    return state


def time_per_iteration(inst: Instance, policy, depths=(0.0, 0.25, 0.5, 0.75), iterations: int = 30,
                       repeats: int = 3, seed: int = 0, sigma: Optional[float] = None,
                       max_seconds: float = 5.0) -> dict:
    """Seconds per UCT iteration as a function of the number of unscheduled operations: ``a + b * r``."""
    points = []
    for fraction in depths:
        state = _state_at_depth(inst, fraction, seed)
        if len(state.active) < 2:
            continue
        samples = []
        for r in range(repeats):
            search = UCTSearch(policy, make_return(inst, sigma, seed + r), seed=seed + r)
            search.start(state)
            done, t0 = 0, time.perf_counter()
            while done < iterations and (done < 3 or time.perf_counter() - t0 < max_seconds):
                search.run_iterations(state, 1)
                done += 1
            samples.append((time.perf_counter() - t0) / done)
        points.append((inst.n_ops - state.n_done, statistics.median(samples)))
    r_values = np.array([p[0] for p in points], dtype=float)
    t_values = np.array([p[1] for p in points], dtype=float)
    if len(points) >= 2:
        b, a = np.polyfit(r_values, t_values, 1)
        a, b = max(float(a), 0.0), max(float(b), 0.0)
    else:
        a, b = 0.0, float(t_values[0] / max(r_values[0], 1.0))
    return {"a": a, "b": b, "points": points}


def _policy_for(name: str, size: tuple, model: Optional[str], artifacts: Artifacts, backend: str):
    if name == "U":
        return UniformRollout()
    if name == "MWKR":
        return MWKRRollout(0.5, 0.02)                 # a sampled level; the greedy rule is cheaper
    if name in FIXED_RULES:
        return make_rule_rollout(name, 0.5, 0.02)
    if model and artifacts.has_model(model):
        infer = artifacts.inference(model)
    else:                                             # untrained network of the same architecture
        spec = (artifacts.cfg.get("models") or {}).get(model or "", {})
        kind = spec.get("kind") or ("gin" if name == "LR" else "ffnn")
        hparams = dict(get(artifacts.cfg, f"training.networks.{kind}", {}) or {})
        if kind in ("ffnn", "fast_az"):
            hparams.update(n=size[0], m=size[1])
        infer = make_inference(build_model(kind, hparams), backend)
    return make_learned_rollout(name, infer, 0.5, 1.0)


def run_seconds(p: Planned, coeff: dict, n_ops: int) -> float:
    """Estimated time of one planned run."""
    stochastic = bool(p.sigma)
    a, b = coeff[(p.name, EXPECTED_SIZE[p.instance], stochastic)]
    remaining = np.arange(n_ops, 0, -1, dtype=float)
    per_iteration = a + b * remaining
    if p.budget_kind == "iterations":
        return float(p.budget_value * per_iteration.sum())
    a_u, b_u = coeff[("U", EXPECTED_SIZE[p.instance], stochastic)]
    t_ref = p.budget_value * float((a_u + b_u * remaining).mean())          # T_N of the instance
    return float(np.maximum(t_ref, per_iteration).sum())


def pilot_study(cfg: dict, log: Callable[[str], None] = print) -> dict:
    """Measure and cost the designs listed under ``pilot.designs``; returns tables as DataFrames."""
    pcfg = cfg.get("pilot", {})
    artifacts = Artifacts(cfg, log)
    backend = cfg.get("inference_backend", "numpy")
    config_dir = Path(cfg["_file"]).resolve().parent
    designs = {}
    for rel in pcfg.get("designs", []):
        dcfg = load_config(config_dir / rel)
        designs[dcfg["experiment"]] = plan(dcfg, artifacts=None)
    instances = load_orlib(data_dir=cfg["paths"]["data"])

    # one measurement per (configuration, instance size, static or stochastic)
    needed = sorted({(p.name, EXPECTED_SIZE[p.instance], bool(p.sigma)) for runs in designs.values() for p in runs}
                    | {("U", EXPECTED_SIZE[p.instance], bool(p.sigma)) for runs in designs.values() for p in runs})
    representative = {}
    for name in instances:
        representative.setdefault(EXPECTED_SIZE[name], name)
    coeff, timing_rows = {}, []
    for name, size, stochastic in needed:
        inst = instances[representative[size]]
        model = None
        for runs in designs.values():
            model = next((p.model for p in runs if p.name == name and p.calibration == size and p.model), model)
        policy = _policy_for(name, size, model, artifacts, backend)
        fit = time_per_iteration(inst, policy, tuple(pcfg.get("depths", (0.0, 0.25, 0.5, 0.75))),
                                 int(pcfg.get("iterations", 30)), int(pcfg.get("repeats", 3)),
                                 seed=int(cfg.get("seed", 0)), sigma=0.1 if stochastic else None)
        coeff[(name, size, stochastic)] = (fit["a"], fit["b"])
        first = fit["a"] + fit["b"] * inst.n_ops
        timing_rows.append({"configuration": name, "size": f"{size[0]}x{size[1]}", "stochastic": stochastic,
                            "instance": inst.name, "ms_per_iteration_first_decision": first * 1e3,
                            "ms_per_iteration_mid_schedule": (fit["a"] + fit["b"] * inst.n_ops / 2) * 1e3,
                            "a_seconds": fit["a"], "b_seconds_per_operation": fit["b"]})
        log(f"  {name:5s} {size[0]:2d}x{size[1]:<2d} {'stochastic' if stochastic else 'static    '} "
            f"{first * 1e3:9.3f} ms per iteration at the first decision")
    # the fast networks as rollout distributions (not part of the design; timed for comparison)
    if pcfg.get("fast_networks", True):
        models = cfg.get("models") or {}
        fast_lr = next((k for k, v in models.items() if v.get("kind") == "fast_lr"), None)
        fast_az = {(int(v["n"]), int(v["m"])): k for k, v in models.items() if v.get("kind") == "fast_az"}
        sizes = sorted({(size, st) for _, size, st in needed})
        for size, stochastic in sizes:
            for label, model in (("FastLR", fast_lr), ("FastAZ", fast_az.get(size))):
                if model is None:
                    continue
                inst = instances[representative[size]]
                policy = _policy_for("LR" if label == "FastLR" else "AZ", size, model, artifacts, backend)
                fit = time_per_iteration(inst, policy, tuple(pcfg.get("depths", (0.0, 0.25, 0.5, 0.75))),
                                         int(pcfg.get("iterations", 30)), int(pcfg.get("repeats", 3)),
                                         seed=int(cfg.get("seed", 0)), sigma=0.1 if stochastic else None)
                first = fit["a"] + fit["b"] * inst.n_ops
                timing_rows.append({"configuration": label, "size": f"{size[0]}x{size[1]}",
                                    "stochastic": stochastic, "instance": inst.name,
                                    "ms_per_iteration_first_decision": first * 1e3,
                                    "ms_per_iteration_mid_schedule": (fit["a"] + fit["b"] * inst.n_ops / 2) * 1e3,
                                    "a_seconds": fit["a"], "b_seconds_per_operation": fit["b"]})
                log(f"  {label:6s} {size[0]:2d}x{size[1]:<2d} {'stochastic' if stochastic else 'static    '} "
                    f"{first * 1e3:9.3f} ms per iteration at the first decision")
    timing = pd.DataFrame(timing_rows)

    def cost(runs: list[Planned]) -> float:
        return sum(run_seconds(p, coeff, EXPECTED_SIZE[p.instance][0] * EXPECTED_SIZE[p.instance][1])
                   for p in runs) / 3600.0

    # the design and its reductions (Section 3.6.3)
    subset = set(pcfg.get("large_subset", []))
    reduced = {k: list(v) for k, v in designs.items()}
    steps = [("full design of Table 3.6", {k: list(v) for k, v in reduced.items()})]
    if "rq4" in reduced:
        reduced["rq4"] = [p for p in reduced["rq4"] if not (p.name == "LR" and p.calibration == (10, 10))]
        steps.append(("1. RQ4 with the 6x6 and 15x15 networks", {k: list(v) for k, v in reduced.items()}))
    if "rq3" in reduced:
        reduced["rq3"] = [p for p in reduced["rq3"] if p.name == "U" or p.level in (0.0, 0.5)]
        steps.append(("2. RQ3 with the entropy levels 0 and 0.5", {k: list(v) for k, v in reduced.items()}))

    def is_large(name: str) -> bool:
        n, m = EXPECTED_SIZE[name]
        return n * m > 250

    for key, runs in reduced.items():
        budgets = [p.budget_value for p in runs if p.name == "LR" and p.budget_kind == "iterations"]
        if budgets:
            largest = max(budgets)
            reduced[key] = [p for p in runs if not (p.name == "LR" and p.budget_kind == "iterations"
                                                    and p.budget_value == largest and is_large(p.instance)
                                                    and p.instance not in subset)]
    steps.append(("3. largest iteration budget of LR on a subset of the large instances",
                  {k: list(v) for k, v in reduced.items()}))

    workers = int(pcfg.get("workers", 1))
    available = float(pcfg.get("available_hours", 0) or 0)
    design_rows, detail_rows = [], []
    for label, state in steps:
        total = 0.0
        row = {"design": label}
        for key, runs in state.items():
            hours = cost(runs)
            row[f"{key}_core_hours"] = hours
            row[f"{key}_runs"] = len(runs)
            total += hours
        row["total_core_hours"] = total
        row["wall_hours"] = total / workers
        row["fits"] = (total / workers <= available) if available else None
        design_rows.append(row)
    for key, runs in designs.items():
        for name in sorted({p.name for p in runs}):
            sel = [p for p in runs if p.name == name]
            detail_rows.append({"experiment": key, "configuration": name, "runs": len(sel),
                                "core_hours": cost(sel)})
    design = pd.DataFrame(design_rows)
    detail = pd.DataFrame(detail_rows)

    training = training_cost(cfg, coeff, instances, representative, artifacts, backend)
    return {"timing": timing, "design": design, "by_configuration": detail, "training": training,
            "workers": workers, "available_hours": available}


def training_cost(cfg: dict, coeff: dict, instances: dict, representative: dict, artifacts: Artifacts,
                  backend: str) -> pd.DataFrame:
    """Hours per generation of self-play for every network (one process)."""
    from ..mcts.puct import PUCTSearch

    tcfg = cfg.get("training", {})
    episodes = int(tcfg.get("episodes_per_generation", 40))
    iterations = int(tcfg.get("iterations", 100))
    rows = []
    for name, spec in (cfg.get("models") or {}).items():
        n, m = int(spec["n"]), int(spec["m"])
        size = (n, m)
        n_ops = n * m
        inst = instances.get(representative.get(size, ""))
        if inst is None:
            continue
        key = ("U", size, bool(spec.get("sigma")))
        if key not in coeff:
            fit = time_per_iteration(inst, UniformRollout(), iterations=20, repeats=2,
                                     sigma=spec.get("sigma"))
            coeff[key] = (fit["a"], fit["b"])
        a, b = coeff[key]
        gen0 = episodes * iterations * float((a + b * np.arange(n_ops, 0, -1)).sum()) / 3600.0
        hparams = dict(get(cfg, f"training.networks.{spec['kind']}", {}) or {})
        if spec["kind"] in ("ffnn", "fast_az"):
            hparams.update(n=n, m=m)
        infer = make_inference(build_model(spec["kind"], hparams), backend)
        state = _state_at_depth(inst, 0.5, 0)
        search = PUCTSearch(infer, lambda st: 0.0, seed=0)   # PUCT self-play for every network
        search.start(state)
        t0 = time.perf_counter()
        search.run_iterations(state, 50)
        per_iteration = (time.perf_counter() - t0) / 50
        later = episodes * iterations * n_ops * per_iteration / 3600.0
        rows.append({"model": name, "kind": spec["kind"], "size": f"{n}x{m}", "sigma": spec.get("sigma"),
                     "hours_generation_0": gen0, "hours_per_later_generation": later,
                     "ms_per_puct_iteration": per_iteration * 1e3})
    return pd.DataFrame(rows)
