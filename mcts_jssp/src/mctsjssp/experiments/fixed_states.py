"""Comparison of the rollout distributions on a fixed set of states (Section 3.6.4, RQ3).

A fixed set of states, visited by the rollouts of U on the 24 instances of RQ3, is evaluated by every
distribution: uniform sampling and MWKR, LR and AZ at each entropy level.  This shows how the learned
distributions differ from the fixed one and from each other, independent of the search.

For every state and distribution the table holds the normalised entropy, the probability of the most
probable operation, and its distance from U, the reference of the study:

* ``kl_u`` - the Kullback-Leibler divergence from U, ``D_KL(p || U) = ln k - H(p)`` (Equation 2.22, nats);
  ``kl_u_norm`` divides it by ``ln k`` (0 = uniform, 1 = deterministic);
* ``tv_u`` - the total variation distance from U, ``0.5 * sum |p_i - 1/k|``: the share of the probability moved
  away from uniform (0 = uniform; at most ``1 - 1/k``);
* ``agree_u_search`` and ``visit_overlap_u`` - agreement with U in the search: from the state, UCT is run with
  the distribution's rollouts and with U's rollouts for the same number of iterations; the first records
  whether both commit the same operation (the most visited one), the second the overlap ``sum min(pi, pi_U)``
  of their visit distributions at the root.  U against U with another seed gives the baseline: how often
  two searches with U agree with each other.  Computed on the first ``fixed_states.search_per_instance``
  states of every instance with ``fixed_states.search_iterations`` iterations (0 switches it off).

It also holds the divergence from and agreement with MWKR at the same level (levels above 0 for the
divergence, where it is finite) and, for LR and AZ, with the other learned distribution.
"""
from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable

import numpy as np
import pandas as pd

from ..distributions.entropy import kl_divergence, normalised_entropy
from ..distributions.fixed import MWKRRollout, UniformRollout
from ..distributions.learned import make_learned_rollout
from ..env.model import State, replay
from ..mcts.evaluation import make_return
from ..mcts.uct import UCTSearch
from ..instances.instance import Instance
from ..instances.orlib import load_orlib, select_names
from .artifacts import Artifacts
from .config import get


def sample_states(inst: Instance, count: int, seed: int) -> list[list[int]]:
    """Job sequences that lead to ``count`` states visited by rollouts of U (only states with a choice)."""
    rnd = random.Random(seed)
    uniform = UniformRollout()
    prefixes: list[list[int]] = []
    while len(prefixes) < count:
        trace: list[int] = []
        uniform.rollout(State(inst), rnd, trace)
        # one state per rollout, at a uniformly drawn depth, so that the states are independent
        state = State(inst)
        choice_depths = []
        for depth, job in enumerate(trace):
            if len(state.active) > 1:
                choice_depths.append(depth)
            state.step(job)
        prefixes.append(trace[: rnd.choice(choice_depths)])
    return prefixes


def root_visits(inst: Instance, prefix: list, policy, iterations: int, c: float, seed: int) -> dict:
    """Visit distribution at the root of a UCT search from the state reached by ``prefix``."""
    state = replay(inst, prefix)
    search = UCTSearch(policy, make_return(inst, None, seed), c=c, seed=seed)
    search.start(state)
    search.run_iterations(state, iterations)
    total = sum(ch.N for ch in search.root.children) or 1
    return {ch.action: ch.N / total for ch in search.root.children}


def _agreement(pi: dict, ref: dict) -> tuple[bool, float]:
    best = max(pi, key=lambda a: (pi[a], -a))
    best_ref = max(ref, key=lambda a: (ref[a], -a))
    overlap = sum(min(pi.get(a, 0.0), ref.get(a, 0.0)) for a in set(pi) | set(ref))
    return best == best_ref, float(overlap)


def _settings(cfg: dict) -> dict:
    return {"per_instance": int(get(cfg, "fixed_states.per_instance", 50)),
            "levels": [float(v) for v in get(cfg, "entropy.levels", [0.0, 0.25, 0.5, 0.75])],
            "seed": int(cfg.get("seed", 0)),
            "learned": {spec["name"]: spec["model"] for spec in cfg["configurations"] if spec["name"] in ("LR", "AZ")},
            "search_iterations": int(get(cfg, "fixed_states.search_iterations", 100)),
            "search_states": int(get(cfg, "fixed_states.search_per_instance", 10)),
            "c": float(get(cfg, "search.c", 1.0))}


def _temperature_requests(artifacts: Artifacts, instances: list, settings: dict) -> list[tuple]:
    """The temperatures that the evaluation of ``instances`` looks up (calibrated beforehand, in parallel)."""
    out = []
    for inst in instances:
        for level in settings["levels"]:
            out.append(("MWKR", level, inst.shape, None))
            for conf, template in settings["learned"].items():
                model = template.format(size=inst.size_label, sigma="000")
                if artifacts.has_model(model):
                    out.append((conf, level, artifacts.model_size(model), model))
    return out


def _instance_rows(task: tuple) -> tuple[int, list[dict]]:
    """The rows of one instance (top-level: it runs in a worker process).  Its seeds depend only on the
    position of the instance, so the rows are the same in any process."""
    cfg, index, name, inst = task
    artifacts = Artifacts(cfg, log=lambda *_: None)
    s = _settings(cfg)
    per_instance, levels, seed, learned = s["per_instance"], s["levels"], s["seed"], s["learned"]
    search_iterations, search_states, c = s["search_iterations"], s["search_states"], s["c"]
    rows = []
    size = inst.shape
    policies: dict[tuple, object] = {("U", 1.0): UniformRollout()}
    for level in levels:
        policies[("MWKR", level)] = MWKRRollout(level, artifacts.temperature("MWKR", level, size))
        for conf, template in learned.items():
            model = template.format(size=inst.size_label, sigma="000")
            if not artifacts.has_model(model):
                continue
            tau = artifacts.temperature(conf, level, artifacts.model_size(model), model=model)
            policies[(conf, level)] = make_learned_rollout(conf, artifacts.inference(model), level, tau)
    for state_id, prefix in enumerate(sample_states(inst, per_instance, seed + index)):
        state = replay(inst, prefix)
        dists = {key: np.asarray(policy.distribution(state)) for key, policy in policies.items()}
        searched = search_iterations > 0 and state_id < search_states
        if searched:                                   # U twice: the reference and its own baseline
            s0 = seed + 1000 * index + state_id
            ref_u = root_visits(inst, prefix, policies[("U", 1.0)], search_iterations, c, s0)
        for (conf, level), p in dists.items():
            row = {"instance": name, "family": inst.family, "size": inst.size_label, "state": state_id,
                   "depth": state.n_done, "depth_fraction": state.n_done / inst.n_ops,
                   "n_candidates": len(state.active), "configuration": conf, "level": level,
                   "norm_entropy": normalised_entropy(p), "top_probability": float(p.max())}
            k = len(p)
            kl_u = float(np.log(k) + (p[p > 0] * np.log(p[p > 0])).sum()) if k > 1 else 0.0
            row["kl_u"] = max(kl_u, 0.0)
            row["kl_u_norm"] = row["kl_u"] / float(np.log(k)) if k > 1 else 0.0
            row["tv_u"] = float(0.5 * np.abs(p - 1.0 / k).sum())
            if searched:
                pi = root_visits(inst, prefix, policies[(conf, level)], search_iterations, c,
                                 s0 + 7919)           # another seed, also for U (the baseline)
                row["agree_u_search"], row["visit_overlap_u"] = _agreement(pi, ref_u)
            ref = dists.get(("MWKR", level))
            if conf in ("LR", "AZ") and ref is not None:
                row["agree_mwkr"] = bool(int(np.argmax(p)) == int(np.argmax(ref)))
                if level > 0.0:
                    row["kl_mwkr"] = kl_divergence(p, ref)
            other = {"LR": "AZ", "AZ": "LR"}.get(conf)
            if other and (other, level) in dists:
                q = dists[(other, level)]
                row["agree_other_learned"] = bool(int(np.argmax(p)) == int(np.argmax(q)))
                if level > 0.0:
                    row["kl_other_learned"] = kl_divergence(p, q)
            rows.append(row)
    return index, rows


def evaluate_fixed_states(cfg: dict, log: Callable[[str], None] = print, workers: int = 1) -> pd.DataFrame:
    """Evaluate every distribution of the experiment (normally RQ3) on the fixed states, one instance per
    worker process (``workers`` > 1); the table is the same as with one process."""
    artifacts = Artifacts(cfg, log)
    names = select_names(cfg["instances"])
    instances = load_orlib(names, cfg["paths"]["data"])
    s = _settings(cfg)
    tasks = [(cfg, index, name, instances[name]) for index, name in enumerate(names)]
    pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init) if workers > 1 and len(tasks) > 1 \
        else None
    parts: dict = {}
    try:
        artifacts.calibrate_missing(_temperature_requests(artifacts, list(instances.values()), s), pool)
        biggest_first = sorted(tasks, key=lambda t: -t[3].n_ops)
        done = map(_instance_rows, biggest_first) if pool is None else \
            pool.imap_unordered(_instance_rows, biggest_first, chunksize=1)
        for index, rows in done:
            parts[index] = rows
            kinds = len({(r["configuration"], r["level"]) for r in rows})
            log(f"  fixed states: {names[index]} done ({s['per_instance']} states, {kinds} distributions"
                + (f", searches on {min(s['search_states'], s['per_instance'])} states)" if s["search_iterations"]
                   else ")"))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return pd.DataFrame([row for index in sorted(parts) for row in parts[index]])


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)
