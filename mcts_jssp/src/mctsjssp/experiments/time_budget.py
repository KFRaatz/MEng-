"""Time budgets T50 and T200 (Section 3.6.3).

T_N gives every configuration the computation time per decision that U needs for N iterations on the same
instance, so that cheaper distributions run more iterations in the same time.  The time is measured by
running U with N iterations per decision for complete episodes; the mean time per decision is taken from
every repetition and the median over the repetitions is stored.  Decisions with a single legal action need no
search and take no time under any configuration, so they are not counted.

Two profiles are offered (``time_budget.profile``):

* ``mean`` (default): every decision of an instance gets the same time T_N, as a dispatcher would allow.
  Early decisions, whose rollouts are long, then get fewer than N iterations of U and late ones more.
* ``per_decision``: a decision with ``r`` unscheduled operations gets ``a + b * r`` seconds, the time that U
  needs for N iterations at that depth (fitted to the measured decisions), so that U itself runs about N
  iterations at every decision.

Times depend on the hardware: an entry measured on another host is measured again.
"""
from __future__ import annotations

import platform
import statistics
import time
from typing import Callable, Optional

import numpy as np

from ..distributions.fixed import UniformRollout
from ..instances.orlib import load_orlib
from ..mcts.agent import Budget, solve
from .artifacts import Artifacts
from .config import get
from .design import Planned


def budget_key(instance: str, iterations: int, sigma: Optional[float]) -> str:
    return f"{instance}|I{iterations}|s{(sigma or 0):g}"


def measure_time_budget(task: dict) -> dict:
    """Run U for ``repeats`` episodes and return the time per decision (top-level for multiprocessing)."""
    inst = load_orlib([task["instance"]], task["data_dir"])[task["instance"]]
    budget = Budget.of_iterations(task["iterations"])
    means, remaining, seconds = [], [], []
    for r in range(task["repeats"]):
        res = solve(inst, UniformRollout(), budget, seed=task["seed"] + r, c=task["c"], sigma=task["sigma"],
                    skip_forced=task["skip_forced"], check=False)
        means.append(res.time_per_decision)
        remaining += [inst.n_ops - d for d in res.decision_depths]
        seconds += res.decision_times
    # time of a decision as a function of the unscheduled operations: a + b * remaining (least squares)
    if len(set(remaining)) >= 2:
        b, a = np.polyfit(np.asarray(remaining, dtype=float), np.asarray(seconds, dtype=float), 1)
        a, b = max(float(a), 0.0), max(float(b), 0.0)
    else:
        a, b = statistics.median(means), 0.0
    return {
        "key": budget_key(task["instance"], task["iterations"], task["sigma"]),
        "mean": statistics.median(means),
        "per_decision": [a, b],
        "repeats": task["repeats"],
        "iterations": task["iterations"],
        "sigma": task["sigma"],
        "host": platform.node(),
        "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def ensure_time_budgets(cfg: dict, planned: list[Planned], artifacts: Artifacts, pool=None,
                        log: Callable[[str], None] = print) -> dict:
    """Time budget of every ``(instance, N, sigma)`` used by the planned runs; missing ones are measured.

    If a pool of workers is given, the measurements run in it, under the same load as the experiment.  A budget
    measured on another machine is measured again (T_N is the time that U needs on the machine of the runs);
    the ratio of the new to the old times is logged and kept in the entry (``replaced``), so that a later
    session on another machine of the same type can be compared with the earlier one.
    """
    needed = sorted({(p.instance, p.budget_value, p.sigma) for p in planned if p.budget_kind == "time"},
                    key=lambda t: (t[0], t[1], t[2] or 0.0))
    host = platform.node()
    out, tasks, previous = {}, [], {}
    for instance, iterations, sigma in needed:
        entry = artifacts.time_budgets.get(budget_key(instance, iterations, sigma))
        if entry is not None and entry.get("host") == host:
            out[(instance, iterations, sigma)] = entry
        else:
            if entry is not None:
                previous[budget_key(instance, iterations, sigma)] = entry
            tasks.append({
                "instance": instance, "iterations": iterations, "sigma": sigma,
                "repeats": int(get(cfg, "time_budget.repeats", 3)), "seed": int(cfg.get("seed", 0)),
                "c": float(get(cfg, "search.c", 1.0)), "skip_forced": bool(get(cfg, "search.skip_forced", True)),
                "data_dir": cfg["paths"]["data"],
            })
    if tasks:
        log(f"measuring {len(tasks)} time budgets with U ...")
        results = pool.imap_unordered(measure_time_budget, tasks, chunksize=1) if pool is not None \
            else map(measure_time_budget, tasks)
        ratios = []
        for entry in results:
            old = previous.get(entry["key"])
            if old is not None and float(old.get("mean", 0.0)) > 0:
                entry["replaced"] = {"host": old.get("host"), "mean": old["mean"], "measured": old.get("measured"),
                                     "ratio": float(entry["mean"]) / float(old["mean"])}
                ratios.append(entry["replaced"]["ratio"])
            artifacts.time_budgets.data[entry["key"]] = entry
            instance, tag, _ = entry["key"].split("|")
            out[(instance, int(tag[1:]), entry["sigma"])] = entry
        artifacts.time_budgets.save()
        if ratios:
            r = sorted(ratios)
            log(f"time budgets measured again on {host} (stored ones came from another machine): new / old time "
                f"median {r[len(r) // 2]:.3f}, range {r[0]:.3f} to {r[-1]:.3f} over {len(r)} budgets")
    return out
