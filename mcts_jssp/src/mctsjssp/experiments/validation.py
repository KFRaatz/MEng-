"""Validation of the dynamics model and the environment before the experiments (Section 3.7).

Three checks are made with methods whose behaviour is known.

1. Every complete schedule is checked independently of the model for precedence, machine overlaps and
   makespan, and executing a schedule without disturbances must reproduce its makespan.  (The same check
   runs on every schedule of the experiments, see :func:`mctsjssp.mcts.agent.solve`.)
2. The dispatching rules compared by Ho et al. (SPT, MWKR, MOR, FIFO) are run as complete policies and
   their gaps on the OR-Library instances are compared with the published values
   (``data/published_rule_gaps.csv``).  The comparison uses the rules as non-delay rules, the way they are
   implemented in the literature; the gaps of the same rules in the MDP of the thesis, where every
   unfinished job is a legal action, are reported next to them.
3. The learning pipeline is checked on the smallest training size: the GIN of LR, trained by self-play on
   6 x 6 instances, must lower the mean ``Cmax / LB0`` of its greedy policy on the validation instances
   below that of uniform random dispatching.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..distributions.fixed import MWKRRollout, UniformRollout
from ..env.checker import schedule_violations
from ..env.model import State, execute
from ..instances.disturbance import scenario
from ..instances.generator import validation_set
from ..instances.instance import Instance
from ..networks.inference import make_inference
from ..networks.checkpoint import load_checkpoint
from ..training.trainer import greedy_ratio, uniform_ratio
from .rules import RULES, run_rule


def check_schedules(instances: Sequence[Instance], random_schedules: int = 20, seed: int = 0) -> dict:
    """First check: random, MWKR and rule-based schedules of every instance pass the independent checker,
    execution without disturbances reproduces the makespan, and the schedule executed in a disturbed
    scenario is feasible for the realised processing times."""
    rnd = random.Random(seed)
    uniform, mwkr = UniformRollout(), MWKRRollout(0.5, 0.05)
    checked, failures = 0, []
    for inst in instances:
        nominal = inst.durations.ravel().tolist()
        states = [run_rule(inst, rule, non_delay) for rule in RULES.values() for non_delay in (False, True)]
        for policy in (uniform, mwkr):
            for _ in range(random_schedules):
                state = State(inst)
                policy.rollout(state, rnd, [])
                states.append(state)
        for state in states:
            checked += 1
            problems = schedule_violations(inst, state.start_times(), state.makespan)
            if execute(inst, state.m_op, nominal) != state.makespan:
                problems.append("execution without disturbances does not reproduce the makespan")
            # a disturbed scenario: the executed schedule must be feasible for the realised durations
            realised = scenario(inst, 0.2, 0)
            ends: list = []
            cmax = execute(inst, state.m_op, realised, ends_out=ends)
            dur = np.array(realised).reshape(inst.n, inst.m)
            start = np.array(ends).reshape(inst.n, inst.m) - dur
            problems += [f"disturbed: {v}" for v in schedule_violations(inst, start, cmax, durations=dur)]
            if problems:
                failures.append({"instance": inst.name, "problems": problems})
    return {"schedules_checked": checked, "failures": failures, "passed": not failures}


def rule_gaps(instances: Sequence[Instance]) -> pd.DataFrame:
    """Gap of every dispatching rule on every instance, as a non-delay rule and in the MDP of the thesis."""
    rows = []
    for inst in instances:
        for name, rule in RULES.items():
            for mode, non_delay in (("non_delay", True), ("free", False)):
                makespan = run_rule(inst, rule, non_delay=non_delay).makespan
                rows.append({"instance": inst.name, "family": inst.family, "size": inst.size_label, "rule": name,
                             "mode": mode, "makespan": makespan, "gap": inst.gap(makespan)})
    return pd.DataFrame(rows)


def compare_with_published(gaps: pd.DataFrame, published_csv: Optional[str | Path]) -> pd.DataFrame:
    """Second check: mean gap per family and rule next to the published value.

    ``published_csv`` has the columns ``family, rule, gap`` (gap in per cent) and optionally ``source``;
    lines that start with ``#`` are comments.  ``difference`` is the gap of the non-delay rule minus the
    published gap.  Without the file only the gaps obtained here are returned.
    """
    table = gaps.groupby(["family", "rule", "mode"])["gap"].mean().unstack("mode").reset_index()
    table = table.rename(columns={"non_delay": "gap_non_delay", "free": "gap_free_dispatching"})
    table.columns.name = None
    if published_csv and Path(published_csv).exists():
        with open(published_csv, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(line for line in fh if not line.startswith("#"))]
        published = pd.DataFrame(rows)
        if len(published):
            published = published[published["gap"].astype(str).str.strip() != ""]
        if len(published):
            published = published.assign(gap_published=published["gap"].astype(float))
            table = table.merge(published[["family", "rule", "gap_published"]], on=["family", "rule"], how="left")
            table["difference"] = table["gap_non_delay"] - table["gap_published"]
    return table


def learning_check(checkpoint: str | Path, backend: str = "numpy", instances: int = 100) -> dict:
    """Third check for a trained network: greedy policy against uniform random dispatching."""
    model, meta = load_checkpoint(checkpoint)
    conf = meta.get("config", {})
    val = list(validation_set(int(conf["n"]), int(conf["m"]))[:instances])
    greedy = greedy_ratio(make_inference(model, backend), val)
    uniform = uniform_ratio(val, seed=0, repeats=10)
    return {"model": meta.get("name"), "size": f"{conf['n']}x{conf['m']}", "greedy_ratio": greedy,
            "uniform_ratio": uniform, "passed": bool(greedy < uniform), "instances": len(val),
            "generation": meta.get("generation")}
