"""States visited by the rollouts of the experiments, compared across configurations and experiments.

The experiments store the complete job sequence of a random sample of their rollouts
(``recording.rollout_sequences``; ``records/<run>.sequences.npz``).  Every state a rollout visited is a prefix
of its sequence, so the states are rebuilt exactly by stepping through it.  By default only the states of the
rollout itself are used (from the end of the tree path to the complete schedule); ``part="all"`` adds the
states of the tree path.

A state is identified by its partial schedule: the set of (operation, start time) pairs of the scheduled
operations.  Two different sequences that build the same partial schedule are the same state.  The key is a
64-bit hash of that set, updated in constant time per step (a Zobrist hash).

For every experiment, configuration (label), budget, instance and depth bin (share of the operations
scheduled) the tables give:

* ``states`` and ``distinct`` - states visited by the sampled rollouts and how many of them differ;
  ``distinct_share`` = distinct / states measures how diverse the rollouts are;
* ``partial_ratio`` - the mean makespan of the partial schedule divided by LB0: how good the visited states are;
* ``overlap_u`` - the Jaccard overlap |A ∩ B| / |A ∪ B| of the distinct states with those of U on the same
  instance and budget (U is the reference of the study); ``overlap`` holds every pair of configurations.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..env.model import State
from ..instances.orlib import load_orlib

MASK = (1 << 64) - 1


def _zobrist(n_ops: int, seed: int = 20240611) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(1, 2 ** 63 - 1, size=n_ops, dtype=np.int64)]


def _mix(z: int, start: int) -> int:
    x = (z ^ ((start + 1) * 0x9E3779B97F4A7C15)) & MASK      # splitmix64 finaliser of (operation, start)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def walk(inst, sequence: Iterable[int], begin: int, zob: list[int]) -> list[tuple[int, float, float]]:
    """(key, depth share, partial makespan / LB0) of every state from position ``begin`` of the sequence on."""
    state = State(inst)
    key = 0
    out = []
    n_ops = inst.n_ops
    for i, job in enumerate(sequence):
        if i >= begin:
            out.append((key, state.n_done / n_ops, state.makespan / inst.lb0))
        state.step(int(job))
        op = state.last[0]
        key ^= _mix(zob[op], state.op_start[op])
    return out


def _read_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collect(run_dirs: dict, part: str = "rollout", bins: int = 10, instances: Optional[set] = None,
            max_sequences: Optional[int] = None, data_dir: Optional[str] = None, log=print) -> dict:
    """Key sets and statistics per (experiment, label, budget, instance, depth bin)."""
    keys: dict = defaultdict(set)
    stats: dict = defaultdict(lambda: [0, 0.0])            # states, sum of partial ratios
    meta: dict = {}
    cache: dict = {}
    for experiment, run_dir in run_dirs.items():
        run_dir = Path(run_dir)
        rows = _read_results(run_dir)
        found = 0
        for row in rows:
            name = row["instance"]
            if instances and name not in instances:
                continue
            if row.get("sigma"):                        # disturbed runs: states depend on the scenario
                continue
            path = run_dir / "records" / f"{row['run_id']}.sequences.npz"
            if not path.exists():
                continue
            if name not in cache:
                inst = load_orlib([name], data_dir)[name] if data_dir else load_orlib([name])[name]
                cache[name] = (inst, _zobrist(inst.n_ops))
            inst, zob = cache[name]
            data = np.load(path)
            seqs = data["sequence"]
            starts = data["rollout_start"] if part == "rollout" else data["tree_start"]
            if max_sequences is not None:
                seqs, starts = seqs[:max_sequences], starts[:max_sequences]
            label = row["label"]
            group = (experiment, label, row["budget"], name)
            meta[group] = {"config": row["config"], "level": row.get("level")}
            for seq, begin in zip(seqs, starts):
                for key, depth, ratio in walk(inst, seq, int(begin), zob):
                    b = min(int(depth * bins), bins - 1)
                    keys[group + (b,)].add(key)
                    s = stats[group + (b,)]
                    s[0] += 1
                    s[1] += ratio
            found += 1
        log(f"  {experiment}: {found} runs with stored rollout sequences")
    return {"keys": keys, "stats": stats, "meta": meta, "bins": bins}


def tables(collected: dict, reference: str = "U") -> dict:
    keys, stats, meta, bins = collected["keys"], collected["stats"], collected["meta"], collected["bins"]
    rows = []
    for k, (n, total) in stats.items():
        experiment, label, budget, inst, b = k
        rows.append({"experiment": experiment, "label": label, "config": meta[k[:4]]["config"],
                     "level": meta[k[:4]]["level"], "budget": budget, "instance": inst,
                     "depth_bin": b, "depth": (b + 0.5) / bins, "states": n, "distinct": len(keys[k]),
                     "distinct_share": len(keys[k]) / n if n else np.nan, "partial_ratio": total / n})
    per_state = pd.DataFrame(rows)
    # overlap between every pair of configurations on the same instance, budget and depth bin
    groups = defaultdict(list)
    for k in keys:
        experiment, label, budget, inst, b = k
        groups[(budget, inst, b)].append((f"{experiment}:{label}", k))
    pairs = []
    for (budget, inst, b), members in groups.items():
        for i, (name_a, ka) in enumerate(members):
            for name_b, kb in members[i + 1:]:
                a, c = keys[ka], keys[kb]
                union = len(a | c)
                pairs.append({"budget": budget, "instance": inst, "depth_bin": b, "depth": (b + 0.5) / bins,
                              "a": name_a, "b": name_b, "overlap": len(a & c) / union if union else np.nan})
    overlap = pd.DataFrame(pairs)
    # overlap with U: the U of the same experiment if it has one, otherwise the U of any experiment
    if not per_state.empty:
        def u_key(experiment, budget, inst, b):
            own = (experiment, reference, budget, inst, b)
            if own in keys:
                return own
            return next((k for k in keys if k[1] == reference and k[2:] == (budget, inst, b)), None)
        vals = []
        for r in per_state.itertuples():
            ku = u_key(r.experiment, r.budget, r.instance, r.depth_bin)
            mine = keys[(r.experiment, r.label, r.budget, r.instance, r.depth_bin)]
            if ku is None or r.label == reference:
                vals.append(np.nan if ku is None else 1.0)
                continue
            union = len(mine | keys[ku])
            vals.append(len(mine & keys[ku]) / union if union else np.nan)
        per_state["overlap_u"] = vals
    summary = (per_state.groupby(["experiment", "label", "config", "budget", "depth_bin", "depth"], dropna=False)
               .agg(instances=("instance", "nunique"), states=("states", "sum"),
                    distinct_share=("distinct_share", "mean"), partial_ratio=("partial_ratio", "mean"),
                    overlap_u=("overlap_u", "mean"))
               .reset_index()) if not per_state.empty else per_state
    return {"visited_states": per_state, "visited_summary": summary, "visited_overlap": overlap}
