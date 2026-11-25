"""Execution of the experiments (Sections 3.6.3 and 3.6.5).

:func:`run_experiment` expands a configuration file into runs, resolves their artifacts, executes the runs -
in this process or in a pool of workers - and stores every result under a unique identifier::

    results/<experiment>-<hash of the configuration>/
      config.yaml        the configuration that produced the results
      meta.json          software versions and hardware
      resolved.json      entropy levels, temperatures and checkpoints actually used
      results.jsonl      one line per run (instance, configuration, budget, seed, makespan, gap, ...)
      records/           detailed records of the runs (decisions, rollouts, sampled rollout steps)

A run that is already in ``results.jsonl`` is not repeated, so an interrupted experiment can be resumed.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..distributions.base import RolloutPolicy
from ..distributions.fixed import FIXED_RULES, MWKRRollout, UniformRollout, make_rule_rollout
from ..distributions.learned import make_learned_rollout
from ..env.model import execute, replay
from ..instances.disturbance import scenarios
from ..instances.orlib import load_orlib
from ..mcts.agent import Budget, solve
from ..mcts.recorder import RunRecorder
from ..networks.checkpoint import file_fingerprint
from .artifacts import Artifacts, load_inference
from .config import config_hash, dump_config
from .design import plan, resolve, temperature_requests
from .time_budget import ensure_time_budgets


# ------------------------------------------------------------------------------------------- single run
def build_policy(job: dict) -> RolloutPolicy:
    """The rollout distribution of a job (Table 3.2)."""
    p = job["planned"]
    name, level = p["name"], p["level"]
    if name == "U":
        return UniformRollout()
    if name in FIXED_RULES:
        return make_rule_rollout(name, level, job["tau"])
    infer = load_inference(job["model_path"], job["backend"])
    policy = make_learned_rollout(name, infer, level, job["tau"])
    if job.get("ref_tau") is not None:
        policy.reference = MWKRRollout(level, job["ref_tau"]).distribution
    return policy


def write_records(run_dir: Path, run_id: str, tables: dict) -> None:
    rec_dir = run_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decisions", "steps"):
        rows = tables.get(name)
        if rows:
            with gzip.open(rec_dir / f"{run_id}.{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
    if "rollouts" in tables:
        np.savez_compressed(rec_dir / f"{run_id}.rollouts.npz", **tables["rollouts"])
    if "sequences" in tables:
        np.savez_compressed(rec_dir / f"{run_id}.sequences.npz", **tables["sequences"])


def execute_job(job: dict) -> dict:
    """Run one job and return its result row (top-level function, so that it can run in a worker)."""
    p = job["planned"]
    inst = load_orlib([p["instance"]], job["data_dir"])[p["instance"]]
    policy = build_policy(job)
    recorder = None
    if job["record_level"] != "none" or float(job.get("sequence_rate", 0.0)) > 0:
        recorder = RunRecorder(job["record_level"], job["step_rate"], seed=job["seed"] ^ 0x5EED,
                               sequence_rate=float(job.get("sequence_rate", 0.0)))
    b = job["budget"]
    budget = Budget(b["kind"], iterations=b["iterations"], seconds=b["seconds"],
                    profile=tuple(b["profile"]) if b["profile"] else None, label=b["label"])
    calls_before = getattr(getattr(policy, "infer", None), "calls", 0)
    started = time.time()
    res = solve(inst, policy, budget, seed=job["seed"], c=p["c"], sigma=p["sigma"], recorder=recorder,
                skip_forced=job["skip_forced"])
    row = {
        "experiment": job["experiment"], "run_id": job["run_id"], "key": job["key"],
        "instance": inst.name, "family": inst.family, "n": inst.n, "m": inst.m, "n_ops": inst.n_ops,
        "size_class": inst.size_class, "bks": inst.bks, "bks_optimal": inst.optimal_known, "lb0": inst.lb0,
        "config": p["name"], "variant": p["variant"], "label": job["label"], "level": p["level"],
        "tau": job["tau"], "model": p["model"],
        "train_size": "x".join(str(v) for v in p["calibration"]) if p["name"] in ("LR", "AZ") else None,
        "c": p["c"],
        "budget": b["label"], "budget_kind": b["kind"], "budget_iterations": b["iterations"],
        "budget_seconds": b["seconds"],
        "run": p["run"], "seed": job["seed"], "sigma": p["sigma"],
        "makespan": res.makespan, "gap": inst.gap(res.makespan), "ratio": res.makespan / inst.lb0,
        "best_rollout_makespan": res.best_rollout_makespan,
        "best_rollout_gap": inst.gap(res.best_rollout_makespan),
        "decisions": res.decisions, "iterations": res.iterations,
        "iterations_per_decision": res.iterations_per_decision,
        "time_search": res.time_search, "time_per_decision": res.time_per_decision,
        "time_total": res.time_total,
        "anytime": [[round(t, 5), m] for t, m in res.anytime],   # best makespan so far against time
        "model_steps": res.model_steps,
        "network_calls": getattr(getattr(policy, "infer", None), "calls", 0) - calls_before,
        "started": started, "host": platform.node(),
    }
    if job["store_sequence"]:
        row["sequence"] = res.sequence
    row.update(res.summary)
    if p["sigma"]:
        # RQ5: execute the finished schedule in the fixed scenarios of the instance and level
        final = replay(inst, res.sequence)
        realised = np.array([execute(inst, final.m_op, d) for d in scenarios(inst, p["sigma"], job["scenarios"])],
                            dtype=np.float64)
        mean, std = float(realised.mean()), float(realised.std(ddof=1)) if len(realised) > 1 else 0.0
        row.update(
            realised=[int(v) for v in realised],
            realised_mean=mean,
            realised_std=std,
            realised_gap=(mean - inst.bks) / inst.bks * 100.0,        # relative to the BKS without disturbances
            realised_std_pct=std / inst.bks * 100.0,
            realised_increase=(mean - res.makespan) / res.makespan * 100.0,   # over the undisturbed makespan
        )
    if recorder is not None and job.get("run_dir"):
        write_records(Path(job["run_dir"]), job["run_id"], recorder.tables())
    return row


# ------------------------------------------------------------------------------------------- experiment
def worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def environment_info() -> dict:
    import scipy
    import torch

    info = {
        "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
        "host": platform.node(), "cpu_count": mp.cpu_count(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch_geometric

        info["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=Path(__file__).resolve().parent)
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def run_directory(cfg: dict) -> Path:
    return Path(cfg["paths"]["results"]) / f"{cfg['experiment']}-{config_hash(cfg)}"


def read_results(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _signatures(jobs: list[dict]) -> dict:
    """What every configuration was resolved to; used to refuse mixing results of different artifacts."""
    out = {}
    cache: dict = {}
    for job in jobs:
        p = job["planned"]
        key = f"{job['label']}|{p['model'] or '-'}|{p['calibration']}"
        if key in out:
            continue
        path = job["model_path"]
        if path and path not in cache:
            cache[path] = file_fingerprint(path)
        out[key] = {"level": p["level"], "tau": job["tau"], "checkpoint": cache.get(path)}
    return out


def run_experiment(cfg: dict, workers: int = 1, limit: Optional[int] = None,
                   log: Callable[[str], None] = print, hours: Optional[float] = None) -> Path:
    """Run (or resume) the experiment defined by ``cfg``; returns the directory of its results.

    ``hours`` stops the run after that many hours of wall time: finished runs are kept, the runs in progress
    are dropped, and running the same command again continues with the remaining runs."""
    artifacts = Artifacts(cfg, log)
    planned = plan(cfg, artifacts)
    run_dir = run_directory(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("spawn").Pool(workers, initializer=worker_init) if workers > 1 else None
    try:
        time_seconds = ensure_time_budgets(cfg, planned, artifacts, pool, log)
        artifacts.calibrate_missing(temperature_requests(planned), pool)    # after the time budgets: no overlap
        jobs = resolve(cfg, planned, artifacts, time_seconds)
        for job in jobs:
            job["run_dir"] = str(run_dir)

        signatures = _signatures(jobs)
        resolved_path = run_dir / "resolved.json"
        done = {row["key"] for row in read_results(run_dir)}
        if resolved_path.exists() and done:
            with open(resolved_path, encoding="utf-8") as fh:
                previous = json.load(fh)
            changed = [k for k, v in signatures.items() if k in previous and previous[k] != v]
            if changed:
                raise RuntimeError(
                    f"{run_dir} holds results that were produced with other temperatures or checkpoints for "
                    f"{changed[:3]}{' ...' if len(changed) > 3 else ''}. Move the directory away or change "
                    f"the configuration before running again.")
        with open(resolved_path, "w", encoding="utf-8") as fh:
            json.dump(signatures, fh, indent=1, sort_keys=True)
        dump_config(cfg, run_dir / "config.yaml")
        with open(run_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump({**environment_info(), "workers": workers, "runs_planned": len(jobs)}, fh, indent=1)

        todo = [job for job in jobs if job["key"] not in done]
        if limit is not None:
            todo = todo[:limit]
        log(f"{cfg['experiment']}: {len(jobs)} runs planned, {len(done)} already done, {len(todo)} to run "
            f"-> {run_dir}")
        if not todo:
            return run_dir
        t0 = time.time()
        deadline = t0 + hours * 3600.0 if hours else None
        stopped = False
        results = pool.imap_unordered(execute_job, todo, chunksize=1) if pool is not None \
            else map(execute_job, todo)
        with open(run_dir / "results.jsonl", "a", encoding="utf-8") as fh:
            for i, row in enumerate(results, 1):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if i == len(todo) or i % max(1, len(todo) // 20) == 0:
                    elapsed = time.time() - t0
                    log(f"  {i}/{len(todo)} runs, {elapsed / 60:.1f} min elapsed, "
                        f"about {elapsed / i * (len(todo) - i) / 60:.1f} min left")
                if deadline is not None and time.time() >= deadline and i < len(todo):
                    stopped = True
                    log(f"  stopped after {hours:g} hours: {len(done) + i} of {len(jobs)} runs done; "
                        f"run the same command again to continue")
                    break
    finally:
        if pool is not None:
            if stopped:
                pool.terminate()
            else:
                pool.close()
            pool.join()
    return run_dir
