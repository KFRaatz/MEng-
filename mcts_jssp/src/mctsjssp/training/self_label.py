"""Self-labelling of the rollout policies (SLIM, Corsini et al., NeurIPS 2024).

Every iteration draws new random training instances.  On every instance the current policy samples
``samples`` complete schedules from its own softmax at ``temperature`` (the rollout distribution itself);
the schedule with the smallest makespan becomes the label, and the policy is fitted by cross-entropy to
that schedule's decisions (the labels of the last ``window`` iterations, ``epochs`` passes).  As the policy
improves, its samples and therefore its labels improve: the cross-entropy method with the rollout
distribution as the sampler.  No value network, tree search or reward model is involved.

The label compares schedules of the same instance, so the difficulty of an instance does not enter it, and
it is one complete schedule, not a distribution of visits.  The best network is the one whose rollouts are
best on the validation instances: by default the mean of its greedy schedule (entropy 0) and of its own
samples (temperature 1), the two ends of the range of entropies that the experiments use.
With disturbances (``sigma``, RQ5) all samples of an instance are executed in the same sampled scenario.

Start (``init``).  A policy that is close to uniform labels its instances with nearly random schedules, and
on 15 x 15 instances the networks learned too little from such labels.  With ``init: curriculum`` the
networks that do not depend on the size (the GIN of LR, fast LR) start from the best network of the next
smaller size (6 x 6 -> 10 x 10 -> 15 x 15; the 6 x 6 network starts from random weights), and a network for
disturbances starts from the deterministic network of the same kind and size.  Every network is still
trained by self-labelling only: no dispatching rule or other teacher is copied.

Sampling temperature.  The samples are drawn at temperature 1, as by Corsini et al.  An option that was
tested and is off by default (``temperature: auto``): fitted to labels that are still noisy, the policy stays
nearly uniform at temperature 1 (normalised entropy about 0.99 on 15 x 15 instances) even when its ranking
of the jobs is good; samples at temperature 1 are then nearly random schedules, and their best is no better
than the greedy schedule.  With ``auto``, once the greedy schedules of the network beat uniform random
dispatching on the validation instances, half of the samples are drawn at temperature 1 (exploration) and
half at the temperature at which the policy's mean normalised entropy on states of its own rollouts is
``target_entropy`` (0.5, the middle of the entropies of the experiments), calibrated every iteration; the
validation samples use that temperature too.  Before that, and whenever the greedy schedules are worse than
uniform, all samples use temperature 1.  In the tests the sharper samples gave better labels, but the
networks did not become better: the best sample was often close to the network's own greedy schedule, so the
network confirmed its current rule (GIN 6 x 6: greedy 1.46 after 15 iterations against 1.39 at temperature
1; fast LR 10 x 10: 1.73 against 1.51 after 6 iterations).

Job relabelling (``augment``).  The networks of AZ and fast AZ read the jobs in a fixed order and have an
output per job, so they must learn the same rule for every position.  Every epoch shows them every label
with the jobs renumbered at random (the same instance, the same decision), which teaches them that the
order of the jobs carries no information.

Teacher (``teacher``).  AZ cannot start from a smaller size, and from random weights its fully connected
networks did not learn on 10 x 10 and 15 x 15 instances (greedy schedules worse than uniform random dispatching
after 10-15 iterations, with and without job relabelling).  With ``teacher: {ffnn: lr}`` a network of AZ that
has no other start first copies the greedy schedules of the trained LR network of its size (``lr_<n>x<m>``,
``teacher_instances`` instances, ``teacher_epochs`` epochs), then continues by self-labelling; with
``teacher_keep`` the teacher's schedules are part of every later fit, beside the labels of the window (fitted to
the noisy labels of its own samples alone, AZ forgot what it had copied within a few iterations).  A string
applies to AZ and fast AZ alike.

Disturbances (RQ5).  A network for disturbed processing times (``sigma``) is validated under disturbances: its
greedy and sampled schedules of the validation instances are executed in ``validation_scenarios`` fixed
scenarios, and uniform random dispatching, the reference of the validation score, is executed in the same
scenarios (:mod:`mctsjssp.training.disturbed`).  After every iteration the history also records how its rollout
distribution changes on fixed probe states: entropy, agreement with MWKR and with the network for nominal
processing times of the same kind and size.  It starts from that network (fine-tuning), or, where no such
network exists (``paths.artifacts`` of the networks trained from scratch), like the nominal networks: LR from
the next smaller size for the same disturbances, AZ by copying the LR network of its size for the same
disturbances (``lr_<n>x<m>_s<sigma>``).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..distributions.learned import make_learned_rollout
from ..env.model import State
from ..env.observation import FeatureTracker
from ..instances.generator import training_instances, validation_set
from ..mcts.evaluation import make_return
from ..networks.checkpoint import build_model, load_checkpoint, save_checkpoint
from ..networks.inference import make_inference
from .imitation import imitate, imitate_fast
from .replay import Record
from . import disturbed as dist
from .trainer import TrainConfig, uniform_ratio

DEFAULTS = {
    "samples": 32,             # schedules sampled per training instance (Corsini et al.: robust from 32 to 256)
    "instances": 64,           # new training instances per iteration
    "epochs": 4,               # passes over the labels of the window after every iteration
    "window": 3,               # iterations whose labels are fitted
    "learning_rate": 1e-3,
    "temperature": 1.0,        # sampling temperature of the training rollouts (Corsini et al.: 1), or "auto"
    "target_entropy": 0.5,     # auto: the other samples use the temperature at which the policy has this entropy
    "explore": 0.5,            # auto: share of the samples drawn at temperature 1
    "iterations": 50,
    "patience": 10,            # stop after this many iterations without a better network ...
    "min_iterations": 20,      # ... but not before this many iterations (null patience: always all iterations)
    "select": "mean",          # best network by the mean of greedy and sampled ratios, or "sampled" / "greedy"
    "init": "curriculum",      # "curriculum", null (random weights) or the path of a checkpoint
    "augment": True,           # renumber the jobs at random in every epoch (AZ and fast AZ)
    "teacher": {"ffnn": "lr", "fast_az": "flr"},   # AZ first copies the greedy schedules of lr_<size>, fast AZ
    "teacher_instances": 500,  #   those of flr_<size>, on this many instances (None: from random weights)
    "teacher_epochs": 20,      #   for this many epochs, then self-labelling continues
    "teacher_keep": True,      # the teacher's schedules stay in every later fit, so they are not forgotten
    "validation_samples": 4,   # sampled rollouts per validation instance (the same seeds every iteration)
    "fit_threads": "auto",     # PyTorch threads while fitting (the sampling workers are idle then): auto =
                               #   one per physical core (half of the logical CPUs), or a number
    "validation_scenarios": dist.VALIDATION_SCENARIOS,   # networks for disturbances: fixed scenarios per
                               #   validation instance in which their schedules are executed (RQ5 uses 30)
}
SIZE_FREE = ("gin", "fast_lr")     # networks that work for every size
_NAME = re.compile(r"(.+)_(\d+)x(\d+)(.*)")

# ------------------------------------------------------------------------------------------ workers
_WORKER: dict = {}


def _init_worker() -> None:
    torch.set_num_threads(1)


def _sample_task(task: dict) -> list:
    """Sample schedules on a chunk of instances; returns (best Cmax/LB0, best sequence, mean, steps, extra) each.

    With ``scenarios`` (and ``sigma``) every schedule is executed in that many fixed scenarios and its mean
    realised Cmax/LB0 counts; ``extra`` then holds the mean nominal Cmax/LB0 of the schedules, the mean standard
    deviation of their realised values and, for one schedule per instance, the realised values themselves."""
    if _WORKER.get("key") != task["key"]:
        _WORKER.clear()
        _WORKER.update(key=task["key"], infer=make_inference(task["model"]))
    taus = task["tau"] if isinstance(task["tau"], (list, tuple)) else [task["tau"]] * task["samples"]
    policies = {t: make_learned_rollout("self_label", _WORKER["infer"], level=task["level"], tau=t) for t in set(taus)}
    count = int(task.get("scenarios") or 0) if task["sigma"] else 0
    out = []
    for inst, seed in zip(task["instances"], task["seeds"]):
        rnd = random.Random(seed)
        best, best_seq, total = math.inf, None, 0.0
        nominal, spread, values = 0.0, 0.0, None
        for tau in taus:
            policy = policies[tau]
            trace: list = []
            state = State(inst)
            policy.rollout(state, rnd, trace)
            if count:                                       # validation: the fixed scenarios of the instance
                real = dist.realised(inst, state, task["sigma"], count)
                cmax = float(real.mean()) * inst.lb0
                nominal += state.makespan / inst.lb0
                spread += float(real.std(ddof=1)) if count > 1 else 0.0
                if values is None:
                    values = [float(v) for v in real]
            else:
                cmax = (make_return(inst, task["sigma"], seed).makespan(state) if task["sigma"]
                        else state.makespan)                # same seed: the same scenario for every sample
            total += cmax
            if cmax < best:
                best, best_seq = cmax, trace
        k = len(taus)
        extra = {"nominal": nominal / k, "std": spread / k, "values": values} if count else None
        out.append((best / inst.lb0, best_seq if task["keep"] else None, total / task["samples"] / inst.lb0,
                    task["samples"] * inst.n_ops, extra))
    return out


class Sampler:
    """Sampling in worker processes (or in this process with one worker)."""

    def __init__(self, workers: int):
        self.workers = max(1, int(workers or 1))
        self.pool = None
        if self.workers > 1:
            self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker)

    def run(self, model, key, instances, seeds, samples, level, tau, sigma, keep, scenarios: int = 0) -> list:
        chunks = np.array_split(np.arange(len(instances)), min(self.workers, len(instances)))
        tasks = [{"key": key, "model": model, "instances": [instances[i] for i in c],
                  "seeds": [int(seeds[i]) for i in c], "samples": samples, "level": level, "tau": tau,
                  "sigma": sigma, "keep": keep, "scenarios": scenarios} for c in chunks if len(c)]
        if self.pool is None:
            _WORKER.clear()
            return [r for t in tasks for r in _sample_task(t)]
        return [r for part in self.pool.map(_sample_task, tasks) for r in part]

    def baselines(self, instances, seeds, sigma, scenarios: int, repeats: int = 10) -> list:
        """Uniform random dispatching and greedy MWKR of every instance in the fixed scenarios."""
        chunks = np.array_split(np.arange(len(instances)), min(self.workers, len(instances)))
        tasks = [{"instances": [instances[i] for i in c], "seeds": [int(seeds[i]) for i in c], "sigma": sigma,
                  "scenarios": scenarios, "repeats": repeats} for c in chunks if len(c)]
        if self.pool is None:
            return [r for t in tasks for r in dist.baseline_task(t)]
        return [r for part in self.pool.map(dist.baseline_task, tasks) for r in part]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ------------------------------------------------------------------------------------------ records
def label_records(inst, sequence, fast: bool):
    """The decisions of a schedule as training records (one-hot targets over the jobs)."""
    from ..networks.fast import state_features

    n = inst.n
    state = State(inst)
    tracker = None if fast else FeatureTracker(inst).reset(state)
    out = []
    for job in sequence:
        if len(state.active) > 1:
            mask = np.zeros(n, dtype=bool)
            mask[state.active] = True
            target = np.zeros(n, dtype=np.float32)
            target[job] = 1.0
            if fast:
                out.append((state_features(state), mask, target))
            else:
                x, mpred = tracker.snapshot()
                out.append(Record(x, mpred, np.array(state.next_op, dtype=np.int16), mask, target, 0.0))
        state.step(job)
        if tracker is not None and state.active:
            tracker.update(state)
    return out


# ------------------------------------------------------------------------------------------ trainer
class SelfLabelTrainer:
    def __init__(self, cfg: TrainConfig, out_dir: str | Path, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.sl = {**DEFAULTS, **(cfg.self_label or {})}
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg.kind, cfg.network_hparams())
        self.model.eval()
        self.fast = bool(getattr(self.model, "uses_features", False))
        self.validation = list(validation_set(cfg.n, cfg.m)[: cfg.validation_instances])
        self.uniform_score = uniform_ratio(self.validation, seed=cfg.seed, repeats=10)   # stable reference
        self.history: list[dict] = []
        self.window: list[list] = []
        self.model_steps = 0
        self.best_score = float("inf")
        self.best_generation = -1
        self.best_scores: dict = {}
        self.initial: dict | None = None
        self.teacher_path: Path | None = None
        self.teacher_records: list = []
        self.sharpen = False                   # auto: sharpened samples once the greedy schedules beat uniform
        self.disturbed = bool(cfg.sigma)
        self.scenarios = int(self.sl.get("validation_scenarios") or dist.VALIDATION_SCENARIOS)
        self.baseline: dict = {}
        self._val_extra: dict = {}
        self.init_from = self._initialise()
        self.probes = self.probe_mwkr = self.probe_reference = None
        self.reference_path = None
        if self.disturbed:                     # how the rollout distribution changes, on fixed probe states
            self.probes = dist.probe_states(self.validation, int(cfg.seed) % (2 ** 31))
            self.probe_mwkr = dist.mwkr_choices(self.probes)
            me = _NAME.fullmatch(cfg.name)
            ref_dir = Path(cfg.reference_models) if cfg.reference_models else self.out_dir.parent
            if me is not None:
                path = ref_dir / f"{me.group(1)}_{me.group(2)}x{me.group(3)}" / "best.pt"
                if path.exists():
                    ref, _ = load_checkpoint(path)
                    self.reference_path = str(path)
                    self.probe_reference = dist.choices(dist.network_logits(make_inference(ref), self.probes))

    # ------------------------------------------------------------------------------------- start
    def _curriculum_source(self) -> Path | None:
        """The trained network to start from: the deterministic twin of a network for disturbances, else the
        largest smaller size of the same model family (only for networks that work for every size)."""
        me = _NAME.fullmatch(self.cfg.name)
        models = self.out_dir.parent
        if me is None or not models.is_dir():
            return None
        prefix, suffix = me.group(1), me.group(4)
        best = None
        for d in models.iterdir():
            other = _NAME.fullmatch(d.name)
            if other is None or other.group(1) != prefix or d == self.out_dir:
                continue
            if not ((d / "best.pt").exists() and (d / "summary.json").exists()):
                continue                                           # not (completely) trained
            size = (int(other.group(2)), int(other.group(3)))
            if suffix and not other.group(4) and size == (self.cfg.n, self.cfg.m):
                rank = (2, 0)                                      # deterministic twin
            elif other.group(4) == suffix and size[0] * size[1] < self.cfg.n * self.cfg.m \
                    and self.cfg.kind in SIZE_FREE:
                rank = (1, size[0] * size[1])                      # the largest smaller size
            else:
                continue
            if best is None or rank > best[0]:
                best = (rank, d / "best.pt")
        return None if best is None else best[1]

    def _initialise(self) -> str | None:
        init = self.sl.get("init")
        path = (self._curriculum_source() if init == "curriculum" else Path(init)) if init else None
        if path is None:
            teacher = self.sl.get("teacher")
            if isinstance(teacher, dict):
                teacher = teacher.get(self.cfg.kind)
            me = _NAME.fullmatch(self.cfg.name)
            if teacher and self.cfg.kind not in SIZE_FREE and me is not None:
                base = self.out_dir.parent / f"{teacher}_{me.group(2)}x{me.group(3)}"
                same = Path(f"{base}{me.group(4)}")                 # the teacher for the same disturbances
                self.teacher_path = (same if me.group(4) and (same / "best.pt").exists() else base) / "best.pt"
                if not self.teacher_path.exists():
                    raise SystemExit(f"[{self.cfg.name}] the teacher {self.teacher_path} has not been trained; train "
                                     f"it first (it must come before {self.cfg.name} under 'models')")
                if self.cfg.verbose >= 1:
                    self.log(f"[{self.cfg.name}] copies the greedy schedules of {self.teacher_path.parent.name} first")
                return str(self.teacher_path)
            if self.cfg.verbose >= 1:
                self.log(f"[{self.cfg.name}] starts from random weights")
            return None
        try:
            source, meta = load_checkpoint(path)
        except (OSError, RuntimeError, KeyError) as exc:
            raise SystemExit(f"[{self.cfg.name}] cannot start from {path}: {exc}")
        if source.kind != self.model.kind or dict(source.hparams) != dict(self.model.hparams):
            raise SystemExit(f"[{self.cfg.name}] cannot start from {path}: {source.kind} {source.hparams} differs "
                             f"from {self.model.kind} {self.model.hparams}")
        if init == "curriculum" and meta.get("method") != "self_label":
            raise SystemExit(f"[{self.cfg.name}] {path} was not trained by self-labelling; move the old networks "
                             f"aside or set training.self_label.init")
        self.model.load_state_dict(source.state_dict())
        self.model.eval()
        if self.cfg.verbose >= 1:
            self.log(f"[{self.cfg.name}] starts from {path.parent.name} ({path})")
        return str(path)

    def _meta(self, iteration: int, score: float) -> dict:
        return {"name": self.cfg.name, "generation": iteration, "validation_score": score,
                "uniform_score": self.uniform_score, "model_steps": self.model_steps, "method": "self_label",
                "self_label": self.sl, "config": asdict(self.cfg)}

    def _seeds(self, iteration: int, count: int, stream: int) -> np.ndarray:
        ss = np.random.SeedSequence([self.cfg.seed, stream, iteration])
        return ss.generate_state(count, dtype=np.uint32)

    def _fit_relabelled(self, data: list) -> list[float]:
        """Cross-entropy fit of AZ or fast AZ; every epoch renumbers the jobs of every record at random."""
        cfg, sl = self.cfg, self.sl
        n = cfg.n
        if self.fast:
            x = torch.from_numpy(np.stack([r[0] for r in data]))                       # (N, n, F)
            masks = torch.from_numpy(np.stack([r[1] for r in data]))
            targets = torch.from_numpy(np.stack([r[2] for r in data]))
        else:
            x = torch.from_numpy(np.stack([r.x for r in data]).reshape(len(data), n, cfg.m, -1))   # (N, n, m, 2)
            masks = torch.from_numpy(np.stack([r.mask for r in data]))
            targets = torch.from_numpy(np.stack([r.pi for r in data]))
        params = self.model.policy_parameters()
        optimizer = torch.optim.Adam(params, lr=float(sl["learning_rate"]), weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(int(cfg.seed) * 1000 + len(self.history))
        losses = []
        for _ in range(int(sl["epochs"])):
            self.model.train()
            order = torch.randperm(len(data), generator=gen)
            total, batches = 0.0, 0
            for start in range(0, len(data), cfg.batch_size):
                idx = order[start: start + cfg.batch_size]
                perm = torch.argsort(torch.rand(len(idx), n, generator=gen), dim=1)     # new job i = old perm[i]
                rows = torch.arange(len(idx)).unsqueeze(1)
                xb, mb, tb = x[idx][rows, perm], masks[idx][rows, perm], targets[idx][rows, perm]
                if self.fast:
                    logits = self.model.forward_features(xb, mb)
                else:
                    logits, _ = self.model.forward_batch(xb.reshape(len(idx), n * cfg.m, -1), None, None, mb)
                logp = torch.where(mb, torch.log_softmax(logits, dim=1), torch.zeros_like(logits))
                loss = -(tb * logp).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            losses.append(total / max(batches, 1))
        self.model.eval()
        return losses

    def _fit(self) -> float:
        cfg, sl = self.cfg, self.sl
        from .trainer import fit_thread_count

        threads = torch.get_num_threads()
        torch.set_num_threads(fit_thread_count(sl.get("fit_threads")))
        try:
            data = [r for part in self.window for r in part] + self.teacher_records
            if sl.get("augment") and self.cfg.kind not in SIZE_FREE:
                losses = self._fit_relabelled(data)
            elif self.fast:
                f, m, t = (np.stack([r[i] for r in data]) for i in range(3))
                losses = imitate_fast(self.model, f, m, t, epochs=int(sl["epochs"]),
                                      learning_rate=float(sl["learning_rate"]), batch_size=cfg.batch_size,
                                      weight_decay=cfg.weight_decay, seed=cfg.seed + len(self.history))
            else:
                losses = imitate(self.model, data, cfg.n, cfg.m, epochs=int(sl["epochs"]),
                                 learning_rate=float(sl["learning_rate"]), batch_size=cfg.batch_size,
                                 weight_decay=cfg.weight_decay, seed=cfg.seed + len(self.history))
        finally:
            torch.set_num_threads(threads)
        self.model.eval()
        return float(losses[-1])

    def _logits(self, records: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits (finished jobs at -1e9) and masks of the policy on up to 512 of the recorded states."""
        sample = records[:: max(1, len(records) // 512)][:512]
        with torch.no_grad():
            if self.fast:
                f = torch.from_numpy(np.stack([r[0] for r in sample]))
                masks = torch.from_numpy(np.stack([r[1] for r in sample]))
                logits = self.model.forward_features(f, masks)
            else:
                x = torch.from_numpy(np.stack([r.x for r in sample]))
                mp_ = torch.from_numpy(np.stack([r.mpred for r in sample]).astype(np.int64))
                nx = torch.from_numpy(np.stack([r.next_op for r in sample]).astype(np.int64))
                masks = torch.from_numpy(np.stack([r.mask for r in sample]))
                logits, _ = self.model.forward_batch(x, mp_, nx, masks)
        return logits.masked_fill(~masks, -1e9), masks

    @staticmethod
    def _entropy(logits: torch.Tensor, masks: torch.Tensor, tau: float) -> float:
        """Mean normalised entropy (1 = uniform) of softmax(logits / tau) over states with 2+ jobs."""
        keep = masks.sum(dim=1) > 1
        logits, masks = logits[keep], masks[keep]
        if not len(logits):
            return float("nan")
        logp = torch.log_softmax(logits / tau, dim=1)
        h = -(logp.exp() * torch.where(masks, logp, torch.zeros_like(logp))).sum(dim=1)
        return float((h / torch.log(masks.sum(dim=1).double()).float()).mean())

    def _policy_entropy(self, records: list) -> float:
        """Mean normalised entropy of the policy (temperature 1) on up to 512 labelled states."""
        if not records:
            return float("nan")
        return self._entropy(*self._logits(records), 1.0)

    def _calibrate(self, instances: list) -> float:
        """Temperature at which the policy's mean normalised entropy on states of its own rollouts (temperature 1,
        on up to 4 of this iteration's training instances) is ``target_entropy``."""
        target = float(self.sl["target_entropy"])
        policy = make_learned_rollout("calibrate", make_inference(self.model), level=None, tau=1.0)
        records = []
        for i, inst in enumerate(instances[:4]):
            trace: list = []
            policy.rollout(State(inst), random.Random(self.cfg.seed * 7919 + len(self.history) * 31 + i), trace)
            records += label_records(inst, trace, self.fast)
        logits, masks = self._logits(records)
        lo, hi = math.log(1e-4), math.log(1e3)
        if self._entropy(logits, masks, math.exp(hi)) <= target:          # flat already at a high temperature
            return math.exp(hi)
        for _ in range(40):                                                # entropy grows with the temperature
            mid = (lo + hi) / 2
            if self._entropy(logits, masks, math.exp(mid)) < target:
                lo = mid
            else:
                hi = mid
        return math.exp((lo + hi) / 2)

    def _tau(self, instances: list) -> float:
        """The sharpened temperature (auto, once the network beats uniform), else the fixed one (auto: 1)."""
        temperature = self.sl["temperature"]
        if temperature != "auto":
            return float(temperature)
        return self._calibrate(instances) if self.sharpen else 1.0

    def _temperatures(self, instances: list) -> tuple[float, list]:
        """The sharpened temperature and the temperatures of the training samples."""
        k = int(self.sl["samples"])
        tau = self._tau(instances)
        if self.sl["temperature"] != "auto" or not self.sharpen:
            return tau, [tau] * k
        explore = int(round(k * float(self.sl["explore"])))
        return tau, [1.0] * explore + [tau] * (k - explore)

    def _probe(self) -> dict:
        """Entropy and agreements of the current policy on the probe states (networks for disturbances)."""
        if self.probes is None:
            return {}
        return dist.probe_metrics(dist.network_logits(make_inference(self.model), self.probes), self.probe_mwkr,
                                  self.probe_reference)

    def _baselines(self, sampler: Sampler) -> None:
        """Uniform random dispatching and greedy MWKR in the fixed scenarios: the references of a network for
        disturbances (uniform random dispatching replaces the nominal one as the reference of its score)."""
        seeds = self._seeds(0, len(self.validation), 11)
        rows = sampler.baselines(self.validation, seeds, self.cfg.sigma, self.scenarios)
        self.baseline = {"uniform_realised": float(np.mean([r[0] for r in rows])),
                         "mwkr_realised": float(np.mean([r[1] for r in rows])),
                         "uniform_nominal": float(np.mean([r[2] for r in rows])),
                         "mwkr_nominal": float(np.mean([r[3] for r in rows]))}
        self.uniform_score = self.baseline["uniform_realised"]
        if self.cfg.verbose >= 1:
            self.log(f"[{self.cfg.name}] validation in {self.scenarios} fixed scenarios (sigma {self.cfg.sigma:g}): "
                     f"uniform {self.baseline['uniform_realised']:.4f}, MWKR {self.baseline['mwkr_realised']:.4f}"
                     f" (without disturbances {self.baseline['uniform_nominal']:.4f} and "
                     f"{self.baseline['mwkr_nominal']:.4f})")

    def _validate(self, sampler: Sampler, iteration: int, instances: list) -> tuple[float, float, float]:
        """Greedy and sampled Cmax/LB0 on the validation instances, and the temperature of the samples.  The
        greedy result decides whether the samples are sharpened (auto)."""
        key = ("val", iteration)
        seeds = self._seeds(0, len(self.validation), 7)            # the same rollouts' seeds every iteration
        sigma = self.cfg.sigma if self.disturbed else None
        count = self.scenarios if self.disturbed else 0
        greedy_runs = sampler.run(self.model, key, self.validation, seeds, 1, 0.0, None, sigma, False, count)
        greedy = float(np.mean([r[2] for r in greedy_runs]))
        self.sharpen = greedy < self.uniform_score
        tau = self._tau(instances)
        sampled_runs = sampler.run(self.model, key, self.validation, seeds, int(self.sl["validation_samples"]),
                                   None, float(tau), sigma, False, count)
        sampled = float(np.mean([r[2] for r in sampled_runs]))
        self._val_extra = {}
        if self.disturbed:                     # realised values of the greedy schedules in all scenarios
            values = np.concatenate([r[4]["values"] for r in greedy_runs])
            self._val_extra = {
                "greedy_nominal": float(np.mean([r[4]["nominal"] for r in greedy_runs])),
                "sampled_nominal": float(np.mean([r[4]["nominal"] for r in sampled_runs])),
                "greedy_std": float(np.mean([r[4]["std"] for r in greedy_runs])),
                "greedy_p10": float(np.percentile(values, 10)), "greedy_p50": float(np.percentile(values, 50)),
                "greedy_p90": float(np.percentile(values, 90)),
                **self._probe()}
        return sampled, greedy, tau

    def _copy_teacher(self, sampler: Sampler) -> None:
        """Fit the policy to the greedy schedules of the teacher network (``teacher_epochs`` epochs)."""
        cfg, sl = self.cfg, self.sl
        teacher, _ = load_checkpoint(self.teacher_path)
        instances = training_instances(cfg.n, cfg.m, int(sl["teacher_instances"]), cfg.seed + 7777, 0)
        runs = sampler.run(teacher, ("teacher",), instances, self._seeds(0, len(instances), 9), 1, 0.0, None,
                           cfg.sigma, True)
        records = [rec for inst, r in zip(instances, runs) for rec in label_records(inst, r[1], self.fast)]
        self.window, self.teacher_records = [], records
        epochs = sl["epochs"]
        sl["epochs"] = int(sl["teacher_epochs"])
        try:
            loss = self._fit()
        finally:
            sl["epochs"] = epochs
        if not sl.get("teacher_keep", True):
            self.teacher_records = []
        if cfg.verbose >= 1:
            self.log(f"[{cfg.name}] copied {self.teacher_path.parent.name} (greedy Cmax/LB0 "
                     f"{np.mean([r[0] for r in runs]):.4f}) from {len(records)} decisions, loss {loss:.4f}")

    def _score(self, sampled: float, greedy: float) -> float:
        select = self.sl["select"]
        if select not in ("mean", "sampled", "greedy"):
            raise SystemExit(f"training.self_label.select must be mean, sampled or greedy, not {select!r}")
        return {"mean": (sampled + greedy) / 2, "sampled": sampled, "greedy": greedy}[select]

    def run(self) -> dict:
        cfg, sl = self.cfg, self.sl
        sampler = Sampler(cfg.workers)
        stale, stop = 0, "iterations"
        patience = int(sl["patience"]) if sl.get("patience") else None
        started = time.perf_counter()
        try:
            if self.disturbed:
                self._baselines(sampler)
            if self.teacher_path is not None:
                self._copy_teacher(sampler)
            if self.init_from:                         # the network it starts from competes for best.pt
                sampled, greedy, tau = self._validate(sampler, -1, training_instances(cfg.n, cfg.m, 4, cfg.seed, 0))
                score = self._score(sampled, greedy)
                self.initial = {"from": self.init_from, "sampled": sampled, "greedy": greedy, "score": score,
                                "temperature": tau, **self._val_extra}
                self.best_score, self.best_generation = score, -1
                self.best_scores = {"sampled": sampled, "greedy": greedy}
                save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(-1, score))
                if cfg.verbose >= 1:
                    self.log(f"[{cfg.name}] start      | val sampled {sampled:.4f}  greedy {greedy:.4f}  (uniform "
                             f"{self.uniform_score:.4f}) tau {tau:.3g}")
            for iteration in range(int(sl["iterations"])):
                t0 = time.perf_counter()
                instances = training_instances(cfg.n, cfg.m, int(sl["instances"]), cfg.seed, iteration)
                train_tau, taus = self._temperatures(instances)
                labels = sampler.run(self.model, ("train", iteration), instances,
                                     self._seeds(iteration, len(instances), 5), int(sl["samples"]), None,
                                     taus, cfg.sigma, True)
                self.model_steps += int(sum(r[3] for r in labels))
                records = [rec for inst, r in zip(instances, labels) for rec in label_records(inst, r[1], self.fast)]
                self.window.append(records)
                self.window = self.window[-int(sl["window"]):]
                loss = self._fit()
                sampled, greedy, tau = self._validate(sampler, iteration, instances)
                score = self._score(sampled, greedy)
                row = {
                    "generation": iteration, "search": "SL",
                    "selfplay_ratio": float(np.mean([r[2] for r in labels])),     # the policy's own samples
                    "label_ratio": float(np.mean([r[0] for r in labels])),        # best of the samples
                    "validation_ratio": score, "sampled_ratio": sampled, "greedy_ratio": greedy, "temperature": tau,
                    "train_temperature": train_tau,
                    "uniform_ratio": self.uniform_score, "policy_loss": loss, "value_loss": float("nan"),
                    "policy_entropy": self._policy_entropy(records), "records": len(records),
                    "model_steps": self.model_steps,
                    "best_ratio": min(self.best_score, score), "stale": 0 if score < self.best_score else stale + 1,
                    "elapsed_minutes": (time.perf_counter() - started) / 60,
                    "seconds": time.perf_counter() - t0,
                    **self._val_extra,
                }
                self.history.append(row)
                better = score < self.best_score
                if cfg.verbose >= 1:
                    self.log(f"[{cfg.name}] iter {iteration:3d} SL samples {row['selfplay_ratio']:.4f} best-of-"
                             f"{sl['samples']} {row['label_ratio']:.4f} | val sampled {sampled:.4f}  greedy "
                             f"{greedy:.4f}{'*' if better else ' '} (uniform {self.uniform_score:.4f}) tau {tau:.3g} "
                             f"loss p {loss:.4f} {row['seconds']:.1f}s")
                if cfg.verbose >= 2:
                    self.log(f"           network: entropy {row['policy_entropy']:.3f} | records {len(records)} "
                             f"| steps {self.model_steps:.3g} | no improvement {row['stale']}/"
                             f"{patience or '-'} | {row['elapsed_minutes']:.1f} min")
                if cfg.verbose >= 2 and self.disturbed:
                    self.log(f"           disturbed: greedy without disturbances {row['greedy_nominal']:.4f} | "
                             f"spread {row['greedy_std']:.4f} | probes: entropy {row['probe_entropy']:.3f}, "
                             f"same job as MWKR {row['agree_mwkr']:.1%}, as the nominal network "
                             f"{row['agree_reference']:.1%}")
                save_checkpoint(self.out_dir / "last.pt", self.model, self._meta(iteration, score))
                if better:
                    self.best_score, self.best_generation, stale = score, iteration, 0
                    self.best_scores = {"sampled": sampled, "greedy": greedy}
                    save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(iteration, score))
                else:
                    stale += 1
                with open(self.out_dir / "history.json", "w", encoding="utf-8") as fh:
                    json.dump(self.history, fh, indent=1)
                if patience and stale >= patience and iteration + 1 >= int(sl.get("min_iterations") or 0):
                    stop = "no_improvement"
                    break
                if cfg.max_hours and time.perf_counter() - started >= cfg.max_hours * 3600:
                    stop = "time"
                    break
        finally:
            sampler.close()
        summary = {
            "name": cfg.name, "kind": cfg.kind, "size": f"{cfg.n}x{cfg.m}", "method": "self_label",
            "stopped_by": stop, "generations": len(self.history), "best_generation": self.best_generation,
            "best_validation_ratio": self.best_score, "uniform_ratio": self.uniform_score,
            "best_sampled_ratio": self.best_scores.get("sampled"), "best_greedy_ratio": self.best_scores.get("greedy"),
            "initial": self.initial,
            "model_steps": self.model_steps, "checkpoint": str(self.out_dir / "best.pt"),
            "self_label": sl, "config": asdict(cfg),
            "validation": "disturbed" if self.disturbed else "nominal",
            "sigma": cfg.sigma, "scenarios": self.scenarios if self.disturbed else 0, "baseline": self.baseline,
            "reference": self.reference_path,
        }
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        return summary
