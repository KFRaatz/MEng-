"""Self-play training of the networks of LR and AZ (Sections 3.6.1 and 3.6.2, Table 3.4).

One generation

1. plays 40 episodes on newly generated instances (generation 0 with the searches of U, later generations
   with PUCT and the networks of the previous generation);
2. labels their records with the ranked reward and adds them to the replay buffer (36 000 records, oldest
   removed first);
3. fits the networks for one epoch over the buffer in mini-batches of 256 records: cross-entropy to the visit
   distribution ``pi_t`` for the policy and squared error to ``z`` for the value, with equal weights.  In LR
   the two heads share the GIN, which is therefore trained by both losses; AZ has separate networks;
4. scores the policy network on the validation instances by the mean ``Cmax / LB0`` that it achieves when it
   dispatches on its own, choosing greedily.

Training stops at the first of: no improvement of the validation score for ``patience`` generations (from
``min_generations`` on; ``patience`` null switches it off), the training budget in steps of the dynamics model,
``max_hours`` of wall-clock time, or ``max_generations``.  The best checkpoint is kept.

Warm start (optional, ``warm_start``): before generation 0 the policy network learns to copy a dispatching
rule (:mod:`mctsjssp.training.imitation`).  Generation 0 then already searches with PUCT and the networks,
the results of the rule seed the window of the ranked reward, and the copied network is the first candidate
for the best checkpoint, so that self-play has to improve on it.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..distributions.fixed import UniformRollout
from ..distributions.learned import make_learned_rollout
from ..env.model import State
from ..instances.generator import training_instances, validation_set
from ..instances.instance import Instance
from ..networks.checkpoint import build_model, save_checkpoint
from ..networks.inference import make_inference
from ..env.rules import RULES, run_rule
from .imitation import imitate, imitate_fast, rule_episodes, rule_feature_records
from .ranked_reward import RankedReward
from .replay import Record, ReplayBuffer
from .selfplay import EpisodeRunner


@dataclass
class TrainConfig:
    """Settings of one training run; the defaults are the values of Table 3.4."""

    name: str = "model"
    kind: str = "gin"                         # "gin" (LR) or "ffnn" (AZ)
    n: int = 6
    m: int = 6
    hparams: dict = field(default_factory=dict)
    # self-play
    episodes_per_generation: int = 40
    iterations: int = 100                     # search iterations per decision
    c: float = 1.0                            # exploration constant of UCT and PUCT
    noise_weight: float = 0.25                # Dirichlet noise at the root
    noise_alpha_scale: float = 10.0           # alpha = 10 / |A(s)|
    q_init: str = "zero"                      # value of unvisited children in PUCT: "zero" or "parent"
    percentile: float = 75.0                  # ranked reward (tuned, Table 3.5)
    window: int = 250
    rank_against_generation0: bool = False    # keep the results of U (generation 0) in the ranking window
    # training searches after Kemmerling et al. [126]: "outcome" = the value network learns the ranked reward
    # of the episode; "search" = it learns the value of the root of every search (V_MCTS, on the scale of
    # search_value, 3 - 2 Cmax/LB0), and complete schedules in the tree are valued on that scale
    value_target: str = "outcome"
    leaf_rollouts: bool = False               # complete every new leaf by a uniform random rollout ...
    rollout_mix: float = 0.5                  # ... whose outcome has this weight next to the value network
    normalize_q: bool = False                 # PUCT compares Q rescaled to [0, 1] within each search (MuZero)
    # training method: "self_label" (self_label.py; the method of the study) or "alphazero" (PUCT self-play,
    # this module; run 1, kept so that it can be reproduced)
    method: str = "alphazero"
    self_label: Optional[dict] = None         # settings of self-labelling, see self_label.DEFAULTS
    sigma: Optional[float] = None             # disturbances during self-play (RQ5)
    # warm start: {"rule": "FIFO", "instances": 300, "epochs": 8, "learning_rate": 1e-3, "non_delay": False};
    # a nested dict per network kind overrides these values for that kind, e.g. "ffnn": {"epochs": 60}
    warm_start: Optional[dict] = None
    # fitting
    buffer_size: int = 36_000
    batch_size: int = 256
    learning_rate: float = 5e-5               # (tuned, Table 3.5)
    weight_decay: float = 5e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    # stopping
    patience: Optional[int] = 5               # None or 0: no early stopping
    min_generations: int = 0                  # early stopping is not applied before this generation
    max_hours: Optional[float] = None         # wall-clock limit of the training of one network (hours)
    verbose: int = 1                          # 0 = summary only, 1 = a line per generation, 2 = diagnostics
    max_generations: int = 100
    budget_model_steps: Optional[int] = None  # training budget in steps of the dynamics model
    # validation
    validation_instances: int = 100
    # technical
    seed: int = 0
    workers: int = 1
    fit_threads: int | str = "auto"          # PyTorch threads while fitting: auto = one per physical core
    reference_models: str | None = None       # networks for disturbances: where the nominal networks are
    inference_backend: str = "numpy"

    def network_hparams(self) -> dict:
        hp = dict(self.hparams)
        if self.kind in ("ffnn", "fast_az"):
            hp.setdefault("n", self.n)
            hp.setdefault("m", self.m)
        return hp


def greedy_ratio(infer, instances: Sequence[Instance], name: str = "net") -> float:
    """Mean ``Cmax / LB0`` of the policy network dispatching on its own, choosing greedily."""
    policy = make_learned_rollout(name, infer, level=0.0)
    rnd = random.Random(0)
    return float(np.mean([policy.rollout(State(inst), rnd, []) / inst.lb0 for inst in instances]))


def uniform_ratio(instances: Sequence[Instance], seed: int = 0, repeats: int = 1) -> float:
    """Mean ``Cmax / LB0`` of uniform random dispatching (reference of the third check of Section 3.7)."""
    policy = UniformRollout()
    rnd = random.Random(seed)
    return float(np.mean([policy.rollout(State(inst), rnd, []) / inst.lb0
                          for inst in instances for _ in range(repeats)]))


def _norm_entropy(p: np.ndarray, mask: np.ndarray) -> float:
    """Entropy of a distribution over the legal actions divided by ln k (1 = uniform, 0 = deterministic)."""
    k = int(mask.sum())
    if k <= 1:
        return np.nan
    q = np.asarray(p, dtype=np.float64)[mask]
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(k))


def fit_thread_count(setting=None) -> int:
    """PyTorch threads while a network is fitted.  The worker processes wait meanwhile, so ``auto`` (or null)
    uses one thread per physical core (half of the logical CPUs); a number is used as given, at most the
    logical CPUs."""
    cpus = os.cpu_count() or 1
    if setting is None or str(setting).lower() == "auto":
        return max(1, cpus // 2)
    return max(1, min(int(setting), cpus))


class Trainer:
    def __init__(self, cfg: TrainConfig, out_dir: str | Path, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg.kind, cfg.network_hparams())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.n, cfg.m)
        self.ranked = RankedReward(cfg.percentile, cfg.window)
        self.validation = list(validation_set(cfg.n, cfg.m)[: cfg.validation_instances])
        self.rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 31]))
        self.history: list[dict] = []
        self.model_steps = 0
        self.best_score = float("inf")
        self.best_generation = -1
        self.uniform_score = uniform_ratio(self.validation, seed=cfg.seed)
        self.warm_start_summary: Optional[dict] = None

    # ------------------------------------------------------------------------------------- warm start
    def _warm_start(self) -> None:
        """Train the policy network to copy a dispatching rule before self-play."""
        cfg = self.cfg
        ws = dict(cfg.warm_start or {})
        per_kind = {k: ws.pop(k) for k in list(ws) if isinstance(ws[k], dict)}   # e.g. ffnn: {epochs: 60}
        ws.update(per_kind.get(cfg.kind) or {})
        rule = str(ws.get("rule", "FIFO"))
        non_delay = bool(ws.get("non_delay", False))
        t0 = time.perf_counter()
        count, epochs = int(ws.get("instances", 300)), int(ws.get("epochs", 8))
        lr = float(ws.get("learning_rate", 1e-3))
        if getattr(self.model, "uses_features", False):          # fast networks: on their own features
            feats, masks, targets, outcomes = rule_feature_records(rule, cfg.n, cfg.m, count, cfg.seed, non_delay)
            losses = imitate_fast(self.model, feats, masks, targets, epochs=epochs, learning_rate=lr,
                                  batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, seed=cfg.seed)
            records = feats
        else:
            records, outcomes = rule_episodes(rule, cfg.n, cfg.m, count, cfg.seed, non_delay)
            losses = imitate(self.model, records, cfg.n, cfg.m, epochs=epochs, learning_rate=lr,
                             batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, seed=cfg.seed)
        self.ranked.update(outcomes)                 # the first episodes of self-play must beat the rule
        score = self.validate()
        rule_score = float(np.mean([run_rule(inst, RULES[rule], non_delay).makespan / inst.lb0
                                    for inst in self.validation]))
        self.warm_start_summary = {
            "rule": rule, "non_delay": non_delay, "records": len(records), "losses": losses,
            "validation_ratio": score, "rule_validation_ratio": rule_score,
            "seconds": time.perf_counter() - t0,
        }
        self.best_score, self.best_generation = score, -1
        save_checkpoint(self.out_dir / "warm_start.pt", self.model, self._meta(-1, score))
        save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(-1, score))
        self.log(f"[{cfg.name}] warm start: copied {rule} from {len(records)} decisions, loss {losses[0]:.3f} -> "
                 f"{losses[-1]:.3f}; greedy val {score:.4f} ({rule} itself {rule_score:.4f}, uniform "
                 f"{self.uniform_score:.4f})")

    # ------------------------------------------------------------------------------------ one generation
    def _play(self, runner: EpisodeRunner, generation: int) -> tuple[list, Optional[float]]:
        cfg = self.cfg
        instances = training_instances(cfg.n, cfg.m, cfg.episodes_per_generation, cfg.seed, generation)
        seeds = [int(s) for s in np.random.SeedSequence([cfg.seed, generation, 17]).generate_state(len(instances))]
        if generation == 0 and not cfg.warm_start:
            episodes = runner.run(instances, seeds, cfg.iterations, cfg.c, cfg.sigma)
            threshold = None
        else:
            threshold = self.ranked.threshold()
            infer = make_inference(self.model, cfg.inference_backend)
            episodes = runner.run(instances, seeds, cfg.iterations, cfg.c, cfg.sigma, model=self.model,
                                  infer=infer, threshold=threshold, noise_weight=cfg.noise_weight,
                                  noise_alpha_scale=cfg.noise_alpha_scale, backend=cfg.inference_backend,
                                  q_init=cfg.q_init, leaf_rollouts=cfg.leaf_rollouts, mix=cfg.rollout_mix,
                                  value_target=cfg.value_target, normalize_q=cfg.normalize_q)
        return episodes, threshold

    def _store(self, episodes: list, threshold: Optional[float], generation: int) -> tuple[float, float]:
        """Label the episodes with the ranked reward and add their records to the buffer.

        Generation 0 (searches of U) has no earlier results and is ranked against its own outcomes.  The
        searches of U with rollouts are much stronger than the first networks, so if their results stayed
        in the ranking window no episode of the next six generations would be rewarded and the value network
        would receive no signal.  By default the first generation that searches with the networks is
        therefore also ranked against its own outcomes, and the ranking window then holds only results of
        the networks (``rank_against_generation0 = True`` keeps the results of U in the window instead).
        With a warm start the window starts with the results of the copied rule and is never cleared.
        """
        outcomes = [ep.outcome for ep in episodes]
        own = threshold is None or (generation == 1 and not self.cfg.rank_against_generation0
                                    and not self.cfg.warm_start)
        if own:
            threshold = self.ranked.threshold(outcomes)
            if generation == 1:
                self.ranked.history.clear()        # drop the results of U
        wins = 0
        ent, top = [], []
        search_targets = self.cfg.value_target == "search"
        for ep in episodes:
            z = RankedReward.label(ep.outcome, threshold)
            wins += z > 0
            for i in range(len(ep)):
                if search_targets:
                    z = float(ep.v[i])                          # V_MCTS: the value of the root of the search
                if ep.mask[i].sum() > 1:                        # targets of the policy: how sharp are they?
                    ent.append(_norm_entropy(ep.pi[i], ep.mask[i]))
                    top.append(float(np.max(ep.pi[i])))
                self.buffer.add(Record(ep.x[i], ep.mpred[i], ep.next_op[i], ep.mask[i], ep.pi[i], z,
                                       ep.f[i] if ep.f else None))
            self.model_steps += ep.model_steps
        self.ranked.update(outcomes)
        self._target_entropy = float(np.nanmean(ent)) if ent else np.nan
        self._target_top = float(np.mean(top)) if top else np.nan
        return threshold, wins / max(len(episodes), 1)

    def _fit(self) -> tuple[float, float]:
        """One pass over the replay buffer, with ``fit_threads`` PyTorch threads (the episode workers are idle
        meanwhile, so the fit can use every physical core)."""
        threads = torch.get_num_threads()
        torch.set_num_threads(fit_thread_count(self.cfg.fit_threads))
        try:
            return self._fit_epoch()
        finally:
            torch.set_num_threads(threads)

    def _fit_epoch(self) -> tuple[float, float]:
        cfg = self.cfg
        self.model.train()
        loss_p, loss_v, batches = 0.0, 0.0, 0
        ent_sum, ent_n, v_abs = 0.0, 0, 0.0
        for batch in self.buffer.epoch(cfg.batch_size, self.rng):
            if getattr(self.model, "uses_features", False):                      # fast networks
                logits, value = self.model.forward_features_value(batch["f"], batch["mask"])
            else:
                logits, value = self.model.forward_batch(batch["x"], batch["mpred"], batch["next_op"],
                                                         batch["mask"])
            logp = torch.log_softmax(logits, dim=1)
            logp = torch.where(batch["mask"], logp, torch.zeros_like(logp))      # finished jobs: pi = 0
            policy_loss = -(batch["pi"] * logp).sum(dim=1).mean()                # cross-entropy to pi_t
            value_loss = ((value - batch["z"]) ** 2).mean()                      # squared error to z
            loss = cfg.policy_weight * policy_loss + cfg.value_weight * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_p += policy_loss.item()
            loss_v += value_loss.item()
            batches += 1
            with torch.no_grad():                                   # entropy of the network's own policy
                k = batch["mask"].sum(dim=1).float()
                p = logp.exp()
                h = -(p * logp).sum(dim=1)
                ok = k > 1
                ent_sum += float((h[ok] / torch.log(k[ok])).sum())
                ent_n += int(ok.sum())
                v_abs += float(value.abs().mean())
        self.model.eval()
        self._policy_entropy = ent_sum / ent_n if ent_n else np.nan
        self._value_abs = v_abs / max(batches, 1)
        return loss_p / max(batches, 1), loss_v / max(batches, 1)

    def validate(self) -> float:
        return greedy_ratio(make_inference(self.model, self.cfg.inference_backend), self.validation)

    def _meta(self, generation: int, score: float) -> dict:
        return {
            "name": self.cfg.name, "generation": generation, "validation_score": score,
            "uniform_score": self.uniform_score, "model_steps": self.model_steps,
            "config": asdict(self.cfg), "ranked_reward": self.ranked.state(),
        }

    # --------------------------------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg
        runner = EpisodeRunner(cfg.workers)
        stale = 0
        stop = "max_generations"
        started = time.perf_counter()
        if cfg.warm_start:
            self._warm_start()
        try:
            for generation in range(cfg.max_generations):
                t0 = time.perf_counter()
                episodes, threshold = self._play(runner, generation)
                threshold, win_share = self._store(episodes, threshold, generation)
                loss_p, loss_v = self._fit()
                score = self.validate()
                row = {
                    "generation": generation,
                    "search": "U" if generation == 0 and not cfg.warm_start else "PUCT",
                    "selfplay_ratio": float(np.mean([ep.outcome for ep in episodes])),
                    "threshold": threshold, "win_share": win_share,
                    "policy_loss": loss_p, "value_loss": loss_v,
                    "validation_ratio": score, "uniform_ratio": self.uniform_score,
                    "buffer": len(self.buffer), "model_steps": self.model_steps,
                    "target_entropy": getattr(self, "_target_entropy", np.nan),
                    "target_top_share": getattr(self, "_target_top", np.nan),
                    "policy_entropy": getattr(self, "_policy_entropy", np.nan),
                    "value_abs": getattr(self, "_value_abs", np.nan),
                    "best_ratio": min(self.best_score, score), "stale": 0 if score < self.best_score else stale + 1,
                    "elapsed_minutes": (time.perf_counter() - started) / 60,
                    "seconds": time.perf_counter() - t0,
                }
                self.history.append(row)
                if cfg.verbose >= 1:
                    better = "*" if score < self.best_score else " "
                    self.log(f"[{cfg.name}] gen {generation:3d} {row['search']:4s} self-play {row['selfplay_ratio']:.4f} "
                             f"val {score:.4f}{better} (uniform {self.uniform_score:.4f}) loss p {loss_p:.4f} "
                             f"v {loss_v:.4f} {row['seconds']:.1f}s")
                if cfg.verbose >= 2:
                    self.log(f"           targets: entropy {row['target_entropy']:.3f} top share "
                             f"{row['target_top_share']:.3f} | network: entropy {row['policy_entropy']:.3f} "
                             f"|value| {row['value_abs']:.3f} | ranked threshold {threshold:.4f} wins "
                             f"{win_share:.0%} | buffer {len(self.buffer)} steps {self.model_steps:.3g} "
                             f"| no improvement {row['stale']}/{cfg.patience or '-'} "
                             f"| {row['elapsed_minutes']:.1f} min")
                save_checkpoint(self.out_dir / "last.pt", self.model, self._meta(generation, score))
                if score < self.best_score:
                    self.best_score, self.best_generation, stale = score, generation, 0
                    save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(generation, score))
                else:
                    stale += 1
                with open(self.out_dir / "history.json", "w", encoding="utf-8") as fh:
                    json.dump(self.history, fh, indent=1)
                if cfg.patience and stale >= cfg.patience and generation + 1 >= cfg.min_generations:
                    stop = "no_improvement"
                    break
                if cfg.budget_model_steps is not None and self.model_steps >= cfg.budget_model_steps:
                    stop = "budget"
                    break
                if cfg.max_hours and time.perf_counter() - started >= cfg.max_hours * 3600:
                    stop = "time"
                    break
        finally:
            runner.close()
        summary = {
            "name": cfg.name, "kind": cfg.kind, "size": f"{cfg.n}x{cfg.m}", "stopped_by": stop,
            "generations": len(self.history), "best_generation": self.best_generation,
            "best_validation_ratio": self.best_score, "uniform_ratio": self.uniform_score,
            "model_steps": self.model_steps, "checkpoint": str(self.out_dir / "best.pt"),
            "warm_start": self.warm_start_summary, "config": asdict(cfg),
        }
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        return summary
