"""EARLIER ALTERNATIVE, no longer used by the commands: the fast networks now have a value output and are
trained by the PUCT self-play of LR and AZ (trainer.py).  Kept for comparison.

Self-play training of the fast policy networks from scratch (no rule to copy).

The fast networks have no value output, so their self-play search uses them the way the experiments do: in
the rollouts of UCT.  One generation

1. plays 40 episodes on newly generated instances.  Generation 0 searches with U (uniform rollouts), as in
   Section 3.6.1; later generations sample the rollouts from the current fast network at temperature
   ``rollout_tau``;
2. records at every decision with more than one legal action the features of all jobs and the visit
   distribution ``pi_t`` at the root, and commits the most visited operation;
3. adds the records to a replay buffer (36 000 records, oldest removed first) and fits the network for one
   epoch in mini-batches of 256 with cross-entropy to ``pi_t``;
4. scores the network on the validation instances by the mean ``Cmax / LB0`` of its greedy policy.

Training stops after ``patience`` generations without improvement, at the training budget (steps of the
dynamics model) or after ``max_generations``; the best checkpoint is kept.  This is expert iteration: the
search, guided by the current rollout policy, produces targets that are better than the policy itself.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..distributions.fixed import UniformRollout
from ..distributions.learned import FastRollout
from ..env.environment import JobShopEnv
from ..instances.generator import training_instances, validation_set
from ..instances.instance import Instance
from ..mcts.evaluation import make_return
from ..mcts.uct import UCTSearch
from ..networks.checkpoint import build_model, save_checkpoint
from ..networks.fast import N_FEATURES, state_features
from ..networks.inference import make_inference
from .trainer import greedy_ratio, uniform_ratio


@dataclass
class FastTrainConfig:
    name: str = "fast"
    kind: str = "fast_lr"                    # "fast_lr" or "fast_az"
    n: int = 6
    m: int = 6
    hparams: dict = field(default_factory=dict)
    episodes_per_generation: int = 40
    iterations: int = 100
    c: float = 1.0
    rollout_tau: float = 1.0                 # rollouts of the self-play search: the network's own softmax
    target: str = "visits"                   # "visits": pi_t of Section 3.6.1; "chosen": the committed action
    sigma: Optional[float] = None
    buffer_size: int = 36_000
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    patience: int = 5
    min_generations: int = 0
    max_generations: int = 100
    budget_model_steps: Optional[int] = None
    validation_instances: int = 100
    seed: int = 0

    def network_hparams(self) -> dict:
        hp = dict(self.hparams)
        if self.kind == "fast_az":
            hp.setdefault("n", self.n)
            hp.setdefault("m", self.m)
        return hp


class FeatureBuffer:
    """Replay buffer of feature records: features ``(n, 6)``, legal jobs ``(n,)`` and the target ``(n,)``."""

    def __init__(self, capacity: int, n: int):
        self.capacity = int(capacity)
        self.feats = np.zeros((self.capacity, n, N_FEATURES), dtype=np.float32)
        self.masks = np.zeros((self.capacity, n), dtype=bool)
        self.targets = np.zeros((self.capacity, n), dtype=np.float32)
        self.size = 0
        self._head = 0

    def __len__(self) -> int:
        return self.size

    def add(self, feats: np.ndarray, mask: np.ndarray, target: np.ndarray) -> None:
        i = self._head
        self.feats[i], self.masks[i], self.targets[i] = feats, mask, target
        self._head = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


def play_fast_episode(inst: Instance, policy, iterations: int, seed: int, c: float = 1.0,
                      sigma: Optional[float] = None, target: str = "visits") -> dict:
    """One self-play episode: UCT with the given rollout policy; returns the records and the outcome.

    ``target = "visits"`` records the visit distribution at the root (Section 3.6.1); ``"chosen"`` records the
    operation the search commits to (one-hot), which stays informative when the visits are nearly uniform.
    """
    if target not in ("visits", "chosen"):
        raise ValueError("target must be 'visits' or 'chosen'")
    env = JobShopEnv(inst)
    state = env.reset()
    evaluate = make_return(inst, sigma, seed)
    search = UCTSearch(policy, evaluate, c=c, seed=seed)
    search.start(state)
    feats, masks, targets = [], [], []
    while not env.done:
        if len(state.active) == 1:
            action = state.active[0]
        else:
            search.run_iterations(state, iterations)
            pi = np.zeros(inst.n, dtype=np.float32)
            for child in search.root.children:
                pi[child.action] = child.N
            pi /= pi.sum()
            mask = np.zeros(inst.n, dtype=bool)
            mask[state.active] = True
            action = search.choose()
            if target == "chosen":
                pi = np.zeros(inst.n, dtype=np.float32)
                pi[action] = 1.0
            feats.append(state_features(state))
            masks.append(mask)
            targets.append(pi)
        env.step(action)
        search.advance(action, state)
    return {"feats": feats, "masks": masks, "targets": targets, "makespan": env.makespan,
            "outcome": evaluate.makespan(state) / inst.lb0, "model_steps": search.model_steps + inst.n_ops}


class FastTrainer:
    def __init__(self, cfg: FastTrainConfig, out_dir: str | Path, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg.kind, cfg.network_hparams())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
        self.buffer = FeatureBuffer(cfg.buffer_size, cfg.n)
        self.validation = list(validation_set(cfg.n, cfg.m)[: cfg.validation_instances])
        self.rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 37]))
        self.uniform_score = uniform_ratio(self.validation, seed=cfg.seed)
        self.history: list[dict] = []
        self.model_steps = 0
        self.best_score = math.inf
        self.best_generation = -1

    def _policy(self, generation: int):
        if generation == 0:
            return UniformRollout()
        return FastRollout(self.cfg.name, make_inference(self.model), level=None, tau=self.cfg.rollout_tau)

    def _fit(self) -> float:
        cfg = self.cfg
        self.model.train()
        order = self.rng.permutation(len(self.buffer))
        total, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = np.sort(order[start: start + cfg.batch_size])
            x = torch.from_numpy(self.buffer.feats[idx])
            mask = torch.from_numpy(self.buffer.masks[idx])
            target = torch.from_numpy(self.buffer.targets[idx])
            logits = self.model.forward_features(x, mask)
            logp = torch.where(mask, torch.log_softmax(logits, dim=1), torch.zeros_like(logits))
            loss = -(target * logp).sum(dim=1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            batches += 1
        self.model.eval()
        return total / max(batches, 1)

    def _meta(self, generation: int, score: float) -> dict:
        return {"name": self.cfg.name, "generation": generation, "validation_score": score,
                "uniform_score": self.uniform_score, "model_steps": self.model_steps, "config": asdict(self.cfg),
                "training": "self-play from scratch"}

    def run(self) -> dict:
        cfg = self.cfg
        stale, stop = 0, "max_generations"
        for generation in range(cfg.max_generations):
            t0 = time.perf_counter()
            policy = self._policy(generation)
            instances = training_instances(cfg.n, cfg.m, cfg.episodes_per_generation, cfg.seed, generation)
            seeds = np.random.SeedSequence([cfg.seed, generation, 19]).generate_state(len(instances))
            outcomes, target_entropy, top_share = [], [], []
            for inst, seed in zip(instances, seeds):
                ep = play_fast_episode(inst, policy, cfg.iterations, int(seed), cfg.c, cfg.sigma, cfg.target)
                for f, mk, t in zip(ep["feats"], ep["masks"], ep["targets"]):
                    self.buffer.add(f, mk, t)
                    k = int(mk.sum())
                    nz = t[t > 0]
                    target_entropy.append(float(-(nz * np.log(nz)).sum()) / math.log(k))
                    top_share.append(float(t.max()) * k)     # 1 = uniform visits, k = all visits on one action
                outcomes.append(ep["outcome"])
                self.model_steps += ep["model_steps"]
            loss = self._fit()
            score = greedy_ratio(make_inference(self.model), self.validation, cfg.name)
            row = {"generation": generation, "search": "U" if generation == 0 else "fast rollouts",
                   "selfplay_ratio": float(np.mean(outcomes)), "validation_ratio": score,
                   "uniform_ratio": self.uniform_score, "policy_loss": loss,
                   "target_entropy": float(np.mean(target_entropy)), "top_visit_vs_uniform": float(np.mean(top_share)),
                   "buffer": len(self.buffer), "model_steps": self.model_steps,
                   "seconds": time.perf_counter() - t0}
            self.history.append(row)
            self.log(f"[{cfg.name}] gen {generation:3d} {row['search']:13s} self-play {row['selfplay_ratio']:.4f} "
                     f"val {score:.4f} (uniform {self.uniform_score:.4f}) loss {loss:.4f} "
                     f"target entropy {row['target_entropy']:.3f} buffer {len(self.buffer)} {row['seconds']:.1f}s")
            save_checkpoint(self.out_dir / "last.pt", self.model, self._meta(generation, score))
            if score < self.best_score:
                self.best_score, self.best_generation, stale = score, generation, 0
                save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(generation, score))
            else:
                stale += 1
            with open(self.out_dir / "history.json", "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=1)
            if stale >= cfg.patience and generation + 1 >= cfg.min_generations:
                stop = "no_improvement"
                break
            if cfg.budget_model_steps is not None and self.model_steps >= cfg.budget_model_steps:
                stop = "budget"
                break
        summary = {"name": cfg.name, "kind": cfg.kind, "size": f"{cfg.n}x{cfg.m}", "stopped_by": stop,
                   "generations": len(self.history), "best_generation": self.best_generation,
                   "best_validation_ratio": self.best_score, "uniform_ratio": self.uniform_score,
                   "model_steps": self.model_steps, "training": "self-play from scratch", "config": asdict(cfg)}
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        return summary
