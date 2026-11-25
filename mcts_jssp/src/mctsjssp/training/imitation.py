"""Warm start: the policy network first learns to copy a dispatching rule, then self-play continues from there.

This is imitation learning (behaviour cloning).  The rule is run on newly generated instances; at every
decision with more than one legal action the observation and the job the rule chose are stored, and the
policy network is fitted to predict that choice with the same cross-entropy loss as in self-play, the
target being the rule's choice instead of the visit distribution of a search.  Only the parameters that the
policy loss reaches are trained (the shared encoder and the policy head of the GIN, the policy network of AZ);
the value network is left to self-play.

The outcomes ``Cmax / LB0`` of the rule's episodes are returned as well: they seed the window of the ranked
reward, so that the first episodes of self-play are rewarded only if they beat most of the rule's results.

The fast policy networks (:mod:`mctsjssp.networks.fast`) are trained the same way, on their own features:
by copying a dispatching rule (:func:`rule_feature_records`) or a trained network (:func:`network_feature_records`,
soft targets), with :func:`imitate_fast`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..env.model import State
from ..env.observation import FeatureTracker
from ..env.rules import RULES, startable_now
from ..instances.generator import generate_instance
from .replay import Record, ReplayBuffer

#: Seed offset of the warm-start instances (distinct from the validation and self-play instances).
WARM_START_STREAM = 4_711


def rule_episodes(rule: str, n: int, m: int, count: int, seed: int,
                  non_delay: bool = False) -> tuple[list[Record], list[float]]:
    """Records of the rule's decisions on ``count`` generated instances, and the outcome of every episode."""
    if rule not in RULES:
        raise ValueError(f"unknown dispatching rule {rule!r}; expected one of {sorted(RULES)}")
    choose = RULES[rule]
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, WARM_START_STREAM]))
    records: list[Record] = []
    outcomes: list[float] = []
    for i in range(count):
        inst = generate_instance(n, m, rng, name=f"warm{n}x{m}_{i}")
        state = State(inst)
        tracker = FeatureTracker(inst).reset(state)
        while state.active:
            job = choose(state, startable_now(state) if non_delay else None)
            if len(state.active) > 1:
                pi = np.zeros(n, dtype=np.float32)
                pi[job] = 1.0
                mask = np.zeros(n, dtype=bool)
                mask[state.active] = True
                x, mpred = tracker.snapshot()
                records.append(Record(x, mpred, np.array(state.next_op, dtype=np.int16), mask, pi, 0.0))
            state.step(job)
            tracker.update(state)
        outcomes.append(state.makespan / inst.lb0)
    return records, outcomes


def imitate(model, records: Sequence[Record], n: int, m: int, epochs: int = 8, learning_rate: float = 1e-3,
            batch_size: int = 256, weight_decay: float = 5e-4, seed: int = 0) -> list[float]:
    """Fit the policy of ``model`` to the recorded choices; returns the mean policy loss of every epoch."""
    buffer = ReplayBuffer(len(records), n, m)
    buffer.extend(records)
    optimizer = torch.optim.Adam(model.policy_parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(np.random.SeedSequence([seed, WARM_START_STREAM, 1]))
    losses = []
    for _ in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for batch in buffer.epoch(batch_size, rng):
            logits, _ = model.forward_batch(batch["x"], batch["mpred"], batch["next_op"], batch["mask"])
            logp = torch.where(batch["mask"], torch.log_softmax(logits, dim=1), torch.zeros_like(logits))
            loss = -(batch["pi"] * logp).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        losses.append(total / max(batches, 1))
    model.eval()
    return losses


# --------------------------------------------------------------------------------- fast policy networks
def _feature_episode(inst, choose, n: int, feats: list, masks: list, targets: list) -> float:
    """Run one episode with ``choose(state) -> (job, target distribution or None)``; store the records."""
    from ..networks.fast import state_features

    state = State(inst)
    while state.active:
        job, target = choose(state)
        if len(state.active) > 1:
            mask = np.zeros(n, dtype=bool)
            mask[state.active] = True
            if target is None:
                target = np.zeros(n, dtype=np.float32)
                target[job] = 1.0
            feats.append(state_features(state))
            masks.append(mask)
            targets.append(target)
        state.step(job)
    return state.makespan / inst.lb0


def rule_feature_records(rule: str, n: int, m: int, count: int, seed: int, non_delay: bool = False):
    """Feature records of a dispatching rule's decisions, for the fast policy networks.

    Returns ``(feats (N, n, 6), mask (N, n), targets (N, n), outcomes)``; a target is the rule's choice.
    """
    if rule not in RULES:
        raise ValueError(f"unknown dispatching rule {rule!r}; expected one of {sorted(RULES)}")
    pick = RULES[rule]
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, WARM_START_STREAM, 2]))
    feats, masks, targets, outcomes = [], [], [], []
    for i in range(count):
        inst = generate_instance(n, m, rng, name=f"fast{n}x{m}_{i}")
        outcomes.append(_feature_episode(
            inst, lambda st: (pick(st, startable_now(st) if non_delay else None), None), n, feats, masks, targets))
    return np.stack(feats), np.stack(masks), np.stack(targets), outcomes


def network_feature_records(infer, n: int, m: int, count: int, seed: int, tau: float = 1.0):
    """Feature records for copying a trained network (distillation).

    The teacher dispatches by sampling from its own softmax at temperature ``tau``; the target of every
    record is that softmax over all jobs (zero for finished jobs).
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, WARM_START_STREAM, 3]))
    feats, masks, targets, outcomes = [], [], [], []
    for i in range(count):
        inst = generate_instance(n, m, rng, name=f"distil{n}x{m}_{i}")
        infer.check(inst)
        tracker = FeatureTracker(inst)

        def choose(state, tracker=tracker):
            if len(state.active) == 1:
                return state.active[0], None
            z = np.asarray(infer.logits(tracker.reset(state), state), dtype=np.float64)
            p = np.exp((z - z.max()) / tau)
            p /= p.sum()
            index = int(rng.choice(len(p), p=p))
            target = np.zeros(n, dtype=np.float32)
            target[state.active] = p
            return state.active[index], target

        outcomes.append(_feature_episode(inst, choose, n, feats, masks, targets))
    return np.stack(feats), np.stack(masks), np.stack(targets), outcomes


def imitate_fast(model, feats: np.ndarray, masks: np.ndarray, targets: np.ndarray, epochs: int = 8,
                 learning_rate: float = 1e-3, batch_size: int = 256, weight_decay: float = 5e-4,
                 seed: int = 0) -> list[float]:
    """Fit a fast policy network to the targets with cross-entropy; returns the mean loss of every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(np.random.SeedSequence([seed, WARM_START_STREAM, 4]))
    x_all, m_all, t_all = (torch.from_numpy(a) for a in (feats, masks, targets))
    losses = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(feats))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(np.sort(order[start: start + batch_size]))
            logits = model.forward_features(x_all[idx], m_all[idx])
            logp = torch.where(m_all[idx], torch.log_softmax(logits, dim=1), torch.zeros_like(logits))
            loss = -(t_all[idx] * logp).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        losses.append(total / max(batches, 1))
    model.eval()
    return losses
