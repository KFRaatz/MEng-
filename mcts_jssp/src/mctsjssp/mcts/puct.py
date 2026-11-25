"""PUCT search with a policy and a value network, used only in the self-play training of LR and AZ and of
their fast networks
(Sections 2.6.1 and 3.6.1).

When selection reaches a leaf, the network evaluates it: the leaf is expanded with all its children, whose
priors are the softmax of the policy logits at temperature 1, and the value estimate takes the place of the
rollout.  In the tree, the child that maximises ``Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))``
(Equation 2.19) is followed.  Dirichlet noise is added to the priors at the root so that operations with a
low prior are explored.  A complete schedule is valued by ``terminal_value`` (the ranked reward of its
makespan), which is on the same scale as the output of the value network.

A child that has not been visited needs a value in Equation 2.19.  ``q_init = "zero"`` (default) is the
AlphaZero convention described in Section 2.6.1: children are created with Q = 0.  The ranked reward is not
centred on 0: with the 75th percentile three episodes in four receive -1, so the mean value is about -0.5
and an unvisited child looks better than a visited one; the search then tries every child of a node before
it returns to any of them.  ``q_init = "parent"`` gives an unvisited child the current mean value of its
parent instead, which is neutral.

Leaf rollouts (optional, after Kemmerling et al. and AlphaGo): with ``rollout`` set, a newly expanded leaf is
also completed by a rollout of that distribution (uniform random in training), and its value is
``(1 - mix) * v + mix * terminal_value(completed schedule)``, the value estimate ``v`` of the network mixed
with the outcome of the rollout.  The search then knows something about every operation it tries before
the networks have learned anything.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

import numpy as np

from ..env.model import State
from ..env.observation import FeatureTracker
from .node import Node


class PUCTSearch:
    def __init__(self, infer, terminal_value: Callable[[State], float], c: float = 1.0,
                 noise_weight: float = 0.25, noise_alpha_scale: float = 10.0, seed: int = 0,
                 q_init: str = "zero", rollout=None, mix: float = 0.5, normalize_q: bool = False):
        if q_init not in ("parent", "zero"):
            raise ValueError("q_init must be 'parent' or 'zero'")
        self.infer = infer
        self.terminal_value = terminal_value
        self.c = float(c)
        self.q_init = q_init                      # value of a child that has not been visited yet
        self.rollout = rollout                    # rollout distribution at the leaves (None: value network only)
        self.mix = float(mix)                     # weight of the rollout outcome in the value of a leaf
        self._rnd = random.Random(seed * 7_919 + 13)
        self.normalize_q = bool(normalize_q)      # rescale Q to [0, 1] with the extremes of the search (MuZero)
        self._qmin, self._qmax = math.inf, -math.inf
        self.noise_weight = float(noise_weight)
        self.noise_alpha_scale = float(noise_alpha_scale)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 4_242]))
        self.root: Node = Node()
        self.iterations = 0
        self.model_steps = 0
        self.evaluations = 0
        self._tracker: FeatureTracker | None = None
        self._needs_tracker = bool(getattr(infer, "needs_tracker", True))

    # --------------------------------------------------------------------------------------- episode
    def start(self, state: State) -> None:
        self.infer.check(state.inst)
        self._tracker = FeatureTracker(state.inst)
        self.root = Node()

    def advance(self, action: int, state: State) -> None:
        child = self.root.child(action)
        self.root = child if child is not None else Node(action)

    def choose(self) -> int:
        return self.root.most_visited().action

    def visit_distribution(self, n_jobs: int) -> np.ndarray:
        """``pi_t(a) = N(s_t, a) / sum_b N(s_t, b)`` over all jobs (zero for finished jobs)."""
        pi = np.zeros(n_jobs, dtype=np.float32)
        for child in self.root.children:
            pi[child.action] = child.N
        total = pi.sum()
        return pi / total if total > 0 else pi

    # ---------------------------------------------------------------------------------------- search
    def run_iterations(self, state: State, iterations: int) -> int:
        self._qmin, self._qmax = math.inf, -math.inf      # extremes of Q seen by this search
        done = 0
        if not self.root.expanded and state.active:
            self._iterate(state)                          # the first iteration expands the root
            done = 1
        self._add_noise(self.root)
        for _ in range(iterations - done):
            self._iterate(state)
        return iterations

    def _add_noise(self, root: Node) -> None:
        children = root.children
        k = len(children)
        if k < 2 or self.noise_weight <= 0.0:
            for child in children:
                child.P = child.P0
            return
        eta = self.rng.dirichlet(np.full(k, self.noise_alpha_scale / k))
        w = self.noise_weight
        for child, e in zip(children, eta):
            child.P = (1.0 - w) * child.P0 + w * float(e)

    def _iterate(self, root_state: State) -> None:
        node = self.root
        st = root_state.copy()
        path = [node]
        c = self.c
        while True:
            if node.terminal:
                value = self.terminal_value(st)
                break
            if not node.expanded:
                if not st.active:
                    node.terminal = True
                    value = self.terminal_value(st)
                    break
                tracker = self._tracker.reset(st) if self._needs_tracker else None   # fast networks: none
                logits, value = self.infer.evaluate(tracker, st)
                self.evaluations += 1
                z = np.asarray(logits, dtype=np.float64)
                p = np.exp(z - z.max())                       # priors: softmax of the logits at tau = 1
                p /= p.sum()
                node.children = [Node(a, prior=float(pa)) for a, pa in zip(st.active, p)]
                node.expanded = True
                if self.rollout is not None:              # complete the schedule; mix its outcome in
                    self.rollout.rollout(st, self._rnd, [])
                    value = (1.0 - self.mix) * float(value) + self.mix * self.terminal_value(st)
                break
            sqrt_n = math.sqrt(node.N)                        # PUCT, Equation 2.19
            unvisited = node.W / node.N if self.q_init == "parent" else 0.0
            lo, span = 0.0, 1.0
            if self.normalize_q and self._qmax > self._qmin:
                lo, span = self._qmin, self._qmax - self._qmin
            best = None
            best_score = -math.inf
            for child in node.children:
                q = ((child.W / child.N if child.N else unvisited) - lo) / span
                score = q + c * child.P * sqrt_n / (1 + child.N)
                if score > best_score:
                    best_score = score
                    best = child
            node = best
            st.step(node.action)
            path.append(node)
        for node in path:
            node.N += 1
            node.W += value
            if self.normalize_q:
                q = node.W / node.N
                if q < self._qmin:
                    self._qmin = q
                if q > self._qmax:
                    self._qmax = q
        self.iterations += 1
        self.model_steps += st.n_done - root_state.n_done
