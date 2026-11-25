"""The MCTS agent of the experiments: UCT selection with rollouts (Section 3.2.1, Table 3.1).

One iteration

1. *selection*: from the root, a node with untried operations is expanded; otherwise the child that
   maximises ``Q(s,a) + c * sqrt(ln N(s) / N(s,a))`` (Equation 2.18) is followed.  Untried operations have
   priority, so all children of a node are added before any of them is visited a second time;
2. *expansion*: one new node is added per iteration;
3. *simulation*: the schedule is completed from the new node by the rollout distribution, and its return
   ``R = -Cmax / LB0`` is the value ``v`` of the leaf.  If the node reached is a complete schedule, its
   makespan gives the value directly;
4. *backpropagation*: ``N += 1`` and ``W += v`` on every node of the path, with the same ``v`` at every
   level (single decision maker), so that ``Q = W / N`` is the mean return.

The four configurations (U, MWKR, LR, AZ) share this class and differ only in the rollout policy.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

from ..distributions.base import RolloutPolicy
from ..env.model import State
from .node import Node
from .recorder import RunRecorder


class UCTSearch:
    def __init__(self, policy: RolloutPolicy, evaluate, c: float = 1.0, seed: int = 0,
                 recorder: Optional[RunRecorder] = None):
        self.policy = policy
        self.evaluate = evaluate                  # StaticReturn or StochasticReturn
        self.c = float(c)
        self.rnd = random.Random(seed)
        self.recorder = recorder
        self._steps = recorder.steps if recorder is not None else None
        self.root: Node = Node()
        self.prefix: list[int] = []               # jobs committed so far
        self.best_makespan = math.inf             # best complete schedule found by any rollout
        self.best_sequence: Optional[list[int]] = None
        self.anytime: list[tuple[float, int]] = []   # (seconds since start, best makespan) at every improvement
        self._t_start = time.perf_counter()
        self.iterations = 0
        self.model_steps = 0                      # steps of the dynamics model made by the search

    # --------------------------------------------------------------------------------------- episode
    def start(self, state: State) -> None:
        """Begin a new episode at ``state`` (normally the empty schedule)."""
        self.root = Node(untried=state.active[:])
        self.root.terminal = not state.active
        self.prefix = []
        self._t_start = time.perf_counter()

    def advance(self, action: int, state: State) -> None:
        """Re-root the tree after ``action`` was executed; ``state`` is the state after the action.

        The subtree of the committed action is kept for the next decision.
        """
        child = self.root.child(action)
        if child is None:
            child = Node(action, untried=state.active[:])
            child.terminal = not state.active
        self.root = child
        self.prefix.append(action)

    def choose(self) -> int:
        """The most visited action at the root (the committed action)."""
        return self.root.most_visited().action

    # ---------------------------------------------------------------------------------------- budget
    def run_iterations(self, state: State, iterations: int) -> int:
        iterate = self._iterate
        for _ in range(iterations):
            iterate(state)
        return iterations

    def run_time(self, state: State, seconds: float) -> int:
        """Iterate until ``seconds`` have passed; an iteration that has started is always completed."""
        iterate = self._iterate
        clock = time.perf_counter
        deadline = clock() + seconds
        done = 0
        while True:
            iterate(state)
            done += 1
            if clock() >= deadline:
                return done

    # ------------------------------------------------------------------------------------- iteration
    def _iterate(self, root_state: State) -> None:
        node = self.root
        st = root_state.copy()
        step = st.step
        path = [node]
        c = self.c
        trace: list[int] = []
        while True:
            if node.terminal:
                break                                         # complete schedule: no rollout
            untried = node.untried
            if untried:                                       # expansion of one new node
                i = int(self.rnd.random() * len(untried))
                action = untried[i]
                untried[i] = untried[-1]
                untried.pop()
                step(action)
                child = Node(action, st.active[:])
                node.children.append(child)
                path.append(child)
                if st.active:                                 # simulation
                    if self._steps is None:
                        self.policy.rollout(st, self.rnd, trace)
                    else:
                        self._steps.context["rollout"] = self.iterations
                        self.policy.rollout_recorded(st, self.rnd, trace, self._steps)
                else:
                    child.terminal = True
                break
            log_n = math.log(node.N)                          # UCT, Equation 2.18
            best = None
            best_score = -math.inf
            for child in node.children:
                score = child.W / child.N + c * math.sqrt(log_n / child.N)
                if score > best_score:
                    best_score = score
                    best = child
            node = best
            step(node.action)
            path.append(node)
        value = self.evaluate(st)
        for node in path:                                     # backpropagation, the same v at every level
            node.N += 1
            node.W += value
        self.iterations += 1
        self.model_steps += st.n_done - root_state.n_done
        if st.makespan < self.best_makespan:
            self.best_makespan = st.makespan
            self.anytime.append((time.perf_counter() - self._t_start, int(st.makespan)))
            self.best_sequence = self.prefix + [n.action for n in path[1:]] + trace
        if self.recorder is not None:
            self.recorder.on_rollout(value, len(trace))
            if self.recorder.sequence_rate > 0 and self.recorder.want_sequence():
                tree = [n.action for n in path[1:]]
                self.recorder.add_sequence(self.prefix + tree + trace, len(self.prefix),
                                           len(self.prefix) + len(tree), value)
