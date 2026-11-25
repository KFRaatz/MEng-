"""Solving one instance with the MCTS agent: the loop of decisions around the search (Section 3.2.1).

At every decision the search runs for its budget, the most visited action at the root is committed and
executed in the environment, and the subtree of that action is kept for the next decision.  The schedule
committed in this way is the reported solution; the best complete schedule found by a rollout is recorded
separately.  Decisions with a single legal action need no search and are executed directly.

Every complete schedule is checked independently of the dynamics model, and executing it without
disturbances must reproduce its makespan (first check of Section 3.7).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..distributions.base import RolloutPolicy
from ..env.checker import assert_valid_schedule
from ..env.environment import JobShopEnv
from ..env.model import execute
from ..instances.instance import Instance
from .evaluation import make_return
from .recorder import RunRecorder
from .uct import UCTSearch


@dataclass(frozen=True)
class Budget:
    """Search budget per decision: a number of iterations or a computation time (Section 3.6.3).

    For a time budget ``seconds`` is used for every decision.  If ``profile = (a, b)`` is given, a decision
    with ``r`` unscheduled operations gets ``a + b * r`` seconds instead: the time that U needs for its
    iterations at that depth of the schedule (its rollouts become shorter as the schedule fills).
    """

    kind: str                                   # "iterations" | "time"
    iterations: int = 0
    seconds: float = 0.0
    profile: Optional[tuple] = None             # (a, b): seconds = a + b * unscheduled operations
    label: str = ""

    @staticmethod
    def of_iterations(n: int) -> "Budget":
        return Budget("iterations", iterations=int(n), label=f"I{int(n)}")

    @staticmethod
    def of_time(seconds: float, label: str = "", profile: Optional[Sequence[float]] = None) -> "Budget":
        return Budget("time", seconds=float(seconds), profile=tuple(profile) if profile else None,
                      label=label or f"T{seconds:g}s")

    def seconds_for(self, remaining: int) -> float:
        """Seconds for a decision with ``remaining`` unscheduled operations."""
        if self.profile:
            return self.profile[0] + self.profile[1] * remaining
        return self.seconds


@dataclass
class EpisodeResult:
    instance: str
    makespan: int                               # the committed schedule (reported solution)
    sequence: list
    best_rollout_makespan: float                # best complete schedule found by a rollout
    best_rollout_sequence: Optional[list]
    decisions: int                              # decisions that needed a search
    iterations: int
    model_steps: int
    time_search: float                          # total time spent in the searches
    decision_times: list = field(default_factory=list)
    decision_iterations: list = field(default_factory=list)
    decision_depths: list = field(default_factory=list)     # operations scheduled before each decision
    summary: dict = field(default_factory=dict)
    anytime: list = field(default_factory=list)  # (seconds since the start, best makespan so far) at improvements
    time_total: float = 0.0                      # wall time of the whole episode

    @property
    def time_per_decision(self) -> float:
        return self.time_search / self.decisions if self.decisions else 0.0

    @property
    def iterations_per_decision(self) -> float:
        return self.iterations / self.decisions if self.decisions else 0.0


def solve(inst: Instance, policy: RolloutPolicy, budget: Budget, seed: int = 0, c: float = 1.0,
          sigma: Optional[float] = None, recorder: Optional[RunRecorder] = None,
          skip_forced: bool = True, check: bool = True) -> EpisodeResult:
    """Schedule ``inst`` with UCT and the rollout distribution ``policy``.

    ``sigma`` switches on the stochastic setting of RQ5: the schedule of every rollout is executed in a
    scenario sampled from the disturbance model before its return is computed.
    """
    env = JobShopEnv(inst)
    state = env.reset()
    search = UCTSearch(policy, make_return(inst, sigma, seed), c=c, seed=seed, recorder=recorder)
    search.start(state)
    clock = time.perf_counter
    times: list[float] = []
    iters: list[int] = []
    depths: list[int] = []
    n_ops = inst.n_ops
    decision = 0
    while not env.done:
        active = state.active
        if len(active) == 1 and skip_forced:
            action = active[0]
        else:
            if recorder is not None:
                recorder.begin_decision(decision)
            n_candidates = len(active)
            depth = state.n_done
            t0 = clock()
            if budget.kind == "iterations":
                done = search.run_iterations(state, budget.iterations)
            else:
                done = search.run_time(state, budget.seconds_for(n_ops - depth))
            action = search.choose()
            elapsed = clock() - t0
            times.append(elapsed)
            iters.append(done)
            depths.append(depth)
            if recorder is not None:
                recorder.end_decision(search.root, depth, n_candidates, done, elapsed, action)
            decision += 1
        env.step(action)
        search.advance(action, state)
    makespan = env.makespan
    time_total = clock() - search._t_start
    anytime = list(search.anytime)
    if not anytime or makespan < anytime[-1][1]:     # the committed schedule is also a complete schedule
        anytime.append((time_total, int(makespan)))
        assert_valid_schedule(inst, state.start_times(), makespan)
        if execute(inst, state.m_op, inst.durations.ravel().tolist()) != makespan:
            raise AssertionError(f"{inst.name}: executing the schedule without disturbances does not "
                                 f"reproduce its makespan")
    summary = recorder.summary(final_return=-makespan / inst.lb0) if recorder is not None else {}
    return EpisodeResult(
        instance=inst.name, makespan=makespan, sequence=list(env.sequence),
        best_rollout_makespan=search.best_makespan, best_rollout_sequence=search.best_sequence,
        decisions=decision, iterations=search.iterations, model_steps=search.model_steps + inst.n_ops,
        time_search=sum(times), decision_times=times, decision_iterations=iters, decision_depths=depths,
        summary=summary, anytime=anytime, time_total=time_total,
    )
