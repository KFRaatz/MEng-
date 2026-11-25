"""Priority dispatching rules run as complete policies (second check of Section 3.7).

The rules are the four that Ho et al. compare on the OR-Library families: SPT, MWKR, MOR and FIFO.  At every
step the rule chooses a job and the dynamics model dispatches its next operation; ties are broken by the
lower job index.

A rule can be run in two ways.

* ``non_delay=True`` - as dispatching rules are normally implemented in the literature: only the operations
  that can start at the earliest possible time are candidates, so no machine is kept idle while an operation
  waits for it.  Run in this way the four rules reproduce the gaps published by Ho et al. to within 0.1
  percentage points on all six families, which validates the dynamics model.
* ``non_delay=False`` - in the MDP of the thesis (Section 3.4.2), where every unfinished job is a legal
  action.  The rule may then choose an operation that cannot start yet, and the gaps are larger (for MWKR,
  MOR and SPT; FIFO chooses the earliest available operation and is the same in both).  The MWKR rollout
  distribution of the search works on this set of actions.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..instances.instance import Instance
from .model import State

Rule = Callable[[State, Optional[Sequence[int]]], int]


def spt(state: State, candidates: Optional[Sequence[int]] = None) -> int:
    """Shortest processing time of the candidate operation."""
    inst = state.inst
    return min(candidates or state.active, key=lambda j: (inst.dur[j][state.next_op[j]], j))


def mwkr(state: State, candidates: Optional[Sequence[int]] = None) -> int:
    """Most work remaining for the job (including the candidate operation)."""
    inst = state.inst
    return min(candidates or state.active, key=lambda j: (-inst.rem[j][state.next_op[j]], j))


def mor(state: State, candidates: Optional[Sequence[int]] = None) -> int:
    """Most operations remaining for the job."""
    return min(candidates or state.active, key=lambda j: (state.next_op[j], j))


def fifo(state: State, candidates: Optional[Sequence[int]] = None) -> int:
    """First in, first out: the operation that became available first (its job predecessor ended earliest)."""
    return min(candidates or state.active, key=lambda j: (state.job_ready[j], j))


RULES: dict[str, Rule] = {"SPT": spt, "MWKR": mwkr, "MOR": mor, "FIFO": fifo}


def startable_now(state: State) -> list[int]:
    """Jobs whose next operation can start at the earliest possible time (the non-delay candidates)."""
    inst = state.inst
    earliest = {}
    for j in state.active:
        ends = state.m_end[inst.mach[j][state.next_op[j]]]
        earliest[j] = max(state.job_ready[j], ends[-1] if ends else 0)
    first = min(earliest.values())
    return [j for j in state.active if earliest[j] == first]


def run_rule(inst: Instance, rule: Rule, non_delay: bool = False) -> State:
    """Complete schedule of a dispatching rule."""
    state = State(inst)
    while state.active:
        state.step(rule(state, startable_now(state) if non_delay else None))
    return state
