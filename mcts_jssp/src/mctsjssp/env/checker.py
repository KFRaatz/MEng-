"""Independent check of complete schedules (first check of Section 3.7).

The checker does not use the dynamics model: it receives only the instance and the start times and verifies
precedence within the jobs, that no two operations overlap on a machine, and the makespan.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..instances.instance import Instance


def schedule_violations(inst: Instance, start, makespan: Optional[int] = None,
                        durations: Optional[np.ndarray] = None) -> list[str]:
    """Return a list of violations (empty if the schedule is feasible).

    ``start`` is an ``n x m`` array of start times, ``durations`` defaults to the nominal processing times.
    """
    start = np.asarray(start, dtype=np.int64).reshape(inst.n, inst.m)
    dur = inst.durations if durations is None else np.asarray(durations, dtype=np.int64).reshape(inst.n, inst.m)
    end = start + dur
    problems: list[str] = []
    if (start < 0).any():
        problems.append("negative start time")
    # precedence within every job
    late = start[:, 1:] < end[:, :-1]
    if late.any():
        j, k = np.argwhere(late)[0]
        problems.append(f"job {j}: operation {k + 1} starts at {start[j, k + 1]} before operation {k} ends at {end[j, k]}")
    # no overlap on any machine
    for mc in range(inst.m):
        mask = inst.machines == mc
        s, e = start[mask], end[mask]
        if s.size != inst.n:
            problems.append(f"machine {mc}: {s.size} operations instead of {inst.n}")
            continue
        order = np.lexsort((e, s))
        s, e = s[order], e[order]
        clash = e[:-1] > s[1:]
        if clash.any():
            i = int(np.argmax(clash))
            problems.append(f"machine {mc}: operations overlap at time {s[i + 1]}")
    cmax = int(end.max())
    if makespan is not None and int(makespan) != cmax:
        problems.append(f"reported makespan {makespan} differs from the makespan {cmax} of the start times")
    return problems


def assert_valid_schedule(inst: Instance, start, makespan: Optional[int] = None,
                          durations: Optional[np.ndarray] = None) -> int:
    """Raise ``AssertionError`` if the schedule is infeasible; return its makespan otherwise."""
    problems = schedule_violations(inst, start, makespan, durations)
    if problems:
        raise AssertionError(f"{inst.name}: infeasible schedule: " + "; ".join(problems))
    start = np.asarray(start, dtype=np.int64).reshape(inst.n, inst.m)
    dur = inst.durations if durations is None else np.asarray(durations, dtype=np.int64).reshape(inst.n, inst.m)
    return int((start + dur).max())
