"""A job shop instance in the representation used by the dynamics model (Sections 2.1 and 3.5.1)."""
from __future__ import annotations

from typing import Optional

import numpy as np

#: Size classes of Section 3.5.1: small <= 100 operations, medium 101-250, large > 250.
SMALL_MAX_OPS = 100
MEDIUM_MAX_OPS = 250


class Instance:
    """Classical JSSP instance: ``n`` jobs, each with ``m`` operations, one per machine.

    Operations are numbered ``op = j * m + k`` (operation ``k`` of job ``j``).  The arrays are kept twice:
    as NumPy arrays for vectorised code and as nested Python lists for the hot loops of the dynamics model,
    where list indexing is several times faster than NumPy scalar indexing.
    """

    __slots__ = (
        "name", "n", "m", "n_ops", "machines", "durations", "mach", "dur", "rem", "cum", "lb0",
        "bks", "lower_bound", "family", "job_pred",
    )

    def __init__(self, name: str, machines, durations, bks: Optional[int] = None,
                 lower_bound: Optional[int] = None, family: str = "generated"):
        machines = np.asarray(machines, dtype=np.int64)
        durations = np.asarray(durations, dtype=np.int64)
        if machines.ndim != 2 or machines.shape != durations.shape:
            raise ValueError(f"{name}: machines and durations must be two n x m arrays of the same shape")
        self.name = name
        self.n, self.m = (int(v) for v in machines.shape)
        self.n_ops = self.n * self.m
        self.machines = machines
        self.durations = durations
        self.mach = machines.tolist()
        self.dur = durations.tolist()
        # rem[j][k]: work remaining for job j from operation k onwards (rem[j][m] = 0); the MWKR score.
        rem = np.zeros((self.n, self.m + 1), dtype=np.int64)
        rem[:, : self.m] = durations[:, ::-1].cumsum(axis=1)[:, ::-1]
        self.rem = rem.tolist()
        #: cum[j, k] = total duration of operations 0..k of job j
        self.cum = durations.cumsum(axis=1)
        # LB0 (Section 3.4.1): the larger of the longest job and the highest machine workload.
        longest_job = int(durations.sum(axis=1).max())
        workload = np.bincount(machines.ravel(), weights=durations.ravel(), minlength=self.m)
        self.lb0 = max(longest_job, int(round(float(workload.max()))))
        if self.lb0 <= 0:
            raise ValueError(f"{name}: all processing times are zero")
        self.bks = None if bks is None else int(bks)
        self.lower_bound = None if lower_bound is None else int(lower_bound)
        self.family = family
        # job predecessor of every operation; n_ops marks "none" (index of a zero padding row)
        jp = np.arange(self.n_ops, dtype=np.int64) - 1
        jp[:: self.m] = self.n_ops
        self.job_pred = jp

    # ------------------------------------------------------------------ descriptive properties
    @property
    def shape(self) -> tuple[int, int]:
        return self.n, self.m

    @property
    def size_label(self) -> str:
        return f"{self.n}x{self.m}"

    @property
    def size_class(self) -> str:
        if self.n_ops <= SMALL_MAX_OPS:
            return "small"
        if self.n_ops <= MEDIUM_MAX_OPS:
            return "medium"
        return "large"

    @property
    def optimal_known(self) -> Optional[bool]:
        """True if the BKS is proven optimal, None if no bounds are attached."""
        if self.bks is None or self.lower_bound is None:
            return None
        return self.bks == self.lower_bound

    def gap(self, makespan: float) -> Optional[float]:
        """Optimality gap in per cent, (Cmax - BKS) / BKS * 100 (Section 3.6.4)."""
        if self.bks is None:
            return None
        return (makespan - self.bks) / self.bks * 100.0

    def __repr__(self) -> str:
        return f"Instance({self.name!r}, {self.n}x{self.m}, LB0={self.lb0}, BKS={self.bks})"


def verify_instance(inst: Instance) -> None:
    """Checks made when an instance is read (Section 3.5.1): every job visits every machine exactly once
    and all processing times are non-negative integers."""
    expected = np.arange(inst.m)
    if not np.array_equal(np.sort(inst.machines, axis=1), np.broadcast_to(expected, inst.machines.shape)):
        bad = [j for j in range(inst.n) if sorted(inst.mach[j]) != list(range(inst.m))]
        raise ValueError(f"{inst.name}: jobs {bad} do not visit every machine exactly once")
    if (inst.durations < 0).any():
        raise ValueError(f"{inst.name}: negative processing time")
