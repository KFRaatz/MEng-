"""The 82 OR-Library job shop instances used as the test set (Section 3.5.1, Table 3.3).

The parser reads both the single-instance "standard" files of JSPLib (``n m`` followed by one row of
``machine duration`` pairs per job, ``#`` comment lines allowed) and the collection file ``jobshop1.txt`` of
the OR-Library, in which every instance is introduced by a line ``instance <name>``.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .instance import Instance, verify_instance

#: Rows of Table 3.3: (family, instance names, jobs n, machines m)
TABLE_3_3 = [
    ("ft", ["ft06"], 6, 6),
    ("ft", ["ft10"], 10, 10),
    ("ft", ["ft20"], 20, 5),
    ("la", [f"la{i:02d}" for i in range(1, 6)], 10, 5),
    ("la", [f"la{i:02d}" for i in range(6, 11)], 15, 5),
    ("la", [f"la{i:02d}" for i in range(11, 16)], 20, 5),
    ("la", [f"la{i:02d}" for i in range(16, 21)], 10, 10),
    ("la", [f"la{i:02d}" for i in range(21, 26)], 15, 10),
    ("la", [f"la{i:02d}" for i in range(26, 31)], 20, 10),
    ("la", [f"la{i:02d}" for i in range(31, 36)], 30, 10),
    ("la", [f"la{i:02d}" for i in range(36, 41)], 15, 15),
    ("abz", ["abz5", "abz6"], 10, 10),
    ("abz", ["abz7", "abz8", "abz9"], 20, 15),
    ("orb", [f"orb{i:02d}" for i in range(1, 11)], 10, 10),
    ("swv", [f"swv{i:02d}" for i in range(1, 6)], 20, 10),
    ("swv", [f"swv{i:02d}" for i in range(6, 11)], 20, 15),
    ("swv", [f"swv{i:02d}" for i in range(11, 21)], 50, 10),
    ("yn", [f"yn{i}" for i in range(1, 5)], 20, 20),
]

#: Sizes for which networks are trained (Section 3.5.2).
TRAINING_SIZES = [(6, 6), (10, 10), (15, 15)]

EXPECTED_SIZE = {name: (n, m) for _, names, n, m in TABLE_3_3 for name in names}
FAMILY = {name: fam for fam, names, _, _ in TABLE_3_3 for name in names}


# ----------------------------------------------------------------------------------------- name subsets
def all_names() -> list[str]:
    """All 82 instances in the order of Table 3.3 (RQ1, RQ2 and RQ4)."""
    return [name for _, names, _, _ in TABLE_3_3 for name in names]


def rq3_names() -> list[str]:
    """The 24 instances whose size equals a training size (RQ3)."""
    return [name for name in all_names() if EXPECTED_SIZE[name] in TRAINING_SIZES]


def rq5_names() -> list[str]:
    """The first instance of each row of Table 3.3: 18 instances that cover every family and size (RQ5)."""
    return [names[0] for _, names, _, _ in TABLE_3_3]


def rq5_az_names() -> list[str]:
    """The six RQ5 instances whose size equals a training size, on which AZ can be run."""
    return [name for name in rq5_names() if EXPECTED_SIZE[name] in TRAINING_SIZES]


def select_names(spec) -> list[str]:
    """Resolve an ``instances`` entry of a configuration file to a list of instance names.

    ``all`` | ``rq3`` | ``rq5`` | ``rq5_az`` | ``small`` / ``medium`` / ``large`` | ``size:10x10`` |
    ``family:la`` | a single name | a list of any of these.
    """
    if isinstance(spec, (list, tuple)):
        out: list[str] = []
        for item in spec:
            for name in select_names(item):
                if name not in out:
                    out.append(name)
        return out
    spec = str(spec)
    if spec == "all":
        return all_names()
    if spec == "rq3":
        return rq3_names()
    if spec == "rq5":
        return rq5_names()
    if spec == "rq5_az":
        return rq5_az_names()
    if spec in ("small", "medium", "large"):
        from .instance import MEDIUM_MAX_OPS, SMALL_MAX_OPS

        def cls(name):
            ops = EXPECTED_SIZE[name][0] * EXPECTED_SIZE[name][1]
            return "small" if ops <= SMALL_MAX_OPS else "medium" if ops <= MEDIUM_MAX_OPS else "large"
        return [name for name in all_names() if cls(name) == spec]
    if spec.startswith("size:"):
        n, m = (int(v) for v in spec[5:].lower().split("x"))
        return [name for name in all_names() if EXPECTED_SIZE[name] == (n, m)]
    if spec.startswith("family:"):
        return [name for name in all_names() if FAMILY[name] == spec[7:]]
    if spec in EXPECTED_SIZE:
        return [spec]
    raise ValueError(f"unknown instance selection {spec!r}")


# ----------------------------------------------------------------------------------------------- parsing
def _rows_to_instance(name: str, header: list[str], rows: list[list[str]], **kw) -> Instance:
    try:
        n, m = int(header[0]), int(header[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{name}: the first data line must hold the numbers of jobs and machines") from exc
    if len(rows) != n:
        raise ValueError(f"{name}: expected {n} job rows, found {len(rows)}")
    machines, durations = [], []
    for j, row in enumerate(rows):
        if len(row) != 2 * m:
            raise ValueError(f"{name}: job {j} has {len(row)} numbers, expected {2 * m}")
        try:
            values = [int(v) for v in row]
        except ValueError as exc:
            raise ValueError(f"{name}: job {j} holds a value that is not an integer") from exc
        machines.append(values[0::2])
        durations.append(values[1::2])
    inst = Instance(name, machines, durations, **kw)
    verify_instance(inst)
    return inst


def parse_standard(text: str, name: str, **kw) -> Instance:
    """Parse one instance in the standard format (comment lines start with ``#``)."""
    lines = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{name}: empty instance file")
    n = int(lines[0][0])
    return _rows_to_instance(name, lines[0], lines[1: 1 + n], **kw)


def parse_orlib_collection(text: str) -> dict[str, Instance]:
    """Parse the OR-Library file ``jobshop1.txt``, which holds all instances one after another."""
    lines = text.splitlines()
    out: dict[str, Instance] = {}
    i = 0
    while i < len(lines):
        match = re.match(r"^\s*instance\s+(\S+)\s*$", lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1).lower()
        i += 1
        # the size line is the first line after the header that holds exactly two integers
        while i < len(lines) and not re.match(r"^\s*\d+\s+\d+\s*$", lines[i]):
            i += 1
        if i >= len(lines):
            raise ValueError(f"{name}: no size line found")
        header = lines[i].split()
        n = int(header[0])
        rows = [ln.split() for ln in lines[i + 1: i + 1 + n]]
        out[name] = _rows_to_instance(name, header, rows)
        i += 1 + n
    return out


# ----------------------------------------------------------------------------------------------- loading
def default_data_dir() -> Path:
    """``$MCTSJSSP_DATA`` if set, otherwise ``<repository>/data/orlib``."""
    env = os.environ.get("MCTSJSSP_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "orlib"


def load_bks(data_dir: Optional[Path] = None) -> dict[str, dict]:
    """Best-known makespans and lower bounds from JSPLib (Section 3.5.1)."""
    path = Path(data_dir or default_data_dir()) / "bks.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["instances"]


def load_instance(name: str, data_dir: Optional[Path] = None, bks: Optional[dict] = None) -> Instance:
    """Read one OR-Library instance, verify it against Table 3.3 and attach its BKS."""
    data_dir = Path(data_dir or default_data_dir())
    if name not in EXPECTED_SIZE:
        raise KeyError(f"{name} is not one of the 82 OR-Library instances of Table 3.3")
    bks = bks if bks is not None else load_bks(data_dir)
    entry = bks[name]
    path = data_dir / "instances" / f"{name}.txt"
    inst = parse_standard(path.read_text(encoding="utf-8"), name, bks=entry["upper_bound"],
                          lower_bound=entry["lower_bound"], family=FAMILY[name])
    if inst.shape != EXPECTED_SIZE[name]:
        raise ValueError(f"{name}: size {inst.shape} does not agree with Table 3.3 {EXPECTED_SIZE[name]}")
    return inst


@lru_cache(maxsize=4)
def _load_all_cached(data_dir: str) -> dict[str, Instance]:
    bks = load_bks(Path(data_dir))
    return {name: load_instance(name, Path(data_dir), bks) for name in all_names()}


def load_orlib(names: Optional[Iterable[str]] = None, data_dir: Optional[Path] = None) -> dict[str, Instance]:
    """Load the OR-Library test set (all 82 instances, or the named subset), cached per process."""
    everything = _load_all_cached(str(Path(data_dir or default_data_dir())))
    if names is None:
        return dict(everything)
    return {name: everything[name] for name in names}
