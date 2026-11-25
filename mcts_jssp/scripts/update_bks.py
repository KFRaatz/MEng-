#!/usr/bin/env python3
"""Rebuild ``data/orlib`` from a local copy of JSPLib (https://github.com/ScheduleOpt/benchmarks).

    python scripts/update_bks.py /path/to/benchmarks [--instances]

Without ``--instances`` only ``bks.json`` (best-known makespans and lower bounds) is rewritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mctsjssp.instances.orlib import all_names  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="local clone of ScheduleOpt/benchmarks")
    parser.add_argument("--instances", action="store_true", help="also copy the 82 instance files")
    args = parser.parse_args()
    job_shop = args.repository / "jobshop"
    with open(job_shop / "solutions" / "bks.json", encoding="utf-8") as fh:
        by_name = {entry["instance"]: entry for entry in json.load(fh)}
    names = all_names()
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"no best-known value for {missing}")
    commit = subprocess.run(["git", "-C", str(args.repository), "log", "-1", "--format=%H %cs"],
                            capture_output=True, text=True).stdout.split()
    out = {
        "_source": "JSPLib, ScheduleOpt/benchmarks (https://github.com/ScheduleOpt/benchmarks), "
                   "file jobshop/solutions/bks.json",
        "_commit": commit[0] if commit else "unknown",
        "_commit_date": commit[1] if len(commit) > 1 else "unknown",
        "_licence": "CC BY-SA 4.0",
        "_note": "upper_bound is the best-known makespan (BKS); status 'closed' means lower_bound == "
                 "upper_bound (optimal).",
        "instances": {n: {"lower_bound": by_name[n]["lower_bound"], "upper_bound": by_name[n]["upper_bound"],
                          "status": by_name[n]["status"]} for n in names},
    }
    target = ROOT / "data" / "orlib"
    with open(target / "bks.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"bks.json: {len(names)} instances, commit {out['_commit'][:10]} of {out['_commit_date']}")
    if args.instances:
        files = {p.stem: p for p in (job_shop / "instances" / "text").rglob("*.txt")}
        for name in names:
            shutil.copy(files[name], target / "instances" / f"{name}.txt")
        print(f"copied {len(names)} instance files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
