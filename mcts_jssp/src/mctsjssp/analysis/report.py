"""Writes the tables and figures of an experiment (Section 3.6.5: scripts that reproduce Chapter 4).

``write_report`` reads ``results.jsonl`` of a run directory, runs the analysis of its research question and
stores every table as CSV and every figure as PNG and PDF under ``<run directory>/analysis``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .metrics import load_frame
from .plots import make_figures
from .rq import analyse


def write_report(run_dir: str | Path, experiment: str, alpha: float = 0.05, zero_method: str = "zsplit",
                 figures: bool = True, log: Callable[[str], None] = print, extra_dirs=(),
                 entropy_levels: Optional[dict] = None, temperatures: Optional[dict] = None) -> dict:
    """Analyse the results in ``run_dir`` (with the rows of ``extra_dirs``, for example the U, MWKR and LR runs
    of RQ2 for the rules extension); the tables and figures go to ``run_dir/analysis``.  ``entropy_levels``
    (artifacts/entropy_levels.json) names the best fixed distribution on the validation instances."""
    run_dir = Path(run_dir)
    frame = load_frame(run_dir)
    for other in extra_dirs or ():
        more = load_frame(other)
        log(f"  with {len(more)} runs of {Path(other).name}")
        frame = pd.concat([frame, more], ignore_index=True)
    tables = analyse(frame, experiment, alpha, zero_method, run_dir, entropy_levels=entropy_levels,
                     temperatures=temperatures)
    from .anytime import anytime_summary, anytime_table
    tables["anytime"] = anytime_table(frame)                 # best makespan so far against time
    tables["anytime_summary"] = anytime_summary(tables["anytime"])
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.to_csv(out_dir / f"{name}.csv", index=False)
    runs = len(frame)
    log(f"{experiment}: {runs} runs, {frame['instance'].nunique()} instances, "
        f"{frame['label'].nunique()} configurations, {frame['budget'].nunique()} budgets")
    written = [n for n, t in tables.items() if isinstance(t, pd.DataFrame) and not t.empty]
    log(f"  tables: {', '.join(written)} -> {out_dir}")
    if figures:
        paths = make_figures(experiment, frame, tables, out_dir)
        if paths:
            log(f"  figures: {', '.join(sorted({p.stem for p in paths}))}")
    return tables
