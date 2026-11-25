"""Pilot study on AWS SageMaker (or any machine), in plain Python.

Run it from the mcts_jssp folder (the one with pyproject.toml):

    python sagemaker_pilot.py                    # install, quick tests, environment checks, pilot, zip the results
    python sagemaker_pilot.py --smoke            # also the smoke run of every command (about 10 minutes)
    python sagemaker_pilot.py --skip-install     # when the packages are already installed
    python sagemaker_pilot.py --hours 40         # hours available for the experiments (default 40)
    python sagemaker_pilot.py --all              # the pilot and then the whole study in one go

or in a notebook cell:   exec(open("sagemaker_pilot.py").read())

Output: results/pilot/*.csv, logs/pilot.log and pilot_results.zip (to download).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOGS = ROOT / "logs"


def sh(cmd: list, log=None, check: bool = True) -> int:
    """Run a command, print its output as it comes and append it to the log."""
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log:
            log.write(line)
    code = proc.wait()
    if check and code != 0:
        raise SystemExit(f"failed ({code}): {' '.join(map(str, cmd))}")
    return code


def physical_cores() -> int:
    try:
        out = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True)
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="also run every command with tiny settings")
    ap.add_argument("--workers", type=int, help="default: one per physical core (24 on ml.c6i.12xlarge)")
    ap.add_argument("--hours", type=float, default=40.0, help="hours available for the experiments")
    ap.add_argument("--all", action="store_true",
                    help="after the pilot: training (self-labelling and the self-play comparison), checks, "
                         "temperatures, experiments and analysis")
    ap.add_argument("--run-hours", type=float, help="with --all: stop the experiments after this many hours")
    args = ap.parse_args(argv)

    if sys.version_info < (3, 10):
        raise SystemExit(f"Python {sys.version.split()[0]} is too old: use a kernel/image with Python 3.10+")
    os.chdir(ROOT)
    LOGS.mkdir(exist_ok=True)
    py = sys.executable
    t0 = time.time()
    with open(LOGS / "pilot.log", "a", encoding="utf-8") as log:
        # 1. install --------------------------------------------------------------------------------
        if not args.skip_install:
            sh([py, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], log)
            sh([py, "-m", "pip", "install", "-q", "torch_geometric"], log)
            sh([py, "-m", "pip", "install", "-q", "-e", ".[test]"], log)

        # 2. workers and settings (study.yaml) --------------------------------------------------------
        import yaml

        cores = physical_cores()
        workers = args.workers or max(cores, 1)       # all physical cores run workers, as in the pilot (Section 4.2)
        study = yaml.safe_load(open("study.yaml", encoding="utf-8"))
        study.setdefault("pipeline", {})["workers"] = workers
        study["pipeline"]["available_hours"] = args.hours
        yaml.safe_dump(study, open("study.yaml", "w", encoding="utf-8"), sort_keys=False)
        msg = f"{os.cpu_count()} vCPUs, {cores} physical cores -> {workers} workers"
        print(msg)
        log.write(msg + "\n")

        # 3. tests ----------------------------------------------------------------------------------
        sh([py, "-m", "pytest", "-q", "-m", "not slow"], log)
        if args.smoke:
            sh(["bash", "scripts/smoke.sh"], log)
            shutil.rmtree(ROOT / "smoke_out", ignore_errors=True)

        # 4. pilot: environment checks and pilot study ------------------------------------------------
        sh([py, "run_pipeline.py", "--steps", "validate,pilot"], log)

        # 5. (--all) the rest of the study -------------------------------------------------------------
        if args.all:
            steps = "train,check,self_play,compare_training,select_entropy,disturbed,run,fixed_states,analyse,compare"
            sh([py, "run_pipeline.py", "--steps", steps]
               + (["--hours", str(args.run_hours)] if args.run_hours else []), log)
            for name, folder in (("study_results", "results"), ("study_artifacts", "artifacts"),
                                 ("study_logs", "logs")):
                if (ROOT / folder).exists():
                    shutil.make_archive(str(ROOT / name), "zip", str(ROOT), folder)
                    print(f"{name}.zip written")

    # 5. results ------------------------------------------------------------------------------------
    out = ROOT / "pilot_results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "results" / "pilot").glob("*.csv")):
            z.write(f, f"pilot/{f.name}")
        z.write(LOGS / "pilot.log", "pilot/pilot.log")
    try:
        import pandas as pd

        for name in ("design", "training"):
            print(f"\n=== {name} ===")
            print(pd.read_csv(ROOT / "results" / "pilot" / f"{name}.csv").round(3).to_string(index=False))
    except Exception:
        pass
    print(f"\nDone in {(time.time() - t0) / 60:.0f} min.  Download {out} and send it back "
          f"(it holds the pilot tables and the log).")
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:        # a notebook cell: exec(open("sagemaker_pilot.py").read())
        main([])
    else:
        sys.exit(main())
