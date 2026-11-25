"""Run the study with the settings of study.yaml (Windows, Linux, Kaggle).

    python run_pipeline.py                         # every step listed under pipeline.steps in study.yaml
    python run_pipeline.py --steps run,analyse     # only these steps
    python run_pipeline.py --settings other.yaml   # another settings file
    python run_pipeline.py --dry-run               # print the commands only
    python run_pipeline.py --steps run --hours 10  # experiments for at most 10 hours, then stop cleanly

Every value under ``fixed``, ``settings`` and ``experiment_settings`` is passed to every command as
``--set KEY=VALUE``.  Finished networks and finished runs are skipped, so the same command can be run again
after an interruption or after ``hours_per_experiment`` has stopped an experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "configs" / "experiments"
STEPS = ["validate", "pilot", "budget_test", "tune", "train", "check", "self_play", "compare_training",
         "select_entropy", "disturbed", "run", "rules", "fixed_states", "analyse", "compare"]
SELF_PLAY_MODELS = ["lr_6x6", "lr_10x10", "lr_15x15", "az_6x6", "az_10x10", "az_15x15"]
DISTURBED_SIGMAS = [0.1, 0.2]
FINE_TUNED = ["lr_15x15", "az_6x6", "az_10x10", "az_15x15"]                # networks of RQ5, from their N
FROM_SCRATCH = ["lr_6x6", "lr_10x10", "lr_15x15", "az_6x6", "az_10x10", "az_15x15"]   # in this order


def sigma_tag(sigma: float) -> str:
    return f"_s{int(round(float(sigma) * 100)):03d}"


# ----------------------------------------------------------------------------------------------- settings
def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and "." not in k:     # a dotted key keeps its dict value whole
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def own_keys(config_file: Path) -> set:
    """Keys an experiment file sets itself (not inherited)."""
    with open(config_file, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("inherits", None)
    return set(flatten(raw))


def overrides_for(study: dict, experiment: str | None = None) -> dict:
    """fixed < settings < experiment_settings; values an experiment file sets itself are kept."""
    shared = {**flatten(study.get("fixed")), **flatten(study.get("settings"))}
    if experiment:
        mine = own_keys(EXPERIMENTS / f"{experiment}.yaml")
        shared = {k: v for k, v in shared.items()
                  if not any(k == m or k.startswith(m + ".") or m.startswith(k + ".") for m in mine)}
        shared.update(flatten((study.get("experiment_settings") or {}).get(experiment)))
    return shared


def set_args(values: dict) -> list:
    args = []
    for k, v in values.items():
        text = yaml.safe_dump(v, default_flow_style=True).strip()
        if text.endswith("..."):
            text = text[:-3].strip()
        args += ["--set", f"{k}={text}"]
    return args


# ----------------------------------------------------------------------------------------------- running
class Runner:
    def __init__(self, study: dict, dry_run: bool, workers: int | None = None, session_hours: float | None = None):
        self.study, self.dry = study, dry_run
        self.p = study.get("pipeline") or {}
        self.workers = int(workers or self.p.get("workers", os.cpu_count() or 1))
        hours = self.p.get("session_hours", 0) if session_hours is None else session_hours
        self.deadline = time.time() + float(hours) * 3600 if hours and float(hours) > 0 else None
        self.env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def cmd(self, *args, experiment: str | None = None, overrides: bool = True, last: dict | None = None,
            watch=None) -> None:
        """Run one command; ``last`` holds step-specific values, passed after the study's so that they win.
        ``watch`` is called every ``pipeline.check_minutes``; if it returns a reason, the command is stopped."""
        full = [sys.executable, "-W", "ignore", "-m", "mctsjssp", *map(str, args)]
        if overrides:
            full += set_args(overrides_for(self.study, experiment))
        full += set_args(last or {})
        shown = " ".join(a if " " not in a else f'"{a}"' for a in full[3:])
        print(f"\n>>> python {shown}", flush=True)
        if self.dry:
            return
        t0 = time.time()
        if watch is None:
            res = subprocess.run(full, cwd=ROOT, env=self.env)
        else:
            proc = subprocess.Popen(full, cwd=ROOT, env=self.env, start_new_session=True)
            every = float(self.p.get("check_minutes", 30)) * 60
            while True:
                try:
                    proc.wait(timeout=every)
                    break
                except subprocess.TimeoutExpired:
                    reason = watch()
                    if reason:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                        raise SystemExit(f"\nSTOPPED during training: {reason}\n"
                                         f"See artifacts/models/<name>/history.json; run training-report for curves.")
            res = proc
            reason = watch(final=True)                          # the last network finished after the check
            if reason:
                raise SystemExit(f"\nSTOPPED after training: {reason}")
        if res.returncode == 3 and args[0] == "training-report":
            raise SystemExit("\nSTOPPED before the experiments: at least one network has not learned. "
                             "See results/training/training_summary.csv and training_curves.png.")
        if res.returncode != 0:
            raise SystemExit(f"step failed ({res.returncode}): {shown}")
        print(f"    ({(time.time() - t0) / 60:.1f} min)", flush=True)

    def run_dir(self, experiment: str) -> Path:
        sys.path.insert(0, str(ROOT / "src"))
        from mctsjssp.experiments.config import load_config
        from mctsjssp.experiments.runner import run_directory

        overrides = [a for a in set_args(overrides_for(self.study, experiment)) if a != "--set"]
        return Path(run_directory(load_config(EXPERIMENTS / f"{experiment}.yaml", overrides)))

    # --------------------------------------------------------------------------------------- the steps
    def validate(self):
        self.cmd("validate-env", "--config", "configs/base.yaml")

    def pilot(self):
        self.cmd("pilot", "--config", "configs/pilot.yaml",
                 last={"pilot.workers": self.workers, "pilot.available_hours": self.p.get("available_hours", 40)})

    def budget_test(self):
        out = ROOT / "artifacts_budget_test"
        self.cmd("train", "--config", "configs/base.yaml", "--model", "lr_15x15", "--workers", self.workers,
                 "--force", last={"training.max_generations": 2, "training.budget_model_steps": None,
                                  "training.min_generations": 0, "paths.artifacts": out.as_posix()})
        if self.dry:
            return
        hist = json.loads((out / "models" / "lr_15x15" / "history.json").read_text(encoding="utf-8"))
        s0, s1 = hist[0]["model_steps"], hist[1]["model_steps"]
        n = int(self.p.get("budget_generations", 50))
        budget = int(s0 + (n - 1) * (s1 - s0))
        print(f"\n    model steps: generation 0 = {s0:,}, one later generation = {s1 - s0:,}")
        print(f"    suggested training budget ({n} generations of lr_15x15): {budget:,}")
        print(f"    -> set  training.budget_model_steps: {budget}  under 'settings' in study.yaml")

    def tune(self):
        for conf in ("LR", "AZ"):
            self.cmd("tune", "--config", "configs/base.yaml", "--configuration", conf, "--parallel", self.workers)

    def artifacts_dir(self) -> Path:
        sys.path.insert(0, str(ROOT / "src"))
        from mctsjssp.experiments.config import load_config

        overrides = [a for a in set_args(overrides_for(self.study)) if a != "--set"]
        path = Path(load_config(ROOT / "configs" / "base.yaml", overrides)["paths"]["artifacts"])
        return path if path.is_absolute() else ROOT / path

    def must_learn(self, name: str) -> bool:
        """stop_if_not_learned: true (every network), false (none) or a list of name prefixes, e.g. [lr_, flr_]."""
        stop = self.p.get("stop_if_not_learned", True)
        if isinstance(stop, (list, tuple)):
            return any(name.startswith(str(prefix)) for prefix in stop)
        return bool(stop)

    def watch_training(self, final: bool = False):
        """Status of the networks every check_minutes; a reason to stop when a finished network has not learned."""
        models = self.artifacts_dir() / "models"
        need = float(self.p.get("min_improvement_pct", 0.0))
        done, failed, current = [], [], None
        for d in sorted(models.glob("*")) if models.exists() else []:
            summ, hist = d / "summary.json", d / "history.json"
            if summ.exists():
                s = json.loads(summ.read_text(encoding="utf-8"))
                gain = (s["uniform_ratio"] - s["best_validation_ratio"]) / s["uniform_ratio"] * 100
                done.append(f"{d.name} {gain:+.1f}%")
                unit = "iterations" if s.get("method") == "self_label" else "generations"
                if gain <= need and self.must_learn(d.name):
                    failed.append(f"{d.name} ({gain:+.1f}% vs uniform random after {s['generations']} {unit})")
            elif hist.exists():
                h = json.loads(hist.read_text(encoding="utf-8"))
                if h:
                    r = h[-1]
                    if "greedy_ratio" in r:                 # self-labelling
                        current = (f"{d.name} iteration {r['generation']}: val {r['validation_ratio']:.4f} (greedy "
                                   f"{r['greedy_ratio']:.4f}, sampled {r['sampled_ratio']:.4f}; uniform "
                                   f"{r['uniform_ratio']:.4f})")
                    else:
                        current = (f"{d.name} gen {r['generation']}: val {r['validation_ratio']:.4f} "
                                   f"(uniform {r['uniform_ratio']:.4f}), target entropy "
                                   f"{r.get('target_entropy', float('nan')):.3f}")
        stamp = time.strftime("%H:%M")
        print(f"\n[check {stamp}] finished {len(done)}: {', '.join(done) or '-'}", flush=True)
        if current and not final:
            print(f"[check {stamp}] training now: {current}", flush=True)
        if failed:
            return "not learned: " + "; ".join(failed)
        return None

    def train(self):
        self.cmd("train", "--config", "configs/base.yaml", "--all", "--workers", self.workers,
                 watch=self.watch_training)

    def check(self):
        # curves + which networks learned; stops the pipeline before the experiments if one has not
        stop = self.p.get("stop_if_not_learned", True)
        strict = ["--strict"] if stop else []
        if isinstance(stop, (list, tuple)):                  # only these networks must have learned
            strict += [a for prefix in stop for a in ("--require", str(prefix))]
        self.cmd("training-report", "--config", "configs/base.yaml", *strict,
                 "--min-improvement", self.p.get("min_improvement_pct", 0.0))
        self.cmd("validate-env", "--config", "configs/base.yaml", "--learning")
        self.cmd("assess", "--config", "configs/base.yaml", "--all")

    def self_play(self):
        """AlphaZero-style self-play of the networks under self_play.models, the training style that
        self-labelling is compared with (step compare_training): from random weights and untuned, into
        self_play.artifacts.  Early stopping ends it soon if it does not learn; it never stops the pipeline, and
        finished networks are skipped when the step runs again."""
        sp = self.study.get("self_play") or {}
        models = [a for name in sp.get("models", SELF_PLAY_MODELS) for a in ("--model", name)]
        last = {"paths.artifacts": sp.get("artifacts", "artifacts/self_play"), "training.method": "alphazero",
                **flatten(sp.get("settings"))}
        self.cmd("train", "--config", "configs/base.yaml", *models, "--workers", self.workers, last=last)

    def disturbed(self):
        """The networks for disturbed processing times (RQ5), each validated in the fixed scenarios of the
        disturbances: fine-tuned from the network for nominal processing times (artifacts/models) and trained with
        disturbances from scratch (artifacts/scratch: LR through the sizes, AZ copying that LR); then both, with the
        nominal networks, U and MWKR, under disturbances on the validation instances (robustness) and their
        development over the iterations (disturbed-report).  Networks validated without disturbances by an earlier
        version are moved to artifacts/old_disturbed first; finished networks are skipped when the step runs again."""
        ds = self.study.get("disturbed") or {}
        sigmas = [float(v) for v in ds.get("sigmas", DISTURBED_SIGMAS)]
        models_dir, old_dir = ROOT / "artifacts" / "models", ROOT / "artifacts" / "old_disturbed"
        fine_tuned = [f"{n}{sigma_tag(s)}" for s in sigmas for n in FINE_TUNED]
        for name in fine_tuned:
            d, summary = models_dir / name, models_dir / name / "summary.json"
            if summary.exists():
                with open(summary, encoding="utf-8") as fh:
                    validated = json.load(fh).get("validation")
                if validated != "disturbed":
                    target = old_dir / name
                    if target.exists():
                        target = old_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
                    old_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(d), str(target))
                    print(f"[{name}] validated without disturbances: moved to {target}", flush=True)
        self.cmd("train", "--config", "configs/base.yaml", *[a for n in fine_tuned for a in ("--model", n)],
                 "--workers", self.workers)
        scratch = [f"{n}{sigma_tag(s)}" for s in sigmas for n in FROM_SCRATCH]
        specs = {}
        for s in sigmas:
            for n in FROM_SCRATCH:
                family, size = n.split("_")
                k, m = (int(v) for v in size.split("x"))
                specs[f"models.{n}{sigma_tag(s)}"] = {"kind": "gin" if family == "lr" else "ffnn", "n": k, "m": m,
                                                      "sigma": s}
        last = {"paths.artifacts": ds.get("scratch", "artifacts/scratch"),
                "training.reference_models": "artifacts/models", **specs}
        self.cmd("train", "--config", "configs/base.yaml", *[a for n in scratch for a in ("--model", n)],
                 "--workers", self.workers, last=last)
        self.cmd("robustness", "--config", "configs/base.yaml", *[a for s in sigmas for a in ("--sigma", s)],
                 "--workers", self.workers)
        self.cmd("disturbed-report", "--config", "configs/base.yaml")

    def compare_training(self):
        """Self-labelling against self-play: the same validation of both, table and figure in results/training."""
        sp = self.study.get("self_play") or {}
        models = [a for name in sp.get("models", SELF_PLAY_MODELS) for a in ("--model", name)]
        self.cmd("compare-training", "--config", "configs/base.yaml", *models, "--other",
                 sp.get("artifacts", "artifacts/self_play"), "--workers", self.workers)

    def select_entropy(self):
        self.cmd("select-entropy", "--config", "configs/base.yaml", "--workers", self.workers)

    def left_hours(self) -> float | None:
        return None if self.deadline is None else (self.deadline - time.time()) / 3600

    def run(self):
        for e in self.p.get("experiments", []):
            left = self.left_hours()
            if left is not None and left < 0.05:
                print(f"\n    session time used up before {e}; run the same command again to continue")
                raise SystemExit(0)
            extra = ["--hours", f"{left:.2f}"] if left is not None else []
            self.cmd("run", "--config", f"configs/experiments/{e}.yaml", "--workers", self.workers, *extra,
                     experiment=e)

    def rules(self):
        """Extension of RQ1 and RQ2 (configs/experiments/rules.yaml): MOR, SPT and FIFO as fixed rollout
        distributions.  Their entropy levels are chosen on the validation instances as for MWKR (with the score of
        U; the levels of MWKR, LR and AZ are kept), then their runs; U, MWKR and LR come from RQ2."""
        self.cmd("select-entropy", "--config", "configs/experiments/rules.yaml", "--workers", self.workers,
                 "--missing", experiment="rules")
        left = self.left_hours()
        if left is not None and left < 0.05:
            print("\n    session time used up before rules; run the same command again to continue")
            raise SystemExit(0)
        extra = ["--hours", f"{left:.2f}"] if left is not None else []
        self.cmd("run", "--config", "configs/experiments/rules.yaml", "--workers", self.workers, *extra,
                 experiment="rules")

    def fixed_states(self):
        self.cmd("fixed-states", "--config", "configs/experiments/rq3.yaml", "--workers", self.workers,
                 experiment="rq3")

    def analyse(self):
        names = list(self.p.get("experiments", []))
        if "rules" not in names and (self.dry or (self.run_dir("rules") / "results.jsonl").exists()):
            names.append("rules")                 # the extension, once its runs exist (step rules)
        for e in names:
            extra = []
            if e == "rules":                      # analysed with U, MWKR and LR of RQ2
                extra = ["--with", "<results of rq2>" if self.dry else str(self.run_dir("rq2"))]
            self.cmd("analyse", "--config", f"configs/experiments/{e}.yaml", *extra, experiment=e)

    def compare(self):
        names = list(self.p.get("compare", []))
        dirs = [] if self.dry else [self.run_dir(e) for e in names]
        found = [d for d in dirs if (d / "results.jsonl").exists()]
        args = [a for d in found for a in ("--run-dir", d)] if not self.dry else \
            [a for e in names for a in ("--run-dir", f"<results of {e}>")]
        if not args:
            print("\n    compare: no finished experiments yet")
            return
        self.cmd("compare", *args, "--out", "results/compare", overrides=False)
        self.cmd("compare-states", *args, "--out", "results/visited_states", overrides=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default="study.yaml", help="settings file (default study.yaml)")
    ap.add_argument("--steps", help=f"comma-separated steps to run, out of: {', '.join(STEPS)}")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--workers", type=int, help="override pipeline.workers")
    ap.add_argument("--hours", type=float, help="override pipeline.session_hours (stop the experiments after H hours)")
    args = ap.parse_args()
    with open(ROOT / args.settings, encoding="utf-8") as fh:
        study = yaml.safe_load(fh) or {}
    steps = args.steps.split(",") if args.steps else list((study.get("pipeline") or {}).get("steps", STEPS))
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; choose from {STEPS}")
    runner = Runner(study, args.dry_run, args.workers, args.hours)
    print(f"settings: {args.settings}   workers: {runner.workers}   steps: {', '.join(steps)}")
    t0 = time.time()
    for step in steps:
        print(f"\n=== {step} " + "=" * (60 - len(step)), flush=True)
        getattr(runner, step)()
    print(f"\nDone in {(time.time() - t0) / 3600:.2f} h.  Results: results/, analyses: results/<experiment>/analysis/, "
          f"comparisons: results/compare/ and results/visited_states/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
