"""Expansion of an experiment file into single runs (Section 3.6.3, Table 3.6).

An experiment file lists instances, configurations, budgets and the number of runs.  :func:`plan` expands
them into one :class:`Planned` entry per run - instance x configuration x entropy level x budget x
disturbance level x run - without touching any artifact, so that the pilot study can cost a design before
anything is trained.  :func:`resolve` then attaches what a run needs: the temperature of its entropy level,
the checkpoint of its network, the seconds of its time budget and its seed.

Configuration entries::

    - {name: U}
    - {name: MWKR, entropy: tuned}                        # level chosen on the validation instances
    - {name: MWKR, entropy: [0, 0.25, 0.5, 0.75], calibration: own_size}
    - {name: MOR, entropy: tuned}                         # also SPT and FIFO (extension of RQ1 and RQ2)
    - {name: LR, model: lr_15x15, entropy: tuned}
    - {name: LR, model: "lr_{size}", entropy: [0, 0.5]}    # the network of the size of the instance
    - {name: AZ, model: "az_{size}_s{sigma}", instances: rq5_az}

``{size}`` is replaced by the size of the instance (``10x10``) and ``{sigma}`` by the disturbance level
(``010`` for 0.1).  Budgets are ``{iterations: N}`` or ``{time: N}``, where ``time: N`` is T_N, the time per
decision that U needs for N iterations on the same instance.
"""
from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..distributions.fixed import FIXED_RULES
from ..instances.orlib import EXPECTED_SIZE, select_names
from .artifacts import Artifacts
from .config import get

LEARNED = ("LR", "AZ")
KNOWN = ("U",) + FIXED_RULES + LEARNED       # FIXED_RULES: MWKR, MOR, SPT, FIFO


@dataclass(frozen=True)
class Planned:
    instance: str
    name: str                     # U | MWKR | MOR | SPT | FIFO | LR | AZ
    variant: str                  # configuration as written in the file, e.g. "LR[lr_{size}]"
    level: Optional[float]        # normalised entropy level (1.0 for U)
    model: Optional[str]          # name of the trained network
    calibration: Optional[tuple]  # size of the validation set used for the temperature
    budget_kind: str              # "iterations" | "time"
    budget_value: int             # N iterations, or N of T_N
    run: int
    sigma: Optional[float]
    c: float

    @property
    def label(self) -> str:
        return self.variant if self.name == "U" else f"{self.variant}@H{self.level:g}"

    @property
    def budget_label(self) -> str:
        return f"I{self.budget_value}" if self.budget_kind == "iterations" else f"T{self.budget_value}"

    @property
    def key(self) -> str:
        sigma = f"{self.sigma:g}" if self.sigma else "0"
        return f"{self.instance}|{self.label}|{self.budget_label}|s{sigma}|r{self.run}"


def sigma_tag(sigma: Optional[float]) -> str:
    return f"{int(round((sigma or 0.0) * 100)):03d}"


def _size_from_model_name(model: str) -> Optional[tuple]:
    match = re.search(r"(\d+)x(\d+)", model)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _levels(spec: dict, artifacts: Optional[Artifacts], tuned_default: float) -> list:
    if spec["name"] == "U":
        return [1.0]
    entropy = spec.get("entropy", "tuned")
    if entropy == "tuned":
        return [artifacts.tuned_level(spec["name"]) if artifacts is not None else tuned_default]
    if isinstance(entropy, (int, float)):
        return [float(entropy)]
    return [float(v) for v in entropy]


def parse_budgets(cfg: dict) -> list[tuple]:
    out = []
    for item in cfg.get("budgets", []):
        if "iterations" in item:
            out.append(("iterations", int(item["iterations"])))
        elif "time" in item:
            out.append(("time", int(item["time"])))
        else:
            raise ValueError(f"budget {item!r} needs 'iterations' or 'time'")
    if not out:
        raise ValueError("the experiment defines no budgets")
    return out


def plan(cfg: dict, artifacts: Optional[Artifacts] = None, tuned_default: float = 0.5) -> list[Planned]:
    """Expand an experiment configuration into its runs.

    Without ``artifacts`` (pilot study) a tuned entropy level is replaced by ``tuned_default`` and the
    training size of a network is read from its name.
    """
    names = select_names(cfg["instances"])
    budgets = parse_budgets(cfg)
    runs = int(cfg.get("runs", 5))
    sigmas = get(cfg, "stochastic.sigmas") or [None]
    default_size = tuple(get(cfg, "entropy.default_size", [15, 15]))
    c_default = float(get(cfg, "search.c", 1.0))
    out: list[Planned] = []
    for spec in cfg["configurations"]:
        name = spec["name"]
        if name not in KNOWN:
            raise ValueError(f"unknown configuration {name!r}; expected one of {KNOWN}")
        allowed = set(select_names(spec["instances"])) if "instances" in spec else None
        template = spec.get("model")
        if name in LEARNED and not template:
            raise ValueError(f"configuration {name} needs a 'model'")
        variant = spec.get("label") or (name if not template else f"{name}[{template}]")
        c = float(spec.get("c", c_default))
        if "c" in spec and "label" not in spec:
            variant = f"{variant}(c={c:g})"
        for sigma in sigmas:
            sigma = float(sigma) if sigma else None
            for inst_name in names:
                if allowed is not None and inst_name not in allowed:
                    continue
                n, m = EXPECTED_SIZE[inst_name]
                model = template.format(size=f"{n}x{m}", sigma=sigma_tag(sigma)) if template else None
                if name in FIXED_RULES:
                    cal = spec.get("calibration", default_size)
                    calibration = (n, m) if cal == "own_size" else tuple(int(v) for v in cal)
                elif name in LEARNED:
                    if artifacts is not None:
                        calibration = artifacts.model_size(model)
                    else:
                        calibration = _size_from_model_name(model) or (n, m)
                else:
                    calibration = None
                for level in _levels(spec, artifacts, tuned_default):
                    for kind, value in budgets:
                        for run in range(runs):
                            out.append(Planned(inst_name, name, variant, level, model, calibration, kind, value,
                                               run, sigma, c))
    keys = [p.key for p in out]
    if len(set(keys)) != len(keys):
        raise ValueError("two configurations of the experiment have the same label; give one a 'label'")
    return out


def job_seed(master: int, key: str) -> int:
    """Seed of a run, derived from the master seed and the key of the run (stable across processes)."""
    instance_part, run_part = key.rsplit("|r", 1)
    ss = np.random.SeedSequence([int(master), zlib.crc32(instance_part.encode("utf-8")), int(run_part)])
    return int(ss.generate_state(1)[0])


def run_identifier(experiment: str, key: str) -> str:
    return f"{experiment}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def temperature_requests(planned: list[Planned]) -> list[tuple]:
    """The temperatures that :func:`resolve` looks up, as (name, level, validation size, model) tuples, so that
    they can be calibrated in parallel beforehand (``Artifacts.calibrate_missing``)."""
    out = []
    for p in planned:
        if not 0.0 < p.level < 1.0 or p.calibration is None:
            continue
        if p.name in FIXED_RULES:
            out.append((p.name, p.level, p.calibration, None))
        elif p.name in LEARNED:
            out.append((p.name, p.level, p.calibration, p.model))
            out.append(("MWKR", p.level, p.calibration, None))       # pi' of Equation 2.22
    return out


def resolve(cfg: dict, planned: list[Planned], artifacts: Artifacts, time_seconds: dict) -> list[dict]:
    """Turn planned runs into jobs: attach temperatures, checkpoints, time budgets, seeds and recording.

    ``time_seconds`` maps ``(instance, N, sigma)`` to the entry of the time budget T_N.
    """
    experiment = cfg["experiment"]
    master = int(cfg.get("seed", 0))
    level_iter = get(cfg, "recording.level", "decisions")
    level_time = get(cfg, "recording.timed_level", level_iter)
    profile_mode = get(cfg, "time_budget.profile", "mean")
    jobs = []
    for p in planned:
        tau = None
        ref_tau = None
        model_path = None
        if p.name in FIXED_RULES:
            tau = artifacts.temperature(p.name, p.level, p.calibration)
        elif p.name in LEARNED:
            model_path = str(artifacts.model_path(p.model))
            tau = artifacts.temperature(p.name, p.level, p.calibration, model=p.model)
            if 0.0 < p.level < 1.0:
                # MWKR distribution of the same level and validation size: pi' of Equation 2.22
                ref_tau = artifacts.temperature("MWKR", p.level, p.calibration)
        budget = {"kind": p.budget_kind, "label": p.budget_label, "iterations": 0, "seconds": 0.0, "profile": None}
        if p.budget_kind == "iterations":
            budget["iterations"] = p.budget_value
        else:
            entry = time_seconds[(p.instance, p.budget_value, p.sigma)]
            budget["seconds"] = float(entry["mean"])
            if profile_mode == "per_decision":
                budget["profile"] = [float(v) for v in entry["per_decision"]]     # (a, b)
        jobs.append({
            "experiment": experiment,
            "key": p.key,
            "run_id": run_identifier(experiment, p.key),
            "seed": job_seed(master, p.key),
            "planned": asdict(p),
            "label": p.label,
            "tau": tau,
            "ref_tau": ref_tau,
            "model_path": model_path,
            "budget": budget,
            "record_level": level_iter if p.budget_kind == "iterations" else level_time,
            "step_rate": float(get(cfg, "recording.step_rate", 0.001)),
            "sequence_rate": float(get(cfg, "recording.rollout_sequences", 0.0)),
            "skip_forced": bool(get(cfg, "search.skip_forced", True)),
            "backend": cfg.get("inference_backend", "numpy"),
            "scenarios": int(get(cfg, "stochastic.scenarios", 30)),
            "data_dir": cfg["paths"]["data"],
            "store_sequence": bool(get(cfg, "recording.store_sequence", True)),
        })
    return jobs
