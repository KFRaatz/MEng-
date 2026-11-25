"""Artifacts shared by the experiments: trained networks, temperatures, tuned entropy levels, time budgets.

Everything that is fixed before the main experiments (Section 3.6.3) lives under one directory::

    artifacts/
      models/<name>/best.pt        trained networks (with history.json and summary.json)
      temperatures.json            temperature of every (distribution, model, size, entropy level)
      entropy_levels.json          entropy level used outside RQ3 for MWKR, LR and AZ (and MOR, SPT, FIFO)
      time_budgets.json            T50 and T200 of every instance (time that U needs per decision)
      tuning/<name>/tuning.json    random-search trials
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..distributions.base import RolloutPolicy
from ..distributions.calibration import calibrate_tau
from ..distributions.fixed import FIXED_RULES, make_rule_rollout
from ..distributions.learned import make_learned_rollout
from ..instances.generator import validation_set
from ..networks.checkpoint import file_fingerprint, load_checkpoint
from ..networks.inference import make_inference
from .config import get


class JsonStore:
    """A small dictionary persisted as JSON (written atomically)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                self.data = json.load(fh)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        tmp.replace(self.path)


_INFERENCE_CACHE: dict = {}


def load_inference(path: str | Path, backend: str = "numpy"):
    """Inference object of a checkpoint, cached per process."""
    key = (str(path), backend)
    if key not in _INFERENCE_CACHE:
        model, meta = load_checkpoint(path)
        infer = make_inference(model, backend)
        infer.meta = meta
        _INFERENCE_CACHE[key] = infer
    return _INFERENCE_CACHE[key]


class Artifacts:
    def __init__(self, cfg: dict, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.root = Path(cfg["paths"]["artifacts"])
        self.log = log
        self.temperatures = JsonStore(self.root / "temperatures.json")
        self.entropy_levels = JsonStore(self.root / "entropy_levels.json")
        self.time_budgets = JsonStore(self.root / "time_budgets.json")
        self.backend = cfg.get("inference_backend", "numpy")

    # ---------------------------------------------------------------------------------------- models
    def model_dir(self, name: str) -> Path:
        return self.root / "models" / name

    def model_path(self, name: str) -> Path:
        return self.model_dir(name) / "best.pt"

    def has_model(self, name: str) -> bool:
        """True when the training of the network finished: best.pt and summary.json (written at the end).  A
        network whose training was interrupted has best.pt only; it is trained again (from the start)."""
        path = self.model_path(name)
        return path.exists() and (path.parent / "summary.json").exists()

    def inference(self, name: str):
        path = self.model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"no trained network '{name}' at {path}; train it first "
                                    f"(python -m mctsjssp train ...)")
        return load_inference(path, self.backend)

    def model_size(self, name: str) -> tuple[int, int]:
        """Training size of a network, from the metadata of its checkpoint."""
        meta = self.inference(name).meta
        conf = meta.get("config", {})
        return int(conf["n"]), int(conf["m"])

    # ---------------------------------------------------------------------------------- temperatures
    def temperature(self, name: str, level: float, size: Sequence[int], model: Optional[str] = None) -> Optional[float]:
        """Temperature of a distribution at an entropy level, calibrated on the validation instances of
        ``size`` (found by bisection on first use and stored).

        Returns None for level 0 (the greedy rule) and infinity for level 1 (uniform sampling).
        """
        level = float(level)
        if level <= 0.0:
            return None
        if level >= 1.0:
            return float("inf")
        key, fingerprint = self._temperature_key(name, level, size, model)
        tau = self._stored_tau(key, fingerprint)
        if tau is not None:
            return tau
        return self._store_temperature(key, self.calibration_entry(name, level, size, model))

    def _temperature_key(self, name: str, level: float, size: Sequence[int], model: Optional[str]) -> tuple:
        n, m = int(size[0]), int(size[1])
        key = f"{name}|{model or '-'}|{n}x{m}|H{float(level):g}"
        return key, (file_fingerprint(self.model_path(model)) if model else None)

    def _stored_tau(self, key: str, fingerprint: Optional[str]) -> Optional[float]:
        entry = self.temperatures.get(key)
        if entry is not None and entry.get("fingerprint") == fingerprint:
            return float(entry["tau"])
        return None

    def _store_temperature(self, key: str, entry: dict) -> float:
        self.temperatures.set(key, entry)
        self.log(f"temperature {key}: tau = {entry['tau']:.5g}, realised entropy {entry['realised']:.3f}"
                 f"{'' if entry['converged'] else ' (NOT within tolerance)'}")
        return float(entry["tau"])

    def calibration_entry(self, name: str, level: float, size: Sequence[int], model: Optional[str] = None) -> dict:
        """Calibrate the temperature of a level by bisection (0 < level < 1) and return the entry; nothing is
        stored.  The result depends only on the settings and the network, not on the process that computes it."""
        level = float(level)
        n, m = int(size[0]), int(size[1])
        instances = validation_set(n, m)[: int(get(self.cfg, "entropy.calibration.instances", 100))]
        if name in FIXED_RULES:
            def make(tau: float) -> RolloutPolicy:
                return make_rule_rollout(name, level, tau)
        else:
            infer = self.inference(model)

            def make(tau: float) -> RolloutPolicy:
                return make_learned_rollout(name, infer, level, tau)
        t0 = time.perf_counter()
        cal = calibrate_tau(make, instances, level,
                            tolerance=float(get(self.cfg, "entropy.calibration.tolerance", 0.01)),
                            rollouts_per_instance=int(get(self.cfg, "entropy.calibration.rollouts_per_instance", 1)),
                            seed=int(self.cfg.get("seed", 0)),
                            start_depths=int(get(self.cfg, "entropy.calibration.start_depths", 10)))
        entry = cal.as_dict()
        entry.update(fingerprint=file_fingerprint(self.model_path(model)) if model else None,
                     instances=len(instances), seconds=time.perf_counter() - t0)
        return entry

    def calibrate_missing(self, requests: Sequence[tuple], pool=None) -> int:
        """Calibrate the temperatures of ``requests`` ((name, level, size, model) tuples) that are not stored yet,
        in the worker processes of ``pool`` (one calibration per process, the longest first), and store them,
        so that ``temperature`` finds them.  Every calibration has its own fixed seed, so the temperatures are
        the ones that ``temperature`` finds one after the other.  Returns the number of calibrations."""
        todo: dict = {}
        for name, level, size, model in requests:
            level = float(level)
            if not 0.0 < level < 1.0 or size is None:
                continue
            key, fingerprint = self._temperature_key(name, level, size, model)
            if key not in todo and self._stored_tau(key, fingerprint) is None:
                todo[key] = (name, level, (int(size[0]), int(size[1])), model)
        if pool is None or len(todo) < 2:
            for name, level, size, model in todo.values():
                self.temperature(name, level, size, model)
            return len(todo)
        order = sorted(todo, key=lambda k: -_calibration_cost(*todo[k]))
        self.log(f"calibrating {len(todo)} temperatures in parallel")
        entries = pool.map(_calibration_task, [(self.cfg, *todo[k]) for k in order], chunksize=1)
        for key, entry in zip(order, entries):
            self._store_temperature(key, entry)
        return len(todo)

    # -------------------------------------------------------------------------------- entropy levels
    def tuned_level(self, name: str) -> float:
        """Entropy level used outside RQ3 for a configuration (chosen by ``select-entropy``)."""
        entry = self.entropy_levels.get(name)
        if entry is None:
            raise KeyError(f"no tuned entropy level for {name}; run 'python -m mctsjssp select-entropy' first "
                           f"or give the level in the configuration file")
        return float(entry["best"])


# ------------------------------------------------------------------------------- calibration in workers
def _calibration_cost(name: str, level: float, size: Sequence[int], model: Optional[str]) -> float:
    """Rough relative cost of a calibration, to start the longest ones first (a network costs ~30 x MWKR)."""
    return float(size[0] * size[1]) * (1.0 if name in FIXED_RULES else 30.0)


def _calibration_task(task: tuple) -> dict:
    """One calibration in a worker process (top-level, so that it can be sent to the process)."""
    cfg, name, level, size, model = task
    return Artifacts(cfg, log=lambda *_: None).calibration_entry(name, level, size, model)
