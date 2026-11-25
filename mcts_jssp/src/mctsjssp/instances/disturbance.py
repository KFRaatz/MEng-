"""Processing-time disturbances for the stochastic setting of RQ5 (Section 3.5.3).

An operation with nominal processing time ``p`` takes ``max(0, round(p * (1 + eps)))`` time units, where
``eps`` is drawn independently for every operation from a normal distribution with mean 0 and standard
deviation ``sigma``.
"""
from __future__ import annotations

import zlib

import numpy as np

from .instance import Instance

#: Base seed of the fixed evaluation scenarios, so that every configuration sees the same realisations.
SCENARIO_SEED = 5_150_000
SCENARIO_COUNT = 30


def disturb(durations: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Realised processing times for one scenario (same shape as ``durations``)."""
    eps = rng.standard_normal(durations.shape) * sigma
    return np.maximum(0, np.rint(durations * (1.0 + eps))).astype(np.int64)


def _name_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def scenario(inst: Instance, sigma: float, index: int, seed: int = SCENARIO_SEED) -> list[int]:
    """Fixed evaluation scenario ``index`` of an instance: realised durations indexed by operation."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, _name_key(inst.name), int(round(sigma * 1000)), index]))
    return disturb(inst.durations, sigma, rng).ravel().tolist()


def scenarios(inst: Instance, sigma: float, count: int = SCENARIO_COUNT, seed: int = SCENARIO_SEED) -> list[list[int]]:
    """The ``count`` fixed scenarios of one instance and disturbance level."""
    return [scenario(inst, sigma, i, seed) for i in range(count)]


class ScenarioSampler:
    """Samples scenarios from the disturbance model during the search.

    The scheduler knows the nominal times and the disturbance model but not the realisations, so this
    sampler is seeded independently of the evaluation scenarios.
    """

    def __init__(self, inst: Instance, sigma: float, seed: int):
        self.sigma = float(sigma)
        self._flat = inst.durations.ravel().astype(np.float64)
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 9_091]))

    def sample(self) -> list[int]:
        eps = self._rng.standard_normal(self._flat.shape[0])
        return np.maximum(0, np.rint(self._flat * (1.0 + self.sigma * eps))).astype(np.int64).tolist()
