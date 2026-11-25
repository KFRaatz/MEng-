"""Generated training and validation instances (Section 3.5.2).

Instances follow the method of Taillard: every job visits the machines in a random order and the processing
times are drawn uniformly from the integers 1..99.  (Taillard's own linear congruential generator, which
reproduces his published benchmark instances, is not needed here: only the distribution matters.)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .instance import Instance

#: Base seed of the fixed validation sets; training instances use the seed of the training run instead.
VALIDATION_SEED = 20_261_116
VALIDATION_COUNT = 100


def generate_instance(n: int, m: int, rng: np.random.Generator, low: int = 1, high: int = 99,
                      name: str | None = None) -> Instance:
    """One random ``n x m`` instance with processing times uniform on ``low..high``."""
    durations = rng.integers(low, high + 1, size=(n, m))
    machines = np.argsort(rng.random((n, m)), axis=1)  # an independent random permutation per job
    return Instance(name or f"gen{n}x{m}", machines, durations, family="generated")


@lru_cache(maxsize=16)
def validation_set(n: int, m: int, count: int = VALIDATION_COUNT, seed: int = VALIDATION_SEED) -> tuple[Instance, ...]:
    """The fixed set of validation instances of one size, used for tuning and model selection."""
    out = []
    for i in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, i]))
        out.append(generate_instance(n, m, rng, name=f"val{n}x{m}_{i:03d}"))
    return tuple(out)


def training_instances(n: int, m: int, count: int, seed: int, generation: int) -> list[Instance]:
    """New training instances for one generation of self-play; never reused between generations."""
    out = []
    for i in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, generation, i, 7]))
        out.append(generate_instance(n, m, rng, name=f"train{n}x{m}_g{generation}_{i}"))
    return out
