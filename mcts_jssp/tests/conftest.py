"""Shared fixtures of the test suite."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# worker processes are started with "spawn": they find the package through PYTHONPATH
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

from mctsjssp.env import FeatureTracker, State  # noqa: E402
from mctsjssp.instances import Instance, generate_instance, load_orlib  # noqa: E402
from mctsjssp.networks import AZNetworks, GINPolicyValue  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def orlib() -> dict:
    return load_orlib()


@pytest.fixture(scope="session")
def ft06(orlib) -> Instance:
    return orlib["ft06"]


@pytest.fixture()
def tiny() -> Instance:
    """A 4 x 3 instance with a zero processing time, small enough to check by hand."""
    machines = [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]
    durations = [[3, 2, 4], [2, 0, 5], [4, 3, 1], [1, 6, 2]]
    return Instance("tiny", machines, durations)


def random_instance(n: int, m: int, seed: int, low: int = 0, high: int = 20) -> Instance:
    return generate_instance(n, m, np.random.default_rng(seed), low=low, high=high, name=f"rand{n}x{m}_{seed}")


def random_state(inst: Instance, steps: int, seed: int) -> State:
    rnd = random.Random(seed)
    state = State(inst)
    for _ in range(min(steps, inst.n_ops)):
        state.step(rnd.choice(state.active))
    return state


def _warm_up(model, n: int, m: int) -> None:
    """A few training-mode passes so that the batch-normalisation statistics are not the initial ones."""
    torch.manual_seed(0)
    model.train()
    n_ops = n * m
    for _ in range(3):
        x = torch.rand(8, n_ops, 2)
        model.forward_batch(x, torch.full((8, n_ops), n_ops), torch.zeros(8, n, dtype=torch.long),
                            torch.ones(8, n, dtype=torch.bool))
    model.eval()


@pytest.fixture(scope="session")
def gin() -> GINPolicyValue:
    torch.manual_seed(1)
    model = GINPolicyValue()
    _warm_up(model, 6, 6)
    return model


@pytest.fixture(scope="session")
def az66() -> AZNetworks:
    torch.manual_seed(2)
    return AZNetworks(6, 6).eval()


@pytest.fixture()
def tracker_of():
    def make(state: State) -> FeatureTracker:
        return FeatureTracker(state.inst).reset(state)
    return make
