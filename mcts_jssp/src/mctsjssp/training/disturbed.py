"""Networks for disturbed processing times (RQ5): their validation under disturbances and how their rollout
distribution changes during training.

Validation.  A network for disturbances is judged as RQ5 judges a schedule: its greedy schedule and its
sampled schedules of every validation instance are executed in the same ``count`` fixed scenarios of the
disturbance model (:func:`mctsjssp.instances.disturbance.scenarios`), and the mean realised ``Cmax / LB0``
over the scenarios is its score.  The scenarios are the same in every iteration and for every network, so
iterations and networks are compared on the same disturbances.

Distribution.  On fixed probe states (states of uniform random schedules of the validation instances, the same
for every network of a size) the policy at temperature 1 is described by its mean normalised entropy, the share
of states on which its most probable job is the job of MWKR, and the share on which it is the job of a
reference network, the network for nominal processing times of the same kind and size.
"""
from __future__ import annotations

import math
import random
import re
import zlib
import numpy as np

from ..distributions.fixed import MWKRRollout
from ..env.model import State, execute
from ..env.observation import FeatureTracker
from ..instances.disturbance import scenarios
from ..instances.instance import Instance

VALIDATION_SCENARIOS = 30
PROBES_PER_INSTANCE = 10
SIGMA_SUFFIX = re.compile(r"_s\d{3}$")


def base_name(name: str) -> str:
    """The network for nominal processing times of a network for disturbances: ``lr_15x15_s010`` ->
    ``lr_15x15`` (the name stays as it is for other networks)."""
    return SIGMA_SUFFIX.sub("", name)


def network_seed(seed: int, name: str) -> int:
    """The training seed of a network.  A network for disturbances has the seed of its nominal network, so that
    trained from scratch it starts from the same random weights and sees the same training instances."""
    return int(np.random.SeedSequence([int(seed), zlib.crc32(base_name(name).encode())]).generate_state(1)[0])


_SCENARIOS: dict = {}


def _scenarios(inst: Instance, sigma: float, count: int) -> list:
    key = (inst.name, inst.n, inst.m, float(sigma), int(count))
    got = _SCENARIOS.get(key)
    if got is None:
        if len(_SCENARIOS) > 4096:
            _SCENARIOS.clear()
        got = _SCENARIOS[key] = scenarios(inst, float(sigma), int(count))
    return got


def realised(inst: Instance, state: State, sigma: float, count: int = VALIDATION_SCENARIOS) -> np.ndarray:
    """Realised makespans of the complete schedule ``state`` in the ``count`` fixed scenarios, divided by LB0."""
    return np.array([execute(inst, state.m_op, d) for d in _scenarios(inst, float(sigma), int(count))],
                    dtype=np.float64) / inst.lb0


def baseline_task(task: dict) -> list:
    """Uniform random dispatching (``repeats`` schedules) and greedy MWKR of every instance, executed in the fixed
    scenarios: per instance (uniform realised, MWKR realised, uniform nominal, MWKR nominal), all / LB0."""
    out = []
    mwkr = MWKRRollout(0.0)
    for inst, seed in zip(task["instances"], task["seeds"]):
        rnd = random.Random(seed)
        u_real, u_nom = [], []
        for _ in range(int(task["repeats"])):
            state = State(inst)
            while state.active:
                state.step(state.active[int(rnd.random() * len(state.active))])
            u_real.append(float(realised(inst, state, task["sigma"], task["scenarios"]).mean()))
            u_nom.append(state.makespan / inst.lb0)
        state = State(inst)
        mwkr.rollout(state, rnd, [])
        out.append((float(np.mean(u_real)), float(realised(inst, state, task["sigma"], task["scenarios"]).mean()),
                    float(np.mean(u_nom)), state.makespan / inst.lb0))
    return out


# ------------------------------------------------------------------------------------------ probe states
def probe_states(instances, seed: int, per_instance: int = PROBES_PER_INSTANCE) -> list[State]:
    """``per_instance`` states of a uniform random schedule of every instance, evenly spaced over the decisions
    with at least two jobs to choose from; the same states for the same instances and seed."""
    out = []
    for i, inst in enumerate(instances):
        rnd = random.Random(int(np.random.SeedSequence([seed, 4_242, i]).generate_state(1)[0]))
        state, sequence, depths = State(inst), [], []
        while state.active:
            if len(state.active) > 1:
                depths.append(len(sequence))
            j = state.active[int(rnd.random() * len(state.active))]
            sequence.append(j)
            state.step(j)
        if not depths:
            continue
        picks = sorted({depths[int(round(k))] for k in np.linspace(0, len(depths) - 1, per_instance)})
        for d in picks:
            s = State(inst)
            for j in sequence[:d]:
                s.step(j)
            out.append(s)
    return out


def mwkr_choices(states: list[State]) -> np.ndarray:
    rule = MWKRRollout(0.0)
    return np.array([int(np.argmax(rule.scores(s))) for s in states], dtype=np.int64)


def network_logits(infer, states: list[State]) -> list[np.ndarray]:
    """Logits of the network over the jobs that can be dispatched (in the order of ``state.active``)."""
    trackers: dict = {}
    out = []
    for s in states:
        tr = trackers.get(id(s.inst))
        if tr is None:
            infer.check(s.inst)
            tr = trackers[id(s.inst)] = FeatureTracker(s.inst)
        out.append(np.asarray(infer.logits(tr.reset(s), s), dtype=np.float64))
    return out


def choices(logits: list[np.ndarray]) -> np.ndarray:
    return np.array([int(np.argmax(z)) for z in logits], dtype=np.int64)


def probe_metrics(logits: list[np.ndarray], mwkr: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """Mean normalised entropy at temperature 1 and the agreement of the most probable job with MWKR and with
    the reference network."""
    ent = []
    for z in logits:
        p = np.exp(z - z.max())
        p /= p.sum()
        ent.append(float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / math.log(len(z))))
    top = choices(logits)
    out = {"probe_entropy": float(np.mean(ent)), "agree_mwkr": float(np.mean(top == mwkr))}
    out["agree_reference"] = float(np.mean(top == reference)) if reference is not None else float("nan")
    return out
