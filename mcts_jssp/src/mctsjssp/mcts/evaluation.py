"""Return of a complete schedule (Sections 3.2.1 and 3.4.3).

In the static setting the return is ``R = -Cmax / LB0``.  In the stochastic setting (RQ5) the schedule is
first executed in a scenario sampled from the disturbance model, so that the mean value of a node estimates
the expected realised makespan.
"""
from __future__ import annotations

from ..env.model import State, execute
from ..instances.disturbance import ScenarioSampler
from ..instances.instance import Instance


class StaticReturn:
    """``R = -Cmax / LB0`` of the nominal schedule."""

    stochastic = False

    def __init__(self, inst: Instance):
        self.inst = inst
        self._scale = -1.0 / inst.lb0

    def __call__(self, state: State) -> float:
        return state.makespan * self._scale

    def makespan(self, state: State) -> float:
        return state.makespan


class StochasticReturn:
    """``R = -Cmax(scenario) / LB0`` with a scenario sampled from the disturbance model for every call."""

    stochastic = True

    def __init__(self, inst: Instance, sigma: float, seed: int):
        self.inst = inst
        self.sigma = float(sigma)
        self._scale = -1.0 / inst.lb0
        self._sampler = ScenarioSampler(inst, sigma, seed)

    def makespan(self, state: State) -> float:
        return execute(self.inst, state.m_op, self._sampler.sample())

    def __call__(self, state: State) -> float:
        return self.makespan(state) * self._scale


def make_return(inst: Instance, sigma: float | None = None, seed: int = 0):
    """Static return if ``sigma`` is None or 0, otherwise the return under sampled disturbances."""
    if not sigma:
        return StaticReturn(inst)
    return StochasticReturn(inst, sigma, seed)
