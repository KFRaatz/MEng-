"""Self-play episodes that produce the training records of LR and AZ (Section 3.6.1).

In each generation the MCTS agent solves newly generated instances, one episode per instance.  After the
search of every decision the visit counts at the root are recorded as the policy target ``pi_t`` and the most
visited operation is dispatched.  When the schedule is complete its outcome ``Cmax / LB0`` decides the ranked
reward ``z`` of all records of the episode.

* Generation 0 uses the searches of U (UCT with uniform rollouts), so that the first networks learn from
  searches without learned guidance.
* Later generations search with PUCT: the policy network of the previous generation gives the priors and its
  value network evaluates the leaves instead of a rollout; Dirichlet noise is added at the root.

For RQ5 the searches are performed with disturbances: complete schedules are executed in a sampled scenario
before their outcome is computed.
"""
from __future__ import annotations

import io
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..distributions.fixed import UniformRollout
from ..env.environment import JobShopEnv
from ..env.model import State
from ..instances.instance import Instance
from ..mcts.evaluation import make_return
from ..mcts.puct import PUCTSearch
from ..mcts.uct import UCTSearch
from ..networks.checkpoint import build_model
from ..networks.fast import state_features
from ..networks.inference import make_inference


@dataclass
class EpisodeData:
    """Records of one episode, still without the ranked reward."""

    x: list = field(default_factory=list)
    mpred: list = field(default_factory=list)
    next_op: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    pi: list = field(default_factory=list)
    f: list = field(default_factory=list)       # (n, 6) features of the fast networks
    v: list = field(default_factory=list)       # value of the root after each search, scale of search_value
    outcome: float = 0.0                 # Cmax / LB0 of the episode (realised in a scenario for RQ5)
    makespan: int = 0
    model_steps: int = 0
    evaluations: int = 0

    def __len__(self) -> int:
        return len(self.pi)


def _record(data: EpisodeData, env: JobShopEnv, pi: np.ndarray) -> None:
    x, mpred = env.observation().snapshot()
    data.x.append(x)
    data.mpred.append(mpred)
    data.next_op.append(np.array(env.state.next_op, dtype=np.int16))
    data.mask.append(env.action_mask())
    data.pi.append(pi)
    data.f.append(state_features(env.state))


def search_value(ratio: float) -> float:
    """Value of a schedule on the scale of the value networks: ``3 - 2 Cmax/LB0`` within [-1, 1].

    1 is a schedule at the lower bound LB0, 0 one at 1.5 LB0 and -1 one at 2 LB0 or worse.  It is linear in
    the return ``R = -Cmax/LB0``, so the mean value of a node is the value of its mean return.
    """
    return max(-1.0, min(1.0, 3.0 - 2.0 * float(ratio)))


def _root_visits(root, n_jobs: int) -> np.ndarray:
    pi = np.zeros(n_jobs, dtype=np.float32)
    for child in root.children:
        pi[child.action] = child.N
    total = pi.sum()
    return pi / total if total > 0 else pi


def play_uct_episode(inst: Instance, iterations: int, seed: int, c: float = 1.0,
                     sigma: Optional[float] = None) -> EpisodeData:
    """One episode of generation 0: the search of U (UCT selection, uniform rollouts)."""
    env = JobShopEnv(inst)
    state = env.reset()
    env.observation()
    evaluate = make_return(inst, sigma, seed)
    search = UCTSearch(UniformRollout(), evaluate, c=c, seed=seed)
    search.start(state)
    data = EpisodeData()
    while not env.done:
        if len(state.active) == 1:
            action = state.active[0]                       # no decision, no record
        else:
            search.run_iterations(state, iterations)
            _record(data, env, _root_visits(search.root, inst.n))
            data.v.append(search_value(-search.root.W / search.root.N))   # mean return -> value scale
            action = search.choose()
        env.step(action)
        search.advance(action, state)
    data.makespan = env.makespan
    data.outcome = evaluate.makespan(state) / inst.lb0
    data.model_steps = search.model_steps + inst.n_ops
    return data


def play_puct_episode(inst: Instance, infer, threshold: float, iterations: int, seed: int, c: float = 1.0,
                      noise_weight: float = 0.25, noise_alpha_scale: float = 10.0,
                      sigma: Optional[float] = None, q_init: str = "zero", leaf_rollouts: bool = False,
                      mix: float = 0.5, value_target: str = "outcome", normalize_q: bool = False) -> EpisodeData:
    """One episode of a later generation: PUCT with the priors and values of the current networks.

    With ``value_target = "outcome"`` a complete schedule (reached in the tree or by a leaf rollout) is valued
    by the ranked reward: +1 if its ``Cmax / LB0`` beats ``threshold``, else -1.  With ``"search"`` it is
    valued by :func:`search_value`, the scale on which the value networks then learn the root values of the
    searches.  ``leaf_rollouts`` completes every new leaf by a uniform random rollout (weight ``mix``).
    """
    env = JobShopEnv(inst)
    state = env.reset()
    env.observation()
    evaluate = make_return(inst, sigma, seed)
    lb0 = inst.lb0

    if value_target == "search":
        def terminal_value(st: State) -> float:
            return search_value(evaluate.makespan(st) / lb0)
    else:
        def terminal_value(st: State) -> float:
            return 1.0 if evaluate.makespan(st) / lb0 < threshold else -1.0

    search = PUCTSearch(infer, terminal_value, c=c, noise_weight=noise_weight,
                        noise_alpha_scale=noise_alpha_scale, seed=seed, q_init=q_init,
                        rollout=UniformRollout() if leaf_rollouts else None, mix=mix, normalize_q=normalize_q)
    search.start(state)
    data = EpisodeData()
    while not env.done:
        if len(state.active) == 1:
            action = state.active[0]
        else:
            search.run_iterations(state, iterations)
            _record(data, env, search.visit_distribution(inst.n))
            data.v.append(search.root.W / search.root.N)
            action = search.choose()
        env.step(action)
        search.advance(action, state)
    data.makespan = env.makespan
    data.outcome = evaluate.makespan(state) / lb0
    data.model_steps = search.model_steps + inst.n_ops
    data.evaluations = search.evaluations
    return data


# ------------------------------------------------------------------------------------- parallel episodes
def pack_model(model) -> bytes:
    """Serialise a network so that worker processes can rebuild it."""
    import torch

    buffer = io.BytesIO()
    torch.save({"kind": model.kind, "hparams": dict(model.hparams),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, buffer)
    return buffer.getvalue()


def unpack_model(blob: bytes):
    import torch

    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    model = build_model(payload["kind"], payload["hparams"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


_WORKER: dict = {}


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _worker_episode(task: dict) -> EpisodeData:
    inst = Instance(task["name"], task["machines"], task["durations"])
    if task["model"] is None:
        return play_uct_episode(inst, task["iterations"], task["seed"], task["c"], task["sigma"])
    key = hash(task["model"])
    if _WORKER.get("key") != key:
        _WORKER["key"] = key
        _WORKER["infer"] = make_inference(unpack_model(task["model"]), task["backend"])
    return play_puct_episode(inst, _WORKER["infer"], task["threshold"], task["iterations"], task["seed"],
                             task["c"], task["noise_weight"], task["noise_alpha_scale"], task["sigma"],
                             task["q_init"], task["leaf_rollouts"], task["mix"], task["value_target"],
                             task["normalize_q"])


class EpisodeRunner:
    """Plays the episodes of a generation, in this process or in a pool of worker processes.

    The result does not depend on the number of workers: every episode has its own seed and the ranked
    reward threshold is fixed for the generation.
    """

    def __init__(self, workers: int = 1):
        self.workers = max(1, int(workers))
        self._pool = None
        if self.workers > 1:
            self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def run(self, instances: Sequence[Instance], seeds: Sequence[int], iterations: int, c: float,
            sigma: Optional[float], model=None, infer=None, threshold: Optional[float] = None,
            noise_weight: float = 0.25, noise_alpha_scale: float = 10.0, backend: str = "numpy",
            q_init: str = "zero", leaf_rollouts: bool = False, mix: float = 0.5,
            value_target: str = "outcome", normalize_q: bool = False) -> list:
        if self._pool is None:
            out = []
            for inst, seed in zip(instances, seeds):
                if model is None:
                    out.append(play_uct_episode(inst, iterations, seed, c, sigma))
                else:
                    out.append(play_puct_episode(inst, infer, threshold, iterations, seed, c, noise_weight,
                                                 noise_alpha_scale, sigma, q_init, leaf_rollouts, mix,
                                                 value_target, normalize_q))
            return out
        blob = None if model is None else pack_model(model)
        tasks = [dict(name=inst.name, machines=inst.machines, durations=inst.durations, seed=seed,
                      iterations=iterations, c=c, sigma=sigma, model=blob, threshold=threshold,
                      noise_weight=noise_weight, noise_alpha_scale=noise_alpha_scale, backend=backend,
                      q_init=q_init, leaf_rollouts=leaf_rollouts, mix=mix, value_target=value_target,
                      normalize_q=normalize_q)
                 for inst, seed in zip(instances, seeds)]
        return self._pool.map(_worker_episode, tasks, chunksize=1)
