"""Fast policy networks for the rollouts of LR and AZ, after AlphaGo's fast rollout policy.

A rollout calls its policy at every step, so the cost of one call decides how many iterations the search
completes in a given time.  The full networks read the whole disjunctive graph (GIN) or the whole state
(fully connected) and cost tens of microseconds per call.  The fast networks read six cheap features of every
candidate operation, computed directly from the partial schedule:

1. work remaining of its job / the largest among the candidates            (as MWKR)
2. operations remaining of its job / m                                      (as MOR)
3. its processing time / the largest among the candidates                   (as SPT)
4. (its earliest start - the earliest start of any candidate) / LB0, where the earliest start is the later
   of its job's ready time and the end of the last operation on its machine
5. (its job's ready time - the earliest ready time of any candidate) / LB0  (as FIFO)
6. (its earliest completion - the earliest completion of any candidate) / LB0

They come from the operation's neighbourhood in the disjunctive graph (its job predecessor, the last
operation on its machine and the rest of its job), so fixed local features take the place of message
passing, as AlphaGo's local patterns did.

* :class:`FastLR` - one scorer shared by all candidates (6 -> 16, ReLU, 16 -> 1); reads any instance size.
* :class:`FastAZ` - the features of all jobs in job order as one vector (6n -> 32, ReLU, 32 -> n); one network
  per instance size.

``hidden = 0`` gives the linear versions (6 -> 1 and 6n -> n).  Like the GIN of LR and the networks of AZ, both
also have a value output, so that they are trained by the same PUCT self-play (Section 3.6.1); in the
experiments only the policy is used, to sample the rollouts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from ..env.model import State
from ..instances.instance import Instance

N_FEATURES = 6


# ------------------------------------------------------------------------------------------- features
def candidate_features(state: State) -> list[list[float]]:
    """The six features of every candidate, in the order of ``state.active``."""
    inst = state.inst
    nxt, ready, m_end = state.next_op, state.job_ready, state.m_end
    mach, dur, rem = inst.mach, inst.dur, inst.rem
    m = inst.m
    work, ops, proc, start, avail = [], [], [], [], []
    for j in state.active:
        k = nxt[j]
        ends = m_end[mach[j][k]]
        r = ready[j]
        a = ends[-1] if ends else 0
        work.append(rem[j][k])
        ops.append(m - k)
        proc.append(dur[j][k])
        start.append(r if r > a else a)
        avail.append(r)
    wmax = max(work) or 1
    pmax = max(proc) or 1
    smin = min(start)
    rmin = min(avail)
    cmin = min(s + p for s, p in zip(start, proc))
    inv_m = 1.0 / m
    inv_lb0 = 1.0 / inst.lb0
    return [[w / wmax, o * inv_m, p / pmax, (s - smin) * inv_lb0, (r - rmin) * inv_lb0, (s + p - cmin) * inv_lb0]
            for w, o, p, s, r in zip(work, ops, proc, start, avail)]


def state_features(state: State) -> np.ndarray:
    """``(n, 6)`` array: the features of every job's candidate, zeros for finished jobs (training records)."""
    out = np.zeros((state.inst.n, N_FEATURES), dtype=np.float32)
    if state.active:
        out[state.active] = candidate_features(state)
    return out


# --------------------------------------------------------------------------------------------- models
def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    if hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class FastLR(nn.Module):
    """Fast network of LR: a policy scorer applied to every candidate (shared weights, any size) and a value
    output on the mean of the candidates' hidden features.

    Like the GIN of LR it has a policy and a value output, so that it is trained by the same PUCT self-play;
    in the experiments only the policy is used, as the rollout distribution.
    """

    kind = "fast_lr"
    uses_features = True

    def __init__(self, hidden: int = 16, in_features: int = N_FEATURES):
        super().__init__()
        self.hparams = dict(hidden=hidden, in_features=in_features)
        self.scorer = _mlp(in_features, hidden, 1)                       # policy: 6 -> hidden -> 1 per candidate
        self.value_enc = (nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU()) if hidden > 0
                          else nn.Identity())                            # value: 6 -> hidden per candidate,
        self.value_out = nn.Linear(hidden if hidden > 0 else in_features, 1)   # mean over candidates -> 1

    def forward_features(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, n)`` from features ``(B, n, 6)``; finished jobs get ``-inf``."""
        return self.scorer(feats).squeeze(-1).masked_fill(~mask, float("-inf"))

    def value_features(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.value_enc(feats) * mask.unsqueeze(-1)
        h = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.tanh(self.value_out(h)).squeeze(-1)

    def forward_features_value(self, feats: torch.Tensor, mask: torch.Tensor):
        return self.forward_features(feats, mask), self.value_features(feats, mask)

    def policy_parameters(self) -> list:
        return list(self.scorer.parameters())


class FastAZ(nn.Module):
    """Fast networks of AZ: the features of all jobs as one vector; a policy network with an output per job and
    a separate value network with one output (one pair per size, like the networks of AZ).  Only the policy is
    used in the experiments, as the rollout distribution."""

    kind = "fast_az"
    uses_features = True

    def __init__(self, n: int, m: int, hidden: int = 32, in_features: int = N_FEATURES):
        super().__init__()
        self.n, self.m = int(n), int(m)
        self.hparams = dict(n=self.n, m=self.m, hidden=hidden, in_features=in_features)
        self.net = _mlp(in_features * self.n, hidden, self.n)           # policy
        self.value = _mlp(in_features * self.n, hidden, 1)              # value

    def _flat(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.shape[1] != self.n:
            raise ValueError(f"the fast AZ network was built for {self.n} jobs")
        return feats.reshape(feats.shape[0], -1)

    def forward_features(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(self._flat(feats)).masked_fill(~mask, float("-inf"))

    def value_features(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value(self._flat(feats))).squeeze(-1)

    def forward_features_value(self, feats: torch.Tensor, mask: torch.Tensor):
        return self.forward_features(feats, mask), self.value_features(feats, mask)

    def policy_parameters(self) -> list:
        return list(self.net.parameters())


# ------------------------------------------------------------------------------------------ inference
def _layers(module: nn.Module) -> list[tuple[np.ndarray, np.ndarray]]:
    linears = [module] if isinstance(module, nn.Linear) else [x for x in module if isinstance(x, nn.Linear)]
    return [(lin.weight.detach().cpu().numpy().astype(np.float32).T.copy(),
             lin.bias.detach().cpu().numpy().astype(np.float32).copy()) for lin in linears]


class FastInference:
    """Evaluates a fast network on single states without PyTorch.

    ``needs_tracker = False`` tells the rollout that no graph observation has to be kept.  ``score_list``
    returns the logits of the candidates as a Python list, which the fast rollout samples in pure Python.
    The linear version of the fast LR network is evaluated entirely in Python.
    """

    needs_tracker = False

    def __init__(self, model):
        self.model = model.eval()
        self.kind = model.kind
        self.calls = 0
        self._layers = _layers(model.scorer if isinstance(model, FastLR) else model.net)
        self.n: Optional[int] = getattr(model, "n", None)
        self.m: Optional[int] = getattr(model, "m", None)
        self._linear_lr = isinstance(model, FastLR) and len(self._layers) == 1
        if isinstance(model, FastLR):                           # value: per-candidate encoder, mean, output
            enc = model.value_enc
            self._v_enc = _layers(enc[0]) if isinstance(enc, nn.Sequential) else []
            self._v_out = _layers(model.value_out)
        else:
            self._v_enc = []
            self._v_out = _layers(model.value)
        if self._linear_lr:
            w, b = self._layers[0]
            self._w = [float(v) for v in w[:, 0]]
            self._b = float(b[0])

    def check(self, inst: Instance) -> None:
        if self.n is not None and inst.shape != (self.n, self.m):
            raise ValueError(f"the fast AZ network was trained for {self.n}x{self.m} instances and cannot be "
                             f"applied to {inst.name} ({inst.n}x{inst.m})")

    # rollout interface shared with the full networks (no observation to maintain)
    def begin(self, tracker, state: State) -> None:
        """Nothing to prepare."""

    def advance(self, tracker, state: State) -> None:
        """Nothing to update."""

    def score_list(self, state: State) -> list[float]:
        """Logits of the candidates, in the order of ``state.active``."""
        self.calls += 1
        feats = candidate_features(state)
        if self._linear_lr:
            w0, w1, w2, w3, w4, w5 = self._w
            b = self._b
            return [w0 * f[0] + w1 * f[1] + w2 * f[2] + w3 * f[3] + w4 * f[4] + w5 * f[5] + b for f in feats]
        if self.n is None:                                   # fast LR: the same scorer for every candidate
            h = np.array(feats, dtype=np.float32)
        else:                                                # fast AZ: one vector in job order
            h = np.zeros((self.n, N_FEATURES), dtype=np.float32)
            h[state.active] = feats
            h = h.reshape(-1)
        last = len(self._layers) - 1
        for i, (w, b) in enumerate(self._layers):
            h = h @ w
            h += b
            if i < last:
                np.maximum(h, 0.0, out=h)
        z = h[:, 0] if self.n is None else h[state.active]
        return z.tolist()

    def rollout_logits(self, tracker, state: State) -> np.ndarray:
        return np.asarray(self.score_list(state), dtype=np.float64)

    def logits(self, tracker, state: State) -> np.ndarray:
        return self.rollout_logits(tracker, state)

    def value(self, state: State, feats: Optional[list] = None) -> float:
        """Value output (tanh) of the state; used only by the PUCT search of the self-play training."""
        feats = candidate_features(state) if feats is None else feats
        if self.n is None:                                   # fast LR: encode, mean over candidates, output
            h = np.array(feats, dtype=np.float32)
            for w, b in self._v_enc:
                h = np.maximum(h @ w + b, 0.0)
            h = h.mean(axis=0)
        else:
            h = np.zeros((self.n, N_FEATURES), dtype=np.float32)
            h[state.active] = feats
            h = h.reshape(-1)
        last = len(self._v_out) - 1
        for i, (w, b) in enumerate(self._v_out):
            h = h @ w + b
            if i < last:
                h = np.maximum(h, 0.0)
        return float(np.tanh(h[0]))

    def evaluate(self, tracker, state: State):
        """Policy logits of the candidates and the value of the state, for PUCT (self-play only)."""
        logits = self.rollout_logits(tracker, state)
        return logits, self.value(state)
