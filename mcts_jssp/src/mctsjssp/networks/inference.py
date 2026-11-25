"""Evaluation of the trained networks on single states, as needed by the rollouts and by self-play.

A rollout of LR or AZ calls its policy network once per step, so the cost of one call decides how many
iterations the search completes in a given time.  The default backend therefore exports the weights of a
trained network to NumPy (with the batch normalisation of the GIN folded into the preceding linear layers)
and evaluates single states without the per-call overhead of PyTorch.  The ``torch`` backend evaluates the
PyTorch module itself; both give the same numbers (see ``tests/test_networks.py``).

Both inference classes offer

* ``logits(tracker, state)`` - logits of the legal actions, in the order of ``state.active``;
* ``evaluate(tracker, state)`` - the same logits and the value of the state (used by PUCT in self-play);
* ``begin`` / ``rollout_logits`` / ``advance`` - the same logits along a rollout.  One step of a rollout
  changes the features of the operations of one job and at most two machine arcs, so only a handful of node
  embeddings of the GIN change.  After a full evaluation at the start of the rollout (``begin``), ``advance``
  recomputes just those embeddings, layer by layer, and the pooled sum.  The result equals a full
  evaluation up to rounding (see ``tests/test_networks.py``); on large instances it is several times
  cheaper.  The networks of AZ are evaluated in full at every step, which is already cheap.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..env.model import State
from ..env.observation import FeatureTracker
from ..instances.instance import Instance
from .ffnn import AZNetworks
from .gin import GINPolicyValue


def _fold(linear: nn.Linear, norm: nn.BatchNorm1d | None) -> tuple[np.ndarray, np.ndarray]:
    """Weights for ``a @ W + b`` with an evaluation-mode batch normalisation folded in."""
    w = linear.weight.detach().cpu().numpy().astype(np.float32).T.copy()
    b = linear.bias.detach().cpu().numpy().astype(np.float32).copy()
    if norm is not None:
        scale = (norm.weight / torch.sqrt(norm.running_var + norm.eps)).detach().cpu().numpy().astype(np.float32)
        shift = (norm.bias - norm.running_mean * norm.weight / torch.sqrt(norm.running_var + norm.eps))
        shift = shift.detach().cpu().numpy().astype(np.float32)
        w = w * scale[None, :]
        b = b * scale + shift
    return np.ascontiguousarray(w), b


def _linears(seq: nn.Sequential) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_fold(layer, None) for layer in seq if isinstance(layer, nn.Linear)]


def _mish(a: np.ndarray) -> np.ndarray:
    return a * np.tanh(np.logaddexp(0.0, a))


class GINInference:
    """Single-state evaluation of the GIN of LR."""

    kind = "gin"

    def __init__(self, model: GINPolicyValue, backend: str = "numpy"):
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        self.backend = backend
        self.model = model.eval()
        self.calls = 0
        if backend == "numpy":
            self._layers = []
            for conv, outer in zip(model.convs, model.norms):
                mods = list(conv.nn)
                layer = []
                for i, mod in enumerate(mods):
                    if isinstance(mod, nn.Linear):
                        nxt = mods[i + 1] if i + 1 < len(mods) else None
                        layer.append(_fold(mod, nxt if isinstance(nxt, nn.BatchNorm1d) else outer))
                self._layers.append(layer)
            self._policy = _linears(model.policy_head)
            self._value = _linears(model.value_head)
        self._h: list = []                  # per layer: padded node embeddings of the current rollout
        self._sum = None                    # sum of the final node embeddings (float64)

    def check(self, inst: Instance) -> None:
        """The GIN reads graphs of any size."""

    # ----------------------------------------------------------------------- incremental (rollouts)
    @staticmethod
    def _mlp(layer, a: np.ndarray) -> np.ndarray:
        for w, b in layer:                      # every folded linear layer is followed by a ReLU
            a = a @ w
            a += b
            np.maximum(a, 0.0, out=a)
        return a

    def begin(self, tracker: FeatureTracker, state: State) -> None:
        """Full evaluation of the encoder at the start of a rollout."""
        if self.backend != "numpy":
            return
        n_ops = state.inst.n_ops
        jpred, mpred = tracker.jpred, tracker.mpred
        h_in = tracker.xp
        self._h = []
        for layer in self._layers:
            out = self._mlp(layer, h_in[:n_ops] + h_in[jpred] + h_in[mpred])
            padded = np.zeros((n_ops + 1, out.shape[1]), dtype=np.float32)
            padded[:n_ops] = out
            self._h.append(padded)
            h_in = padded
        self._sum = self._h[-1][:n_ops].sum(axis=0, dtype=np.float64)

    def advance(self, tracker: FeatureTracker, state: State) -> None:
        """Update the embeddings after one step (call after ``tracker.update``)."""
        if self.backend != "numpy":
            return
        inst = state.inst
        m, n_ops = inst.m, inst.n_ops
        op, _, _ = state.last
        jpred, mpred, msucc = tracker.jpred, tracker.mpred, tracker.msucc
        # layer 1: the features of the operations op .. end of its job changed, and the machine successor
        # of op has a new machine predecessor
        dirty = list(range(op, (op // m + 1) * m))
        succ = int(msucc[op])
        if succ < n_ops:
            dirty.append(succ)
        h_in = tracker.xp
        last = len(self._layers) - 1
        for depth, layer in enumerate(self._layers):
            idx = np.array(dirty, dtype=np.int64)
            new = self._mlp(layer, h_in[idx] + h_in[jpred[idx]] + h_in[mpred[idx]])
            h_out = self._h[depth]
            if depth == last:
                self._sum += new.sum(axis=0, dtype=np.float64) - h_out[idx].sum(axis=0, dtype=np.float64)
            h_out[idx] = new
            if depth < last:                    # the next layer also changes at the successors of these nodes
                seen = set(dirty)
                for v in dirty[:]:
                    nxt = v + 1
                    if nxt % m and nxt not in seen:
                        seen.add(nxt)
                        dirty.append(nxt)
                    ms = int(msucc[v])
                    if ms < n_ops and ms not in seen:
                        seen.add(ms)
                        dirty.append(ms)
            h_in = h_out

    def rollout_logits(self, tracker: FeatureTracker, state: State) -> np.ndarray:
        """Logits of the legal actions from the embeddings maintained by ``begin`` and ``advance``."""
        if self.backend != "numpy":
            return self.logits(tracker, state)
        self.calls += 1
        m = state.inst.m
        nxt = state.next_op
        cand = [j * m + nxt[j] for j in state.active]
        h_graph = (self._sum / state.inst.n_ops).astype(np.float32)
        return self._policy_np(self._h[-1], h_graph, cand)

    # ------------------------------------------------------------------------------------- numpy
    def _encode(self, tracker: FeatureTracker) -> np.ndarray:
        h = tracker.x
        jpred, mpred = tracker.jpred, tracker.mpred
        for layer in self._layers:
            padded = np.concatenate([h, np.zeros((1, h.shape[1]), dtype=np.float32)])
            a = h + padded[jpred]
            a += padded[mpred]
            h = self._mlp(layer, a)
        return h

    def _policy_np(self, h: np.ndarray, h_graph: np.ndarray, cand: list[int]) -> np.ndarray:
        (w0, b0), *hidden, (w_out, b_out) = self._policy
        d = h.shape[1]
        z = np.tanh(h[cand] @ w0[:d] + (h_graph @ w0[d:] + b0))     # [h_a; h_G] @ W without concatenating
        for w, b in hidden:
            z = np.tanh(z @ w + b)
        return (z @ w_out + b_out)[:, 0]

    def _value_np(self, h_graph: np.ndarray) -> float:
        z = h_graph
        for w, b in self._value[:-1]:
            z = np.tanh(z @ w + b)
        w, b = self._value[-1]
        return float(np.tanh(z @ w + b)[0])

    # --------------------------------------------------------------------------------------- API
    def logits(self, tracker: FeatureTracker, state: State) -> np.ndarray:
        self.calls += 1
        m = state.inst.m
        nxt = state.next_op
        cand = [j * m + nxt[j] for j in state.active]
        if self.backend == "numpy":
            h = self._encode(tracker)
            return self._policy_np(h, h.mean(axis=0), cand)
        with torch.inference_mode():
            logits, _ = self.model.forward_single(torch.from_numpy(tracker.x), torch.from_numpy(tracker.jpred),
                                                  torch.from_numpy(tracker.mpred), torch.as_tensor(cand))
        return logits.numpy()

    def evaluate(self, tracker: FeatureTracker, state: State) -> tuple[np.ndarray, float]:
        self.calls += 1
        m = state.inst.m
        nxt = state.next_op
        cand = [j * m + nxt[j] for j in state.active]
        if self.backend == "numpy":
            h = self._encode(tracker)
            h_graph = h.mean(axis=0)
            return self._policy_np(h, h_graph, cand), self._value_np(h_graph)
        with torch.inference_mode():
            logits, value = self.model.forward_single(torch.from_numpy(tracker.x), torch.from_numpy(tracker.jpred),
                                                      torch.from_numpy(tracker.mpred), torch.as_tensor(cand))
        return logits.numpy(), float(value)


class FFNNInference:
    """Single-state evaluation of the fully connected networks of AZ."""

    kind = "ffnn"

    def __init__(self, model: AZNetworks, backend: str = "numpy"):
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        self.backend = backend
        self.model = model.eval()
        self.n, self.m = model.n, model.m
        self.calls = 0
        if backend == "numpy":
            self._policy = _linears(model.policy)
            self._value = _linears(model.value)

    def check(self, inst: Instance) -> None:
        if inst.shape != (self.n, self.m):
            raise ValueError(f"the networks of AZ were trained for {self.n}x{self.m} instances and cannot be "
                             f"applied to {inst.name} ({inst.n}x{inst.m})")

    @staticmethod
    def _forward(layers, v: np.ndarray) -> np.ndarray:
        for w, b in layers[:-1]:
            v = _mish(v @ w + b)
        w, b = layers[-1]
        return v @ w + b

    def begin(self, tracker: FeatureTracker, state: State) -> None:
        """Nothing to prepare: the fully connected network is evaluated in full at every step."""

    def advance(self, tracker: FeatureTracker, state: State) -> None:
        """Nothing to update."""

    def rollout_logits(self, tracker: FeatureTracker, state: State) -> np.ndarray:
        return self.logits(tracker, state)

    def logits(self, tracker: FeatureTracker, state: State) -> np.ndarray:
        self.calls += 1
        if self.backend == "numpy":
            return self._forward(self._policy, tracker.x.reshape(-1))[state.active]
        with torch.inference_mode():
            out = self.model.policy(torch.from_numpy(tracker.x).reshape(1, -1))[0]
        return out.numpy()[state.active]

    def evaluate(self, tracker: FeatureTracker, state: State) -> tuple[np.ndarray, float]:
        self.calls += 1
        if self.backend == "numpy":
            flat = tracker.x.reshape(-1)
            value = float(np.tanh(self._forward(self._value, flat))[0])
            return self._forward(self._policy, flat)[state.active], value
        with torch.inference_mode():
            flat = torch.from_numpy(tracker.x).reshape(1, -1)
            logits = self.model.policy(flat)[0].numpy()[state.active]
            value = float(torch.tanh(self.model.value(flat))[0, 0])
        return logits, value


def make_inference(model, backend: str = "numpy"):
    """Inference object for a trained network (GIN of LR, networks of AZ, or a fast policy network)."""
    from .fast import FastAZ, FastInference, FastLR

    if isinstance(model, GINPolicyValue):
        return GINInference(model, backend)
    if isinstance(model, AZNetworks):
        return FFNNInference(model, backend)
    if isinstance(model, (FastLR, FastAZ)):
        return FastInference(model)
    raise TypeError(f"no inference class for {type(model).__name__}")
