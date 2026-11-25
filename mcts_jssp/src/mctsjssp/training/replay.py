"""Replay buffer of the self-play records (Section 3.6.1).

Every record holds the observation of a state ``s_t`` (features and machine arcs), the legal actions, the
visit distribution ``pi_t`` at the root and the ranked reward ``z`` of the episode.  The buffer holds 36 000
records and the oldest are removed first.
"""
from __future__ import annotations

from typing import Iterator, NamedTuple, Optional

import numpy as np
import torch


class Record(NamedTuple):
    x: np.ndarray          # (n_ops, 2) features: scheduled flag, LB / LB0
    mpred: np.ndarray      # (n_ops,) machine predecessor, n_ops = none
    next_op: np.ndarray    # (n,) index of the next operation of every job
    mask: np.ndarray       # (n,) True for unfinished jobs
    pi: np.ndarray         # (n,) visit distribution at the root
    z: float               # ranked reward of the episode
    f: Optional[np.ndarray] = None   # (n, 6) features of the fast networks


class ReplayBuffer:
    def __init__(self, capacity: int, n: int, m: int):
        self.capacity = int(capacity)
        self.n, self.m = n, m
        n_ops = n * m
        self.x = np.zeros((self.capacity, n_ops, 2), dtype=np.float32)
        self.mpred = np.zeros((self.capacity, n_ops), dtype=np.int32)
        self.next_op = np.zeros((self.capacity, n), dtype=np.int16)
        self.mask = np.zeros((self.capacity, n), dtype=bool)
        self.pi = np.zeros((self.capacity, n), dtype=np.float32)
        self.z = np.zeros(self.capacity, dtype=np.float32)
        self.f = np.zeros((self.capacity, n, 6), dtype=np.float32)
        self.size = 0
        self._head = 0                 # position of the next record; the oldest record once the buffer is full
        self.added = 0

    def __len__(self) -> int:
        return self.size

    def add(self, rec: Record) -> None:
        i = self._head
        self.x[i] = rec.x
        self.mpred[i] = rec.mpred
        self.next_op[i] = rec.next_op
        self.mask[i] = rec.mask
        self.pi[i] = rec.pi
        self.z[i] = rec.z
        self.f[i] = rec.f if rec.f is not None else 0.0
        self._head = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.added += 1

    def extend(self, records) -> None:
        for rec in records:
            self.add(rec)

    def batch(self, idx: np.ndarray) -> dict:
        """Tensors of the records ``idx`` in the form expected by ``forward_batch``."""
        return {
            "x": torch.from_numpy(self.x[idx]),
            "mpred": torch.from_numpy(self.mpred[idx].astype(np.int64)),
            "next_op": torch.from_numpy(self.next_op[idx].astype(np.int64)),
            "mask": torch.from_numpy(self.mask[idx]),
            "pi": torch.from_numpy(self.pi[idx]),
            "z": torch.from_numpy(self.z[idx]),
            "f": torch.from_numpy(self.f[idx]),
        }

    def epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[dict]:
        """One pass over the buffer in shuffled mini-batches."""
        order = rng.permutation(self.size)
        for start in range(0, self.size, batch_size):
            yield self.batch(np.sort(order[start: start + batch_size]))
