"""Softmax rollout distribution, entropy and Kullback-Leibler divergence (Section 2.6.2).

* Equation 2.20: ``pi(a|s) = exp(f(s,a)/tau) / sum_b exp(f(s,b)/tau)``
* Equation 2.21: ``H(pi(.|s)) = -sum_a pi(a|s) ln pi(a|s)``
* normalised entropy (Section 3.2.2): ``H / ln|A(s)|``, 1 for uniform sampling and 0 for a deterministic rule
* Equation 2.22: ``D_KL(pi || pi') = sum_a pi(a|s) ln(pi(a|s) / pi'(a|s))`` in nats
"""
from __future__ import annotations

import math

import numpy as np


def softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    """Equation 2.20.  ``tau = inf`` gives the uniform distribution; ``tau`` must be positive."""
    scores = np.asarray(scores, dtype=np.float64)
    if math.isinf(tau):
        return np.full(scores.shape[0], 1.0 / scores.shape[0])
    if tau <= 0:
        raise ValueError("the temperature must be positive; use greedy() for the deterministic rule")
    z = (scores - scores.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def greedy(scores: np.ndarray) -> np.ndarray:
    """The deterministic rule (normalised entropy 0): all probability on the highest score (first on ties)."""
    p = np.zeros(len(scores), dtype=np.float64)
    p[int(np.argmax(scores))] = 1.0
    return p


def entropy(p: np.ndarray) -> float:
    """Equation 2.21, in nats."""
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def normalised_entropy(p: np.ndarray) -> float:
    """``H / ln|A(s)|``; not defined (NaN) for a single candidate, where every distribution is the same."""
    k = len(p)
    if k < 2:
        return float("nan")
    return entropy(p) / math.log(k)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Equation 2.22, ``D_KL(p || q)`` in nats; infinite if ``q`` is zero where ``p`` is positive."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    if (q[mask] <= 0).any():
        return float("inf")
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def same_top(p: np.ndarray, q: np.ndarray) -> bool:
    """Whether the most probable candidates of two distributions agree."""
    return int(np.argmax(p)) == int(np.argmax(q))
