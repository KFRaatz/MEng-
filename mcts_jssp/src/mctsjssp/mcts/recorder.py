"""Recording of the search behaviour at three levels, linked by the identifier of the run (Section 3.6.4).

* ``steps``     - a sample of the rollout steps (one in a thousand, drawn with a fixed seed): the number of
  candidates, the probabilities, the operation chosen and its rank under MWKR, the normalised entropy, the
  Kullback-Leibler divergence from the MWKR distribution and the agreement of the most probable operations;
  the mean entropy is accumulated over all steps;
* ``rollouts``  - for every rollout its return and its length;
* ``sequences`` (any level, when ``sequence_rate > 0``) - the complete job sequence of a random sample of the
  rollouts, with the position where the tree ends and the rollout begins: every state the rollout visited is a
  prefix of its sequence and is rebuilt exactly by :func:`mctsjssp.env.replay`;
* ``decisions`` - for every decision the visit counts and values at the root, the entropy of the visit
  distribution, the visit share of the most visited action, the variance of the rollout returns and of the
  value estimates at the root, the number of iterations and the time.

Every level includes the levels below it in this list (``steps`` records everything).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from ..distributions.base import StepStats
from .node import Node

LEVELS = {"none": 0, "decisions": 1, "rollouts": 2, "steps": 3}


class RunRecorder:
    def __init__(self, level: str = "decisions", step_rate: float = 0.001, seed: int = 0,
                 sequence_rate: float = 0.0):
        if level not in LEVELS:
            raise ValueError(f"recording level must be one of {sorted(LEVELS)}")
        self.level_name = level
        self.level = LEVELS[level]
        self.decisions: list[dict] = []
        self.rollout_return: list[float] = []
        self.rollout_length: list[int] = []
        self.rollout_decision: list[int] = []
        self.steps: Optional[StepStats] = StepStats(step_rate, seed) if self.level >= 3 else None
        self._decision = 0
        self._n = 0
        self._sum = 0.0
        self._sq = 0.0
        self._rollouts_total = 0
        self.final_return: Optional[float] = None
        # sampled rollout sequences; their own random stream, so the search itself is not changed
        self.sequence_rate = float(sequence_rate)
        self._seq_rnd = random.Random(seed ^ 0x5E9) if self.sequence_rate > 0 else None
        self.seq: list = []
        self.seq_tree_start: list[int] = []
        self.seq_rollout_start: list[int] = []
        self.seq_decision: list[int] = []
        self.seq_return: list[float] = []

    def want_sequence(self) -> bool:
        return self._seq_rnd is not None and self._seq_rnd.random() < self.sequence_rate

    def add_sequence(self, sequence: list, tree_start: int, rollout_start: int, value: float) -> None:
        """The complete job sequence of one rollout: ``sequence[:tree_start]`` are the committed decisions,
        ``sequence[tree_start:rollout_start]`` the path in the tree and the rest the rollout itself."""
        self.seq.append(np.asarray(sequence, dtype=np.uint16))
        self.seq_tree_start.append(tree_start)
        self.seq_rollout_start.append(rollout_start)
        self.seq_decision.append(self._decision)
        self.seq_return.append(value)

    # ------------------------------------------------------------------------------------- callbacks
    def begin_decision(self, index: int) -> None:
        self._decision = index
        self._n = 0
        self._sum = 0.0
        self._sq = 0.0
        if self.steps is not None:
            self.steps.context = {"decision": index}

    def on_rollout(self, value: float, length: int) -> None:
        """Called by the search once per iteration with the return and the length of the rollout."""
        self._n += 1
        self._sum += value
        self._sq += value * value
        self._rollouts_total += 1
        if self.level >= 2:
            self.rollout_return.append(value)
            self.rollout_length.append(length)
            self.rollout_decision.append(self._decision)

    def end_decision(self, root: Node, depth: int, n_candidates: int, iterations: int, seconds: float,
                     action: int) -> None:
        if self.level < 1:
            return
        children = sorted(root.children, key=lambda c: c.action)
        visits = np.array([c.N for c in children], dtype=np.float64)
        values = [c.Q for c in children if c.N > 0]
        total = visits.sum()
        if total > 0:
            p = visits / total
            nz = p[p > 0]
            h = float(-(nz * np.log(nz)).sum())
            top = float(p.max())
        else:
            h, top = float("nan"), float("nan")
        n = self._n
        mean = self._sum / n if n else float("nan")
        var = max(self._sq / n - mean * mean, 0.0) if n else float("nan")
        self.decisions.append({
            "decision": self._decision,
            "depth": depth,                                   # operations scheduled before the decision
            "n_candidates": n_candidates,
            "iterations": iterations,
            "time_s": seconds,
            "action": action,
            "actions": [c.action for c in children],
            "visits": [int(c.N) for c in children],
            "values": [round(c.Q, 6) for c in children],
            "visit_entropy": h,                               # Equation 2.21 of the visit distribution
            "visit_entropy_norm": h / math.log(n_candidates) if n_candidates > 1 else float("nan"),
            "top_visit_share": top,
            "value_var": float(np.var(values)) if len(values) > 1 else 0.0,
            "return_mean": mean,                              # rollouts of this decision
            "return_var": var,
            "rollouts": n,
        })

    # --------------------------------------------------------------------------------------- summary
    def summary(self, final_return: Optional[float] = None) -> dict:
        """Run-level aggregates stored with the result of the run."""
        self.final_return = final_return
        out: dict = {"record_level": self.level_name}
        if self.level >= 1 and self.decisions:
            def mean(key):
                vals = [d[key] for d in self.decisions if not math.isnan(d[key])]
                return float(np.mean(vals)) if vals else float("nan")
            out.update(
                visit_entropy=mean("visit_entropy"),
                visit_entropy_norm=mean("visit_entropy_norm"),
                top_visit_share=mean("top_visit_share"),
                value_var=mean("value_var"),
                return_var=mean("return_var"),
            )
            if final_return is not None:
                # bias: mean return from a state minus the return of the schedule finally completed from it
                out["return_bias"] = mean("return_mean") - final_return
        if self.steps is not None:
            s = self.steps
            out["rollout_entropy"] = s.mean_entropy
            out["rollout_steps"] = s.h_cnt
            out["sampled_steps"] = len(s.samples)
            if s.samples:
                out["agree_mwkr"] = float(np.mean([r["agree_mwkr"] for r in s.samples]))
                out["mwkr_rank"] = float(np.mean([r["mwkr_rank"] for r in s.samples]))
                kl = [r["kl_mwkr"] for r in s.samples if "kl_mwkr" in r and math.isfinite(r["kl_mwkr"])]
                if kl:
                    out["kl_mwkr"] = float(np.mean(kl))
        return out

    def tables(self) -> dict:
        """The detailed records of the run, ready to be written to disk."""
        out = {"decisions": self.decisions}
        if self.level >= 2:
            out["rollouts"] = {
                "decision": np.asarray(self.rollout_decision, dtype=np.int32),
                "return": np.asarray(self.rollout_return, dtype=np.float32),
                "length": np.asarray(self.rollout_length, dtype=np.int32),
            }
        if self.steps is not None:
            out["steps"] = self.steps.samples
        if self.seq:
            out["sequences"] = {
                "sequence": np.vstack(self.seq),
                "tree_start": np.asarray(self.seq_tree_start, dtype=np.int32),
                "rollout_start": np.asarray(self.seq_rollout_start, dtype=np.int32),
                "decision": np.asarray(self.seq_decision, dtype=np.int32),
                "return": np.asarray(self.seq_return, dtype=np.float32),
            }
        return out
