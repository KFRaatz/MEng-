"""Statistical tests of Section 3.6.4 (after Demsar, 2006).

All configurations are run on the same instances, so the comparisons are paired, with the instance as the
unit of analysis and the mean over its runs or scenarios as the observation.

* Two configurations: Wilcoxon signed-rank test.  The differences between their gaps on the instances are
  ranked by size, and the test asks whether the instances that favour one configuration carry clearly more
  rank than those that favour the other.
* Several configurations: Friedman test.  The configurations are ranked from best to worst on every instance
  and the test asks whether their average ranks are equal.
* Several comparisons: Holm's step-down procedure keeps the probability of at least one false difference
  below the significance level.
* Effect sizes: the median difference between the gaps (percentage points) and the rank-biserial
  correlation, the difference between the rank sums of the instances that favour each configuration as a
  share of the total rank sum.

All tests are two-sided.  None assumes normally distributed gaps.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


# ------------------------------------------------------------------------------------------- Wilcoxon
@dataclass
class PairedResult:
    n: int                      # instances
    n_zero: int                 # instances on which the two configurations are equal
    rank_sum_first_better: float
    rank_sum_second_better: float
    statistic: float            # the smaller of the two rank sums
    p_value: float
    median_diff: float          # median of (first - second); negative: the first has the smaller gaps
    rank_biserial: float        # in [-1, 1]; positive: the first configuration is better (smaller values)

    def as_dict(self) -> dict:
        return asdict(self)


def signed_rank_sums(diff: np.ndarray, zero_method: str = "zsplit") -> tuple[float, float]:
    """Rank sums ``(R+, R-)`` of the positive and negative differences.

    ``zsplit`` ranks the zero differences with the others and splits their ranks evenly between the two sums
    (Demsar); ``wilcox`` discards zero differences before ranking; ``pratt`` ranks them and then drops them.
    """
    d = np.asarray(diff, dtype=np.float64)
    if zero_method == "wilcox":
        d = d[d != 0]
    if d.size == 0:
        return 0.0, 0.0
    ranks = sps.rankdata(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    if zero_method == "zsplit":
        zero = float(ranks[d == 0].sum())
        r_plus += zero / 2.0
        r_minus += zero / 2.0
    elif zero_method not in ("wilcox", "pratt"):
        raise ValueError("zero_method must be 'zsplit', 'wilcox' or 'pratt'")
    return r_plus, r_minus


def wilcoxon_signed_rank(first: Sequence[float], second: Sequence[float], zero_method: str = "zsplit") -> PairedResult:
    """Two-sided Wilcoxon signed-rank test of two configurations on the same instances.

    ``first`` and ``second`` hold one observation per instance (smaller is better, for example the gap).
    """
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("the two configurations must be observed on the same instances")
    d = x - y
    n = int(d.size)
    r_plus, r_minus = signed_rank_sums(d, zero_method)
    total = r_plus + r_minus
    if n == 0 or not np.any(d != 0):
        p_value, statistic = 1.0, 0.0
    else:
        res = sps.wilcoxon(x, y, zero_method=zero_method, alternative="two-sided")
        p_value, statistic = float(res.pvalue), float(min(r_plus, r_minus))
    # positive differences favour the second configuration (its values are smaller)
    return PairedResult(
        n=n, n_zero=int((d == 0).sum()),
        rank_sum_first_better=r_minus, rank_sum_second_better=r_plus,
        statistic=statistic, p_value=p_value,
        median_diff=float(np.median(d)) if n else float("nan"),
        rank_biserial=(r_minus - r_plus) / total if total > 0 else 0.0,
    )


# --------------------------------------------------------------------------------------------- Friedman
@dataclass
class FriedmanResult:
    n: int                      # instances
    k: int                      # configurations
    statistic: float            # chi-square statistic (corrected for ties)
    p_value: float
    iman_davenport: float       # F statistic of Iman and Davenport (less conservative)
    p_iman_davenport: float
    average_ranks: dict         # configuration -> average rank (1 = best)

    def as_dict(self) -> dict:
        return asdict(self)


def friedman_test(table: pd.DataFrame) -> FriedmanResult:
    """Friedman test on a table with one row per instance and one column per configuration."""
    data = table.to_numpy(dtype=np.float64)
    n, k = data.shape
    if k < 3:
        raise ValueError("the Friedman test needs at least three configurations; use the Wilcoxon test for two")
    if n < 2:
        raise ValueError("the Friedman test needs at least two instances")
    ranks = np.apply_along_axis(sps.rankdata, 1, data)
    average = dict(zip(table.columns, ranks.mean(axis=0)))
    if np.allclose(data, data[:, :1]):
        return FriedmanResult(n, k, 0.0, 1.0, 0.0, 1.0, average)
    chi2, p = sps.friedmanchisquare(*[data[:, j] for j in range(k)])
    chi2, p = float(chi2), float(p)
    denom = n * (k - 1) - chi2
    if denom > 0:
        f_stat = (n - 1) * chi2 / denom
        p_f = float(sps.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    else:
        f_stat, p_f = float("inf"), 0.0
    return FriedmanResult(n, k, chi2, p, float(f_stat), p_f, average)


# ------------------------------------------------------------------------------------------------- Holm
def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm's step-down adjustment of a family of p-values (returned in the original order).

    The smallest p-value is multiplied by the number of comparisons ``m``, the next by ``m - 1`` and so on;
    the adjusted values are made monotone and capped at 1.  A comparison is significant if its adjusted
    p-value is at most alpha.  NaN entries are left out of the family.
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m == 0:
        return out
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


# -------------------------------------------------------------------------------------- comparison tables
def compare_pairs(wide: pd.DataFrame, pairs: Sequence[tuple], alpha: float = 0.05,
                  zero_method: str = "zsplit", correct: bool = True) -> pd.DataFrame:
    """Wilcoxon tests for the listed pairs of columns of ``wide`` (rows: instances), Holm over the pairs.

    Only instances observed for both configurations of a pair are used.
    """
    rows = []
    for first, second in pairs:
        if first not in wide.columns or second not in wide.columns:
            continue
        both = wide[[first, second]].dropna()
        if both.empty:
            continue
        res = wilcoxon_signed_rank(both[first], both[second], zero_method)
        rows.append({"first": first, "second": second, "n": res.n, "n_equal": res.n_zero,
                     "mean_first": float(both[first].mean()), "mean_second": float(both[second].mean()),
                     "median_first": float(both[first].median()), "median_second": float(both[second].median()),
                     "median_diff": res.median_diff, "rank_biserial": res.rank_biserial,
                     "rank_sum_first_better": res.rank_sum_first_better,
                     "rank_sum_second_better": res.rank_sum_second_better, "p_value": res.p_value})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"].to_numpy()) if correct else out["p_value"]
    out["significant"] = out["p_holm"] <= alpha
    return out


def compare_many(wide: pd.DataFrame, columns: Sequence[str], posthoc_pairs: Sequence[tuple], alpha: float = 0.05,
                 zero_method: str = "zsplit") -> tuple[Optional[FriedmanResult], pd.DataFrame]:
    """Friedman test over ``columns`` followed by Wilcoxon tests of ``posthoc_pairs`` with Holm's procedure.

    The post-hoc comparisons are always reported; ``friedman_rejects`` says whether the Friedman test
    rejected equal performance, which is the condition for interpreting them.
    """
    columns = [c for c in columns if c in wide.columns]
    data = wide[columns].dropna()
    friedman = friedman_test(data) if len(columns) >= 3 and len(data) >= 2 else None
    post = compare_pairs(data, posthoc_pairs, alpha, zero_method)
    if not post.empty:
        post["friedman_p"] = friedman.p_value if friedman else float("nan")
        post["friedman_rejects"] = bool(friedman and friedman.p_value <= alpha)
    return friedman, post


def rank_posthoc(friedman: FriedmanResult, pairs: Sequence[tuple], alpha: float = 0.05) -> pd.DataFrame:
    """Demsar's rank-based post-hoc test with Holm's procedure (alternative to the pairwise Wilcoxon tests).

    ``z = (R_i - R_j) / sqrt(k (k + 1) / (6 N))`` with the average ranks of the Friedman test.
    """
    se = math.sqrt(friedman.k * (friedman.k + 1) / (6.0 * friedman.n))
    rows = []
    for first, second in pairs:
        z = (friedman.average_ranks[first] - friedman.average_ranks[second]) / se
        rows.append({"first": first, "second": second, "z": z, "p_value": float(2 * sps.norm.sf(abs(z)))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].to_numpy())
        out["significant"] = out["p_holm"] <= alpha
    return out
