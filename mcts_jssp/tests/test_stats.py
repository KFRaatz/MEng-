"""Statistical tests of Section 3.6.4: Wilcoxon signed-rank, Friedman, Holm's procedure, effect sizes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from mctsjssp.analysis import (compare_many, compare_pairs, friedman_test, holm, rank_posthoc, signed_rank_sums,
                               wilcoxon_signed_rank)


# ------------------------------------------------------------------------------------------- Wilcoxon
def test_signed_rank_sums_by_hand():
    # differences 3, -1, 2, 5 -> ranks of |d|: 3, 1, 2, 4 -> R+ = 3 + 2 + 4 = 9, R- = 1
    assert signed_rank_sums(np.array([3.0, -1.0, 2.0, 5.0])) == (9.0, 1.0)
    # ties share the average rank: |d| = 1, 1, 2 -> ranks 1.5, 1.5, 3
    assert signed_rank_sums(np.array([1.0, -1.0, 2.0])) == (4.5, 1.5)


def test_zero_differences_are_split_as_in_demsar():
    d = np.array([0.0, 0.0, 1.0, -2.0, 3.0])          # ranks of |d|: 1.5, 1.5, 3, 4, 5
    r_plus, r_minus = signed_rank_sums(d, "zsplit")
    assert (r_plus, r_minus) == (3 + 5 + 1.5, 4 + 1.5)
    assert r_plus + r_minus == 5 * 6 / 2              # all ranks are used
    assert signed_rank_sums(d, "wilcox") == (1 + 3, 2)  # zeros discarded, then ranked
    assert signed_rank_sums(d, "pratt") == (3 + 5, 4)   # ranked with the zeros, zero ranks dropped
    with pytest.raises(ValueError):
        signed_rank_sums(d, "other")


@pytest.mark.parametrize("zero_method", ["zsplit", "wilcox", "pratt"])
def test_wilcoxon_agrees_with_scipy(zero_method):
    rng = np.random.default_rng(0)
    for n in (8, 24, 82):
        x = rng.gamma(2.0, 5.0, n).round(1)
        y = (x + rng.normal(0.8, 2.0, n)).round(1)
        y[:2] = x[:2]                                  # two instances without a difference
        res = wilcoxon_signed_rank(x, y, zero_method)
        ref = sps.wilcoxon(x, y, zero_method=zero_method, alternative="two-sided")
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.statistic == pytest.approx(float(ref.statistic))
        assert res.n == n and res.n_zero == int((x == y).sum()) >= 2


def test_effect_sizes_have_the_documented_direction():
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = first + np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.2, 0.5, 4.0])     # the first is better on every instance
    res = wilcoxon_signed_rank(first, second)
    assert res.rank_biserial == pytest.approx(1.0) and res.median_diff < 0
    assert res.rank_sum_first_better == 36 and res.rank_sum_second_better == 0
    assert res.p_value == pytest.approx(2 / 2 ** 8)                           # exact two-sided p for n = 8
    swapped = wilcoxon_signed_rank(second, first)
    assert swapped.rank_biserial == pytest.approx(-1.0) and swapped.median_diff == -res.median_diff
    assert swapped.p_value == pytest.approx(res.p_value)
    mixed = wilcoxon_signed_rank(np.array([1.0, 5.0, 2.0, 9.0]), np.array([2.0, 3.0, 5.0, 5.0]))
    assert -1.0 < mixed.rank_biserial < 1.0
    # (R- - R+) / (R- + R+) with d = first - second = -1, 2, -3, 4 -> R+ = 2 + 4, R- = 1 + 3
    assert mixed.rank_biserial == pytest.approx((4 - 6) / 10)


def test_identical_configurations_are_not_different():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    res = wilcoxon_signed_rank(x, x)
    assert res.p_value == 1.0 and res.rank_biserial == 0.0 and res.median_diff == 0.0 and res.n_zero == 5
    with pytest.raises(ValueError):
        wilcoxon_signed_rank(x, x[:3])


# ------------------------------------------------------------------------------------------------ Holm
def test_holm_by_hand():
    # sorted: 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> monotone: 0.02, 0.03, 0.06, 0.06
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])                     # capped at 1
    assert np.allclose(holm([0.02]), [0.02])                             # one comparison: unchanged
    assert holm([]).size == 0


def test_holm_is_at_most_as_strict_as_bonferroni_and_ignores_missing_values():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, 12)
    adjusted = holm(p)
    assert (adjusted >= p - 1e-15).all() and (adjusted <= np.minimum(1.0, p * 12) + 1e-15).all()
    order = np.argsort(p)
    assert (np.diff(adjusted[order]) >= -1e-15).all()                    # monotone in the raw p-values
    with_nan = holm([0.01, np.nan, 0.04])
    assert np.isnan(with_nan[1]) and np.allclose(with_nan[[0, 2]], [0.02, 0.04])


# -------------------------------------------------------------------------------------------- Friedman
def test_friedman_agrees_with_scipy_and_ranks_from_best_to_worst():
    rng = np.random.default_rng(2)
    base = rng.gamma(2.0, 5.0, 30)
    table = pd.DataFrame({"U": base + 3 + rng.normal(0, 1, 30), "MWKR": base + 1.5 + rng.normal(0, 1, 30),
                          "LR": base + rng.normal(0, 1, 30)})
    res = friedman_test(table)
    chi2, p = sps.friedmanchisquare(table["U"], table["MWKR"], table["LR"])
    assert res.statistic == pytest.approx(chi2) and res.p_value == pytest.approx(p)
    assert res.n == 30 and res.k == 3
    assert res.average_ranks["U"] > res.average_ranks["MWKR"] > res.average_ranks["LR"]   # 1 = best (smallest)
    assert sum(res.average_ranks.values()) == pytest.approx(6.0)          # 1 + 2 + 3
    assert chi2 < 60                                                      # not a perfect ordering
    expected_f = (30 - 1) * chi2 / (30 * 2 - chi2)                        # Iman and Davenport
    assert res.iman_davenport == pytest.approx(expected_f)
    assert res.p_iman_davenport == pytest.approx(float(sps.f.sf(expected_f, 2, 58)))


def test_friedman_edge_cases():
    same = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [1.0, 2.0, 3.0]})
    res = friedman_test(same)
    assert res.p_value == 1.0 and set(res.average_ranks.values()) == {2.0}
    with pytest.raises(ValueError):
        friedman_test(same[["a", "b"]])                                   # two configurations: use Wilcoxon
    with pytest.raises(ValueError):
        friedman_test(same.iloc[:1])


# ----------------------------------------------------------------------------------- comparison tables
def make_table(n=40, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 5.0, n)
    return pd.DataFrame({"U": base + 4 + rng.normal(0, 1, n), "MWKR": base + 2 + rng.normal(0, 1, n),
                         "LR": base + rng.normal(0, 1, n)}, index=[f"i{k}" for k in range(n)])


def test_compare_pairs_applies_holm_over_the_listed_pairs():
    table = make_table()
    res = compare_pairs(table, [("LR", "U"), ("LR", "MWKR"), ("MWKR", "U")], alpha=0.05)
    assert list(res["first"]) == ["LR", "LR", "MWKR"] and (res["n"] == 40).all()
    assert np.allclose(res["p_holm"], holm(res["p_value"].to_numpy()))
    assert res["significant"].all() and (res["rank_biserial"] > 0).all() and (res["median_diff"] < 0).all()
    raw = compare_pairs(table, [("LR", "U")], correct=False)
    assert raw["p_holm"].iloc[0] == raw["p_value"].iloc[0]
    assert compare_pairs(table, [("LR", "missing")]).empty


def test_compare_pairs_uses_only_common_instances():
    table = make_table(12)
    table.loc[table.index[:4], "LR"] = np.nan                            # e.g. AZ, run on fewer instances
    res = compare_pairs(table, [("LR", "U")])
    assert res["n"].iloc[0] == 8


def test_compare_many_reports_friedman_and_posthoc():
    table = make_table()
    friedman, post = compare_many(table, ["U", "MWKR", "LR"], [("LR", "U"), ("LR", "MWKR")])
    assert friedman.p_value < 0.05 and post["friedman_rejects"].all()
    assert len(post) == 2 and (post["p_holm"] >= post["p_value"]).all()
    ranks = rank_posthoc(friedman, [("LR", "U"), ("LR", "MWKR")])
    assert (ranks["z"] < 0).all() and ranks["significant"].iloc[0]       # LR has the smaller average rank
    noise = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)), columns=["a", "b", "c"])
    friedman, post = compare_many(noise, ["a", "b", "c"], [("a", "b")])
    assert post["friedman_rejects"].iloc[0] == (friedman.p_value <= 0.05)
