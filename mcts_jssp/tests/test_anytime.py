"""Anytime curves: recording the best makespan against time, the table and the figure."""
import math

import pandas as pd

from mctsjssp.analysis.anytime import _gaps_on_grid, anytime_summary, anytime_table
from mctsjssp.analysis.plots import plot_anytime
from mctsjssp.distributions.fixed import UniformRollout
from mctsjssp.instances.orlib import load_orlib
from mctsjssp.mcts.agent import Budget, solve


def test_solve_records_improvements_in_time_order():
    inst = load_orlib(["ft06"])["ft06"]
    res = solve(inst, UniformRollout(), Budget.of_iterations(20), seed=1)
    times = [t for t, _ in res.anytime]
    spans = [m for _, m in res.anytime]
    assert times == sorted(times) and times[-1] <= res.time_total + 1e-9
    assert all(a > b for a, b in zip(spans, spans[1:]))          # recorded only when it improves
    assert spans[-1] == min(res.makespan, res.best_rollout_makespan)


def test_gaps_on_grid_is_a_step_function():
    gaps = _gaps_on_grid([[0.1, 120], [0.5, 110]], 100.0, [0.05, 0.1, 0.3, 0.5, 9.0])
    assert math.isnan(gaps[0])
    assert list(gaps[1:]) == [20.0, 20.0, 10.0, 10.0]


def _frame():
    rows = []
    for label, traces in {"U": [[[0.01, 110], [1.0, 100]], [[0.02, 105]]],
                          "MWKR@H0": [[[0.01, 102]], [[0.01, 101], [0.1, 100]]]}.items():
        for run, tr in enumerate(traces):
            rows.append({"label": label, "config": label.split("@")[0], "level": 0.0, "budget": "I100",
                         "size_class": "small", "bks": 100, "anytime": tr, "time_total": 1.0, "run": run})
    return pd.DataFrame(rows)


def test_table_and_figure(tmp_path):
    table = anytime_table(_frame(), points=20)
    end = table[(table["group"] == "all")].groupby("label")["mean_gap"].last()
    assert end["U"] == 2.5 and end["MWKR@H0"] == 1.0
    share = table[(table["group"] == "all") & (table["label"] == "MWKR@H0")]["share_at_bks"].iloc[-1]
    assert share == 0.5
    assert not anytime_summary(table).empty
    paths = plot_anytime(table, tmp_path)
    assert any(p.name == "anytime_I100.png" for p in paths)


def test_old_results_without_traces_are_skipped():
    assert anytime_table(pd.DataFrame({"budget": ["I100"], "label": ["U"]})).empty


def test_rollout_sequences_rebuild_the_visited_states(tmp_path):
    """Sampled rollout sequences replay to the rollout's own return, and identical states get one key."""
    import numpy as np

    from mctsjssp.analysis.visited import _zobrist, walk
    from mctsjssp.env import replay
    from mctsjssp.mcts.recorder import RunRecorder

    inst = load_orlib(["ft06"])["ft06"]
    rec = RunRecorder("decisions", seed=3, sequence_rate=0.2)
    solve(inst, UniformRollout(), Budget.of_iterations(30), seed=2, recorder=rec)
    seqs = rec.tables()["sequences"]
    assert 0 < len(seqs["sequence"]) and seqs["sequence"].shape[1] == inst.n_ops
    for seq, value, tree, roll in zip(seqs["sequence"], seqs["return"], seqs["tree_start"], seqs["rollout_start"]):
        state = replay(inst, [int(j) for j in seq])
        assert state.complete and abs(-state.makespan / inst.lb0 - value) < 1e-6 and tree <= roll
    zob = _zobrist(inst.n_ops)
    a = walk(inst, [int(j) for j in seqs["sequence"][0]], 0, zob)
    assert len(a) == inst.n_ops and a[0][0] == 0                     # the empty schedule has key 0
    # two orders of independent first steps build the same partial schedule -> the same key
    k1 = walk(inst, [0, 1, 0], 0, zob)[2][0]
    k2 = walk(inst, [1, 0, 0], 0, zob)[2][0]
    s1, s2 = replay(inst, [0, 1]), replay(inst, [1, 0])
    assert (s1.op_start == s2.op_start) == (k1 == k2)


def test_compare_experiments_instance_by_instance(tmp_path):
    """Two experiments are combined per instance and budget, ranked, and tested against U."""
    import json

    from mctsjssp.analysis.compare import compare_experiments

    def write(name, rows):
        d = tmp_path / name
        d.mkdir()
        with open(d / "results.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return d

    def row(label, inst, gap, run):
        return {"label": label, "config": label.split("@")[0], "level": 0.0, "instance": inst, "budget": "I100",
                "gap": gap, "run": run, "n_ops": 36, "time_total": 1.0, "iterations": 100, "bks": 55}

    insts = [f"i{k}" for k in range(8)]
    a = write("A", [row("U", i, 5.0 + k, r) for k, i in enumerate(insts) for r in range(2)]
              + [row("MWKR@H0", i, 3.0 + k, r) for k, i in enumerate(insts) for r in range(2)])
    b = write("B", [row("LR@H0.5", i, 6.0 + k, r) for k, i in enumerate(insts) for r in range(2)])
    res = compare_experiments({"A": a, "B": b})
    t = res["tables"]
    assert set(t["per_instance"]["name"]) == {"A:U", "A:MWKR@H0", "B:LR@H0.5"}
    ranks = t["mean_rank"].set_index("configuration")["mean_rank"]
    assert ranks["A:MWKR@H0"] == 1.0 and ranks["B:LR@H0.5"] == 3.0
    vs = t["vs_reference"].set_index("configuration")
    assert vs.loc["A:MWKR@H0", "better"] == 8 and vs.loc["B:LR@H0.5", "worse"] == 8
    assert (vs["reference"] == "A:U").all() and "p_holm" in vs
