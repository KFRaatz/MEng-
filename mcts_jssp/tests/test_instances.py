"""Instances: parser and its checks, the OR-Library registry (Table 3.3), generator and disturbances."""
from __future__ import annotations

import numpy as np
import pytest

from mctsjssp.instances import (EXPECTED_SIZE, TABLE_3_3, TRAINING_SIZES, all_names, disturb,
                                generate_instance, parse_orlib_collection, parse_standard, rq3_names,
                                rq5_az_names, rq5_names, scenario, scenarios, select_names, training_instances,
                                validation_set)
from mctsjssp.instances.disturbance import ScenarioSampler

STANDARD = """# a comment
# another comment
2 3
0 3 1 2 2 2
1 4 0 1 2 5
"""


def test_parse_standard_reads_machines_and_durations():
    inst = parse_standard(STANDARD, "demo")
    assert inst.shape == (2, 3)
    assert inst.mach == [[0, 1, 2], [1, 0, 2]]
    assert inst.dur == [[3, 2, 2], [4, 1, 5]]
    # LB0 = max(longest job, highest machine workload) = max(10, machine 2: 2 + 5 = 7) = 10
    assert inst.lb0 == 10
    assert inst.rem[0] == [7, 4, 2, 0] and inst.rem[1] == [10, 6, 5, 0]


@pytest.mark.parametrize("text, message", [
    ("2 2\n0 1 1 1\n", "expected 2 job rows"),
    ("2 2\n0 1 1 1\n0 1 0 2\n", "do not visit every machine"),
    ("2 2\n0 1 1 1\n0 1 1\n", "has 3 numbers"),
    ("2 2\n0 1 1 x\n1 1 0 1\n", "not an integer"),
    ("2 2\n0 1 1 -4\n1 1 0 1\n", "negative processing time"),
])
def test_parser_rejects_malformed_instances(text, message):
    with pytest.raises(ValueError, match=message):
        parse_standard(text, "bad")


def test_parse_orlib_collection_file_format():
    text = """
 +++++++++++++++++++++++++++++
 instance one
 +++++++++++++++++++++++++++++
 A 2x2 instance (description with numbers 10 10 inside a sentence)
 2 2
 0 5 1 3
 1 2 0 4
 +++++++++++++++++++++++++++++
 instance two
 +++++++++++++++++++++++++++++
 another
 1 2
 1 7 0 1
"""
    parsed = parse_orlib_collection(text)
    assert set(parsed) == {"one", "two"}
    assert parsed["one"].dur == [[5, 3], [2, 4]]
    assert parsed["two"].mach == [[1, 0]]


def test_registry_matches_table_3_3(orlib):
    assert len(all_names()) == 82 and len(orlib) == 82
    for name, inst in orlib.items():
        assert inst.shape == EXPECTED_SIZE[name]
        assert sorted(inst.mach[0]) == list(range(inst.m))
    classes = [inst.size_class for inst in orlib.values()]
    assert (classes.count("small"), classes.count("medium"), classes.count("large")) == (35, 20, 27)
    assert sum(len(names) for _, names, _, _ in TABLE_3_3) == 82
    ops = [inst.n_ops for inst in orlib.values()]
    assert min(ops) == 36 and max(ops) == 500


def test_subsets_of_the_research_questions(orlib):
    assert len(rq3_names()) == 24
    assert all(orlib[n].shape in TRAINING_SIZES for n in rq3_names())
    assert rq5_names() == ["ft06", "ft10", "ft20", "la01", "la06", "la11", "la16", "la21", "la26", "la31", "la36",
                           "abz5", "abz7", "orb01", "swv01", "swv06", "swv11", "yn1"]
    assert rq5_az_names() == ["ft06", "ft10", "la16", "la36", "abz5", "orb01"]
    assert select_names("size:15x15") == ["la36", "la37", "la38", "la39", "la40"]
    assert select_names("family:yn") == ["yn1", "yn2", "yn3", "yn4"]
    assert select_names(["ft06", "small"])[0] == "ft06" and len(select_names(["ft06", "small"])) == 35
    assert len(select_names("large")) == 27
    with pytest.raises(ValueError):
        select_names("nonsense")


def test_best_known_makespans_are_attached(orlib):
    assert (orlib["ft06"].bks, orlib["ft10"].bks, orlib["la01"].bks) == (55, 930, 666)
    assert orlib["ft06"].optimal_known is True
    open_instances = sorted(n for n, inst in orlib.items() if not inst.optimal_known)
    assert open_instances == ["swv07", "swv08", "swv10"]
    assert all(inst.lower_bound <= inst.bks for inst in orlib.values())
    assert all(inst.lb0 <= inst.bks for inst in orlib.values())      # LB0 is a lower bound of the makespan
    assert orlib["ft06"].gap(55) == 0.0 and orlib["ft06"].gap(66) == pytest.approx(20.0)


def test_lb0_is_the_larger_of_job_length_and_machine_workload(orlib):
    for name in ("ft06", "la01", "swv11", "orb07"):
        inst = orlib[name]
        longest_job = max(sum(row) for row in inst.dur)
        load = [0] * inst.m
        for j in range(inst.n):
            for k in range(inst.m):
                load[inst.mach[j][k]] += inst.dur[j][k]
        assert inst.lb0 == max(longest_job, max(load))
    assert min(min(row) for row in orlib["orb07"].dur) == 0        # the zero processing time of Table 3.3


def test_generator_follows_taillard_distribution():
    inst = generate_instance(10, 7, np.random.default_rng(0))
    assert inst.shape == (10, 7)
    assert inst.durations.min() >= 1 and inst.durations.max() <= 99
    assert all(sorted(row) == list(range(7)) for row in inst.mach)
    many = np.concatenate([generate_instance(20, 20, np.random.default_rng(s)).durations.ravel() for s in range(5)])
    assert set(np.unique(many)) == set(range(1, 100))


def test_validation_and_training_sets_are_fixed_and_distinct():
    a, b = validation_set(6, 6), validation_set(6, 6)
    assert len(a) == 100 and a is b
    assert np.array_equal(a[3].durations, validation_set.__wrapped__(6, 6)[3].durations)
    assert not np.array_equal(a[0].durations, a[1].durations)
    g0 = training_instances(6, 6, 4, seed=1, generation=0)
    g1 = training_instances(6, 6, 4, seed=1, generation=1)
    again = training_instances(6, 6, 4, seed=1, generation=0)
    assert np.array_equal(g0[2].durations, again[2].durations)
    assert not np.array_equal(g0[0].durations, g1[0].durations)
    assert not np.array_equal(g0[0].durations, a[0].durations)


def test_disturbance_model(ft06):
    rng = np.random.default_rng(0)
    assert np.array_equal(disturb(ft06.durations, 0.0, rng), ft06.durations)
    realised = disturb(np.full((200, 200), 50), 0.2, rng)
    assert realised.min() >= 0 and realised.dtype == np.int64
    assert abs(realised.mean() - 50) < 0.5 and abs(realised.std() - 10) < 0.5     # p * (1 + N(0, 0.2))
    assert disturb(np.array([[1]]), 5.0, np.random.default_rng(3)).min() >= 0     # clipped at zero


def test_evaluation_scenarios_are_fixed(ft06, orlib):
    first = scenarios(ft06, 0.1, 30)
    assert len(first) == 30 and len(first[0]) == ft06.n_ops
    assert first == scenarios(ft06, 0.1, 30)
    assert first[0] != first[1]
    assert scenario(ft06, 0.1, 0) != scenario(ft06, 0.2, 0)
    assert scenario(ft06, 0.1, 0) != scenario(orlib["la01"], 0.1, 0)[: ft06.n_ops]


def test_search_scenarios_differ_from_evaluation_scenarios(ft06):
    sampler = ScenarioSampler(ft06, 0.1, seed=4)
    drawn = [sampler.sample() for _ in range(3)]
    assert drawn[0] != drawn[1] and all(len(d) == ft06.n_ops and min(d) >= 0 for d in drawn)
    assert ScenarioSampler(ft06, 0.1, seed=4).sample() == drawn[0]
    assert drawn[0] not in scenarios(ft06, 0.1, 30)
