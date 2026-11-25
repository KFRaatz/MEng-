"""Networks for disturbed processing times (RQ5): validation under disturbances, fine-tuning and training from
scratch, robustness without search and the report over the iterations."""
import json

import numpy as np
import pytest

from mctsjssp.training import TrainConfig


def _tiny(models, name, kind, n, sigma=None, reference=None, **extra):
    from mctsjssp.training.self_label import SelfLabelTrainer

    sl = {"samples": 2, "instances": 2, "epochs": 1, "window": 1, "iterations": 2, "validation_samples": 1,
          "validation_scenarios": 3, "teacher_instances": 4, "teacher_epochs": 1, **extra}
    cfg = TrainConfig(name=name, kind=kind, n=n, m=n, validation_instances=3, seed=1, workers=1, sigma=sigma,
                      method="self_label", self_label=sl, reference_models=str(reference) if reference else None)
    return SelfLabelTrainer(cfg, models / name, log=lambda *_: None)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("disturbed") / "artifacts"
    main, scratch = root / "models", root / "scratch" / "models"
    _tiny(main, "lr_6x6", "gin", 6).run()
    _tiny(main, "az_6x6", "ffnn", 6).run()
    out = {"root": root, "main": main, "scratch": scratch}
    ft = _tiny(main, "lr_6x6_s010", "gin", 6, sigma=0.1)
    out["ft"] = (ft.init_from, ft.reference_path, ft.run())
    sc = _tiny(scratch, "lr_6x6_s010", "gin", 6, sigma=0.1, reference=main)
    out["sc"] = (sc.init_from, sc.reference_path, sc.run())
    az = _tiny(scratch, "az_6x6_s010", "ffnn", 6, sigma=0.1, reference=main)
    out["az"] = (az.init_from, az.teacher_path, az.run())
    return out


def test_fine_tuned_network_is_validated_in_the_fixed_scenarios(trained):
    main = trained["main"]
    init_from, reference, summary = trained["ft"]
    assert init_from == reference == str(main / "lr_6x6" / "best.pt")        # starts as the nominal network
    assert summary["validation"] == "disturbed" and summary["scenarios"] == 3 and summary["sigma"] == 0.1
    assert summary["uniform_ratio"] == summary["baseline"]["uniform_realised"] > 0
    assert summary["initial"]["agree_reference"] == 1.0                     # the same network at the start
    history = json.loads((main / "lr_6x6_s010" / "history.json").read_text())
    for row in history:
        for key in ("greedy_nominal", "greedy_std", "greedy_p10", "greedy_p50", "greedy_p90", "probe_entropy",
                    "agree_mwkr", "agree_reference"):
            assert key in row and np.isfinite(row[key])
        assert row["greedy_p10"] <= row["greedy_p50"] <= row["greedy_p90"]
        assert 0 <= row["agree_mwkr"] <= 1 and 0 <= row["probe_entropy"] <= 1


def test_networks_from_scratch_start_like_the_nominal_ones(trained):
    main, scratch = trained["main"], trained["scratch"]
    init_from, reference, summary = trained["sc"]
    assert init_from is None and reference == str(main / "lr_6x6" / "best.pt")   # random weights, nominal reference
    assert summary["validation"] == "disturbed" and summary["initial"] is None
    az_from, teacher, az = trained["az"]
    assert az_from == str(teacher) == str(scratch / "lr_6x6_s010" / "best.pt")   # copies the LR of its disturbances
    assert az["validation"] == "disturbed" and az["initial"]["from"] == az_from


def test_the_validation_scenarios_are_the_same_for_every_network():
    from mctsjssp.env.model import State
    from mctsjssp.instances.generator import validation_set
    from mctsjssp.training.disturbed import realised

    inst = validation_set(6, 6)[0]
    st = State(inst)
    while st.active:
        st.step(st.active[0])
    a, b = realised(inst, st, 0.1, 5), realised(inst, st, 0.1, 5)
    assert np.array_equal(a, b) and a.std() > 0 and abs(a.mean() - st.makespan / inst.lb0) < 0.5


def test_a_network_for_disturbances_has_the_seed_of_its_nominal_network(tmp_path):
    from mctsjssp.cli import _train_config

    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    cfg = {"seed": 3, "paths": {"artifacts": str(root / "artifacts")}, "_file": str(root / "configs" / "base.yaml"),
           "training": {"reference_models": "artifacts/models"},
           "models": {"lr_6x6": {"kind": "gin", "n": 6, "m": 6},
                      "lr_6x6_s010": {"kind": "gin", "n": 6, "m": 6, "sigma": 0.1},
                      "lr_10x10": {"kind": "gin", "n": 10, "m": 10}}}
    a, b, c = (_train_config(cfg, n) for n in ("lr_6x6", "lr_6x6_s010", "lr_10x10"))
    assert a.seed == b.seed != c.seed                    # the same random weights and training instances
    assert b.reference_models == str(root / "artifacts" / "models")


def test_robustness_and_report_over_the_iterations(trained, tmp_path):
    import pandas as pd

    from mctsjssp.analysis.disturbed_report import histories, plot
    from mctsjssp.training.robustness import evaluate, write

    root = trained["root"]
    cfg = {"seed": 1, "paths": {"artifacts": str(root), "results": str(tmp_path / "results")},
           "training": {"validation_instances": 3}}
    result = evaluate(cfg, [0.1], workers=1, samples=1, scenarios=3, log=lambda *_: None)
    s = result["summary"]
    six = s[s["size"] == "6x6"]
    assert {(f, v) for f, v in zip(six["family"], six["variant"])} == {
        ("U", "-"), ("MWKR", "-"), ("LR", "N"), ("LR", "FT"), ("LR", "SC"), ("AZ", "N"), ("AZ", "SC")}
    lr_n = six[(six["family"] == "LR") & (six["variant"] == "N")].iloc[0]
    assert lr_n["agree_reference"] == 1.0 and lr_n["validation"] == "nominal"
    assert six[(six["family"] == "LR") & (six["variant"] == "FT")].iloc[0]["validation"] == "disturbed"
    tests = result["tests"]
    assert {("FT", "N"), ("SC", "N"), ("FT", "SC"), ("N", "U"), ("N", "MWKR")} <= set(zip(tests["first"], tests["second"]))
    assert (tests["p_holm"] >= tests["p_value"] - 1e-12).all()
    paths = write(result, tmp_path / "results" / "training")
    assert all(p.exists() for p in paths)
    frame = histories({"FT": root / "models", "SC": root / "scratch" / "models"})
    assert set(zip(frame["network"], frame["variant"])) == {("lr_6x6_s010", "FT"), ("lr_6x6_s010", "SC"),
                                                            ("az_6x6_s010", "SC")}
    assert (frame[frame["variant"] == "FT"]["iteration"] == -1).any()        # the start of the fine-tuned network
    figures = plot(frame, pd.read_csv(tmp_path / "results" / "training" / "robustness.csv"), tmp_path / "fig")
    assert [p.name for p in figures] == ["disturbed_over_time.png", "disturbed_distribution.png"]
    assert all(p.stat().st_size > 10_000 for p in figures)
