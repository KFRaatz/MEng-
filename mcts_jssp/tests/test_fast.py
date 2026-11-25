"""Fast policy networks for the rollouts of LR and AZ (after AlphaGo's fast rollout policy)."""
from __future__ import annotations

import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from conftest import random_instance, random_state
from mctsjssp.distributions import FastRollout, NetworkRollout, StepStats, make_learned_rollout
from mctsjssp.env import State
from mctsjssp.instances import validation_set
from mctsjssp.networks import (FastAZ, FastLR, build_model, candidate_features, load_checkpoint, make_inference,
                               save_checkpoint, state_features)
from mctsjssp.training import (greedy_ratio, imitate_fast, network_feature_records, rule_feature_records,
                               uniform_ratio)


def test_features_by_hand(tiny):
    """After job 0 has been dispatched (machine 0 busy until 3); LB0 = 19."""
    state = State(tiny)
    state.step(0)
    feats = candidate_features(state)
    expected = [  # work share, operations share, time share, start delay, ready delay, completion delay
        [6 / 9, 2 / 3, 2 / 4, 3 / 19, 3 / 19, 3 / 19],
        [7 / 9, 1.0, 2 / 4, 0.0, 0.0, 0.0],
        [8 / 9, 1.0, 4 / 4, 0.0, 0.0, 2 / 19],
        [9 / 9, 1.0, 1 / 4, 3 / 19, 0.0, 2 / 19],
    ]
    assert np.allclose(feats, expected)
    full = state_features(state)
    assert full.shape == (4, 6) and np.allclose(full, expected)
    for _ in range(2):
        state.step(0)                                       # job 0 is finished: its row is zero
    assert not state_features(state)[0].any() and len(candidate_features(state)) == 3


def test_architectures_and_sizes():
    def count(params):
        return sum(p.numel() for p in params)

    # policy (used in the rollouts) and value (used only by the PUCT self-play)
    assert count(FastLR().policy_parameters()) == 129                           # 6*16+16 + 16+1
    assert count(FastLR().parameters()) == 129 + 129                             # value: 6*16+16, mean, 16+1
    assert count(FastLR(hidden=0).policy_parameters()) == 7
    assert count(FastLR(hidden=0).parameters()) == 7 + 7
    assert count(FastAZ(15, 15).policy_parameters()) == 3407                     # 90*32+32 + 32*15+15
    assert count(FastAZ(15, 15).parameters()) == 3407 + 2945                     # value: 90*32+32 + 32+1
    assert count(FastAZ(15, 15, hidden=0).policy_parameters()) == 1365
    assert count(FastAZ(15, 15, hidden=0).parameters()) == 1365 + 91
    with pytest.raises(ValueError):
        FastAZ(6, 6).forward_features(torch.zeros(1, 10, 6), torch.ones(1, 10, dtype=torch.bool))


@pytest.mark.parametrize("make", [lambda: FastLR(), lambda: FastLR(hidden=0), lambda: FastAZ(10, 5),
                                  lambda: FastAZ(10, 5, hidden=0)])
def test_numpy_inference_equals_pytorch(make):
    torch.manual_seed(0)
    model = make().eval()
    inst = random_instance(10, 5, 3, low=1, high=40)
    infer = make_inference(model)
    for steps in (0, 12, 31):
        state = random_state(inst, steps, steps)
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, state.active] = True
        with torch.no_grad():
            reference = model.forward_features(torch.from_numpy(state_features(state))[None], mask)[0, state.active]
        assert np.allclose(infer.score_list(state), reference.numpy(), atol=1e-5)
        assert infer.logits(None, state).shape == (len(state.active),)
        with torch.no_grad():
            value = model.value_features(torch.from_numpy(state_features(state))[None], mask)[0]
        logits, v = infer.evaluate(None, state)
        assert np.allclose(logits, reference.numpy(), atol=1e-5) and abs(v - float(value)) < 1e-5


def test_fast_rollouts(ft06, orlib):
    torch.manual_seed(0)
    for model in (FastLR(), FastLR(hidden=0), FastAZ(6, 6)):
        infer = make_inference(model)
        for level, tau in ((0.0, None), (0.5, 0.4), (1.0, None)):
            policy = make_learned_rollout("LR", infer, level, tau)
            assert isinstance(policy, FastRollout)
            for seed in range(3):
                fast, recorded = [], []
                a = policy.rollout(random_state(ft06, 4, seed), random.Random(seed), fast)
                stats = StepStats(rate=0.3, seed=1)
                b = policy.rollout_recorded(random_state(ft06, 4, seed), random.Random(seed), recorded, stats)
                assert fast == recorded and a == b and len(fast) == ft06.n_ops - 4
            p = policy.distribution(random_state(ft06, 9, 0))
            assert p.sum() == pytest.approx(1.0)
            if level == 0.0:
                assert sorted(p)[-1] == 1.0                                     # greedy
            if level == 1.0:
                assert np.allclose(p, 1.0 / len(p))                             # uniform
    lr_any_size = make_learned_rollout("LR", make_inference(FastLR()), 0.5, 0.5)
    assert lr_any_size.rollout(State(orlib["swv11"]), random.Random(0), []) > 0   # any instance size
    az = make_learned_rollout("AZ", make_inference(FastAZ(6, 6)), 0.5, 0.5)
    with pytest.raises(ValueError, match="6x6"):
        az.rollout(State(orlib["la01"]), random.Random(0), [])


def test_full_networks_keep_their_rollout(gin):
    assert type(make_learned_rollout("LR", make_inference(gin), 0.5, 1.0)) is NetworkRollout


def test_copying_a_rule_beats_random_dispatching():
    feats, masks, targets, outcomes = rule_feature_records("FIFO", 6, 6, count=60, seed=0)
    assert feats.shape[1:] == (6, 6) and len(feats) == len(masks) == len(targets) and len(outcomes) == 60
    assert np.allclose(targets.sum(axis=1), 1.0) and (targets[~masks] == 0).all()
    instances = validation_set(6, 6)[:30]
    for model in (FastLR(), FastAZ(6, 6)):
        torch.manual_seed(0)
        losses = imitate_fast(model, feats, masks, targets, epochs=6, learning_rate=2e-3)
        assert losses[-1] < losses[0]
        assert greedy_ratio(make_inference(model), instances) < uniform_ratio(instances, seed=0)


def test_copying_a_network_uses_its_softmax(gin):
    feats, masks, targets, _ = network_feature_records(make_inference(gin), 6, 6, count=4, seed=0)
    assert np.allclose(targets.sum(axis=1), 1.0, atol=1e-5) and (targets[~masks] == 0).all()
    assert (targets[masks] > 0).all()                                            # soft targets
    with pytest.raises(ValueError):
        rule_feature_records("LPT", 6, 6, count=1, seed=0)


def test_checkpoints(tmp_path):
    for model in (FastLR(), FastAZ(6, 6, hidden=0)):
        path = save_checkpoint(tmp_path / model.kind / "best.pt", model, {"config": {"n": 6, "m": 6}})
        loaded, meta = load_checkpoint(path)
        assert type(loaded) is type(model) and meta["config"]["n"] == 6
        state = random_state(validation_set(6, 6)[0], 10, 0)
        assert np.allclose(make_inference(model).score_list(state), make_inference(loaded).score_list(state))
    assert isinstance(build_model("fast_lr", {"hidden": 8}), FastLR)


@pytest.mark.slow
def test_fast_networks_in_the_pipeline(tmp_path, root):
    from mctsjssp.cli import main
    from mctsjssp.experiments import load_config, run_experiment
    from mctsjssp.experiments.runner import read_results

    base = {
        "inherits": str(root / "configs" / "base.yaml"),
        "paths": {"data": str(root / "data" / "orlib"), "artifacts": str(tmp_path / "artifacts"),
                  "results": str(tmp_path / "results")},
        "runs": 1,
        "entropy": {"calibration": {"instances": 5, "start_depths": 2, "tolerance": 0.05}},
        "time_budget": {"repeats": 1},
        "training": {"fast": {"copy": {"instances": 20, "epochs": 3}}, "validation_instances": 10},
        "models": {"_replace": True, "flr_6x6": {"kind": "fast_lr", "n": 6, "m": 6, "teacher": {"rule": "FIFO"}}},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    assert main(["train-fast", "--config", str(tmp_path / "base.yaml"), "--all"]) == 0
    assert (tmp_path / "artifacts" / "models" / "flr_6x6" / "best.pt").exists()
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({
        "inherits": "base.yaml", "experiment": "fastexp", "instances": ["ft06"],
        "budgets": [{"iterations": 4}, {"time": 4}],
        "configurations": [{"name": "U"}, {"name": "LR", "model": "flr_6x6", "entropy": 0.5}]}))
    rows = read_results(run_experiment(load_config(tmp_path / "exp.yaml"), log=lambda *_: None))
    fast = [r for r in rows if r["config"] == "LR"]
    assert len(rows) == 4 and len(fast) == 2
    assert all(r["network_calls"] > 0 and math.isfinite(r["tau"]) for r in fast)
    with pytest.raises(SystemExit):
        main(["train", "--config", str(tmp_path / "base.yaml"), "--model", "flr_6x6"])


# ------------------------------------------------------------------------------- self-play from scratch
def test_fast_selfplay_episode_records_visit_distributions(ft06):
    from mctsjssp.distributions import UniformRollout
    from mctsjssp.training.fast_trainer import play_fast_episode

    ep = play_fast_episode(ft06, UniformRollout(), iterations=12, seed=3)
    assert ep["makespan"] >= ft06.bks and ep["outcome"] == pytest.approx(ep["makespan"] / ft06.lb0)
    assert len(ep["feats"]) == len(ep["masks"]) == len(ep["targets"]) > 0
    for f, mask, pi in zip(ep["feats"], ep["masks"], ep["targets"]):
        assert f.shape == (6, 6) and pi.sum() == pytest.approx(1.0) and (pi[~mask] == 0).all()
        assert mask.sum() >= 2 and (pi[mask] > 0).all()            # UCT tries every child first
    again = play_fast_episode(ft06, UniformRollout(), iterations=12, seed=3)
    assert again["makespan"] == ep["makespan"]


@pytest.mark.parametrize("kind", ["fast_lr", "fast_az"])
def test_fast_trainer_from_scratch(tmp_path, kind):
    import json

    from mctsjssp.training.fast_trainer import FastTrainConfig, FastTrainer

    cfg = FastTrainConfig(name="scratch", kind=kind, n=4, m=3, episodes_per_generation=3, iterations=8,
                          max_generations=3, validation_instances=8, learning_rate=1e-3, seed=2)
    trainer = FastTrainer(cfg, tmp_path, log=lambda *_: None)
    summary = trainer.run()
    history = json.loads((tmp_path / "history.json").read_text())
    assert [h["search"] for h in history] == ["U", "fast rollouts", "fast rollouts"]   # generation 0 searches with U
    assert all(0.0 < h["target_entropy"] <= 1.0 + 1e-9 for h in history)
    assert summary["best_validation_ratio"] == min(h["validation_ratio"] for h in history)
    model, meta = load_checkpoint(tmp_path / "best.pt")
    assert model.kind == kind and meta["config"]["n"] == 4 and meta["training"] == "self-play from scratch"
    assert greedy_ratio(make_inference(model), trainer.validation) == pytest.approx(summary["best_validation_ratio"])
    budget = FastTrainConfig(name="b", kind=kind, n=4, m=3, episodes_per_generation=2, iterations=4,
                             max_generations=9, validation_instances=4, budget_model_steps=1, seed=0)
    assert FastTrainer(budget, tmp_path / "b", log=lambda *_: None).run()["stopped_by"] == "budget"


def test_sampling_without_an_entropy_level():
    policy = FastRollout("LR", make_inference(FastLR()), level=None, tau=0.5)
    assert policy.level is None and policy.tau == 0.5 and policy.label == "LR"
    assert policy.rollout(State(validation_set(6, 6)[0]), random.Random(0), []) > 0
    with pytest.raises(ValueError):
        FastRollout("LR", make_inference(FastLR()), level=None, tau=None)


def test_chosen_action_targets(ft06):
    from mctsjssp.distributions import UniformRollout
    from mctsjssp.training.fast_trainer import play_fast_episode

    visits = play_fast_episode(ft06, UniformRollout(), iterations=12, seed=3)
    chosen = play_fast_episode(ft06, UniformRollout(), iterations=12, seed=3, target="chosen")
    assert chosen["makespan"] == visits["makespan"]                       # the same search and decisions
    for pv, pc in zip(visits["targets"], chosen["targets"]):
        assert pc.sum() == 1.0 and pc.max() == 1.0 and pv[int(np.argmax(pc))] == pv.max()
    with pytest.raises(ValueError):
        play_fast_episode(ft06, UniformRollout(), iterations=4, seed=0, target="other")


def test_pilot_training_cost_includes_fast_networks():
    """The pilot times fast networks with UCT and network rollouts (they have no value output)."""
    from mctsjssp.experiments.artifacts import Artifacts
    from mctsjssp.experiments.config import load_config
    from mctsjssp.experiments.pilot import training_cost
    from mctsjssp.instances.orlib import load_orlib

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    cfg["training"]["episodes_per_generation"] = 1
    cfg["training"]["iterations"] = 2
    cfg["models"] = {"flr_6x6": {"kind": "fast_lr", "n": 6, "m": 6},
                     "faz_6x6": {"kind": "fast_az", "n": 6, "m": 6}}
    rows = training_cost(cfg, {}, load_orlib(["ft06"]), {(6, 6): "ft06"}, Artifacts(cfg), "numpy")
    assert list(rows["model"]) == ["flr_6x6", "faz_6x6"]
    assert (rows["hours_per_later_generation"] > 0).all()


@pytest.mark.parametrize("kind", ["fast_lr", "fast_az"])
def test_fast_networks_train_by_puct_self_play(tmp_path, kind):
    """The fast networks are trained like LR and AZ: PUCT self-play with policy and value targets."""
    from mctsjssp.training.trainer import TrainConfig, Trainer

    cfg = TrainConfig(name=kind, kind=kind, n=6, m=6, hparams={}, episodes_per_generation=3, iterations=8,
                      max_generations=2, validation_instances=4, buffer_size=500, batch_size=32,
                      learning_rate=5e-4, patience=5)
    summary = Trainer(cfg, tmp_path, log=lambda *_: None).run()
    assert summary["generations"] == 2 and (tmp_path / "best.pt").exists()
    model, _ = load_checkpoint(tmp_path / "best.pt")
    assert model.kind == kind and hasattr(model, "value_features")


def test_training_stops_after_max_hours_without_early_stopping(tmp_path):
    """With patience off, training runs until the wall-clock limit (here a few seconds)."""
    from mctsjssp.training.trainer import TrainConfig, Trainer

    cfg = TrainConfig(name="t", kind="fast_lr", n=6, m=6, hparams={}, episodes_per_generation=2, iterations=4,
                      max_generations=1000, validation_instances=3, buffer_size=200, batch_size=16,
                      learning_rate=5e-4, patience=None, max_hours=3 / 3600)
    summary = Trainer(cfg, tmp_path, log=lambda *_: None).run()
    assert summary["stopped_by"] == "time" and summary["generations"] >= 2
