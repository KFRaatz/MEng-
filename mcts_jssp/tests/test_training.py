"""Self-play training (Section 3.6.1): replay buffer, ranked reward, episodes, trainer, tuning, assessment."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mctsjssp.instances import generate_instance, validation_set
from mctsjssp.networks import load_checkpoint, make_inference
from mctsjssp.training import (EpisodeRunner, RankedReward, Record, ReplayBuffer, TrainConfig, Trainer,
                               greedy_ratio, play_puct_episode, play_uct_episode, uniform_ratio)
from mctsjssp.training.offline import assess_offline
from mctsjssp.training.tuning import random_search, sample_trials, select_entropy_level


def record(i: int, n: int = 2, m: int = 2) -> Record:
    return Record(np.full((n * m, 2), i, np.float32), np.full(n * m, n * m, np.int32), np.zeros(n, np.int16),
                  np.ones(n, bool), np.full(n, 1 / n, np.float32), float(i))


# --------------------------------------------------------------------------------------- replay buffer
def test_replay_buffer_removes_the_oldest_records_first():
    buffer = ReplayBuffer(5, 2, 2)
    for i in range(7):
        buffer.add(record(i))
    assert len(buffer) == 5 and buffer.added == 7
    assert sorted(buffer.z.tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0]       # records 0 and 1 were removed
    buffer.extend(record(i) for i in range(7, 12))
    assert sorted(buffer.z.tolist()) == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_one_epoch_visits_every_record_once():
    buffer = ReplayBuffer(50, 2, 2)
    buffer.extend(record(i) for i in range(23))
    seen = []
    for batch in buffer.epoch(8, np.random.default_rng(0)):
        assert batch["x"].shape[1:] == (4, 2) and batch["mask"].dtype == torch.bool
        assert batch["mpred"].dtype == torch.int64 and batch["next_op"].dtype == torch.int64
        seen += batch["z"].tolist()
    assert sorted(seen) == [float(i) for i in range(23)]
    sizes = [len(b["z"]) for b in buffer.epoch(8, np.random.default_rng(0))]
    assert sizes == [8, 8, 7]


# ---------------------------------------------------------------------------------------- ranked reward
def test_ranked_reward_threshold_and_labels():
    ranked = RankedReward(percentile=75, window=250)
    assert ranked.threshold() is None
    ranked.update([1.0 + 0.01 * i for i in range(100)])                 # outcomes Cmax / LB0, lower is better
    threshold = ranked.threshold()
    assert threshold == pytest.approx(np.percentile([1.0 + 0.01 * i for i in range(100)], 25))
    labels = [RankedReward.label(o, threshold) for o in ranked.history]
    assert sum(1 for z in labels if z > 0) == 25                        # an episode must beat 75 % of the results
    assert RankedReward.label(threshold, threshold) == -1.0             # equal is not better
    assert RankedReward(50).threshold([1, 2, 3, 4, 5]) == 3
    assert RankedReward(90).threshold(range(101)) == pytest.approx(10)


def test_ranked_reward_window_keeps_recent_results():
    ranked = RankedReward(percentile=50, window=10)
    ranked.update(range(100))
    assert list(ranked.history) == list(range(90, 100))
    state = ranked.state()
    assert RankedReward(**state).threshold() == ranked.threshold()
    with pytest.raises(ValueError):
        RankedReward(percentile=100)


# --------------------------------------------------------------------------------------------- episodes
def test_generation_zero_episode_records_visit_distributions():
    inst = generate_instance(4, 3, np.random.default_rng(0))
    data = play_uct_episode(inst, iterations=15, seed=1)
    assert 0 < len(data) <= inst.n_ops and data.makespan > 0
    assert data.outcome == pytest.approx(data.makespan / inst.lb0) and data.outcome >= 1.0
    for x, mpred, next_op, mask, pi in zip(data.x, data.mpred, data.next_op, data.mask, data.pi):
        assert x.shape == (12, 2) and mpred.shape == (12,) and pi.sum() == pytest.approx(1.0)
        assert mask.sum() >= 2                                           # only real decisions are recorded
        assert (pi[~mask] == 0).all()
        assert (x[:, 0].reshape(4, 3).sum(axis=1) == next_op).all()      # flags agree with the next operations
    depth = [int(x[:, 0].sum()) for x in data.x]
    assert depth == sorted(depth)
    assert data.model_steps > inst.n_ops
    again = play_uct_episode(inst, iterations=15, seed=1)
    assert again.makespan == data.makespan and np.array_equal(again.pi[0], data.pi[0])


def test_puct_episode_uses_the_networks(gin, az66):
    inst = generate_instance(6, 6, np.random.default_rng(1))
    for model in (gin, az66):
        infer = make_inference(model)
        data = play_puct_episode(inst, infer, threshold=1.3, iterations=12, seed=2)
        assert len(data) > 0 and data.evaluations > 0
        assert all(pi.sum() == pytest.approx(1.0) and (pi[~mask] == 0).all() for pi, mask in zip(data.pi, data.mask))
        again = play_puct_episode(inst, infer, threshold=1.3, iterations=12, seed=2)
        assert again.makespan == data.makespan
    noisy = play_puct_episode(inst, make_inference(gin), threshold=1.3, iterations=12, seed=2, sigma=0.2)
    assert noisy.outcome != pytest.approx(noisy.makespan / inst.lb0)    # realised in a sampled scenario


# ---------------------------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("kind", ["gin", "ffnn"])
def test_trainer_runs_generations_and_keeps_the_best_checkpoint(tmp_path, kind):
    cfg = TrainConfig(name=f"t_{kind}", kind=kind, n=4, m=3, episodes_per_generation=4, iterations=10,
                      max_generations=3, validation_instances=8, learning_rate=1e-3, seed=3)
    logs: list = []
    trainer = Trainer(cfg, tmp_path / kind, log=logs.append)
    summary = trainer.run()
    assert summary["generations"] == 3 and summary["stopped_by"] == "max_generations"
    history = json.loads((tmp_path / kind / "history.json").read_text())
    assert [h["search"] for h in history] == ["U", "PUCT", "PUCT"]      # generation 0 uses the searches of U
    assert all(np.isfinite([h["policy_loss"], h["value_loss"], h["validation_ratio"]]).all() for h in history)
    assert history[0]["win_share"] <= 0.5                                # 75th percentile: few episodes win
    assert history[-1]["buffer"] == len(trainer.buffer) > 0
    assert summary["best_validation_ratio"] == min(h["validation_ratio"] for h in history)
    model, meta = load_checkpoint(tmp_path / kind / "best.pt")
    assert meta["generation"] == summary["best_generation"] and meta["config"]["n"] == 4
    assert meta["validation_score"] == pytest.approx(summary["best_validation_ratio"])
    assert history[1]["win_share"] <= 0.5                                # also ranked against its own outcomes
    last_meta = load_checkpoint(tmp_path / kind / "last.pt")[1]
    assert len(last_meta["ranked_reward"]["history"]) == 8               # generations 1 and 2; U is dropped
    assert greedy_ratio(make_inference(model), trainer.validation) == pytest.approx(summary["best_validation_ratio"])
    assert len(logs) == 3 and (tmp_path / kind / "last.pt").exists()


def test_training_stops_on_budget_and_patience(tmp_path):
    cfg = TrainConfig(name="budget", kind="ffnn", n=4, m=3, episodes_per_generation=2, iterations=5,
                      max_generations=50, validation_instances=4, budget_model_steps=1, seed=0)
    assert Trainer(cfg, tmp_path / "b", log=lambda *_: None).run()["stopped_by"] == "budget"
    cfg = TrainConfig(name="patience", kind="ffnn", n=4, m=3, episodes_per_generation=2, iterations=5,
                      max_generations=50, validation_instances=4, patience=1, learning_rate=0.0, seed=0)
    summary = Trainer(cfg, tmp_path / "p", log=lambda *_: None).run()
    assert summary["stopped_by"] == "no_improvement" and summary["generations"] == 2   # nothing can improve


def test_ranking_window_and_minimum_generations(tmp_path):
    cfg = TrainConfig(name="keep", kind="ffnn", n=4, m=3, episodes_per_generation=4, iterations=6,
                      max_generations=3, validation_instances=4, rank_against_generation0=True, seed=0)
    trainer = Trainer(cfg, tmp_path / "keep", log=lambda *_: None)
    trainer.run()
    assert len(trainer.ranked.history) == 12                             # the results of U stay in the window
    cfg = TrainConfig(name="min", kind="ffnn", n=4, m=3, episodes_per_generation=2, iterations=5,
                      max_generations=50, validation_instances=4, patience=1, min_generations=4,
                      learning_rate=0.0, seed=0)
    summary = Trainer(cfg, tmp_path / "min", log=lambda *_: None).run()
    assert summary["stopped_by"] == "no_improvement" and summary["generations"] == 4


def test_training_is_reproducible(tmp_path):
    def run(sub):
        cfg = TrainConfig(name="r", kind="gin", n=4, m=3, episodes_per_generation=3, iterations=8,
                          max_generations=2, validation_instances=6, seed=11)
        return Trainer(cfg, tmp_path / sub, log=lambda *_: None).run()
    a, b = run("a"), run("b")
    assert a["best_validation_ratio"] == pytest.approx(b["best_validation_ratio"], rel=1e-6)
    assert a["model_steps"] == b["model_steps"]


@pytest.mark.slow
def test_parallel_self_play_gives_the_same_episodes(gin):
    instances = [generate_instance(4, 3, np.random.default_rng(s)) for s in range(4)]
    seeds = [5, 6, 7, 8]
    inline = EpisodeRunner(1).run(instances, seeds, 8, 1.0, None, model=gin, infer=make_inference(gin), threshold=1.3)
    runner = EpisodeRunner(2)
    try:
        pooled = runner.run(instances, seeds, 8, 1.0, None, model=gin, threshold=1.3)
        zero = runner.run(instances, seeds, 8, 1.0, None)
    finally:
        runner.close()
    assert [e.makespan for e in inline] == [e.makespan for e in pooled]
    assert all(np.allclose(a.pi[0], b.pi[0]) for a, b in zip(inline, pooled))
    assert [e.makespan for e in zero] == [play_uct_episode(i, 8, s).makespan for i, s in zip(instances, seeds)]


# ------------------------------------------------------------------- the learning signal, in isolation
def test_networks_learn_from_informative_targets(tmp_path):
    """The fitting step works: a policy network trained to imitate a dispatching rule dispatches better
    than before and better than uniform random dispatching."""
    from mctsjssp.env import FeatureTracker, State
    from mctsjssp.experiments.rules import fifo

    cfg = TrainConfig(name="imitate", kind="ffnn", n=6, m=6, learning_rate=2e-3, validation_instances=40, seed=0)
    trainer = Trainer(cfg, tmp_path, log=lambda *_: None)
    before = trainer.validate()
    rng = np.random.default_rng(0)
    for _ in range(120):
        inst = generate_instance(6, 6, rng)
        state = State(inst)
        tracker = FeatureTracker(inst).reset(state)
        while state.active:
            job = fifo(state)
            if len(state.active) > 1:
                pi = np.zeros(6, np.float32)
                pi[job] = 1.0
                mask = np.zeros(6, bool)
                mask[state.active] = True
                x, mpred = tracker.snapshot()
                trainer.buffer.add(Record(x, mpred, np.array(state.next_op, np.int16), mask, pi, 0.0))
            state.step(job)
            tracker.update(state)
    losses = [trainer._fit()[0] for _ in range(8)]
    after = trainer.validate()
    assert losses[-1] < losses[0]
    assert after < before and after < trainer.uniform_score


def test_self_play_search_uses_an_informative_value(gin):
    """The PUCT search works: with a value that reflects the quality of a state it dispatches far better
    than with a constant value, with which it is as good as random dispatching."""
    import random

    from mctsjssp.distributions import UniformRollout

    uniform = UniformRollout()

    class Value:
        kind = "stub"

        def __init__(self, informative):
            self.informative, self.rnd = informative, random.Random(0)

        def check(self, inst):
            pass

        def evaluate(self, tracker, state):
            if not self.informative:
                return np.zeros(len(state.active)), -0.5
            ratios = [uniform.rollout(state.copy(), self.rnd, []) / state.inst.lb0 for _ in range(6)]
            return np.zeros(len(state.active)), float(np.clip((1.45 - np.mean(ratios)) / 0.15, -1, 1))

    instances = validation_set(6, 6)[:8]

    def mean_outcome(informative, q_init):
        return float(np.mean([play_puct_episode(inst, Value(informative), threshold=1.3, iterations=50, seed=s,
                                                q_init=q_init).outcome for s, inst in enumerate(instances)]))

    blind = mean_outcome(False, "zero")
    assert mean_outcome(True, "zero") < blind - 0.1
    assert mean_outcome(True, "parent") < blind - 0.1


def test_value_of_unvisited_children(gin, ft06):
    from mctsjssp.env import State
    from mctsjssp.mcts import PUCTSearch

    with pytest.raises(ValueError):
        PUCTSearch(make_inference(gin), lambda st: 0.0, q_init="other")
    # with a value of -1 everywhere, Q = 0 makes every unvisited child look better than a visited one, so
    # the six children of the root are visited once each before any is visited again
    class Constant:
        kind = "stub"

        def check(self, inst):
            pass

        def evaluate(self, tracker, state):
            logits = np.zeros(len(state.active))
            logits[0] = 3.0                                   # a prior that strongly prefers the first job
            return logits, -1.0

    state = State(ft06)
    zero = PUCTSearch(Constant(), lambda st: -1.0, noise_weight=0.0, q_init="zero")
    zero.start(state)
    zero.run_iterations(state, 7)
    assert sorted(c.N for c in zero.root.children) == [1, 1, 1, 1, 1, 1]
    parent = PUCTSearch(Constant(), lambda st: -1.0, noise_weight=0.0, q_init="parent")
    parent.start(state)
    parent.run_iterations(state, 7)
    assert max(c.N for c in parent.root.children) > 1         # the search follows the prior instead


def test_leaf_rollouts_and_search_values(ft06):
    """Training searches after Kemmerling: leaves completed by a rollout, Q rescaled, root values recorded."""
    from mctsjssp.distributions import UniformRollout
    from mctsjssp.env import State
    from mctsjssp.mcts import PUCTSearch
    from mctsjssp.training.selfplay import play_puct_episode, search_value

    assert search_value(1.0) == 1.0 and search_value(1.5) == 0.0 and search_value(2.0) == -1.0
    assert search_value(3.0) == -1.0 and search_value(0.9) == 1.0          # clipped to [-1, 1]

    class Blind:                                              # the value network knows nothing: v = 0
        kind = "stub"

        def check(self, inst):
            pass

        def evaluate(self, tracker, state):
            return np.zeros(len(state.active)), 0.0

    lb0 = ft06.lb0
    values = []
    search = PUCTSearch(Blind(), lambda st: values.append(st.makespan) or search_value(st.makespan / lb0),
                        noise_weight=0.0, q_init="parent", rollout=UniformRollout(), mix=0.5, normalize_q=True)
    state = State(ft06)
    search.start(state)
    search.run_iterations(state, 20)
    assert len(values) >= 19                                  # every new leaf was completed by a rollout
    assert search.model_steps > 19 * (ft06.n_ops - 2)        # ... and its steps count as model steps
    mean_leaf = np.mean([0.5 * search_value(c / lb0) for c in values])
    assert abs(search.root.W / search.root.N - mean_leaf) < 1e-9   # leaf value = 0.5 v + 0.5 rollout
    assert search._qmax > search._qmin                        # extremes for the rescaled Q

    blind = Blind()
    ep = play_puct_episode(ft06, blind, 1.3, 20, 3, q_init="parent", leaf_rollouts=True, mix=0.5,
                           value_target="search", normalize_q=True)
    assert len(ep.v) == len(ep) and all(-1.0 <= v <= 1.0 for v in ep.v)
    assert len(set(np.round(ep.v, 6))) > 3                   # values of the searches, not a +-1 label


def test_trainer_learns_search_values(tmp_path):
    cfg = TrainConfig(name="kem", kind="ffnn", n=6, m=6, episodes_per_generation=3, iterations=12,
                      max_generations=2, validation_instances=10, seed=4, q_init="parent",
                      value_target="search", leaf_rollouts=True, rollout_mix=0.5, normalize_q=True)
    trainer = Trainer(cfg, tmp_path, log=lambda *_: None)
    trainer.run()
    z = trainer.buffer.z[: trainer.buffer.size]
    history = json.loads((tmp_path / "history.json").read_text())
    assert [h["search"] for h in history] == ["U", "PUCT"]
    assert history[1]["model_steps"] - history[0]["model_steps"] > 3 * 30 * 12   # rollouts are counted
    assert np.all(np.abs(z) <= 1.0) and len(np.unique(np.round(z, 4))) > 5      # search values, not +-1


def test_self_label_records_reproduce_the_schedule(ft06):
    from mctsjssp.env import State
    from mctsjssp.training.self_label import label_records

    rnd = np.random.default_rng(0)
    st, seq = State(ft06), []
    while st.active:
        j = int(rnd.choice(st.active))
        seq.append(j)
        st.step(j)
    for fast in (False, True):
        recs = label_records(ft06, seq, fast)
        decisions = [j for j, k in zip(seq, _active_counts(ft06, seq)) if k > 1]
        assert len(recs) == len(decisions)
        targets = [r[2] if fast else r.pi for r in recs]
        assert [int(np.argmax(t)) for t in targets] == decisions          # one-hot at the label's choice
        assert all(abs(float(t.sum()) - 1.0) < 1e-6 for t in targets)


def _active_counts(inst, seq):
    from mctsjssp.env import State

    st, out = State(inst), []
    for j in seq:
        out.append(len(st.active))
        st.step(j)
    return out


@pytest.mark.parametrize("kind,hp", [("gin", {}), ("fast_lr", {"hidden": 16})])
def test_self_label_trainer(tmp_path, kind, hp):
    from mctsjssp.training.self_label import SelfLabelTrainer

    sl = {"samples": 4, "instances": 3, "epochs": 2, "window": 2, "iterations": 2, "validation_samples": 1}
    cfg = TrainConfig(name=kind, kind=kind, n=6, m=6, hparams=hp, validation_instances=8, seed=2, workers=1,
                      method="self_label", self_label=sl)
    summary = SelfLabelTrainer(cfg, tmp_path, log=lambda *_: None).run()
    history = json.loads((tmp_path / "history.json").read_text())
    assert summary["method"] == "self_label" and summary["generations"] == 2 == len(history)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert history[-1]["model_steps"] == 2 * 3 * 4 * 36                   # iterations x instances x samples x ops
    for row in history:
        assert row["label_ratio"] <= row["selfplay_ratio"] + 1e-9         # best of the samples <= their mean
        assert row["validation_ratio"] == pytest.approx((row["sampled_ratio"] + row["greedy_ratio"]) / 2)
        assert 0 <= row["policy_entropy"] <= 1
    assert summary["best_validation_ratio"] == min(r["validation_ratio"] for r in history)
    assert summary["initial"] is None                                      # no smaller network to start from


def _tiny_self_label(models, name, kind, n, sigma=None, **extra):
    from mctsjssp.training.self_label import SelfLabelTrainer

    sl = {"samples": 2, "instances": 2, "epochs": 1, "window": 1, "iterations": 1, "validation_samples": 1, **extra}
    cfg = TrainConfig(name=name, kind=kind, n=n, m=n, validation_instances=3, seed=1, workers=1, sigma=sigma,
                      method="self_label", self_label=sl)
    return SelfLabelTrainer(cfg, models / name, log=lambda *_: None)


def test_self_label_curriculum(tmp_path):
    import torch

    models = tmp_path / "models"
    _tiny_self_label(models, "lr_6x6", "gin", 6).run()
    _tiny_self_label(models, "az_6x6", "ffnn", 6).run()
    big = _tiny_self_label(models, "lr_10x10", "gin", 10)                  # GIN: from the smaller size
    assert big.init_from == str(models / "lr_6x6" / "best.pt")
    small = torch.load(models / "lr_6x6" / "best.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(v, big.model.state_dict()[k]) for k, v in small.items())
    summary = big.run()
    assert summary["initial"]["from"] == big.init_from and summary["best_validation_ratio"] <= summary["initial"]["score"]
    assert _tiny_self_label(models, "az_10x10", "ffnn", 10, teacher=None).init_from is None   # depends on the size
    assert _tiny_self_label(models, "az_10x10", "ffnn", 10).init_from == str(models / "lr_10x10" / "best.pt")
    twin = _tiny_self_label(models, "az_6x6_s010", "ffnn", 6, sigma=0.1)            # disturbances: the twin
    assert twin.init_from == str(models / "az_6x6" / "best.pt")
    assert _tiny_self_label(models, "lr_10x10_x", "gin", 10, init=None).init_from is None
    assert _tiny_self_label(models, "lr_15x15", "gin", 15).init_from == str(models / "lr_10x10" / "best.pt")
    for d in ("lr_6x6", "lr_10x10"):
        (models / d / "summary.json").unlink()                             # unfinished networks are not used
    assert _tiny_self_label(models, "lr_15x15", "gin", 15).init_from is None
    from mctsjssp.networks.checkpoint import build_model, save_checkpoint

    save_checkpoint(models / "lr_6x6" / "best.pt", build_model("gin", {}), {"method": "alphazero"})
    (models / "lr_6x6" / "summary.json").write_text("{}")
    with pytest.raises(SystemExit):                                        # a self-play network is refused
        _tiny_self_label(models, "lr_15x15", "gin", 15)


def test_self_label_teacher(tmp_path):
    models = tmp_path / "models"
    with pytest.raises(SystemExit):                                        # the teacher must be trained first
        _tiny_self_label(models, "az_6x6", "ffnn", 6, teacher="lr")
    _tiny_self_label(models, "lr_6x6", "gin", 6).run()
    student = _tiny_self_label(models, "az_6x6", "ffnn", 6, teacher="lr", teacher_instances=4, teacher_epochs=2)
    assert student.init_from == str(models / "lr_6x6" / "best.pt")
    summary = student.run()
    assert summary["initial"]["from"] == student.init_from and summary["generations"] == 1
    assert _tiny_self_label(models, "flr_6x6", "fast_lr", 6, teacher="lr").teacher_path is None   # size-free
    per_kind = _tiny_self_label(models, "az_6x6", "ffnn", 6, teacher={"ffnn": "lr", "fast_az": "flr"})
    assert per_kind.init_from == str(models / "lr_6x6" / "best.pt")
    assert _tiny_self_label(models, "az_6x6", "ffnn", 6, teacher={"fast_az": "flr"}).init_from is None


@pytest.mark.parametrize("kind", ["ffnn", "fast_az"])
def test_job_relabelling_fits_only_the_policy(tmp_path, kind):
    import torch

    from mctsjssp.env import RULES, State
    from mctsjssp.instances.generator import training_instances
    from mctsjssp.training.self_label import label_records

    trainer = _tiny_self_label(tmp_path, kind, kind, 6, epochs=6, learning_rate=3e-3)
    data = []
    for inst in training_instances(6, 6, 30, 3, 0):
        st, seq = State(inst), []
        while st.active:
            seq.append(RULES["MWKR"](st, None))
            st.step(seq[-1])
        data += label_records(inst, seq, trainer.fast)
    value = trainer.model.value
    before = [p.detach().clone() for p in value.parameters()]
    losses = trainer._fit_relabelled(data)
    assert len(losses) == 6 and losses[-1] < losses[0]                     # it learns the rule
    assert all(torch.equal(a, b) for a, b in zip(before, value.parameters()))


def test_self_label_is_the_default_training_method():
    import yaml

    base = yaml.safe_load(open("configs/base.yaml"))
    assert base["training"]["method"] == "self_label" and base["training"]["self_label"]["samples"] == 32
    names = list(base["models"])                  # curriculum and teachers: smaller sizes and teachers come first
    for name in names:
        kind, size = base["models"][name]["kind"], name.split("_")[1]
        teacher = {"ffnn": "lr", "fast_az": "flr"}.get(kind)
        if teacher:
            assert names.index(f"{teacher}_{size}") < names.index(name)


# ------------------------------------------------------------------------------------------ warm start
def test_rule_episodes_record_the_rule_choices():
    from mctsjssp.env import RULES, State
    from mctsjssp.training import rule_episodes

    records, outcomes = rule_episodes("MWKR", 4, 3, count=5, seed=0)
    assert len(outcomes) == 5 and all(o >= 1.0 for o in outcomes)
    assert all(r.pi.sum() == 1.0 and r.mask.sum() >= 2 and r.pi[r.mask].max() == 1.0 for r in records)
    # the recorded choice is the job that MWKR picks among the unfinished jobs
    first = records[0]
    remaining = [j for j in range(4) if first.mask[j]]
    assert int(np.argmax(first.pi)) in remaining
    again, _ = rule_episodes("MWKR", 4, 3, count=5, seed=0)
    assert all(np.array_equal(a.pi, b.pi) for a, b in zip(records, again))
    nd_records, nd_outcomes = rule_episodes("MWKR", 4, 3, count=5, seed=0, non_delay=True)
    assert len(nd_outcomes) == 5
    with pytest.raises(ValueError):
        rule_episodes("LPT", 4, 3, count=1, seed=0)
    assert set(RULES) >= {"FIFO", "MWKR"} and State is not None


@pytest.mark.parametrize("kind", ["gin", "ffnn"])
def test_imitation_trains_only_the_policy(kind):
    import torch

    from mctsjssp.networks import build_model
    from mctsjssp.training import imitate, rule_episodes

    torch.manual_seed(0)
    model = build_model(kind, {"n": 6, "m": 6} if kind == "ffnn" else {})
    value_head = model.value_head if kind == "gin" else model.value
    before = [p.detach().clone() for p in value_head.parameters()]
    records, _ = rule_episodes("FIFO", 6, 6, count=40, seed=1)
    losses = imitate(model, records, 6, 6, epochs=4, learning_rate=2e-3)
    assert len(losses) == 4 and losses[-1] < losses[0]
    assert all(torch.equal(a, b) for a, b in zip(before, value_head.parameters()))   # value untouched
    assert not model.training


def test_trainer_with_warm_start(tmp_path):
    cfg = TrainConfig(name="warm", kind="ffnn", n=6, m=6, episodes_per_generation=3, iterations=8,
                      max_generations=2, validation_instances=20, seed=2,
                      warm_start={"rule": "FIFO", "instances": 60, "epochs": 6, "learning_rate": 2e-3})
    trainer = Trainer(cfg, tmp_path, log=lambda *_: None)
    summary = trainer.run()
    ws = summary["warm_start"]
    assert ws["rule"] == "FIFO" and ws["records"] > 0 and ws["losses"][-1] < ws["losses"][0]
    assert ws["validation_ratio"] < trainer.uniform_score                     # copying FIFO beats random
    history = json.loads((tmp_path / "history.json").read_text())
    assert [h["search"] for h in history] == ["PUCT", "PUCT"]                  # no generation of U
    assert history[0]["threshold"] is not None                                # ranked against the rule
    assert summary["best_validation_ratio"] <= ws["validation_ratio"]
    assert (tmp_path / "warm_start.pt").exists()
    model, meta = load_checkpoint(tmp_path / "best.pt")
    assert meta["generation"] == summary["best_generation"] >= -1
    assert len(trainer.ranked.history) == min(250, 60 + 3 * 2)                # rule results stay in the window


def test_warm_start_of_fast_networks_and_per_kind_values(tmp_path):
    """The fast networks copy the rule on their own features; a nested entry overrides values per kind."""
    ws = {"rule": "FIFO", "instances": 10, "epochs": 6, "learning_rate": 2e-3, "fast_lr": {"instances": 40}}
    for kind, hp, count in (("fast_lr", {"hidden": 16}, 40), ("fast_az", {"n": 6, "m": 6, "hidden": 32}, 10)):
        cfg = TrainConfig(name=kind, kind=kind, n=6, m=6, hparams=hp, episodes_per_generation=2, iterations=8,
                          max_generations=1, validation_instances=20, seed=3, warm_start=ws)
        trainer = Trainer(cfg, tmp_path / kind, log=lambda *_: None)
        summary = trainer.run()
        w = summary["warm_start"]
        assert w["losses"][-1] < w["losses"][0]
        assert len(trainer.ranked.history) == count + 2                      # outcomes of the rule's episodes
        assert w["records"] > count * 6                                       # several decisions per instance
        assert (tmp_path / kind / "warm_start.pt").exists()
    assert ws["fast_lr"] == {"instances": 40}                                 # the setting itself is unchanged


# ------------------------------------------------------------------------------- tuning and assessment
def test_random_search_samples_the_spaces_of_table_3_5():
    trials = sample_trials(200, seed=0)
    rates = np.array([t["learning_rate"] for t in trials])
    assert rates.min() >= 1e-5 and rates.max() <= 1e-3
    assert 0.3 < np.mean(np.log10(rates) < -4) < 0.7                     # log-uniform: half below 1e-4
    assert {t["percentile"] for t in trials} == {50.0, 75.0, 90.0}
    assert sample_trials(5, seed=0) == sample_trials(5, seed=0) != sample_trials(5, seed=1)


def test_random_search_picks_the_best_trial(tmp_path):
    base = TrainConfig(name="tune", kind="ffnn", n=4, m=3, episodes_per_generation=2, iterations=5,
                       max_generations=1, validation_instances=4, seed=0)
    result = random_search(base, tmp_path, trials=3, seed=0, log=lambda *_: None)
    assert len(result["trials"]) == 3
    assert result["best"]["validation_ratio"] == min(t["validation_ratio"] for t in result["trials"])
    assert json.loads((tmp_path / "tuning.json").read_text())["best"] == result["best"]


def test_entropy_level_is_chosen_by_mcts_on_validation_instances():
    from mctsjssp.distributions import MWKRRollout

    instances = [generate_instance(4, 3, np.random.default_rng(s)) for s in range(3)]
    taus = {0.0: None, 0.5: 0.05}
    result = select_entropy_level(lambda level: MWKRRollout(level, taus[level]), instances, [0.0, 0.5],
                                  iterations=10, seed=0)
    assert set(result["scores"]) == {0.0, 0.5} and result["best"] in (0.0, 0.5)
    assert result["scores"][result["best"]] == min(result["scores"].values())
    assert all(score >= 1.0 for score in result["scores"].values())


def test_offline_assessment(gin, az66):
    instances = validation_set(6, 6)[:5]
    for model in (gin, az66):
        report = assess_offline(make_inference(model), instances,
                                {"percentile": 75, "window": 250, "history": [1.2, 1.3, 1.4, 1.5]}, timing_calls=5)
        assert report["greedy_ratio"] >= 1.0 and report["uniform_ratio"] >= 1.0 and report["mwkr_ratio"] >= 1.0
        assert 0.0 <= report["agreement_with_mwkr"] <= 1.0 and 0.0 <= report["policy_entropy_tau1"] <= 1.0
        assert report["value_mse"] >= 0 and 0.0 <= report["value_sign_accuracy"] <= 1.0
        assert report["seconds_per_evaluation"] > 0
    assert uniform_ratio(instances, seed=0) == uniform_ratio(instances, seed=0)


def test_training_report_requires_only_some_networks(tmp_path):
    from mctsjssp.cli import main

    for name, val in [("lr_6x6", 1.40), ("az_6x6", 1.70)]:            # uniform 1.57: LR learned, AZ did not
        d = tmp_path / "art" / "models" / name
        d.mkdir(parents=True)
        row = {"generation": 0, "validation_ratio": val, "uniform_ratio": 1.57, "policy_loss": 1.0,
               "value_loss": float("nan"), "greedy_ratio": val, "sampled_ratio": val, "label_ratio": val - 0.1,
               "policy_entropy": 0.9, "seconds": 1.0, "model_steps": 10}
        (d / "history.json").write_text(json.dumps([row]))
        (d / "summary.json").write_text(json.dumps({"best_validation_ratio": val, "best_generation": 0,
                                                    "uniform_ratio": 1.57, "method": "self_label"}))
    base = ["training-report", "--config", "configs/base.yaml", "--set", f"paths.artifacts={tmp_path / 'art'}",
            "--set", f"paths.results={tmp_path / 'res'}", "--strict"]
    assert main(base) == 3                                                  # every network must have learned
    assert main(base + ["--require", "lr_"]) == 0                           # only LR must have learned
    assert main(base + ["--require", "lr_", "--require", "az_"]) == 3


def test_compare_training_judges_both_styles_alike(tmp_path):
    import pandas as pd

    from mctsjssp.cli import main

    _tiny_self_label(tmp_path / "art" / "models", "lr_6x6", "gin", 6).run()
    cfg = TrainConfig(name="lr_6x6", kind="gin", n=6, m=6, episodes_per_generation=2, iterations=4,
                      max_generations=1, validation_instances=3, seed=1, method="alphazero")
    Trainer(cfg, tmp_path / "sp" / "models" / "lr_6x6", log=lambda *_: None).run()
    args = ["compare-training", "--config", "configs/base.yaml", "--set", f"paths.artifacts={tmp_path / 'art'}",
            "--set", f"paths.results={tmp_path / 'res'}", "--set", "training.validation_instances=3",
            "--model", "lr_6x6", "--model", "az_6x6", "--other", str(tmp_path / "sp"), "--workers", "1"]
    assert main(args) == 0
    table = pd.read_csv(tmp_path / "res" / "training" / "compare_training.csv")
    assert list(table["style"]) == ["self-labelling", "self-play"] * 2
    done = table[table["status"] == "trained"]
    assert list(done["model"]) == ["lr_6x6", "lr_6x6"] and list(done["method"]) == ["self_label", "alphazero"]
    assert done["greedy"].between(1, 3).all() and done["sampled"].between(1, 3).all()
    assert done["entropy"].between(0, 1).all() and (table["uniform"] > 1).all()
    assert (table[table["model"] == "az_6x6"]["status"] == "not trained").all()
    assert done["p_greedy"].between(0, 1).all() and done["rank_biserial_sampled"].between(-1, 1).all()
    assert (tmp_path / "res" / "training" / "compare_training.png").exists()


def test_pipeline_runs_self_play_into_its_own_folder():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "run_pipeline.py", "--dry-run", "--steps", "self_play,compare_training"],
                         capture_output=True, text=True, check=True).stdout
    train = out[out.index(">>> python -m mctsjssp train"): out.index("=== compare_training")]
    assert "--set paths.artifacts=artifacts/self_play" in train
    assert train.index("training.method=self_label") < train.index("training.method=alphazero")   # the last wins
    assert "compare-training" in out and "--other artifacts/self_play" in out


def test_fit_threads_use_one_thread_per_physical_core_unless_given():
    import os

    import torch

    from mctsjssp.training.trainer import fit_thread_count

    cpus = os.cpu_count() or 1
    assert fit_thread_count("auto") == fit_thread_count(None) == max(1, cpus // 2)
    assert fit_thread_count(1) == 1 and fit_thread_count(10_000) == cpus
    before = torch.get_num_threads()
    tc = TrainConfig(name="t", kind="ffnn", n=3, m=3, episodes_per_generation=2, iterations=4, max_generations=1,
                     validation_instances=2, seed=1)
    assert tc.fit_threads == "auto"
    assert torch.get_num_threads() == before
