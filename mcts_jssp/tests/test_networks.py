"""Networks (Section 3.6.2, Table 3.4): architecture, the three equivalent ways of evaluating them, checkpoints."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from conftest import random_instance, random_state
from mctsjssp.env import FeatureTracker, State
from mctsjssp.networks import (AZNetworks, build_model, file_fingerprint, load_checkpoint, make_inference,
                               save_checkpoint)


def batch_of(state: State, tracker: FeatureTracker):
    inst = state.inst
    mask = torch.zeros(1, inst.n, dtype=torch.bool)
    mask[0, state.active] = True
    return (torch.from_numpy(tracker.x.copy())[None], torch.from_numpy(tracker.mpred.copy())[None],
            torch.tensor([state.next_op]), mask)


# ------------------------------------------------------------------------------------------ architecture
def test_gin_architecture_follows_table_3_4(gin):
    assert len(gin.convs) == 2                                         # two GIN layers
    linears = [m for m in gin.convs[0].nn if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(2, 64), (64, 64), (64, 64)]   # 2 hidden layers
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in gin.convs[0].nn) == 2
    assert float(gin.convs[0].eps) == 0.0
    policy = [m for m in gin.policy_head if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in policy] == [(128, 32), (32, 32), (32, 1)]    # [h_a; h_G]
    value = [m for m in gin.value_head if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in value] == [(64, 32), (32, 32), (32, 1)]
    assert sum(isinstance(m, torch.nn.Tanh) for m in gin.policy_head) == 2


def test_az_architecture_follows_table_3_4():
    for (n, m), size in (((6, 6), 72), ((10, 10), 200), ((15, 15), 450)):
        net = AZNetworks(n, m)
        policy = [l for l in net.policy if isinstance(l, torch.nn.Linear)]
        value = [l for l in net.value if isinstance(l, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in policy] == [(size, 256), (256, 256), (256, n)]
        assert [(l.in_features, l.out_features) for l in value] == [(size, 256), (256, 256), (256, 1)]
        assert sum(isinstance(l, torch.nn.Mish) for l in net.policy) == 2
        assert not set(map(id, net.policy.parameters())) & set(map(id, net.value.parameters()))   # separate


# ------------------------------------------------------------------------------- equivalent evaluations
@pytest.mark.parametrize("n, m, steps", [(6, 6, 0), (6, 6, 17), (10, 5, 23), (5, 10, 31), (15, 15, 120)])
def test_gin_batch_single_and_numpy_agree(gin, n, m, steps):
    """PyTorch Geometric message passing, the gather form and the NumPy export compute the same function."""
    inst = random_instance(n, m, seed=n * m + steps, low=1, high=99)
    state = random_state(inst, steps, seed=steps)
    tracker = FeatureTracker(inst).reset(state)
    with torch.no_grad():
        logits_batch, value_batch = gin.forward_batch(*batch_of(state, tracker))
    reference = logits_batch[0, state.active].numpy()
    assert torch.isinf(logits_batch[0, [j for j in range(n) if j not in state.active]]).all()
    for backend in ("torch", "numpy"):
        logits, value = make_inference(gin, backend).evaluate(tracker, state)
        assert np.allclose(logits, reference, atol=1e-5), backend
        assert value == pytest.approx(float(value_batch[0]), abs=1e-5)
        assert -1.0 <= value <= 1.0
        assert np.allclose(make_inference(gin, backend).logits(tracker, state), reference, atol=1e-5)


def test_gin_reads_graphs_of_any_size_in_one_batch_shape(gin):
    for n, m in ((3, 2), (20, 5), (50, 10)):
        inst = random_instance(n, m, seed=1, low=1, high=50)
        state = State(inst)
        logits = make_inference(gin).logits(FeatureTracker(inst).reset(state), state)
        assert logits.shape == (n,) and np.isfinite(logits).all()


@pytest.mark.parametrize("n, m", [(6, 6), (10, 5), (15, 15), (20, 10)])
def test_incremental_gin_evaluation_equals_full_evaluation(gin, n, m):
    """Along a rollout only the changed node embeddings are recomputed; the logits must not change."""
    inst = random_instance(n, m, seed=7 * n + m, low=0, high=40)       # zero durations and insertions included
    infer = make_inference(gin, "numpy")
    rnd = random.Random(n + m)
    for start in (0, inst.n_ops // 3):
        state = random_state(inst, start, seed=start)
        live = FeatureTracker(inst).reset(state)
        fresh = FeatureTracker(inst)
        infer.begin(live, state)
        while len(state.active) > 1:
            incremental = infer.rollout_logits(live, state)
            full = infer.logits(fresh.reset(state), state)
            assert np.allclose(incremental, full, atol=1e-4)
            state.step(rnd.choice(state.active))
            live.update(state)
            infer.advance(live, state)


def test_az_backends_agree_and_check_the_size(az66, ft06, orlib):
    state = random_state(ft06, 14, 2)
    tracker = FeatureTracker(ft06).reset(state)
    with torch.no_grad():
        logits_batch, value_batch = az66.forward_batch(*batch_of(state, tracker))
    for backend in ("torch", "numpy"):
        infer = make_inference(az66, backend)
        logits, value = infer.evaluate(tracker, state)
        assert logits.shape == (len(state.active),)
        assert np.allclose(logits, logits_batch[0, state.active].numpy(), atol=1e-5)
        assert value == pytest.approx(float(value_batch[0]), abs=1e-5) and -1 <= value <= 1
        with pytest.raises(ValueError, match="6x6"):
            infer.check(orlib["la01"])                                  # one pair of networks per size
    with pytest.raises(ValueError):
        az66.forward_batch(torch.zeros(1, 50, 2), torch.zeros(1, 50, dtype=torch.long),
                           torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.bool))


def test_finished_jobs_are_masked(gin, az66, ft06):
    state = State(ft06)
    for _ in range(ft06.m):
        state.step(3)                                                    # job 3 is finished
    tracker = FeatureTracker(ft06).reset(state)
    for model in (gin, az66):
        with torch.no_grad():
            logits, _ = model.forward_batch(*batch_of(state, tracker))
        assert torch.isinf(logits[0, 3]) and torch.isfinite(logits[0, [0, 1, 2, 4, 5]]).all()
        probabilities = torch.softmax(logits, dim=1)
        assert probabilities[0, 3] == 0 and probabilities.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------- training
@pytest.mark.parametrize("kind", ["gin", "ffnn"])
def test_both_losses_reach_the_weights(kind, ft06):
    torch.manual_seed(0)
    model = build_model(kind, {"n": 6, "m": 6} if kind == "ffnn" else {})
    model.train()
    states = [random_state(ft06, s, s) for s in (0, 5, 11, 20)]
    trackers = [FeatureTracker(ft06).reset(s) for s in states]
    parts = [batch_of(s, t) for s, t in zip(states, trackers)]
    x, mpred, next_op, mask = (torch.cat([p[i] for p in parts]) for i in range(4))
    logits, value = model.forward_batch(x, mpred, next_op, mask)
    pi = torch.where(mask, torch.ones_like(logits), torch.zeros_like(logits))
    pi = pi / pi.sum(dim=1, keepdim=True)
    logp = torch.where(mask, torch.log_softmax(logits, dim=1), torch.zeros_like(logits))
    loss = -(pi * logp).sum(dim=1).mean() + ((value - torch.tensor([1.0, -1.0, 1.0, -1.0])) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    if kind == "gin":       # the shared encoder is trained by both heads
        assert grads["convs.0.nn.0.weight"].abs().sum() > 0
        assert grads["policy_head.0.weight"].abs().sum() > 0 and grads["value_head.0.weight"].abs().sum() > 0


def test_checkpoint_round_trip(tmp_path, gin, az66, ft06):
    state = random_state(ft06, 9, 0)
    tracker = FeatureTracker(ft06).reset(state)
    for model in (gin, az66):
        path = save_checkpoint(tmp_path / model.kind / "best.pt", model, {"name": "x", "config": {"n": 6, "m": 6},
                                                                         "ranked_reward": {"history": [1.2, 1.3]}})
        loaded, meta = load_checkpoint(path)
        assert type(loaded) is type(model) and meta["config"] == {"n": 6, "m": 6}
        assert not loaded.training
        a = make_inference(model).evaluate(tracker, state)
        b = make_inference(loaded).evaluate(tracker, state)
        assert np.allclose(a[0], b[0]) and a[1] == pytest.approx(b[1])
        assert len(file_fingerprint(path)) == 16 and file_fingerprint(path) == file_fingerprint(path)
    with pytest.raises(ValueError):
        build_model("transformer")
    with pytest.raises(TypeError):
        make_inference(torch.nn.Linear(2, 2))
