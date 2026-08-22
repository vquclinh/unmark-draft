"""Torch-gated Stage-1 contracts on a TOY module. **No real model, no corpus.**

Skipped in the ML-free local `.venv`; these run on Colab. They exercise the two
things only tensors can prove -- that the optimizer touches the adapter and
nothing else, and that a resumed run is numerically the run it continues.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - ML-free local environment
    torch = None
    nn = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


class ToyFrozenEncoder(nn.Module if nn else object):
    """Stands in for the frozen backbone. Deliberately tiny."""

    def __init__(self, hidden=8):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


class ToyAdapter(nn.Module if nn else object):
    """Parameter names mirror the real adapter's decay groups."""

    def __init__(self, hidden=8):
        super().__init__()
        self.fusion = nn.Linear(3 * hidden, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.layernorm = nn.LayerNorm(hidden)
        self.tone_embedding = nn.Embedding(7, hidden)
        self.letter_embedding = nn.Embedding(5, hidden)


@requires_torch
def test_the_optimizer_receives_only_trainable_adapter_parameters():
    from unmark.stage1.optim import build_optimizer
    from unmark.stage1.protocol import WEIGHT_DECAY_EXEMPT, WEIGHT_DECAY_WEIGHTS

    adapter = ToyAdapter()
    optimizer = build_optimizer(list(adapter.named_parameters()), 3e-4)
    groups = optimizer.param_groups
    assert len(groups) == 2
    assert {g["weight_decay"] for g in groups} == {WEIGHT_DECAY_WEIGHTS, WEIGHT_DECAY_EXEMPT}

    decayed = next(g for g in groups if g["weight_decay"] == WEIGHT_DECAY_WEIGHTS)
    # only the two weight matrices are decayed; embeddings/LayerNorm/bias are not
    assert len(decayed["params"]) == 2
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == sum(p.numel() for p in adapter.parameters())


@requires_torch
def test_a_frozen_parameter_reaching_the_optimizer_is_refused():
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.optim import build_optimizer

    encoder = ToyFrozenEncoder()
    with pytest.raises(Stage1ContractViolation, match="do not require grad"):
        build_optimizer(list(encoder.named_parameters()), 3e-4)


@requires_torch
def test_optimizer_hyperparameters_are_the_locked_ones():
    from unmark.stage1.optim import build_optimizer
    from unmark.stage1.protocol import ADAMW_BETAS, ADAMW_EPS, AMSGRAD

    optimizer = build_optimizer(list(ToyAdapter().named_parameters()), 1e-3)
    assert isinstance(optimizer, torch.optim.AdamW)
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-3
        assert tuple(group["betas"]) == ADAMW_BETAS
        assert group["eps"] == ADAMW_EPS
        assert group["amsgrad"] is AMSGRAD


@requires_torch
def test_frozen_parameters_receive_no_gradient_through_a_backward():
    encoder = ToyFrozenEncoder()
    adapter = ToyAdapter()
    x = torch.randn(4, 8)
    out = encoder.linear(adapter.layernorm(x))
    out.sum().backward()
    assert all(p.grad is None for p in encoder.parameters()), "the frozen encoder got gradients"
    assert adapter.layernorm.weight.grad is not None


@requires_torch
def test_resumed_training_matches_uninterrupted_training_on_a_toy_state():
    """Same optimizer, same order, split into two segments -> identical weights."""
    from unmark.stage1.optim import build_optimizer
    from unmark.stage1.sampler import DeterministicSampler

    ids = tuple(f"c{i}" for i in range(24))
    data = {cid: torch.full((8,), float(i) / 24) for i, cid in enumerate(ids)}

    def run(segments):
        torch.manual_seed(0)
        adapter = ToyAdapter()
        optimizer = build_optimizer(list(adapter.named_parameters()), 1e-2)
        sampler = DeterministicSampler(ids, seed=36930)
        state = None
        for steps in segments:
            if state is not None:
                sampler = DeterministicSampler.from_state(ids, state)
            for _ in range(steps):
                batch = sampler.next_batch(4)
                x = torch.stack([data[cid] for cid, _ in batch])
                loss = adapter.layernorm(x).pow(2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            state = sampler.state_dict()
        return adapter.layernorm.weight.detach().clone()

    straight = run([6])
    resumed = run([2, 4])
    assert torch.allclose(straight, resumed), (
        "a resumed run must be numerically the run it continues"
    )


@requires_torch
def test_the_cosine_objective_is_finite_and_batch_meaned_on_a_toy_input():
    from unmark.stage1.objective import representation_distance

    a = torch.randn(5, 8)
    b = torch.randn(5, 8)
    distances = representation_distance(a, b)
    assert distances.shape == (5,)
    assert torch.isfinite(distances).all()
    assert (distances >= 0).all() and (distances <= 2).all()
