"""Tensor-level ViUnMark checks: readouts, robust MLP head, adapter handles.

Needs **real torch**. Follows the repository convention of a per-test `skipif`
rather than a module-level `importorskip`, so skipped tests stay visible in the
report. No model is downloaded and no dataset is read; every tensor is synthetic.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.viunmark.config import (  # noqa: E402
    ReadoutKind,
    RobustMLPHeadConfig,
    ViUnMarkContractError,
)

try:  # pragma: no cover - depends on the environment
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - the normal ML-free path
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(
    not TORCH, reason="torch is not installed locally; tensor checks run where available"
)


def synthetic_batch():
    """Two rows, length 5, hidden 4. Row 0: <s> a b </s> <pad>; row 1: <s> a b c </s>."""
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    special_tokens_mask = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 0, 0, 1]])
    return hidden, attention_mask, special_tokens_mask


@requires_torch
def test_first_token_is_position_zero():
    from unmark.viunmark.readout import first_token

    hidden, _, _ = synthetic_batch()
    assert torch.equal(first_token(hidden), hidden[:, 0, :])


@requires_torch
def test_masked_mean_excludes_padding_and_tokenizer_special_tokens():
    from unmark.viunmark.readout import masked_mean

    hidden, attention_mask, special = synthetic_batch()
    result = masked_mean(hidden, attention_mask, special)
    expected = torch.stack([hidden[0, 1:3].mean(dim=0), hidden[1, 1:4].mean(dim=0)])
    assert torch.allclose(result, expected)


@requires_torch
def test_concat_is_first_token_then_masked_mean_from_one_tensor():
    from unmark.viunmark.readout import concat_first_token_masked_mean, first_token, masked_mean

    hidden, attention_mask, special = synthetic_batch()
    result = concat_first_token_masked_mean(hidden, attention_mask, special)
    ft, mm = first_token(hidden), masked_mean(hidden, attention_mask, special)
    assert result.shape == (2, 8)
    assert torch.equal(result, torch.cat([ft, mm], dim=-1))
    assert torch.equal(result[:, :4], ft)
    assert torch.allclose(result[:, 4:], mm)
    assert not torch.allclose(result, torch.cat([mm, ft], dim=-1))


@requires_torch
@pytest.mark.parametrize("kind, width", [
    (ReadoutKind.FIRST_TOKEN, 768), (ReadoutKind.MASKED_MEAN, 768), (ReadoutKind.CONCAT, 1536),
])
def test_readout_widths_on_phobert_base_hidden_states(kind, width):
    from unmark.viunmark.readout import readout

    hidden = torch.randn(3, 6, 768)
    attention_mask = torch.ones(3, 6, dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 0, 0, 1]] * 3)
    assert readout(kind, hidden, attention_mask, special).shape == (3, width)
    assert kind.feature_dim(768) == width


@requires_torch
def test_readout_refuses_non_sequence_hidden_states():
    from unmark.viunmark.readout import first_token

    with pytest.raises(ViUnMarkContractError):
        first_token(torch.zeros(2, 768))


@requires_torch
@pytest.mark.parametrize("input_dim, expected", [(768, 199171), (1536, 397315)])
def test_robust_mlp_head_structure_and_parameter_count(input_dim, expected):
    from torch import nn

    from unmark.viunmark.heads import build_robust_mlp_head

    head = build_robust_mlp_head(RobustMLPHeadConfig(input_dim=input_dim, num_labels=3))
    kinds = [type(layer) for layer in head]
    assert kinds == [nn.LayerNorm, nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    assert head[0].normalized_shape == (input_dim,)
    assert (head[1].in_features, head[1].out_features) == (input_dim, 256)
    assert head[3].p == 0.1
    assert (head[4].in_features, head[4].out_features) == (256, 3)
    assert sum(p.numel() for p in head.parameters()) == expected
    assert head(torch.randn(4, input_dim)).shape == (4, 3)


@requires_torch
def test_adapter_handles_build_the_recorded_fusion():
    from unmark.viunmark.system import ViUnMarkGate, ViUnMarkScale

    gate = ViUnMarkGate.build_adapter(51800)
    scale = ViUnMarkScale.build_adapter(51800)
    assert gate.config.fusion_id == "historical-fusion-v1"
    assert scale.config.fusion_id == "scale-calibrated-fusion-v1"
    assert scale.config.is_scale_calibrated and not gate.config.is_scale_calibrated
    assert sum(p.numel() for p in gate.parameters()) == 3551232


@requires_torch
def test_adapter_handles_load_their_own_fusion_strictly():
    from unmark.viunmark.system import ViUnMarkGate, ViUnMarkScale

    source = ViUnMarkScale.build_adapter(7)
    payload = {
        "provenance": {"fusion": {"fusion_id": "scale-calibrated-fusion-v1"}},
        "adapter_state": source.state_dict(),
    }
    loaded = ViUnMarkScale.load_adapter(payload)
    assert loaded.config.fusion_id == "scale-calibrated-fusion-v1"
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    with pytest.raises(ViUnMarkContractError):
        ViUnMarkGate.load_adapter(payload)


# ---------------------------------------------------------------------------
# Evidence amendment: recovered initialisation, optimizer, historical readout sources
# ---------------------------------------------------------------------------
@requires_torch
def test_initialisation_policy_on_a_built_head():
    from torch import nn

    from unmark.viunmark.heads import build_robust_mlp_head, initialize_robust_mlp_head

    head = initialize_robust_mlp_head(
        build_robust_mlp_head(RobustMLPHeadConfig(input_dim=1536, num_labels=3)))
    layer_norm, first, last = head[0], head[1], head[4]
    assert isinstance(layer_norm, nn.LayerNorm)
    assert torch.equal(layer_norm.weight, torch.ones(1536))
    assert torch.equal(layer_norm.bias, torch.zeros(1536))
    for linear in (first, last):
        assert torch.equal(linear.bias, torch.zeros_like(linear.bias))
        fan_out, fan_in = linear.weight.shape
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert float(linear.weight.abs().max()) <= bound


@requires_torch
def test_optimizer_policy_on_a_built_head():
    from unmark.viunmark.heads import build_robust_mlp_head, build_robust_mlp_optimizer

    head = build_robust_mlp_head(RobustMLPHeadConfig(input_dim=768, num_labels=3))
    optimizer = build_robust_mlp_optimizer(head)
    assert type(optimizer).__name__ == "AdamW"
    decayed, undecayed = optimizer.param_groups
    assert decayed["weight_decay"] == 0.01 and undecayed["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() == 1 for p in undecayed["params"])
    assert len(decayed["params"]) == 2      # two Linear weight matrices
    assert len(undecayed["params"]) == 4    # LayerNorm weight/bias and two Linear biases
    for group in optimizer.param_groups:
        assert group["lr"] == 0.01 and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8


@requires_torch
def test_public_readouts_equal_the_historical_bank_sources():
    """FIRST_TOKEN matches the Stage-2 extractor; MASKED_MEAN is the Stage-1 pooling."""
    from unmark.evaluation.stage2_dual_finalist import stage2_first_token_representation
    from unmark.modeling.pooling import masked_mean_non_special
    from unmark.viunmark.readout import first_token, masked_mean

    hidden = torch.randn(2, 6, 768)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    special = torch.tensor([[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
    assert torch.equal(first_token(hidden), stage2_first_token_representation(hidden))
    assert torch.equal(masked_mean(hidden, attention_mask, special),
                       masked_mean_non_special(hidden, attention_mask, special))
