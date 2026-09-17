"""ViUnMark method definitions: pathways, readouts, head, losses, calibration.

Everything here is torch-free. Tensor-level checks live in
`test_viunmark_torch.py`.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.corruption import CONDITIONS  # noqa: E402
from unmark.viunmark.config import (  # noqa: E402
    CONCAT_ORDER,
    FINAL_BRANCH_STREAMS,
    ROBUST_MLP_DROPOUT,
    ROBUST_MLP_HIDDEN_DIM,
    ROBUST_MLP_LAYERS,
    SIX_CONDITIONS,
    AdaptedOnlyFusionConfig,
    BranchReadoutConfig,
    CalibrationConfig,
    GateReadoutConfig,
    LogitStream,
    LossKind,
    Pathway,
    PhoBERTReadoutConfig,
    ReadoutKind,
    RobustMLPHeadConfig,
    ScaleUnweightedReadoutConfig,
    ScaleWeightedReadoutConfig,
    TrainingDistribution,
    ViUnMarkConfig,
    ViUnMarkContractError,
    ViUnMarkGateConfig,
    ViUnMarkScaleConfig,
)
from unmark.viunmark.heads import robust_mlp_layers  # noqa: E402
from unmark.viunmark.losses import (  # noqa: E402
    class_counts_from_labels,
    class_weights_for,
    sqrt_inverse_frequency_weights,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
UIT_VSFC_PROTOCOL_TRAIN_COUNTS = (4259, 366, 4514)
UIT_VSFC_WEIGHT_VECTOR = (0.5573523045, 1.9012669325, 0.5413808227)
SEEDS = (53148, 59945, 42941, 720, 9428)


# ---------------------------------------------------------------------------
# Stage-I pathways
# ---------------------------------------------------------------------------
def test_gate_and_scale_share_everything_but_the_fusion():
    gate, scale = ViUnMarkGateConfig(), ViUnMarkScaleConfig()
    assert gate.fusion_id == "historical-fusion-v1"
    assert scale.fusion_id == "scale-calibrated-fusion-v1"
    assert gate.fusion_equation == "z = g * f + (1 - g) * e"
    assert scale.fusion_equation == (
        "scale = ||e||_2 / max(||f||_2, 1e-8); z = g * (scale * f) + (1 - g) * e"
    )
    for name in ("encoder_checkpoint", "encoder_revision", "hidden_size", "tone_rows",
                 "letter_rows", "gate_weight_initial", "gate_initial_value", "objective_id",
                 "corpus_dataset", "corpus_revision", "precision", "encoder_frozen"):
        assert getattr(gate, name) == getattr(scale, name), name
    assert gate.encoder_checkpoint == "vinai/phobert-base"
    assert gate.encoder_revision == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert (gate.hidden_size, gate.tone_rows, gate.letter_rows) == (768, 7, 5)
    assert gate.gate_weight_initial == 0.0 and gate.gate_initial_value == 0.01
    assert gate.objective_id == "align-clean-pooled-v1"
    assert gate.corpus_dataset == "undertheseanlp/UVW-2026"
    assert gate.precision == "fp32" and gate.encoder_frozen is True
    assert scale.scale_epsilon == 1e-8
    assert gate.adapter_trainable_parameters == scale.adapter_trainable_parameters == 3551232
    assert gate.pathway is Pathway.VIUNMARK_GATE and scale.pathway is Pathway.VIUNMARK_SCALE


@pytest.mark.parametrize("field, value", [
    ("hidden_size", 1024), ("encoder_revision", "main"), ("tone_rows", 6),
    ("precision", "fp16"), ("encoder_frozen", False), ("objective_id", "other"),
])
def test_stage1_identity_cannot_be_reconfigured(field, value):
    with pytest.raises(ViUnMarkContractError):
        ViUnMarkGateConfig(**{field: value})


@pytest.mark.parametrize("handle_name, other_fusion", [
    ("ViUnMarkGate", "scale-calibrated-fusion-v1"),
    ("ViUnMarkScale", "historical-fusion-v1"),
])
def test_adapter_loader_refuses_the_other_pathways_checkpoint(handle_name, other_fusion):
    """Refused on the recorded fusion, before any tensor is touched (torch-free).

    The two adapters have identical tensor shapes, so this recorded-identity check
    is the only thing standing between a checkpoint and the wrong equation.
    """
    import unmark.viunmark.system as system

    payload = {"provenance": {"fusion": {"fusion_id": other_fusion}}, "adapter_state": {}}
    with pytest.raises(ViUnMarkContractError, match="not a"):
        getattr(system, handle_name).load_adapter(payload)


def test_a_checkpoint_recording_no_fusion_is_not_a_scale_adapter():
    from unmark.viunmark.system import ViUnMarkScale

    with pytest.raises(ViUnMarkContractError):
        ViUnMarkScale.load_adapter({"provenance": {}, "adapter_state": {}})


def test_gate_bias_initial_is_logit_of_one_percent():
    from unmark.modeling.contracts import GATE_INIT_BIAS

    assert GATE_INIT_BIAS == pytest.approx(math.log(0.01 / 0.99))


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
def test_readout_feature_dimensions():
    assert ReadoutKind.FIRST_TOKEN.feature_dim(768) == 768
    assert ReadoutKind.MASKED_MEAN.feature_dim(768) == 768
    assert ReadoutKind.CONCAT.feature_dim(768) == 1536


def test_concat_order_is_first_token_then_masked_mean():
    assert CONCAT_ORDER == (ReadoutKind.FIRST_TOKEN, ReadoutKind.MASKED_MEAN)


def test_concat_is_built_from_one_hidden_tensor_in_ft_mm_order():
    """AST: CONCAT reads both halves from one `hidden` and concatenates [first, mean]."""
    tree = ast.parse((REPO / "unmark/viunmark/readout.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    pair = functions["first_token_and_masked_mean"]
    calls = [n for n in ast.walk(pair) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert [c.func.id for c in calls] == ["first_token", "masked_mean"]
    assert all(isinstance(c.args[0], ast.Name) and c.args[0].id == "hidden" for c in calls)

    concat = functions["concat_first_token_masked_mean"]
    inner = [n for n in ast.walk(concat)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert [c.func.id for c in inner] == ["first_token_and_masked_mean"]
    cat = next(n for n in ast.walk(concat)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "cat")
    assert [e.id for e in cat.args[0].elts] == ["first", "mean"]
    unpack = next(n for n in ast.walk(concat) if isinstance(n, ast.Assign))
    assert [e.id for e in unpack.targets[0].elts] == ["first", "mean"]


def test_masked_mean_uses_the_tokenizer_special_tokens_mask():
    source = (REPO / "unmark/viunmark/readout.py").read_text(encoding="utf-8")
    assert "masked_mean_non_special(hidden, attention_mask, special_tokens_mask)" in source


# ---------------------------------------------------------------------------
# Robust MLP head
# ---------------------------------------------------------------------------
def test_robust_mlp_layer_order_and_shapes():
    assert ROBUST_MLP_LAYERS == ("LayerNorm", "Linear", "GELU", "Dropout", "Linear")
    assert ROBUST_MLP_HIDDEN_DIM == 256 and ROBUST_MLP_DROPOUT == 0.1
    assert robust_mlp_layers(RobustMLPHeadConfig(input_dim=768, num_labels=3)) == (
        ("LayerNorm", (768,)),
        ("Linear", (768, 256)),
        ("GELU", ()),
        ("Dropout", (0.1,)),
        ("Linear", (256, 3)),
    )


@pytest.mark.parametrize("input_dim, expected", [(768, 199171), (1536, 397315)])
def test_robust_mlp_parameter_counts(input_dim, expected):
    assert RobustMLPHeadConfig(input_dim=input_dim, num_labels=3).parameter_count == expected


# ---------------------------------------------------------------------------
# Branch recipes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_cls, stream, pathway, readout, loss, input_dim", [
    (GateReadoutConfig, LogitStream.GATE_ROBUST_READOUT, Pathway.VIUNMARK_GATE,
     ReadoutKind.MASKED_MEAN, LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY, 768),
    (ScaleUnweightedReadoutConfig, LogitStream.SCALE_UNWEIGHTED_READOUT, Pathway.VIUNMARK_SCALE,
     ReadoutKind.CONCAT, LossKind.UNWEIGHTED_CROSS_ENTROPY, 1536),
    (ScaleWeightedReadoutConfig, LogitStream.SCALE_WEIGHTED_READOUT, Pathway.VIUNMARK_SCALE,
     ReadoutKind.CONCAT, LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY, 1536),
    (PhoBERTReadoutConfig, LogitStream.PHOBERT_READOUT, Pathway.PHOBERT,
     ReadoutKind.CONCAT, LossKind.UNWEIGHTED_CROSS_ENTROPY, 1536),
])
def test_final_branch_recipes(config_cls, stream, pathway, readout, loss, input_dim):
    config = config_cls(num_labels=3)
    assert config.stream is stream and config.pathway is pathway
    assert config.readout is readout and config.loss is loss
    assert config.training_distribution is TrainingDistribution.AUGMENTED_SIX_CONDITIONS
    assert config.input_dim == input_dim
    assert config.head == RobustMLPHeadConfig(input_dim=input_dim, num_labels=3)


def test_branch_recipe_base_is_abstract():
    with pytest.raises(ViUnMarkContractError):
        BranchReadoutConfig(num_labels=3)


def test_six_conditions_are_the_authoritative_corruption_conditions():
    assert SIX_CONDITIONS == ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")
    assert tuple(CONDITIONS) == SIX_CONDITIONS
    assert TrainingDistribution.AUGMENTED_SIX_CONDITIONS.conditions == SIX_CONDITIONS


def test_final_system_has_four_branches_and_twenty_heads():
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS)
    assert config.streams == FINAL_BRANCH_STREAMS
    assert config.heads_per_branch == 5
    assert config.head_count == 20
    assert config.stage2_parameter_count == 6955580


def test_adapted_only_fusion_is_a_diagnostic_over_three_branches():
    config = AdaptedOnlyFusionConfig(num_labels=3, head_seeds=SEEDS)
    assert AdaptedOnlyFusionConfig.IS_DIAGNOSTIC is True
    assert ViUnMarkConfig.IS_DIAGNOSTIC is False
    assert LogitStream.PHOBERT_READOUT not in config.streams
    assert config.head_count == 15


@pytest.mark.parametrize("seeds", [(), (1, 1), [1, 2], (1, True)])
def test_head_seed_set_is_validated(seeds):
    with pytest.raises(ViUnMarkContractError):
        ViUnMarkConfig(num_labels=3, head_seeds=seeds)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_weight_rule_is_inverse_sqrt_frequency_normalised_to_mean_one():
    counts = (100, 25, 400)
    weights = sqrt_inverse_frequency_weights(counts)
    raw = [1 / math.sqrt(c) for c in counts]
    expected = [r / (sum(raw) / 3) for r in raw]
    assert weights == pytest.approx(expected, abs=1e-15)
    assert sum(weights) / 3 == pytest.approx(1.0, abs=1e-15)


def test_uit_vsfc_counts_reproduce_the_recorded_weight_vector():
    float32 = sqrt_inverse_frequency_weights(UIT_VSFC_PROTOCOL_TRAIN_COUNTS, arithmetic="float32")
    assert float32 == pytest.approx(UIT_VSFC_WEIGHT_VECTOR, abs=1e-9)
    float64 = sqrt_inverse_frequency_weights(UIT_VSFC_PROTOCOL_TRAIN_COUNTS)
    assert float64 == pytest.approx(UIT_VSFC_WEIGHT_VECTOR, abs=1e-7)


def test_weights_do_not_depend_on_a_uniform_count_scale():
    base = sqrt_inverse_frequency_weights(UIT_VSFC_PROTOCOL_TRAIN_COUNTS)
    scaled = sqrt_inverse_frequency_weights(tuple(6 * c for c in UIT_VSFC_PROTOCOL_TRAIN_COUNTS))
    assert scaled == pytest.approx(base, abs=1e-15)


def test_no_weight_vector_is_a_default():
    with pytest.raises(ViUnMarkContractError):
        class_weights_for(LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY, None)
    assert class_weights_for(LossKind.UNWEIGHTED_CROSS_ENTROPY, None) is None
    with pytest.raises(ViUnMarkContractError):
        class_weights_for(LossKind.UNWEIGHTED_CROSS_ENTROPY, UIT_VSFC_PROTOCOL_TRAIN_COUNTS)
    for path in (REPO / "unmark/viunmark").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number in ("4259", "4514", "0.5573", "1.9012"):
            assert number not in text, (path, number)


@pytest.mark.parametrize("counts", [(10, 0, 5), (10,), (10, -1), (10, 2.5)])
def test_weight_rule_refuses_undefined_counts(counts):
    with pytest.raises(ViUnMarkContractError):
        sqrt_inverse_frequency_weights(counts)


def test_class_counts_from_labels():
    assert class_counts_from_labels([0, 2, 2, 1, 2], 3) == (1, 1, 3)
    with pytest.raises(ViUnMarkContractError):
        class_counts_from_labels([0, 3], 3)


# ---------------------------------------------------------------------------
# Calibration is dataset-specific
# ---------------------------------------------------------------------------
def test_generic_configs_carry_no_calibration():
    for config in (ViUnMarkConfig(num_labels=3, head_seeds=SEEDS),
                   AdaptedOnlyFusionConfig(num_labels=3, head_seeds=SEEDS)):
        assert config.calibration == CalibrationConfig.identity()
        assert config.calibration.is_identity
        assert config.calibration.bias_vector(3) == (0.0, 0.0, 0.0)


def test_no_calibration_value_is_hard_coded_in_the_package():
    for path in (REPO / "unmark/viunmark").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "1.25" not in text and "0.75" not in text, path


def test_calibration_bias_vector_touches_one_class_only():
    calibration = CalibrationConfig(class_index=1, additive_logit_bias=1.25, label_name="neutral")
    assert calibration.bias_vector(3) == (0.0, 1.25, 0.0)
    with pytest.raises(ViUnMarkContractError):
        calibration.bias_vector(1)
    with pytest.raises(ViUnMarkContractError):
        CalibrationConfig(class_index=1, additive_logit_bias=1.25).bias_vector(2 - 1)


@pytest.mark.parametrize("kwargs", [
    {"class_index": None, "additive_logit_bias": 0.5},
    {"class_index": None, "label_name": "neutral"},
    {"class_index": -1, "additive_logit_bias": 1.0},
    {"class_index": 1, "additive_logit_bias": float("nan")},
    {"class_index": True, "additive_logit_bias": 1.0},
])
def test_malformed_calibration_is_refused(kwargs):
    with pytest.raises(ViUnMarkContractError):
        CalibrationConfig(**kwargs)


def test_configs_are_frozen():
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.calibration = CalibrationConfig(1, 1.25)  # type: ignore[misc]


def test_system_calibration_must_fit_the_label_space():
    with pytest.raises(ViUnMarkContractError):
        ViUnMarkConfig(num_labels=3, head_seeds=SEEDS,
                       calibration=CalibrationConfig(class_index=5, additive_logit_bias=1.0))
