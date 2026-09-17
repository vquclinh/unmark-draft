"""ViUnMark analyses under descriptive names. Torch-free, synthetic inputs only."""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.viunmark.config import (  # noqa: E402
    CalibrationConfig,
    LogitStream,
    Pathway,
    ReadoutKind,
    ViUnMarkContractError,
)
from unmark.viunmark.diagnostics import (  # noqa: E402
    COMPUTABLE_FACTORS,
    FACTORIZATION_ORDER,
    UNRESOLVED_FACTORS,
    MatchedPoolingComparisonPlan,
    RobustnessFactor,
    cosine_distances,
    final_system_factors,
    matched_recipe_average,
    matched_recipe_streams,
    prediction_agreement,
)
from unmark.viunmark.diagnostics import decision_geometry, pooling_bridge  # noqa: E402
from unmark.viunmark.fusion import (  # noqa: E402
    CalibratedLogits,
    FusedLogits,
    HeadLogits,
    ensemble_branch,
    fuse_adapted_only_raw,
    fuse_viunmark_raw,
)

SEEDS = (53148, 59945, 42941, 720, 9428)
ROWS = ("a", "b")


def branch(stream, base, seeds=SEEDS):
    heads = [HeadLogits(stream, s, ROWS, [[base + i, 0.0, -base], [0.0, base, float(i)]])
             for i, s in enumerate(seeds)]
    return ensemble_branch(heads, expected_seeds=seeds)


# ---------------------------------------------------------------------------
# Native-Adapted Decision Geometry
# ---------------------------------------------------------------------------
def test_cosine_distances_per_example():
    native = [[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
    adapted = [[1.0, 0.0], [0.0, 1.0], [0.0, -3.0]]
    distances = cosine_distances(native, adapted)
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(1.0 - 1.0 / math.sqrt(2.0))
    assert distances[2] == pytest.approx(2.0)


def test_cosine_distance_refuses_zero_norm_and_shape_mismatch():
    with pytest.raises(ViUnMarkContractError, match="zero-norm"):
        cosine_distances([[0.0, 0.0]], [[1.0, 0.0]])
    with pytest.raises(ViUnMarkContractError):
        cosine_distances([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


def test_prediction_agreement():
    native = [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    adapted = [[3.0, 0.0, 0.0], [5.0, 1.0, 1.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]
    assert prediction_agreement(native, adapted) == pytest.approx(0.75)


def test_centered_logit_cosine_is_deliberately_not_implemented():
    assert decision_geometry.CENTERED_LOGIT_COSINE_IMPLEMENTED is False
    assert not hasattr(decision_geometry, "centered_logit_cosine")


# ---------------------------------------------------------------------------
# Checkpoint-Free Pooling Bridge
# ---------------------------------------------------------------------------
def test_pooling_bridge_trains_no_head_and_implements_no_guessed_decoder():
    assert pooling_bridge.HEAD_TRAINED is False
    assert pooling_bridge.DECODERS_IMPLEMENTED is False
    source = pathlib.Path(pooling_bridge.__file__).read_text(encoding="utf-8")
    assert "first_token_and_masked_mean(hidden, attention_mask, special_tokens_mask)" in source


# ---------------------------------------------------------------------------
# Matched-Head Pooling Comparison
# ---------------------------------------------------------------------------
def plan(**overrides):
    fields = dict(
        pathway=Pathway.VIUNMARK_GATE,
        seeds_by_readout={ReadoutKind.FIRST_TOKEN: SEEDS, ReadoutKind.MASKED_MEAN: SEEDS},
        schedule_by_readout={ReadoutKind.FIRST_TOKEN: {"epochs": 30},
                             ReadoutKind.MASKED_MEAN: {"epochs": 30}},
        initial_state_digest_by_readout={
            ReadoutKind.FIRST_TOKEN: {s: f"init-{s}" for s in SEEDS},
            ReadoutKind.MASKED_MEAN: {s: f"init-{s}" for s in SEEDS},
        },
    )
    fields.update(overrides)
    return MatchedPoolingComparisonPlan(**fields)


def test_a_matched_plan_is_accepted():
    assert plan().paired_seeds == SEEDS


def test_plan_refuses_unpaired_seeds():
    with pytest.raises(ViUnMarkContractError, match="same seeds"):
        plan(seeds_by_readout={ReadoutKind.FIRST_TOKEN: SEEDS,
                               ReadoutKind.MASKED_MEAN: tuple(reversed(SEEDS))})


def test_plan_refuses_different_schedules():
    with pytest.raises(ViUnMarkContractError, match="schedule"):
        plan(schedule_by_readout={ReadoutKind.FIRST_TOKEN: {"epochs": 30},
                                  ReadoutKind.MASKED_MEAN: {"epochs": 20}})


def test_plan_refuses_different_initial_parameters():
    digests = {s: f"init-{s}" for s in SEEDS}
    other = {**digests, SEEDS[2]: "different"}
    with pytest.raises(ViUnMarkContractError, match="different parameters"):
        plan(initial_state_digest_by_readout={ReadoutKind.FIRST_TOKEN: digests,
                                              ReadoutKind.MASKED_MEAN: other})


def test_plan_refuses_best_seed_selection_and_cross_head_selection():
    with pytest.raises(ViUnMarkContractError, match="best seed"):
        plan(best_seed_selection=True)
    with pytest.raises(ViUnMarkContractError, match="within each head"):
        plan(checkpoint_selection_scope="across_heads")


def test_plan_compares_first_token_with_masked_mean_only():
    with pytest.raises(ViUnMarkContractError):
        plan(seeds_by_readout={ReadoutKind.FIRST_TOKEN: SEEDS, ReadoutKind.CONCAT: SEEDS})


# ---------------------------------------------------------------------------
# Matched-Recipe Complementarity Analysis
# ---------------------------------------------------------------------------
def test_matched_recipe_streams_are_derived_from_the_recipes():
    assert set(matched_recipe_streams()) == {
        LogitStream.GATE_MATCHED_RECIPE_READOUT,
        LogitStream.SCALE_UNWEIGHTED_READOUT,
        LogitStream.PHOBERT_READOUT,
    }


def test_matched_recipe_average_is_equal_weight():
    gate = branch(LogitStream.GATE_MATCHED_RECIPE_READOUT, 2.0)
    scale = branch(LogitStream.SCALE_UNWEIGHTED_READOUT, 4.0)
    result = matched_recipe_average([gate, scale])
    expected = tuple(
        tuple(0.5 * g + 0.5 * s for g, s in zip(gr, sr))
        for gr, sr in zip(gate.values, scale.values)
    )
    assert result.values == expected
    assert result.head_seeds == SEEDS


def test_matched_recipe_average_refuses_off_recipe_and_duplicate_pathways():
    with pytest.raises(ViUnMarkContractError, match="matched recipe"):
        matched_recipe_average([branch(LogitStream.GATE_ROBUST_READOUT, 1.0),
                                branch(LogitStream.SCALE_UNWEIGHTED_READOUT, 1.0)])
    with pytest.raises(ViUnMarkContractError):
        matched_recipe_average([branch(LogitStream.SCALE_UNWEIGHTED_READOUT, 1.0)])


def test_matched_recipe_uses_five_heads_per_branch():
    four = SEEDS[:4]
    with pytest.raises(ViUnMarkContractError, match="heads"):
        matched_recipe_average([branch(LogitStream.GATE_MATCHED_RECIPE_READOUT, 1.0, four),
                                branch(LogitStream.PHOBERT_READOUT, 1.0, four)])


# ---------------------------------------------------------------------------
# Robustness Gain Factorization
# ---------------------------------------------------------------------------
def test_factorization_order_and_unresolved_steps():
    assert [f.value for f in FACTORIZATION_ORDER] == [
        "common_matched_pair", "gate_specialization", "scale_dual_readout",
        "adapted_only_raw", "adapted_only_calibrated", "add_native_phobert",
        "viunmark_calibrated",
    ]
    assert UNRESOLVED_FACTORS == FACTORIZATION_ORDER[:3]
    assert COMPUTABLE_FACTORS == FACTORIZATION_ORDER[3:]


def test_final_system_factors_follow_the_fusion_graph_without_stacking():
    gate = branch(LogitStream.GATE_ROBUST_READOUT, 1.0)
    scale_u = branch(LogitStream.SCALE_UNWEIGHTED_READOUT, 2.0)
    scale_w = branch(LogitStream.SCALE_WEIGHTED_READOUT, 4.0)
    phobert = branch(LogitStream.PHOBERT_READOUT, 8.0)
    adapted_cal = CalibrationConfig(class_index=1, additive_logit_bias=0.75, label_name="neutral")
    final_cal = CalibrationConfig(class_index=1, additive_logit_bias=1.25, label_name="neutral")
    factors = final_system_factors(gate, scale_u, scale_w, phobert,
                                   adapted_only_calibration=adapted_cal,
                                   viunmark_calibration=final_cal)
    assert set(factors) == set(COMPUTABLE_FACTORS)

    adapted_raw = fuse_adapted_only_raw(gate, scale_u, scale_w)
    viunmark_raw = fuse_viunmark_raw(gate, scale_u, scale_w, phobert)
    assert factors[RobustnessFactor.ADAPTED_ONLY_RAW].values == adapted_raw.values
    assert factors[RobustnessFactor.ADD_NATIVE_PHOBERT].values == viunmark_raw.values
    assert type(factors[RobustnessFactor.ADD_NATIVE_PHOBERT]) is FusedLogits

    calibrated = factors[RobustnessFactor.VIUNMARK_CALIBRATED]
    assert isinstance(calibrated, CalibratedLogits)
    assert calibrated.values == tuple((r[0], r[1] + 1.25, r[2]) for r in viunmark_raw.values)
    adapted_calibrated = factors[RobustnessFactor.ADAPTED_ONLY_CALIBRATED]
    assert adapted_calibrated.values == tuple(
        (r[0], r[1] + 0.75, r[2]) for r in adapted_raw.values)
