"""ViUnMark fusion: within-branch head means, hierarchical raw fusion, one calibration.

All logits are synthetic and chosen so every intermediate value is an exact
binary fraction, which lets the equations be checked with `==` rather than a
tolerance. Torch-free.

The final system is ONE 20-head ensemble: each branch averages the logits of its
five heads, the four RAW branch means are fused, and the system calibration is
applied once at the end.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.viunmark.config import (  # noqa: E402
    FINAL_BRANCH_STREAMS,
    AdaptedOnlyFusionConfig,
    CalibrationConfig,
    LogitStream,
    ViUnMarkConfig,
    ViUnMarkContractError,
)
from unmark.viunmark.fusion import (  # noqa: E402
    BEST_SEED_SELECTION,
    PARENT_BIASES_STACKED,
    WITHIN_BRANCH_LOGIT_ENSEMBLE,
    BranchLogits,
    CalibratedLogits,
    FusedLogits,
    FusionNode,
    HeadLogits,
    SystemKind,
    adapted_only_fusion,
    apply_system_calibration,
    argmax_rows,
    ensemble_branch,
    expanded_adapted_only_weights,
    expanded_per_head_weights,
    expanded_viunmark_weights,
    fuse_adapted_only_raw,
    fuse_scale,
    fuse_viunmark_raw,
    viunmark_fusion,
)
from unmark.viunmark.system import AdaptedOnlyFusion, ViUnMark  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SEEDS = (53148, 59945, 42941, 720, 9428)
ROWS = ("r0", "r1", "r2")
GATE = LogitStream.GATE_ROBUST_READOUT
SCALE_U = LogitStream.SCALE_UNWEIGHTED_READOUT
SCALE_W = LogitStream.SCALE_WEIGHTED_READOUT
PHOBERT = LogitStream.PHOBERT_READOUT
UIT_FINAL = CalibrationConfig(class_index=1, additive_logit_bias=1.25, label_name="neutral")
UIT_ADAPTED = CalibrationConfig(class_index=1, additive_logit_bias=0.75, label_name="neutral")

# Per-branch offsets make every branch distinguishable in the fused result.
OFFSET = {GATE: 1.0, SCALE_U: 2.0, SCALE_W: 4.0, PHOBERT: 8.0}


def head_values(stream: LogitStream, index: int) -> list[list[float]]:
    base = OFFSET[stream]
    return [
        [base * (index + 1), -base, 0.5 * index],
        [0.25 * index, base, -(index + 1.0)],
        [base, 0.125 * index, base * index],
    ]


def heads(stream: LogitStream, seeds=SEEDS) -> list[HeadLogits]:
    return [HeadLogits(stream, seed, ROWS, head_values(stream, i)) for i, seed in enumerate(seeds)]


def mean_of(matrices):
    n = len(matrices)
    return tuple(
        tuple(sum(m[r][c] for m in matrices) / n for c in range(3)) for r in range(3)
    )


def branch(stream: LogitStream) -> BranchLogits:
    return ensemble_branch(heads(stream), expected_seeds=SEEDS)


def all_branches():
    return {stream: branch(stream) for stream in FINAL_BRANCH_STREAMS}


def add(*terms):
    return tuple(
        tuple(sum(w * m[r][c] for w, m in terms) for c in range(3)) for r in range(3)
    )


# ---------------------------------------------------------------------------
# 1-4. Five heads per branch, arithmetic mean, no selection
# ---------------------------------------------------------------------------
def test_declared_ensemble_semantics():
    assert WITHIN_BRANCH_LOGIT_ENSEMBLE == "MEAN"
    assert BEST_SEED_SELECTION is False
    assert PARENT_BIASES_STACKED is False


@pytest.mark.parametrize("stream", FINAL_BRANCH_STREAMS)
def test_branch_raw_logit_is_the_arithmetic_mean_of_its_five_heads(stream):
    result = branch(stream)
    expected = mean_of([head_values(stream, i) for i in range(5)])
    assert result.values == expected
    assert result.stream is stream
    assert result.head_seeds == SEEDS
    assert len(result.head_seeds) == 5


def test_head_order_does_not_change_the_branch_mean():
    shuffled = list(reversed(heads(GATE)))
    assert ensemble_branch(shuffled, expected_seeds=SEEDS).values == branch(GATE).values


def test_a_missing_head_is_refused_so_no_seed_is_ever_selected():
    for dropped in range(5):
        subset = [h for i, h in enumerate(heads(GATE)) if i != dropped]
        with pytest.raises(ViUnMarkContractError, match="missing"):
            ensemble_branch(subset, expected_seeds=SEEDS)


def test_a_single_best_head_cannot_stand_in_for_the_branch():
    with pytest.raises(ViUnMarkContractError):
        ensemble_branch(heads(GATE)[:1], expected_seeds=SEEDS)
    with pytest.raises(ViUnMarkContractError):
        fuse_scale(heads(SCALE_U)[0], branch(SCALE_W))  # type: ignore[arg-type]


def test_extra_duplicated_and_foreign_heads_are_refused():
    with pytest.raises(ViUnMarkContractError, match="unexpected"):
        ensemble_branch(heads(GATE, SEEDS + (1,)), expected_seeds=SEEDS)
    with pytest.raises(ViUnMarkContractError, match="duplicated"):
        ensemble_branch(heads(GATE) + heads(GATE)[:1], expected_seeds=SEEDS)
    mixed = heads(GATE)[:4] + [HeadLogits(SCALE_U, SEEDS[4], ROWS, head_values(SCALE_U, 4))]
    with pytest.raises(ViUnMarkContractError, match="several branches"):
        ensemble_branch(mixed, expected_seeds=SEEDS)


def test_heads_on_different_rows_are_refused():
    moved = heads(GATE)
    moved[2] = HeadLogits(GATE, SEEDS[2], ("r0", "r2", "r1"), head_values(GATE, 2))
    with pytest.raises(ViUnMarkContractError, match="rows"):
        ensemble_branch(moved, expected_seeds=SEEDS)


WITHIN_HEAD_BOUNDARY_SELECTION = "select_checkpoint_boundary"
"""The one permitted `select` name: checkpoint selection over boundaries INSIDE one head."""


def test_no_function_selects_a_best_seed_or_head():
    for path in (REPO / "unmark/viunmark").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lowered = node.name.lower()
                assert "best" not in lowered, (path, node.name)
                if "select" in lowered:
                    assert node.name == WITHIN_HEAD_BOUNDARY_SELECTION, (path, node.name)


def test_the_only_selection_function_ranks_boundaries_not_seeds_or_heads():
    """Structural: it takes BoundaryScore values, and a BoundaryScore has no seed or head."""
    import dataclasses

    from unmark.viunmark.training import BoundaryScore

    source = (REPO / "unmark/viunmark/training.py").read_text(encoding="utf-8")
    function = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == WITHIN_HEAD_BOUNDARY_SELECTION
    )
    assert ast.unparse(function.args.args[0].annotation) == "Sequence[BoundaryScore]"
    assert {f.name for f in dataclasses.fields(BoundaryScore)} == {
        "boundary", "macro_f1_by_condition"}


# ---------------------------------------------------------------------------
# 5-7. Hierarchical raw fusion
# ---------------------------------------------------------------------------
def test_scale_raw_is_the_equal_mean_of_the_two_scale_branch_means():
    result = fuse_scale(branch(SCALE_U), branch(SCALE_W))
    assert result.node is FusionNode.SCALE_RAW
    assert result.values == add((0.5, branch(SCALE_U).values), (0.5, branch(SCALE_W).values))


def test_adapted_raw_is_half_gate_mean_plus_half_scale_raw():
    b = all_branches()
    result = fuse_adapted_only_raw(b[GATE], b[SCALE_U], b[SCALE_W])
    scale = add((0.5, b[SCALE_U].values), (0.5, b[SCALE_W].values))
    assert result.node is FusionNode.ADAPTED_ONLY_RAW
    assert result.values == add((0.5, b[GATE].values), (0.5, scale))


def test_viunmark_raw_is_half_adapted_raw_plus_half_phobert_mean():
    b = all_branches()
    result = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT])
    adapted = fuse_adapted_only_raw(b[GATE], b[SCALE_U], b[SCALE_W]).values
    assert result.node is FusionNode.VIUNMARK_RAW
    assert result.values == add((0.5, adapted), (0.5, b[PHOBERT].values))


def test_raw_fusion_equals_the_expanded_branch_weights():
    b = all_branches()
    weights = expanded_viunmark_weights()
    result = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT])
    assert result.values == add(*((weights[s], b[s].values) for s in FINAL_BRANCH_STREAMS))


def test_raw_fusion_equals_a_weighted_sum_over_all_twenty_heads():
    per_head = expanded_per_head_weights(expanded_viunmark_weights(), 5)
    b = all_branches()
    result = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT]).values
    terms = [(per_head[s], head_values(s, i)) for s in FINAL_BRANCH_STREAMS for i in range(5)]
    assert len(terms) == 20
    flat = [cell for row in result for cell in row]
    expected = [cell for row in add(*terms) for cell in row]
    assert flat == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 9. Weights
# ---------------------------------------------------------------------------
def test_expanded_branch_weights_are_exact_and_sum_to_one():
    assert expanded_adapted_only_weights() == {GATE: 0.5, SCALE_U: 0.25, SCALE_W: 0.25}
    assert expanded_viunmark_weights() == {
        PHOBERT: 0.5, GATE: 0.25, SCALE_U: 0.125, SCALE_W: 0.125,
    }
    assert sum(expanded_adapted_only_weights().values()) == 1.0
    assert sum(expanded_viunmark_weights().values()) == 1.0


def test_expanded_per_head_weights_are_exact_and_sum_to_one_over_twenty_heads():
    per_head = expanded_per_head_weights(expanded_viunmark_weights(), 5)
    assert per_head == pytest.approx({PHOBERT: 0.1, GATE: 0.05, SCALE_U: 0.025, SCALE_W: 0.025},
                                     abs=1e-15)
    assert sum(w * 5 for w in per_head.values()) == pytest.approx(1.0, abs=1e-15)
    adapted = expanded_per_head_weights(expanded_adapted_only_weights(), 5)
    assert adapted == pytest.approx({GATE: 0.1, SCALE_U: 0.05, SCALE_W: 0.05}, abs=1e-15)


# ---------------------------------------------------------------------------
# 8, 10-11. Calibration after all raw fusion, one class only
# ---------------------------------------------------------------------------
def test_final_calibration_adds_one_point_two_five_to_class_one_only_after_fusion():
    b = all_branches()
    raw = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT]).values
    final = viunmark_fusion(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT], UIT_FINAL)
    assert isinstance(final, CalibratedLogits)
    assert final.system is SystemKind.VIUNMARK
    assert final.values == tuple((r[0], r[1] + 1.25, r[2]) for r in raw)
    assert final.calibration == UIT_FINAL


def test_adapted_only_calibration_adds_zero_point_seven_five_to_class_one_only():
    b = all_branches()
    raw = fuse_adapted_only_raw(b[GATE], b[SCALE_U], b[SCALE_W]).values
    final = adapted_only_fusion(b[GATE], b[SCALE_U], b[SCALE_W], UIT_ADAPTED)
    assert final.system is SystemKind.ADAPTED_ONLY_FUSION
    assert final.values == tuple((r[0], r[1] + 0.75, r[2]) for r in raw)


def test_generic_system_output_is_uncalibrated():
    b = all_branches()
    raw = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT]).values
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS)
    output = ViUnMark(config).predict_logits({s: heads(s) for s in FINAL_BRANCH_STREAMS})
    assert output.values == raw
    assert output.calibration.is_identity


# ---------------------------------------------------------------------------
# 9 (correction). Parent biases are never stacked
# ---------------------------------------------------------------------------
def test_calibrated_adapted_only_output_cannot_enter_final_fusion():
    b = all_branches()
    calibrated = adapted_only_fusion(b[GATE], b[SCALE_U], b[SCALE_W], UIT_ADAPTED)
    for position in range(4):
        args = [b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT]]
        args[position] = calibrated
        with pytest.raises(ViUnMarkContractError):
            fuse_viunmark_raw(*args)


def test_no_public_path_produces_a_calibrated_single_branch():
    """The standalone PhoBERT calibration has no public construction at all."""
    with pytest.raises(ViUnMarkContractError):
        apply_system_calibration(branch(PHOBERT), UIT_ADAPTED)  # type: ignore[arg-type]
    with pytest.raises(ViUnMarkContractError):
        apply_system_calibration(heads(PHOBERT)[0], UIT_ADAPTED)  # type: ignore[arg-type]


def test_internal_nodes_and_calibrated_outputs_are_never_recalibrated():
    b = all_branches()
    with pytest.raises(ViUnMarkContractError, match="internal"):
        apply_system_calibration(fuse_scale(b[SCALE_U], b[SCALE_W]), UIT_FINAL)
    once = viunmark_fusion(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT], UIT_FINAL)
    with pytest.raises(ViUnMarkContractError):
        apply_system_calibration(once, UIT_FINAL)  # type: ignore[arg-type]


def test_stacked_parent_biases_would_give_a_different_answer():
    """The invariant is not vacuous: stacking parent biases changes the logits."""
    b = all_branches()
    correct = viunmark_fusion(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT], UIT_FINAL).values
    adapted_raw = fuse_adapted_only_raw(b[GATE], b[SCALE_U], b[SCALE_W]).values
    adapted_biased = tuple((r[0], r[1] + 0.75, r[2]) for r in adapted_raw)
    phobert_biased = tuple((r[0], r[1] + 0.75, r[2]) for r in b[PHOBERT].values)
    stacked = add((0.5, adapted_biased), (0.5, phobert_biased))
    stacked = tuple((r[0], r[1] + 1.25, r[2]) for r in stacked)
    assert stacked != correct
    assert all(s[1] - c[1] == pytest.approx(0.75) for s, c in zip(stacked, correct))


def test_containers_cannot_be_laundered_into_new_raw_logits():
    once = viunmark_fusion(*all_branches().values(), UIT_FINAL)
    with pytest.raises(ViUnMarkContractError, match="container"):
        BranchLogits(PHOBERT, SEEDS, ROWS, once)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Head-set and row alignment across branches
# ---------------------------------------------------------------------------
def test_branches_averaged_over_different_head_sets_are_refused():
    other = (1, 2, 3, 4, 5)
    odd = ensemble_branch(heads(SCALE_W, other), expected_seeds=other)
    with pytest.raises(ViUnMarkContractError, match="head seed sets"):
        fuse_scale(branch(SCALE_U), odd)


def test_branches_on_different_rows_are_refused():
    moved = [HeadLogits(SCALE_W, s, ("r0", "r2", "r1"), head_values(SCALE_W, i))
             for i, s in enumerate(SEEDS)]
    with pytest.raises(ViUnMarkContractError, match="rows"):
        fuse_scale(branch(SCALE_U), ensemble_branch(moved, expected_seeds=SEEDS))


def test_branches_in_the_wrong_position_are_refused():
    b = all_branches()
    with pytest.raises(ViUnMarkContractError):
        fuse_viunmark_raw(b[PHOBERT], b[SCALE_U], b[SCALE_W], b[GATE])
    with pytest.raises(ViUnMarkContractError):
        fuse_scale(b[SCALE_W], b[SCALE_U])


# ---------------------------------------------------------------------------
# System objects
# ---------------------------------------------------------------------------
def test_viunmark_system_is_one_twenty_head_ensemble():
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS, calibration=UIT_FINAL)
    supplied = {s: heads(s) for s in FINAL_BRANCH_STREAMS}
    assert sum(len(v) for v in supplied.values()) == 20
    output = ViUnMark(config).predict_logits(supplied)
    b = all_branches()
    assert output.values == viunmark_fusion(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT],
                                            UIT_FINAL).values
    assert output.head_seeds == SEEDS
    assert isinstance(output, CalibratedLogits)


def test_viunmark_system_refuses_a_missing_or_extra_branch():
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS)
    partial = {s: heads(s) for s in FINAL_BRANCH_STREAMS[:3]}
    with pytest.raises(ViUnMarkContractError):
        ViUnMark(config).predict_logits(partial)


def test_viunmark_system_refuses_heads_filed_under_the_wrong_branch():
    config = ViUnMarkConfig(num_labels=3, head_seeds=SEEDS)
    supplied = {s: heads(s) for s in FINAL_BRANCH_STREAMS}
    supplied[GATE], supplied[PHOBERT] = supplied[PHOBERT], supplied[GATE]
    with pytest.raises(ViUnMarkContractError, match="belong to"):
        ViUnMark(config).predict_logits(supplied)


def test_adapted_only_system_uses_three_branches():
    config = AdaptedOnlyFusionConfig(num_labels=3, head_seeds=SEEDS, calibration=UIT_ADAPTED)
    b = all_branches()
    output = AdaptedOnlyFusion(config).predict_logits({s: heads(s) for s in (GATE, SCALE_U, SCALE_W)})
    assert output.values == adapted_only_fusion(b[GATE], b[SCALE_U], b[SCALE_W], UIT_ADAPTED).values


def test_predictions_take_the_first_maximum_on_ties():
    assert argmax_rows(((1.0, 3.0, 3.0), (2.0, 2.0, 1.0), (0.0, -1.0, 5.0))) == (1, 0, 2)


def test_fused_logits_type_is_distinct_from_calibrated():
    b = all_branches()
    raw = fuse_viunmark_raw(b[GATE], b[SCALE_U], b[SCALE_W], b[PHOBERT])
    assert type(raw) is FusedLogits and not isinstance(raw, CalibratedLogits)
