"""Recovered robust MLP training recipe, corruption protocol and input identity.

Torch-free. Also checks the one training fact that is deliberately NOT recovered,
the cross-entropy reduction, against the repository evidence that justifies
leaving it open.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.corruption import CorruptionPurpose, corrupt  # noqa: E402
from unmark.orthography import canon  # noqa: E402
from unmark.viunmark.config import Pathway, ViUnMarkContractError  # noqa: E402
from unmark.viunmark.inputs import (  # noqa: E402
    ADAPTED_PATHWAYS,
    BASE_GRID_INVARIANCE_REQUIRED,
    CROSS_PATHWAY_INPUT_IDENTITY_REQUIRED,
    FULL_CONDITION_API_SEED,
    FULL_CONDITION_API_SEED_IS_SCIENTIFIC,
    CorruptionProtocol,
    CorruptionRealizationKey,
    require_cross_pathway_input_identity,
)
from unmark.viunmark.losses import sqrt_inverse_frequency_weights  # noqa: E402
from unmark.viunmark.training import (  # noqa: E402
    CROSS_ENTROPY_REDUCTION_RECOVERED,
    LAYERNORM_BIAS_INIT,
    LAYERNORM_WEIGHT_INIT,
    LINEAR_BIAS_INIT,
    LINEAR_WEIGHT_INIT,
    RECIPE_EVIDENCE_SCOPE,
    ROBUST_MLP_OPTIMIZER_POLICY,
    ROBUST_MLP_TRAINING_BUDGET,
    TRAINING_AMP,
    TRAINING_DTYPE,
    TRAINING_TF32,
    BoundaryScore,
    HeadTrainingBudget,
    augmented_labels,
    augmented_row_order,
    cycle_seed,
    dropout_seed,
    select_checkpoint_boundary,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CONDITIONS = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")
UIT_VECTOR = (0.5573523044586182, 1.9012669324874878, 0.5413808226585388)


# ---------------------------------------------------------------------------
# Initialisation, optimizer, precision
# ---------------------------------------------------------------------------
def test_initialisation_policy():
    assert (LAYERNORM_WEIGHT_INIT, LAYERNORM_BIAS_INIT) == (1.0, 0.0)
    assert (LINEAR_WEIGHT_INIT, LINEAR_BIAS_INIT) == ("xavier_uniform", 0.0)


def test_optimizer_policy():
    policy = ROBUST_MLP_OPTIMIZER_POLICY
    assert policy.name == "AdamW"
    assert policy.learning_rate == 0.01
    assert policy.betas == (0.9, 0.999)
    assert policy.eps == 1e-8
    assert policy.weight_decay_for(2) == 0.01
    assert policy.weight_decay_for(1) == 0.0


def test_precision_policy():
    assert (TRAINING_DTYPE, TRAINING_AMP, TRAINING_TF32) == ("float32", False, False)


def test_recipe_is_scoped_to_the_adapted_pathway_heads_only():
    assert RECIPE_EVIDENCE_SCOPE == (
        "gate_robust_readout", "scale_unweighted_readout", "scale_weighted_readout")
    assert "phobert_readout" not in RECIPE_EVIDENCE_SCOPE


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def test_budget_is_thirty_boundaries_of_seventy_two_updates():
    budget = ROBUST_MLP_TRAINING_BUDGET
    assert (budget.batch_size, budget.max_optimizer_updates, budget.selection_boundaries) == (
        128, 2160, 30)
    assert budget.updates_per_boundary == 72
    assert 2160 == 30 * 72
    assert budget.boundary_update(1) == 72
    assert budget.boundary_update(30) == 2160
    with pytest.raises(ViUnMarkContractError):
        budget.boundary_update(0)
    with pytest.raises(ViUnMarkContractError):
        budget.boundary_update(31)


def test_budget_refuses_uneven_boundaries():
    with pytest.raises(ViUnMarkContractError):
        HeadTrainingBudget(batch_size=128, max_optimizer_updates=2161, selection_boundaries=30)


# ---------------------------------------------------------------------------
# Augmented training set
# ---------------------------------------------------------------------------
def test_augmented_order_is_condition_major_in_the_exact_order():
    order = augmented_row_order(3)
    assert order == tuple((c, r) for c in CONDITIONS for r in range(3))
    assert [c for c, _ in order[::3]] == list(CONDITIONS)
    for index, condition in enumerate(CONDITIONS):
        block = order[index * 3:(index + 1) * 3]
        assert block == ((condition, 0), (condition, 1), (condition, 2))


def test_augmented_labels_repeat_once_per_condition_in_order():
    assert augmented_labels([2, 0, 1]) == (2, 0, 1) * 6
    assert len(augmented_row_order(7)) == len(augmented_labels(list(range(3)) * 2 + [0])) == 42


def test_no_row_count_or_seed_of_a_dataset_is_baked_into_the_package():
    for path in (REPO / "unmark/viunmark").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for literal in ("9139", "54834", "19225", "3500", "8000"):
            assert literal not in text, (path, literal)


# ---------------------------------------------------------------------------
# Deterministic streams
# ---------------------------------------------------------------------------
def test_cycle_seed_rule():
    assert cycle_seed(53148, 1) == 53148001
    assert cycle_seed(720, 6) == 720006
    with pytest.raises(ViUnMarkContractError):
        cycle_seed(53148, 0)


def test_dropout_seed_rule():
    assert dropout_seed(53148, 72) == 5314800072
    assert dropout_seed(9428, 2160) == 942802160


# ---------------------------------------------------------------------------
# Checkpoint selection inside one head
# ---------------------------------------------------------------------------
def score(boundary, **overrides):
    values = {c: 0.5 for c in CONDITIONS}
    values.update(overrides)
    return BoundaryScore(boundary, values)


def test_selection_primary_is_the_six_condition_mean():
    chosen = select_checkpoint_boundary([score(1), score(2, P50=0.8), score(3)])
    assert chosen.boundary == 2


def test_selection_tie_break_one_is_the_worst_condition():
    a = score(1, FULL=0.7, STRIP_ALL=0.3)       # mean 0.5, worst 0.3
    b = score(2, FULL=0.6, STRIP_ALL=0.4)       # mean 0.5, worst 0.4
    assert select_checkpoint_boundary([a, b]).boundary == 2


def test_selection_tie_break_two_is_full():
    a = score(1, FULL=0.6, P25=0.4)             # mean 0.5, worst 0.4, FULL 0.6
    b = score(2, FULL=0.4, P25=0.6)             # mean 0.5, worst 0.4, FULL 0.4
    assert select_checkpoint_boundary([b, a]).boundary == 1


def test_selection_tie_break_three_is_the_earlier_boundary():
    assert select_checkpoint_boundary([score(9), score(4), score(17)]).boundary == 4


def test_selection_refuses_duplicates_and_partial_condition_sets():
    with pytest.raises(ViUnMarkContractError):
        select_checkpoint_boundary([score(3), score(3)])
    with pytest.raises(ViUnMarkContractError):
        BoundaryScore(1, {"FULL": 0.5})


# ---------------------------------------------------------------------------
# Weighted CE: the rule, recomputed per dataset
# ---------------------------------------------------------------------------
def test_weighted_ce_uses_a_new_datasets_own_counts():
    external = sqrt_inverse_frequency_weights((1700, 1700, 1700))
    assert external == pytest.approx((1.0, 1.0, 1.0), abs=1e-15)
    assert external != pytest.approx(UIT_VECTOR, abs=1e-3)
    skewed = sqrt_inverse_frequency_weights((400, 100, 25))
    # raw = (1/20, 1/10, 1/5), mean = 7/60, so weights = (3/7, 6/7, 12/7)
    assert skewed == pytest.approx((3 / 7, 6 / 7, 12 / 7), abs=1e-12)


def test_uit_vsfc_vector_is_the_float32_rule_exactly():
    assert sqrt_inverse_frequency_weights((4259, 366, 4514), arithmetic="float32") == UIT_VECTOR


# ---------------------------------------------------------------------------
# Corruption protocol
# ---------------------------------------------------------------------------
def test_corruption_protocol_has_no_default_seed():
    with pytest.raises(TypeError):
        CorruptionProtocol()  # type: ignore[call-arg]
    field = next(f for f in dataclasses.fields(CorruptionProtocol)
                 if f.name == "scientific_corruption_seed")
    assert field.default is dataclasses.MISSING


def test_full_placeholder_is_zero_and_labelled_non_scientific():
    assert FULL_CONDITION_API_SEED == 0
    assert FULL_CONDITION_API_SEED_IS_SCIENTIFIC is False
    protocol = CorruptionProtocol(scientific_corruption_seed=12345)
    assert protocol.api_seed_for("FULL") == 0
    for condition in CONDITIONS[1:]:
        assert protocol.api_seed_for(condition) == 12345


def test_the_full_placeholder_cannot_change_the_text():
    """FULL removes nothing: any integer gives the canonical clean text."""
    text = "Tôi đã học ở trường"
    outputs = {
        corrupt(text, "FULL", seed, "sample-1", purpose=CorruptionPurpose.SELF_CHECK).corrupted_text
        for seed in (FULL_CONDITION_API_SEED, 19225, 7)
    }
    assert outputs == {canon(text)}


def test_corruption_protocol_conditions_are_fixed():
    with pytest.raises(ViUnMarkContractError):
        CorruptionProtocol(scientific_corruption_seed=1, conditions=("FULL", "P50"))


# ---------------------------------------------------------------------------
# Cross-pathway input identity
# ---------------------------------------------------------------------------
def key(sample="s1", condition="P50"):
    return CorruptionRealizationKey("UIT-VSFC", "protocol-train", sample, condition, 19225)


def test_cross_pathway_identity_is_required():
    assert CROSS_PATHWAY_INPUT_IDENTITY_REQUIRED is True
    assert BASE_GRID_INVARIANCE_REQUIRED is True
    assert ADAPTED_PATHWAYS == (Pathway.VIUNMARK_GATE, Pathway.VIUNMARK_SCALE)


def test_shared_realizations_pass():
    shared = {key(): "toi da hoc", key("s2", "STRIP_ALL"): "toi da hoc o truong"}
    require_cross_pathway_input_identity(
        {Pathway.VIUNMARK_GATE: dict(shared), Pathway.VIUNMARK_SCALE: dict(shared)})


def test_independently_redrawn_corruption_is_refused():
    gate = {key(): "tôi da học"}
    scale = {key(): "toi đã hoc"}
    with pytest.raises(ViUnMarkContractError, match="drawn once and shared"):
        require_cross_pathway_input_identity(
            {Pathway.VIUNMARK_GATE: gate, Pathway.VIUNMARK_SCALE: scale})


def test_missing_pathway_or_mismatched_keys_are_refused():
    with pytest.raises(ViUnMarkContractError):
        require_cross_pathway_input_identity({Pathway.VIUNMARK_GATE: {key(): "x"}})
    with pytest.raises(ViUnMarkContractError, match="different"):
        require_cross_pathway_input_identity(
            {Pathway.VIUNMARK_GATE: {key(): "x"}, Pathway.VIUNMARK_SCALE: {key("s9"): "x"}})


# ---------------------------------------------------------------------------
# Cross-entropy reduction: the claim must match the repository evidence
# ---------------------------------------------------------------------------
REPOSITORY_CROSS_ENTROPY_CALL_SITES = (
    ("unmark/evaluation/preg1_head.py", "train_head"),
    ("unmark/evaluation/stage2_head_campaign.py", "train_stage2_head"),
    ("unmark/evaluation/stage2_scf_campaign.py", "train_scf_head"),
    ("unmark/baselines/restore/stage2.py", "train_restore_head"),
)


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_reduction_is_declared_unresolved():
    assert CROSS_ENTROPY_REDUCTION_RECOVERED is False


def test_every_repository_cross_entropy_call_is_a_bare_call_in_a_linear_head_trainer():
    """The evidence cited for leaving the reduction open, checked rather than asserted.

    Each call passes no `reduction` and no `weight`, so the effective library default
    ("mean") applies there. Each also trains `build_head`, a single `nn.Linear`. None
    of them trains a robust MLP head or a weighted loss.
    """
    for path, name in REPOSITORY_CROSS_ENTROPY_CALL_SITES:
        function = _function(path, name)
        calls = [
            n for n in ast.walk(function)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "nn.CrossEntropyLoss"
        ]
        assert len(calls) == 1, (path, name)
        assert calls[0].args == [] and calls[0].keywords == [], (path, name)
        builders = {ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)}
        assert "build_head" in builders, (path, name)

    head_builder = _function("unmark/evaluation/preg1_head.py", "build_head")
    assert "nn.Linear" in {ast.unparse(n.func) for n in ast.walk(head_builder)
                           if isinstance(n, ast.Call)}


def test_the_repository_has_no_other_cross_entropy_call_site():
    """AST calls only. `preg1_protocol.LOSS_SPEC` spells the call inside a string
    constant documenting the linear pre-G1 loss; a string is not a call site."""
    found = set()
    for directory in ("unmark", "scripts", "docs/colab"):
        for path in (REPO / directory).rglob("*.py"):
            relative = str(path.relative_to(REPO))
            if relative.startswith("unmark/viunmark"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in {
                    "CrossEntropyLoss", "cross_entropy", "nll_loss",
                }:
                    found.add(relative)
    assert found == {path for path, _ in REPOSITORY_CROSS_ENTROPY_CALL_SITES}


def test_the_documented_linear_protocol_loss_is_mean_reduction():
    """Related, non-authoritative evidence: the pre-G1 LINEAR protocol records "mean"."""
    from unmark.evaluation.preg1_protocol import LOSS_CLASS_WEIGHTS, LOSS_REDUCTION

    assert LOSS_REDUCTION == "mean" and LOSS_CLASS_WEIGHTS is None


def test_the_robust_mlp_runner_is_absent_from_the_repository():
    """No GELU MLP head is trained anywhere outside the new public layer."""
    for directory in ("unmark", "scripts", "docs/colab"):
        for path in (REPO / directory).rglob("*.py"):
            if str(path.relative_to(REPO)).startswith("unmark/viunmark"):
                continue
            text = path.read_text(encoding="utf-8")
            assert "nn.GELU" not in text and "dropout_seed" not in text, path


def test_spec_records_the_reduction_as_unresolved():
    from unmark.viunmark.provenance import load_final_system_spec

    method = load_final_system_spec()["method"]
    assert method["robust_mlp_training_recipe"]["cross_entropy_reduction"]["status"] == "UNRESOLVED"
    assert load_final_system_spec()["status"]["recovery"]["cross_entropy_reduction"] == "UNRESOLVED"
