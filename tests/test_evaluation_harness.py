"""Minimal Stage-2 / G1 evaluation harness.

Three tiers: pure metric tests, torch-free contract tests, and torch-gated
pathway tests that skip cleanly in the ML-free local environment.

**Nothing here trains, downloads a dataset, or selects a task.**
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.evaluation import (
    LOCKED_EVALUATION_VALUES,
    OPEN_EVALUATION_VALUES,
    SCIENTIFIC_REQUIRED_VALUES,
    STAGE1_POOLING_DOES_NOT_TRANSFER,
    EvaluationContractViolation,
    EvaluationPurpose,
    EvaluationRunConfig,
    GRR_FORMULA,
    HeadConfig,
    Split,
    SplitLeakage,
    SystemPathway,
    TaskExample,
    TaskSplit,
    UndefinedGRR,
    UnresolvedEvaluationValue,
    accuracy,
    assert_disjoint_splits,
    gap_recovery_rate,
    is_grr_defined,
    macro_f1,
    per_class_scores,
    require_resolved,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
EVAL_MODULES = (
    "unmark/evaluation/contracts.py",
    "unmark/evaluation/metrics.py",
    "unmark/evaluation/pathways.py",
    "unmark/evaluation/__init__.py",
)


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    return ast.parse(source(name))


def imported(name: str) -> set[str]:
    parsed = tree(name)
    return {
        (n.module or "").split(".")[0]
        for n in ast.walk(parsed)
        if isinstance(n, ast.ImportFrom)
    } | {
        a.name.split(".")[0]
        for n in ast.walk(parsed)
        if isinstance(n, ast.Import)
        for a in n.names
    }


def called_names(name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


# TEST-ONLY diagnostic head values. These are NOT a scientific protocol; §5.2's
# concrete head values are pinned during spec lock and remain OPEN.
def diagnostic_head(num_labels: int = 2, **overrides) -> HeadConfig:
    values = dict(
        pooling="TEST_ONLY_masked_mean",
        hidden_size=None,
        learning_rate=1e-3,
        batch_size=2,
        epochs=1,
        early_stopping_patience=None,
        max_length=16,
        seed=0,
        num_labels=num_labels,
    )
    values.update(overrides)
    return HeadConfig(**values)


# ---------------------------------------------------------------------------
# 1. GRR — the proposal's formula, verified numerically
# ---------------------------------------------------------------------------
def test_grr_formula_string_matches_the_proposal():
    assert GRR_FORMULA == "(S_system - S_FLOOR) / (S_UPPER - S_FLOOR)"


def test_grr_canonical_case_84_60_72():
    """The required case: vanilla_full=84, vanilla_c=60, system_c=72 -> 0.5."""
    assert gap_recovery_rate(score_system=72, score_floor=60, score_upper=84) == 0.5


def test_grr_floor_and_upper_endpoints():
    assert gap_recovery_rate(60, 60, 84) == 0.0
    assert gap_recovery_rate(84, 60, 84) == 1.0


def test_grr_is_not_clamped_above_one():
    """§6.5 prescribes no clamping. Beating clean vanilla is informative."""
    assert gap_recovery_rate(90, 60, 84) == pytest.approx(1.25)
    assert gap_recovery_rate(96, 60, 84) == pytest.approx(1.5)


def test_grr_is_not_clamped_below_zero():
    """Doing worse than the corrupted unmodified model is equally informative."""
    assert gap_recovery_rate(48, 60, 84) == pytest.approx(-0.5)


def test_grr_zero_denominator_is_undefined_not_epsilon_patched():
    """§6.5 defines no epsilon or fallback, so none is invented."""
    with pytest.raises(UndefinedGRR, match="undefined"):
        gap_recovery_rate(70, 84, 84)
    assert not is_grr_defined(score_floor=84, score_upper=84)
    assert is_grr_defined(score_floor=60, score_upper=84)


def test_grr_degenerate_policy_is_registered_open():
    assert "grr_degenerate_denominator_policy" in OPEN_EVALUATION_VALUES
    with pytest.raises(UnresolvedEvaluationValue):
        require_resolved("grr_degenerate_denominator_policy")


def test_no_epsilon_or_clamp_appears_in_the_metric():
    """Structural: no clamping call and no epsilon constant."""
    calls = called_names("unmark/evaluation/metrics.py")
    assert not calls & {"clamp", "clip", "max", "min"}
    literals = [
        n.value
        for n in ast.walk(tree("unmark/evaluation/metrics.py"))
        if isinstance(n, ast.Constant) and isinstance(n.value, float)
    ]
    assert all(v in (0.0, 2.0) for v in literals), f"unexpected float constants: {literals}"


def test_grr_anchors_are_documented_as_vanilla():
    """§6.4: UPPER and FLOOR are both the unmodified model."""
    assert "unmodified model" in LOCKED_EVALUATION_VALUES["grr_anchors"]
    assert "VANILLA" in LOCKED_EVALUATION_VALUES["grr_anchors"]


# ---------------------------------------------------------------------------
# 2. Accuracy and macro-F1 (§6.5)
# ---------------------------------------------------------------------------
def test_accuracy_basics():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert accuracy([1, 1], [1, 1]) == 1.0
    assert accuracy([0, 0], [1, 1]) == 0.0


def test_macro_f1_is_unweighted_across_classes():
    """Macro, so a rare class counts as much as a common one."""
    # 9 of class 0 all correct; 1 of class 1 missed entirely.
    predictions = [0] * 10
    labels = [0] * 9 + [1]
    assert accuracy(predictions, labels) == 0.9
    # class 0 F1 = 2*0.9*1/1.9 = 0.947...; class 1 F1 = 0 -> macro ~0.4737
    assert macro_f1(predictions, labels) == pytest.approx(0.9473684 / 2, abs=1e-6)


def test_macro_f1_perfect_and_degenerate():
    assert macro_f1([0, 1], [0, 1]) == 1.0
    assert macro_f1([0, 0], [1, 1]) == 0.0


def test_predicted_but_absent_class_is_penalised():
    """A model inventing a class it never saw must not be silently ignored."""
    scores = per_class_scores([0, 2], [0, 1])
    assert {s.label for s in scores} == {0, 1, 2}
    assert macro_f1([0, 2], [0, 1]) < macro_f1([0, 1], [0, 1])


def test_num_labels_fixes_the_class_set():
    scores = per_class_scores([0, 0], [0, 0], num_labels=3)
    assert [s.label for s in scores] == [0, 1, 2]


def test_metrics_reject_mismatched_or_empty_input():
    with pytest.raises(EvaluationContractViolation, match="differ in length"):
        accuracy([0], [0, 1])
    with pytest.raises(EvaluationContractViolation, match="empty"):
        macro_f1([], [])


def test_metrics_need_no_ml_dependency():
    modules = imported("unmark/evaluation/metrics.py")
    assert not modules & {"torch", "numpy", "sklearn", "scipy", "pandas"}


# ---------------------------------------------------------------------------
# 3. Pathway identity
# ---------------------------------------------------------------------------
def test_only_the_two_diagnostic_pathways_exist():
    """RESTORE and ALIGN are §6.4 systems but are not implemented here."""
    assert {p.name for p in SystemPathway} == {"VANILLA", "BASE_ONLY"}


def test_base_only_is_not_called_unmark():
    """§4.5: the gate recovers the base-only pathway, not UNMARK."""
    for name in EVAL_MODULES:
        body = source(name)
        assert "BASE_ONLY is UNMARK" not in body
    assert SystemPathway.BASE_ONLY.value == "BASE_ONLY"
    assert "UNMARK" not in SystemPathway.BASE_ONLY.value


def test_neither_pathway_uses_orthography_channels():
    for pathway in SystemPathway:
        assert not pathway.uses_orthography_channels


def test_base_only_uses_base_text_and_vanilla_does_not():
    assert SystemPathway.BASE_ONLY.uses_base_text
    assert not SystemPathway.VANILLA.uses_base_text


def test_pathway_text_strips_only_for_base_only():
    from unmark.evaluation.pathways import pathway_text

    text = "Tôi đang học nghiên cứu"
    assert pathway_text(text, SystemPathway.VANILLA) == "Tôi đang học nghiên cứu"
    assert pathway_text(text, SystemPathway.BASE_ONLY) == "Toi dang hoc nghien cuu"


def test_pathway_text_canonicalises_both():
    """D-B2-004 / D-S1A-001: one canonical identity per example."""
    from unmark.evaluation.pathways import pathway_text
    import unicodedata

    nfc = "hòa"
    nfd = unicodedata.normalize("NFD", nfc)
    for pathway in SystemPathway:
        assert pathway_text(nfc, pathway) == pathway_text(nfd, pathway)


def test_no_restoration_anywhere_in_the_harness():
    for name in EVAL_MODULES:
        calls = called_names(name)
        assert not calls & {"restore", "recompose", "predict_diacritics"}
        assert "unmark.gates" not in source(name)


# ---------------------------------------------------------------------------
# 4. Split discipline and leakage (§5.4)
# ---------------------------------------------------------------------------
def test_split_permissions_follow_the_proposal():
    assert Split.TRAIN.may_train_head
    assert not Split.DEV.may_train_head
    assert not Split.TEST.may_train_head
    assert Split.TEST.is_final_evaluation


def test_head_cannot_be_trained_on_dev_or_test():
    for split in (Split.DEV, Split.TEST):
        task_split = TaskSplit("t", split, (TaskExample("a", "xin chao", 0),))
        with pytest.raises(SplitLeakage, match="train only"):
            task_split.require_trainable()


def test_train_split_is_trainable():
    TaskSplit("t", Split.TRAIN, (TaskExample("a", "xin chao", 0),)).require_trainable()


def test_split_identity_is_required():
    with pytest.raises(EvaluationContractViolation, match="no default"):
        TaskSplit("t", "TRAIN", ())  # type: ignore[arg-type]


def test_duplicate_sample_ids_within_a_split_are_rejected():
    with pytest.raises(EvaluationContractViolation, match="duplicate sample_id"):
        TaskSplit(
            "t", Split.TRAIN,
            (TaskExample("a", "x", 0), TaskExample("a", "y", 1)),
        )


def test_sample_id_shared_across_splits_is_rejected():
    """The cheapest possible leak, and the hardest to notice once cached."""
    train = TaskSplit("t", Split.TRAIN, (TaskExample("dup", "x", 0),))
    test = TaskSplit("t", Split.TEST, (TaskExample("dup", "x", 0),))
    with pytest.raises(SplitLeakage, match="appears in both"):
        assert_disjoint_splits([train, test])


def test_same_id_in_different_tasks_is_allowed():
    a = TaskSplit("task-a", Split.TRAIN, (TaskExample("s1", "x", 0),))
    b = TaskSplit("task-b", Split.TRAIN, (TaskExample("s1", "x", 0),))
    assert_disjoint_splits([a, b])


def test_stable_sample_id_required():
    with pytest.raises(EvaluationContractViolation, match="stable identity"):
        TaskExample("", "x", 0)


def test_label_must_be_an_int():
    with pytest.raises(EvaluationContractViolation):
        TaskExample("a", "x", "positive")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Head contract — required values, clean-only, per-pathway
# ---------------------------------------------------------------------------
def test_head_config_has_no_defaults():
    import inspect

    empty = inspect.Parameter.empty
    parameters = inspect.signature(HeadConfig).parameters
    for name in parameters:
        assert parameters[name].default is empty, f"{name} has an experiment-facing default"


def test_head_open_values_are_registered():
    for name in (
        "head_architecture", "head_pooling", "head_optimizer", "head_learning_rate",
        "head_batch_size", "head_epochs", "head_early_stopping", "seed_list", "max_length",
    ):
        assert name in OPEN_EVALUATION_VALUES, name
        with pytest.raises(UnresolvedEvaluationValue):
            require_resolved(name)


def test_head_config_validates_values():
    with pytest.raises(EvaluationContractViolation):
        diagnostic_head(num_labels=1)
    with pytest.raises(EvaluationContractViolation):
        diagnostic_head(batch_size=0)
    with pytest.raises(EvaluationContractViolation):
        diagnostic_head(learning_rate=0.0)


def test_head_protocol_identity_is_checkable():
    """§5.2: one architecture, identical across all five systems."""
    a, b = diagnostic_head(), diagnostic_head()
    assert a.identical_protocol_to(b)
    assert not a.identical_protocol_to(diagnostic_head(epochs=2))


def test_head_binding_refuses_non_train_splits():
    from unmark.evaluation.pathways import HeadBinding

    for split in (Split.DEV, Split.TEST):
        with pytest.raises(SplitLeakage, match="train only"):
            HeadBinding("t", SystemPathway.VANILLA, split, diagnostic_head())


def test_head_binding_refuses_corrupted_training():
    """§5.2 locks clean-only head training."""
    from unmark.evaluation.pathways import HeadBinding

    binding = HeadBinding("t", SystemPathway.VANILLA, Split.TRAIN, diagnostic_head())
    binding.require_clean_training("FULL")
    for condition in ("P50", "STRIP_ALL", "P100"):
        with pytest.raises(SplitLeakage, match="clean-only"):
            binding.require_clean_training(condition)


def test_clean_only_head_training_is_recorded_as_locked():
    assert "clean data only" in LOCKED_EVALUATION_VALUES["head_trained_on_clean_only"]
    assert "identically for all five systems" in LOCKED_EVALUATION_VALUES[
        "head_protocol_identical_across_systems"
    ]


# ---------------------------------------------------------------------------
# 6. Task/dataset must not be chosen here
# ---------------------------------------------------------------------------
def test_no_dataset_or_benchmark_name_is_baked_in():
    for name in EVAL_MODULES:
        parsed = tree(name)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                for banned in ("uit-", "vsfc", "vsmec", "hsd", "vlsp", "huggingface.co", "http"):
                    assert banned not in lowered, f"{name} names a dataset/URL: {banned}"
        assert not imported(name) & {"datasets", "requests", "urllib"}


def test_task_dataset_is_registered_open():
    assert "task_dataset" in OPEN_EVALUATION_VALUES
    assert "g1_task_choice" in OPEN_EVALUATION_VALUES
    with pytest.raises(UnresolvedEvaluationValue):
        require_resolved("task_dataset")


def test_scientific_config_cannot_be_built_while_values_are_open():
    """Uses a non-TEST_ONLY pooling name so the *missing values* guard is the one
    exercised, rather than the pooling-placeholder guard tested separately."""
    head = diagnostic_head(pooling="some_resolved_rule")
    with pytest.raises(UnresolvedEvaluationValue, match="task_dataset"):
        EvaluationRunConfig(
            purpose=EvaluationPurpose.SCIENTIFIC,
            task_id="whatever",
            head=head,
            pathways=(SystemPathway.VANILLA, SystemPathway.BASE_ONLY),
        )


def test_diagnostic_config_is_allowed_and_labelled():
    config = EvaluationRunConfig(
        purpose=EvaluationPurpose.DIAGNOSTIC,
        task_id="TEST_ONLY_synthetic",
        head=diagnostic_head(),
        pathways=(SystemPathway.VANILLA, SystemPathway.BASE_ONLY),
        note="harness self-test",
    )
    payload = config.to_dict()
    assert payload["diagnostic_only"] is True
    assert payload["values_are_scientific"] is False
    assert payload["resolved_values"] == []
    assert payload["pathways"] == ["VANILLA", "BASE_ONLY"]


def test_scientific_requirements_subset_of_open_register():
    assert set(SCIENTIFIC_REQUIRED_VALUES) <= set(OPEN_EVALUATION_VALUES)


def test_backbone_decision_remains_open():
    assert "backbone_finalisation" in OPEN_EVALUATION_VALUES
    decisions = source("docs/spec/decisions.md")
    assert "D-B3B0-002" in decisions and "remains OPEN" in decisions


# ---------------------------------------------------------------------------
# 6b. Stage-2 pooling must remain OPEN (it is not Stage-1's pooling)
# ---------------------------------------------------------------------------
def test_stage2_pooling_is_registered_open():
    assert "head_pooling" in OPEN_EVALUATION_VALUES
    with pytest.raises(UnresolvedEvaluationValue):
        require_resolved("head_pooling")
    assert "Stage-1" in OPEN_EVALUATION_VALUES["head_pooling"]


def test_stage1_pooling_is_documented_as_not_transferring():
    assert "does NOT transfer to Stage-2" in OPEN_EVALUATION_VALUES["head_pooling"]
    assert "two different decisions" in STAGE1_POOLING_DOES_NOT_TRANSFER


def test_stage1_pooling_itself_is_unchanged():
    """The repair must not touch the locked Stage-1 rule."""
    from unmark.modeling.contracts import STAGE1_POOLING

    assert STAGE1_POOLING == "masked_mean_over_non_special_content_tokens"
    assert "def masked_mean_non_special" in source("unmark/modeling/pooling.py")


def test_extraction_does_not_pool_on_the_scientific_path():
    """Structural: the scientific extractor returns hidden states, not vectors."""
    body = source("unmark/evaluation/pathways.py")
    assert "def encoder_hidden_states" in body
    assert "def pooled_representations" not in body
    for node in ast.walk(tree("unmark/evaluation/pathways.py")):
        if isinstance(node, ast.FunctionDef) and node.name == "encoder_hidden_states":
            unparsed = ast.unparse(node)
            assert "masked_mean" not in unparsed, "scientific extraction pools implicitly"


def test_only_the_test_only_helper_may_pool():
    """`masked_mean_non_special` may be reached from exactly one place, and that
    place is named TEST_ONLY and gated on purpose."""
    poolers = [
        node.name
        for node in ast.walk(tree("unmark/evaluation/pathways.py"))
        if isinstance(node, ast.FunctionDef) and "masked_mean" in ast.unparse(node)
    ]
    assert poolers == ["TEST_ONLY_masked_mean_pool"]


def test_head_pooling_is_required_and_not_decorative():
    """`HeadConfig.pooling` must matter: extraction is unpooled, so a head has to
    apply this rule rather than inherit one."""
    import inspect

    assert inspect.signature(HeadConfig).parameters["pooling"].default is inspect.Parameter.empty
    with pytest.raises(EvaluationContractViolation, match="no default"):
        diagnostic_head(pooling="")
    assert diagnostic_head().pooling_is_test_only


def test_scientific_config_rejects_a_test_only_pooling_name():
    with pytest.raises(UnresolvedEvaluationValue, match="TEST_ONLY placeholder"):
        EvaluationRunConfig(
            purpose=EvaluationPurpose.SCIENTIFIC,
            task_id="x",
            head=diagnostic_head(),
            pathways=(SystemPathway.VANILLA,),
            resolved_values=frozenset(SCIENTIFIC_REQUIRED_VALUES),
        )


def test_no_pooling_option_was_invented():
    """The proposal has not chosen among CLS/max/attention pooling, so neither
    does this harness."""
    body = source("unmark/evaluation/contracts.py") + source("unmark/evaluation/pathways.py")
    for invented in ("class PoolingStrategy", "CLS_TOKEN", "MAX_POOL", "ATTENTION_POOL"):
        assert invented not in body


# ---------------------------------------------------------------------------
# 7. Hygiene — no training, no second pooling, ML-free package
# ---------------------------------------------------------------------------
def test_evaluation_package_is_torch_free():
    assert "torch" not in imported("unmark/evaluation/__init__.py")
    actually_imported = {
        n.module
        for n in tree("unmark/evaluation/__init__.py").body
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "unmark.evaluation.pathways" not in actually_imported


@pytest.mark.parametrize("name", EVAL_MODULES)
def test_no_optimizer_or_training_loop(name):
    calls = called_names(name)
    assert not calls & {"step", "zero_grad", "backward", "save_pretrained"}
    assert not imported(name) & {"optim", "wandb", "tensorboard"}
    attributes = {n.attr for n in ast.walk(tree(name)) if isinstance(n, ast.Attribute)}
    assert not attributes & {"AdamW", "SGD", "lr_scheduler"}


def test_no_second_pooling_implementation():
    body = source("unmark/evaluation/pathways.py")
    assert "def masked_mean" not in body
    assert "masked_mean_non_special" in body


def test_representation_extraction_is_frozen_and_no_grad():
    for node in ast.walk(tree("unmark/evaluation/pathways.py")):
        if isinstance(node, ast.FunctionDef) and node.name == "pooled_representations":
            body = ast.unparse(node)
            assert "no_grad" in body
            assert "encoder.eval()" in body


def test_no_stage1_training_or_restore_or_align():
    for name in EVAL_MODULES:
        body = source(name).lower()
        for banned in ("class alignadapter", "def train_stage1", "restorer"):
            assert banned not in body


# ---------------------------------------------------------------------------
# 8. Torch-gated pathway plumbing
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


class StubTokenizer:
    """Whitespace stand-in returning tensors in the HF shape."""

    def __call__(self, texts, padding=True, truncation=True, max_length=None,
                 return_tensors=None, return_special_tokens_mask=False):
        rows = [["<s>"] + t.split()[:max_length - 2] + ["</s>"] for t in texts]
        width = max(len(r) for r in rows)
        ids, attention, special = [], [], []
        for row in rows:
            pad = width - len(row)
            ids.append([7 + (len(tok) % 5) for tok in row] + [1] * pad)
            attention.append([1] * len(row) + [0] * pad)
            special.append([1] + [0] * (len(row) - 2) + [1] + [1] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attention),
            "special_tokens_mask": torch.tensor(special),
        }


def stub_encoder(d: int = 8):
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(64, d, padding_idx=1)

        def forward(self, input_ids=None, attention_mask=None, **_):
            return self.embed(input_ids)

    encoder = Encoder()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


def synthetic_split(split: Split = Split.TRAIN) -> TaskSplit:
    return TaskSplit(
        "TEST_ONLY_synthetic",
        split,
        (
            TaskExample("s1", "Tôi đang học nghiên cứu", 0),
            TaskExample("s2", "Chào bạn", 1),
        ),
    )


@requires_torch
def test_runtime_pathways_produce_different_token_ids():
    """The whole diagnostic rests on the two pathways differing."""
    from unmark.evaluation.pathways import encode_split

    tokenizer, config = StubTokenizer(), diagnostic_head()
    vanilla = encode_split(synthetic_split(), SystemPathway.VANILLA, tokenizer, config)
    base = encode_split(synthetic_split(), SystemPathway.BASE_ONLY, tokenizer, config)
    assert vanilla.pathway is SystemPathway.VANILLA
    assert base.pathway is SystemPathway.BASE_ONLY
    assert not torch.equal(vanilla.input_ids, base.input_ids)


@requires_torch
def test_runtime_encoded_split_carries_identity():
    from unmark.evaluation.pathways import encode_split

    encoded = encode_split(
        synthetic_split(Split.DEV), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()
    )
    assert encoded.identity == ("TEST_ONLY_synthetic", "DEV", "VANILLA")
    assert encoded.sample_ids == ("s1", "s2")
    assert encoded.labels == (0, 1)


@requires_torch
def test_runtime_extraction_returns_unpooled_hidden_states():
    """Stage-2 pooling is OPEN (§5.2), so extraction must NOT pool."""
    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

    encoded = encode_split(
        synthetic_split(), SystemPathway.BASE_ONLY, StubTokenizer(), diagnostic_head()
    )
    states = encoder_hidden_states(encoded, stub_encoder(8))
    assert states.hidden_states.dim() == 3, "extraction pooled; Stage-2 pooling is OPEN"
    assert states.hidden_states.shape[0] == 2
    assert states.hidden_states.shape[2] == 8
    assert not states.hidden_states.requires_grad
    assert states.identity == ("TEST_ONLY_synthetic", "TRAIN", "BASE_ONLY")


@requires_torch
def test_runtime_masks_travel_with_the_hidden_states():
    """Whatever pooling is eventually pinned needs the masks to apply it."""
    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

    encoded = encode_split(
        synthetic_split(), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()
    )
    states = encoder_hidden_states(encoded, stub_encoder())
    assert states.attention_mask.shape == states.hidden_states.shape[:2]
    assert states.special_tokens_mask.shape == states.hidden_states.shape[:2]


@requires_torch
def test_runtime_scientific_path_cannot_reach_masked_mean():
    """The core of the repair: a SCIENTIFIC purpose cannot pool implicitly."""
    from unmark.evaluation.pathways import (
        TEST_ONLY_masked_mean_pool,
        encode_split,
        encoder_hidden_states,
    )

    states = encoder_hidden_states(
        encode_split(synthetic_split(), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()),
        stub_encoder(),
    )
    with pytest.raises(UnresolvedEvaluationValue, match="diagnostic-only"):
        TEST_ONLY_masked_mean_pool(states, EvaluationPurpose.SCIENTIFIC)


@requires_torch
def test_runtime_test_only_pooling_works_for_diagnostics():
    from unmark.evaluation.pathways import (
        TEST_ONLY_masked_mean_pool,
        encode_split,
        encoder_hidden_states,
    )

    states = encoder_hidden_states(
        encode_split(synthetic_split(), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()),
        stub_encoder(8),
    )
    pooled = TEST_ONLY_masked_mean_pool(states, EvaluationPurpose.DIAGNOSTIC)
    assert pooled.shape == (2, 8)


@requires_torch
def test_runtime_encoder_is_not_mutated_by_extraction():
    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

    encoder = stub_encoder()
    before = [p.detach().clone() for p in encoder.parameters()]
    encoder_hidden_states(
        encode_split(synthetic_split(), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()),
        encoder,
    )
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    for original, current in zip(before, encoder.parameters()):
        assert torch.equal(original, current)


@requires_torch
def test_runtime_cross_pathway_head_reuse_is_refused():
    """A head fitted to one pathway has no defined meaning on another."""
    from unmark.evaluation.pathways import HeadBinding, encode_split, encoder_hidden_states

    encoder = stub_encoder()
    base_states = encoder_hidden_states(
        encode_split(synthetic_split(), SystemPathway.BASE_ONLY, StubTokenizer(), diagnostic_head()),
        encoder,
    )
    vanilla_head = HeadBinding(
        "TEST_ONLY_synthetic", SystemPathway.VANILLA, Split.TRAIN, diagnostic_head()
    )
    with pytest.raises(SplitLeakage, match="pathway mismatch"):
        base_states.require_same_pathway(vanilla_head)

    base_head = HeadBinding(
        "TEST_ONLY_synthetic", SystemPathway.BASE_ONLY, Split.TRAIN, diagnostic_head()
    )
    base_states.require_same_pathway(base_head)


@requires_torch
def test_runtime_dev_representations_cannot_train_a_head():
    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

    states = encoder_hidden_states(
        encode_split(
            synthetic_split(Split.DEV), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()
        ),
        stub_encoder(),
    )
    with pytest.raises(SplitLeakage, match="train only"):
        states.require_trainable()


@requires_torch
def test_runtime_task_mismatch_is_refused():
    from unmark.evaluation.pathways import HeadBinding, encode_split, encoder_hidden_states

    states = encoder_hidden_states(
        encode_split(synthetic_split(), SystemPathway.VANILLA, StubTokenizer(), diagnostic_head()),
        stub_encoder(),
    )
    other = HeadBinding("a-different-task", SystemPathway.VANILLA, Split.TRAIN, diagnostic_head())
    with pytest.raises(SplitLeakage, match="task mismatch"):
        states.require_same_pathway(other)
