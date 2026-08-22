"""Pre-G1 frozen-encoder head trainer / evaluator.

Network-free and download-free. Torch-free tests cover every selection rule,
boundary guard and cache-provenance check; the tests that genuinely need tensors
are marked `@requires_torch` and skip in the ML-free local environment, exactly
as the existing harness tests do.

No real UIT-VSFC, no real split artifacts, no PhoBERT.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import statistics

import pytest

from unmark.evaluation.contracts import (
    EvaluationContractViolation,
    SplitLeakage,
    SystemPathway,
)
from unmark.evaluation.preg1_head import (
    BoundRepresentations,
    DETERMINISM_SCOPE,
    LABEL_ORDER,
    NO_SIGNIFICANCE_TEST,
    PREG1_HEAD_SCHEMA_VERSION,
    EpochScore,
    FrozenLearningRate,
    HeadRun,
    LrCandidate,
    PairedDiagnostic,
    PairedSeedResult,
    Preg1Role,
    RepresentationCache,
    RepresentationKey,
    SplitMembership,
    deterministic_batches,
    freeze_learning_rate,
    load_membership,
    ordered_id_digest,
    require_full_schedule,
    require_protocol_settings,
    require_training_roles,
    sample_stdev,
    score_predictions,
    select_checkpoint,
    select_learning_rate,
)
from unmark.evaluation.preg1_protocol import (
    ADAMW_BETAS,
    ADAMW_EPS,
    BATCH_SIZE,
    EPOCHS,
    LR_GRID,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PREG1_POOLING,
    TUNING_SEEDS,
    WEIGHT_DECAY_BIAS,
    WEIGHT_DECAY_WEIGHT,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

try:  # pragma: no cover - depends on the environment
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover - the normal local state
    TORCH = False

requires_torch = pytest.mark.skipif(not TORCH, reason="torch is not installed locally")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flat_run(pathway, lr, seed, f1=0.5, acc=0.5, hidden=8):
    return HeadRun(
        pathway=pathway,
        learning_rate=lr,
        seed=seed,
        scores=tuple(EpochScore(e, f1, acc) for e in range(1, EPOCHS + 1)),
        hidden_size=hidden,
    )


def full_grid(f1_by_lr=None, acc=0.5, pathway=SystemPathway.VANILLA):
    """The whole precommitted grid as candidates, all on one pathway.

    `pathway` defaults to VANILLA (the primary). The secondary own-LR
    sensitivity sweeps the identical grid and seeds on BASE_ONLY.
    """
    f1_by_lr = f1_by_lr or {}
    return [
        LrCandidate(
            lr,
            tuple(
                flat_run(pathway, lr, seed, f1_by_lr.get(lr, 0.5), acc)
                for seed in TUNING_SEEDS
            ),
            pathway=pathway,
        )
        for lr in LR_GRID
    ]


class FakeMatrix:
    """Duck-typed stand-in for a tensor, so the binding contract is testable
    without torch. Real tensors satisfy the same three attributes."""

    def __init__(self, count, hidden, dtype="torch.float32", requires_grad=False):
        self.shape = (count, hidden)
        self.dtype = dtype
        self.requires_grad = requires_grad


def bound(role=None, pathway=None, **overrides):
    """A `BoundRepresentations` over a fake matrix, for role-contract tests."""
    key = sample_key(
        **{k: v for k, v in
           dict(role=role, pathway=pathway).items() if v is not None},
        **overrides,
    )
    return BoundRepresentations(
        values=FakeMatrix(key.count, key.hidden_size, key.dtype), key=key
    )


def sample_key(**overrides):
    base = dict(
        dataset="UIT-VSFC",
        dataset_version="1.0",
        task="sentiment",
        role=Preg1Role.PROTOCOL_TRAIN,
        pathway=SystemPathway.VANILLA,
        source_identity="a" * 64,
        ordered_id_digest=ordered_id_digest(["train:00000", "train:00001"]),
        tokenizer_id="vinai/phobert-base",
        model_revision="01daacda68afe13d83023d16ec647239e344a1e6",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        pooling=PREG1_POOLING.value,
        dtype="torch.float32",
        hidden_size=8,
        count=2,
    )
    base.update(overrides)
    return RepresentationKey(**base)


# ===========================================================================
# A. Data / membership boundaries
# ===========================================================================
def test_membership_loads_and_partitions(tmp_path):
    membership = SplitMembership(("a", "b", "c"), ("d",), assignment_digest="x")
    membership.require_partitions(["a", "b", "c", "d"])
    assert membership.all_ids == ("a", "b", "c", "d")


def test_duplicate_ids_within_a_part_fail():
    with pytest.raises(EvaluationContractViolation, match="duplicate ids"):
        SplitMembership(("a", "a"), ("b",), assignment_digest="x")


def test_overlapping_parts_fail():
    with pytest.raises(SplitLeakage, match="overlap"):
        SplitMembership(("a", "b"), ("b",), assignment_digest="x")


def test_empty_part_fails():
    with pytest.raises(EvaluationContractViolation, match="empty"):
        SplitMembership((), ("b",), assignment_digest="x")


def test_incomplete_partition_fails():
    membership = SplitMembership(("a",), ("b",), assignment_digest="x")
    with pytest.raises(EvaluationContractViolation, match="in neither part"):
        membership.require_partitions(["a", "b", "c"])


def test_unknown_membership_id_fails():
    membership = SplitMembership(("a", "zzz"), ("b",), assignment_digest="x")
    with pytest.raises(EvaluationContractViolation, match="not in the pool"):
        membership.require_partitions(["a", "b"])


def test_duplicate_pool_ids_fail():
    membership = SplitMembership(("a",), ("b",), assignment_digest="x")
    with pytest.raises(EvaluationContractViolation, match="pool has duplicate ids"):
        membership.require_partitions(["a", "b", "b"])


def test_official_validation_has_no_internal_membership():
    membership = SplitMembership(("a",), ("b",), assignment_digest="x")
    with pytest.raises(EvaluationContractViolation, match="no internal membership"):
        membership.ids_for(Preg1Role.OFFICIAL_VALIDATION)


def write_split_dir(tmp_path, train=("t1", "t2"), dev=("d1",), schema="preg1-split-v1"):
    out = tmp_path / "split"
    out.mkdir()
    (out / "protocol-train.ids.txt").write_text("".join(f"{i}\n" for i in train), encoding="utf-8")
    (out / "protocol-dev.ids.txt").write_text("".join(f"{i}\n" for i in dev), encoding="utf-8")
    (out / "split-manifest.json").write_text(
        json.dumps({"schema_version": schema, "result": {"assignment_digest": "deadbeef"}}),
        encoding="utf-8",
    )
    return out


def test_load_membership_reads_a_valid_directory(tmp_path):
    membership = load_membership(write_split_dir(tmp_path))
    assert membership.protocol_train == ("t1", "t2")
    assert membership.assignment_digest == "deadbeef"


def test_load_membership_rejects_a_wrong_schema(tmp_path):
    with pytest.raises(EvaluationContractViolation, match="schema must be"):
        load_membership(write_split_dir(tmp_path, schema="preg1-split-v99"))


def test_load_membership_rejects_a_malformed_manifest(tmp_path):
    out = write_split_dir(tmp_path)
    (out / "split-manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="malformed"):
        load_membership(out)


def test_load_membership_rejects_a_missing_id_file(tmp_path):
    out = write_split_dir(tmp_path)
    (out / "protocol-dev.ids.txt").unlink()
    with pytest.raises(EvaluationContractViolation, match="missing protocol-dev"):
        load_membership(out)


def test_load_membership_enforces_pinned_digests(tmp_path):
    out = write_split_dir(tmp_path)
    with pytest.raises(EvaluationContractViolation, match="digest mismatch"):
        load_membership(out, expected_digests={"protocol-train.ids.txt": "0" * 64})


def test_load_membership_accepts_correct_pinned_digests(tmp_path):
    out = write_split_dir(tmp_path)
    digest = hashlib.sha256((out / "protocol-train.ids.txt").read_bytes()).hexdigest()
    membership = load_membership(out, expected_digests={"protocol-train.ids.txt": digest})
    assert membership.protocol_train == ("t1", "t2")


def test_official_test_is_not_representable():
    """Not merely forbidden — unnameable, so no code path can reach it."""
    assert "OFFICIAL_TEST" not in {role.name for role in Preg1Role}
    assert {role.value for role in Preg1Role} == {
        "protocol-train", "protocol-dev", "official-validation"
    }
    with pytest.raises(ValueError):
        Preg1Role("official-test")


def test_only_protocol_dev_may_select():
    assert Preg1Role.PROTOCOL_DEV.may_select
    assert not Preg1Role.OFFICIAL_VALIDATION.may_select
    assert not Preg1Role.PROTOCOL_TRAIN.may_select


def test_only_protocol_train_may_train_the_head():
    assert Preg1Role.PROTOCOL_TRAIN.may_train_head
    assert not Preg1Role.PROTOCOL_DEV.may_train_head
    assert not Preg1Role.OFFICIAL_VALIDATION.may_train_head


def test_cli_exposes_no_official_test_and_no_constant_overrides():
    tree = ast.parse((REPO / "scripts/preg1_head_diagnostic.py").read_text(encoding="utf-8"))
    flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert not {"--test", "--official-test", "--test-split"} & flags
    assert not {"--seeds", "--tuning-seeds", "--measurement-seeds", "--grid",
                "--epochs", "--batch-size", "--max-length"} & flags
    assert "--frozen-lr" in flags  # measure cannot invent its own LR


# ===========================================================================
# B. Pathways
# ===========================================================================
def test_vanilla_canonicalises_but_does_not_strip():
    from unmark.evaluation.pathways import pathway_text
    from unmark.orthography import canon, decompose

    text = "Tôi học ở trường"
    assert pathway_text(text, SystemPathway.VANILLA) == canon(text)
    assert pathway_text(text, SystemPathway.VANILLA) != decompose(canon(text)).base_text


def test_base_only_canonicalises_then_strips():
    from unmark.evaluation.pathways import pathway_text
    from unmark.orthography import canon, decompose

    text = "Tôi học ở trường"
    assert pathway_text(text, SystemPathway.BASE_ONLY) == decompose(canon(text)).base_text


def test_the_two_pathways_differ_only_by_the_stripping_step():
    from unmark.evaluation.pathways import pathway_text
    from unmark.orthography import decompose

    text = "Tôi học ở trường"
    vanilla = pathway_text(text, SystemPathway.VANILLA)
    assert decompose(vanilla).base_text == pathway_text(text, SystemPathway.BASE_ONLY)


def test_no_word_segmentation_in_either_pathway():
    """RAW_BASE: no segmenter, so no underscore-joined compounds appear."""
    from unmark.evaluation.pathways import pathway_text

    text = "sinh viên học tập"
    for pathway in (SystemPathway.VANILLA, SystemPathway.BASE_ONLY):
        assert "_" not in pathway_text(text, pathway)


def test_pathway_module_imports_no_segmenter():
    source = (REPO / "unmark/evaluation/preg1_head.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert not modules & {"py_vncorenlp", "vncorenlp", "underthesea", "pyvi"}


def test_tokenization_contract_is_the_locked_one():
    from unmark.evaluation.preg1_head import PREG1_TOKENIZATION

    assert f"max_length={MAX_LENGTH}" in PREG1_TOKENIZATION
    assert "'max_length'" in PREG1_TOKENIZATION  # padding
    assert "no word segmentation" in PREG1_TOKENIZATION


def test_pooling_is_first_token_and_scoped_to_preg1():
    from unmark.evaluation.preg1_head import PREG1_ONLY_first_token_pool, POOLING_SCOPE_WARNING

    assert PREG1_POOLING.value == "FIRST_TOKEN"
    assert PREG1_ONLY_first_token_pool.__name__.startswith("PREG1_ONLY_")
    assert "OPEN" in POOLING_SCOPE_WARNING


# ===========================================================================
# C. Frozen encoder  (torch)
# ===========================================================================
def make_fake_encoder(hidden=8, layers=1):
    import torch
    from torch import nn

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, hidden)

        def forward(self, input_ids, attention_mask=None):
            return self.embed(input_ids)

    encoder = FakeEncoder()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


@requires_torch
def test_frozen_encoder_check_accepts_a_frozen_eval_encoder():
    from unmark.evaluation.preg1_head import require_frozen_encoder

    require_frozen_encoder(make_fake_encoder())


@requires_torch
def test_a_trainable_encoder_parameter_fails():
    from unmark.evaluation.preg1_head import require_frozen_encoder

    encoder = make_fake_encoder()
    next(encoder.parameters()).requires_grad_(True)
    with pytest.raises(EvaluationContractViolation, match="require grad"):
        require_frozen_encoder(encoder)


@requires_torch
def test_a_training_mode_encoder_fails():
    from unmark.evaluation.preg1_head import require_frozen_encoder

    encoder = make_fake_encoder()
    encoder.train()
    with pytest.raises(EvaluationContractViolation, match="eval mode"):
        require_frozen_encoder(encoder)


@requires_torch
def test_extraction_is_fp32_detached_and_first_token():
    import torch
    from unmark.evaluation.preg1_head import extract_representations

    encoder = make_fake_encoder()
    ids = torch.randint(0, 50, (4, 7))
    mask = torch.ones_like(ids)
    features = extract_representations(encoder, ids, mask)

    assert features.shape == (4, 8)
    assert features.dtype is torch.float32
    assert not features.requires_grad
    assert features.grad_fn is None
    expected = encoder.embed(ids)[:, 0, :]
    assert torch.equal(features, expected.detach().to(torch.float32))


@requires_torch
def test_a_real_head_backward_cannot_reach_the_encoder():
    """Drives an actual backward through a head fitted on the features.

    An earlier version of this test only called `backward()` when the features
    required grad -- which they never do, so it could not fail. This one always
    backpropagates, so if extraction ever leaked a graph the encoder would
    accumulate gradients here.
    """
    import torch
    from unmark.evaluation.preg1_head import build_head, extract_representations

    encoder = make_fake_encoder()
    ids = torch.randint(0, 50, (3, 5))
    features = extract_representations(encoder, ids, torch.ones_like(ids))

    head = build_head(features.shape[1], seed=1)
    loss = head(features).sum()
    assert loss.requires_grad, "the head itself must still be trainable"
    loss.backward()

    assert all(p.grad is None for p in encoder.parameters())
    assert head.weight.grad is not None


@requires_torch
def test_first_token_pool_rejects_a_pooled_input():
    import torch
    from unmark.evaluation.preg1_head import PREG1_ONLY_first_token_pool

    with pytest.raises(EvaluationContractViolation, match=r"\[N, L, d\]"):
        PREG1_ONLY_first_token_pool(torch.zeros(4, 8))


# ===========================================================================
# D. Head  (torch)
# ===========================================================================
@requires_torch
def test_head_shape_init_and_absence_of_dropout():
    import torch
    from torch import nn
    from unmark.evaluation.preg1_head import build_head

    head = build_head(16, seed=7)
    assert isinstance(head, nn.Linear)
    assert head.weight.shape == (3, 16)
    assert head.bias.shape == (3,)
    assert torch.equal(head.bias, torch.zeros(3))
    assert not any(isinstance(m, nn.Dropout) for m in head.modules())
    # Xavier-uniform bound for fan_in=16, fan_out=3
    bound = (6.0 / (16 + 3)) ** 0.5
    assert head.weight.abs().max().item() <= bound + 1e-6


@requires_torch
def test_same_seed_gives_bit_identical_parameters():
    import torch
    from unmark.evaluation.preg1_head import build_head

    a, b = build_head(16, seed=11), build_head(16, seed=11)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)


@requires_torch
def test_different_seed_changes_initialisation():
    import torch
    from unmark.evaluation.preg1_head import build_head

    assert not torch.equal(build_head(16, seed=11).weight, build_head(16, seed=12).weight)


@requires_torch
def test_paired_seed_gives_identical_starts_for_both_pathways():
    """The paired guarantee, and it must survive RNG consumed in between."""
    import torch
    from unmark.evaluation.preg1_head import build_head

    vanilla = build_head(16, seed=53148)
    torch.rand(1000)  # the Vanilla arm runs and advances the global RNG
    base_only = build_head(16, seed=53148)
    assert torch.equal(vanilla.weight, base_only.weight)
    assert torch.equal(vanilla.bias, base_only.bias)


@requires_torch
def test_head_rejects_a_bad_hidden_size():
    from unmark.evaluation.preg1_head import build_head

    with pytest.raises(EvaluationContractViolation, match="positive int"):
        build_head(0, seed=1)


# ===========================================================================
# E. Optimizer  (torch)
# ===========================================================================
@requires_torch
def test_optimizer_is_adamw_with_the_locked_settings():
    import torch
    from unmark.evaluation.preg1_head import build_head, build_optimizer

    head = build_head(8, seed=3)
    optimizer = build_optimizer(head, 1e-3)
    assert isinstance(optimizer, torch.optim.AdamW)
    for group in optimizer.param_groups:
        assert group["betas"] == ADAMW_BETAS
        assert group["eps"] == ADAMW_EPS
        assert group["amsgrad"] is False
        assert group["lr"] == 1e-3


@requires_torch
def test_weight_decays_but_bias_does_not():
    from unmark.evaluation.preg1_head import build_head, build_optimizer

    head = build_head(8, seed=3)
    optimizer = build_optimizer(head, 1e-3)
    decays = {
        id(group["params"][0]): group["weight_decay"] for group in optimizer.param_groups
    }
    assert decays[id(head.weight)] == WEIGHT_DECAY_WEIGHT == 0.01
    assert decays[id(head.bias)] == WEIGHT_DECAY_BIAS == 0.0


@requires_torch
def test_learning_rate_is_constant_across_steps():
    """No schedule object is created, so the LR cannot drift."""
    import torch
    from unmark.evaluation.preg1_head import build_head, build_optimizer

    head = build_head(8, seed=3)
    optimizer = build_optimizer(head, 3e-4)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        head(torch.zeros(2, 8)).sum().backward()
        optimizer.step()
        assert all(group["lr"] == 3e-4 for group in optimizer.param_groups)


def test_trainer_uses_no_scheduler_clipping_or_accumulation():
    """Structural: those APIs must not appear in the module at all."""
    source = (REPO / "unmark/evaluation/preg1_head.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "clip_grad_norm_" not in called
    assert "clip_grad_value_" not in called
    assert not any(name.startswith("lr_scheduler") for name in called)


def test_protocol_settings_guard_matches_the_trainer():
    require_protocol_settings()


# ===========================================================================
# F. Training protocol
# ===========================================================================
def test_all_thirty_epochs_are_required():
    with pytest.raises(EvaluationContractViolation, match="no early stopping"):
        require_full_schedule([EpochScore(e, 0.5, 0.5) for e in range(1, 10)])


def test_a_full_schedule_passes():
    require_full_schedule([EpochScore(e, 0.5, 0.5) for e in range(1, EPOCHS + 1)])


def test_epoch_zero_is_not_checkpoint_eligible():
    with pytest.raises(EvaluationContractViolation, match="not checkpoint-eligible"):
        EpochScore(0, 0.9, 0.9)


def test_epoch_beyond_the_budget_is_rejected():
    with pytest.raises(EvaluationContractViolation, match="not checkpoint-eligible"):
        EpochScore(EPOCHS + 1, 0.9, 0.9)


def test_checkpoint_prefers_macro_f1():
    scores = [EpochScore(e, 0.5, 0.9) for e in range(1, EPOCHS + 1)]
    scores[5] = EpochScore(6, 0.7, 0.1)
    assert select_checkpoint(scores).epoch == 6


def test_checkpoint_tie_breaks_on_accuracy_then_earliest_epoch():
    scores = [EpochScore(e, 0.5, 0.5) for e in range(1, EPOCHS + 1)]
    scores[9] = EpochScore(10, 0.5, 0.9)
    scores[19] = EpochScore(20, 0.5, 0.9)
    assert select_checkpoint(scores).epoch == 10


def test_full_tie_selects_the_earliest_epoch():
    assert select_checkpoint([EpochScore(e, 0.5, 0.5) for e in range(1, EPOCHS + 1)]).epoch == 1


def test_checkpoint_selection_does_not_depend_on_input_order():
    scores = [EpochScore(e, 0.5, 0.5) for e in range(1, EPOCHS + 1)]
    scores[9] = EpochScore(10, 0.5, 0.9)
    scores[19] = EpochScore(20, 0.5, 0.9)
    assert select_checkpoint(list(reversed(scores))).epoch == 10


def test_duplicate_epochs_are_rejected():
    with pytest.raises(EvaluationContractViolation, match="duplicate epochs"):
        select_checkpoint([EpochScore(1, 0.5, 0.5), EpochScore(1, 0.6, 0.6)])


def test_macro_f1_on_synthetic_edge_cases():
    assert score_predictions([0, 1, 2], [0, 1, 2])[0] == pytest.approx(1.0)
    # A class never predicted scores 0 and still counts in the average.
    f1, acc = score_predictions([0, 0, 0], [0, 1, 2])
    assert acc == pytest.approx(1 / 3)
    assert f1 == pytest.approx((0.5 + 0.0 + 0.0) / 3)


def test_label_order_is_explicit_and_by_index():
    assert LABEL_ORDER == ("negative", "neutral", "positive")


def test_deterministic_batches_are_stable_and_complete():
    a = deterministic_batches(1000, seed=42)
    assert a == deterministic_batches(1000, seed=42)
    assert sorted(i for batch in a for i in batch) == list(range(1000))
    assert all(len(b) <= BATCH_SIZE for b in a)
    assert sum(len(b) for b in a) == 1000  # drop_last is False


def test_batch_order_changes_with_the_seed():
    assert deterministic_batches(1000, seed=1) != deterministic_batches(1000, seed=2)


def test_batch_order_ignores_the_ambient_global_rng():
    import random

    random.seed(1)
    first = deterministic_batches(500, seed=7)
    random.seed(999)
    assert deterministic_batches(500, seed=7) == first


def real_bound(count, hidden, role, pathway=SystemPathway.VANILLA, **overrides):
    """A `BoundRepresentations` over a real torch tensor."""
    import torch

    key = sample_key(role=role, pathway=pathway, count=count, hidden_size=hidden,
                     **overrides)
    return BoundRepresentations(values=torch.randn(count, hidden), key=key)


@requires_torch
def test_train_head_runs_every_epoch_on_correctly_bound_representations():
    from unmark.evaluation.preg1_head import train_head

    train = real_bound(40, 6, Preg1Role.PROTOCOL_TRAIN)
    dev = real_bound(12, 6, Preg1Role.PROTOCOL_DEV)
    run = train_head(
        train, [i % 3 for i in range(40)], dev, [i % 3 for i in range(12)],
        learning_rate=1e-3, seed=5, epochs=EPOCHS,
    )
    assert len(run.scores) == EPOCHS
    assert [s.epoch for s in run.scores] == list(range(1, EPOCHS + 1))
    assert run.pathway is SystemPathway.VANILLA  # taken from provenance
    assert run.scored_on is Preg1Role.PROTOCOL_DEV


@requires_torch
def test_train_head_refuses_a_mutually_consistent_non_fp32_pair():
    """The FP32 rule, isolated from the geometry-agreement rule.

    C24-1-R1 finding 1: the previous fixture made train float64 and dev
    float32, which trips `require_same_geometry` first — so the test proved the
    *agreement* rule, not the FP32 rule it claimed. Both sets are now float64
    and mutually consistent, so geometry agreement passes and the failure can
    only come from the pre-G1 contract's absolute FP32 requirement.
    """
    import torch
    from unmark.evaluation.preg1_head import require_training_roles, train_head

    train_key = sample_key(role=Preg1Role.PROTOCOL_TRAIN, count=10, hidden_size=4,
                           dtype="torch.float64")
    dev_key = sample_key(role=Preg1Role.PROTOCOL_DEV, count=4, hidden_size=4,
                         dtype="torch.float64")
    train = BoundRepresentations(torch.randn(10, 4, dtype=torch.float64), train_key)
    dev = BoundRepresentations(torch.randn(4, 4, dtype=torch.float64), dev_key)

    # The pair agrees on geometry — this must NOT raise, or the test is not isolated.
    require_training_roles(train, dev)

    with pytest.raises(EvaluationContractViolation, match="FP32"):
        train_head(train, [0] * 10, dev, [0] * 4, learning_rate=1e-3, seed=1, epochs=1)


@requires_torch
def test_bound_representations_refuse_a_gradient_path():
    import torch

    key = sample_key(count=10, hidden_size=4)
    with pytest.raises(EvaluationContractViolation, match="detached"):
        BoundRepresentations(torch.randn(10, 4, requires_grad=True), key)


@requires_torch
def test_measurement_path_works_on_official_validation_bound_features():
    from unmark.evaluation.preg1_head import build_head, score_measurement

    head = build_head(6, seed=1)
    measurement = real_bound(9, 6, Preg1Role.OFFICIAL_VALIDATION)
    f1, acc = score_measurement(head, measurement, [i % 3 for i in range(9)])
    assert 0.0 <= f1 <= 1.0 and 0.0 <= acc <= 1.0


# ===========================================================================
# G. LR selection
# ===========================================================================
def test_primary_selection_rejects_base_only_runs():
    with pytest.raises(SplitLeakage, match="VANILLA runs only"):
        LrCandidate(1e-3, (flat_run(SystemPathway.BASE_ONLY, 1e-3, TUNING_SEEDS[0]),))


def test_grid_must_be_the_precommitted_one():
    partial = full_grid()[:3]
    with pytest.raises(EvaluationContractViolation, match="precommitted"):
        select_learning_rate(partial)


def test_tuning_seeds_must_be_the_precommitted_ones():
    candidates = full_grid()
    bad = LrCandidate(
        LR_GRID[0], tuple(flat_run(SystemPathway.VANILLA, LR_GRID[0], s) for s in (1, 2, 3))
    )
    with pytest.raises(EvaluationContractViolation, match="expected"):
        select_learning_rate([bad] + candidates[1:])


def test_lr_winner_prefers_mean_macro_f1():
    winner = select_learning_rate(full_grid({3e-3: 0.9}))
    assert winner.learning_rate == 3e-3


def test_lr_ties_break_on_mean_accuracy():
    candidates = [
        LrCandidate(lr, tuple(
            flat_run(SystemPathway.VANILLA, lr, s, 0.5, 0.9 if lr == 1e-2 else 0.4)
            for s in TUNING_SEEDS))
        for lr in LR_GRID
    ]
    assert select_learning_rate(candidates).learning_rate == 1e-2


def test_lr_ties_break_on_lower_sample_sd_then_smaller_lr():
    def spread(lr, values):
        return LrCandidate(lr, tuple(
            flat_run(SystemPathway.VANILLA, lr, s, v, 0.5)
            for s, v in zip(TUNING_SEEDS, values)))

    candidates = [
        spread(1e-4, [0.4, 0.5, 0.6]),   # same mean, wider spread
        spread(3e-4, [0.49, 0.5, 0.51]),  # same mean, tighter spread -> wins
        *[LrCandidate(lr, tuple(flat_run(SystemPathway.VANILLA, lr, s, 0.1, 0.1)
                                for s in TUNING_SEEDS)) for lr in LR_GRID[2:]],
    ]
    assert select_learning_rate(candidates).learning_rate == 3e-4


def test_total_tie_selects_the_smallest_lr():
    assert select_learning_rate(full_grid()).learning_rate == min(LR_GRID)


def test_lr_selection_does_not_depend_on_candidate_order():
    candidates = full_grid()
    assert (
        select_learning_rate(list(reversed(candidates))).learning_rate
        == select_learning_rate(candidates).learning_rate
    )


def test_sample_standard_deviation_is_used_not_population():
    values = [0.4, 0.5, 0.6]
    assert sample_stdev(values) == pytest.approx(statistics.stdev(values))
    assert sample_stdev(values) != pytest.approx(statistics.pstdev(values))
    assert sample_stdev([0.5]) == 0.0


def test_frozen_lr_must_come_from_vanilla_and_the_grid():
    frozen = freeze_learning_rate(select_learning_rate(full_grid({1e-2: 0.9})))
    assert frozen.value == 1e-2
    assert frozen.selected_on is SystemPathway.VANILLA
    assert frozen.require_primary("ctx") is frozen
    with pytest.raises(EvaluationContractViolation, match="not in the precommitted grid"):
        FrozenLearningRate(value=5e-3)


def test_a_base_only_lr_is_representable_but_never_passes_as_the_primary():
    """The secondary own-LR sensitivity needs a Base-only-selected LR to EXIST.

    Audit 027 relaxed the constructor, which by itself would have removed the
    guarantee that a Base-only LR cannot be used as the shared one. The
    guarantee now lives in `require_primary`, which every primary consumer
    calls, so this test is what keeps the relaxation from being a hole.
    """
    secondary = FrozenLearningRate(value=1e-3, selected_on=SystemPathway.BASE_ONLY)
    assert secondary.selected_on is SystemPathway.BASE_ONLY
    with pytest.raises(SplitLeakage, match="must not replace the primary"):
        secondary.require_primary("the primary paired measurement")

    # A pathway that is neither arm remains unrepresentable.
    other = next(
        (p for p in SystemPathway
         if p not in (SystemPathway.VANILLA, SystemPathway.BASE_ONLY)),
        None,
    )
    if other is not None:
        with pytest.raises(SplitLeakage):
            FrozenLearningRate(value=1e-3, selected_on=other)


def test_freeze_carries_the_selecting_pathway_so_the_two_cannot_be_confused():
    primary = freeze_learning_rate(select_learning_rate(full_grid({1e-2: 0.9})))
    assert primary.selected_on is SystemPathway.VANILLA

    base_runs = full_grid({3e-3: 0.8}, pathway=SystemPathway.BASE_ONLY)
    secondary = freeze_learning_rate(
        select_learning_rate(base_runs)
    )
    assert secondary.value == 3e-3
    assert secondary.selected_on is SystemPathway.BASE_ONLY, (
        "a Base-only sweep must not freeze as though Vanilla had selected it"
    )


# ===========================================================================
# H. Paired final diagnostic
# ===========================================================================
def paired(**overrides):
    results = tuple(
        PairedSeedResult(seed=s, vanilla_macro_f1=0.70, vanilla_accuracy=0.80,
                         base_only_macro_f1=0.60, base_only_accuracy=0.72)
        for s in MEASUREMENT_SEEDS
    )
    kwargs = dict(learning_rate=FrozenLearningRate(1e-3), results=results)
    kwargs.update(overrides)
    return PairedDiagnostic(**kwargs)


def test_measurement_uses_exactly_the_five_precommitted_seeds():
    assert len(MEASUREMENT_SEEDS) == 5
    report = paired().to_dict()
    assert [r["seed"] for r in report["per_seed"]] == list(MEASUREMENT_SEEDS)


def test_wrong_measurement_seeds_are_rejected():
    bad = tuple(
        PairedSeedResult(s, 0.7, 0.8, 0.6, 0.7) for s in (1, 2, 3, 4, 5)
    )
    with pytest.raises(EvaluationContractViolation, match="precommitted seeds"):
        PairedDiagnostic(learning_rate=FrozenLearningRate(1e-3), results=bad)


def test_measurement_is_reported_on_official_validation_only():
    with pytest.raises(EvaluationContractViolation, match="official validation"):
        paired(measured_on=Preg1Role.PROTOCOL_DEV)


def test_report_contains_raw_scores_deltas_means_and_sample_sds():
    report = paired().to_dict()
    row = report["per_seed"][0]
    for key in ("vanilla_macro_f1", "base_only_macro_f1", "macro_f1_delta", "accuracy_delta"):
        assert key in row
    for arm in ("vanilla", "base_only", "delta_vanilla_minus_base_only"):
        assert "mean_macro_f1" in report[arm]
        assert "sample_stdev_macro_f1" in report[arm]
    assert report["delta_vanilla_minus_base_only"]["mean_macro_f1"] == pytest.approx(0.10)


def test_delta_is_vanilla_minus_base_only():
    result = PairedSeedResult(MEASUREMENT_SEEDS[0], 0.7, 0.8, 0.6, 0.75)
    assert result.macro_f1_delta == pytest.approx(0.1)
    assert result.accuracy_delta == pytest.approx(0.05)


def test_report_carries_no_p_value_or_significance_threshold():
    blob = json.dumps(paired().to_dict())
    payload = json.loads(blob)

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    found = set(keys(payload))
    assert not found & {
        "p_value", "pvalue", "significant", "significance", "threshold",
        "confidence_interval", "ci_low", "ci_high", "test_statistic", "verdict", "passed",
    }
    assert "no significance threshold" in NO_SIGNIFICANCE_TEST.lower()


def test_report_states_what_determinism_is_not_guaranteed():
    assert "NOT GUARANTEED" in DETERMINISM_SCOPE
    assert "hardware" in DETERMINISM_SCOPE
    assert paired().to_dict()["determinism"] == DETERMINISM_SCOPE


def test_report_is_json_serialisable_and_schema_stamped():
    report = paired().to_dict()
    json.dumps(report)
    assert report["schema_version"] == PREG1_HEAD_SCHEMA_VERSION == "preg1-head-v1"


# ===========================================================================
# I. Cache provenance
# ===========================================================================
def test_identical_metadata_is_compatible():
    sample_key().require_compatible(sample_key())


@pytest.mark.parametrize(
    "override",
    [
        {"pathway": SystemPathway.BASE_ONLY},
        {"role": Preg1Role.PROTOCOL_DEV},
        {"model_revision": "0" * 40},
        {"max_length": 128},
        {"pooling": "MEAN"},
        {"dtype": "torch.float16"},
        {"truncation": False},
        {"padding": "longest"},
        {"hidden_size": 16},
        {"count": 3},
        {"source_identity": "b" * 64},
        {"tokenizer_id": "other/model"},
        {"dataset_version": "2.0"},
        {"schema_version": "preg1-head-v0"},
    ],
)
def test_incompatible_metadata_fails_closed(override):
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        sample_key(**override).require_compatible(sample_key())


def test_a_vanilla_cache_cannot_be_reused_as_base_only():
    """The single most dangerous reuse: it would report a burden of zero."""
    with pytest.raises(EvaluationContractViolation, match="pathway"):
        sample_key(pathway=SystemPathway.BASE_ONLY).require_compatible(
            sample_key(pathway=SystemPathway.VANILLA)
        )


def test_reordered_ids_change_the_identity():
    a = ordered_id_digest(["x", "y"])
    b = ordered_id_digest(["y", "x"])
    assert a != b
    with pytest.raises(EvaluationContractViolation, match="ordered_id_digest"):
        sample_key(ordered_id_digest=a).require_compatible(sample_key(ordered_id_digest=b))


def test_cache_metadata_contains_no_raw_text():
    payload = sample_key().to_dict()
    blob = json.dumps(payload)
    marked = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    assert not set(blob.lower()) & marked
    assert "text" not in payload


def test_cache_metadata_round_trips():
    key = sample_key()
    assert RepresentationKey.from_dict(json.loads(json.dumps(key.to_dict()))) == key


def test_malformed_cache_metadata_fails():
    with pytest.raises(EvaluationContractViolation, match="malformed"):
        RepresentationKey.from_dict({"dataset": "UIT-VSFC"})


def test_unknown_enum_value_in_cache_metadata_fails():
    payload = sample_key().to_dict()
    payload["pathway"] = "RESTORE"
    with pytest.raises(EvaluationContractViolation, match="malformed"):
        RepresentationKey.from_dict(payload)


def test_missing_cache_metadata_fails(tmp_path):
    with pytest.raises(EvaluationContractViolation, match="no cache metadata"):
        RepresentationCache(tmp_path / "absent").read_key()


def test_unparseable_cache_metadata_fails(tmp_path):
    directory = tmp_path / "c"
    directory.mkdir()
    (directory / RepresentationCache.METADATA_NAME).write_text("{oops", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="not valid JSON"):
        RepresentationCache(directory).read_key()


@requires_torch
def test_cache_round_trip_preserves_values_and_stays_bound(tmp_path):
    """Round-trip fidelity, through the **bound** API.

    C24-1-R1 finding 2: this previously read
    `torch.equal(cache.load(...), tensor)`, which encoded the pre-Revision-1
    bare-Tensor return. `load` deliberately returns `BoundRepresentations` now,
    so that assertion was stale, not a defect. Reverting the API would undo the
    provenance binding, so the test moved instead.
    """
    import torch

    cache = RepresentationCache(tmp_path / "c")
    key = sample_key()
    tensor = torch.randn(key.count, key.hidden_size)
    cache.save(key, tensor)
    assert cache.exists()

    loaded = cache.load(sample_key())
    assert not isinstance(loaded, torch.Tensor), "no bare tensor may leave the cache"
    assert isinstance(loaded, BoundRepresentations)
    assert loaded.key == key
    assert loaded.pathway is key.pathway
    assert loaded.key.source_identity == key.source_identity
    assert loaded.key.ordered_id_digest == key.ordered_id_digest
    assert torch.equal(loaded.values, tensor)

    # A second load is equally bound -- the binding is not a one-shot wrapper.
    assert torch.equal(cache.load(sample_key()).values, tensor)


@requires_torch
def test_cache_refuses_a_reload_under_a_different_pathway(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    cache.save(sample_key(), torch.randn(2, 8))
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        cache.load(sample_key(pathway=SystemPathway.BASE_ONLY))


@requires_torch
def test_cache_refuses_non_fp32(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    with pytest.raises(EvaluationContractViolation, match="FP32"):
        cache.save(sample_key(), torch.randn(2, 8, dtype=torch.float64))


@requires_torch
def test_cache_refuses_a_shape_that_contradicts_its_key(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    with pytest.raises(EvaluationContractViolation, match="does not match"):
        cache.save(sample_key(), torch.randn(5, 8))


# ===========================================================================
# Module hygiene
# ===========================================================================
def test_no_unmark_adapter_concepts_leak_into_the_diagnostic():
    """This is a burden probe, not an UNMARK experiment."""
    tree = ast.parse((REPO / "unmark/evaluation/preg1_head.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
    assert not any(m.startswith("unmark.modeling") for m in modules)
    assert not any(m.startswith("unmark.stage1") for m in modules)
    assert not any(m.startswith("unmark.alignment") for m in modules)


def test_scientific_literals_are_imported_not_restated():
    tree = ast.parse((REPO / "unmark/evaluation/preg1_head.py").read_text(encoding="utf-8"))
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    forbidden = {30, 128, 256, 0.01, 1e-8, 0.9, 0.999, 17486, 5509, 19422, 11800,
                 53148, 59945, 42941, 720, 9428}
    assert not numbers & forbidden, f"restated locked values: {sorted(numbers & forbidden)}"


def test_head_run_records_which_role_it_was_scored_on():
    run = flat_run(SystemPathway.VANILLA, 1e-3, TUNING_SEEDS[0])
    assert run.scored_on is Preg1Role.PROTOCOL_DEV
    assert run.to_dict()["scored_on"] == "protocol-dev"


def test_a_run_scored_on_official_validation_cannot_be_constructed():
    with pytest.raises(SplitLeakage, match="cannot drive selection"):
        HeadRun(
            pathway=SystemPathway.VANILLA, learning_rate=1e-3, seed=TUNING_SEEDS[0],
            scores=tuple(EpochScore(e, 0.5, 0.5) for e in range(1, EPOCHS + 1)),
            hidden_size=8, scored_on=Preg1Role.OFFICIAL_VALIDATION,
        )


def test_lr_selection_rejects_runs_not_scored_on_protocol_dev():
    """Defence in depth: the guard holds at the candidate layer too."""
    run = flat_run(SystemPathway.VANILLA, 1e-3, TUNING_SEEDS[0])
    smuggled = HeadRun.__new__(HeadRun)
    object.__setattr__(smuggled, "pathway", SystemPathway.VANILLA)
    object.__setattr__(smuggled, "learning_rate", 1e-3)
    object.__setattr__(smuggled, "seed", TUNING_SEEDS[0])
    object.__setattr__(smuggled, "scores", run.scores)
    object.__setattr__(smuggled, "hidden_size", 8)
    object.__setattr__(smuggled, "scored_on", Preg1Role.OFFICIAL_VALIDATION)
    with pytest.raises(SplitLeakage, match="protocol-dev"):
        LrCandidate(1e-3, (smuggled,))


# ===========================================================================
# J. Provenance binding — the role travels with the tensor (Finding B)
# ===========================================================================
def test_a_bound_representation_takes_its_role_from_provenance():
    rep = bound(role=Preg1Role.PROTOCOL_DEV, pathway=SystemPathway.BASE_ONLY)
    assert rep.role is Preg1Role.PROTOCOL_DEV
    assert rep.pathway is SystemPathway.BASE_ONLY
    assert len(rep) == rep.key.count


def test_there_is_no_argument_by_which_a_role_can_be_declared():
    """The repair, stated structurally: the trainer has no role parameter.

    Previously `train_head(dev_role=...)` was a claim *about* a tensor rather
    than a property *of* it, so official-validation features could be presented
    as protocol-dev and every guard would pass.
    """
    import inspect
    from unmark.evaluation.preg1_head import train_head

    parameters = set(inspect.signature(train_head).parameters)
    assert "dev_role" not in parameters
    assert "role" not in parameters
    assert "pathway" not in parameters
    assert {"train", "dev"} <= parameters


def test_training_representations_must_be_protocol_train():
    with pytest.raises(SplitLeakage, match="head training requires"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_DEV), bound(role=Preg1Role.PROTOCOL_DEV)
        )


def test_official_validation_cannot_be_used_for_training():
    with pytest.raises(SplitLeakage, match="head training requires"):
        require_training_roles(
            bound(role=Preg1Role.OFFICIAL_VALIDATION),
            bound(role=Preg1Role.PROTOCOL_DEV),
        )


def test_checkpoint_representations_must_be_protocol_dev():
    with pytest.raises(SplitLeakage, match="checkpoint selection requires"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_TRAIN), bound(role=Preg1Role.PROTOCOL_TRAIN)
        )


def test_official_validation_cannot_drive_checkpoint_selection():
    """The substitution that would invalidate the diagnostic silently."""
    with pytest.raises(SplitLeakage, match="checkpoint selection requires"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_TRAIN),
            bound(role=Preg1Role.OFFICIAL_VALIDATION),
        )


def test_a_correctly_bound_pair_passes():
    require_training_roles(
        bound(role=Preg1Role.PROTOCOL_TRAIN), bound(role=Preg1Role.PROTOCOL_DEV)
    )


def test_training_refuses_a_pathway_mismatch():
    with pytest.raises(SplitLeakage, match="pathway mismatch"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_TRAIN, pathway=SystemPathway.VANILLA),
            bound(role=Preg1Role.PROTOCOL_DEV, pathway=SystemPathway.BASE_ONLY),
        )


@pytest.mark.parametrize(
    "override", [{"model_revision": "0" * 40}, {"max_length": 128},
                 {"pooling": "MEAN"}, {"tokenizer_id": "other/model"},
                 {"padding": "longest"}, {"truncation": False},
                 {"dtype": "torch.float64"}]
)
def test_training_refuses_representation_sets_from_different_geometry(override):
    """Includes a train/dev **dtype** mismatch, which must fail closed as a
    geometry disagreement — the property the old FP32 test was accidentally
    proving. Both rules now have their own test."""
    with pytest.raises(EvaluationContractViolation, match="disagree on"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_TRAIN),
            bound(role=Preg1Role.PROTOCOL_DEV, **override),
        )


def test_training_refuses_a_hidden_size_mismatch():
    with pytest.raises(EvaluationContractViolation, match="hidden size mismatch"):
        require_training_roles(
            bound(role=Preg1Role.PROTOCOL_TRAIN),
            bound(role=Preg1Role.PROTOCOL_DEV, hidden_size=16),
        )


def test_measurement_rejects_protocol_dev_bound_representations():
    rep = bound(role=Preg1Role.PROTOCOL_DEV)
    with pytest.raises(SplitLeakage, match="primary paired measurement requires"):
        rep.require_role(Preg1Role.OFFICIAL_VALIDATION, "the primary paired measurement")


def test_measurement_rejects_protocol_train_bound_representations():
    rep = bound(role=Preg1Role.PROTOCOL_TRAIN)
    with pytest.raises(SplitLeakage, match="primary paired measurement requires"):
        rep.require_role(Preg1Role.OFFICIAL_VALIDATION, "the primary paired measurement")


def test_a_tensor_cannot_be_reinterpreted_by_rebinding_a_contradictory_key():
    """Rebinding is not a loophole: the key must still describe the values."""
    values = FakeMatrix(2, 8)
    with pytest.raises(EvaluationContractViolation, match="contradicts its key"):
        BoundRepresentations(values, sample_key(count=99))
    with pytest.raises(EvaluationContractViolation, match="contradicts its key"):
        BoundRepresentations(values, sample_key(hidden_size=99))
    with pytest.raises(EvaluationContractViolation, match="dtype"):
        BoundRepresentations(FakeMatrix(2, 8, dtype="torch.float16"), sample_key())


def test_rebinding_to_another_role_still_requires_a_matching_source_identity():
    """A relabel is detectable wherever the source identity is pinned.

    Rebinding the same values under a different role produces a key whose
    `role` and `source_identity` no longer agree with the cache that produced
    it, so a cache reload fails -- which is where the supported path checks.
    """
    original = sample_key(role=Preg1Role.PROTOCOL_DEV)
    relabelled = sample_key(role=Preg1Role.OFFICIAL_VALIDATION)
    with pytest.raises(EvaluationContractViolation, match="role"):
        relabelled.require_compatible(original)


@requires_torch
def test_cache_load_returns_a_bound_object_preserving_the_validated_key(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    key = sample_key(role=Preg1Role.PROTOCOL_DEV)
    tensor = torch.randn(key.count, key.hidden_size)
    cache.save(key, tensor)

    loaded = cache.load(sample_key(role=Preg1Role.PROTOCOL_DEV))
    assert isinstance(loaded, BoundRepresentations)
    assert loaded.role is Preg1Role.PROTOCOL_DEV
    assert loaded.key == key
    assert torch.equal(loaded.values, tensor)


@requires_torch
def test_a_protocol_dev_cache_cannot_be_loaded_as_official_validation(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    cache.save(sample_key(role=Preg1Role.PROTOCOL_DEV), torch.randn(2, 8))
    with pytest.raises(EvaluationContractViolation, match="role"):
        cache.load(sample_key(role=Preg1Role.OFFICIAL_VALIDATION))


@requires_torch
def test_an_official_validation_cache_cannot_be_loaded_as_protocol_dev(tmp_path):
    import torch

    cache = RepresentationCache(tmp_path / "c")
    cache.save(sample_key(role=Preg1Role.OFFICIAL_VALIDATION), torch.randn(2, 8))
    with pytest.raises(EvaluationContractViolation, match="role"):
        cache.load(sample_key(role=Preg1Role.PROTOCOL_DEV))


@requires_torch
def test_fresh_extraction_returns_the_same_bound_shape_as_a_cache_load():
    import torch
    from unmark.evaluation.preg1_head import extract_bound_representations

    encoder = make_fake_encoder()
    ids = torch.randint(0, 50, (2, 5))
    key = sample_key(role=Preg1Role.PROTOCOL_TRAIN, count=2, hidden_size=8)
    fresh = extract_bound_representations(encoder, ids, torch.ones_like(ids), key)
    assert isinstance(fresh, BoundRepresentations)
    assert fresh.role is Preg1Role.PROTOCOL_TRAIN
    assert fresh.values.dtype is torch.float32


@requires_torch
def test_the_full_supported_training_and_measurement_path(tmp_path):
    """End to end on synthetic data: train on PROTOCOL_TRAIN, select on
    PROTOCOL_DEV, measure on OFFICIAL_VALIDATION."""
    from unmark.evaluation.preg1_head import score_measurement, train_head

    train = real_bound(30, 6, Preg1Role.PROTOCOL_TRAIN)
    dev = real_bound(9, 6, Preg1Role.PROTOCOL_DEV)
    run = train_head(
        train, [i % 3 for i in range(30)], dev, [i % 3 for i in range(9)],
        learning_rate=1e-3, seed=MEASUREMENT_SEEDS[0], epochs=EPOCHS,
    )
    assert run.selected.epoch in range(1, EPOCHS + 1)

    from unmark.evaluation.preg1_head import build_head

    head = build_head(6, seed=MEASUREMENT_SEEDS[0])
    measurement = real_bound(9, 6, Preg1Role.OFFICIAL_VALIDATION)
    f1, acc = score_measurement(head, measurement, [i % 3 for i in range(9)])
    assert 0.0 <= f1 <= 1.0 and 0.0 <= acc <= 1.0


@requires_torch
def test_score_measurement_refuses_protocol_dev_features():
    from unmark.evaluation.preg1_head import build_head, score_measurement

    head = build_head(6, seed=1)
    with pytest.raises(SplitLeakage, match="primary paired measurement"):
        score_measurement(head, real_bound(9, 6, Preg1Role.PROTOCOL_DEV), [0] * 9)


def test_cache_load_is_annotated_to_return_bound_representations():
    """Structural guard against reverting the provenance binding.

    C24-1-R1 finding 2 was a stale test, not a defect — but the tempting "fix"
    was to make `load` return a bare tensor again, which would undo Revision 1.
    This runs without torch, so the regression is caught in the local suite.
    """
    import ast

    tree = ast.parse((REPO / "unmark/evaluation/preg1_head.py").read_text(encoding="utf-8"))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load"
    ]
    assert len(loads) == 1
    assert ast.unparse(loads[0].returns) == "BoundRepresentations"
    returns = [n for n in ast.walk(loads[0]) if isinstance(n, ast.Return)]
    assert returns and all(
        isinstance(r.value, ast.Call)
        and isinstance(r.value.func, ast.Name)
        and r.value.func.id == "BoundRepresentations"
        for r in returns
    )


# ===========================================================================
# I. SECONDARY OWN-LR SENSITIVITY (Audit 027)
#
# The same delta/aggregation arithmetic serves two report shapes. These tests
# pin that the shapes stay distinguishable, that the primary shape is byte-for
# -byte what the completed primary run emitted, and that a secondary result
# cannot pass itself off as the primary.
# ===========================================================================
def test_a_base_only_candidate_must_be_declared_and_stays_single_pathway():
    base_runs = tuple(
        flat_run(SystemPathway.BASE_ONLY, 1e-3, seed, 0.6, 0.7) for seed in TUNING_SEEDS
    )
    # declaring the pathway is what makes Base-only aggregation legal at all
    candidate = LrCandidate(1e-3, base_runs, pathway=SystemPathway.BASE_ONLY)
    assert candidate.pathway is SystemPathway.BASE_ONLY
    assert candidate.to_dict()["pathway"] == "BASE_ONLY"

    # the default is still VANILLA, so the primary cannot absorb Base-only runs
    with pytest.raises(SplitLeakage, match="VANILLA runs only"):
        LrCandidate(1e-3, base_runs)

    # and a declared Base-only candidate refuses Vanilla runs, symmetrically
    vanilla_runs = tuple(
        flat_run(SystemPathway.VANILLA, 1e-3, seed, 0.6, 0.7) for seed in TUNING_SEEDS
    )
    with pytest.raises(SplitLeakage, match="BASE_ONLY runs only"):
        LrCandidate(1e-3, vanilla_runs, pathway=SystemPathway.BASE_ONLY)


def test_the_primary_report_shape_is_unchanged_by_the_secondary_feature():
    """Audit 027 must not alter what the completed primary run emits."""
    report = paired().to_dict()
    assert "analysis" not in report
    assert "base_only_learning_rate" not in report
    assert "vanilla_learning_rate" not in report
    assert "secondary_caveat" not in report
    assert report["learning_rate"]["selected_on"] == "VANILLA"


def secondary(**overrides):
    kwargs = dict(
        base_only_learning_rate=FrozenLearningRate(
            3e-3, selected_on=SystemPathway.BASE_ONLY
        )
    )
    kwargs.update(overrides)
    return paired(**kwargs)


def test_the_secondary_report_names_itself_and_carries_both_learning_rates():
    from unmark.evaluation.preg1_protocol import SECONDARY_ANALYSIS_LABEL

    report = secondary().to_dict()
    assert report["analysis"] == SECONDARY_ANALYSIS_LABEL == "SECONDARY OWN-LR SENSITIVITY"
    assert report["vanilla_learning_rate"]["learning_rate"] == 1e-3
    assert report["vanilla_learning_rate"]["selected_on"] == "VANILLA"
    assert report["base_only_learning_rate"]["learning_rate"] == 3e-3
    assert report["base_only_learning_rate"]["selected_on"] == "BASE_ONLY"
    # the precommitment travels with the result
    assert "MUST NOT replace" in report["secondary_caveat"]
    assert "does NOT make Vanilla an upper bound" in report["primary_lr_caveat"]
    # still descriptive: no significance machinery in either shape
    assert not {"p_value", "threshold", "passed"} & set(report)


def test_the_secondary_pairs_the_primary_vanilla_lr_against_a_base_only_lr():
    # the Vanilla comparator must itself be a primary (VANILLA) selection
    with pytest.raises(SplitLeakage, match="must not replace the primary"):
        paired(learning_rate=FrozenLearningRate(1e-3, selected_on=SystemPathway.BASE_ONLY))

    # and the retuned arm must actually have been selected on BASE_ONLY
    with pytest.raises(SplitLeakage, match="pairs the primary Vanilla LR"):
        secondary(base_only_learning_rate=FrozenLearningRate(3e-3))


def test_the_secondary_may_not_widen_the_grid_or_the_tuning_budget():
    widened = FrozenLearningRate(
        3e-3, selected_on=SystemPathway.BASE_ONLY, grid=tuple(LR_GRID) + (5e-2,)
    )
    with pytest.raises(EvaluationContractViolation, match="same precommitted grid"):
        secondary(base_only_learning_rate=widened)

    extra_seeds = FrozenLearningRate(
        3e-3, selected_on=SystemPathway.BASE_ONLY,
        tuning_seeds=tuple(TUNING_SEEDS) + (1234,),
    )
    with pytest.raises(EvaluationContractViolation, match="same three tuning seeds"):
        secondary(base_only_learning_rate=extra_seeds)


def test_both_shapes_share_one_delta_implementation():
    """The arithmetic is identical; only the labelling differs."""
    primary_report = paired().to_dict()
    secondary_report = secondary().to_dict()
    for section in ("vanilla", "base_only", "delta_vanilla_minus_base_only", "per_seed"):
        assert primary_report[section] == secondary_report[section]
    assert primary_report["delta_vanilla_minus_base_only"]["mean_macro_f1"] == pytest.approx(0.10)
