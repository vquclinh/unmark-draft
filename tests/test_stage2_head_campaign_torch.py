"""Tensor-level Stage-2 head-campaign behaviour: init pairing, optimiser, schedule.

Needs **real torch**. Follows the marker convention of
`test_stage2_dual_finalist_infra.py` -- a per-test `skipif` rather than a
module-level `importorskip` -- so the tests stay *collected* and their skips are
visible in the report instead of the whole file vanishing.

Nothing here trains on UIT-VSFC, reads a downstream row, or loads PhoBERT: every
fixture is a synthetic `[N, 768]` FP32 tensor. That is the point of the cached
design -- once representations exist, head training touches no model.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import build_head, build_optimizer  # noqa: E402
from unmark.evaluation.preg1_protocol import EPOCHS, MEASUREMENT_SEEDS  # noqa: E402
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    Stage2UnmarkArm,
    finalist_for_arm,
)
from unmark.evaluation.stage2_head_campaign import (  # noqa: E402
    STAGE2_CAMPAIGN_SEEDS,
    STAGE2_CLEAN_CONDITION,
    STAGE2_HEAD_LEARNING_RATE,
    STAGE2_MEASUREMENT_ROLE,
    STAGE2_PROTOCOL_VERSION,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    Stage2BoundRepresentations,
    Stage2HeadRunStore,
    Stage2RepresentationCache,
    Stage2RepresentationKey,
    build_stage2_head_artifact,
    label_digest,
    measure_stage2_head,
    require_head_only_optimizer,
    train_stage2_head,
    validate_stage2_head_artifact,
)
from unmark.evaluation.preg1_head import ordered_id_digest  # noqa: E402
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE  # noqa: E402

try:  # pragma: no cover - depends on the environment
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - the normal ML-free path
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(
    not TORCH, reason="torch is not installed locally; tensor checks run where available"
)

HEAD_SHA = "a" * 40
N_TRAIN, N_DEV = 40, 24


def make_key(*, arm="UNMARK-A", role, condition=STAGE2_CLEAN_CONDITION,
             corruption_seed=None, count, ids, labels):
    return Stage2RepresentationKey(
        repository_head=HEAD_SHA,
        arm=arm,
        finalist_checkpoint_sha256=finalist_for_arm(arm).checkpoint_sha256,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=STAGE2_PROTOCOL_VERSION,
        dataset="UIT-VSFC",
        dataset_version="1.0",
        task="sentiment",
        role=role.value,
        condition=condition,
        corruption_seed=corruption_seed,
        pooling=STAGE2_FIRST_TOKEN_POOLING,
        max_length=256,
        truncation=True,
        padding="max_length",
        ordered_id_digest=ordered_id_digest(ids),
        label_digest=label_digest(labels),
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=count,
    )


def synthetic(count, *, seed, arm="UNMARK-A", role, condition=STAGE2_CLEAN_CONDITION,
              corruption_seed=None):
    """A separable synthetic split: labels are recoverable, so training moves."""
    generator = torch.Generator().manual_seed(seed)
    labels = [i % 3 for i in range(count)]
    values = torch.randn(count, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    for i, label in enumerate(labels):
        values[i, label] += 6.0
    ids = [f"s{i:04d}" for i in range(count)]
    key = make_key(arm=arm, role=role, condition=condition,
                   corruption_seed=corruption_seed, count=count, ids=ids, labels=labels)
    return Stage2BoundRepresentations(values=values, key=key), labels


# ==========================================================================================
# B. Head init and A/B pairing
# ==========================================================================================

@requires_torch
def test_matched_arms_at_the_same_seed_get_bit_identical_heads():
    for seed in STAGE2_CAMPAIGN_SEEDS:
        a = build_head(HIDDEN_SIZE, seed)
        b = build_head(HIDDEN_SIZE, seed)
        for name, tensor in a.state_dict().items():
            assert torch.equal(tensor, b.state_dict()[name]), name


@requires_torch
def test_the_pairing_survives_intervening_rng_consumption():
    """Run order and ambient RNG must not change either arm's initial head."""
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    a = build_head(HIDDEN_SIZE, seed)
    torch.manual_seed(12345)
    _ = torch.randn(1000)
    b = build_head(HIDDEN_SIZE, seed)
    for name, tensor in a.state_dict().items():
        assert torch.equal(tensor, b.state_dict()[name]), name


@requires_torch
def test_different_seeds_give_different_initialisations():
    a = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    b = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[1])
    assert not torch.equal(a.weight, b.weight)


@requires_torch
def test_the_bias_starts_at_zero_and_the_weight_does_not():
    head = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    assert torch.equal(head.bias, torch.zeros_like(head.bias))
    assert not torch.equal(head.weight, torch.zeros_like(head.weight))


@requires_torch
def test_no_weight_is_shared_between_two_arms_heads():
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    a, b = build_head(HIDDEN_SIZE, seed), build_head(HIDDEN_SIZE, seed)
    assert a.weight is not b.weight
    with torch.no_grad():
        a.weight.add_(1.0)
    assert not torch.equal(a.weight, b.weight)


# ==========================================================================================
# C. Optimiser isolation
# ==========================================================================================

@requires_torch
def test_only_head_parameters_reach_the_optimiser():
    head = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    optimizer = build_optimizer(head, STAGE2_HEAD_LEARNING_RATE)
    require_head_only_optimizer(optimizer, head)
    optimised = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(optimised) == 2
    assert {id(p) for p in optimised} == {id(p) for p in head.parameters()}


@requires_torch
def test_a_foreign_parameter_in_the_optimiser_is_refused():
    """Stands in for an encoder or adapter tensor leaking into a param group."""
    head = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    optimizer = build_optimizer(head, STAGE2_HEAD_LEARNING_RATE)
    foreign = torch.nn.Parameter(torch.zeros(4))
    optimizer.param_groups[0]["params"] = list(optimizer.param_groups[0]["params"]) + [foreign]
    with pytest.raises(EvaluationContractViolation, match="do not belong to the head"):
        require_head_only_optimizer(optimizer, head)


@requires_torch
def test_a_missing_head_parameter_is_refused():
    head = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    optimizer = build_optimizer(head, STAGE2_HEAD_LEARNING_RATE)
    optimizer.param_groups = optimizer.param_groups[:1]
    with pytest.raises(EvaluationContractViolation, match="absent from the optimiser"):
        require_head_only_optimizer(optimizer, head)


@requires_torch
def test_weight_decay_applies_to_the_weight_and_not_the_bias():
    head = build_head(HIDDEN_SIZE, STAGE2_CAMPAIGN_SEEDS[0])
    optimizer = build_optimizer(head, STAGE2_HEAD_LEARNING_RATE)
    decays = {id(g["params"][0]): g["weight_decay"] for g in optimizer.param_groups}
    assert decays[id(head.weight)] == 0.01
    assert decays[id(head.bias)] == 0.0
    for group in optimizer.param_groups:
        assert group["lr"] == STAGE2_HEAD_LEARNING_RATE


# ==========================================================================================
# D. Full schedule, no early stopping
# ==========================================================================================

@requires_torch
def test_a_run_completes_all_thirty_epochs_and_selects_afterwards():
    train, train_y = synthetic(N_TRAIN, seed=1, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=2, role=STAGE2_SELECTION_ROLE)
    run = train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])
    assert len(run.scores) == EPOCHS
    assert [s.epoch for s in run.scores] == list(range(1, EPOCHS + 1))
    assert 1 <= run.selected.epoch <= EPOCHS
    # The selected epoch may be early; the schedule still ran to completion.
    assert run.scores[-1].epoch == EPOCHS


@requires_torch
def test_the_selected_epoch_is_the_locked_total_order_over_the_history():
    from unmark.evaluation.preg1_head import select_checkpoint

    train, train_y = synthetic(N_TRAIN, seed=3, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=4, role=STAGE2_SELECTION_ROLE)
    run = train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])
    assert run.selected == select_checkpoint(list(run.scores))


@requires_torch
def test_two_arms_at_one_seed_are_trained_independently():
    """Same seed, different representations -> separate heads, no shared state."""
    a_train, a_y = synthetic(N_TRAIN, seed=5, arm="UNMARK-A", role=STAGE2_TRAINING_ROLE)
    a_dev, a_dy = synthetic(N_DEV, seed=6, arm="UNMARK-A", role=STAGE2_SELECTION_ROLE)
    b_train, b_y = synthetic(N_TRAIN, seed=7, arm="UNMARK-B", role=STAGE2_TRAINING_ROLE)
    b_dev, b_dy = synthetic(N_DEV, seed=8, arm="UNMARK-B", role=STAGE2_SELECTION_ROLE)
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    a = train_stage2_head(a_train, a_y, a_dev, a_dy, seed=seed)
    b = train_stage2_head(b_train, b_y, b_dev, b_dy, seed=seed)
    assert a.arm == "UNMARK-A" and b.arm == "UNMARK-B"
    assert a.initial_head_fingerprint == b.initial_head_fingerprint  # same seed
    assert a.selected_head_state_sha256 != b.selected_head_state_sha256  # different data


@requires_torch
def test_a_rerun_is_bit_identical_which_is_why_midrun_resume_is_unnecessary():
    train, train_y = synthetic(N_TRAIN, seed=9, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=10, role=STAGE2_SELECTION_ROLE)
    first = train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])
    second = train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])
    assert first.history_digest == second.history_digest
    assert first.selected_head_state_sha256 == second.selected_head_state_sha256


@requires_torch
def test_training_never_touches_an_encoder_or_adapter():
    """`train_stage2_head` takes tensors, never a pathway -- assert by signature."""
    import inspect

    parameters = set(inspect.signature(train_stage2_head).parameters)
    for banned in ("pathway", "encoder", "adapter", "model", "tokenizer"):
        assert banned not in parameters


# ==========================================================================================
# Frozen budget / seed guards
# ==========================================================================================

@requires_torch
@pytest.mark.parametrize(("kwargs", "match"), [
    ({"learning_rate": 0.001}, "LR is frozen"),
    ({"epochs": 10}, "epoch budget is frozen"),
])
def test_a_drifted_budget_is_refused(kwargs, match):
    train, train_y = synthetic(N_TRAIN, seed=11, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=12, role=STAGE2_SELECTION_ROLE)
    with pytest.raises(EvaluationContractViolation, match=match):
        train_stage2_head(train, train_y, dev, dev_y,
                          seed=STAGE2_CAMPAIGN_SEEDS[0], **kwargs)


@requires_torch
def test_an_unfrozen_seed_is_refused():
    train, train_y = synthetic(N_TRAIN, seed=13, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=14, role=STAGE2_SELECTION_ROLE)
    with pytest.raises(EvaluationContractViolation, match="not one of the frozen"):
        train_stage2_head(train, train_y, dev, dev_y, seed=999)


# ==========================================================================================
# I. Role and condition safety on the training path
# ==========================================================================================

@requires_torch
def test_measurement_representations_cannot_train_or_select():
    train, train_y = synthetic(N_TRAIN, seed=15, role=STAGE2_MEASUREMENT_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=16, role=STAGE2_SELECTION_ROLE)
    with pytest.raises(EvaluationContractViolation, match="requires role 'protocol-train'"):
        train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])

    train2, y2 = synthetic(N_TRAIN, seed=17, role=STAGE2_TRAINING_ROLE)
    meas, my = synthetic(N_DEV, seed=18, role=STAGE2_MEASUREMENT_ROLE)
    with pytest.raises(EvaluationContractViolation, match="requires role 'protocol-dev'"):
        train_stage2_head(train2, y2, meas, my, seed=STAGE2_CAMPAIGN_SEEDS[0])


@requires_torch
def test_a_corrupted_dev_split_cannot_select_the_head():
    train, train_y = synthetic(N_TRAIN, seed=19, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=20, role=STAGE2_SELECTION_ROLE,
                           condition="P50", corruption_seed=77)
    with pytest.raises(EvaluationContractViolation, match="only use the clean"):
        train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])


@requires_torch
def test_cross_arm_training_is_refused():
    train, train_y = synthetic(N_TRAIN, seed=21, arm="UNMARK-A", role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=22, arm="UNMARK-B", role=STAGE2_SELECTION_ROLE)
    with pytest.raises(EvaluationContractViolation, match="cross-arm"):
        train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])


@requires_torch
def test_labels_must_match_the_cached_digest():
    train, train_y = synthetic(N_TRAIN, seed=23, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=24, role=STAGE2_SELECTION_ROLE)
    shuffled = list(reversed(train_y))
    with pytest.raises(EvaluationContractViolation, match="cached label digest"):
        train_stage2_head(train, shuffled, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])


# ==========================================================================================
# Cache round trip and immutability
# ==========================================================================================

@requires_torch
def test_the_cache_round_trips_and_refuses_a_foreign_key(tmp_path):
    bound, labels = synthetic(16, seed=25, role=STAGE2_TRAINING_ROLE)
    cache = Stage2RepresentationCache(tmp_path / "cache")
    cache.save(bound.key, bound.values)
    assert cache.exists()
    loaded = cache.load(bound.key)
    assert torch.equal(loaded.values, bound.values)
    assert loaded.key == bound.key

    other_arm = replace(
        bound.key, arm="UNMARK-B",
        finalist_checkpoint_sha256=finalist_for_arm("UNMARK-B").checkpoint_sha256,
    )
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        cache.load(other_arm)


@requires_torch
def test_the_cache_refuses_to_overwrite_a_different_key(tmp_path):
    bound, _ = synthetic(16, seed=26, role=STAGE2_TRAINING_ROLE)
    cache = Stage2RepresentationCache(tmp_path / "cache")
    cache.save(bound.key, bound.values)
    cache.save(bound.key, bound.values)  # idempotent re-extraction is allowed
    other = replace(bound.key, repository_head="9" * 40)
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        cache.save(other, bound.values)


@requires_torch
def test_the_cache_refuses_a_tensor_that_contradicts_its_key(tmp_path):
    bound, _ = synthetic(16, seed=27, role=STAGE2_TRAINING_ROLE)
    cache = Stage2RepresentationCache(tmp_path / "cache")
    with pytest.raises(EvaluationContractViolation, match="does not match key"):
        cache.save(bound.key, torch.zeros(15, HIDDEN_SIZE, dtype=torch.float32))
    with pytest.raises(EvaluationContractViolation, match="not the frozen"):
        cache.save(bound.key, torch.zeros(16, HIDDEN_SIZE, dtype=torch.float16))


# ==========================================================================================
# Artifact + store
# ==========================================================================================

@requires_torch
def test_a_completed_run_produces_a_validating_artifact_and_commits_once(tmp_path):
    train, train_y = synthetic(N_TRAIN, seed=28, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=29, role=STAGE2_SELECTION_ROLE)
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    run = train_stage2_head(train, train_y, dev, dev_y, seed=seed)
    artifact = build_stage2_head_artifact(run, repository_head=HEAD_SHA)
    validate_stage2_head_artifact(artifact, expected_arm="UNMARK-A", expected_seed=seed)

    store = Stage2HeadRunStore(tmp_path / "run")
    store.require_writable(expected_arm="UNMARK-A", expected_seed=seed)
    store.mark_in_progress(arm="UNMARK-A", seed=seed)
    store.commit(artifact, run.selected_head_state)
    assert store.is_complete()
    assert not (store.directory / store.IN_PROGRESS_NAME).exists()
    assert store.read_artifact(expected_arm="UNMARK-A", expected_seed=seed) == artifact
    with pytest.raises(EvaluationContractViolation, match="immutable"):
        store.require_writable(expected_arm="UNMARK-A", expected_seed=seed)


@requires_torch
def test_an_artifact_from_a_foreign_commit_is_refused():
    train, train_y = synthetic(N_TRAIN, seed=30, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=31, role=STAGE2_SELECTION_ROLE)
    run = train_stage2_head(train, train_y, dev, dev_y, seed=STAGE2_CAMPAIGN_SEEDS[0])
    with pytest.raises(EvaluationContractViolation, match="different repository head"):
        build_stage2_head_artifact(run, repository_head="b" * 40)


# ==========================================================================================
# Measurement -- reporting only
# ==========================================================================================

@requires_torch
def test_measurement_scores_a_frozen_head_and_refuses_a_selection_split():
    train, train_y = synthetic(N_TRAIN, seed=32, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=33, role=STAGE2_SELECTION_ROLE)
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    run = train_stage2_head(train, train_y, dev, dev_y, seed=seed)
    head = build_head(HIDDEN_SIZE, seed)
    head.load_state_dict(dict(run.selected_head_state))

    meas, meas_y = synthetic(N_DEV, seed=34, role=STAGE2_MEASUREMENT_ROLE,
                             condition="P50", corruption_seed=99)
    score = measure_stage2_head(head, meas, meas_y, seed=seed)
    assert score.condition == "P50"
    assert score.arm == "UNMARK-A"
    assert len(score.per_class_f1) == 3
    assert 0.0 <= score.macro_f1 <= 1.0 and 0.0 <= score.accuracy <= 1.0

    with pytest.raises(EvaluationContractViolation, match="requires role 'official-validation'"):
        measure_stage2_head(head, dev, dev_y, seed=seed)


@requires_torch
def test_measurement_does_not_mutate_the_head():
    train, train_y = synthetic(N_TRAIN, seed=35, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=36, role=STAGE2_SELECTION_ROLE)
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    run = train_stage2_head(train, train_y, dev, dev_y, seed=seed)
    head = build_head(HIDDEN_SIZE, seed)
    head.load_state_dict(dict(run.selected_head_state))
    before = {k: v.clone() for k, v in head.state_dict().items()}
    meas, meas_y = synthetic(N_DEV, seed=37, role=STAGE2_MEASUREMENT_ROLE)
    measure_stage2_head(head, meas, meas_y, seed=seed)
    for name, tensor in head.state_dict().items():
        assert torch.equal(tensor, before[name]), name


# ==========================================================================================
# Anti-drift: the Stage-2 loop matches the closed pre-G1 trainer
# ==========================================================================================

@requires_torch
def test_the_stage2_loop_reproduces_the_preg1_trainer_on_identical_inputs():
    """The epoch loop is written once here; this proves it has not drifted.

    Both trainers use the same locked primitives, so on identical features,
    labels and seed they must produce identical per-epoch scores.
    """
    from unmark.evaluation.contracts import SystemPathway
    from unmark.evaluation.preg1_head import (
        BoundRepresentations,
        Preg1Role,
        RepresentationKey,
        train_head,
    )

    seed = STAGE2_CAMPAIGN_SEEDS[0]
    train, train_y = synthetic(N_TRAIN, seed=41, role=STAGE2_TRAINING_ROLE)
    dev, dev_y = synthetic(N_DEV, seed=42, role=STAGE2_SELECTION_ROLE)

    def preg1_bound(values, labels, role):
        key = RepresentationKey(
            dataset="UIT-VSFC", dataset_version="1.0", task="sentiment", role=role,
            pathway=SystemPathway.VANILLA, source_identity="x" * 64,
            ordered_id_digest=ordered_id_digest([f"s{i}" for i in range(len(labels))]),
            tokenizer_id=ENCODER_CHECKPOINT, model_revision=ENCODER_REVISION,
            max_length=256, truncation=True, padding="max_length",
            pooling="FIRST_TOKEN", dtype=STAGE2_REPRESENTATION_DTYPE,
            hidden_size=HIDDEN_SIZE, count=len(labels),
        )
        return BoundRepresentations(values=values, key=key)

    reference = train_head(
        preg1_bound(train.values, train_y, Preg1Role.PROTOCOL_TRAIN), train_y,
        preg1_bound(dev.values, dev_y, Preg1Role.PROTOCOL_DEV), dev_y,
        learning_rate=STAGE2_HEAD_LEARNING_RATE, seed=seed,
    )
    stage2 = train_stage2_head(train, train_y, dev, dev_y, seed=seed)

    assert len(reference.scores) == len(stage2.scores) == EPOCHS
    for a, b in zip(reference.scores, stage2.scores):
        assert a.epoch == b.epoch
        assert a.macro_f1 == pytest.approx(b.macro_f1, abs=0.0, rel=0.0)
        assert a.accuracy == pytest.approx(b.accuracy, abs=0.0, rel=0.0)
