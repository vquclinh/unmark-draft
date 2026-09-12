"""Post-hoc V2-SCF Stage-2 campaign: cache identity, five-seed plan, head protocol.

Audit 071. Two things are being proven here.

**Separation.** A V2-SCF cache, head artifact and campaign manifest must be
impossible to confuse with a historical UNMARK-A/UNMARK-B one, in *both*
directions and by construction rather than convention -- different key schema,
different on-disk filenames, disjoint artifact fields.

**Inheritance.** Everything scientific about the head is the corrected historical
Stage-2 protocol, unchanged: the same five frozen seeds, the same 30 complete
epochs, the same LR, the same clean-only training and selection, the same
macro-F1 -> accuracy -> earliest-epoch rule, and no best-seed selection anywhere.

Torch-free wherever the contract allows it. Nothing here trains on UIT-VSFC,
reads a downstream row, opens official validation, or names official TEST.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.evaluation.stage2_scf_campaign as campaign  # noqa: E402
import unmark.evaluation.stage2_scf_measurement as measurement  # noqa: E402
import unmark.evaluation.stage2_scf_pathway as scf  # noqa: E402
from unmark.evaluation.preg1_head import Preg1Role, ordered_id_digest  # noqa: E402
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_TASK,
    TRUNCATION,
)
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_CONDITIONS,
)
from unmark.evaluation.stage2_head_campaign import (  # noqa: E402
    STAGE2_CLEAN_CONDITION,
    STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
    STAGE2_HEAD_LEARNING_RATE,
    STAGE2_MEASUREMENT_CORRUPTION_SEED,
    Stage2RepresentationCache,
    Stage2RepresentationKey,
    label_digest,
)
from unmark.stage1.protocol import (  # noqa: E402
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    HIDDEN_SIZE,
    HISTORICAL_FUSION_ID,
    SCALE_CALIBRATED_FUSION_ID,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN_MODULE = "unmark/evaluation/stage2_scf_campaign.py"
MEASUREMENT_MODULE = "unmark/evaluation/stage2_scf_measurement.py"

try:  # pragma: no cover - depends on the environment
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - the normal ML-free path
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(
    not TORCH, reason="torch is not installed locally; tensor checks run where available"
)

IDS_TRAIN = tuple(f"train-{i}" for i in range(8))
IDS_DEV = tuple(f"dev-{i}" for i in range(4))
LABELS_TRAIN = (0, 1, 2, 0, 1, 2, 0, 1)
LABELS_DEV = (0, 1, 2, 0)
HEAD = "c0ffee" * 6 + "abcd"


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def scf_key(
    *,
    role=campaign.SCF_TRAINING_ROLE,
    condition=STAGE2_CLEAN_CONDITION,
    corruption_seed=None,
    ids=IDS_TRAIN,
    labels=LABELS_TRAIN,
    stage2_head=HEAD,
    **overrides,
) -> campaign.ScfRepresentationKey:
    identity = scf.V2_SCF_CHECKPOINT
    fields = dict(
        stage2_repository_head=stage2_head,
        pathway_id=scf.V2_SCF_PATHWAY_ID,
        stage1_stage=identity.stage,
        stage1_source_repository_head=identity.source_repository_head,
        stage1_checkpoint_sha256=identity.checkpoint_sha256,
        stage1_selected_update=identity.update,
        fusion_id=identity.fusion_id,
        objective_id=identity.objective_id,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=scf.V2_SCF_POSTHOC_PROTOCOL_VERSION,
        dataset=PRIMARY_DATASET,
        dataset_version=PRIMARY_DATASET_VERSION,
        task=PRIMARY_TASK,
        role=role.value if hasattr(role, "value") else role,
        condition=condition,
        corruption_seed=corruption_seed,
        pooling=STAGE2_FIRST_TOKEN_POOLING,
        max_length=MAX_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING,
        ordered_id_digest=ordered_id_digest(list(ids)),
        label_digest=label_digest(labels),
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=len(ids),
    )
    fields.update(overrides)
    return campaign.ScfRepresentationKey(**fields)


def historical_key(**overrides) -> Stage2RepresentationKey:
    from unmark.stage1.finalists import FINALIST_A

    fields = dict(
        repository_head=HEAD,
        arm="UNMARK-A",
        finalist_checkpoint_sha256=FINALIST_A.checkpoint_sha256,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version="stage2-dual-finalist-protocol-v1",
        dataset=PRIMARY_DATASET,
        dataset_version=PRIMARY_DATASET_VERSION,
        task=PRIMARY_TASK,
        role=campaign.SCF_TRAINING_ROLE.value,
        condition=STAGE2_CLEAN_CONDITION,
        corruption_seed=None,
        pooling=STAGE2_FIRST_TOKEN_POOLING,
        max_length=MAX_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING,
        ordered_id_digest=ordered_id_digest(list(IDS_TRAIN)),
        label_digest=label_digest(LABELS_TRAIN),
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=len(IDS_TRAIN),
    )
    fields.update(overrides)
    return Stage2RepresentationKey(**fields)


# ---------------------------------------------------------------------------
# A. Cache identity: bound to everything, and disjoint from the A/B family
# ---------------------------------------------------------------------------
def test_scf_cache_key_binds_the_full_required_identity():
    key = scf_key().to_dict()
    for required in (
        "stage1_checkpoint_sha256",
        "fusion_id",
        "stage1_source_repository_head",
        "stage2_repository_head",
        "stage1_selected_update",
        "role",
        "condition",
        "corruption_seed",
        "ordered_id_digest",
        "label_digest",
        "pooling",
        "dtype",
        "hidden_size",
        "count",
        "representation_shape",
    ):
        assert required in key, f"the V2-SCF cache key does not bind {required}"
    assert key["fusion_id"] == SCALE_CALIBRATED_FUSION_ID
    assert key["stage1_checkpoint_sha256"] == scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    assert key["stage1_source_repository_head"] == (
        scf.V2_SCF_CHECKPOINT.source_repository_head
    )
    assert key["stage1_selected_update"] == 8000
    assert key["representation_shape"] == [len(IDS_TRAIN), HIDDEN_SIZE]
    assert key["pooling"] == "FIRST_TOKEN"
    assert key["dtype"] == "torch.float32"


def test_scf_cache_key_and_historical_key_are_mutually_unreadable():
    """Neither schema can parse the other. This is the separation, in one test."""
    scf_payload = scf_key().to_dict()
    ab_payload = historical_key().to_dict()

    with pytest.raises(scf.ScfPathwayViolation):
        campaign.ScfRepresentationKey.from_dict(ab_payload)
    with pytest.raises(Exception):
        Stage2RepresentationKey.from_dict(scf_payload)

    assert "arm" not in scf_payload
    assert "pathway_id" not in ab_payload
    assert scf_payload["schema_version"] != ab_payload["schema_version"]


def test_scf_cache_key_refuses_a_historical_arm_as_pathway():
    with pytest.raises(scf.ScfPathwayViolation):
        scf_key(pathway_id="UNMARK-A")


def test_scf_cache_key_refuses_the_historical_fusion():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf_key(fusion_id=HISTORICAL_FUSION_ID)
    assert "fusion" in str(error.value)


def test_scf_cache_key_refuses_another_checkpoint():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf_key(stage1_checkpoint_sha256="f" * 64)
    assert "checkpoint digest" in str(error.value)


def test_scf_cache_key_refuses_another_selected_update():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf_key(stage1_selected_update=7500)
    assert "selected update" in str(error.value)


def test_scf_cache_key_refuses_another_stage1_source_head():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf_key(stage1_source_repository_head="0" * 40)
    assert "source HEAD" in str(error.value)


def test_scf_cache_key_differs_when_the_stage2_commit_differs():
    a = scf_key(stage2_head="a" * 40)
    b = scf_key(stage2_head="b" * 40)
    assert a.to_dict() != b.to_dict()
    with pytest.raises(scf.ScfPathwayViolation) as error:
        a.require_compatible(b)
    assert "stage2_repository_head" in str(error.value)


def test_clean_cache_may_not_carry_a_corruption_seed():
    with pytest.raises(scf.ScfPathwayViolation):
        scf_key(condition="FULL", corruption_seed=19225)


def test_degraded_cache_must_carry_a_corruption_seed():
    with pytest.raises(scf.ScfPathwayViolation):
        scf_key(condition="P50", corruption_seed=None)


def test_degraded_measurement_cache_must_bind_the_frozen_seed():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf_key(
            role=campaign.SCF_MEASUREMENT_ROLE,
            condition="P50",
            corruption_seed=12345,
        )
    assert str(STAGE2_MEASUREMENT_CORRUPTION_SEED) in str(error.value)
    # The frozen one is accepted.
    scf_key(
        role=campaign.SCF_MEASUREMENT_ROLE,
        condition="P50",
        corruption_seed=STAGE2_MEASUREMENT_CORRUPTION_SEED,
    )


def test_scf_cache_uses_its_own_filenames():
    """A path typo cannot turn one artifact family into the other."""
    assert campaign.ScfRepresentationCache.METADATA_NAME != (
        Stage2RepresentationCache.METADATA_NAME
    )
    assert campaign.ScfRepresentationCache.TENSOR_NAME != (
        Stage2RepresentationCache.TENSOR_NAME
    )


def test_scf_cache_refuses_a_historical_cache_directory(tmp_path):
    """Pointed at an A/B cache, the V2-SCF cache finds nothing and fails closed."""
    (tmp_path / Stage2RepresentationCache.METADATA_NAME).write_text(
        json.dumps(historical_key().to_dict()), encoding="utf-8"
    )
    cache = campaign.ScfRepresentationCache(tmp_path)
    assert not cache.exists()
    with pytest.raises(scf.ScfPathwayViolation) as error:
        cache.read_key()
    assert "not a V2-SCF cache" in str(error.value)


def test_historical_cache_refuses_an_scf_cache_directory(tmp_path):
    (tmp_path / campaign.ScfRepresentationCache.METADATA_NAME).write_text(
        json.dumps(scf_key().to_dict()), encoding="utf-8"
    )
    cache = Stage2RepresentationCache(tmp_path)
    assert not cache.exists()
    with pytest.raises(Exception):
        cache.read_key()


def test_scf_cache_key_round_trips():
    key = scf_key()
    assert campaign.ScfRepresentationKey.from_dict(key.to_dict()) == key


def test_scf_cache_key_schema_is_closed():
    payload = scf_key().to_dict()
    payload["extra_field"] = 1
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.ScfRepresentationKey.from_dict(payload)
    assert "unknown" in str(error.value)


# ---------------------------------------------------------------------------
# B. The plan is exactly five seeds
# ---------------------------------------------------------------------------
def test_campaign_plan_is_exactly_five_scf_seeds():
    plan = campaign.scf_campaign_plan()
    assert len(plan) == 5 == campaign.SCF_CAMPAIGN_RUN_COUNT
    assert [r.seed for r in plan] == [53148, 59945, 42941, 720, 9428]
    assert {r.pathway_id for r in plan} == {scf.V2_SCF_PATHWAY_ID}


def test_campaign_seeds_are_the_frozen_historical_five_unchanged():
    assert campaign.SCF_CAMPAIGN_SEEDS == tuple(MEASUREMENT_SEEDS)
    assert campaign.SCF_CAMPAIGN_SEEDS == (53148, 59945, 42941, 720, 9428)


def test_campaign_plan_refuses_a_dropped_seed():
    plan = campaign.scf_campaign_plan()
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.require_scf_campaign_plan(plan[:4])
    assert "exactly 5 runs" in str(error.value)


def test_campaign_plan_refuses_an_extra_seed():
    plan = list(campaign.scf_campaign_plan())
    plan.append(campaign.ScfCampaignRun(pathway_id=scf.V2_SCF_PATHWAY_ID, seed=11111))
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.require_scf_campaign_plan(plan)


def test_campaign_plan_refuses_a_duplicated_seed():
    plan = list(campaign.scf_campaign_plan())
    plan[-1] = plan[0]
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.require_scf_campaign_plan(plan)
    assert "duplicate" in str(error.value)


def test_campaign_plan_refuses_a_historical_arm_run():
    plan = list(campaign.scf_campaign_plan())
    plan[0] = campaign.ScfCampaignRun(pathway_id="UNMARK-A", seed=plan[0].seed)
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.require_scf_campaign_plan(plan)


def test_no_best_seed_selection_exists():
    assert campaign.SCF_BEST_SEED_RULE is None
    assert campaign.SCF_WINNER_RULE is None
    assert campaign.SCF_TIE_BREAK is None
    assert campaign.SCF_BEST_SEED_SELECTION_IMPLEMENTED is False
    assert campaign.SCF_RANKS_AGAINST_HISTORICAL_ARMS is False
    assert measurement.SCF_MEASUREMENT_BEST_SEED_RULE is None
    assert measurement.SCF_MEASUREMENT_WINNER is None


def test_no_best_seed_helper_can_be_reached_by_name():
    """There is no function whose name could select among the five heads."""
    for name in (CAMPAIGN_MODULE, MEASUREMENT_MODULE):
        tree = ast.parse(source(name))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lowered = node.name.lower()
                for banned in ("best_seed", "winner", "rank_", "pick_", "choose_"):
                    assert banned not in lowered, (
                        f"{name} defines {node.name}, which looks like a selection helper"
                    )


# ---------------------------------------------------------------------------
# C. The head protocol is the corrected historical one, unchanged
# ---------------------------------------------------------------------------
def test_head_protocol_constants_are_inherited_not_redefined():
    assert campaign.STAGE2_CLEAN_CONDITION == "FULL"
    assert STAGE2_HEAD_LEARNING_RATE == 0.01
    assert EPOCHS == 30
    assert BATCH_SIZE == 128
    assert campaign.SCF_TRAINING_ROLE is Preg1Role.PROTOCOL_TRAIN
    assert campaign.SCF_SELECTION_ROLE is Preg1Role.PROTOCOL_DEV
    assert campaign.SCF_MEASUREMENT_ROLE is Preg1Role.OFFICIAL_VALIDATION


def test_campaign_reuses_the_locked_head_primitives():
    """Asserted by AST: the scientific primitives are imported, never redefined."""
    tree = ast.parse(source(CAMPAIGN_MODULE))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
    for primitive in (
        "build_head",
        "build_optimizer",
        "deterministic_batches",
        "select_checkpoint",
        "require_full_schedule",
        "score_predictions",
        "EpochScore",
        "Preg1Role",
        "require_head_only_optimizer",
        "require_clean_condition",
        "label_digest",
    ):
        assert primitive in imported, f"{primitive} is not reused"
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for primitive in (
        "build_head",
        "build_optimizer",
        "deterministic_batches",
        "select_checkpoint",
        "require_full_schedule",
        "score_predictions",
    ):
        assert primitive not in defined, f"{primitive} is re-implemented, not reused"


def test_thirty_epochs_are_required():
    tree = ast.parse(source(CAMPAIGN_MODULE))
    trainers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_scf_head"
    ]
    assert len(trainers) == 1
    body = ast.dump(trainers[0])
    assert "require_full_schedule" in body
    assert "EPOCHS" in body


def test_no_early_stopping_or_scheduler_reaches_the_scf_head():
    """No scheduler, no clipping, no smoothing, no accumulation, no early exit.

    `early_stopping` appears in the artifact schema as a field whose value is
    `False`, which is a declaration and not a mechanism -- so the mechanism is
    checked on the AST instead: the epoch loop in `train_scf_head` contains no
    `break`, which is what early stopping would actually look like here.
    """
    text = source(CAMPAIGN_MODULE)
    for banned in (
        "lr_scheduler",
        "get_linear_schedule",
        "clip_grad_norm",
        "patience",
        "label_smoothing",
        "gradient_accumulation",
    ):
        assert banned not in text, f"{banned} reached the V2-SCF head runner"

    tree = ast.parse(text)
    (trainer,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_scf_head"
    ]
    breaks = [node for node in ast.walk(trainer) if isinstance(node, ast.Break)]
    assert not breaks, "train_scf_head can exit its epoch loop early"


# ---------------------------------------------------------------------------
# D. Official validation cannot enter head training or selection
# ---------------------------------------------------------------------------
def test_training_extraction_plan_has_no_measurement_role():
    plan = campaign.scf_training_extraction_plan()
    assert len(plan) == 2
    roles = {r.role for r in plan}
    assert roles == {"protocol-train", "protocol-dev"}
    assert campaign.SCF_MEASUREMENT_ROLE.value not in roles
    assert all(r.condition == "FULL" for r in plan)
    assert all(r.corruption_seed is None for r in plan)


def test_clean_campaign_cache_slots_exclude_official_validation():
    assert campaign.SCF_CAMPAIGN_CACHE_SLOTS == (
        "UNMARK-V2-SCF/protocol-train",
        "UNMARK-V2-SCF/protocol-dev",
    )
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.scf_cache_slot(Preg1Role.OFFICIAL_VALIDATION)
    assert "no slot in the clean campaign" in str(error.value)


def test_campaign_manifest_refuses_an_official_validation_cache_slot():
    keys = {
        campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(),
        campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
            role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
        ),
        "UNMARK-V2-SCF/official-validation": scf_key(
            role=campaign.SCF_MEASUREMENT_ROLE, ids=IDS_DEV, labels=LABELS_DEV
        ),
    }
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.build_scf_campaign_manifest(
            stage2_repository_head=HEAD, cache_keys=keys
        )
    assert "unknown cache slot" in str(error.value)


def test_campaign_manifest_refuses_a_measurement_role_in_a_clean_slot():
    keys = {
        campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(
            role=campaign.SCF_MEASUREMENT_ROLE
        ),
        campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
            role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
        ),
    }
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.build_scf_campaign_manifest(
            stage2_repository_head=HEAD, cache_keys=keys
        )
    assert "role" in str(error.value)


def test_campaign_manifest_refuses_a_degraded_clean_slot():
    keys = {
        campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(
            condition="P50", corruption_seed=19225
        ),
        campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
            role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
        ),
    }
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.build_scf_campaign_manifest(
            stage2_repository_head=HEAD, cache_keys=keys
        )
    assert "clean" in str(error.value)


def test_campaign_manifest_refuses_identical_train_and_dev_rows():
    keys = {
        campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(),
        campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
            role=campaign.SCF_SELECTION_ROLE
        ),
    }
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.build_scf_campaign_manifest(
            stage2_repository_head=HEAD, cache_keys=keys
        )
    assert "same ordered-id digest" in str(error.value)


def valid_manifest() -> campaign.ScfCampaignManifest:
    return campaign.build_scf_campaign_manifest(
        stage2_repository_head=HEAD,
        cache_keys={
            campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(),
            campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
                role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
            ),
        },
    )


def test_valid_manifest_binds_five_runs_and_the_frozen_checkpoint():
    manifest = valid_manifest()
    payload = manifest.to_dict()
    assert len(payload["expected_runs"]) == 5
    assert payload["stage1_checkpoint_sha256"] == (
        scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    assert payload["fusion_id"] == SCALE_CALIBRATED_FUSION_ID
    assert payload["stage1_selected_update"] == 8000
    assert payload["posthoc_exploratory"] is True
    assert payload["best_seed_rule"] is None
    assert payload["winner_rule"] is None
    assert payload["ranks_against_historical_arms"] is False


def test_manifest_refuses_a_mixed_stage2_commit():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.build_scf_campaign_manifest(
            stage2_repository_head="d" * 40,
            cache_keys={
                campaign.scf_cache_slot(campaign.SCF_TRAINING_ROLE): scf_key(),
                campaign.scf_cache_slot(campaign.SCF_SELECTION_ROLE): scf_key(
                    role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
                ),
            },
        )
    assert "may not mix commits" in str(error.value)


# ---------------------------------------------------------------------------
# E. Head artifact schema: disjoint from the historical one, fail-closed
# ---------------------------------------------------------------------------
def head_artifact(**overrides) -> dict:
    identity = scf.V2_SCF_CHECKPOINT
    artifact = {
        "schema_version": campaign.V2_SCF_CAMPAIGN_SCHEMA_VERSION,
        "pathway_id": scf.V2_SCF_PATHWAY_ID,
        "posthoc_exploratory": True,
        "stage1_stage": identity.stage,
        "stage1_source_repository_head": identity.source_repository_head,
        "stage1_checkpoint_sha256": identity.checkpoint_sha256,
        "stage1_selected_update": identity.update,
        "fusion_id": identity.fusion_id,
        "objective_id": identity.objective_id,
        "stage2_repository_head": HEAD,
        "protocol_version": scf.V2_SCF_POSTHOC_PROTOCOL_VERSION,
        "seed": 53148,
        "initial_head_fingerprint": "a" * 64,
        "head_architecture": "Linear(768, 3, bias=True)",
        "optimizer": "AdamW",
        "learning_rate": STAGE2_HEAD_LEARNING_RATE,
        "epochs": EPOCHS,
        "early_stopping": False,
        "selected_epoch": 7,
        "selected_macro_f1": 0.5,
        "selected_accuracy": 0.6,
        "selected_head_state_sha256": "b" * 64,
        "train_cache_key": scf_key().to_dict(),
        "protocol_dev_cache_key": scf_key(
            role=campaign.SCF_SELECTION_ROLE, ids=IDS_DEV, labels=LABELS_DEV
        ).to_dict(),
        "history_digest": "c" * 64,
        "precision": STAGE2_REPRESENTATION_DTYPE,
        "selection_role": "protocol-dev",
        "selection_condition": "FULL",
        "measurement_used_for_selection": False,
        "best_seed_selection_performed": False,
        "ranked_against_historical_arms": False,
        "official_test_used": False,
    }
    artifact.update(overrides)
    return artifact


def test_valid_head_artifact_validates():
    campaign.validate_scf_head_artifact(head_artifact(), expected_seed=53148)


def test_head_artifact_schema_is_disjoint_from_the_historical_one():
    from unmark.evaluation.stage2_head_campaign import STAGE2_HEAD_ARTIFACT_FIELDS

    assert "arm" in STAGE2_HEAD_ARTIFACT_FIELDS
    assert "arm" not in campaign.SCF_HEAD_ARTIFACT_FIELDS
    assert "pathway_id" in campaign.SCF_HEAD_ARTIFACT_FIELDS
    assert "pathway_id" not in STAGE2_HEAD_ARTIFACT_FIELDS
    assert campaign.V2_SCF_CAMPAIGN_SCHEMA_VERSION != (
        STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION
    )


def test_head_artifact_refuses_an_unknown_field():
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.validate_scf_head_artifact(
            head_artifact(arm="UNMARK-A"), expected_seed=53148
        )
    assert "unknown field" in str(error.value)


def test_head_artifact_refuses_a_missing_field():
    artifact = head_artifact()
    del artifact["fusion_id"]
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.validate_scf_head_artifact(artifact, expected_seed=53148)
    assert "missing" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fusion_id", HISTORICAL_FUSION_ID),
        ("stage1_checkpoint_sha256", "f" * 64),
        ("stage1_source_repository_head", "0" * 40),
        ("stage1_selected_update", 7500),
        ("learning_rate", 0.001),
        ("epochs", 10),
        ("early_stopping", True),
        ("selection_condition", "P50"),
        ("selection_role", "official-validation"),
        ("posthoc_exploratory", False),
        ("measurement_used_for_selection", True),
        ("best_seed_selection_performed", True),
        ("ranked_against_historical_arms", True),
        ("official_test_used", True),
    ],
)
def test_head_artifact_fails_closed_on_drift(field, value):
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.validate_scf_head_artifact(
            head_artifact(**{field: value}), expected_seed=53148
        )


def test_head_artifact_refuses_an_unfrozen_seed():
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.validate_scf_head_artifact(
            head_artifact(seed=11111), expected_seed=11111
        )


def test_head_run_store_uses_its_own_filenames():
    from unmark.evaluation.stage2_head_campaign import Stage2HeadRunStore

    assert campaign.ScfHeadRunStore.ARTIFACT_NAME != Stage2HeadRunStore.ARTIFACT_NAME
    assert campaign.ScfHeadRunStore.STATE_NAME != Stage2HeadRunStore.STATE_NAME


def test_completed_head_run_is_immutable(tmp_path):
    store = campaign.ScfHeadRunStore(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store.artifact_path.write_text(json.dumps(head_artifact()), encoding="utf-8")
    with pytest.raises(scf.ScfPathwayViolation) as error:
        store.require_writable(expected_seed=53148)
    assert "immutable" in str(error.value)


# ---------------------------------------------------------------------------
# F. Measurement is a later phase, and it is gated
# ---------------------------------------------------------------------------
def test_measurement_lives_in_its_own_module():
    campaign_text = source(CAMPAIGN_MODULE)
    assert "def measure_" not in campaign_text
    assert "aggregate_scf_campaign" not in campaign_text
    # The measurement role exists as a declared constant in the campaign module so
    # a cache key can validate one, but nothing there reads such a tensor.
    assert campaign.SCF_MEASUREMENT_ROLE is Preg1Role.OFFICIAL_VALIDATION
    measurement_text = source(MEASUREMENT_MODULE)
    assert "def measure_scf_head" in measurement_text
    assert "def train_" not in measurement_text


def test_measurement_requires_the_explicit_authorisation_string(tmp_path):
    manifest = valid_manifest()
    registry = campaign.ScfCampaignRegistry(tmp_path)
    with pytest.raises(scf.ScfPathwayViolation) as error:
        measurement.require_scf_measurement_authorised(
            manifest, registry, authorisation="yes"
        )
    assert measurement.SCF_MEASUREMENT_AUTHORISATION in str(error.value)


def test_measurement_requires_all_five_clean_heads(tmp_path):
    manifest = valid_manifest()
    registry = campaign.ScfCampaignRegistry(tmp_path)
    registry.open(manifest)
    with pytest.raises(scf.ScfPathwayViolation) as error:
        measurement.require_scf_measurement_authorised(
            manifest,
            registry,
            authorisation=measurement.SCF_MEASUREMENT_AUTHORISATION,
        )
    assert "0/5" in str(error.value)


def test_measurement_extraction_plan_is_gated_before_it_names_a_seed(tmp_path):
    manifest = valid_manifest()
    registry = campaign.ScfCampaignRegistry(tmp_path)
    with pytest.raises(scf.ScfPathwayViolation):
        measurement.scf_measurement_extraction_plan(
            manifest,
            registry,
            corruption_seed=STAGE2_MEASUREMENT_CORRUPTION_SEED,
            authorisation="",
        )


def test_measurement_corruption_seed_has_no_default():
    import inspect

    signature = inspect.signature(measurement.scf_measurement_extraction_plan)
    assert signature.parameters["corruption_seed"].default is inspect.Parameter.empty
    assert signature.parameters["authorisation"].default is inspect.Parameter.empty


def test_measurement_conditions_are_the_six_frozen_ones():
    payload = scf.require_frozen_scf_protocol_spec()
    assert payload["measurement"]["conditions"]["value"] == [
        "FULL",
        "P25",
        "P50",
        "P75",
        "P100",
        "STRIP_ALL",
    ]
    assert list(STAGE2_UNMARK_CONDITIONS) == [
        "FULL",
        "P25",
        "P50",
        "P75",
        "P100",
        "STRIP_ALL",
    ]


def test_aggregate_refuses_a_report_missing_a_seed():
    scores = [
        measurement.ScfConditionScore(
            pathway_id=scf.V2_SCF_PATHWAY_ID,
            seed=seed,
            condition="FULL",
            macro_f1=0.5,
            accuracy=0.6,
            per_class_f1=(0.5, 0.5, 0.5),
        )
        for seed in campaign.SCF_CAMPAIGN_SEEDS[:4]
    ]
    with pytest.raises(scf.ScfPathwayViolation) as error:
        measurement.aggregate_scf_campaign(scores)
    assert "never be dropped" in str(error.value)


def test_aggregate_reports_all_five_seeds_and_no_winner():
    scores = [
        measurement.ScfConditionScore(
            pathway_id=scf.V2_SCF_PATHWAY_ID,
            seed=seed,
            condition="FULL",
            macro_f1=0.5 + index / 100,
            accuracy=0.6,
            per_class_f1=(0.5, 0.5, 0.5),
        )
        for index, seed in enumerate(campaign.SCF_CAMPAIGN_SEEDS)
    ]
    report = measurement.aggregate_scf_campaign(scores)
    assert report["winner"] is None
    assert report["best_seed"] is None
    assert report["ranks_against_historical_arms"] is False
    assert report["posthoc_exploratory"] is True
    assert report["confirmatory"] is False
    assert report["official_test_used"] is False
    assert len(report["conditions"]["FULL"]["per_seed"]) == 5


def test_aggregate_refuses_a_historical_arm_score():
    scores = [
        measurement.ScfConditionScore(
            pathway_id="UNMARK-A",
            seed=seed,
            condition="FULL",
            macro_f1=0.5,
            accuracy=0.6,
            per_class_f1=(0.5, 0.5, 0.5),
        )
        for seed in campaign.SCF_CAMPAIGN_SEEDS
    ]
    with pytest.raises(scf.ScfPathwayViolation) as error:
        measurement.aggregate_scf_campaign(scores)
    assert "not ranked into" in str(error.value)


# ---------------------------------------------------------------------------
# G. Tensor-level behaviour (torch-gated)
# ---------------------------------------------------------------------------
def make_encoder():
    import torch
    from torch import nn
    from types import SimpleNamespace

    class Embeddings:
        def __init__(self, word_embeddings):
            self.word_embeddings = word_embeddings

    class RobertaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(512, HIDDEN_SIZE, padding_idx=1)
            self.embeddings = Embeddings(self.word_embeddings)
            self.name_or_path = ENCODER_CHECKPOINT
            self.config = SimpleNamespace(
                hidden_size=HIDDEN_SIZE,
                model_type="roberta",
                pad_token_id=1,
                _commit_hash=ENCODER_REVISION,
            )

        def get_input_embeddings(self):
            return self.word_embeddings

        def forward(
            self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None
        ):
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            batch, length, dim = inputs_embeds.shape
            positions = torch.arange(
                length, dtype=torch.float32, device=inputs_embeds.device
            ).view(1, length, 1)
            offsets = torch.arange(
                batch, dtype=torch.float32, device=inputs_embeds.device
            ).view(batch, 1, 1) * 1000.0
            return SimpleNamespace(
                last_hidden_state=positions.expand(batch, length, dim) + offsets
            )

    return RobertaModel()


def make_scf_pathway():
    from unmark.stage1.initialisation import fresh_adapter

    encoder = make_encoder()
    adapter = fresh_adapter(HIDDEN_SIZE, 51800, SCALE_CALIBRATED_FUSION_ID)
    for module in (encoder, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    evidence = {
        "kind": "stage2_v2_scf_checkpoint_evidence",
        "pathway_id": scf.V2_SCF_PATHWAY_ID,
        "stage": scf.V2_SCF_CHECKPOINT.stage,
        "source_repository_head": scf.V2_SCF_CHECKPOINT.source_repository_head,
        "run_seed": scf.V2_SCF_CHECKPOINT.run_seed,
        "update": scf.V2_SCF_CHECKPOINT.update,
        "fusion_id": scf.V2_SCF_CHECKPOINT.fusion_id,
        "objective_id": scf.V2_SCF_CHECKPOINT.objective_id,
        "checkpoint_sha256": scf.V2_SCF_CHECKPOINT.checkpoint_sha256,
        "checkpoint_schema_version": "stage1-checkpoint-v2",
        "adapter_tensor_count": 8,
        "adapter_trainable_parameters": 3_551_232,
        "adapter_dtype": "fp32",
        "all_finite": True,
    }
    binding = scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    return scf.FrozenScfPathway(binding=binding, encoder=encoder, adapter=adapter)


@requires_torch
def test_frozen_scf_pathway_accepts_a_correctly_frozen_pathway():
    pathway = make_scf_pathway()
    pathway.require_frozen(check_values=True)
    assert pathway.fusion_id == SCALE_CALIBRATED_FUSION_ID


@requires_torch
def test_every_encoder_and_adapter_parameter_is_frozen():
    pathway = make_scf_pathway()
    for module in (pathway.encoder, pathway.adapter):
        assert module.training is False
        assert not any(p.requires_grad for p in module.parameters())
        assert all(p.dtype is torch.float32 for p in module.parameters())


@requires_torch
@pytest.mark.parametrize("which", ["encoder", "adapter"])
def test_an_unfrozen_parameter_fails_closed(which):
    pathway = make_scf_pathway()
    module = getattr(pathway, which)
    next(iter(module.parameters())).requires_grad_(True)
    with pytest.raises(scf.ScfPathwayViolation) as error:
        pathway.require_frozen()
    assert "requires grad" in str(error.value)


@requires_torch
@pytest.mark.parametrize("which", ["encoder", "adapter"])
def test_a_training_mode_module_fails_closed(which):
    pathway = make_scf_pathway()
    getattr(pathway, which).train()
    with pytest.raises(scf.ScfPathwayViolation) as error:
        pathway.require_frozen()
    assert "eval mode" in str(error.value)


@requires_torch
def test_a_historical_fusion_adapter_on_the_scf_pathway_fails_closed():
    """The pathway checks the LIVE module, not a string in a binding."""
    from unmark.stage1.initialisation import fresh_adapter

    pathway = make_scf_pathway()
    historical = fresh_adapter(HIDDEN_SIZE, 51800, HISTORICAL_FUSION_ID)
    historical.load_state_dict(pathway.adapter.state_dict(), strict=True)
    for parameter in historical.parameters():
        parameter.requires_grad_(False)
    historical.eval()
    pathway.adapter = historical
    with pytest.raises(scf.ScfPathwayViolation) as error:
        pathway.require_frozen()
    assert SCALE_CALIBRATED_FUSION_ID in str(error.value)


@requires_torch
def test_representation_is_first_token_768_detached_fp32():
    pathway = make_scf_pathway()
    batch_size, length = 3, 6
    batch = {
        "input_ids": torch.full((batch_size, length), 7, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
        "tone_ids": torch.zeros(batch_size, length, dtype=torch.long),
        "tone_mask": torch.ones(batch_size, length, dtype=torch.bool),
        "letter_ids": torch.zeros(batch_size, length, 1, dtype=torch.long),
        "letter_mask": torch.ones(batch_size, length, 1, dtype=torch.bool),
    }
    pooled = scf.extract_scf_representations(pathway, batch)
    assert tuple(pooled.shape) == (batch_size, HIDDEN_SIZE)
    assert pooled.dtype is torch.float32
    assert pooled.requires_grad is False
    assert pooled.grad_fn is None
    # The stub encoder emits `position + 1000 * row`, so FIRST_TOKEN is row*1000.
    expected = torch.tensor([0.0, 1000.0, 2000.0])
    assert torch.allclose(pooled[:, 0], expected)


@requires_torch
def test_scf_head_training_matches_the_historical_head_protocol_exactly():
    """Same inputs, same seed: `train_scf_head` and `train_stage2_head` agree.

    This is what makes "the corrected historical Stage-2 head protocol, reused
    unchanged" a checked statement rather than a claim in a docstring.
    """
    from unmark.evaluation.stage2_head_campaign import (
        Stage2BoundRepresentations,
        train_stage2_head,
    )

    rows_train, rows_dev = 40, 16
    torch.manual_seed(7)
    train_values = torch.randn(rows_train, HIDDEN_SIZE, dtype=torch.float32)
    dev_values = torch.randn(rows_dev, HIDDEN_SIZE, dtype=torch.float32)
    train_labels = [i % 3 for i in range(rows_train)]
    dev_labels = [i % 3 for i in range(rows_dev)]
    ids_train = [f"t{i}" for i in range(rows_train)]
    ids_dev = [f"d{i}" for i in range(rows_dev)]

    scf_train = campaign.ScfBoundRepresentations(
        values=train_values,
        key=scf_key(ids=ids_train, labels=train_labels),
    )
    scf_dev = campaign.ScfBoundRepresentations(
        values=dev_values,
        key=scf_key(
            role=campaign.SCF_SELECTION_ROLE, ids=ids_dev, labels=dev_labels
        ),
    )
    ab_train = Stage2BoundRepresentations(
        values=train_values,
        key=historical_key(
            ordered_id_digest=ordered_id_digest(ids_train),
            label_digest=label_digest(train_labels),
            count=rows_train,
        ),
    )
    ab_dev = Stage2BoundRepresentations(
        values=dev_values,
        key=historical_key(
            role="protocol-dev",
            ordered_id_digest=ordered_id_digest(ids_dev),
            label_digest=label_digest(dev_labels),
            count=rows_dev,
        ),
    )

    seed = campaign.SCF_CAMPAIGN_SEEDS[0]
    scf_run = campaign.train_scf_head(
        scf_train, train_labels, scf_dev, dev_labels, seed=seed
    )
    ab_run = train_stage2_head(ab_train, train_labels, ab_dev, dev_labels, seed=seed)

    assert [s.to_dict() for s in scf_run.scores] == [s.to_dict() for s in ab_run.scores]
    assert scf_run.selected.epoch == ab_run.selected.epoch
    assert scf_run.initial_head_fingerprint == ab_run.initial_head_fingerprint
    assert scf_run.selected_head_state_sha256 == ab_run.selected_head_state_sha256
    assert len(scf_run.scores) == EPOCHS == 30


@requires_torch
def test_scf_head_training_refuses_an_official_validation_tensor():
    """Official validation cannot enter head training or checkpoint selection."""
    rows = 8
    values = torch.randn(rows, HIDDEN_SIZE, dtype=torch.float32)
    labels = [i % 3 for i in range(rows)]
    ids = [f"m{i}" for i in range(rows)]
    measurement_tensor = campaign.ScfBoundRepresentations(
        values=values,
        key=scf_key(role=campaign.SCF_MEASUREMENT_ROLE, ids=ids, labels=labels),
    )
    clean_dev = campaign.ScfBoundRepresentations(
        values=values,
        key=scf_key(role=campaign.SCF_SELECTION_ROLE, ids=ids, labels=labels),
    )
    with pytest.raises(Exception) as error:
        campaign.train_scf_head(
            measurement_tensor, labels, clean_dev, labels,
            seed=campaign.SCF_CAMPAIGN_SEEDS[0],
        )
    assert "protocol-train" in str(error.value)

    with pytest.raises(Exception) as error:
        campaign.train_scf_head(
            clean_dev, labels, measurement_tensor, labels,
            seed=campaign.SCF_CAMPAIGN_SEEDS[0],
        )
    assert "protocol-train" in str(error.value) or "protocol-dev" in str(error.value)


@requires_torch
def test_scf_head_training_refuses_an_unfrozen_seed_and_a_retuned_lr():
    rows = 8
    values = torch.randn(rows, HIDDEN_SIZE, dtype=torch.float32)
    labels = [i % 3 for i in range(rows)]
    train = campaign.ScfBoundRepresentations(
        values=values, key=scf_key(ids=[f"t{i}" for i in range(rows)], labels=labels)
    )
    dev = campaign.ScfBoundRepresentations(
        values=values,
        key=scf_key(
            role=campaign.SCF_SELECTION_ROLE,
            ids=[f"d{i}" for i in range(rows)],
            labels=labels,
        ),
    )
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.train_scf_head(train, labels, dev, labels, seed=11111)
    with pytest.raises(scf.ScfPathwayViolation) as error:
        campaign.train_scf_head(
            train, labels, dev, labels,
            seed=campaign.SCF_CAMPAIGN_SEEDS[0],
            learning_rate=0.001,
        )
    assert "no V2-SCF LR pilot" in str(error.value)
    with pytest.raises(scf.ScfPathwayViolation):
        campaign.train_scf_head(
            train, labels, dev, labels,
            seed=campaign.SCF_CAMPAIGN_SEEDS[0],
            epochs=10,
        )
