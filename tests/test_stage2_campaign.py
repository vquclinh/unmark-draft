"""Stage-2 campaign orchestration: one identity, exactly ten runs, no selection.

**Torch-free.** Manifest validation, the registry and the schedule are all
identity logic, which is the part that can silently assemble a meaningless
campaign — ten runs that look complete but were produced at two commits. The
execution path that needs real tensors is exercised in
`test_stage2_campaign_torch.py`.

Nothing here trains, reads a downstream row, or names official TEST.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import ordered_id_digest  # noqa: E402
from unmark.evaluation.stage2_campaign import (  # noqa: E402
    STAGE2_CAMPAIGN_CACHE_SLOTS,
    STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    STAGE2_CAMPAIGN_RANKS_ARMS,
    STAGE2_CAMPAIGN_WINNER,
    Stage2CampaignManifest,
    Stage2CampaignRegistry,
    build_stage2_campaign_manifest,
    cache_slot,
    stage2_campaign_schedule,
    validate_stage2_campaign_manifest,
)
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    Stage2UnmarkArm,
    finalist_for_arm,
)
from unmark.evaluation.stage2_head_campaign import (  # noqa: E402
    STAGE2_CAMPAIGN_RUN_COUNT,
    STAGE2_CAMPAIGN_SEEDS,
    STAGE2_CLEAN_CONDITION,
    STAGE2_PROTOCOL_VERSION,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    Stage2CampaignRun,
    Stage2RepresentationKey,
    label_digest,
)
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULE = REPO / "unmark/evaluation/stage2_campaign.py"
HEAD_SHA = "a" * 40


def make_key(arm, role, *, head=HEAD_SHA, protocol=STAGE2_PROTOCOL_VERSION,
             condition=STAGE2_CLEAN_CONDITION, corruption_seed=None, sha=None, count=8):
    arm_value = arm.value if hasattr(arm, "value") else arm
    return Stage2RepresentationKey(
        repository_head=head,
        arm=arm_value,
        finalist_checkpoint_sha256=sha or finalist_for_arm(arm_value).checkpoint_sha256,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=protocol,
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
        ordered_id_digest=ordered_id_digest([f"{arm_value}-{role.value}-{i}" for i in range(count)]),
        label_digest=label_digest([i % 3 for i in range(count)]),
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=count,
    )


def cache_keys(**overrides):
    keys = {}
    for arm in Stage2UnmarkArm:
        for role in (STAGE2_TRAINING_ROLE, STAGE2_SELECTION_ROLE):
            keys[cache_slot(arm, role)] = make_key(arm, role)
    keys.update(overrides)
    return keys


def manifest(**overrides):
    return build_stage2_campaign_manifest(
        repository_head=overrides.pop("repository_head", HEAD_SHA),
        cache_keys=overrides.pop("cache_keys", cache_keys()),
    )


# ==========================================================================================
# The accepted campaign
# ==========================================================================================

def test_the_exact_two_by_five_campaign_is_accepted():
    m = manifest()
    validate_stage2_campaign_manifest(m)
    assert m.arms == ("UNMARK-A", "UNMARK-B")
    assert m.seeds == tuple(STAGE2_CAMPAIGN_SEEDS)
    assert len(m.expected_runs) == STAGE2_CAMPAIGN_RUN_COUNT == 10
    assert set(m.cache_keys) == set(STAGE2_CAMPAIGN_CACHE_SLOTS)


def test_the_orchestrator_schedules_exactly_ten_runs():
    schedule = stage2_campaign_schedule(manifest())
    assert len(schedule) == 10
    assert len({(r.arm, r.seed) for r in schedule}) == 10


def test_matched_seeds_remain_paired():
    schedule = stage2_campaign_schedule(manifest())
    by_seed = {}
    for run in schedule:
        by_seed.setdefault(run.seed, set()).add(run.arm)
    assert sorted(by_seed) == sorted(STAGE2_CAMPAIGN_SEEDS)
    for seed, arms in by_seed.items():
        assert arms == {"UNMARK-A", "UNMARK-B"}, seed


def test_the_manifest_digest_is_stable_and_identity_sensitive():
    a, b = manifest(), manifest()
    assert a.digest == b.digest
    other = manifest(repository_head="b" * 40, cache_keys=cache_keys(**{
        slot: make_key(slot.split("/")[0],
                       STAGE2_TRAINING_ROLE if slot.endswith("train") else STAGE2_SELECTION_ROLE,
                       head="b" * 40)
        for slot in STAGE2_CAMPAIGN_CACHE_SLOTS
    }))
    assert other.digest != a.digest


# ==========================================================================================
# Fail-closed: arms and seeds
# ==========================================================================================

def test_a_missing_arm_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="are not exactly"):
        validate_stage2_campaign_manifest(replace(m, arms=("UNMARK-A",)))


def test_a_third_arm_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="are not exactly"):
        validate_stage2_campaign_manifest(
            replace(m, arms=("UNMARK-A", "UNMARK-B", "UNMARK-C"))
        )


def test_a_missing_seed_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="not exactly the frozen"):
        validate_stage2_campaign_manifest(replace(m, seeds=STAGE2_CAMPAIGN_SEEDS[:4]))


def test_an_extra_seed_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="not exactly the frozen"):
        validate_stage2_campaign_manifest(
            replace(m, seeds=tuple(STAGE2_CAMPAIGN_SEEDS) + (12345,))
        )


def test_a_duplicate_run_refuses():
    m = manifest()
    runs = list(m.expected_runs)
    runs[1] = runs[0]
    with pytest.raises(EvaluationContractViolation, match="duplicate run|not paired"):
        validate_stage2_campaign_manifest(replace(m, expected_runs=tuple(runs)))


def test_a_short_run_list_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="exactly 10 runs"):
        validate_stage2_campaign_manifest(replace(m, expected_runs=m.expected_runs[:9]))


def test_an_unpaired_run_list_refuses():
    m = manifest()
    runs = [r for r in m.expected_runs if not (r.arm == "UNMARK-B"
                                               and r.seed == STAGE2_CAMPAIGN_SEEDS[0])]
    runs.append(Stage2CampaignRun(arm="UNMARK-A", seed=STAGE2_CAMPAIGN_SEEDS[0]))
    with pytest.raises(EvaluationContractViolation, match="duplicate run|not paired"):
        validate_stage2_campaign_manifest(replace(m, expected_runs=tuple(runs)))


# ==========================================================================================
# Fail-closed: caches
# ==========================================================================================

def test_a_missing_cache_slot_refuses():
    keys = cache_keys()
    del keys[cache_slot(Stage2UnmarkArm.UNMARK_B, STAGE2_SELECTION_ROLE)]
    with pytest.raises(EvaluationContractViolation, match="missing cache slot"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_an_unknown_cache_slot_refuses():
    keys = cache_keys()
    keys["UNMARK-C/protocol-train"] = make_key(Stage2UnmarkArm.UNMARK_A, STAGE2_TRAINING_ROLE)
    with pytest.raises(EvaluationContractViolation, match="unknown cache slot"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_cache_from_the_wrong_arm_refuses():
    """B's slot holding A's cache is the substitution that makes arms identical."""
    keys = cache_keys(**{
        cache_slot(Stage2UnmarkArm.UNMARK_B, STAGE2_TRAINING_ROLE):
            make_key(Stage2UnmarkArm.UNMARK_A, STAGE2_TRAINING_ROLE)
    })
    with pytest.raises(EvaluationContractViolation, match="may never stand in for"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_cache_with_the_wrong_role_refuses():
    keys = cache_keys(**{
        cache_slot(Stage2UnmarkArm.UNMARK_A, STAGE2_TRAINING_ROLE):
            make_key(Stage2UnmarkArm.UNMARK_A, STAGE2_SELECTION_ROLE)
    })
    with pytest.raises(EvaluationContractViolation, match="holds role"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_corrupted_cache_refuses():
    keys = cache_keys(**{
        cache_slot(Stage2UnmarkArm.UNMARK_A, STAGE2_SELECTION_ROLE):
            make_key(Stage2UnmarkArm.UNMARK_A, STAGE2_SELECTION_ROLE,
                     condition="P50", corruption_seed=9)
    })
    with pytest.raises(EvaluationContractViolation, match="clean FULL only"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_mixed_repository_head_refuses():
    keys = cache_keys(**{
        cache_slot(Stage2UnmarkArm.UNMARK_B, STAGE2_TRAINING_ROLE):
            make_key(Stage2UnmarkArm.UNMARK_B, STAGE2_TRAINING_ROLE, head="9" * 40)
    })
    with pytest.raises(EvaluationContractViolation, match="may not mix commits"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_mixed_protocol_refuses():
    keys = cache_keys(**{
        cache_slot(Stage2UnmarkArm.UNMARK_A, STAGE2_SELECTION_ROLE):
            make_key(Stage2UnmarkArm.UNMARK_A, STAGE2_SELECTION_ROLE,
                     protocol="stage2-dual-finalist-protocol-v2")
    })
    with pytest.raises(EvaluationContractViolation, match="may not mix protocols"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_two_arms_sharing_one_checkpoint_refuses():
    shared = finalist_for_arm("UNMARK-A").checkpoint_sha256
    keys = cache_keys()
    for role in (STAGE2_TRAINING_ROLE, STAGE2_SELECTION_ROLE):
        keys[cache_slot(Stage2UnmarkArm.UNMARK_B, role)] = make_key(
            Stage2UnmarkArm.UNMARK_B, role, sha=shared
        )
    with pytest.raises(EvaluationContractViolation, match="same finalist checkpoint"):
        build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)


def test_a_manifest_protocol_that_is_not_frozen_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="not the frozen"):
        validate_stage2_campaign_manifest(
            replace(m, protocol_version="stage2-dual-finalist-protocol-v9")
        )


def test_a_wrong_manifest_schema_refuses():
    m = manifest()
    with pytest.raises(EvaluationContractViolation, match="manifest schema"):
        validate_stage2_campaign_manifest(replace(m, schema_version="stage2-campaign-v0"))


# ==========================================================================================
# Registry: re-entry, overwrite, incompatible state
# ==========================================================================================

def test_a_registry_opens_and_reopens_under_the_same_manifest(tmp_path):
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    m = manifest()
    registry.open(m)
    assert registry.exists()
    registry.open(m)  # re-entry under an identical manifest is fine
    recorded = json.loads(registry.manifest_path.read_text(encoding="utf-8"))
    assert recorded["manifest_digest"] == m.digest


def test_re_entering_with_an_incompatible_manifest_refuses(tmp_path):
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    registry.open(manifest())
    other_head = "b" * 40
    drifted = build_stage2_campaign_manifest(
        repository_head=other_head,
        cache_keys={
            slot: make_key(slot.split("/")[0],
                           STAGE2_TRAINING_ROLE if slot.endswith("train")
                           else STAGE2_SELECTION_ROLE,
                           head=other_head)
            for slot in STAGE2_CAMPAIGN_CACHE_SLOTS
        },
    )
    with pytest.raises(EvaluationContractViolation, match="may not adopt an incompatible"):
        registry.open(drifted)


def test_a_completed_run_cannot_be_overwritten(tmp_path):
    from unmark.evaluation.stage2_head_campaign import (
        STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
        STAGE2_HEAD_LEARNING_RATE,
    )
    from unmark.evaluation.preg1_protocol import EPOCHS, PRIMARY_NUM_LABELS

    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    m = manifest()
    registry.open(m)
    run = m.expected_runs[0]
    store = registry.store_for(run.arm, run.seed)
    store.directory.mkdir(parents=True, exist_ok=True)
    train = m.cache_keys[cache_slot(run.arm, STAGE2_TRAINING_ROLE)]
    dev = m.cache_keys[cache_slot(run.arm, STAGE2_SELECTION_ROLE)]
    store.artifact_path.write_text(json.dumps({
        "schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
        "arm": run.arm,
        "finalist_checkpoint_sha256": finalist_for_arm(run.arm).checkpoint_sha256,
        "repository_head": HEAD_SHA, "protocol_version": STAGE2_PROTOCOL_VERSION,
        "seed": run.seed, "initial_head_fingerprint": "1" * 64,
        "head_architecture": f"Linear({HIDDEN_SIZE}, {PRIMARY_NUM_LABELS}, bias=True)",
        "optimizer": "AdamW", "learning_rate": STAGE2_HEAD_LEARNING_RATE,
        "epochs": EPOCHS, "early_stopping": False, "selected_epoch": 3,
        "selected_macro_f1": 0.5, "selected_accuracy": 0.6,
        "selected_head_state_sha256": "2" * 64,
        "train_cache_key": train.to_dict(), "protocol_dev_cache_key": dev.to_dict(),
        "history_digest": "3" * 64, "precision": STAGE2_REPRESENTATION_DTYPE,
        "selection_role": STAGE2_SELECTION_ROLE.value,
        "selection_condition": STAGE2_CLEAN_CONDITION,
        "measurement_used_for_selection": False, "ab_selection_performed": False,
    }), encoding="utf-8")

    assert store.is_complete()
    with pytest.raises(EvaluationContractViolation, match="immutable"):
        store.require_writable(expected_arm=run.arm, expected_seed=run.seed)
    assert registry.completed_runs(m) == (run,)
    assert len(registry.pending_runs(m)) == 9


def test_a_partial_campaign_may_not_be_reported(tmp_path):
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    m = manifest()
    registry.open(m)
    status = registry.campaign_status(m)
    assert status["complete"] is False
    assert status["completed"] == 0 and status["expected"] == 10
    with pytest.raises(EvaluationContractViolation, match="Both arms must be reported"):
        registry.require_campaign_complete(m)


def test_status_is_a_completion_report_not_a_comparison(tmp_path):
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    m = manifest()
    registry.open(m)
    status = registry.campaign_status(m)
    assert status["winner"] is None
    assert status["ranks_arms"] is False
    # Scan the report MINUS its two safety declarations, whose whole purpose is to
    # record that no winner and no ranking exist. Including them would make this
    # test match its own negative statement.
    scanned = {k: v for k, v in status.items() if k not in {"winner", "ranks_arms"}}
    text = json.dumps(scanned).lower()
    for banned in ("macro_f1", "accuracy", "best", "rank", "score", "delta"):
        assert banned not in text, f"status leaks {banned}: it must not compare arms"


# ==========================================================================================
# No selection, and no duplicated implementation
# ==========================================================================================

def test_the_module_declares_no_winner_or_ranking():
    assert STAGE2_CAMPAIGN_WINNER is None
    assert STAGE2_CAMPAIGN_RANKS_ARMS is False


def test_no_definition_looks_like_selection_machinery():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    banned = ("winner", "best_arm", "rank_arms", "select_arm", "choose_arm",
              "pick_finalist", "drop_arm", "promote_arm")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lowered = node.name.lower()
            assert not any(b in lowered for b in banned), node.name


def test_the_orchestrator_calls_the_existing_implementation():
    """AST: it must delegate, not re-implement training or artifact logic."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    for required in ("train_stage2_head", "build_stage2_head_artifact",
                     "validate_stage2_head_artifact", "Stage2RepresentationCache",
                     "require_paired_campaign_plan", "stage2_campaign_plan"):
        assert required in called, f"orchestrator does not call {required}"


def test_the_orchestrator_does_not_reimplement_training_or_metrics():
    source = MODULE.read_text(encoding="utf-8")
    for reimplementation in ("CrossEntropyLoss", "backward()", "optimizer.step",
                             "macro_f1(", "deterministic_batches(", "build_head("):
        assert reimplementation not in source, f"{reimplementation} suggests duplication"


def test_the_module_reads_no_dataset():
    source = MODULE.read_text(encoding="utf-8")
    for banned in ("read_csv", "uit-vsfc", "UIT_VSFC", ".csv", "official-test",
                   "OFFICIAL_TEST"):
        assert banned not in source, f"orchestrator references {banned}"
