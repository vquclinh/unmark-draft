"""Stage-2 campaign execution over synthetic caches. Needs real torch.

Exercises `run_stage2_campaign` end to end on **synthetic** `[N, 768]` FP32
caches: ten runs, both arms, five seeds, resumable, completed runs never
overwritten. No UIT-VSFC row is read and no model is loaded — the campaign entry
point operates only over already-materialised representation caches.

Per-test `skipif` rather than a module-level `importorskip`, so the tests stay
collected and their skips are visible.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import ordered_id_digest  # noqa: E402
from unmark.evaluation.stage2_campaign import (  # noqa: E402
    STAGE2_CAMPAIGN_CACHE_SLOTS,
    Stage2CampaignRegistry,
    build_stage2_campaign_manifest,
    cache_slot,
    run_stage2_campaign,
)
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    Stage2UnmarkArm,
    finalist_for_arm,
)
from unmark.evaluation.stage2_head_campaign import (  # noqa: E402
    STAGE2_CAMPAIGN_SEEDS,
    STAGE2_CLEAN_CONDITION,
    STAGE2_PROTOCOL_VERSION,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    Stage2RepresentationCache,
    Stage2RepresentationKey,
    label_digest,
)
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
N_TRAIN, N_DEV = 36, 21


def _slot_parts(slot):
    arm, role_value = slot.split("/")
    role = STAGE2_TRAINING_ROLE if role_value == STAGE2_TRAINING_ROLE.value else STAGE2_SELECTION_ROLE
    return arm, role


def build_caches(root):
    """Four synthetic clean caches: {A,B} x {protocol-train, protocol-dev}."""
    keys, directories, labels = {}, {}, {}
    for slot in STAGE2_CAMPAIGN_CACHE_SLOTS:
        arm, role = _slot_parts(slot)
        count = N_TRAIN if role is STAGE2_TRAINING_ROLE else N_DEV
        row_labels = [i % 3 for i in range(count)]
        generator = torch.Generator().manual_seed(abs(hash(slot)) % (2**31))
        values = torch.randn(count, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
        for i, label in enumerate(row_labels):
            values[i, label] += 6.0
        key = Stage2RepresentationKey(
            repository_head=HEAD_SHA,
            arm=arm,
            finalist_checkpoint_sha256=finalist_for_arm(arm).checkpoint_sha256,
            backbone_checkpoint=ENCODER_CHECKPOINT,
            backbone_revision=ENCODER_REVISION,
            protocol_version=STAGE2_PROTOCOL_VERSION,
            dataset="UIT-VSFC", dataset_version="1.0", task="sentiment",
            role=role.value, condition=STAGE2_CLEAN_CONDITION, corruption_seed=None,
            pooling=STAGE2_FIRST_TOKEN_POOLING, max_length=256, truncation=True,
            padding="max_length",
            ordered_id_digest=ordered_id_digest([f"{slot}-{i}" for i in range(count)]),
            label_digest=label_digest(row_labels),
            dtype=STAGE2_REPRESENTATION_DTYPE, hidden_size=HIDDEN_SIZE, count=count,
        )
        directory = pathlib.Path(root) / "caches" / slot.replace("/", "__")
        Stage2RepresentationCache(directory).save(key, values)
        keys[slot], directories[slot], labels[slot] = key, directory, row_labels
    return keys, directories, labels


@requires_torch
def test_a_full_campaign_executes_exactly_ten_runs(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")

    result = run_stage2_campaign(
        manifest, registry, cache_directories=directories, labels=labels,
        repository_head=HEAD_SHA,
    )
    assert len(result["executed"]) == 10
    assert result["status"]["complete"] is True
    assert result["status"]["completed"] == 10
    assert {r["arm"] for r in result["executed"]} == {"UNMARK-A", "UNMARK-B"}
    for arm in ("UNMARK-A", "UNMARK-B"):
        seeds = sorted(r["seed"] for r in result["executed"] if r["arm"] == arm)
        assert seeds == sorted(STAGE2_CAMPAIGN_SEEDS)
    registry.require_campaign_complete(manifest)


@requires_torch
def test_re_entering_a_completed_campaign_executes_nothing(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    run_stage2_campaign(manifest, registry, cache_directories=directories,
                        labels=labels, repository_head=HEAD_SHA)
    again = run_stage2_campaign(manifest, registry, cache_directories=directories,
                                labels=labels, repository_head=HEAD_SHA)
    assert again["executed"] == []
    assert again["status"]["complete"] is True


@requires_torch
def test_a_partial_campaign_resumes_only_its_pending_runs(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")

    first = run_stage2_campaign(
        manifest, registry, cache_directories=directories, labels=labels,
        repository_head=HEAD_SHA, only=list(manifest.expected_runs[:3]),
    )
    assert len(first["executed"]) == 3
    assert first["status"]["complete"] is False
    assert len(registry.pending_runs(manifest)) == 7

    rest = run_stage2_campaign(manifest, registry, cache_directories=directories,
                               labels=labels, repository_head=HEAD_SHA)
    assert len(rest["executed"]) == 7
    assert rest["status"]["complete"] is True


@requires_torch
def test_a_completed_run_is_never_re_executed_or_overwritten(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    run = manifest.expected_runs[0]

    run_stage2_campaign(manifest, registry, cache_directories=directories,
                        labels=labels, repository_head=HEAD_SHA, only=[run])
    store = registry.store_for(run.arm, run.seed)
    before = store.artifact_path.read_bytes()

    with pytest.raises(EvaluationContractViolation, match="immutable"):
        run_stage2_campaign(manifest, registry, cache_directories=directories,
                            labels=labels, repository_head=HEAD_SHA, only=[run])
    assert store.artifact_path.read_bytes() == before


@requires_torch
def test_matched_seeds_produce_paired_but_distinct_heads(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    seed = STAGE2_CAMPAIGN_SEEDS[0]
    pair = [r for r in manifest.expected_runs if r.seed == seed]
    assert len(pair) == 2

    run_stage2_campaign(manifest, registry, cache_directories=directories,
                        labels=labels, repository_head=HEAD_SHA, only=pair)
    a = registry.store_for("UNMARK-A", seed).read_artifact(
        expected_arm="UNMARK-A", expected_seed=seed)
    b = registry.store_for("UNMARK-B", seed).read_artifact(
        expected_arm="UNMARK-B", expected_seed=seed)
    # Same seed -> identical initial head; different caches -> different result.
    assert a["initial_head_fingerprint"] == b["initial_head_fingerprint"]
    assert a["selected_head_state_sha256"] != b["selected_head_state_sha256"]
    assert a["finalist_checkpoint_sha256"] != b["finalist_checkpoint_sha256"]


@requires_torch
def test_a_caller_head_that_disagrees_with_the_campaign_refuses(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    with pytest.raises(EvaluationContractViolation, match="does not match the campaign"):
        run_stage2_campaign(manifest, registry, cache_directories=directories,
                            labels=labels, repository_head="b" * 40)


@requires_torch
def test_a_missing_cache_directory_or_label_vector_refuses(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    slot = STAGE2_CAMPAIGN_CACHE_SLOTS[0]

    thin = {k: v for k, v in directories.items() if k != slot}
    with pytest.raises(EvaluationContractViolation, match="no cache directory"):
        run_stage2_campaign(manifest, registry, cache_directories=thin,
                            labels=labels, repository_head=HEAD_SHA)

    thin_labels = {k: v for k, v in labels.items() if k != slot}
    with pytest.raises(EvaluationContractViolation, match="no labels supplied"):
        run_stage2_campaign(manifest, registry, cache_directories=directories,
                            labels=thin_labels, repository_head=HEAD_SHA)


@requires_torch
def test_a_cache_directory_holding_the_other_arm_refuses(tmp_path):
    """Swapping A's directory into B's slot must fail on the cache key, not train."""
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    swapped = dict(directories)
    swapped[cache_slot(Stage2UnmarkArm.UNMARK_B, STAGE2_TRAINING_ROLE)] = directories[
        cache_slot(Stage2UnmarkArm.UNMARK_A, STAGE2_TRAINING_ROLE)
    ]
    b_runs = [r for r in manifest.expected_runs if r.arm == "UNMARK-B"]
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        run_stage2_campaign(manifest, registry, cache_directories=swapped,
                            labels=labels, repository_head=HEAD_SHA, only=b_runs[:1])


@requires_torch
def test_a_run_outside_the_campaign_refuses(tmp_path):
    from unmark.evaluation.stage2_head_campaign import Stage2CampaignRun

    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    with pytest.raises(EvaluationContractViolation, match="not one of the campaign"):
        run_stage2_campaign(
            manifest, registry, cache_directories=directories, labels=labels,
            repository_head=HEAD_SHA,
            only=[Stage2CampaignRun(arm="UNMARK-A", seed=STAGE2_CAMPAIGN_SEEDS[0] + 1)],
        )


@requires_torch
def test_every_committed_artifact_validates_against_its_own_identity(tmp_path):
    keys, directories, labels = build_caches(tmp_path)
    manifest = build_stage2_campaign_manifest(repository_head=HEAD_SHA, cache_keys=keys)
    registry = Stage2CampaignRegistry(tmp_path / "campaign")
    run_stage2_campaign(manifest, registry, cache_directories=directories,
                        labels=labels, repository_head=HEAD_SHA)
    for run in manifest.expected_runs:
        artifact = registry.store_for(run.arm, run.seed).read_artifact(
            expected_arm=run.arm, expected_seed=run.seed
        )
        assert artifact["repository_head"] == HEAD_SHA
        assert artifact["protocol_version"] == STAGE2_PROTOCOL_VERSION
        assert artifact["ab_selection_performed"] is False
        assert artifact["measurement_used_for_selection"] is False
        # An arm's artifact must not validate as the other arm's.
        other = "UNMARK-B" if run.arm == "UNMARK-A" else "UNMARK-A"
        with pytest.raises(EvaluationContractViolation):
            registry.store_for(run.arm, run.seed).read_artifact(
                expected_arm=other, expected_seed=run.seed
            )
