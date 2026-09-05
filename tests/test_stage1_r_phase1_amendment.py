"""Resource-bounded r-phase1 amendment contract.

These tests use synthetic checkpoint payloads rather than real ``.pt`` files.
The production Colab helper loads the files, then hands the resulting payloads
to the same builder exercised here.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.artifact import (  # noqa: E402
    LOCKED_LR_SELECTION_RULE,
    LR_PILOT_AUTHOR_OVERRIDE_KIND,
    RESOURCE_BOUNDED_R_COMPARISON_WINDOW,
    R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND,
    ArtifactViolation,
    CampaignIdentity,
    validate_selection_artifact,
)
from unmark.stage1.protocol import (  # noqa: E402
    CORRUPTION_SEED,
    ENCODER_REVISION,
    LR_PILOT_GRID,
    LR_PILOT_R,
    R_PHASE1_GRID,
    SELECTION_SEED,
    STAGE1_PROTOCOL_VERSION,
    VALIDATION_CONDITIONS,
    adapter_init_seed,
    lambdas_for_r,
)
from unmark.stage1.r_phase1_amendment import (  # noqa: E402
    RPhase1AmendmentViolation,
    build_resource_bounded_r_phase1_artifact,
    load_r_phase1_last_checkpoints,
)
from unmark.stage1.selection import Candidate, ValidationPoint, select_learning_rate  # noqa: E402
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402


CURRENT_HEAD = "a" * 40
SOURCE_R_HEAD = "b" * 40
CONTROL_HEAD = "c" * 40
DIGEST = "d" * 64

INVENTORY = types.SimpleNamespace(
    source_name="all-vietnamese-syllables.txt",
    source_revision="135a4d9716e49a981624474156d6f247b9b46f6a",
    sha256="78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",
)


WINDOW_SCORES = {
    0.25: (0.31, 0.32, 0.33, 0.34, 0.35, 0.36),
    0.5: (0.21, 0.22, 0.23, 0.24, 0.25, 0.26),
    1.0: (0.10, 0.11, 0.12, 0.13, 0.14, 0.15),
    2.0: (0.22, 0.23, 0.24, 0.25, 0.26, 0.27),
    4.0: (0.41, 0.42, 0.43, 0.44, 0.45, 0.46),
}


def identity(**overrides) -> CampaignIdentity:
    base = CampaignIdentity.from_inputs(
        repository_head=CURRENT_HEAD,
        corpus_manifest_digest=DIGEST,
        encoder_revision=ENCODER_REVISION,
        inventory=INVENTORY,
    )
    return CampaignIdentity(**{**base.__dict__, **overrides})


def point(update: int, score: float, d_clean: float | None = None) -> ValidationPoint:
    return ValidationPoint(
        update=update,
        distances={condition: score for condition in VALIDATION_CONDITIONS},
        d_clean=score / 2.0 if d_clean is None else d_clean,
    )


def execution() -> dict:
    return {
        "backend": "cuda",
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


def checkpoint_payload(r: float, *, head: str = SOURCE_R_HEAD, digest: str = DIGEST) -> dict:
    lambda_align, lambda_clean = lambdas_for_r(r)
    points = [point(0, 0.9).to_dict()]
    points.extend(
        point(update, score).to_dict()
        for update, score in zip(RESOURCE_BOUNDED_R_COMPARISON_WINDOW, WINDOW_SCORES[r])
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": {
            "run_seed": SELECTION_SEED,
            "init_seed": adapter_init_seed(SELECTION_SEED),
            "corruption_seed": CORRUPTION_SEED,
            "learning_rate": 1e-4,
            "r": r,
            "lambda_align": lambda_align,
            "lambda_clean": lambda_clean,
            "corpus_manifest_digest": digest,
            "repository_head": head,
            "backbone_checkpoint": "vinai/phobert-base",
            "backbone_revision": ENCODER_REVISION,
            "protocol_version": STAGE1_PROTOCOL_VERSION,
            "precision": "fp32",
            "inventory": {
                "source_name": INVENTORY.source_name,
                "source_revision": INVENTORY.source_revision,
                "sha256": INVENTORY.sha256,
            },
        },
        "adapter_state": {},
        "optimizer_state": {},
        "global_update": 6500,
        "sampler_state": {},
        "cap": 20000,
        "budget_limited": False,
        "points": points,
        "execution": execution(),
    }


def checkpoint_payloads() -> dict[float, dict]:
    return {float(r): checkpoint_payload(float(r)) for r in R_PHASE1_GRID}


def checkpoint_paths(tmp_path: pathlib.Path) -> dict[float, pathlib.Path]:
    return {
        float(r): tmp_path / f"run-r{r:g}" / "_checkpoint" / "training-checkpoint-last.pt"
        for r in R_PHASE1_GRID
    }


def control_payload() -> dict:
    evaluations = [
        point(update, score).to_dict()
        for update, score in zip(RESOURCE_BOUNDED_R_COMPARISON_WINDOW, WINDOW_SCORES[1.0])
    ]
    # The historical LR-pilot run may have a tail; the amendment compares only
    # the fair window and must not read beyond it.
    evaluations.append(point(7000, 99.0).to_dict())
    return {
        "provenance": {
            "learning_rate": 1e-4,
            "r": 1.0,
            "repository_head": CONTROL_HEAD,
        },
        "evaluations": evaluations,
    }


def build_artifact(tmp_path: pathlib.Path) -> dict:
    return build_resource_bounded_r_phase1_artifact(
        checkpoint_payloads=checkpoint_payloads(),
        checkpoint_paths=checkpoint_paths(tmp_path),
        identity=identity(),
        source_r_phase1_repository_head=SOURCE_R_HEAD,
        reissued_under_repository_head=CURRENT_HEAD,
        fixed_learning_rate=1e-4,
        control_run_payload=control_payload(),
        control_source=tmp_path / "run-lr0.0001.json",
        author="test author",
        created_at="2026-09-05",
    )


def test_resource_bounded_r_artifact_validates_and_selects_author_r(tmp_path):
    artifact = build_artifact(tmp_path)

    winner = validate_selection_artifact(
        artifact,
        expected_stage="r_phase1",
        identity=identity(),
        what="r_phase1.json",
    )

    assert winner.r == 1.0
    assert artifact["selection_override"]["kind"] == R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND
    assert artifact["selection_override"]["observed_cutoff_update"] == 6500
    assert artifact["selection_override"]["original_planned_cap"] == 20000
    assert artifact["selection_override"]["global_optimum_claimed"] is False
    assert artifact["official_test_used"] is False
    assert artifact["downstream_score_used"] is False


def test_helper_fails_if_any_update_6500_durable_state_is_missing(tmp_path):
    with pytest.raises(RPhase1AmendmentViolation, match="missing durable update-6500"):
        load_r_phase1_last_checkpoints(tmp_path)

    payloads = checkpoint_payloads()
    del payloads[4.0]

    with pytest.raises(RPhase1AmendmentViolation, match="exactly r grid"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=payloads,
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(),
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )

    payloads = checkpoint_payloads()
    payloads[4.0]["global_update"] = 6000
    with pytest.raises(RPhase1AmendmentViolation, match="durable checkpoint"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=payloads,
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(),
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )


def test_helper_fails_if_evidence_corpus_or_head_identities_mismatch(tmp_path):
    payloads = checkpoint_payloads()
    payloads[0.25]["provenance"]["corpus_manifest_digest"] = "e" * 64
    with pytest.raises(RPhase1AmendmentViolation, match="corpus_manifest_digest"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=payloads,
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(),
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )

    payloads = checkpoint_payloads()
    payloads[0.25]["provenance"]["repository_head"] = "f" * 40
    with pytest.raises(RPhase1AmendmentViolation, match="repository_head"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=payloads,
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(),
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )


def test_helper_fails_if_r_checkpoint_contains_tail_after_cutoff(tmp_path):
    payloads = checkpoint_payloads()
    payloads[1.0]["points"].append(point(7000, 0.01).to_dict())

    with pytest.raises(RPhase1AmendmentViolation, match="after update 6500"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=payloads,
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(),
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )


def test_selected_r_change_without_matching_amendment_evidence_is_refused(tmp_path):
    artifact = build_artifact(tmp_path)
    r05 = next(candidate for candidate in artifact["candidates"] if candidate["r"] == 0.5)
    artifact["selected"] = copy.deepcopy(r05)
    artifact["selection_override"]["selected_label"] = "r=0.5"
    artifact["selection_override"]["selected_r"] = 0.5

    with pytest.raises(ArtifactViolation, match="resource-bounded summaries select"):
        validate_selection_artifact(
            artifact,
            expected_stage="r_phase1",
            identity=identity(),
            what="r_phase1.json",
        )


def test_normal_historical_r_locked_behavior_remains_fail_closed():
    candidates = [
        Candidate(
            label=f"r={r:g}",
            learning_rate=1e-4,
            r=float(r),
            selected=point(500, {0.25: 0.6, 0.5: 0.4, 1.0: 0.2, 2.0: 0.5, 4.0: 0.8}[r]),
        )
        for r in R_PHASE1_GRID
    ]
    artifact = {
        "stage": "r_phase1",
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": identity().to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected": candidates[1].to_dict(),
    }

    with pytest.raises(ArtifactViolation, match="rerunning the locked selection"):
        validate_selection_artifact(
            artifact,
            expected_stage="r_phase1",
            identity=identity(),
            what="r_phase1.json",
        )


def test_fused_r1_control_evidence_must_be_exact_where_tested(tmp_path):
    bad_control = control_payload()
    for row in bad_control["evaluations"]:
        if row["update"] == 6500:
            row["distances"]["P50"] = row["distances"]["P50"] + 0.001
            row["score"] = max(row["distances"].values())

    with pytest.raises(RPhase1AmendmentViolation, match="differs at update 6500"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=checkpoint_payloads(),
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=bad_control,
            control_source=tmp_path / "run-lr0.0001.json",
            author="test author",
            created_at="2026-09-05",
        )


def lr_artifact() -> dict:
    candidates = [
        Candidate(
            label=f"lr={lr:g}",
            learning_rate=lr,
            r=LR_PILOT_R,
            selected=point(500, {1e-4: 0.1, 3e-4: 0.2, 1e-3: 0.3}[lr]),
        )
        for lr in LR_PILOT_GRID
    ]
    winner = select_learning_rate(candidates)
    artifact = {
        "stage": "lr_pilot",
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": identity().to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected": winner.to_dict(),
    }
    assert winner.learning_rate == 1e-4
    return artifact


def test_final_main_reads_the_validated_r_artifact(tmp_path, monkeypatch):
    import scripts.stage1_runner as runner

    lr_path = tmp_path / "lr_pilot.json"
    r_path = tmp_path / "r_phase1.json"
    lr_path.write_text(json.dumps(lr_artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    r_path.write_text(json.dumps(build_artifact(tmp_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    captured = {}

    def fake_execute(args, schedule, stage, verified):
        captured["stage"] = stage
        captured["schedule"] = schedule
        return 0

    monkeypatch.setattr(runner, "_verified_corpus", lambda args: types.SimpleNamespace(manifest={}))
    monkeypatch.setattr(runner, "_campaign_identity", lambda args, verified: identity())
    monkeypatch.setattr(runner, "_execute", fake_execute)

    args = types.SimpleNamespace(lr_artifact=str(lr_path), r_artifact=str(r_path))
    assert runner.run_final_main(args) == 0

    assert captured["stage"] == "final_main"
    assert [run.learning_rate for run in captured["schedule"]] == [1e-4, 1e-4, 1e-4]
    assert [run.r for run in captured["schedule"]] == [1.0, 1.0, 1.0]


def author_override_for(selected: Candidate, locked: Candidate) -> dict:
    return {
        "kind": LR_PILOT_AUTHOR_OVERRIDE_KIND,
        "author": "test author",
        "created_at": "2026-09-04",
        "selected_label": selected.label,
        "selected_learning_rate": selected.learning_rate,
        "superseded_locked_rule": LOCKED_LR_SELECTION_RULE,
        "superseded_locked_rule_winner": locked.to_dict(),
        "reason": "test override",
        "evidence": {"source": "unit-test"},
    }
