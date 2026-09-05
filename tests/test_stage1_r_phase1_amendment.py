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
    verify_r_phase1_telemetry_evidence,
)
from unmark.stage1.selection import Candidate, ValidationPoint, select_learning_rate  # noqa: E402
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402


CURRENT_HEAD = "a" * 40
SOURCE_R_HEAD = "b" * 40
HISTORICAL_SOURCE_R_HEAD = "3bb2944e6f71865d5a37fe403b78ea640f8a3f1d"
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


def checkpoint_payload(
    r: float,
    *,
    head: str = SOURCE_R_HEAD,
    digest: str = DIGEST,
    scores: dict[float, tuple[float, ...]] | None = None,
) -> dict:
    window = (WINDOW_SCORES if scores is None else scores)[r]
    lambda_align, lambda_clean = lambdas_for_r(r)
    points = [point(0, 0.9).to_dict()]
    points.extend(
        point(update, score).to_dict()
        for update, score in zip(RESOURCE_BOUNDED_R_COMPARISON_WINDOW, window)
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


def checkpoint_payloads(scores: dict[float, tuple[float, ...]] | None = None) -> dict[float, dict]:
    return {
        float(r): checkpoint_payload(float(r), scores=scores) for r in R_PHASE1_GRID
    }


def checkpoint_paths(tmp_path: pathlib.Path) -> dict[float, pathlib.Path]:
    return {
        float(r): tmp_path / f"run-r{r:g}" / "_checkpoint" / "training-checkpoint-last.pt"
        for r in R_PHASE1_GRID
    }


def control_payload(scores: dict[float, tuple[float, ...]] | None = None) -> dict:
    window = (WINDOW_SCORES if scores is None else scores)[1.0]
    evaluations = [
        point(update, score).to_dict()
        for update, score in zip(RESOURCE_BOUNDED_R_COMPARISON_WINDOW, window)
    ]
    # The historical LR-pilot run may have a tail; the amendment compares only
    # the fair window and must not read beyond it.
    evaluations.append(point(7000, 99.0).to_dict())
    return {
        "provenance": {
            "run_seed": SELECTION_SEED,
            "learning_rate": 1e-4,
            "r": 1.0,
            "repository_head": CONTROL_HEAD,
        },
        "evaluations": evaluations,
    }


def write_telemetry(tmp_path: pathlib.Path, *, omit_label: str | None = None) -> pathlib.Path:
    path = tmp_path / "telemetry.jsonl"
    events = [
        {
            "event": "stage_start",
            "stage": "r_phase1",
            "repository_head": SOURCE_R_HEAD,
        }
    ]
    for r in R_PHASE1_GRID:
        label = f"r={r:g}"
        identity_block = {
            "stage": "r_phase1",
            "candidate_index": list(R_PHASE1_GRID).index(r),
            "candidate_count": len(R_PHASE1_GRID),
            "label": label,
            "lr": 1e-4,
            "r": float(r),
            "seed": SELECTION_SEED,
        }
        events.append(
            {
                "event": "run_start",
                "stage": "r_phase1",
                "label": label,
                "lr": 1e-4,
                "r": float(r),
                "repository_head": SOURCE_R_HEAD,
                "execution_mode": "fused",
            }
        )
        if label != omit_label:
            events.extend(
                [
                    {
                        "event": "train_progress",
                        "global_update": 6500,
                        "execution_mode": "fused",
                        "telemetry_identity": identity_block,
                    },
                    {
                        "event": "validation",
                        "update": 6500,
                        "execution_mode": "fused",
                        "telemetry_identity": identity_block,
                    },
                    {
                        "event": "checkpoint",
                        "update": 6500,
                        "checkpoint_name": "training-checkpoint-last.pt",
                        "execution_mode": "fused",
                        "telemetry_identity": identity_block,
                    },
                ]
            )
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def write_raw_historical_telemetry(
    tmp_path: pathlib.Path,
    *,
    mutate_event: str | None = None,
    mutate_label: str = "r=0.25",
    changes: dict | None = None,
    remove: tuple[str, ...] = (),
) -> pathlib.Path:
    path = tmp_path / "raw-historical-telemetry.jsonl"
    events = [
        {
            "schema": "stage1-telemetry-v1",
            "event": "stage_start",
            "stage": "r_phase1",
            "repository_head": HISTORICAL_SOURCE_R_HEAD,
        }
    ]
    for r in R_PHASE1_GRID:
        label = f"r={r:g}"
        common = {
            "schema": "stage1-telemetry-v1",
            "stage": "r_phase1",
            "execution_mode": "fused",
            "label": label,
            "lr": 1e-4,
            "r": float(r),
            "seed": SELECTION_SEED,
        }
        events.extend(
            [
                {
                    **common,
                    "event": "run_start",
                    "repository_head": HISTORICAL_SOURCE_R_HEAD,
                },
                {
                    **common,
                    "event": "train_progress",
                    "global_update": 6500,
                },
                {
                    **common,
                    "event": "validation",
                    "update": 6500,
                },
                {
                    **common,
                    "event": "checkpoint",
                    "update": 6500,
                    "checkpoint_name": "training-checkpoint-last.pt",
                },
            ]
        )
    if mutate_event is not None:
        for event in events:
            if event.get("event") == mutate_event and event.get("label") == mutate_label:
                for key in remove:
                    event.pop(key, None)
                event.update(changes or {})
                break
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def telemetry_evidence(tmp_path: pathlib.Path) -> dict:
    return verify_r_phase1_telemetry_evidence(
        write_telemetry(tmp_path),
        expected_source_repository_head=SOURCE_R_HEAD,
        expected_learning_rate=1e-4,
    )


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
        telemetry_evidence=telemetry_evidence(tmp_path),
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
            telemetry_evidence=telemetry_evidence(tmp_path),
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
            telemetry_evidence=telemetry_evidence(tmp_path),
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
            telemetry_evidence=telemetry_evidence(tmp_path),
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
            telemetry_evidence=telemetry_evidence(tmp_path),
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
            telemetry_evidence=telemetry_evidence(tmp_path),
            author="test author",
            created_at="2026-09-05",
        )


def test_helper_fails_if_telemetry_evidence_missing_update_6500(tmp_path):
    good = verify_r_phase1_telemetry_evidence(
        write_telemetry(tmp_path),
        expected_source_repository_head=SOURCE_R_HEAD,
        expected_learning_rate=1e-4,
    )
    assert set(good["required_events_by_label"]) == {f"r={r:g}" for r in R_PHASE1_GRID}

    with pytest.raises(RPhase1AmendmentViolation, match="r=4.*missing"):
        verify_r_phase1_telemetry_evidence(
            write_telemetry(tmp_path, omit_label="r=4"),
            expected_source_repository_head=SOURCE_R_HEAD,
            expected_learning_rate=1e-4,
        )


def test_raw_historical_r_phase1_telemetry_uses_lr_for_candidate_identity(tmp_path):
    path = write_raw_historical_telemetry(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    run_start = next(
        event
        for event in events
        if event["event"] == "run_start" and event["label"] == "r=0.25"
    )

    assert "telemetry_identity" not in run_start
    assert {
        "event": run_start["event"],
        "execution_mode": run_start["execution_mode"],
        "label": run_start["label"],
        "r": run_start["r"],
        "lr": run_start["lr"],
        "seed": run_start["seed"],
        "repository_head": run_start["repository_head"],
        "stage": run_start["stage"],
        "schema": run_start["schema"],
    } == {
        "event": "run_start",
        "execution_mode": "fused",
        "label": "r=0.25",
        "r": 0.25,
        "lr": 0.0001,
        "seed": 21230,
        "repository_head": HISTORICAL_SOURCE_R_HEAD,
        "stage": "r_phase1",
        "schema": "stage1-telemetry-v1",
    }

    good = verify_r_phase1_telemetry_evidence(
        path,
        expected_source_repository_head=HISTORICAL_SOURCE_R_HEAD,
        expected_learning_rate=1e-4,
    )

    assert set(good["required_events_by_label"]) == {f"r={r:g}" for r in R_PHASE1_GRID}
    assert good["required_events_by_label"]["r=0.25"]["run_start"] is True


@pytest.mark.parametrize(
    ("changes", "remove", "message"),
    [
        ({}, ("lr",), "telemetry r=0.25 lr must be numeric"),
        ({"lr": "not numeric"}, (), "telemetry r=0.25 lr must be numeric"),
        ({"lr": 3e-4}, (), "telemetry r=0.25 lr mismatch"),
        ({"r": 0.5}, (), "telemetry r=0.25 r mismatch"),
        ({"label": "r=bogus"}, (), "telemetry for r=0.25 is missing required event"),
    ],
)
def test_raw_historical_r_phase1_candidate_identity_fails_closed(
    tmp_path,
    changes,
    remove,
    message,
):
    with pytest.raises(RPhase1AmendmentViolation, match=message):
        verify_r_phase1_telemetry_evidence(
            write_raw_historical_telemetry(
                tmp_path,
                mutate_event="run_start",
                changes=changes,
                remove=remove,
            ),
            expected_source_repository_head=HISTORICAL_SOURCE_R_HEAD,
            expected_learning_rate=1e-4,
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
            telemetry_evidence=telemetry_evidence(tmp_path),
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


# ==========================================================================================
# Audit 047 repair - Finding 1: the amendment kind is pinned to ONE decision.
#
# Every other check in this path proves the artifact is INTERNALLY consistent,
# which a coherent forgery also is. These tests build or edit artifacts that pass
# all of those checks and must still be refused.
# ==========================================================================================

# r=0.5 legitimately wins under this table, so the artifact below is coherent:
# its summaries, its recomputed ranking and its selected candidate all agree.
SCORES_FAVOURING_R05 = {
    0.25: (0.31, 0.32, 0.33, 0.34, 0.35, 0.36),
    0.5: (0.10, 0.11, 0.12, 0.13, 0.14, 0.15),
    1.0: (0.21, 0.22, 0.23, 0.24, 0.25, 0.26),
    2.0: (0.22, 0.23, 0.24, 0.25, 0.26, 0.27),
    4.0: (0.41, 0.42, 0.43, 0.44, 0.45, 0.46),
}


def test_a_coherent_artifact_selecting_another_r_is_refused(tmp_path):
    with pytest.raises(ArtifactViolation, match="selected_r must be 1.0"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=checkpoint_payloads(SCORES_FAVOURING_R05),
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=control_payload(SCORES_FAVOURING_R05),
            control_source=tmp_path / "run-lr0.0001.json",
            telemetry_evidence=telemetry_evidence(tmp_path),
            author="test author",
            created_at="2026-09-05",
            selected_r=0.5,
        )


def _rewrite(value, swaps):
    """Recursively rewrite values so an edited artifact stays self-consistent."""
    if isinstance(value, dict):
        return {key: _rewrite(item, swaps) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, swaps) for item in value]
    for old, new in swaps:
        if type(value) is type(old) and value == old:
            return new
    return value


def test_a_coherent_artifact_at_another_learning_rate_is_refused(tmp_path):
    artifact = build_artifact(tmp_path)
    # Move the WHOLE artifact to the historical locked-rule LR. Candidates,
    # summaries, override and control labels all move together, so nothing
    # internal disagrees.
    forged = _rewrite(artifact, [(1e-4, 3e-4), ("lr=0.0001", "lr=0.0003")])
    assert forged["selection_override"]["fixed_learning_rate"] == 3e-4

    with pytest.raises(ArtifactViolation, match="fixed_learning_rate must be 0.0001"):
        validate_selection_artifact(
            forged,
            expected_stage="r_phase1",
            identity=identity(),
            what="r_phase1.json",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observed_cutoff_update", 6000, "observed_cutoff_update must be 6500"),
        ("original_planned_cap", 40000, "original_planned_cap must preserve 20000"),
        (
            "comparison_window",
            [4000, 4500, 5000, 5500, 6000],
            "comparison_window must be",
        ),
        (
            "historical_tail_after_cutoff_used",
            True,
            "historical_tail_after_cutoff_used must be False",
        ),
        ("official_test_used", True, "official_test_used must be False"),
        ("downstream_score_used", True, "downstream_score_used must be False"),
    ],
)
def test_editing_a_pinned_override_field_is_refused(tmp_path, field, value, message):
    artifact = build_artifact(tmp_path)
    artifact["selection_override"][field] = value

    with pytest.raises(ArtifactViolation, match=message):
        validate_selection_artifact(
            artifact,
            expected_stage="r_phase1",
            identity=identity(),
            what="r_phase1.json",
        )


# ==========================================================================================
# Audit 047 repair - Finding 2: the r=1 control is pinned to the LR-pilot seed.
# ==========================================================================================

def test_the_r1_control_must_carry_the_selection_seed(tmp_path):
    wrong_seed = control_payload()
    wrong_seed["provenance"]["run_seed"] = SELECTION_SEED + 1

    with pytest.raises(RPhase1AmendmentViolation, match="is not the historical LR-pilot"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=checkpoint_payloads(),
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=wrong_seed,
            control_source=tmp_path / "run-lr0.0001.json",
            telemetry_evidence=telemetry_evidence(tmp_path),
            author="test author",
            created_at="2026-09-05",
        )


def test_a_control_without_a_seed_is_refused_rather_than_inferred(tmp_path):
    seedless = control_payload()
    del seedless["provenance"]["run_seed"]

    with pytest.raises(RPhase1AmendmentViolation, match="control run_seed must be an integer"):
        build_resource_bounded_r_phase1_artifact(
            checkpoint_payloads=checkpoint_payloads(),
            checkpoint_paths=checkpoint_paths(tmp_path),
            identity=identity(),
            source_r_phase1_repository_head=SOURCE_R_HEAD,
            reissued_under_repository_head=CURRENT_HEAD,
            fixed_learning_rate=1e-4,
            control_run_payload=seedless,
            control_source=tmp_path / "run-lr0.0001.json",
            telemetry_evidence=telemetry_evidence(tmp_path),
            author="test author",
            created_at="2026-09-05",
        )


def test_the_control_seed_is_revalidated_from_the_finished_artifact(tmp_path):
    """The pin must survive into whatever final-main validates, not just the builder."""
    artifact = build_artifact(tmp_path)
    assert artifact["selection_override"]["evidence"]["control_equivalence"][
        "control_run_seed"
    ] == SELECTION_SEED

    artifact["selection_override"]["evidence"]["control_equivalence"][
        "control_run_seed"
    ] = SELECTION_SEED + 1

    with pytest.raises(ArtifactViolation, match="is not the historical LR-pilot"):
        validate_selection_artifact(
            artifact,
            expected_stage="r_phase1",
            identity=identity(),
            what="r_phase1.json",
        )


def test_the_control_window_stops_at_the_cutoff(tmp_path):
    """The historical r=1 tail after 6500 is not selection evidence."""
    artifact = build_artifact(tmp_path)
    control = artifact["selection_override"]["evidence"]["control_equivalence"]

    assert control["comparison_window"] == list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW)
    assert max(int(update) for update in
               control["max_abs_validation_metric_difference_by_update"]) == 6500
    # control_payload() carries a deliberate update-7000 tail; it must not appear.
    assert "7000" not in control["max_abs_validation_metric_difference_by_update"]
    assert set(control["max_abs_validation_metric_difference_by_update"].values()) == {0.0}
