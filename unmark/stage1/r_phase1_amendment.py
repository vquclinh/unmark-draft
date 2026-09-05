"""Resource-bounded r-phase1 handoff reconstruction.

This module is the reusable core behind the Colab reissue cell. It treats the
stopped per-r checkpoints as read-only evidence, verifies that they belong to
the expected campaign, recomputes the observed-window summaries, and produces
the explicit r-phase1 amendment artifact that final-main validates through
``unmark.stage1.artifact``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.stage1.artifact import (
    CampaignIdentity,
    LOCKED_R_SELECTION_RULE,
    RESOURCE_BOUNDED_R_COMPARISON_WINDOW,
    RESOURCE_BOUNDED_R_CUTOFF_UPDATE,
    RESOURCE_BOUNDED_R_PRIMARY_CRITERION,
    RESOURCE_BOUNDED_R_SCORE_STD_KIND,
    RESOURCE_BOUNDED_R_SECONDARY_DIAGNOSTIC,
    RESOURCE_BOUNDED_R_SELECTION_RULE,
    RESOURCE_BOUNDED_R_STABILITY_DIAGNOSTICS,
    R_PHASE1_CONTROL_EQUIVALENCE_KIND,
    R_PHASE1_CONTROL_METRICS,
    R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND,
    resource_bounded_r_order,
    resource_bounded_r_summary,
    validate_selection_artifact,
)
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.device import (
    DETERMINISTIC_CUBLAS_WORKSPACE,
    FLOAT32_MATMUL_PRECISION,
    SCIENTIFIC_DEVICE_BACKEND,
)
from unmark.stage1.protocol import (
    CORRUPTION_SEED,
    ENCODER_CHECKPOINT,
    INITIAL_MAX_UPDATES,
    PRECISION,
    R_PHASE1_GRID,
    SELECTION_SEED,
    STAGE1_PROTOCOL_VERSION,
    VALIDATION_CONDITIONS,
    adapter_init_seed,
    lambdas_for_r,
)
from unmark.stage1.selection import (
    Candidate,
    ValidationPoint,
    select_checkpoint,
)
from unmark.stage1.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    LAST_CHECKPOINT_NAME,
    REQUIRED_CHECKPOINT_KEYS,
)


class RPhase1AmendmentViolation(Stage1ContractViolation):
    """Raised when stopped r-phase1 evidence cannot support the handoff."""


def r_label(r: float) -> str:
    """The production label for one r-phase1 candidate."""
    return f"r={r:g}"


def r_checkpoint_path(r_phase1_dir: Path, r: float) -> Path:
    """The production last-checkpoint path for one r-phase1 candidate."""
    label = r_label(r)
    return Path(r_phase1_dir) / f"run-{label.replace('=', '')}" / "_checkpoint" / LAST_CHECKPOINT_NAME


def load_r_phase1_last_checkpoints(
    r_phase1_dir: Path,
) -> tuple[dict[float, Mapping[str, Any]], dict[float, Path]]:
    """Load the five read-only update-6500 checkpoint payloads.

    Torch is imported only through ``load_training_checkpoint`` and only when a
    caller explicitly asks to load real ``.pt`` files.
    """
    from unmark.stage1.trainer import load_training_checkpoint

    payloads: dict[float, Mapping[str, Any]] = {}
    paths: dict[float, Path] = {}
    for r in R_PHASE1_GRID:
        path = r_checkpoint_path(Path(r_phase1_dir), r)
        if not path.is_file():
            raise RPhase1AmendmentViolation(
                f"missing durable update-{RESOURCE_BOUNDED_R_CUTOFF_UPDATE} "
                f"checkpoint for {r_label(r)}: {path}"
            )
        payload = load_training_checkpoint(path.parent)
        if payload is None:
            raise RPhase1AmendmentViolation(f"checkpoint could not be loaded: {path}")
        payloads[float(r)] = payload
        paths[float(r)] = path
    return payloads, paths


def build_resource_bounded_r_phase1_artifact(
    *,
    checkpoint_payloads: Mapping[float, Mapping[str, Any]],
    checkpoint_paths: Mapping[float, Path | str],
    identity: CampaignIdentity,
    source_r_phase1_repository_head: str,
    reissued_under_repository_head: str,
    fixed_learning_rate: float,
    control_run_payload: Mapping[str, Any],
    control_source: Path | str,
    author: str,
    created_at: str,
    previous_artifact: Mapping[str, Any] | None = None,
    selected_r: float = 1.0,
) -> dict[str, Any]:
    """Build and self-validate the resource-bounded r-phase1 artifact."""
    source_head = _full_sha(source_r_phase1_repository_head, "source r-phase1 head")
    current_head = _full_sha(reissued_under_repository_head, "reissued head")
    if identity.repository_head != current_head:
        raise RPhase1AmendmentViolation(
            f"artifact identity head {identity.repository_head!r} does not match "
            f"the requested reissue head {current_head!r}"
        )
    if not author.strip() or not created_at.strip():
        raise RPhase1AmendmentViolation("author and created_at must be non-empty")

    payloads = _normalise_r_mapping(checkpoint_payloads, "checkpoint payload")
    paths = _normalise_r_mapping(checkpoint_paths, "checkpoint path")

    candidates: list[Candidate] = []
    summaries: list[dict[str, Any]] = []
    window_points_by_label: dict[str, list[ValidationPoint]] = {}
    for r in R_PHASE1_GRID:
        candidate, summary, window_points = _candidate_and_summary_from_checkpoint(
            r=float(r),
            payload=payloads[float(r)],
            source_path=str(paths[float(r)]),
            fixed_learning_rate=fixed_learning_rate,
            identity=identity,
            source_repository_head=source_head,
        )
        candidates.append(candidate)
        summaries.append(summary)
        window_points_by_label[candidate.label] = window_points

    order = resource_bounded_r_order(summaries)
    selected_label = r_label(selected_r)
    if order[0] != selected_label:
        raise RPhase1AmendmentViolation(
            f"resource-bounded r rule selects {order[0]}, not {selected_label}; "
            "refusing to reissue a handoff that contradicts the observed-window evidence"
        )
    selected = next(candidate for candidate in candidates if candidate.r == selected_r)

    previous_head = None
    previous_identity_head = None
    if previous_artifact is not None:
        previous_head = previous_artifact.get("repository_head")
        identity_block = previous_artifact.get("identity")
        if isinstance(identity_block, Mapping):
            previous_identity_head = identity_block.get("repository_head")

    artifact = {
        "stage": "r_phase1",
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": identity.to_dict(),
        "repository_head": current_head,
        "corpus_manifest_digest": identity.corpus_manifest_digest,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected": selected.to_dict(),
        "preparation": {
            "candidate_execution": "fused",
            "reissue_helper": "resource_bounded_r_phase1_amendment",
            "training_not_rerun": True,
        },
        "raw_text_persisted": False,
        "official_test_used": False,
        "downstream_score_used": False,
        "selection_override": {
            "kind": R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND,
            "author": author,
            "created_at": created_at,
            "selected_label": selected.label,
            "selected_r": selected.r,
            "fixed_learning_rate": float(fixed_learning_rate),
            "resource_bounded_selection_rule": RESOURCE_BOUNDED_R_SELECTION_RULE,
            "original_locked_r_selection_rule": LOCKED_R_SELECTION_RULE,
            "original_planned_cap": INITIAL_MAX_UPDATES,
            "observed_cutoff_update": RESOURCE_BOUNDED_R_CUTOFF_UPDATE,
            "comparison_window": list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW),
            "primary_criterion": RESOURCE_BOUNDED_R_PRIMARY_CRITERION,
            "secondary_diagnostic": RESOURCE_BOUNDED_R_SECONDARY_DIAGNOSTIC,
            "stability_diagnostics": list(RESOURCE_BOUNDED_R_STABILITY_DIAGNOSTICS),
            "stopped_resource_bounded": True,
            "global_optimum_claimed": False,
            "official_test_used": False,
            "downstream_score_used": False,
            "historical_tail_after_cutoff_used": False,
            "reason": (
                "Resource-bounded author amendment after observing partial "
                "r-phase1 validation curves. All five r candidates had durable "
                "update-6500 evidence, so r=1.0 is adopted only under the "
                "documented observed-window rule; this is not a completed "
                "20,000-update r-phase1 result and not a claim of global optimality."
            ),
            "evidence": {
                "source_r_phase1_repository_head": source_head,
                "reissued_under_repository_head": current_head,
                "previous_artifact_repository_head": previous_head,
                "previous_artifact_identity_repository_head": previous_identity_head,
                "candidate_summaries": summaries,
                "resource_bounded_order": order,
                "control_equivalence": r1_control_equivalence(
                    r1_points=window_points_by_label["r=1"],
                    control_run_payload=control_run_payload,
                    control_source=str(control_source),
                ),
            },
        },
    }

    winner = validate_selection_artifact(
        artifact,
        expected_stage="r_phase1",
        identity=identity,
        what="resource-bounded r_phase1 artifact",
    )
    if winner.r != selected_r:
        raise RPhase1AmendmentViolation(
            f"self-validation returned r={winner.r}, not r={selected_r}"
        )
    return artifact


def r1_control_equivalence(
    *,
    r1_points: Sequence[ValidationPoint],
    control_run_payload: Mapping[str, Any],
    control_source: str,
) -> dict[str, Any]:
    """Verify fused r=1 matches the historical LR-pilot r=1 window exactly."""
    if tuple(point.update for point in r1_points) != RESOURCE_BOUNDED_R_COMPARISON_WINDOW:
        raise RPhase1AmendmentViolation("r=1 control input does not cover the comparison window")
    provenance = _mapping(control_run_payload.get("provenance"), "control provenance")
    if float(provenance.get("learning_rate")) != 1e-4 or float(provenance.get("r")) != 1.0:
        raise RPhase1AmendmentViolation(
            "control run must be the historical lr=0.0001,r=1 candidate"
        )
    control_head = _full_sha(
        provenance.get("repository_head"),
        "control repository head",
    )
    control_points = _points_by_update(
        control_run_payload.get("evaluations"),
        "control evaluations",
    )
    diffs: dict[str, float] = {}
    for point in r1_points:
        other = control_points.get(point.update)
        if other is None:
            raise RPhase1AmendmentViolation(
                f"control run is missing update {point.update}"
            )
        diff = _max_validation_metric_difference(point, other)
        if diff != 0.0:
            raise RPhase1AmendmentViolation(
                f"r=1 fused/control validation differs at update {point.update}: {diff}"
            )
        diffs[str(point.update)] = diff
    return {
        "kind": R_PHASE1_CONTROL_EQUIVALENCE_KIND,
        "r_phase1_candidate_label": "r=1",
        "control_candidate_label": "lr=0.0001",
        "control_source": str(control_source),
        "control_repository_head": control_head,
        "comparison_window": list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW),
        "metrics_compared": list(R_PHASE1_CONTROL_METRICS),
        "max_abs_validation_metric_difference_by_update": diffs,
    }


def assert_expected_resource_bounded_summaries(
    summaries: Sequence[Mapping[str, Any]],
    expected: Mapping[float | str, Mapping[str, float]],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 5e-10,
) -> None:
    """Compare recomputed summaries with the externally recorded observed table."""
    by_r = {float(summary["r"]): summary for summary in summaries}
    expected_by_r = {float(r): values for r, values in expected.items()}
    if sorted(by_r) != sorted(expected_by_r):
        raise RPhase1AmendmentViolation(
            f"expected summary grid {sorted(expected_by_r)} does not match "
            f"computed grid {sorted(by_r)}"
        )
    for r, fields in sorted(expected_by_r.items()):
        for field, wanted in fields.items():
            if field not in by_r[r]:
                raise RPhase1AmendmentViolation(f"computed summary for r={r:g} has no {field}")
            observed = float(by_r[r][field])
            if not math.isclose(observed, float(wanted), rel_tol=rel_tol, abs_tol=abs_tol):
                raise RPhase1AmendmentViolation(
                    f"r={r:g} {field} recomputed as {observed!r}, expected {wanted!r}"
                )


def _candidate_and_summary_from_checkpoint(
    *,
    r: float,
    payload: Mapping[str, Any],
    source_path: str,
    fixed_learning_rate: float,
    identity: CampaignIdentity,
    source_repository_head: str,
) -> tuple[Candidate, dict[str, Any], list[ValidationPoint]]:
    _require_checkpoint_payload(
        r=r,
        payload=payload,
        fixed_learning_rate=fixed_learning_rate,
        identity=identity,
        source_repository_head=source_repository_head,
    )
    points = _points_from_checkpoint(payload, r_label(r))
    if any(point.update > RESOURCE_BOUNDED_R_CUTOFF_UPDATE for point in points):
        raise RPhase1AmendmentViolation(
            f"{r_label(r)} checkpoint contains validation after update "
            f"{RESOURCE_BOUNDED_R_CUTOFF_UPDATE}; the amendment must not use a tail"
        )
    by_update = {point.update: point for point in points}
    window_points = [
        by_update[update] for update in RESOURCE_BOUNDED_R_COMPARISON_WINDOW
        if update in by_update
    ]
    if len(window_points) != len(RESOURCE_BOUNDED_R_COMPARISON_WINDOW):
        missing = sorted(set(RESOURCE_BOUNDED_R_COMPARISON_WINDOW) - set(by_update))
        raise RPhase1AmendmentViolation(
            f"{r_label(r)} is missing validation update(s) {missing}"
        )
    candidate = Candidate(
        label=r_label(r),
        learning_rate=float(fixed_learning_rate),
        r=float(r),
        selected=select_checkpoint(points),
        budget_limited=bool(payload.get("budget_limited", False)),
    )
    summary = resource_bounded_r_summary(
        label=candidate.label,
        r=candidate.r,
        learning_rate=candidate.learning_rate,
        points=window_points,
        source_checkpoint=source_path,
        source_repository_head=source_repository_head,
        checkpoint_schema_version=str(payload["schema_version"]),
        checkpoint_global_update=int(payload["global_update"]),
        checkpoint_cap=int(payload["cap"]),
    )
    return candidate, summary, window_points


def _require_checkpoint_payload(
    *,
    r: float,
    payload: Mapping[str, Any],
    fixed_learning_rate: float,
    identity: CampaignIdentity,
    source_repository_head: str,
) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    label = r_label(r)
    if missing:
        raise RPhase1AmendmentViolation(f"{label} checkpoint is missing {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise RPhase1AmendmentViolation(
            f"{label} checkpoint schema {payload['schema_version']!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION!r}"
        )
    global_update = _int_field(payload["global_update"], f"{label} global_update")
    cap = _int_field(payload["cap"], f"{label} cap")
    if global_update != RESOURCE_BOUNDED_R_CUTOFF_UPDATE:
        raise RPhase1AmendmentViolation(
            f"{label} durable checkpoint is at update {global_update!r}, "
            f"not {RESOURCE_BOUNDED_R_CUTOFF_UPDATE}"
        )
    if cap != INITIAL_MAX_UPDATES:
        raise RPhase1AmendmentViolation(
            f"{label} checkpoint cap {cap!r} does not preserve the "
            f"original {INITIAL_MAX_UPDATES} plan"
        )
    if payload.get("budget_limited") is not False:
        raise RPhase1AmendmentViolation(f"{label} must not be marked budget_limited at 6500")

    provenance = _mapping(payload.get("provenance"), f"{label} provenance")
    exact = {
        "run_seed": SELECTION_SEED,
        "init_seed": adapter_init_seed(SELECTION_SEED),
        "corruption_seed": CORRUPTION_SEED,
        "corpus_manifest_digest": identity.corpus_manifest_digest,
        "repository_head": source_repository_head,
        "backbone_checkpoint": ENCODER_CHECKPOINT,
        "backbone_revision": identity.encoder_revision,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "precision": PRECISION,
    }
    for field, expected in exact.items():
        if provenance.get(field) != expected:
            raise RPhase1AmendmentViolation(
                f"{label} provenance mismatch on {field}: "
                f"{provenance.get(field)!r} != {expected!r}"
            )
    if not math.isclose(float(provenance.get("learning_rate")), fixed_learning_rate):
        raise RPhase1AmendmentViolation(
            f"{label} learning_rate {provenance.get('learning_rate')!r} "
            f"!= frozen LR {fixed_learning_rate!r}"
        )
    if not math.isclose(float(provenance.get("r")), r):
        raise RPhase1AmendmentViolation(
            f"{label} r {provenance.get('r')!r} != expected {r!r}"
        )
    lambda_align, lambda_clean = lambdas_for_r(r)
    if not math.isclose(float(provenance.get("lambda_align")), lambda_align):
        raise RPhase1AmendmentViolation(f"{label} lambda_align is inconsistent with r")
    if not math.isclose(float(provenance.get("lambda_clean")), lambda_clean):
        raise RPhase1AmendmentViolation(f"{label} lambda_clean is inconsistent with r")

    inventory = _mapping(provenance.get("inventory"), f"{label} inventory")
    inventory_expected = {
        "source_name": identity.inventory_source_name,
        "source_revision": identity.inventory_source_revision,
        "sha256": identity.inventory_sha256,
    }
    for field, expected in inventory_expected.items():
        if expected is None:
            raise RPhase1AmendmentViolation(
                f"current campaign identity has no inventory {field}; cannot "
                "validate stopped r-phase1 checkpoints"
            )
        if inventory.get(field) != expected:
            raise RPhase1AmendmentViolation(
                f"{label} inventory mismatch on {field}: "
                f"{inventory.get(field)!r} != {expected!r}"
            )

    _require_execution_policy(payload.get("execution"), label)


def _require_execution_policy(execution: Any, label: str) -> None:
    execution = _mapping(execution, f"{label} execution")
    expected = {
        "backend": SCIENTIFIC_DEVICE_BACKEND,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": FLOAT32_MATMUL_PRECISION,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for field, value in expected.items():
        if execution.get(field) != value:
            raise RPhase1AmendmentViolation(
                f"{label} execution mismatch on {field}: "
                f"{execution.get(field)!r} != {value!r}"
            )
    if execution.get("cublas_workspace_config") not in {DETERMINISTIC_CUBLAS_WORKSPACE, ":16:8"}:
        raise RPhase1AmendmentViolation(
            f"{label} execution does not record deterministic CUBLAS workspace"
        )


def _points_from_checkpoint(payload: Mapping[str, Any], label: str) -> list[ValidationPoint]:
    raw = payload.get("points")
    if not isinstance(raw, list):
        raise RPhase1AmendmentViolation(f"{label} checkpoint points must be a list")
    points = [ValidationPoint.from_dict(point) for point in raw]
    updates = [point.update for point in points]
    if RESOURCE_BOUNDED_R_CUTOFF_UPDATE not in updates:
        raise RPhase1AmendmentViolation(f"{label} has no validation at update 6500")
    if len(set(updates)) != len(updates):
        raise RPhase1AmendmentViolation(f"{label} has duplicate validation updates")
    return points


def _points_by_update(raw: Any, what: str) -> dict[int, ValidationPoint]:
    if not isinstance(raw, list):
        raise RPhase1AmendmentViolation(f"{what} must be a list")
    points = [ValidationPoint.from_dict(point) for point in raw]
    by_update = {point.update: point for point in points}
    if len(by_update) != len(points):
        raise RPhase1AmendmentViolation(f"{what} contains duplicate updates")
    return by_update


def _max_validation_metric_difference(a: ValidationPoint, b: ValidationPoint) -> float:
    differences = [
        abs(a.d_clean - b.d_clean),
        abs(a.score - b.score),
    ]
    differences.extend(
        abs(float(a.distances[condition]) - float(b.distances[condition]))
        for condition in VALIDATION_CONDITIONS
    )
    return max(differences)


def _normalise_r_mapping(raw: Mapping[float, Any], what: str) -> dict[float, Any]:
    normalised: dict[float, Any] = {}
    for key, value in raw.items():
        r = float(key)
        if r in normalised:
            raise RPhase1AmendmentViolation(f"duplicate {what} for r={r:g}")
        normalised[r] = value
    if sorted(normalised) != sorted(float(r) for r in R_PHASE1_GRID):
        raise RPhase1AmendmentViolation(
            f"{what}s must cover exactly r grid {list(R_PHASE1_GRID)}, got "
            f"{sorted(normalised)}"
        )
    return normalised


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RPhase1AmendmentViolation(f"{what} must be a JSON object")
    return value


def _int_field(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RPhase1AmendmentViolation(f"{what} must be an integer")
    return value


def _full_sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise RPhase1AmendmentViolation(f"{what} must be a full 40-character sha")
    try:
        int(value, 16)
    except ValueError as error:
        raise RPhase1AmendmentViolation(f"{what} must be hexadecimal") from error
    return value.lower()
