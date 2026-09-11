#!/usr/bin/env python3
"""Audit064 RESTORE e22 post-measurement closeout.

This command is a provenance-preserving repair for the e22 scientific
execution. It derives GRR/context JSON from immutable evidence that already
exists; it never invokes the normal RESTORE Stage-2 executor and has no data,
model, representation, measurement, or head-training path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from unmark.baselines.restore.cache import file_sha256, read_json, write_json  # noqa: E402
from unmark.baselines.restore.config import (  # noqa: E402
    RESTORE_BASELINE_SCHEMA_VERSION,
    RESTORE_BEST_SEED_SELECTION,
    RESTORE_CONDITIONS,
    RESTORE_DEGRADED_CONDITIONS,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_PROTOCOL_VERSION,
    RESTORE_STAGE2_HEAD_SEEDS,
    RESTORE_UNMARK_A_EVIDENCE_RELATIVE,
    RESTORE_UNMARK_A_EVIDENCE_SHA256,
    RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE,
    RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
    require_full_commit_sha,
)
from unmark.baselines.restore.evidence import (  # noqa: E402
    authenticate_read_only_evidence,
    compute_grr,
    extract_condition_metrics,
    minimal_comparison,
)
from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402


SOURCE_E22_HEAD = "e22c5ea8dbcbea4060193dca3f40fc97694fbb4a"
SOURCE_E22_HEAD_PREFIX = "e22c5ea8dbcb"
E22_RESTORE_MEASUREMENT_SHA256 = (
    "c175d492834d5bb32e9423fd8f14fbd65874f1c22906f014dbd6727a0ba7b38f"
)
RESTORE_SCORE_UNITS = len(RESTORE_STAGE2_HEAD_SEEDS) * len(RESTORE_CONDITIONS)
POSTPROCESS_SCHEMA_VERSION = "restore-e22-postprocess-closeout-v1"
POSTPROCESS_AUDIT_DIR = "audit064-e22-closeout-v1"
POSTPROCESS_FORBIDDEN_DIRS = frozenset(
    {
        "clean-representations",
        "clean-restored-text",
        "heads",
        "measurement",
        "model",
        "representations",
        "restored-text",
        "validation-representations",
        "validation-restored-text",
    }
)


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def require_repair_head(expected_head: str) -> str:
    expected = require_full_commit_sha(expected_head, name="--expected-repair-head")
    actual = _git(["rev-parse", "HEAD"])
    if actual != expected:
        raise EvaluationContractViolation(
            f"HEAD must equal --expected-repair-head {expected}, got {actual}"
        )
    if actual == SOURCE_E22_HEAD:
        raise EvaluationContractViolation(
            "Audit064 post-processing must run from a committed repair HEAD, not the e22 scientific HEAD"
        )
    status = _git(["status", "--short"])
    if status:
        raise EvaluationContractViolation(
            f"git tree must be clean before Audit064 post-processing, got:\n{status}"
        )
    return actual


def e22_measurement_path(drive_root: Path) -> Path:
    return (
        drive_root
        / "stage2-baselines"
        / "restore"
        / SOURCE_E22_HEAD_PREFIX
        / "audit064-restore-v1"
        / "measurement"
        / "restore-measurement.json"
    )


def e22_scientific_namespace(drive_root: Path) -> Path:
    return (
        drive_root
        / "stage2-baselines"
        / "restore"
        / SOURCE_E22_HEAD_PREFIX
        / "audit064-restore-v1"
    )


def postprocess_namespace(drive_root: Path, repair_head: str) -> Path:
    repair_head = require_full_commit_sha(repair_head, name="postprocessing_generated_by_head")
    return (
        drive_root
        / "stage2-baselines"
        / "restore-postprocess"
        / repair_head[:12]
        / POSTPROCESS_AUDIT_DIR
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def require_postprocess_namespace(path: Path, drive_root: Path) -> None:
    allowed_root = drive_root / "stage2-baselines" / "restore-postprocess"
    if not _is_relative_to(path, allowed_root):
        raise EvaluationContractViolation(
            f"post-processing output must live under {allowed_root}, got {path}"
        )
    if _is_relative_to(path, e22_scientific_namespace(drive_root)):
        raise EvaluationContractViolation("post-processing must not write under the e22 RESTORE namespace")
    for forbidden in POSTPROCESS_FORBIDDEN_DIRS:
        if (path / forbidden).exists():
            raise EvaluationContractViolation(
                f"post-processing namespace must not contain scientific artifact directory {forbidden!r}"
            )


def _require_metric_conditions(payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    metrics = extract_condition_metrics(payload)
    if set(metrics) != set(RESTORE_CONDITIONS):
        raise EvaluationContractViolation("RESTORE metric conditions are not exactly the frozen six-condition set")
    return {condition: metrics[condition] for condition in RESTORE_CONDITIONS}


def authenticate_e22_measurement(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationContractViolation(f"e22 RESTORE measurement not found: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != E22_RESTORE_MEASUREMENT_SHA256:
        raise EvaluationContractViolation(
            f"e22 RESTORE measurement SHA256 mismatch: expected {E22_RESTORE_MEASUREMENT_SHA256}, got {actual_sha}"
        )
    payload = read_json(path)
    validate_e22_measurement(payload)
    return payload


def validate_e22_measurement(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESTORE_BASELINE_SCHEMA_VERSION:
        raise EvaluationContractViolation("e22 RESTORE measurement schema_version mismatch")
    if payload.get("protocol_version") != RESTORE_PROTOCOL_VERSION:
        raise EvaluationContractViolation("e22 RESTORE measurement protocol_version mismatch")
    if payload.get("model_id") != RESTORE_MODEL_ID:
        raise EvaluationContractViolation("e22 RESTORE measurement model_id mismatch")
    if payload.get("model_revision") != RESTORE_MODEL_REVISION:
        raise EvaluationContractViolation("e22 RESTORE measurement model_revision mismatch")
    head_field = payload.get("repository_head", payload.get("source_execution_head"))
    if head_field is not None and head_field != SOURCE_E22_HEAD:
        raise EvaluationContractViolation(
            f"e22 RESTORE measurement head mismatch: expected {SOURCE_E22_HEAD}, got {head_field}"
        )
    if payload.get("score_units") != RESTORE_SCORE_UNITS:
        raise EvaluationContractViolation(
            f"e22 RESTORE measurement score_units must be {RESTORE_SCORE_UNITS}"
        )
    if payload.get("expected_score_units") != RESTORE_SCORE_UNITS:
        raise EvaluationContractViolation(
            f"e22 RESTORE measurement expected_score_units must be {RESTORE_SCORE_UNITS}"
        )
    if tuple(payload.get("seeds", ())) != RESTORE_STAGE2_HEAD_SEEDS:
        raise EvaluationContractViolation("e22 RESTORE measurement head seed set mismatch")
    if payload.get("best_seed_selection") != RESTORE_BEST_SEED_SELECTION:
        raise EvaluationContractViolation("e22 RESTORE measurement best_seed_selection mismatch")

    conditions = payload.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(RESTORE_CONDITIONS):
        raise EvaluationContractViolation("e22 RESTORE measurement conditions are not exactly frozen six")
    _require_metric_conditions({"conditions": conditions})

    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != RESTORE_SCORE_UNITS:
        raise EvaluationContractViolation("e22 RESTORE measurement per_seed grid does not contain 30 score units")
    seen: set[tuple[int, str]] = set()
    for row in per_seed:
        if not isinstance(row, Mapping):
            raise EvaluationContractViolation("e22 RESTORE measurement per_seed row is malformed")
        seed = row.get("seed")
        condition = row.get("condition")
        if seed not in RESTORE_STAGE2_HEAD_SEEDS or condition not in RESTORE_CONDITIONS:
            raise EvaluationContractViolation("e22 RESTORE measurement per_seed identity is outside the frozen grid")
        seen.add((int(seed), str(condition)))
    expected = {(seed, condition) for seed in RESTORE_STAGE2_HEAD_SEEDS for condition in RESTORE_CONDITIONS}
    if seen != expected:
        raise EvaluationContractViolation("e22 RESTORE measurement per_seed grid is incomplete")


def derive_postprocess_payloads(
    *,
    measurement: Mapping[str, Any],
    vanilla_evidence: Mapping[str, Any],
    unmark_a_evidence: Mapping[str, Any] | None,
    drive_root: Path,
    repair_head: str,
) -> dict[str, Any]:
    restore_metrics = _require_metric_conditions({"conditions": measurement["conditions"]})
    vanilla_metrics = _require_metric_conditions(vanilla_evidence)
    unmark_metrics = _require_metric_conditions(unmark_a_evidence) if unmark_a_evidence is not None else None
    grr = compute_grr(restore_aggregate=measurement, vanilla_evidence=vanilla_evidence)
    comparison = minimal_comparison(
        vanilla=vanilla_metrics,
        restore=restore_metrics,
        unmark_a=unmark_metrics,
    )
    closeout = {
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "postprocess_kind": "audit064-e22-grr-evidence-schema-repair",
        "scientific_measurement_generated_by_head": SOURCE_E22_HEAD,
        "postprocessing_generated_by_head": repair_head,
        "cross_head_scientific_execution_reuse": "NO",
        "historical_measurement_consumed_read_only": "YES",
        "source_e22_namespace": str(e22_scientific_namespace(drive_root)),
        "postprocess_namespace": str(postprocess_namespace(drive_root, repair_head)),
        "restore_measurement": {
            "path": str(e22_measurement_path(drive_root)),
            "sha256": E22_RESTORE_MEASUREMENT_SHA256,
            "score_units": RESTORE_SCORE_UNITS,
            "conditions": list(RESTORE_CONDITIONS),
            "payload": dict(measurement),
        },
        "vanilla_evidence": {
            "path": str(drive_root / RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE),
            "sha256": RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
        },
        "unmark_a_evidence": (
            {
                "path": str(drive_root / RESTORE_UNMARK_A_EVIDENCE_RELATIVE),
                "sha256": RESTORE_UNMARK_A_EVIDENCE_SHA256,
            }
            if unmark_a_evidence is not None
            else None
        ),
        "measurement_internal_identity_checked": {
            "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
            "protocol_version": RESTORE_PROTOCOL_VERSION,
            "model_id": RESTORE_MODEL_ID,
            "model_revision": RESTORE_MODEL_REVISION,
            "head_field_present": (
                "repository_head" in measurement or "source_execution_head" in measurement
            ),
            "score_units": RESTORE_SCORE_UNITS,
            "seeds": list(RESTORE_STAGE2_HEAD_SEEDS),
            "conditions": list(RESTORE_CONDITIONS),
            "per_seed_grid_units": RESTORE_SCORE_UNITS,
            "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
        },
        "boundaries": {
            "postprocessing_only_json_reads": "YES",
            "model_loading_reachable": "NO",
            "phobert_loading_reachable": "NO",
            "head_training_reachable": "NO",
            "dataset_reading_reachable": "NO",
            "measurement_recomputed": "NO",
            "official_test_read": "NO",
        },
        "grr_formula": "(S_RESTORE - S_FLOOR) / (S_UPPER - S_FLOOR)",
        "headline_grr_semantics": "average degraded RESTORE scores and degraded Vanilla floors first, then apply ratio once",
        "grr": grr,
        "comparison": comparison,
    }
    return {"grr": grr, "comparison": comparison, "closeout": closeout}


def run_postprocess(
    *,
    drive_root: Path,
    repair_head: str,
    include_unmark_a: bool,
) -> dict[str, Any]:
    output_root = postprocess_namespace(drive_root, repair_head)
    require_postprocess_namespace(output_root, drive_root)
    output_files = [
        output_root / "grr" / "restore-grr.json",
        output_root / "comparison" / "restore-context-comparison.json",
        output_root / "evidence" / "restore-e22-postprocess-closeout-v1.json",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise EvaluationContractViolation(
            "post-processing outputs already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    measurement = authenticate_e22_measurement(e22_measurement_path(drive_root))
    vanilla = authenticate_read_only_evidence(
        drive_root / RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE,
        RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
    )
    unmark_a = None
    if include_unmark_a:
        unmark_a = authenticate_read_only_evidence(
            drive_root / RESTORE_UNMARK_A_EVIDENCE_RELATIVE,
            RESTORE_UNMARK_A_EVIDENCE_SHA256,
        )

    payloads = derive_postprocess_payloads(
        measurement=measurement,
        vanilla_evidence=vanilla,
        unmark_a_evidence=unmark_a,
        drive_root=drive_root,
        repair_head=repair_head,
    )
    write_json(output_files[0], payloads["grr"])
    write_json(output_files[1], payloads["comparison"])
    write_json(output_files[2], payloads["closeout"])
    return {
        "output_root": str(output_root),
        "grr": {"path": str(output_files[0]), "sha256": file_sha256(output_files[0])},
        "comparison": {"path": str(output_files[1]), "sha256": file_sha256(output_files[1])},
        "closeout": {"path": str(output_files[2]), "sha256": file_sha256(output_files[2])},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument(
        "--run-e22-postprocess",
        action="store_true",
        help="derive Audit064 GRR/final closeout from immutable e22 JSON evidence only",
    )
    parser.add_argument("--expected-repair-head", required=True)
    parser.add_argument(
        "--skip-unmark-a-comparison",
        action="store_true",
        help="omit contextual UNMARK-A comparison and therefore do not read UNMARK-A evidence",
    )
    return parser


def print_success(result: Mapping[str, Any]) -> None:
    print("RESTORE_E22_POSTPROCESS=PASS")
    print(f"OUTPUT_ROOT={result['output_root']}")
    print(f"RESTORE_GRR={result['grr']['path']}")
    print(f"RESTORE_GRR_SHA256={result['grr']['sha256']}")
    print(f"RESTORE_CONTEXT_COMPARISON={result['comparison']['path']}")
    print(f"RESTORE_CONTEXT_COMPARISON_SHA256={result['comparison']['sha256']}")
    print(f"RESTORE_E22_POSTPROCESS_CLOSEOUT={result['closeout']['path']}")
    print(f"RESTORE_E22_POSTPROCESS_CLOSEOUT_SHA256={result['closeout']['sha256']}")
    print("SCIENTIFIC_MEASUREMENT_GENERATED_BY_HEAD=" + SOURCE_E22_HEAD)
    print("HISTORICAL_MEASUREMENT_CONSUMED_READ_ONLY=YES")
    print("CROSS_HEAD_SCIENTIFIC_EXECUTION_REUSE=NO")
    print("MEASUREMENT_RECOMPUTED=NO")
    print("OFFICIAL_TEST_READ=NO")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_e22_postprocess:
        parser.error("--run-e22-postprocess is required for the Audit064 recovery executor")
    try:
        repair_head = require_repair_head(args.expected_repair_head)
        result = run_postprocess(
            drive_root=Path(args.drive_root),
            repair_head=repair_head,
            include_unmark_a=not args.skip_unmark_a_comparison,
        )
    except EvaluationContractViolation as exc:
        print(f"RESTORE_E22_POSTPROCESS=FAIL: {exc}", file=sys.stderr)
        return 2
    print_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
