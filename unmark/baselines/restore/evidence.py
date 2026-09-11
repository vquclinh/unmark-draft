"""Read-only RESTORE evidence parsing and GRR derivation helpers."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

from unmark.baselines.restore.cache import file_sha256, read_json
from unmark.baselines.restore.config import (
    RESTORE_BASELINE_SCHEMA_VERSION,
    RESTORE_CONDITIONS,
    RESTORE_DEGRADED_CONDITIONS,
)
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.metrics import UndefinedGRR, gap_recovery_rate


def authenticate_read_only_evidence(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise EvaluationContractViolation(f"read-only evidence not found: {source}")
    actual = file_sha256(source)
    if actual != expected_sha256:
        raise EvaluationContractViolation(
            f"evidence SHA256 mismatch for {source}: expected {expected_sha256}, got {actual}"
        )
    return read_json(source)


def _numeric_mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        if "mean" not in value:
            return None
        value = value["mean"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _condition_row_metrics(row: Any) -> dict[str, float] | None:
    if not isinstance(row, Mapping):
        return None
    if "macro_f1_mean" in row:
        macro = _numeric_mean(row["macro_f1_mean"])
    else:
        macro = _numeric_mean(row.get("macro_f1"))
    if "accuracy_mean" in row:
        acc = _numeric_mean(row["accuracy_mean"])
    else:
        acc = _numeric_mean(row.get("accuracy"))
    if macro is None or acc is None:
        return None
    return {"macro_f1": macro, "accuracy": acc}


def _condition_metrics_from_mapping(payload: Mapping[str, Any]) -> dict[str, dict[str, float]] | None:
    if not isinstance(payload, Mapping):
        return None
    if set(payload) != set(RESTORE_CONDITIONS):
        return None
    result = {}
    for condition in RESTORE_CONDITIONS:
        metrics = _condition_row_metrics(payload[condition])
        if metrics is None:
            return None
        result[condition] = metrics
    return result


def extract_condition_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Find one of the explicitly supported six-condition evidence schemas."""

    metrics = _condition_metrics_from_mapping(payload.get("conditions"))
    if metrics is not None:
        return metrics

    aggregate = (
        payload.get("aggregate")
        if isinstance(payload.get("aggregate"), Mapping)
        else None
    )
    metrics = _condition_metrics_from_mapping(aggregate)
    if metrics is not None:
        return metrics

    aggregate_report = (
        payload.get("aggregate_report")
        if isinstance(payload.get("aggregate_report"), Mapping)
        else None
    )
    aggregate_report_arms = (
        aggregate_report.get("arms") if isinstance(aggregate_report, Mapping) else None
    )
    aggregate_report_unmark_a = (
        aggregate_report_arms.get("UNMARK-A")
        if isinstance(aggregate_report_arms, Mapping)
        else None
    )
    metrics = _condition_metrics_from_mapping(
        aggregate_report_unmark_a.get("conditions")
        if isinstance(aggregate_report_unmark_a, Mapping)
        else None
    )
    if metrics is not None:
        return metrics

    raise EvaluationContractViolation(
        "could not find six-condition aggregate metrics in evidence"
    )


def compute_grr(
    *,
    restore_aggregate: Mapping[str, Any],
    vanilla_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    restore = extract_condition_metrics({"conditions": restore_aggregate["conditions"]})
    vanilla = extract_condition_metrics(vanilla_evidence)
    result: dict[str, Any] = {
        "formula": "(S_RESTORE - S_FLOOR) / (S_UPPER - S_FLOOR)",
        "zero_denominator_policy": "UNDEFINED",
        "no_epsilon": True,
        "no_clipping": True,
        "conditions": {},
    }
    upper_f1 = vanilla["FULL"]["macro_f1"]
    upper_acc = vanilla["FULL"]["accuracy"]
    for condition in RESTORE_DEGRADED_CONDITIONS:
        floor_f1 = vanilla[condition]["macro_f1"]
        floor_acc = vanilla[condition]["accuracy"]
        row: dict[str, Any] = {}
        try:
            row["macro_f1_grr"] = gap_recovery_rate(
                restore[condition]["macro_f1"], floor_f1, upper_f1
            )
        except UndefinedGRR:
            row["macro_f1_grr"] = "UNDEFINED"
        try:
            row["accuracy_grr"] = gap_recovery_rate(
                restore[condition]["accuracy"], floor_acc, upper_acc
            )
        except UndefinedGRR:
            row["accuracy_grr"] = "UNDEFINED"
        result["conditions"][condition] = row

    restore_degraded_f1 = statistics.fmean(
        restore[c]["macro_f1"] for c in RESTORE_DEGRADED_CONDITIONS
    )
    floor_degraded_f1 = statistics.fmean(
        vanilla[c]["macro_f1"] for c in RESTORE_DEGRADED_CONDITIONS
    )
    restore_degraded_acc = statistics.fmean(
        restore[c]["accuracy"] for c in RESTORE_DEGRADED_CONDITIONS
    )
    floor_degraded_acc = statistics.fmean(
        vanilla[c]["accuracy"] for c in RESTORE_DEGRADED_CONDITIONS
    )
    headline: dict[str, Any] = {
        "score_averaging_rule": "average scores over degraded conditions first, apply GRR once",
        "condition_grrs_averaged": False,
    }
    try:
        headline["macro_f1_grr"] = gap_recovery_rate(
            restore_degraded_f1, floor_degraded_f1, upper_f1
        )
    except UndefinedGRR:
        headline["macro_f1_grr"] = "UNDEFINED"
    try:
        headline["accuracy_grr"] = gap_recovery_rate(
            restore_degraded_acc, floor_degraded_acc, upper_acc
        )
    except UndefinedGRR:
        headline["accuracy_grr"] = "UNDEFINED"
    result["headline_degraded"] = headline
    return result


def minimal_comparison(
    *,
    vanilla: Mapping[str, dict[str, float]],
    restore: Mapping[str, dict[str, float]],
    unmark_a: Mapping[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    systems = {"Vanilla": vanilla, "RESTORE": restore}
    if unmark_a is not None:
        systems["UNMARK-A"] = unmark_a
    rows = []
    for system, metrics in systems.items():
        for condition in RESTORE_CONDITIONS:
            rows.append(
                {
                    "system": system,
                    "condition": condition,
                    "macro_f1": metrics[condition]["macro_f1"],
                    "accuracy": metrics[condition]["accuracy"],
                }
            )
        rows.append(
            {
                "system": system,
                "condition": "degraded_mean",
                "macro_f1": statistics.fmean(
                    metrics[c]["macro_f1"] for c in RESTORE_DEGRADED_CONDITIONS
                ),
                "accuracy": statistics.fmean(
                    metrics[c]["accuracy"] for c in RESTORE_DEGRADED_CONDITIONS
                ),
            }
        )
    return {
        "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
        "rows": rows,
        "selection_logic": None,
    }


__all__ = [
    "authenticate_read_only_evidence",
    "compute_grr",
    "extract_condition_metrics",
    "minimal_comparison",
]
