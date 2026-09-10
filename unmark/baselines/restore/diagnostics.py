"""Descriptive restoration diagnostics for completed RESTORE text caches."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.orthography import decompose


RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT = False
RESTORE_DIAGNOSTIC_NORMALIZATION = "NFC"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
        previous = current
    return previous[-1]


def base_form(text: str) -> str:
    return decompose(nfc(text), source_is_clean=False).base_text


@dataclass(frozen=True)
class RestoreDiagnosticRow:
    sample_id: str
    observed_text: str
    restored_text: str
    clean_gold_text: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "observed_text", "restored_text", "clean_gold_text"):
            if not isinstance(getattr(self, name), str):
                raise EvaluationContractViolation(f"{name} must be a string")
        if not self.sample_id:
            raise EvaluationContractViolation("diagnostic sample_id must be non-empty")


def restore_diagnostics(
    rows: Sequence[RestoreDiagnosticRow],
    *,
    condition: str,
) -> dict[str, Any]:
    """Compute descriptive restoration metrics for one finalized condition."""

    if not rows:
        raise EvaluationContractViolation("cannot compute RESTORE diagnostics on no rows")
    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise EvaluationContractViolation("RESTORE diagnostics received duplicate sample ids")

    exact = 0
    output_differs = 0
    base_differs = 0
    char_errors = 0
    char_total = 0
    word_errors = 0
    word_total = 0
    full_unchanged = 0
    for row in rows:
        observed = nfc(row.observed_text)
        restored = nfc(row.restored_text)
        gold = nfc(row.clean_gold_text)
        exact += int(restored == gold)
        output_differs += int(restored != observed)
        base_differs += int(base_form(restored) != base_form(observed))

        char_errors += levenshtein_distance(list(restored), list(gold))
        char_total += len(gold)
        restored_words = restored.split()
        gold_words = gold.split()
        word_errors += levenshtein_distance(restored_words, gold_words)
        word_total += len(gold_words)
        if condition == "FULL":
            full_unchanged += int(restored == observed)

    count = len(rows)
    report: dict[str, Any] = {
        "condition": condition,
        "row_count": count,
        "normalization": RESTORE_DIAGNOSTIC_NORMALIZATION,
        "sentence_exact_match_rate_to_clean_gold": exact / count,
        "character_error_rate_to_clean_gold": char_errors / char_total if char_total else 0.0,
        "word_error_rate_to_clean_gold": word_errors / word_total if word_total else 0.0,
        "fraction_outputs_differing_from_observed_input": output_differs / count,
        "fraction_outputs_whose_stripped_base_differs_from_observed_input": (
            base_differs / count
        ),
        "diagnostics_only": True,
        "selection_input": RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT,
        "diagnostics_used_for_selection": RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT,
    }
    if condition == "FULL":
        report["fraction_restore_leaves_clean_sentence_unchanged"] = full_unchanged / count
        report["fraction_restore_changes_clean_sentence"] = 1.0 - (full_unchanged / count)
    return report


def aggregate_diagnostic_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise EvaluationContractViolation("cannot aggregate no RESTORE diagnostics")
    return {
        "normalization": RESTORE_DIAGNOSTIC_NORMALIZATION,
        "conditions": {str(report["condition"]): dict(report) for report in reports},
        "diagnostics_only": True,
        "selection_input": RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT,
        "diagnostics_used_for_selection": RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT,
    }
