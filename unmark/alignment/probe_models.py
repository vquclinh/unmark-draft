"""Probe result structures and cross-condition analysis (B3B-0).

What the Colab probe records, and the metrics that decide whether the base token
grid actually is corruption-invariant. Pure standard library.

The load-bearing check is `grid_invariance`. Proposal §4.4 asserts that
"corrupting the input changes the tone labels but never the base ids", and §4.5
that "`b(x) = b(x̃)` for every corruption rate, [so] UNMARK sees the *same* token
grid whatever the input condition". B2 already guarantees the *base strings* are
equal. What is unverified is whether everything downstream of that -- word
segmentation especially -- preserves the equality all the way to token ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from unmark.alignment.contracts import AlignmentStatus, OffsetAvailability, PathAvailability, PreprocessingPath
from unmark.alignment.spans import TokenSpan


@dataclass(frozen=True)
class PathObservation:
    """One (case, condition, preprocessing path) observation."""

    case_id: str
    condition: str
    path: PreprocessingPath
    availability: PathAvailability

    source_text: str = ""
    canonical_text: str = ""
    base_text: str = ""
    segmented_text: str | None = None
    tokenizer_input: str = ""

    tokens: tuple[str, ...] = ()
    token_ids: tuple[int, ...] = ()
    spans: tuple[TokenSpan, ...] = ()

    offset_availability: OffsetAvailability = OffsetAvailability.NOT_PROBED
    offset_reason: str = ""
    alignment: AlignmentStatus = AlignmentStatus.NOT_ATTEMPTED
    alignment_reason: str = ""

    coverage: dict[str, Any] = field(default_factory=dict)
    syllable_map: dict[str, Any] = field(default_factory=dict)
    eligibility: tuple[str, ...] = ()
    observed_tones: tuple[str, ...] = ()
    error: str | None = None

    @property
    def content_tokens(self) -> tuple[TokenSpan, ...]:
        return tuple(s for s in self.spans if not s.is_special)

    @property
    def token_count(self) -> int:
        return len(self.content_tokens)

    @property
    def unknown_token_count(self) -> int:
        return sum(1 for s in self.spans if s.is_unknown)

    def fragmentation(self) -> float | None:
        """Subwords per syllable -- the §6.8 diagnostic."""
        syllables = self.syllable_map.get("syllable_count")
        if not syllables:
            return None
        return self.token_count / syllables

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "path": self.path.value,
            "availability": self.availability.value,
            "source_text": self.source_text,
            "canonical_text": self.canonical_text,
            "base_text": self.base_text,
            "segmented_text": self.segmented_text,
            "tokenizer_input": self.tokenizer_input,
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "spans": [s.to_dict() for s in self.spans],
            "offset_availability": self.offset_availability.value,
            "offset_reason": self.offset_reason,
            "alignment": self.alignment.value,
            "alignment_reason": self.alignment_reason,
            "coverage": self.coverage,
            "syllable_map": self.syllable_map,
            "eligibility": list(self.eligibility),
            "observed_tones": list(self.observed_tones),
            "token_count": self.token_count,
            "unknown_token_count": self.unknown_token_count,
            "fragmentation": self.fragmentation(),
            "error": self.error,
        }


def grid_invariance(observations: Sequence[PathObservation]) -> dict[str, Any]:
    """Does the base token grid survive corruption, for one case and path?

    Compares every condition against the `FULL` reference. Three levels are
    distinguished, because they fail differently:

    * `base_text_invariant` -- B2's guarantee. If this breaks, corruption
      changed lexical content and the problem is upstream.
    * `tokenizer_input_invariant` -- whether preprocessing (segmentation)
      preserved the equality. This is the new thing B3B-0 exists to measure.
    * `token_ids_invariant` -- what §4.4 actually requires.
    """
    usable = [o for o in observations if o.availability is PathAvailability.OK and o.error is None]
    if not usable:
        return {"comparable": False, "reason": "no usable observations for this path"}

    reference = next((o for o in usable if o.condition == "FULL"), usable[0])
    base_mismatch: list[str] = []
    input_mismatch: list[str] = []
    id_mismatch: list[str] = []
    token_mismatch: list[str] = []

    for observation in usable:
        if observation is reference:
            continue
        if observation.base_text != reference.base_text:
            base_mismatch.append(observation.condition)
        if observation.tokenizer_input != reference.tokenizer_input:
            input_mismatch.append(observation.condition)
        if observation.token_ids != reference.token_ids:
            id_mismatch.append(observation.condition)
        if observation.tokens != reference.tokens:
            token_mismatch.append(observation.condition)

    return {
        "comparable": True,
        "reference_condition": reference.condition,
        "conditions_compared": [o.condition for o in usable],
        "base_text_invariant": not base_mismatch,
        "base_text_mismatches": base_mismatch,
        "tokenizer_input_invariant": not input_mismatch,
        "tokenizer_input_mismatches": input_mismatch,
        "tokens_invariant": not token_mismatch,
        "token_mismatches": token_mismatch,
        "token_ids_invariant": not id_mismatch,
        "token_id_mismatches": id_mismatch,
        # The proposal's requirement, restated as a single boolean.
        "satisfies_base_grid_invariance": not id_mismatch and not base_mismatch,
    }


def path_summary(observations: Sequence[PathObservation]) -> dict[str, Any]:
    """Aggregate one preprocessing path across every case and condition."""
    usable = [o for o in observations if o.availability is PathAvailability.OK and o.error is None]
    unavailable = [o for o in observations if o.availability is not PathAvailability.OK]
    fragmentations = [f for f in (o.fragmentation() for o in usable) if f is not None]
    aligned = [o for o in usable if o.alignment is AlignmentStatus.ALIGNED]
    return {
        "observations": len(observations),
        "usable": len(usable),
        "unavailable": len(unavailable),
        "unavailable_reasons": sorted({o.availability.value for o in unavailable}),
        "errors": sum(1 for o in observations if o.error is not None),
        "aligned": len(aligned),
        "alignment_rate": (len(aligned) / len(usable)) if usable else None,
        "offset_availability": sorted({o.offset_availability.value for o in usable}),
        "mean_fragmentation": (sum(fragmentations) / len(fragmentations)) if fragmentations else None,
        "max_fragmentation": max(fragmentations) if fragmentations else None,
        "total_unknown_tokens": sum(o.unknown_token_count for o in usable),
        "cases_with_unknown_tokens": sum(1 for o in usable if o.unknown_token_count),
    }


def compare_paths(by_path: dict[str, Sequence[PathObservation]]) -> dict[str, Any]:
    """Side-by-side comparison of the preprocessing paths.

    Reports facts only. Choosing a path is a specification decision that weighs
    deployability and hidden-restoration risk, which no measurement settles;
    see `docs/spec/decisions.md` D-B3B0-001.
    """
    summaries = {name: path_summary(list(obs)) for name, obs in by_path.items()}

    invariance: dict[str, Any] = {}
    for name, observations in by_path.items():
        by_case: dict[str, list[PathObservation]] = {}
        for observation in observations:
            by_case.setdefault(observation.case_id, []).append(observation)
        per_case = {case: grid_invariance(rows) for case, rows in by_case.items()}
        comparable = [r for r in per_case.values() if r.get("comparable")]
        invariance[name] = {
            "per_case": per_case,
            "cases_comparable": len(comparable),
            "cases_satisfying_grid_invariance": sum(
                1 for r in comparable if r.get("satisfies_base_grid_invariance")
            ),
            "all_cases_invariant": bool(comparable)
            and all(r.get("satisfies_base_grid_invariance") for r in comparable),
        }

    return {
        "path_summaries": summaries,
        "grid_invariance": invariance,
        "decision": "NOT_MADE",
        "decision_note": (
            "B3B-0 measures; it does not choose. Selecting a preprocessing path weighs "
            "deployability, hidden-restoration risk and PhoBERT distribution match against "
            "these numbers. See docs/spec/decisions.md D-B3B0-001."
        ),
    }
