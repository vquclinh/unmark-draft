"""Typed results of the corruption operator.

Everything a later stage needs is recorded, including enough paired metadata to
derive the three H4 tone policies (proposal v1.3 section 6.7) without B2
implementing any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unmark.corruption.conditions import CorruptionCondition
from unmark.corruption.eligibility import (
    EligibilityPolicy,
    EligibilityUnresolved,
    is_resolved,
)
from unmark.orthography import DecomposedText, Eligibility, ObservedTone, Tone


@dataclass(frozen=True)
class UnitDecision:
    """The corruption decision for one orthographic unit, and its consequence.

    `clean_lexical_tone` versus `corrupted_observed_tone` is the pair the H4
    oracle needs: it is the only place the difference between a genuine `ngang`
    and a stripped tone survives, because the corrupted *string* cannot express
    it.
    """

    unit_index: int
    base_text: str
    canonical_start: int
    canonical_end: int

    score: float
    selected: bool
    """Whether the stable per-unit score put this unit below the threshold."""

    modified: bool
    """Whether the text of this unit actually changed. A selected `ngang`
    syllable is selected but not modified: there is no mark to remove. Proposal
    section 4.3: "a `ngang` syllable is *invariant*"."""

    clean_lexical_tone: Tone | None
    clean_observed_tone: ObservedTone
    corrupted_observed_tone: ObservedTone

    tone_mark_removed: bool
    letter_diacritics_removed: tuple[str, ...] = ()

    eligibility: Eligibility = Eligibility.UNDECIDED
    """Whether this span is a confirmed Vietnamese syllable.

    `UNDECIDED` while GAP-2 is open, which is every case today. Being scored by
    the corruption engine does *not* upgrade it: the engine scores candidate
    spans, which is a structural notion, not a linguistic one.
    """

    # --- H4 policy views, derived here so no later stage re-derives them ----
    @property
    def oracle_tone_is_missing(self) -> bool:
        """ORACLE policy: this position's tone was destroyed by corruption.

        Available only because corruption is synthetic. Proposal section 6.7:
        the oracle "cannot be used at inference and exists purely as an
        upper-bound diagnostic".
        """
        return self.tone_mark_removed

    @property
    def oracle_tone_is_genuine_ngang(self) -> bool:
        """ORACLE policy: unmarked in the output *and* genuinely `ngang`."""
        return (
            not self.tone_mark_removed
            and self.corrupted_observed_tone is ObservedTone.UNMARKED
            and self.clean_lexical_tone is Tone.NGANG
        )


@dataclass(frozen=True)
class CorruptionResult:
    """The full record of one corruption, reproducible from its key alone."""

    # --- identity and key --------------------------------------------------
    schema_version: str
    condition: CorruptionCondition
    seed: int
    sample_id: str
    text_identity: str

    # --- text --------------------------------------------------------------
    original_text: str
    """Exactly as supplied, never mutated."""
    canonical_clean_text: str
    corrupted_text: str

    # --- eligibility -------------------------------------------------------
    eligibility_policy: EligibilityPolicy
    """The policy in force. `UNRESOLVED` means the counts below are about
    *candidate spans*, not confirmed Vietnamese syllables."""

    # --- decisions ---------------------------------------------------------
    # Named `candidate_*`, never `eligible_*`: while GAP-2 is open these count
    # maximal alphabetic runs, including English words. See `eligible_units`.
    requested_probability: float
    candidate_units: int
    selected_candidates: int
    modified_candidates: int
    decisions: tuple[UnitDecision, ...]

    # --- decompositions ----------------------------------------------------
    clean_decomposition: DecomposedText
    corrupted_decomposition: DecomposedText
    source_is_clean: bool

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def provisional_eligibility(self) -> bool:
        """True while the counts describe candidate spans rather than syllables."""
        return not is_resolved(self.eligibility_policy)

    @property
    def eligible_units(self) -> int:
        """Confirmed Vietnamese syllables.

        Raises while GAP-2 is open. There is no honest provisional value: using
        the candidate count here is exactly the substitution that would turn a
        temporary fallback into the scientific protocol. Read
        `candidate_units` instead, and know what it means.
        """
        if self.provisional_eligibility:
            raise EligibilityUnresolved(
                "eligible_units is undefined while the eligibility policy is "
                f"{self.eligibility_policy.name} (GAP-2 open). This result counted "
                f"{self.candidate_units} candidate span(s), which is not the same thing: "
                "candidates are maximal alphabetic runs and include non-Vietnamese words. "
                "Use `candidate_units` / `candidate_selection_rate`, or close GAP-2 "
                "(owner: B3 / pre-training; see docs/spec/decisions.md D-B2-003)."
            )
        return self.candidate_units

    @property
    def candidate_selection_rate(self) -> float | None:
        """Selected candidate spans / candidate spans.

        **Not** the fraction of Vietnamese syllables corrupted: while GAP-2 is
        open the denominator is every alphabetic run, English words included.
        `None` when there are no candidate spans at all -- a string of digits and
        punctuation has none, and reporting 0.0 would claim a rate that was never
        measured.
        """
        if self.candidate_units == 0:
            return None
        return self.selected_candidates / self.candidate_units

    @property
    def candidate_modification_rate(self) -> float | None:
        """Candidate spans whose text actually changed / candidate spans.

        Lower than `candidate_selection_rate` whenever a selected syllable
        carried no mark to remove, so "selected" and "changed" are never
        conflated.
        """
        if self.candidate_units == 0:
            return None
        return self.modified_candidates / self.candidate_units

    @property
    def is_unchanged(self) -> bool:
        return self.corrupted_text == self.canonical_clean_text

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for run artifacts."""
        return {
            "schema_version": self.schema_version,
            "condition": self.condition.name,
            "condition_scope": self.condition.scope.value,
            "seed": self.seed,
            "sample_id": self.sample_id,
            "text_identity": self.text_identity,
            "original_text": self.original_text,
            "canonical_clean_text": self.canonical_clean_text,
            "corrupted_text": self.corrupted_text,
            "requested_probability": self.requested_probability,
            "eligibility_policy": self.eligibility_policy.value,
            "provisional_eligibility": self.provisional_eligibility,
            "candidate_units": self.candidate_units,
            "selected_candidates": self.selected_candidates,
            "modified_candidates": self.modified_candidates,
            "candidate_selection_rate": self.candidate_selection_rate,
            "candidate_modification_rate": self.candidate_modification_rate,
            # Deliberately absent while GAP-2 is open: eligible_units,
            # realized_probability. There is no honest value for either, and an
            # artifact must not imply one. See docs/spec/decisions.md D-B2-003.
            "source_is_clean": self.source_is_clean,
            "base_text": self.clean_decomposition.base_text,
            "base_invariant": self.clean_decomposition.base_text == self.corrupted_decomposition.base_text,
            "decisions": [
                {
                    "unit_index": d.unit_index,
                    "base_text": d.base_text,
                    "eligibility": d.eligibility.value,
                    "score": d.score,
                    "selected": d.selected,
                    "modified": d.modified,
                    "clean_lexical_tone": d.clean_lexical_tone.name if d.clean_lexical_tone else None,
                    "clean_observed_tone": d.clean_observed_tone.name,
                    "corrupted_observed_tone": d.corrupted_observed_tone.name,
                    "tone_mark_removed": d.tone_mark_removed,
                    "letter_diacritics_removed": list(d.letter_diacritics_removed),
                    "oracle_tone_is_missing": d.oracle_tone_is_missing,
                    "oracle_tone_is_genuine_ngang": d.oracle_tone_is_genuine_ngang,
                }
                for d in self.decisions
            ],
            "metadata": self.metadata,
        }
