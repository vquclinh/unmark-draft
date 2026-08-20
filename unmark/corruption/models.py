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
    requested_probability: float

    candidate_units: int
    """Every maximal alphabetic run in the text, regardless of eligibility.
    Structural, language-blind, always available."""

    scored_units: int
    """Units that entered the lottery. Equal to the eligible Vietnamese
    syllables when the policy is resolved, and to `candidate_units` under the
    provisional self-check fallback. `eligible_units` exposes it only when that
    distinction is real."""

    selected_units: int
    modified_units: int
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
        """Confirmed eligible Vietnamese syllables -- the scientific denominator.

        Raises under the provisional fallback. There is no honest value there:
        substituting the candidate count is exactly what would turn a temporary
        state into the protocol.
        """
        if self.provisional_eligibility:
            raise EligibilityUnresolved(
                "eligible_units is undefined while the eligibility policy is "
                f"{self.eligibility_policy.name}. This result scored "
                f"{self.candidate_units} candidate span(s), which is not the same thing: "
                "candidates are maximal alphabetic runs and include non-Vietnamese words. "
                "Fetch the pinned inventory "
                "(scripts/fetch_vietnamese_syllable_inventory.py), or read "
                "`candidate_units` / `candidate_selection_rate` and know what they mean. "
                "See docs/spec/decisions.md D-B2-003 and D-B3A-001."
            )
        return self.scored_units

    @property
    def realized_probability(self) -> float | None:
        """Selected / eligible Vietnamese syllables.

        The scientific realized rate. Raises under the provisional fallback, for
        the same reason as `eligible_units`; `None` when the text contains no
        eligible syllable at all, never 0.0.
        """
        eligible = self.eligible_units
        return None if eligible == 0 else self.selected_units / eligible

    @property
    def realized_modification_rate(self) -> float | None:
        """Modified / eligible Vietnamese syllables.

        Lower than `realized_probability` whenever a selected syllable carried no
        mark to remove, so "selected" and "changed" are never conflated.
        """
        eligible = self.eligible_units
        return None if eligible == 0 else self.modified_units / eligible

    @property
    def candidate_selection_rate(self) -> float | None:
        """Selected / **candidate** spans. Always available.

        Not the scientific rate when the policy is resolved -- the denominator
        includes non-Vietnamese spans that were never scored. Kept because it is
        the only rate the provisional fallback can honestly report.
        """
        if self.candidate_units == 0:
            return None
        return self.selected_units / self.candidate_units

    @property
    def candidate_modification_rate(self) -> float | None:
        if self.candidate_units == 0:
            return None
        return self.modified_units / self.candidate_units

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
            "scored_units": self.scored_units,
            "selected_units": self.selected_units,
            "modified_units": self.modified_units,
            "candidate_selection_rate": self.candidate_selection_rate,
            "candidate_modification_rate": self.candidate_modification_rate,
            # The scientific denominator and rate appear ONLY when the policy is
            # resolved. Under the provisional fallback there is no honest value
            # and an artifact must not imply one (docs/spec/decisions.md D-B2-003).
            **(
                {}
                if self.provisional_eligibility
                else {
                    "eligible_units": self.eligible_units,
                    "realized_probability": self.realized_probability,
                    "realized_modification_rate": self.realized_modification_rate,
                }
            ),
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
