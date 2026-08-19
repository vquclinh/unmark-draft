"""The deterministic orthographic corruption operator.

Implements proposal v1.3 section 5.3::

    C(x, p, s) -> x̃

with the sampling unit, the removed information and the condition set taken
from section 6.3. Two clarifications of that text are recorded in
`docs/spec/decisions.md`: the corruption key carries an explicit `sample_id`
(D-B2-001), and selection is an independent per-syllable Bernoulli trial rather
than an exact `round(p*N)` count (D-B2-002).

Corruption operates on `canon(x)`, never on arbitrary source spelling
(D-B2-004), so `hòa` and `hoà` corrupt identically. It removes information that
is already represented outside the base channel, which is what makes the
proposal's central invariant hold::

    strip_to_base(canon(x)) == strip_to_base(corrupt(x))

No tokenizer, no model, no word list, no network. Pure standard library plus
the B1A orthography core.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from unmark.corruption.conditions import CorruptionCondition, CorruptionScope, get_condition
from unmark.corruption.deterministic import CORRUPTION_SCHEMA_VERSION, is_selected, text_identity
from unmark.corruption.eligibility import (
    ACTIVE_ELIGIBILITY_POLICY,
    CorruptionPurpose,
    EligibilityPolicy,
    require_resolved_eligibility,
)
from unmark.corruption.models import CorruptionResult, UnitDecision
from unmark.orthography import ObservedTone, canon, decompose
from unmark.orthography.marks import D_STROKE, LETTER_MARK_TO_STATE, TONE_MARK_TO_OBSERVED
from unmark.orthography.units import join_units, split_units

_TONE_MARK_SET = frozenset(TONE_MARK_TO_OBSERVED)
_LETTER_MARK_SET = frozenset(LETTER_MARK_TO_STATE)


def corrupt(
    text: str,
    condition: str | CorruptionCondition,
    seed: int,
    sample_id: str | int,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    source_is_clean: bool = True,
    schema_version: str = CORRUPTION_SCHEMA_VERSION,
    eligibility_policy: EligibilityPolicy = ACTIVE_ELIGIBILITY_POLICY,
) -> CorruptionResult:
    """Corrupt `text` under `condition`, reproducibly.

    Args:
        text: the clean Vietnamese string. Canonicalised before anything else;
            the original is preserved verbatim in the result.
        condition: a `CorruptionCondition` or its name (`"P50"`, `"strip-all"`).
        seed: experiment seed.
        sample_id: stable identity of this example. Must not be row order --
            reordering a dataset must not change any sample's corruption.
        purpose: `SCIENTIFIC` (the default) requires a resolved eligibility
            policy and therefore **raises today**, because GAP-2 is open. Pass
            `SELF_CHECK` for implementation verification; the result is stamped
            `provisional_eligibility=True` and its counts are named
            `candidate_*`. The unsafe path is the one you have to ask for.
        source_is_clean: assert `text` is fully diacritized, which makes an
            unmarked syllable a genuine `ngang`. True by default because that is
            what the operator's input is defined to be; recorded in the result.
            When false, `clean_lexical_tone` stays `None` for unmarked syllables
            and the ORACLE metadata is correspondingly weaker.

    Returns:
        A `CorruptionResult` reproducible from
        `(schema_version, condition, seed, sample_id, canonical text)` alone.
    """
    if purpose is CorruptionPurpose.SCIENTIFIC:
        require_resolved_eligibility(
            context=f"purpose={purpose.name} corruption", policy=eligibility_policy
        )

    cond = get_condition(condition)
    canonical = canon(text)
    identity = text_identity(canonical)

    clean = decompose(canonical, source_is_clean=source_is_clean)

    # CANDIDATE spans, not confirmed eligible Vietnamese syllables. While GAP-2
    # is open no language filter can be applied, so this is every maximal
    # alphabetic run. The single place a resolved eligibility policy would plug
    # in is here, by filtering `spans`; nothing else in this function changes.
    # See unmark/corruption/eligibility.py and docs/spec/decisions.md D-B2-003.
    spans = clean.syllables

    selected_flags: list[bool] = []
    scores: list[float] = []
    for span in spans:
        selected, score = is_selected(
            probability=cond.probability,
            seed=seed,
            sample_id=str(sample_id),
            identity=identity,
            unit_index=span.span_index,
            schema_version=schema_version,
        )
        selected_flags.append(selected and cond.scope is not CorruptionScope.NONE)
        scores.append(score)

    corrupted_text, removed_per_span = _apply(canonical, clean, spans, selected_flags, cond.scope)
    corrupted = decompose(corrupted_text, source_is_clean=False)

    decisions = _build_decisions(spans, corrupted.syllables, selected_flags, scores, removed_per_span)

    return CorruptionResult(
        schema_version=schema_version,
        condition=cond,
        seed=seed,
        sample_id=str(sample_id),
        text_identity=identity,
        original_text=text,
        canonical_clean_text=canonical,
        corrupted_text=corrupted_text,
        eligibility_policy=eligibility_policy,
        requested_probability=cond.probability,
        candidate_units=len(spans),
        selected_candidates=sum(selected_flags),
        modified_candidates=sum(1 for d in decisions if d.modified),
        decisions=tuple(decisions),
        clean_decomposition=clean,
        corrupted_decomposition=corrupted,
        source_is_clean=source_is_clean,
        metadata={
            "condition_description": cond.description,
            "scope": cond.scope.value,
            "unit": "candidate_syllable_span",
            "purpose": purpose.name,
            "eligibility_policy": eligibility_policy.name,
            "eligibility_filter": (
                "none - PROVISIONAL fallback: candidate spans are every maximal alphabetic "
                "run, including non-Vietnamese words. Must be replaced by the stripped-form "
                "Vietnamese syllable eligibility policy before stage-1 training or the main "
                "experiments. GAP-2, owner B3/pre-training; docs/spec/decisions.md D-B2-003."
            ),
        },
    )


def _apply(
    canonical: str,
    clean: Any,
    spans: tuple[Any, ...],
    selected_flags: list[bool],
    scope: CorruptionScope,
) -> tuple[str, list[tuple[bool, tuple[str, ...]]]]:
    """Rebuild the string with the selected syllables stripped.

    Works on the NFD unit stream, so only combining marks and the `đ` stroke are
    ever touched: letters, case, punctuation, whitespace, digits and unsupported
    combining marks pass through untouched by construction.
    """
    units = split_units(unicodedata.normalize("NFD", canonical))
    mutable: list[tuple[str, list[str]]] = [(base, list(marks)) for base, marks in units]

    # Map each canonical character offset to its unit, so a span's units can be
    # found without re-deriving the segmentation.
    unit_of_span: dict[int, list[int]] = {}
    for span in spans:
        unit_of_span[span.span_index] = list(span.unit_indices)

    removed: list[tuple[bool, tuple[str, ...]]] = []
    for span in spans:
        if not selected_flags[span.span_index]:
            removed.append((False, ()))
            continue
        tone_removed = False
        letters_removed: list[str] = []
        for unit_index in unit_of_span[span.span_index]:
            base, marks = mutable[unit_index]
            if scope in (CorruptionScope.TONE, CorruptionScope.TONE_AND_LETTER):
                kept = [m for m in marks if m not in _TONE_MARK_SET]
                if len(kept) != len(marks):
                    tone_removed = True
                marks = kept
            if scope is CorruptionScope.TONE_AND_LETTER:
                kept = [m for m in marks if m not in _LETTER_MARK_SET]
                letters_removed.extend(m for m in marks if m in _LETTER_MARK_SET)
                marks = kept
                if base in D_STROKE:
                    letters_removed.append("STROKE")
                    base = D_STROKE[base]
            mutable[unit_index] = (base, marks)
        removed.append((tone_removed, tuple(letters_removed)))

    rebuilt = join_units([(base, tuple(marks)) for base, marks in mutable])
    return unicodedata.normalize("NFC", rebuilt), removed


def _build_decisions(
    clean_spans: tuple[Any, ...],
    corrupted_spans: tuple[Any, ...],
    selected_flags: list[bool],
    scores: list[float],
    removed_per_span: list[tuple[bool, tuple[str, ...]]],
) -> list[UnitDecision]:
    """Pair each clean syllable with its corrupted counterpart.

    Corruption never inserts, deletes or reorders a syllable, so the two span
    sequences are the same length and align by index. That is asserted rather
    than assumed: if it ever fails, something changed the lexical content.
    """
    if len(clean_spans) != len(corrupted_spans):
        raise AssertionError(
            f"corruption changed the syllable count ({len(clean_spans)} -> {len(corrupted_spans)}); "
            "this means lexical content was altered, which corruption must never do"
        )

    decisions: list[UnitDecision] = []
    for clean_span, corrupted_span, selected, score, (tone_removed, letters_removed) in zip(
        clean_spans, corrupted_spans, selected_flags, scores, removed_per_span
    ):
        modified = tone_removed or bool(letters_removed)
        decisions.append(
            UnitDecision(
                unit_index=clean_span.span_index,
                base_text=clean_span.base_text,
                canonical_start=clean_span.canonical_start,
                canonical_end=clean_span.canonical_end,
                score=score,
                selected=selected,
                modified=modified,
                clean_lexical_tone=clean_span.lexical_tone,
                clean_observed_tone=clean_span.observed_tone,
                corrupted_observed_tone=corrupted_span.observed_tone,
                tone_mark_removed=tone_removed,
                letter_diacritics_removed=letters_removed,
                # Carried through from B1A: still UNDECIDED, never upgraded here.
                eligibility=clean_span.eligibility,
            )
        )
    return decisions


def corrupt_batch(
    samples: list[tuple[str, str | int]],
    condition: str | CorruptionCondition,
    seed: int,
    **kwargs: Any,
) -> list[CorruptionResult]:
    """Corrupt `(text, sample_id)` pairs.

    Order-independent by construction: each result depends only on its own
    `sample_id` and text, so shuffling `samples` permutes the output list
    without changing any individual result.
    """
    return [corrupt(text, condition, seed, sample_id, **kwargs) for text, sample_id in samples]
