"""Deterministic Vietnamese orthographic decomposition and reconstruction.

Implements the rule-based `dec` / `rec` of proposal v1.3 section 4.2 at the
granularities fixed in section 4.3: a character-level base stream, a
character-level letter-diacritic channel, and a syllable-level tone channel.

The guaranteed invariant is::

    recompose(decompose(x)) == canon(x)

It holds by construction, not by luck: each character unit stores the exact NFD
combining-mark sequence it carried, and reconstruction replays that sequence.
Marks this project does not classify as Vietnamese are carried through
untouched rather than dropped, so the round-trip survives text the orthography
layer does not understand.

No model, no lookup table, no word list, no network. Pure standard library.

Naming note: this module and the `decompose` function it defines share a name,
which the proposal (section 8.1) fixes. The package re-exports the function, so
``import unmark.orthography.decompose as d`` binds the *function*, not the
module -- the same shadowing as ``datetime.datetime``. Import either
``from unmark.orthography import decompose`` or
``from unmark.orthography.decompose import decompose, recompose``.
"""

from __future__ import annotations

import unicodedata

from unmark.orthography.canonical import DEFAULT_TONE_PLACEMENT, TonePlacement, canon, nfc, nfd
from unmark.orthography.marks import (
    D_STROKE,
    D_STROKE_INVERSE,
    LETTER_MARK_TO_STATE,
    TONE_MARK_TO_OBSERVED,
    TONE_MARK_TO_TONE,
    Anomaly,
    Eligibility,
    LetterDiacritic,
    ObservedTone,
    Tone,
)
from unmark.orthography.models import CharacterUnit, DecomposedText, SyllableSpan
from unmark.orthography.units import split_units

_TONE_MARK_SET = frozenset(TONE_MARK_TO_OBSERVED)
_LETTER_MARK_SET = frozenset(LETTER_MARK_TO_STATE)


def decompose(
    text: str,
    *,
    source_is_clean: bool = False,
    placement: TonePlacement = DEFAULT_TONE_PLACEMENT,
) -> DecomposedText:
    """Decompose `text` into base, tone and letter-diacritic streams.

    Args:
        text: any string, in NFC, NFD or neither.
        source_is_clean: assert that `text` is fully diacritized Vietnamese, so
            an absent tone mark really is `ngang`. **Default false.** When false,
            an unmarked syllable gets `lexical_tone=None`, because absence of a
            mark is not evidence of `ngang` (proposal 1.2). Set this only for
            text known to be clean, such as a gold corpus before corruption.
        placement: canonical tone placement; only `PRESERVE` is implemented.

    Returns:
        A :class:`DecomposedText` from which `recompose` rebuilds `canon(text)`.
    """
    canonical_text = canon(text, placement)
    text_nfd = nfd(canonical_text)

    units: list[CharacterUnit] = []
    base_parts: list[str] = []
    canonical_cursor = 0
    base_cursor = 0

    for unit_index, (base_cp, unit_marks) in enumerate(split_units(text_nfd)):
        anomalies: list[Anomaly] = []

        tone_marks = [m for m in unit_marks if m in _TONE_MARK_SET]
        letter_marks = [m for m in unit_marks if m in _LETTER_MARK_SET]
        other_marks = [m for m in unit_marks if m not in _TONE_MARK_SET and m not in _LETTER_MARK_SET]

        if other_marks:
            anomalies.append(Anomaly.UNSUPPORTED_COMBINING_MARK)
        if len(tone_marks) > 1:
            anomalies.append(Anomaly.MULTIPLE_TONE_MARKS)
        if len(letter_marks) > 1:
            anomalies.append(Anomaly.MULTIPLE_LETTER_DIACRITICS)

        # đ / Đ: a stroke, not a combining mark.
        has_stroke = base_cp in D_STROKE
        base_letter = D_STROKE[base_cp] if has_stroke else base_cp

        # The base keeps non-Vietnamese combining marks, so `Müller` survives as
        # `Müller` while `Đường` becomes `Duong`. This matches the G-1
        # `base_signature` semantics, which the tests cross-check.
        base_char = nfc(base_letter + "".join(other_marks))

        is_letter = base_letter.isalpha()
        if not is_letter:
            letter_state = LetterDiacritic.NA
            if letter_marks:
                # A letter-forming mark on a non-letter is not Vietnamese; it is
                # preserved for reconstruction but the channel does not apply.
                anomalies.append(Anomaly.UNSUPPORTED_COMBINING_MARK)
            if tone_marks:
                anomalies.append(Anomaly.TONE_MARK_ON_NON_LETTER)
        elif has_stroke:
            letter_state = LetterDiacritic.STROKE
        elif letter_marks:
            letter_state = LETTER_MARK_TO_STATE[letter_marks[0]]
        else:
            letter_state = LetterDiacritic.NONE

        observed = TONE_MARK_TO_OBSERVED[tone_marks[0]] if tone_marks else ObservedTone.UNMARKED

        canonical_unit = nfc(base_cp + "".join(unit_marks))
        units.append(
            CharacterUnit(
                unit_index=unit_index,
                canonical_text=canonical_unit,
                canonical_start=canonical_cursor,
                canonical_end=canonical_cursor + len(canonical_unit),
                base_char=base_char,
                base_start=base_cursor,
                base_end=base_cursor + len(base_char),
                letter_diacritic=letter_state,
                observed_tone=observed,
                nfd_marks=unit_marks,
                has_stroke=has_stroke,
                anomalies=tuple(dict.fromkeys(anomalies)),
            )
        )
        canonical_cursor += len(canonical_unit)
        base_cursor += len(base_char)
        base_parts.append(base_char)

    base_text = "".join(base_parts)
    syllables = _segment_syllables(units, source_is_clean=source_is_clean)

    return DecomposedText(
        original_text=text,
        nfc_text=nfc(text),
        canonical_text=canonical_text,
        base_text=base_text,
        units=tuple(units),
        syllables=tuple(syllables),
        source_is_clean=source_is_clean,
        metadata={"placement": placement.name},
    )


def _segment_syllables(units: list[CharacterUnit], *, source_is_clean: bool) -> list[SyllableSpan]:
    """Split the unit stream into maximal alphabetic runs.

    Vietnamese writes each syllable as its own whitespace-delimited word, so a
    maximal alphabetic run is the orthographic candidate for one syllable. This
    makes no claim that the run *is* Vietnamese -- see `Eligibility`.
    """
    spans: list[SyllableSpan] = []
    current: list[CharacterUnit] = []

    def flush() -> None:
        if not current:
            return
        spans.append(_build_span(len(spans), current, source_is_clean=source_is_clean))
        current.clear()

    for unit in units:
        if unit.is_letter:
            current.append(unit)
        else:
            flush()
    flush()
    return spans


def _build_span(span_index: int, units: list[CharacterUnit], *, source_is_clean: bool) -> SyllableSpan:
    toned = [u for u in units if u.observed_tone is not ObservedTone.UNMARKED]

    anomalies: list[Anomaly] = []
    if len(toned) > 1:
        # Well-formed Vietnamese never marks two tones in one syllable. Recorded,
        # never repaired: the character stream still reconstructs exactly, and
        # the syllable-level channel value is flagged as unreliable.
        anomalies.append(Anomaly.MULTIPLE_TONE_MARKS)

    if toned:
        observed = toned[0].observed_tone
        tone_unit_index = toned[0].unit_index
        lexical: Tone | None = Tone[observed.name]
    else:
        observed = ObservedTone.UNMARKED
        tone_unit_index = None
        # The load-bearing line of this module. No visible mark means either a
        # genuine ngang or a stripped tone, and the string cannot say which
        # (proposal 1.2). Only an explicit clean-source assertion settles it.
        lexical = Tone.NGANG if source_is_clean else None

    for unit in units:
        for anomaly in unit.anomalies:
            if anomaly not in anomalies:
                anomalies.append(anomaly)

    return SyllableSpan(
        span_index=span_index,
        canonical_start=units[0].canonical_start,
        canonical_end=units[-1].canonical_end,
        base_start=units[0].base_start,
        base_end=units[-1].base_end,
        text="".join(u.canonical_text for u in units),
        base_text="".join(u.base_char for u in units),
        observed_tone=observed,
        lexical_tone=lexical,
        tone_unit_index=tone_unit_index,
        # Proposal 4.3 decides this by matching the Vietnamese syllable
        # inventory after stripping. That inventory is not in the proposal and
        # not in this repository, so B1A does not guess it.
        eligibility=Eligibility.UNDECIDED,
        unit_indices=tuple(u.unit_index for u in units),
        anomalies=tuple(anomalies),
    )


def recompose(parts: DecomposedText) -> str:
    """Rebuild the canonical string from a decomposition.

    Exact by construction: each unit replays its stored NFD mark sequence on its
    base codepoint, and the result is renormalised to NFC. Equal to
    `canon(original_text)` for every input.
    """
    pieces: list[str] = []
    for unit in parts.units:
        base_letter = unit.base_char
        # Strip any non-Vietnamese marks back off the base before replaying the
        # full NFD sequence, otherwise they would be applied twice.
        base_letter = nfd(base_letter)[:1] if base_letter else ""
        if unit.has_stroke:
            base_letter = D_STROKE_INVERSE.get(base_letter, base_letter)
        pieces.append(base_letter + "".join(unit.nfd_marks))
    return nfc("".join(pieces))


def strip_to_base(text: str, *, placement: TonePlacement = DEFAULT_TONE_PLACEMENT) -> str:
    """Convenience: the base stream of `text` as a string.

    Equivalent to `decompose(text).base_text`. Note this is *not* the same as
    `signature.base_signature`, which additionally collapses whitespace because
    it serves a comparison purpose rather than a reconstruction one.
    """
    return decompose(text, placement=placement).base_text
