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
from typing import Callable

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
from unmark.orthography.units import split_units_with_offsets

_TONE_MARK_SET = frozenset(TONE_MARK_TO_OBSERVED)
_LETTER_MARK_SET = frozenset(LETTER_MARK_TO_STATE)


def decompose(
    text: str,
    *,
    source_is_clean: bool = False,
    placement: TonePlacement = DEFAULT_TONE_PLACEMENT,
    eligibility_classifier: Callable[[str], Eligibility] | None = None,
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
        eligibility_classifier: optional `str -> Eligibility` applied to each
            syllable's **stripped base form**. Kept as an injected policy layer
            rather than an import so that Unicode decomposition and
            reconstruction never depend on an external resource file: without a
            classifier every span stays `Eligibility.UNDECIDED` and the
            round-trip is unaffected. `unmark.linguistics.make_classifier`
            builds one from the pinned syllable inventory.

    Returns:
        A :class:`DecomposedText` from which `recompose` rebuilds `canon(text)`.
    """
    canonical_text = canon(text, placement)

    units: list[CharacterUnit] = []
    base_parts: list[str] = []
    canonical_cursor = 0
    base_cursor = 0

    # Units are grouped over the **canonical (NFC) text**, then each unit is
    # decomposed individually to separate its marks. Grouping over
    # `nfd(whole text)` instead -- which this did until Audit 029 §AA -- silently
    # split any precomposed character whose NFD expansion is several
    # *non-combining* codepoints. Hangul is the real case: NFD turns one syllable
    # into 2-3 Jamo of combining class 0, so each Jamo became its **own unit**
    # with its own base, and a 98-character region produced a 269-character base
    # stream. That is Unicode decomposition leaking into the base rather than
    # Vietnamese diacritic removal, and proposal §4.2 requires "recomposition of
    # the base".
    for unit_index, unit in enumerate(split_units_with_offsets(canonical_text)):
        anomalies: list[Anomaly] = []

        # Decompose THIS unit and keep every non-combining codepoint. For Latin
        # the skeleton is one character; for Hangul it is the Jamo sequence that
        # NFC puts back together.
        unit_nfd = nfd(unit.text)
        base_cp = "".join(c for c in unit_nfd if not unicodedata.combining(c))
        unit_marks = tuple(c for c in unit_nfd if unicodedata.combining(c))

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

        canonical_unit = unit.text
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
    syllables = _segment_syllables(
        units, source_is_clean=source_is_clean, eligibility_classifier=eligibility_classifier
    )

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


def protects_a_vietnamese_candidate(unit_text: str) -> bool:
    """Whether a chunk cut may **not** fall inside a run of such units.

    Vietnamese is written in the **Latin script**, so a character unit can be
    part of a Vietnamese candidate only if its base letter is a Latin letter.
    That is the whole predicate, and it is deliberately the *smallest* one that
    does the job:

    * **Not `str.isalpha()`.** "Alphabetic in any Unicode script" was the rule
      until Audit 029 §AA, and it made 97 consecutive Hangul syllables one
      indivisible "Vietnamese candidate span" — a region no Vietnamese reader
      would recognise as a syllable, whose RAW_BASE length then exceeded
      `max_length` with no legal cut. Hangul, CJK, Cyrillic and Greek are
      alphabetic; none of them can spell a Vietnamese syllable.

    * **Not the eligibility classifier.** `classify_candidate` answers
      *inventory membership*, and returns `NOT_APPLICABLE` for a syllable that
      is orthographically valid but out of vocabulary. Protecting only what the
      inventory recognises would therefore permit a cut **inside a genuine
      Vietnamese syllable** that the pinned inventory happens not to list. This
      predicate is orthographic and lexicon-free, so an uncommon or OOV
      Vietnamese candidate is protected exactly as a common one is.

    * **Not a script table.** The test is `unicodedata.name`'s script prefix,
      from the same standard-library module the rest of this package already
      uses for normalisation and combining classes. No codepoint list is
      hard-coded here.

    Latin is *wider* than Vietnamese, deliberately: `Müller`, `naïve` and
    `façade` stay protected. Over-protection only costs cut opportunities;
    under-protection would bisect a syllable, so the error is taken in the safe
    direction.
    """
    if not unit_text:
        return False
    base_nfd = nfd(unit_text)
    base_cp = base_nfd[0] if base_nfd else ""
    base_letter = D_STROKE.get(base_cp, base_cp)
    if not base_letter.isalpha():
        return False
    try:
        return unicodedata.name(base_letter).startswith("LATIN ")
    except ValueError:  # unnamed codepoint -- not a Latin letter
        return False


def source_letter_runs(text: str) -> list[tuple[int, int]]:
    """Maximal **protected** runs of `text`, in **source** coordinates.

    The same shape of segmentation `_segment_syllables` performs -- split the
    character unit stream on a per-unit letter predicate -- but over the string
    **as given**, so the ranges address the original rather than its canonical
    form, and with the **narrower** predicate
    :func:`protects_a_vietnamese_candidate`.

    The two are therefore related but not equal, and the difference is
    deliberate. `SyllableSpan` answers "where are the alphabetic runs?" and
    feeds channel metadata, where breadth is harmless. This answers "what may a
    chunk cut never bisect?", where breadth is not harmless: it protected 97
    Hangul syllables as one candidate and stopped Stage 6 (§AA). Every run
    returned here is contained in an alphabetic run; the reverse does not hold.

    Why this exists: `decompose` canonicalises before it unitises, so every
    offset it reports is a canonical offset. For a non-canonically spelled
    source those offsets do not address the original, and Stage-1 chunking must
    never rewrite the corpus to make them fit (Audit 029 §Z). This reuses the
    **existing** unitisation (`split_units_with_offsets`) and the **existing**
    letter predicate (the NFD base codepoint, with a `d`-stroke resolved,
    answering `str.isalpha`). It is not a second Vietnamese parser: it makes no
    tone, eligibility or syllable-identity claim, and is used only to decide
    which offsets a cut must avoid.

    Returns half-open `(start, end)` ranges over `text`, in order.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for unit in split_units_with_offsets(text):
        if protects_a_vietnamese_candidate(unit.text):
            if start is None:
                start = unit.start
            end = unit.end
        elif start is not None:
            runs.append((start, end))
            start = None
    if start is not None:
        runs.append((start, end))
    return runs


def _segment_syllables(
    units: list[CharacterUnit],
    *,
    source_is_clean: bool,
    eligibility_classifier: Callable[[str], Eligibility] | None = None,
) -> list[SyllableSpan]:
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
        spans.append(
            _build_span(
                len(spans),
                current,
                source_is_clean=source_is_clean,
                eligibility_classifier=eligibility_classifier,
            )
        )
        current.clear()

    for unit in units:
        if unit.is_letter:
            current.append(unit)
        else:
            flush()
    flush()
    return spans


def _build_span(
    span_index: int,
    units: list[CharacterUnit],
    *,
    source_is_clean: bool,
    eligibility_classifier: Callable[[str], Eligibility] | None = None,
) -> SyllableSpan:
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

    base_text = "".join(u.base_char for u in units)
    return SyllableSpan(
        span_index=span_index,
        canonical_start=units[0].canonical_start,
        canonical_end=units[-1].canonical_end,
        base_start=units[0].base_start,
        base_end=units[-1].base_end,
        text="".join(u.canonical_text for u in units),
        base_text=base_text,
        observed_tone=observed,
        lexical_tone=lexical,
        tone_unit_index=tone_unit_index,
        # Proposal 4.3: membership of the Vietnamese syllable inventory, tested
        # on the STRIPPED form so clean and corrupted input classify identically.
        # `UNDECIDED` only when no classifier was supplied -- meaning "not
        # resolvable here", never "resolved as non-Vietnamese".
        eligibility=(
            eligibility_classifier(base_text)
            if eligibility_classifier is not None
            else Eligibility.UNDECIDED
        ),
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
        # Keep every *non-combining* codepoint: one character for Latin, the
        # whole Jamo skeleton for Hangul. Taking only the first would silently
        # delete two thirds of a Hangul syllable (§AA).
        base_letter = "".join(
            c for c in nfd(base_letter) if not unicodedata.combining(c)
        )
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
