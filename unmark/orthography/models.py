"""Typed result structures for Vietnamese orthographic decomposition.

Three parallel streams, at the granularities the proposal (v1.3 section 4.3)
fixes:

* base   -- character level, all Vietnamese diacritics removed;
* tone   -- **syllable** level, one tone per syllable;
* letter -- **character** level, because one syllable may carry several
  letter-forming marks on different characters.

Everything needed to rebuild the canonical string exactly is retained on the
character units, so reconstruction never depends on the syllable-level tone
label. That matters while canonical tone placement is undecided (see
`canonical.py`): the *position* of a tone mark inside a syllable is currently
information, not a derivable consequence of a placement rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unmark.orthography.marks import Anomaly, Eligibility, LetterDiacritic, ObservedTone, Tone


@dataclass(frozen=True)
class CharacterUnit:
    """One base character plus whatever combining marks attach to it.

    A "unit" is a base codepoint together with its following combining marks in
    NFD order, which is what a reader sees as a single letter. Units are used
    rather than raw codepoints because a Vietnamese letter is one codepoint in
    NFC and up to three in NFD, and the base stream must be indexable either way.
    """

    unit_index: int
    canonical_text: str
    """This unit as it appears in the canonical (NFC) string."""
    canonical_start: int
    canonical_end: int

    base_char: str
    """The unit with Vietnamese diacritics removed. Case is preserved; non-
    Vietnamese combining marks (diaeresis, cedilla, ...) are preserved too, so
    `Müller` keeps its umlaut while `Đường` becomes `Duong`."""
    base_start: int
    base_end: int

    letter_diacritic: LetterDiacritic
    """`NA` for anything that is not a letter; `NONE` for a letter carrying no
    Vietnamese letter diacritic."""

    observed_tone: ObservedTone
    """The tone mark visible *on this character*. `UNMARKED` when there is none.
    The syllable-level channel value lives on :class:`SyllableSpan`."""

    nfd_marks: tuple[str, ...] = ()
    """Every combining mark on this unit, in NFD canonical order. Reconstruction
    replays this tuple verbatim, which is what makes the round-trip exact even
    for marks this project does not classify as Vietnamese."""

    has_stroke: bool = False
    """True for đ/Đ, whose stroke is not a combining mark."""

    anomalies: tuple[Anomaly, ...] = ()

    @property
    def is_letter(self) -> bool:
        return self.letter_diacritic is not LetterDiacritic.NA


@dataclass(frozen=True)
class SyllableSpan:
    """A maximal alphabetic run: the orthographic candidate for one syllable.

    Vietnamese writes each syllable as a separate whitespace-delimited word, so
    a maximal alphabetic run is the natural candidate. Whether the run actually
    *is* Vietnamese is a separate question this layer does not answer; see
    :class:`~unmark.orthography.marks.Eligibility`.
    """

    span_index: int
    canonical_start: int
    canonical_end: int
    base_start: int
    base_end: int
    text: str
    """The span as it appears in the canonical string."""
    base_text: str

    observed_tone: ObservedTone
    """What is readable from the string: one of five marked tones, or
    `UNMARKED`. This is the tone channel value UNMARK would consume."""

    lexical_tone: Tone | None
    """The true tone, when it is knowable.

    A visible mark settles it. An *absent* mark does not: it is either a genuine
    `ngang` or a `ngang`-looking syllable whose mark was stripped. This field is
    therefore `None` for unmarked syllables unless the caller asserted the source
    text was clean (`decompose(..., source_is_clean=True)`), in which case it is
    `Tone.NGANG`. Never infer `NGANG` from `observed_tone == UNMARKED`."""

    tone_unit_index: int | None
    """Index of the character unit carrying the tone mark, or `None`.

    Retained because canonical tone placement is an open specification decision;
    once it is settled this becomes derivable rather than observed."""

    eligibility: Eligibility
    unit_indices: tuple[int, ...] = ()
    anomalies: tuple[Anomaly, ...] = ()

    @property
    def tone_is_knowable(self) -> bool:
        return self.lexical_tone is not None


@dataclass(frozen=True)
class DecomposedText:
    """The full decomposition of one input string.

    Invariant, verified in tests and by `scripts/g0_orthography_check.py`::

        recompose(decompose(x)) == canon(x)
    """

    original_text: str
    nfc_text: str
    canonical_text: str
    base_text: str
    units: tuple[CharacterUnit, ...]
    syllables: tuple[SyllableSpan, ...]
    source_is_clean: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- channel views ------------------------------------------------------
    @property
    def letter_channel(self) -> tuple[LetterDiacritic, ...]:
        """Per-character letter-diacritic states, aligned to `units`."""
        return tuple(u.letter_diacritic for u in self.units)

    @property
    def observed_tone_channel(self) -> tuple[ObservedTone, ...]:
        """Per-syllable observable tone states, aligned to `syllables`."""
        return tuple(s.observed_tone for s in self.syllables)

    @property
    def lexical_tone_channel(self) -> tuple[Tone | None, ...]:
        """Per-syllable lexical tones; `None` where genuinely unknowable."""
        return tuple(s.lexical_tone for s in self.syllables)

    @property
    def anomalies(self) -> tuple[Anomaly, ...]:
        seen: list[Anomaly] = []
        for item in (*self.units, *self.syllables):
            for anomaly in item.anomalies:
                if anomaly not in seen:
                    seen.append(anomaly)
        return tuple(seen)

    @property
    def is_canonical(self) -> bool:
        """Whether the input was already in canonical form."""
        return self.original_text == self.canonical_text
