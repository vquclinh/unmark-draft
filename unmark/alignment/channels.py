"""Deterministic orthographic channel projection onto the PhoBERT token grid.

This is the metadata layer between validated alignment (B3B-1B) and the neural
adapter that does not exist yet. It produces, for every authoritative subword
position, the tone label and the letter-diacritic contributors that later become
embedding lookups.

**No torch, no trainable parameters, no pooling layer, no model weights.**
Pure data.

What it consumes
----------------
The alignment metadata validated by the corrected B3B-1B run: raw-BPE pieces
over maximal non-whitespace chunks, each carrying an exact global half-open
character range in `b(x)`, overlaid with B1A/B3A orthographic regions. It never
re-derives Unicode structure — character labels come from the canonical
decomposition, never from the BPE token string.

Tone ownership: the unique-candidate rule
-----------------------------------------
Counting **distinct Vietnamese candidate contributors** decides the tone:

* 0 candidates -> `NA`;
* exactly 1 -> that candidate's observed tone, **even if the piece also covers
  punctuation or other non-applicable characters** — nothing competes for the
  label;
* >= 2 -> `NA`, with every contributor recorded.

The real probe showed why the earlier "any mixture -> no tone" rule was too
conservative: of 191 piece overlays, 2 were mixed and both mixed a *single*
candidate with punctuation (`en` + `-`; `.` + `com`). Zero pieces spanned two
distinct candidates. Discarding those two tones would have thrown away
information that was never ambiguous.

A multi-candidate piece is never resolved by majority length, by first or last,
or by averaging categorical tone ids — including when the candidates happen to
share an observed tone. Sharing a value is not the same as having one source.

Tone is never inferred from the stripped BPE surface. It comes from the one
contributing candidate's already-derived orthographic metadata, and the deploy
pathway keeps `UNMARKED` ambiguous: no restoration, no lexical inference, no
reintroduction of lexical `NGANG`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from unmark.alignment.manual import (
    OrthographicRegion,
    PieceAlignment,
    PieceOverlay,
    ToneOwnership,
)
from unmark.orthography import DecomposedText, Eligibility, LetterDiacritic, ObservedTone

LETTER_POOLING_RULE = (
    "mean in embedding space over the APPLICABLE letter contributors; NONE is "
    "included in that mean; NA contributors are excluded; a token with zero "
    "applicable contributors has token-level letter channel NA"
)
"""The intended pooling rule, persisted as documentation.

Proposal §4.4 step 4 pools letter labels in embedding space rather than label
space. Recorded here so the adapter implements the decided rule rather than
re-deciding it. **Not implemented in this task.**
"""


class TokenToneLabel(Enum):
    """Tone label at the **token** level.

    A projection of `ObservedTone`, not a duplicate of it: `NA` is a token-level
    state that no syllable can have. Proposal §4.4 requires every non-Vietnamese
    subword to carry `N/A` in both channels, and a syllable is Vietnamese by
    construction — so the extra state belongs here.
    """

    UNMARKED = "UNMARKED"
    SAC = "SAC"
    HUYEN = "HUYEN"
    HOI = "HOI"
    NGA = "NGA"
    NANG = "NANG"
    NA = "NA"

    @classmethod
    def from_observed_tone(cls, tone: ObservedTone) -> TokenToneLabel:
        return cls[tone.name]

    @property
    def is_not_applicable(self) -> bool:
        return self is TokenToneLabel.NA


@dataclass(frozen=True)
class CharacterContribution:
    """One base character that a subword drew from, with its channel metadata."""

    char_index: int
    """Global index in `b(x)`."""
    character: str
    letter_diacritic: LetterDiacritic
    eligibility: Eligibility
    region_index: int | None = None

    @property
    def is_applicable(self) -> bool:
        """Whether this character participates in the letter channel at all.

        `NONE` is applicable -- it means "a letter that could carry a Vietnamese
        letter diacritic and does not", which is real information. Only `NA`
        (punctuation, digits, symbols) is excluded.
        """
        return self.letter_diacritic is not LetterDiacritic.NA

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_index": self.char_index,
            "character": self.character,
            "letter_diacritic": self.letter_diacritic.value,
            "eligibility": self.eligibility.value,
            "region_index": self.region_index,
            "is_applicable": self.is_applicable,
        }


@dataclass(frozen=True)
class ToneProjection:
    """The tone channel for one subword position."""

    ownership: ToneOwnership
    label: TokenToneLabel
    source_region_index: int | None = None
    candidate_region_indices: tuple[int, ...] = ()
    detail: str = ""

    @property
    def carries_tone(self) -> bool:
        return self.ownership is ToneOwnership.SINGLE_CANDIDATE

    @property
    def is_ambiguous(self) -> bool:
        return self.ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.value,
            "label": self.label.value,
            "source_region_index": self.source_region_index,
            "candidate_region_indices": list(self.candidate_region_indices),
            "carries_tone": self.carries_tone,
            "is_ambiguous": self.is_ambiguous,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LetterProjection:
    """The letter-diacritic channel for one subword position."""

    contributions: tuple[CharacterContribution, ...] = ()
    """Every contributing base character, in deterministic source order."""

    token_label: LetterDiacritic = LetterDiacritic.NA
    """Token-level summary: `NA` when no contributor is applicable. This is a
    label-space convenience for reporting; the adapter pools in embedding space
    over `applicable_labels`."""

    pooling_rule: str = LETTER_POOLING_RULE

    @property
    def applicable(self) -> tuple[CharacterContribution, ...]:
        return tuple(c for c in self.contributions if c.is_applicable)

    @property
    def applicable_labels(self) -> tuple[LetterDiacritic, ...]:
        """What the adapter will pool, in source order. `NONE` is included."""
        return tuple(c.letter_diacritic for c in self.applicable)

    @property
    def has_applicable_contributors(self) -> bool:
        return bool(self.applicable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributions": [c.to_dict() for c in self.contributions],
            "applicable_labels": [label.value for label in self.applicable_labels],
            "token_label": self.token_label.value,
            "has_applicable_contributors": self.has_applicable_contributors,
            "pooling_rule": self.pooling_rule,
        }


@dataclass(frozen=True)
class TokenOrthographyProjection:
    """Both orthography channels for one authoritative subword position."""

    token_index: int
    token: str
    token_id: int | None
    tone: ToneProjection
    letter: LetterProjection
    global_start: int | None = None
    global_end: int | None = None
    is_special: bool = False
    has_unknown_token_id: bool = False

    @property
    def has_source_range(self) -> bool:
        return self.global_start is not None and self.global_end is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token": self.token,
            "token_id": self.token_id,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "is_special": self.is_special,
            "has_unknown_token_id": self.has_unknown_token_id,
            "has_source_range": self.has_source_range,
            "tone": self.tone.to_dict(),
            "letter": self.letter.to_dict(),
        }


# ---------------------------------------------------------------------------
# Character labels from the canonical decomposition
# ---------------------------------------------------------------------------
def character_letter_labels(decomposed: DecomposedText) -> tuple[LetterDiacritic, ...]:
    """Per-base-character letter-diacritic labels, indexed by `base_text`.

    Read from the canonical decomposition's character units -- **never** derived
    from a BPE token string, and never by decomposing Unicode a second time
    inside the alignment layer. A unit whose base form spans several characters
    (a non-Vietnamese mark kept on the base) gives each of them its label.
    """
    labels: list[LetterDiacritic] = [LetterDiacritic.NA] * len(decomposed.base_text)
    for unit in decomposed.units:
        for index in range(unit.base_start, min(unit.base_end, len(labels))):
            labels[index] = unit.letter_diacritic
    return tuple(labels)


def _region_at(regions: Sequence[OrthographicRegion], char_index: int) -> OrthographicRegion | None:
    for region in regions:
        if region.start <= char_index < region.end:
            return region
    return None


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_special_token(
    token_index: int, token: str, token_id: int | None = None
) -> TokenOrthographyProjection:
    """A special token carries no orthography and has no source characters.

    Boundary, padding, mask and unknown *special* tokens are positions in the
    encoder sequence that no character produced. They are never given a
    Vietnamese tone or letter label, and no source range is fabricated for them.
    """
    return TokenOrthographyProjection(
        token_index=token_index,
        token=token,
        token_id=token_id,
        tone=ToneProjection(
            ownership=ToneOwnership.NOT_APPLICABLE,
            label=TokenToneLabel.NA,
            detail="special token: no source characters, so no orthography channels",
        ),
        letter=LetterProjection(token_label=LetterDiacritic.NA),
        global_start=None,
        global_end=None,
        is_special=True,
    )


def project_piece(
    token_index: int,
    piece: PieceAlignment,
    overlay: PieceOverlay,
    base_text: str,
    letter_labels: Sequence[LetterDiacritic],
    regions: Sequence[OrthographicRegion],
    region_tones: dict[int, ObservedTone] | None = None,
) -> TokenOrthographyProjection:
    """Project both channels onto one authoritative subword.

    Args:
        piece: the aligned BPE piece, carrying its exact global range.
        overlay: its orthographic-region attribution, already decided by the
            unique-candidate rule.
        letter_labels: per-base-character labels from the canonical
            decomposition.
        region_tones: observed tone per Vietnamese candidate region.
    """
    region_tones = region_tones or {}

    contributions = tuple(
        CharacterContribution(
            char_index=index,
            character=base_text[index],
            letter_diacritic=(
                letter_labels[index] if index < len(letter_labels) else LetterDiacritic.NA
            ),
            eligibility=(
                _region_at(regions, index).eligibility
                if _region_at(regions, index)
                else Eligibility.NOT_APPLICABLE
            ),
            region_index=(
                _region_at(regions, index).index if _region_at(regions, index) else None
            ),
        )
        for index in range(piece.global_start, min(piece.global_end, len(base_text)))
    )

    applicable = [c for c in contributions if c.is_applicable]
    distinct_labels = {c.letter_diacritic for c in applicable}
    token_label = (
        LetterDiacritic.NA
        if not applicable
        else (next(iter(distinct_labels)) if len(distinct_labels) == 1 else LetterDiacritic.NONE)
    )

    if overlay.tone_ownership is ToneOwnership.SINGLE_CANDIDATE:
        region_index = overlay.tone_region_index
        observed = region_tones.get(region_index) if region_index is not None else None
        label = (
            TokenToneLabel.from_observed_tone(observed) if observed is not None else TokenToneLabel.NA
        )
    else:
        region_index, label = None, TokenToneLabel.NA

    return TokenOrthographyProjection(
        token_index=token_index,
        token=piece.token,
        token_id=piece.token_id,
        tone=ToneProjection(
            ownership=overlay.tone_ownership,
            label=label,
            source_region_index=region_index,
            candidate_region_indices=overlay.candidate_region_indices,
            detail=overlay.detail,
        ),
        letter=LetterProjection(contributions=contributions, token_label=token_label),
        global_start=piece.global_start,
        global_end=piece.global_end,
        is_special=False,
        has_unknown_token_id=piece.has_unknown_token_id,
    )


def summarize_projections(projections: Sequence[TokenOrthographyProjection]) -> dict[str, Any]:
    """Aggregate projections into the numbers a probe reports."""
    ownership: dict[str, int] = {}
    tone_labels: dict[str, int] = {}
    for projection in projections:
        key = projection.tone.ownership.value
        ownership[key] = ownership.get(key, 0) + 1
        label = projection.tone.label.value
        tone_labels[label] = tone_labels.get(label, 0) + 1
    return {
        "total_tokens": len(projections),
        "special_tokens": sum(1 for p in projections if p.is_special),
        "tokens_with_tone": sum(1 for p in projections if p.tone.carries_tone),
        "tokens_tone_ambiguous": sum(1 for p in projections if p.tone.is_ambiguous),
        "tokens_with_applicable_letters": sum(
            1 for p in projections if p.letter.has_applicable_contributors
        ),
        "tokens_with_unknown_token_id": sum(1 for p in projections if p.has_unknown_token_id),
        "tone_ownership_counts": ownership,
        "tone_label_counts": tone_labels,
    }
