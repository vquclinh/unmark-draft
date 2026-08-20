"""Token spans, offset validation and coverage arithmetic.

Pure standard library. This is the analysis the Colab probe's raw output is fed
through, and it is what the local test suite exercises against mock tokenizer
output -- so the logic that will later decide alignment feasibility is testable
without transformers, torch, Java or a network.

Nothing here decides a preprocessing policy. It measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from unmark.alignment.contracts import AlignmentStatus, OffsetAvailability


@dataclass(frozen=True)
class TokenSpan:
    """One tokenizer token and where it sits in the text it was produced from."""

    index: int
    token: str
    token_id: int | None = None
    start: int | None = None
    end: int | None = None
    is_special: bool = False
    is_unknown: bool = False

    @property
    def has_offsets(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def length(self) -> int:
        if not self.has_offsets:
            return 0
        return max(0, self.end - self.start)

    def overlaps(self, start: int, end: int) -> bool:
        """Half-open interval overlap. Zero-width spans never overlap."""
        if not self.has_offsets or self.length == 0 or end <= start:
            return False
        return self.start < end and start < self.end

    def overlap_length(self, start: int, end: int) -> int:
        if not self.overlaps(start, end):
            return 0
        return min(self.end, end) - max(self.start, start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "token": self.token,
            "token_id": self.token_id,
            "start": self.start,
            "end": self.end,
            "is_special": self.is_special,
            "is_unknown": self.is_unknown,
        }


def validate_offsets(
    text: str,
    spans: Sequence[TokenSpan],
) -> tuple[OffsetAvailability, str]:
    """Classify how usable a tokenizer's offsets are, with a stated reason.

    Deliberately strict: §4.4 propagates channel labels by character offset, so
    an offset mapping that *looks* present but does not reproduce token surface
    forms would corrupt every label silently.
    """
    content = [s for s in spans if not s.is_special]
    if not content:
        return OffsetAvailability.NOT_PROBED, "no non-special tokens to check"
    if all(not s.has_offsets for s in content):
        return OffsetAvailability.ABSENT, "tokenizer returned no offset mapping"
    if any(not s.has_offsets for s in content):
        return OffsetAvailability.NATIVE_MALFORMED, "offsets present for only some tokens"

    for span in content:
        if span.start < 0 or span.end > len(text):
            return (
                OffsetAvailability.NATIVE_MALFORMED,
                f"token {span.index} offsets ({span.start}, {span.end}) outside text of length {len(text)}",
            )
        if span.end < span.start:
            return OffsetAvailability.NATIVE_MALFORMED, f"token {span.index} has end < start"

    ordered = sorted(content, key=lambda s: s.index)
    previous_end = None
    for span in ordered:
        if previous_end is not None and span.start < previous_end and span.length > 0:
            return (
                OffsetAvailability.NATIVE_MALFORMED,
                f"token {span.index} starts at {span.start} before the previous token ended at {previous_end}",
            )
        previous_end = max(previous_end or 0, span.end)

    mismatches = [
        span.index
        for span in ordered
        if span.length > 0 and text[span.start : span.end] != _surface(span.token)
    ]
    if mismatches:
        return (
            OffsetAvailability.NATIVE_INEXACT,
            f"{len(mismatches)} token(s) whose slice differs from the token string, first at index {mismatches[0]}",
        )
    return OffsetAvailability.NATIVE_EXACT, "every token slice reproduces its surface form"


def _surface(token: str) -> str:
    """Strip common sub-word continuation markers before comparing to a slice.

    Handles the BPE marker families a Vietnamese encoder may use. This is a
    comparison convenience for diagnosis only; it never rewrites data.
    """
    if token.startswith("##"):
        return token[2:]
    if token.startswith("▁"):  # SentencePiece
        return token[1:]
    if token.endswith("@@"):  # fastBPE, as used by PhoBERT
        return token[:-2]
    return token


def character_coverage(text: str, spans: Sequence[TokenSpan]) -> dict[str, Any]:
    """Which characters of `text` are covered by at least one token span."""
    covered = bytearray(len(text))
    for span in spans:
        if span.is_special or not span.has_offsets:
            continue
        for i in range(max(0, span.start), min(len(text), span.end)):
            covered[i] = 1
    total = len(text)
    hit = sum(covered)
    uncovered = [i for i, flag in enumerate(covered) if not flag]
    return {
        "text_length": total,
        "covered_characters": hit,
        "coverage_rate": (hit / total) if total else None,
        "uncovered_indices": uncovered[:50],
        "fully_covered": total > 0 and hit == total,
    }


def tokens_for_span(spans: Sequence[TokenSpan], start: int, end: int) -> list[int]:
    """Indices of tokens overlapping the half-open character range."""
    return [s.index for s in spans if not s.is_special and s.overlaps(start, end)]


def syllable_token_map(
    spans: Sequence[TokenSpan],
    syllable_ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Map syllable character ranges onto token indices.

    Reports the quantities §4.4 depends on: whether every syllable reaches at
    least one token, how many subwords each syllable costs, and whether any
    token straddles two syllables (which would make a single tone label
    ambiguous for that token).
    """
    mapping: list[list[int]] = []
    for start, end in syllable_ranges:
        mapping.append(tokens_for_span(spans, start, end))

    owners: dict[int, int] = {}
    straddling: list[int] = []
    for syllable_index, token_indices in enumerate(mapping):
        for token_index in token_indices:
            if token_index in owners and owners[token_index] != syllable_index:
                if token_index not in straddling:
                    straddling.append(token_index)
            owners.setdefault(token_index, syllable_index)

    unmapped = [i for i, tokens in enumerate(mapping) if not tokens]
    counts = [len(tokens) for tokens in mapping]
    return {
        "syllable_count": len(syllable_ranges),
        "syllable_to_tokens": mapping,
        "unmapped_syllables": unmapped,
        "all_syllables_mapped": not unmapped,
        "straddling_tokens": straddling,
        "has_straddling_tokens": bool(straddling),
        "subwords_per_syllable_mean": (sum(counts) / len(counts)) if counts else None,
        "subwords_per_syllable_max": max(counts) if counts else None,
    }


def alignment_status(
    availability: OffsetAvailability,
    coverage: dict[str, Any],
    syllable_map: dict[str, Any],
) -> tuple[AlignmentStatus, str]:
    """Overall verdict for one observation, with the reason stated."""
    if availability in (OffsetAvailability.ABSENT, OffsetAvailability.NOT_PROBED):
        return AlignmentStatus.UNALIGNED, f"offsets unusable: {availability.value}"
    if availability is OffsetAvailability.NATIVE_MALFORMED:
        return AlignmentStatus.UNALIGNED, "offset mapping is structurally malformed"
    if not syllable_map.get("all_syllables_mapped", False):
        return (
            AlignmentStatus.PARTIAL,
            f"{len(syllable_map.get('unmapped_syllables', []))} syllable(s) reached no token",
        )
    if not coverage.get("fully_covered", False):
        return AlignmentStatus.PARTIAL, "some characters are covered by no token"
    if availability is OffsetAvailability.NATIVE_INEXACT:
        return AlignmentStatus.PARTIAL, "offsets usable but token slices differ from surface forms"
    return AlignmentStatus.ALIGNED, "offsets exact, full coverage, every syllable mapped"
