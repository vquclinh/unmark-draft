"""Deterministic character alignment for PhoBERT's slow tokenizer (B3B-1B).

What the real tokenizer showed
------------------------------
B3B-1A hypothesised that tokenizing each B1A/B3A linguistic span independently
and composing the pieces would reproduce the authoritative sequence. The first
real Colab run refuted it: **6/13** sentences agreed, despite every span
reconstructing its own surface exactly.

The cause is granularity. PhoBERT's fastBPE runs over **maximal non-whitespace
chunks**, not over linguistic spans, so punctuation, hyphens, URLs and e-mail
addresses change the BPE segmentation of the whole chunk they sit in::

    "nhien."      authoritative  ["nhi@@", "en@@", "."]
                  span-composed  ["nh@@", "ien"] + ["."]        <- different
    "VNU-HCM"     authoritative  ["VN@@", "U-@@", "HCM"]
    "(VAT"        authoritative  ["(@@", "VAT"]
    "Viet-Nam"    authoritative  ["Viet@@", "-@@", "Nam"]

Re-running the composition over whole `\\S+` chunks gave **13/13** token
agreement, **13/13** id agreement, and 119/119 chunk surfaces reconstructed.

The contract this module implements
-----------------------------------
The authoritative model token grid is always `T(b(x))` from the pinned slow
tokenizer. **This module never defines it.** It reconstructs a character map
alongside it:

1. take the exact base text `b(x)`;
2. split it into maximal non-whitespace chunks, keeping global ranges;
3. tokenize each *whole chunk* with the same tokenizer, no special tokens;
4. use **raw BPE pieces**, before any id→token round trip;
5. reconstruct the chunk by stripping the fastBPE continuation marker;
6. require exact surface equality;
7. derive a local half-open range per piece;
8. translate to global ranges in `b(x)`;
9. compose the chunks and verify against the authoritative tokens **and** ids;
10. overlay the B1A/B3A orthographic spans onto the global ranges.

Vocabulary OOV is not alignment failure
---------------------------------------
`tokenizer.tokenize("khut")` returns `["khut"]` — the raw surface — while the id
is `3`, the unknown id, and `convert_ids_to_tokens` gives back `<unk>`. The
*surface* is exactly recoverable; only the vocabulary lookup fails. B3B-1A
conflated the two and wrongly reported an alignment failure. They are now
separate: such a piece is `ALIGNED` with `has_unknown_token_id = True`.

No torch, no transformers, no tokenizer. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from unmark.orthography import Eligibility

CONTINUATION_MARKER = "@@"
"""fastBPE continuation suffix: a piece carrying it continues into the next."""

_CHUNK_PATTERN = re.compile(r"\S+")


class AlignmentStatusB(Enum):
    """Outcome of aligning one unit's raw pieces to its characters."""

    ALIGNED = "ALIGNED"
    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class AlignmentFailureReason(Enum):
    """Why an alignment failed. Recorded, never absorbed.

    `UNKNOWN_TOKEN` is deliberately absent: an unknown *vocabulary id* does not
    prevent the *surface* from being recovered, and treating it as failure was
    the B3B-1A defect.
    """

    NO_TOKENS = "NO_TOKENS"
    SURFACE_MISMATCH = "SURFACE_MISMATCH"
    MALFORMED_CONTINUATION = "MALFORMED_CONTINUATION"
    RANGE_ERROR = "RANGE_ERROR"
    """Piece ranges were non-monotonic, overlapping, or did not cover the unit."""
    UNRESOLVED_ELIGIBILITY = "UNRESOLVED_ELIGIBILITY"
    """A scientific channel assignment was requested for a span whose Vietnamese
    candidacy was never resolved."""


class ToneOwnership(Enum):
    """Whether a BPE piece can be assigned a syllable's tone label."""

    VIETNAMESE = "VIETNAMESE"
    """Every contributing character comes from one Vietnamese candidate span."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """No contributing character comes from a Vietnamese candidate span."""

    MIXED = "MIXED"
    """The piece straddles a Vietnamese candidate and something else. Recorded,
    never resolved by guessing -- see `docs/spec/decisions.md` D-B3B1B-002."""

    UNRESOLVED = "UNRESOLVED"
    """A contributing span's eligibility was `UNDECIDED`."""


# ---------------------------------------------------------------------------
# Whitespace chunking
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """A maximal non-whitespace run of the base text, with its global range."""

    index: int
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "start": self.start, "end": self.end}


def whitespace_chunks(text: str) -> tuple[Chunk, ...]:
    """Split into maximal `\\S+` chunks, preserving exact global ranges.

    These are the units PhoBERT's BPE actually operates on, so they are the
    units the alignment must reconstruct. Whitespace itself produces no tokens
    and belongs to no chunk.
    """
    return tuple(
        Chunk(index=i, text=match.group(), start=match.start(), end=match.end())
        for i, match in enumerate(_CHUNK_PATTERN.finditer(text))
    )


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PieceAlignment:
    """One raw BPE piece and the exact characters that produced it."""

    index: int
    token: str
    """The RAW BPE piece, before any id round trip. For an OOV surface this is
    the surface itself, not `<unk>`."""
    token_id: int | None
    surface: str
    local_start: int
    local_end: int
    """Half-open range within the chunk."""
    global_start: int
    global_end: int
    """Half-open range within the full base text."""
    has_unknown_token_id: bool = False
    """The vocabulary lookup failed. Independent of surface recoverability."""

    @property
    def is_continuation(self) -> bool:
        return self.token.endswith(CONTINUATION_MARKER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "token": self.token,
            "token_id": self.token_id,
            "surface": self.surface,
            "local_start": self.local_start,
            "local_end": self.local_end,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "is_continuation": self.is_continuation,
            "has_unknown_token_id": self.has_unknown_token_id,
        }


def piece_surface(token: str) -> str:
    """The characters a raw BPE piece contributes."""
    if token.endswith(CONTINUATION_MARKER):
        return token[: -len(CONTINUATION_MARKER)]
    return token


def reconstruct_surface(tokens: Sequence[str]) -> str:
    return "".join(piece_surface(token) for token in tokens)


# ---------------------------------------------------------------------------
# Chunk alignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkAlignment:
    """Raw pieces of one whitespace chunk mapped onto its characters."""

    chunk: Chunk
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    pieces: tuple[PieceAlignment, ...]
    status: AlignmentStatusB
    failure_reason: AlignmentFailureReason | None = None
    reconstructed: str = ""
    detail: str = ""

    @property
    def aligned(self) -> bool:
        return self.status is AlignmentStatusB.ALIGNED

    @property
    def subword_count(self) -> int:
        return len(self.tokens)

    @property
    def unknown_id_count(self) -> int:
        return sum(1 for p in self.pieces if p.has_unknown_token_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "pieces": [p.to_dict() for p in self.pieces],
            "status": self.status.value,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "reconstructed": self.reconstructed,
            "subword_count": self.subword_count,
            "unknown_id_count": self.unknown_id_count,
            "detail": self.detail,
        }


def align_chunk(
    chunk: Chunk,
    tokens: Sequence[str],
    token_ids: Sequence[int] | None = None,
    *,
    unk_token_id: int | None = None,
) -> ChunkAlignment:
    """Map one chunk's RAW BPE pieces onto its characters.

    Args:
        chunk: the whitespace chunk, carrying its global range.
        tokens: raw pieces from tokenizing the **whole chunk**, no special
            tokens. Must be the raw strings, not an id round trip, or an OOV
            surface would already have been destroyed.
        token_ids: the corresponding vocabulary ids.
        unk_token_id: the unknown id, so OOV can be *reported* rather than
            mistaken for a surface failure.

    Returns:
        A `ChunkAlignment`. On failure `pieces` is empty — no partial or guessed
        ranges are ever returned.
    """
    tokens = tuple(tokens)
    ids = tuple(token_ids or ())

    def fail(reason: AlignmentFailureReason, detail: str) -> ChunkAlignment:
        return ChunkAlignment(
            chunk=chunk, tokens=tokens, token_ids=ids, pieces=(),
            status=AlignmentStatusB.ALIGNMENT_FAILURE, failure_reason=reason,
            reconstructed=reconstruct_surface(tokens), detail=detail,
        )

    if not tokens:
        return fail(AlignmentFailureReason.NO_TOKENS, "the tokenizer produced no pieces")

    if tokens[-1].endswith(CONTINUATION_MARKER):
        return fail(
            AlignmentFailureReason.MALFORMED_CONTINUATION,
            "the final piece still carries the continuation marker, so this chunk's "
            "tokenization is not self-contained",
        )

    reconstructed = reconstruct_surface(tokens)
    if reconstructed != chunk.text:
        return fail(
            AlignmentFailureReason.SURFACE_MISMATCH,
            f"raw pieces reconstruct {reconstructed!r}, not the chunk {chunk.text!r}",
        )

    pieces: list[PieceAlignment] = []
    cursor = 0
    for index, token in enumerate(tokens):
        surface = piece_surface(token)
        token_id = ids[index] if index < len(ids) else None
        pieces.append(
            PieceAlignment(
                index=index,
                token=token,
                token_id=token_id,
                surface=surface,
                local_start=cursor,
                local_end=cursor + len(surface),
                global_start=chunk.start + cursor,
                global_end=chunk.start + cursor + len(surface),
                # Reported, never a failure: the surface above is exact.
                has_unknown_token_id=(unk_token_id is not None and token_id == unk_token_id),
            )
        )
        cursor += len(surface)

    problem = _range_problem(pieces, chunk)
    if problem:
        return fail(AlignmentFailureReason.RANGE_ERROR, problem)

    return ChunkAlignment(
        chunk=chunk, tokens=tokens, token_ids=ids, pieces=tuple(pieces),
        status=AlignmentStatusB.ALIGNED, reconstructed=reconstructed,
        detail="exact raw-BPE surface reconstruction with monotonic global ranges",
    )


def _range_problem(pieces: Sequence[PieceAlignment], chunk: Chunk) -> str | None:
    """Whether the derived ranges tile the chunk monotonically."""
    previous_end = chunk.start
    for piece in pieces:
        if piece.global_start != previous_end:
            return (
                f"piece {piece.index} starts at {piece.global_start}, "
                f"leaving a gap or overlap after {previous_end}"
            )
        if piece.global_end < piece.global_start:
            return f"piece {piece.index} has end before start"
        previous_end = piece.global_end
    if previous_end != chunk.end:
        return f"pieces cover up to {previous_end}, but the chunk ends at {chunk.end}"
    return None


# ---------------------------------------------------------------------------
# Orthographic overlay
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrthographicRegion:
    """A B1A/B3A region of the base text, with its global range.

    These are **orthographic metadata boundaries, not tokenization boundaries.**
    A BPE piece may cover part of one, or straddle several.
    """

    index: int
    text: str
    start: int
    end: int
    eligibility: Eligibility
    is_syllable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "text": self.text, "start": self.start, "end": self.end,
            "eligibility": self.eligibility.value, "is_syllable": self.is_syllable,
        }


@dataclass(frozen=True)
class PieceContribution:
    """The characters one orthographic region contributed to one BPE piece."""

    region_index: int
    eligibility: Eligibility
    overlap_start: int
    overlap_end: int
    is_syllable: bool = True

    @property
    def length(self) -> int:
        return self.overlap_end - self.overlap_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_index": self.region_index,
            "eligibility": self.eligibility.value,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
            "length": self.length,
            "is_syllable": self.is_syllable,
        }


@dataclass(frozen=True)
class PieceOverlay:
    """A BPE piece together with every orthographic region it draws from."""

    piece_index: int
    global_start: int
    global_end: int
    contributions: tuple[PieceContribution, ...]
    tone_ownership: ToneOwnership
    tone_region_index: int | None = None
    detail: str = ""

    @property
    def is_mixed(self) -> bool:
        return self.tone_ownership is ToneOwnership.MIXED

    @property
    def carries_tone(self) -> bool:
        return self.tone_ownership is ToneOwnership.VIETNAMESE

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece_index": self.piece_index,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "contributions": [c.to_dict() for c in self.contributions],
            "tone_ownership": self.tone_ownership.value,
            "tone_region_index": self.tone_region_index,
            "is_mixed": self.is_mixed,
            "carries_tone": self.carries_tone,
            "detail": self.detail,
        }


def overlay_orthography(
    pieces: Sequence[PieceAlignment],
    regions: Sequence[OrthographicRegion],
) -> tuple[PieceOverlay, ...]:
    """Attribute each BPE piece's characters to orthographic regions.

    Alignment is by **character-range overlap**, because BPE boundaries and
    linguistic boundaries do not coincide. Every contributing region is recorded;
    a piece straddling a Vietnamese candidate and something else is marked
    `MIXED` and is **not** claimed to be Vietnamese.
    """
    overlays: list[PieceOverlay] = []
    for piece in pieces:
        contributions = [
            PieceContribution(
                region_index=region.index,
                eligibility=region.eligibility,
                overlap_start=max(piece.global_start, region.start),
                overlap_end=min(piece.global_end, region.end),
                is_syllable=region.is_syllable,
            )
            for region in regions
            if region.start < piece.global_end and piece.global_start < region.end
        ]
        contributions = [c for c in contributions if c.length > 0]

        vietnamese = [c for c in contributions if c.eligibility is Eligibility.VIETNAMESE_CANDIDATE]
        undecided = [c for c in contributions if c.eligibility is Eligibility.UNDECIDED]
        other = [c for c in contributions if c not in vietnamese and c not in undecided]

        if undecided:
            ownership, region_index = ToneOwnership.UNRESOLVED, None
            detail = "a contributing region has UNDECIDED eligibility; resolve the inventory"
        elif not vietnamese:
            ownership, region_index = ToneOwnership.NOT_APPLICABLE, None
            detail = "no contributing character comes from a Vietnamese candidate span"
        elif len(vietnamese) == 1 and not other:
            ownership = ToneOwnership.VIETNAMESE
            region_index = vietnamese[0].region_index
            detail = "every contributing character comes from one Vietnamese candidate span"
        else:
            ownership, region_index = ToneOwnership.MIXED, None
            detail = (
                f"piece draws from {len(contributions)} regions "
                f"({len(vietnamese)} Vietnamese, {len(other)} other); tone ownership is not "
                "decided here -- see docs/spec/decisions.md D-B3B1B-002"
            )

        overlays.append(
            PieceOverlay(
                piece_index=piece.index,
                global_start=piece.global_start,
                global_end=piece.global_end,
                contributions=tuple(contributions),
                tone_ownership=ownership,
                tone_region_index=region_index,
                detail=detail,
            )
        )
    return tuple(overlays)


def characters_for_piece(text: str, piece: PieceAlignment) -> str:
    """The exact base-text characters that produced a piece."""
    return text[piece.global_start : piece.global_end]


# ---------------------------------------------------------------------------
# Composition and the token-grid invariant
# ---------------------------------------------------------------------------
def compose(alignments: Sequence[ChunkAlignment]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Concatenate the chunks' raw tokens and ids, in order."""
    tokens: list[str] = []
    ids: list[int] = []
    for alignment in alignments:
        tokens.extend(alignment.tokens)
        ids.extend(alignment.token_ids)
    return tuple(tokens), tuple(ids)


def verify_token_grid(
    composed_tokens: Sequence[str],
    composed_ids: Sequence[int],
    authoritative_tokens: Sequence[str],
    authoritative_ids: Sequence[int],
) -> dict[str, Any]:
    """Check chunk composition against the authoritative `T(b(x))`.

    Both the raw tokens and the ids must match **exactly**. The authoritative
    sequence is the token grid; this only verifies that the character map was
    built over the same units. Any authoritative token the composition cannot
    account for is reported.
    """
    tokens_match = tuple(composed_tokens) == tuple(authoritative_tokens)
    ids_match = tuple(composed_ids) == tuple(authoritative_ids)

    unexplained: list[dict[str, Any]] = []
    if not tokens_match:
        for position, expected in enumerate(authoritative_tokens):
            actual = composed_tokens[position] if position < len(composed_tokens) else None
            if actual != expected:
                unexplained.append({"position": position, "authoritative": expected, "composed": actual})

    return {
        "tokens_match": tokens_match,
        "ids_match": ids_match,
        "consistent": tokens_match and ids_match,
        "authoritative_length": len(authoritative_tokens),
        "composed_length": len(composed_tokens),
        "unexplained_tokens": unexplained[:20],
        "detail": (
            "chunk composition reproduces the authoritative token grid exactly"
            if tokens_match and ids_match
            else f"{len(unexplained)} position(s) differ from the authoritative sequence"
        ),
    }


def summarize_chunk_alignments(alignments: Sequence[ChunkAlignment]) -> dict[str, Any]:
    """Aggregate chunk alignments into the numbers the probe reports."""
    aligned = [a for a in alignments if a.aligned]
    failed = [a for a in alignments if a.status is AlignmentStatusB.ALIGNMENT_FAILURE]
    counts = [a.subword_count for a in aligned]
    reasons: dict[str, int] = {}
    for failure in failed:
        key = failure.failure_reason.value if failure.failure_reason else "UNKNOWN"
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "total_chunks": len(alignments),
        "aligned": len(aligned),
        "failed": len(failed),
        "failure_reasons": reasons,
        "surface_reconstruction_failures": reasons.get(
            AlignmentFailureReason.SURFACE_MISMATCH.value, 0
        ),
        "range_failures": reasons.get(AlignmentFailureReason.RANGE_ERROR.value, 0),
        "chunks_with_unknown_token_id": sum(1 for a in aligned if a.unknown_id_count),
        "unknown_token_ids": sum(a.unknown_id_count for a in aligned),
        "mean_subwords_per_chunk": (sum(counts) / len(counts)) if counts else None,
        "max_subwords_per_chunk": max(counts) if counts else None,
    }
