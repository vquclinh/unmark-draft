"""Deterministic manual alignment for PhoBERT's slow tokenizer (B3B-1A).

Why this exists
---------------
The scientifically usable B3B-0 run found `offset_availability = ABSENT` for
every preprocessing path: the authoritative tokenizer is `PhobertTokenizer`
(`is_fast = False`), which returns no `offset_mapping`. Proposal v1.3 §4.4
propagates channel labels "by tracking character offsets through tokenization",
so that step is not implementable as written for this tokenizer.

Switching to a fast tokenizer to obtain offsets is **not** an option: the token
ids produced by the pinned slow tokenizer are the frozen encoder's own
vocabulary and remain authoritative. So the offsets have to be reconstructed.

The hypothesis
--------------
PhoBERT uses fastBPE, whose pieces carry a `@@` continuation suffix on every
piece except the last of a word::

    nghien -> ["ngh@@", "ien"]      ngh + ien == nghien

If that reconstruction is exact, each piece's character range inside the base
syllable follows deterministically, and syllable→subword mapping is recoverable
without native offsets.

**This module implements the hypothesis; it does not validate it.** Validation
requires the real tokenizer and happens in
`scripts/b3b1_phobert_alignment_probe.py` on Colab. Everything here is pure
standard library and is exercised locally against mock BPE sequences only.

Failure is explicit
-------------------
Nothing is ever labelled on a guess. An eligible Vietnamese syllable that
produces an unknown token, or whose pieces do not reconstruct its exact surface,
is reported as `ALIGNMENT_FAILURE` with a reason. Special tokens, punctuation and
non-Vietnamese spans are `NOT_APPLICABLE` in both orthography channels, per
proposal §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from unmark.orthography import Eligibility

CONTINUATION_MARKER = "@@"
"""fastBPE continuation suffix. A piece ending with it continues into the next."""


class SpanAlignmentStatus(Enum):
    """Outcome of aligning one span's tokens to its characters."""

    ALIGNED = "ALIGNED"
    """Every piece has an exact half-open character range in the span."""

    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    """The span could not be aligned. Channels must NOT be assigned."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """The span carries no orthography channels: special token, punctuation,
    digits, or a non-Vietnamese span (proposal 4.3)."""

    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class AlignmentFailureReason(Enum):
    """Why an alignment failed. Recorded, never absorbed."""

    NO_TOKENS = "NO_TOKENS"
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    SURFACE_MISMATCH = "SURFACE_MISMATCH"
    MALFORMED_CONTINUATION = "MALFORMED_CONTINUATION"
    """The final piece still carries the continuation marker, so the span's
    tokenization is not self-contained."""
    UNRESOLVED_ELIGIBILITY = "UNRESOLVED_ELIGIBILITY"
    """Eligibility was `UNDECIDED`. A scientific alignment must never label a
    span whose Vietnamese candidacy was not resolved."""


@dataclass(frozen=True)
class PieceAlignment:
    """One BPE piece and the exact characters of the span that produced it."""

    index: int
    token: str
    token_id: int | None
    surface: str
    """The token with its continuation marker removed."""
    start: int
    end: int
    """Half-open character range within the span text."""

    @property
    def is_continuation(self) -> bool:
        return self.token.endswith(CONTINUATION_MARKER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "token": self.token,
            "token_id": self.token_id,
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "is_continuation": self.is_continuation,
        }


@dataclass(frozen=True)
class SpanAlignment:
    """The alignment of one span's tokens onto its characters."""

    span_text: str
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    pieces: tuple[PieceAlignment, ...]
    status: SpanAlignmentStatus
    failure_reason: AlignmentFailureReason | None = None
    eligibility: Eligibility = Eligibility.UNDECIDED
    reconstructed: str = ""
    detail: str = ""

    @property
    def aligned(self) -> bool:
        return self.status is SpanAlignmentStatus.ALIGNED

    @property
    def subword_count(self) -> int:
        return len(self.tokens)

    @property
    def carries_channels(self) -> bool:
        """Whether orthography channels may be attached to this span.

        Only a successfully aligned, resolved Vietnamese candidate qualifies.
        """
        return self.aligned and self.eligibility is Eligibility.VIETNAMESE_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_text": self.span_text,
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "pieces": [p.to_dict() for p in self.pieces],
            "status": self.status.value,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "eligibility": self.eligibility.value,
            "reconstructed": self.reconstructed,
            "subword_count": self.subword_count,
            "carries_channels": self.carries_channels,
            "detail": self.detail,
        }


def piece_surface(token: str) -> str:
    """The characters a BPE piece contributes: the token minus its marker."""
    if token.endswith(CONTINUATION_MARKER):
        return token[: -len(CONTINUATION_MARKER)]
    return token


def reconstruct_surface(tokens: Sequence[str]) -> str:
    """Concatenate the pieces' surfaces."""
    return "".join(piece_surface(token) for token in tokens)


def align_span(
    span_text: str,
    tokens: Sequence[str],
    token_ids: Sequence[int] | None = None,
    *,
    eligibility: Eligibility = Eligibility.UNDECIDED,
    unk_token: str | None = None,
    require_resolved_eligibility: bool = True,
) -> SpanAlignment:
    """Map `tokens` onto the characters of `span_text`.

    Args:
        span_text: the exact base substring that was tokenized.
        tokens: its pieces, from tokenizing `span_text` alone with no special
            tokens.
        eligibility: the B3A verdict for this span.
        unk_token: the tokenizer's unknown token, so its presence can be
            detected rather than reconstructed around.
        require_resolved_eligibility: when true (the default), an `UNDECIDED`
            span fails rather than being aligned. A scientific run must never
            attach channels to a span whose candidacy was never resolved.

    Returns:
        A `SpanAlignment`. On failure `pieces` is empty and `failure_reason`
        says why -- no partial or guessed ranges are ever returned.
    """
    ids = tuple(token_ids or ())
    tokens = tuple(tokens)

    def fail(reason: AlignmentFailureReason, detail: str) -> SpanAlignment:
        return SpanAlignment(
            span_text=span_text, tokens=tokens, token_ids=ids, pieces=(),
            status=SpanAlignmentStatus.ALIGNMENT_FAILURE, failure_reason=reason,
            eligibility=eligibility, reconstructed=reconstruct_surface(tokens), detail=detail,
        )

    if require_resolved_eligibility and eligibility is Eligibility.UNDECIDED:
        return fail(
            AlignmentFailureReason.UNRESOLVED_ELIGIBILITY,
            "eligibility is UNDECIDED; resolve the Vietnamese syllable inventory before aligning",
        )

    if eligibility is Eligibility.NOT_APPLICABLE:
        return SpanAlignment(
            span_text=span_text, tokens=tokens, token_ids=ids, pieces=(),
            status=SpanAlignmentStatus.NOT_APPLICABLE, eligibility=eligibility,
            reconstructed=reconstruct_surface(tokens),
            detail="non-Vietnamese span: N/A in both orthography channels (proposal 4.3)",
        )

    if not tokens:
        return fail(AlignmentFailureReason.NO_TOKENS, "the tokenizer produced no pieces")

    if unk_token is not None and unk_token in tokens:
        positions = [i for i, t in enumerate(tokens) if t == unk_token]
        return fail(
            AlignmentFailureReason.UNKNOWN_TOKEN,
            f"unknown token at piece index(es) {positions}; the surface cannot be recovered",
        )

    if tokens[-1].endswith(CONTINUATION_MARKER):
        return fail(
            AlignmentFailureReason.MALFORMED_CONTINUATION,
            "the final piece still carries the continuation marker, so this span's "
            "tokenization is not self-contained",
        )

    reconstructed = reconstruct_surface(tokens)
    if reconstructed != span_text:
        return fail(
            AlignmentFailureReason.SURFACE_MISMATCH,
            f"pieces reconstruct {reconstructed!r}, which is not the span {span_text!r}",
        )

    pieces: list[PieceAlignment] = []
    cursor = 0
    for index, token in enumerate(tokens):
        surface = piece_surface(token)
        pieces.append(
            PieceAlignment(
                index=index,
                token=token,
                token_id=ids[index] if index < len(ids) else None,
                surface=surface,
                start=cursor,
                end=cursor + len(surface),
            )
        )
        cursor += len(surface)

    return SpanAlignment(
        span_text=span_text, tokens=tokens, token_ids=ids, pieces=tuple(pieces),
        status=SpanAlignmentStatus.ALIGNED, eligibility=eligibility,
        reconstructed=reconstructed,
        detail="exact surface reconstruction; every piece has a half-open character range",
    )


def characters_for_piece(alignment: SpanAlignment, piece_index: int) -> str:
    """The exact characters of the span that produced one piece."""
    piece = alignment.pieces[piece_index]
    return alignment.span_text[piece.start : piece.end]


def pieces_for_character(alignment: SpanAlignment, char_index: int) -> list[int]:
    """Which pieces a character contributes to.

    Character-level letter-diacritic states pool into the subword covering them
    (proposal §4.4 step 4); this is the lookup that makes that possible.
    """
    return [p.index for p in alignment.pieces if p.start <= char_index < p.end]


@dataclass(frozen=True)
class SequenceAlignment:
    """Per-span alignments for one full tokenizer input, plus reconciliation."""

    text: str
    spans: tuple[SpanAlignment, ...]
    full_sequence_tokens: tuple[str, ...] = ()
    composed_tokens: tuple[str, ...] = ()
    sequence_consistent: bool | None = None
    unexplained_tokens: tuple[str, ...] = ()
    detail: str = ""

    @property
    def aligned_spans(self) -> int:
        return sum(1 for s in self.spans if s.aligned)

    @property
    def failed_spans(self) -> tuple[SpanAlignment, ...]:
        return tuple(s for s in self.spans if s.status is SpanAlignmentStatus.ALIGNMENT_FAILURE)

    @property
    def channel_bearing_spans(self) -> int:
        return sum(1 for s in self.spans if s.carries_channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "full_sequence_tokens": list(self.full_sequence_tokens),
            "composed_tokens": list(self.composed_tokens),
            "sequence_consistent": self.sequence_consistent,
            "unexplained_tokens": list(self.unexplained_tokens),
            "aligned_spans": self.aligned_spans,
            "failed_spans": len(self.failed_spans),
            "channel_bearing_spans": self.channel_bearing_spans,
            "detail": self.detail,
        }


def compare_sequences(
    full_sequence: Sequence[str],
    composed: Sequence[str],
    special_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare the authoritative full-sequence tokenization with a composition.

    Special tokens are excluded from the comparison because they belong to the
    sequence, not to any span. Everything else must match exactly: a token the
    composition cannot account for is reported, never ignored.

    Deliberately compares *whole* sequences rather than concatenating only the
    Vietnamese spans -- punctuation and non-candidate regions are part of the
    tokenizer input and must be accounted for too, even though they carry no
    orthography channels.
    """
    specials = set(special_tokens)
    reference = [t for t in full_sequence if t not in specials]
    candidate = [t for t in composed if t not in specials]
    consistent = reference == candidate

    unexplained: list[str] = []
    if not consistent:
        remaining = list(candidate)
        for token in reference:
            if token in remaining:
                remaining.remove(token)
            else:
                unexplained.append(token)

    return {
        "consistent": consistent,
        "reference_length": len(reference),
        "composed_length": len(candidate),
        "unexplained_tokens": unexplained,
        "detail": (
            "per-span composition reproduces the authoritative sequence"
            if consistent
            else f"{len(unexplained)} token(s) in the authoritative sequence are unaccounted for"
        ),
    }


def summarize_alignments(alignments: Sequence[SpanAlignment]) -> dict[str, Any]:
    """Aggregate span alignments into the numbers the probe reports."""
    aligned = [a for a in alignments if a.aligned]
    failed = [a for a in alignments if a.status is SpanAlignmentStatus.ALIGNMENT_FAILURE]
    not_applicable = [a for a in alignments if a.status is SpanAlignmentStatus.NOT_APPLICABLE]
    counts = [a.subword_count for a in aligned]
    reasons: dict[str, int] = {}
    for failure in failed:
        key = failure.failure_reason.value if failure.failure_reason else "UNKNOWN"
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "total": len(alignments),
        "aligned": len(aligned),
        "failed": len(failed),
        "not_applicable": len(not_applicable),
        "failure_reasons": reasons,
        "alignment_rate": (len(aligned) / len(alignments)) if alignments else None,
        "mean_subwords_per_span": (sum(counts) / len(counts)) if counts else None,
        "max_subwords_per_span": max(counts) if counts else None,
        "spans_with_unknown_token": reasons.get(AlignmentFailureReason.UNKNOWN_TOKEN.value, 0),
        "surface_reconstruction_failures": reasons.get(
            AlignmentFailureReason.SURFACE_MISMATCH.value, 0
        ),
    }
