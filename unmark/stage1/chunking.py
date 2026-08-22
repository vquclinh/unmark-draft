"""Deterministic, tokenizer-aware pre-chunking. **Torch-free.**

Implements the Audit-028 chunking contract (D-S1B-002), as corrected by
D-S1B-009 after the real corpus was inspected.

Seven requirements, each enforced rather than documented:

1. **Preserve the source exactly** -- chunks are contiguous half-open slices of
   the original ``content`` that tile ``[0, len(content))`` with no gaps and no
   overlaps, so ``"".join(texts) == content`` byte for byte. Whitespace is never
   collapsed, regenerated or inserted.
2. **No normalization** -- the chunker never canonicalises, repairs or rewrites.
   It only decides *where* to cut.
3. **Stable ids** -- ``{document_id}#{chunk_index}``.
4. **Fits `max_length` on BOTH tokenizer paths** -- the reference path
   (``canon(x)``) and the base path (``b(canon(x))``) have different lengths and
   separate padding domains, so both are measured on every emitted chunk.
5. **Never cut inside a Vietnamese candidate span**, and never inside a
   character unit (a base codepoint plus its combining marks).
6. **Runs only after the document-level partition exists.**
7. **Every chunk inherits its parent document's partition.**

Requirements 6 and 7 are structural: `chunk_document` *takes* a partition and
copies it onto every chunk. No code path assigns a partition to a chunk.

**Where the cut boundaries come from.** Audit 029 originally implemented "never
split a syllable" as "only ever cut at whitespace". Real UVW-2026 disproved that
implication: the corpus contains maximal **non-whitespace** units far larger than
`max_length` (observed up to 1 707 tokens), typically underscored article titles.
Whitespace-only cutting therefore could not prepare the locked corpus at all.

The authoritative definition lives in `unmark.orthography`: `decompose`
segments text into **maximal alphabetic runs** (`SyllableSpan`), and it is those
runs -- not whitespace-delimited words -- that must not be bisected. An
underscore, hyphen or comma inside a long title is a span *boundary*, so cutting
there is orthographically safe. `safe_cut_offsets` below is a pure **query** over
`decompose`'s own output: it introduces no second syllable parser and no new
linguistic rule.

**No truncation, ever.** A region that genuinely cannot be subdivided raises
`ChunkingViolation` with enough provenance to diagnose it. Text is never dropped.

The length functions are injected, so the whole contract is testable with a
lightweight mock and the real pinned tokenizer is not needed to prove it.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from unmark.orthography import decompose
from unmark.stage1.corpus import CorpusContractViolation, CorpusDocument
from unmark.stage1.protocol import CHUNK_ID_TEMPLATE, CHUNK_SCHEMA_VERSION, MAX_LENGTH


class ChunkingViolation(CorpusContractViolation):
    """Raised when text cannot be chunked within the locked contract."""


LengthFn = Callable[[str], int]
"""`text -> token count INCLUDING the model's special tokens`. Injected so the
contract is testable without the real tokenizer."""


@dataclass(frozen=True)
class PreparedChunk:
    """One Stage-1 training example, bound to its parent and its partition.

    `source_start`/`source_end` are the half-open range of the **original**
    document content this chunk was taken from. They exist so that exact
    reconstruction is a checkable property of the artifact rather than a claim:
    `verify_tiles_source` re-derives the document from the ranges alone.
    """

    chunk_id: str
    document_id: str
    partition: str
    chunk_index: int
    text: str
    source_start: int
    source_end: int
    reference_length: int
    base_length: int
    source_shard: str
    schema_version: str = CHUNK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.partition not in {"train", "dev"}:
            raise ChunkingViolation(f"unknown partition {self.partition!r}")
        expected = CHUNK_ID_TEMPLATE.format(
            document_id=self.document_id, chunk_index=self.chunk_index
        )
        if self.chunk_id != expected:
            raise ChunkingViolation(f"chunk id {self.chunk_id!r} does not match {expected!r}")
        if self.source_end <= self.source_start:
            raise ChunkingViolation(
                f"chunk {self.chunk_id!r} has empty range "
                f"[{self.source_start}, {self.source_end})"
            )
        if len(self.text) != self.source_end - self.source_start:
            raise ChunkingViolation(
                f"chunk {self.chunk_id!r} text length {len(self.text)} does not match its "
                f"source range width {self.source_end - self.source_start}; the chunker "
                "must slice, never rewrite"
            )


# ---------------------------------------------------------------------------
# Safe cut offsets -- a QUERY over the authoritative orthography, not a parser
# ---------------------------------------------------------------------------
def safe_cut_offsets(text: str, classifier: Callable[[str], Any] | None = None) -> frozenset[int]:
    """Offsets in `text` at which a cut changes no orthographic semantics.

    Derived entirely from `unmark.orthography.decompose`:

    * candidate offsets are **character-unit boundaries**, so a cut can never
      land between a base codepoint and its combining marks;
    * offsets strictly inside a `SyllableSpan` -- a maximal alphabetic run --
      are removed, so a Vietnamese candidate is never bisected.

    Returns the empty set when `canon(text) != text`. `decompose` reports offsets
    into the *canonical* string, so for non-canonically-spelled input those
    offsets do not address the original, and using them would cut in the wrong
    place. The chunker treats an empty result as "no safe interior boundary
    here" and stays fail-closed rather than normalising the corpus.
    """
    if not text:
        return frozenset()
    parts = decompose(text, eligibility_classifier=classifier)
    if parts.canonical_text != text:
        return frozenset()
    unsafe: set[int] = set()
    for span in parts.syllables:
        unsafe.update(range(span.canonical_start + 1, span.canonical_end))
    candidates = {unit.canonical_start for unit in parts.units}
    candidates.update((0, len(text)))
    return frozenset(candidates - unsafe)


_SEGMENT = re.compile(r"\s+|\S+")


def _segment_bounds(text: str) -> list[int]:
    """Boundary offsets between maximal whitespace and non-whitespace runs.

    Returns the *ends*, in order, so ``[0] + _segment_bounds(t)`` tiles `t`.
    Lossless by construction: the pattern alternates and covers every character.
    """
    return [match.end() for match in _SEGMENT.finditer(text)]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_document(
    document: CorpusDocument,
    partition: str,
    *,
    reference_length: LengthFn,
    base_length: LengthFn,
    max_length: int = MAX_LENGTH,
    classifier: Callable[[str], Any] | None = None,
) -> list[PreparedChunk]:
    """Cut one document into chunks that fit BOTH paths. Partition is inherited.

    Two-tier, so the common case stays cheap:

    * **fast path** -- extend greedily over whitespace-delimited segments while
      the accumulated text still fits. This is the original behaviour and covers
      essentially all documents.
    * **fallback** -- when a *single* non-whitespace segment cannot fit on its
      own, subdivide it at `safe_cut_offsets`. Only reached for the oversized
      units the real corpus actually contains.

    Every emitted chunk is length-checked, so correctness never depends on the
    token count growing monotonically with the text.
    """
    if partition not in {"train", "dev"}:
        raise ChunkingViolation(
            f"chunk_document requires the parent document's partition, got {partition!r}"
        )
    content = document.content
    if not content:
        raise ChunkingViolation(f"document {document.document_id!r} has no content to chunk")

    def fits(start: int, end: int) -> bool:
        piece = content[start:end]
        return reference_length(piece) <= max_length and base_length(piece) <= max_length

    def provenance(start: int, end: int) -> str:
        piece = content[start:end]
        return (
            f"document {document.document_id!r} (shard {document.source_shard}, source row "
            f"{document.source_row}), range [{start}, {end}) of {len(content)}, "
            f"{len(piece)} chars, reference={reference_length(piece)} "
            f"base={base_length(piece)}, max_length={max_length}"
        )

    bounds = _segment_bounds(content)
    ranges: list[tuple[int, int]] = []
    start = 0
    index = 0

    while start < len(content):
        index = bisect_right(bounds, start)
        best: int | None = None
        cursor = index
        while cursor < len(bounds):
            end = bounds[cursor]
            if end <= start:
                cursor += 1
                continue
            if fits(start, end):
                best = end
                cursor += 1
            else:
                break
        if best is None:
            # Not even the first segment fits on its own: subdivide it at
            # orthographically safe interior boundaries.
            hard_end = bounds[index] if index < len(bounds) else len(content)
            best = _subdivide(content, start, hard_end, fits, provenance, classifier)
        ranges.append((start, best))
        start = best

    if not ranges:
        raise ChunkingViolation(f"document {document.document_id!r} produced no chunks")

    chunks = [
        PreparedChunk(
            chunk_id=CHUNK_ID_TEMPLATE.format(
                document_id=document.document_id, chunk_index=position
            ),
            document_id=document.document_id,
            partition=partition,
            chunk_index=position,
            text=content[begin:finish],
            source_start=begin,
            source_end=finish,
            reference_length=reference_length(content[begin:finish]),
            base_length=base_length(content[begin:finish]),
            source_shard=document.source_shard,
        )
        for position, (begin, finish) in enumerate(ranges)
    ]
    verify_tiles_source(chunks, content, document.document_id)
    for chunk in chunks:
        if chunk.reference_length > max_length or chunk.base_length > max_length:
            raise ChunkingViolation(
                f"emitted chunk {chunk.chunk_id!r} exceeds max_length: "
                f"reference={chunk.reference_length} base={chunk.base_length} "
                f"max_length={max_length}"
            )
    return chunks


def _subdivide(
    content: str,
    start: int,
    hard_end: int,
    fits: Callable[[int, int], bool],
    provenance: Callable[[int, int], str],
    classifier: Callable[[str], Any] | None,
) -> int:
    """Largest safe end in `(start, hard_end]` that fits. Fail closed otherwise.

    Cuts come from `safe_cut_offsets`, so a Vietnamese candidate span is never
    bisected and a combining sequence is never split.
    """
    segment = content[start:hard_end]
    offsets = sorted(o for o in safe_cut_offsets(segment, classifier) if 0 < o <= len(segment))
    best: int | None = None
    for offset in offsets:
        end = start + offset
        if fits(start, end):
            best = end
        elif best is not None:
            break
    if best is None:
        raise ChunkingViolation(
            "indivisible orthographic region does not fit: "
            + provenance(start, hard_end)
            + ". No safe interior cut point produced a fitting chunk -- a cut would have "
            "split a Vietnamese candidate span or a combining sequence. Stage-1 does not "
            "truncate and does not drop text."
        )
    return best


def chunk_corpus(
    documents: Sequence[CorpusDocument],
    partition_of: dict[str, str],
    *,
    reference_length: LengthFn,
    base_length: LengthFn,
    max_length: int = MAX_LENGTH,
    classifier: Callable[[str], Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[PreparedChunk]:
    """Chunk every document, inheriting each one's already-decided partition.

    `partition_of` must already cover every document: the split happened first.
    """
    missing = [d.document_id for d in documents if d.document_id not in partition_of]
    if missing:
        raise ChunkingViolation(
            f"{len(missing)} document(s) have no partition, e.g. {missing[:5]}. The "
            "document-level split must run BEFORE chunking (D-S1B-002 steps 4-5)."
        )
    out: list[PreparedChunk] = []
    for position, document in enumerate(documents, start=1):
        out.extend(
            chunk_document(
                document,
                partition_of[document.document_id],
                reference_length=reference_length,
                base_length=base_length,
                max_length=max_length,
                classifier=classifier,
            )
        )
        if on_progress is not None:
            on_progress(position, len(out))
    return out


# ---------------------------------------------------------------------------
# Invariants, asserted rather than assumed
# ---------------------------------------------------------------------------
def verify_tiles_source(
    chunks: Sequence[PreparedChunk], content: str, document_id: str
) -> None:
    """Chunks must tile `[0, len(content))` exactly and reconstruct it byte-exact.

    This replaces the earlier `" ".join(...) == content` check, which silently
    assumed single-space separation and would have masked collapsed runs of
    whitespace, tabs and newlines -- and could not describe an internal
    non-whitespace cut at all.
    """
    cursor = 0
    for chunk in chunks:
        if chunk.source_start != cursor:
            raise ChunkingViolation(
                f"document {document_id!r}: chunk {chunk.chunk_id!r} starts at "
                f"{chunk.source_start}, expected {cursor} -- chunks must be contiguous "
                "with no gaps and no overlaps"
            )
        cursor = chunk.source_end
    if cursor != len(content):
        raise ChunkingViolation(
            f"document {document_id!r}: chunks cover {cursor} of {len(content)} characters; "
            "text would be lost"
        )
    rebuilt = "".join(chunk.text for chunk in chunks)
    if rebuilt != content:
        raise ChunkingViolation(
            f"document {document_id!r}: reconstruction differs from the source "
            f"({len(rebuilt)} vs {len(content)} chars). The chunker slices; it never "
            "normalises, collapses whitespace or rewrites."
        )


def verify_no_parent_spans_partitions(chunks: Iterable[PreparedChunk]) -> int:
    """Return the parent-document count, raising if any parent spans both sides.

    Structurally impossible given `chunk_corpus`, and asserted anyway -- this is
    the invariant the whole split-before-chunk ordering exists to guarantee.
    """
    seen: dict[str, str] = {}
    offenders: list[str] = []
    for chunk in chunks:
        previous = seen.setdefault(chunk.document_id, chunk.partition)
        if previous != chunk.partition:
            offenders.append(chunk.document_id)
    if offenders:
        raise ChunkingViolation(
            f"{len(set(offenders))} document(s) have chunks in BOTH partitions: "
            f"{sorted(set(offenders))[:5]}. Chunks must inherit their parent's partition."
        )
    return len(seen)
