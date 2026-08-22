"""Deterministic, tokenizer-aware pre-chunking. **Torch-free.**

Implements the Audit-028 chunking contract (D-S1B-002). Seven requirements, each
enforced rather than documented:

1. **Preserve text order** -- chunks are contiguous and emitted in source order.
2. **No extra normalization** -- the chunker never calls `canon()`, never
   repairs, never restores. It only *cuts*, at boundaries that already exist.
3. **Stable ids** -- ``{document_id}#{chunk_index}``.
4. **Fits `max_length` on BOTH tokenizer paths** -- the reference path
   (``canon(x)``) and the base path (``b(canon(x))``) have different lengths and
   separate padding domains, so both are measured.
5. **Never split a syllable span** -- cuts land on whitespace boundaries, which
   never fall inside a syllable.
6. **Runs only after the document-level partition exists.**
7. **Every chunk inherits its parent document's partition.**

Requirements 6 and 7 are structural: `chunk_document` *takes* a partition and
copies it onto every chunk. There is no code path that assigns a partition to a
chunk, so chunks of one article cannot land on opposite sides of the split.

**No truncation, ever.** A span that cannot fit raises `ChunkingViolation` with
enough provenance to diagnose it -- text is never silently dropped.

The length function is injected, so the core logic is testable with a
lightweight mock and the real pinned tokenizer is not needed to prove the
contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from unmark.stage1.corpus import CorpusDocument, CorpusContractViolation
from unmark.stage1.protocol import CHUNK_ID_TEMPLATE, CHUNK_SCHEMA_VERSION, MAX_LENGTH


class ChunkingViolation(CorpusContractViolation):
    """Raised when text cannot be chunked within the locked contract."""


LengthFn = Callable[[str], int]
"""`text -> token count INCLUDING the model's special tokens`. Injected so the
contract is testable without the real tokenizer."""


@dataclass(frozen=True)
class PreparedChunk:
    """One Stage-1 training example, bound to its parent and its partition."""

    chunk_id: str
    document_id: str
    partition: str
    chunk_index: int
    text: str
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


_WHITESPACE = re.compile(r"(\s+)")


def _segments(text: str) -> list[str]:
    """Split into alternating content/whitespace pieces, losslessly.

    ``"".join(_segments(t)) == t`` for every input, which is what lets the
    chunker guarantee that no interior content is lost. Cuts are only ever made
    *between* segments, and a whitespace segment never falls inside a syllable.
    """
    return [s for s in _WHITESPACE.split(text) if s != ""]


def chunk_document(
    document: CorpusDocument,
    partition: str,
    *,
    reference_length: LengthFn,
    base_length: LengthFn,
    max_length: int = MAX_LENGTH,
) -> list[PreparedChunk]:
    """Cut one document into chunks that fit BOTH paths. Partition is inherited.

    `partition` is an argument, never derived here -- requirement 6/7. The
    chunker cannot see the split seed and cannot assign a side.
    """
    if partition not in {"train", "dev"}:
        raise ChunkingViolation(
            f"chunk_document requires the parent document's partition, got {partition!r}"
        )
    segments = _segments(document.content)
    if not segments:
        raise ChunkingViolation(f"document {document.document_id!r} has no content to chunk")

    def fits(text: str) -> bool:
        return reference_length(text) <= max_length and base_length(text) <= max_length

    chunks: list[PreparedChunk] = []
    current: list[str] = []
    index = 0

    def emit(pieces: list[str]) -> None:
        nonlocal index
        text = "".join(pieces).strip()
        if not text:
            return
        chunks.append(
            PreparedChunk(
                chunk_id=CHUNK_ID_TEMPLATE.format(
                    document_id=document.document_id, chunk_index=index
                ),
                document_id=document.document_id,
                partition=partition,
                chunk_index=index,
                text=text,
                reference_length=reference_length(text),
                base_length=base_length(text),
                source_shard=document.source_shard,
            )
        )
        index += 1

    def require_fits_alone(segment: str) -> None:
        """A segment that cannot fit on its own is indivisible.

        Whitespace is the only boundary that never splits a syllable, so there
        is nowhere legal left to cut. Fail closed with full provenance -- never
        truncate, never drop, and never emit an oversized chunk.
        """
        text = segment.strip()
        if not text or fits(text):
            return
        raise ChunkingViolation(
            f"indivisible span does not fit max_length={max_length} in document "
            f"{document.document_id!r} (shard {document.source_shard}, source row "
            f"{document.source_row}), segment {text[:60]!r}… "
            f"reference={reference_length(text)} base={base_length(text)}. "
            "Stage-1 does not truncate and does not drop text."
        )

    for segment in segments:
        if not current:
            require_fits_alone(segment)
            current = [segment]
            continue
        candidate = current + [segment]
        if fits("".join(candidate).strip()):
            current = candidate
            continue
        emit(current)
        # the segment that could not be appended now STARTS a chunk, so it must
        # fit on its own -- checking only the append would let an oversized
        # segment become its own oversized chunk.
        require_fits_alone(segment)
        current = [segment]
    emit(current)

    if not chunks:
        raise ChunkingViolation(f"document {document.document_id!r} produced no chunks")
    return chunks


def chunk_corpus(
    documents: Sequence[CorpusDocument],
    partition_of: dict[str, str],
    *,
    reference_length: LengthFn,
    base_length: LengthFn,
    max_length: int = MAX_LENGTH,
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
    for document in documents:
        out.extend(
            chunk_document(
                document,
                partition_of[document.document_id],
                reference_length=reference_length,
                base_length=base_length,
                max_length=max_length,
            )
        )
    return out


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
