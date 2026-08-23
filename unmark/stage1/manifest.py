"""The prepared-corpus manifest. **Torch-free.**

Training consumes a manifest-bound prepared corpus, never a loose directory of
text: the manifest is what lets the trainer refuse a corpus that was built under
a different protocol, and it is where every claim about the corpus is recorded
as a checkable value rather than a promise.

**It carries no raw text.** Ids, digests, counts and provenance only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from unmark.stage1.chunking import PreparedChunk, verify_no_parent_spans_partitions
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    CHUNK_SCHEMA_VERSION,
    CONTAMINATION_METHOD,
    DEV_DOCUMENTS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    ON_OVERFLOW,
    STAGE1_PROTOCOL_VERSION,
    SPLIT_SEED,
)

MANIFEST_SCHEMA_VERSION = "stage1-prepared-corpus-v1"
MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.jsonl"


class ManifestViolation(Stage1ContractViolation):
    """Raised when a prepared corpus is incompatible with the locked protocol."""


def chunk_membership_digest(chunks: Sequence[PreparedChunk]) -> str:
    payload = "\n".join(sorted(f"{c.chunk_id}\t{c.partition}" for c in chunks))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedCorpusManifest:
    """Everything training must verify before it trusts a prepared corpus."""

    source: dict[str, Any]
    contamination: dict[str, Any]
    partition: dict[str, Any]
    chunking: dict[str, Any]
    counts: dict[str, Any]
    schema_version: str = MANIFEST_SCHEMA_VERSION
    protocol_version: str = STAGE1_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "source": self.source,
            "contamination": self.contamination,
            "partition": self.partition,
            "chunking": self.chunking,
            "counts": self.counts,
            "official_test_used": False,
            "raw_text_in_manifest": False,
        }

    def write(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_manifest(
    *,
    source: dict[str, Any],
    contamination: dict[str, Any],
    partition: dict[str, Any],
    chunks: Sequence[PreparedChunk],
    overflow_count: int,
    base_invariance_violations: int,
) -> PreparedCorpusManifest:
    """Assemble the manifest, re-deriving every count from the chunks themselves."""
    parent_count = verify_no_parent_spans_partitions(chunks)
    by_partition: dict[str, int] = {"train": 0, "dev": 0}
    parents: dict[str, set[str]] = {"train": set(), "dev": set()}
    for chunk in chunks:
        by_partition[chunk.partition] += 1
        parents[chunk.partition].add(chunk.document_id)
    if overflow_count:
        raise ManifestViolation(
            f"{overflow_count} chunk(s) overflowed max_length={MAX_LENGTH}. After correct "
            "pre-chunking this must be zero; on_overflow=FAIL is a guard, not a policy."
        )
    if base_invariance_violations:
        raise ManifestViolation(
            f"{base_invariance_violations} base-invariance violation(s); b(C(x)) != b(x) "
            "means corruption or decomposition changed underneath Stage-1"
        )
    return PreparedCorpusManifest(
        source=source,
        contamination=contamination,
        partition=partition,
        chunking={
            "algorithm": "deterministic_whitespace_boundary",
            "schema_version": CHUNK_SCHEMA_VERSION,
            "max_length": MAX_LENGTH,
            "on_overflow": ON_OVERFLOW,
            "truncation": False,
            "split_before_chunk": True,
            "chunks_inherit_parent_partition": True,
            "tokenizer_checkpoint": ENCODER_CHECKPOINT,
            "tokenizer_revision": ENCODER_REVISION,
        },
        counts={
            "chunks_total": len(chunks),
            "chunks_by_partition": dict(by_partition),
            "parent_documents_total": parent_count,
            "parent_documents_by_partition": {k: len(v) for k, v in parents.items()},
            "parents_spanning_both_partitions": 0,
            "overflow_count": 0,
            "base_invariance_violations": 0,
            "chunk_membership_digest": chunk_membership_digest(chunks),
        },
    )


def build_manifest_from_counts(
    *,
    source: dict[str, Any],
    contamination: dict[str, Any],
    partition: dict[str, Any],
    chunks_total: int,
    chunks_by_partition: dict[str, int],
    parent_documents_total: int,
    parent_documents_by_partition: dict[str, int],
    chunk_membership_digest: str,
    overflow_count: int,
    base_invariance_violations: int,
) -> PreparedCorpusManifest:
    """`build_manifest` for a **streamed** payload. Identical scientific fields.

    The pre-3c writer held every `PreparedChunk` in RAM to derive these counts --
    projected at ~30 GB for the real corpus. The counts are now accumulated while
    the payload streams past, and this function assembles exactly the same
    manifest from them. `build_manifest` is retained unchanged for callers that
    already have the chunks in hand, and the two are asserted equal in tests.
    """
    if overflow_count:
        raise ManifestViolation(
            f"{overflow_count} chunk(s) overflowed max_length={MAX_LENGTH}. After correct "
            "pre-chunking this must be zero; on_overflow=FAIL is a guard, not a policy."
        )
    if base_invariance_violations:
        raise ManifestViolation(
            f"{base_invariance_violations} base-invariance violation(s); b(C(x)) != b(x) "
            "means corruption or decomposition changed underneath Stage-1"
        )
    return PreparedCorpusManifest(
        source=source,
        contamination=contamination,
        partition=partition,
        chunking={
            "algorithm": "deterministic_whitespace_boundary",
            "schema_version": CHUNK_SCHEMA_VERSION,
            "max_length": MAX_LENGTH,
            "on_overflow": ON_OVERFLOW,
            "truncation": False,
            "split_before_chunk": True,
            "chunks_inherit_parent_partition": True,
            "tokenizer_checkpoint": ENCODER_CHECKPOINT,
            "tokenizer_revision": ENCODER_REVISION,
        },
        counts={
            "chunks_total": chunks_total,
            "chunks_by_partition": dict(chunks_by_partition),
            "parent_documents_total": parent_documents_total,
            "parent_documents_by_partition": dict(parent_documents_by_partition),
            "parents_spanning_both_partitions": 0,
            "overflow_count": 0,
            "base_invariance_violations": 0,
            "chunk_membership_digest": chunk_membership_digest,
        },
    )


def load_manifest(directory: Path) -> dict[str, Any]:
    """Read a prepared-corpus manifest and refuse anything off-protocol."""
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise ManifestViolation(f"prepared-corpus manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestViolation(f"manifest is malformed: {error}") from error
    require_compatible(manifest)
    return manifest


def require_compatible(manifest: dict[str, Any]) -> None:
    """Fail closed unless the prepared corpus matches the locked Stage-1 protocol."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestViolation(
            f"manifest schema {manifest.get('schema_version')!r} != {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("protocol_version") != STAGE1_PROTOCOL_VERSION:
        raise ManifestViolation(
            f"manifest protocol {manifest.get('protocol_version')!r} != "
            f"{STAGE1_PROTOCOL_VERSION!r}; artifacts from different protocols are not "
            "comparable and must not be pooled"
        )
    if manifest.get("official_test_used") is not False:
        raise ManifestViolation("manifest does not record official_test_used=false")

    source = manifest.get("source") or {}
    if source.get("revision") != __import__(
        "unmark.stage1.protocol", fromlist=["CORPUS_REVISION"]
    ).CORPUS_REVISION:
        raise ManifestViolation(
            f"manifest corpus revision {source.get('revision')!r} is not the pinned revision"
        )

    chunking = manifest.get("chunking") or {}
    if chunking.get("max_length") != MAX_LENGTH:
        raise ManifestViolation(f"manifest max_length {chunking.get('max_length')} != {MAX_LENGTH}")
    if chunking.get("split_before_chunk") is not True:
        raise ManifestViolation("manifest does not record split_before_chunk=true")
    if chunking.get("chunks_inherit_parent_partition") is not True:
        raise ManifestViolation("manifest does not record chunks_inherit_parent_partition=true")
    if chunking.get("truncation") is not False:
        raise ManifestViolation("manifest records truncation; Stage-1 does not truncate")
    if chunking.get("tokenizer_revision") != ENCODER_REVISION:
        raise ManifestViolation(
            f"manifest tokenizer revision {chunking.get('tokenizer_revision')!r} is not the "
            f"pinned {ENCODER_REVISION!r}"
        )

    partition = manifest.get("partition") or {}
    if partition.get("seed") != SPLIT_SEED:
        raise ManifestViolation(
            f"manifest split seed {partition.get('seed')!r} != locked {SPLIT_SEED}"
        )
    if partition.get("dev_documents") != DEV_DOCUMENTS:
        raise ManifestViolation(
            f"manifest dev document count {partition.get('dev_documents')!r} != "
            f"locked {DEV_DOCUMENTS}"
        )

    counts = manifest.get("counts") or {}
    if counts.get("parents_spanning_both_partitions") != 0:
        raise ManifestViolation("manifest does not record zero parents spanning both partitions")
    if counts.get("overflow_count") != 0:
        raise ManifestViolation("manifest does not record zero overflow")

    contamination = manifest.get("contamination") or {}
    if contamination.get("method") != CONTAMINATION_METHOD:
        raise ManifestViolation(
            f"contamination method {contamination.get('method')!r} != {CONTAMINATION_METHOD!r}; "
            "fuzzy or semantic screening is not part of this pipeline"
        )
    if contamination.get("official_test_screened") is not False:
        raise ManifestViolation(
            "manifest must record official_test_screened=false -- TEST is SEALED and "
            "screening against it would require opening it"
        )
