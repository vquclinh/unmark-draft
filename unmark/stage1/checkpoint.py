"""Durable, cross-runtime Stage-6 checkpointing. **Torch-free, operational only.**

Stage 6 takes ~10.5 hours on the real corpus (Audit 029 §U). A Colab runtime
death must not send it back to document 0. This module adds that durability
**without touching a single scientific value**: the prepared payload, the
manifest's scientific fields, chunk boundaries, ids, ranges, lengths, ordering
and partitioning are all bit-identical to an uninterrupted run.

It also removes a second, independent blocker found while designing this: the
pre-3c writer accumulated **every** `PreparedChunk` in RAM and serialised once
at the end -- projected at ~30 GB for 1.1 M documents. Chunks are now streamed
to disk as they are produced, so peak memory is bounded by one shard.

Durability model
----------------
* **Append-only immutable shards.** Each shard covers one contiguous range of
  source-document indices. A committed shard is never rewritten, so checkpoint
  cost is O(new work), never O(progress).
* **Document boundaries only.** A commit means *every* document below
  `next_document_index` is completely processed. A document is never split
  across a commit.
* **Failure-atomic.** Payload: temp -> flush -> fsync -> close -> sha256 ->
  replace -> **re-verify size and digest from disk** -> only then update state.
  State: temp -> flush -> fsync -> replace. A death at any point leaves the
  previous committed checkpoint valid, and orphan temps are ignored and removed.
* **Google Drive is not trusted to be POSIX.** Every finalised artifact is
  re-read and re-hashed after `replace`; a mismatch fails closed.
* **Identity-bound.** Resume verifies repository HEAD, protocol/schema, corpus
  pin and every shard digest, the tokenizer pin, `max_length`, the split seed
  and dev count, the contamination criterion, and the ordered document-sequence
  and partition-assignment digests. Any difference **fails closed** rather than
  resuming against a different stream.

Caches are **performance-only**: a fresh runtime resumes cold, and correctness
depends solely on the source data, the locked protocol and the committed shards.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from unmark.stage1.contracts import Stage1ContractViolation

CHECKPOINT_SCHEMA_VERSION = "stage1-prepare-checkpoint-v1"
STATE_NAME = "state.json"
SHARD_DIRNAME = "shards"
COMPLETE_NAME = "COMPLETE.json"
TEMP_SUFFIX = ".tmp"

CHECKPOINT_INTERVAL_DOCUMENTS = 5_000
"""Commit every 5 000 completed documents.

**Operational, not scientific.** At the measured ~29.5 documents/s this bounds
lost work at roughly three minutes. It changes no output: the same shards, in
the same order, with the same contents, are produced for any interval.
"""

_EXTERNAL_SORT_BLOCK = 2_000_000
"""Lines held in memory while sorting membership keys. Bounds finalisation
memory to a few hundred MB regardless of corpus size."""


@dataclass
class Stage6Timings:
    """Where Stage-6 wall-clock actually goes. **Operational only.**

    Revision 3c was blamed for a 3.8x real slowdown that a local A/B could not
    reproduce (Audit 029 §Y). The reason the question was open at all is that
    the runner reported one number -- docs/s -- so "the chunker is slow" and
    "the writer is slow" looked identical from the outside. These counters make
    the next real run answer it directly instead of by inference.

    Every field is a count or a duration. **No corpus text, no UIT-VSFC text.**
    Accumulation is a float add per document (not per chunk), so the
    instrumentation cannot itself become the cost it is measuring.
    """

    stage6_total_seconds: float = 0.0
    chunk_compute_seconds: float = 0.0
    serialization_seconds: float = 0.0
    shard_buffer_write_seconds: float = 0.0
    membership_accumulator_seconds: float = 0.0
    membership_spill_seconds: float = 0.0
    checkpoint_commit_seconds: float = 0.0
    checkpoint_bytes: int = 0
    json_records_written: int = 0
    chunks_processed: int = 0
    documents_processed: int = 0
    membership_spills: int = 0
    collector_wait_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def report(self) -> str:
        """One-line-per-phase summary. Called once, at the end of Stage 6."""
        total = max(1e-9, self.stage6_total_seconds)
        rows = [
            ("chunk compute", self.chunk_compute_seconds),
            ("serialisation", self.serialization_seconds),
            ("shard buffer write", self.shard_buffer_write_seconds),
            ("membership accumulate", self.membership_accumulator_seconds),
            ("membership spill", self.membership_spill_seconds),
            ("checkpoint commit", self.checkpoint_commit_seconds),
            ("collector wait", self.collector_wait_seconds),
        ]
        lines = [
            f"    stage 6 total {self.stage6_total_seconds:.1f}s, "
            f"{self.documents_processed} documents, {self.chunks_processed} chunks, "
            f"{self.json_records_written} json records, "
            f"{self.checkpoint_bytes / 1e6:.1f} MB committed, "
            f"{self.membership_spills} membership spills"
        ]
        for label, value in rows:
            lines.append(f"      {label:<24s} {value:8.1f}s  {100 * value / total:5.1f}%")
        return "\n".join(lines)


class CheckpointViolation(Stage1ContractViolation):
    """Raised when checkpoint state is missing, inconsistent, or foreign."""


# ---------------------------------------------------------------------------
# Atomic, verified file writes
# ---------------------------------------------------------------------------
def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(block), b""):
            digest.update(piece)
    return digest.hexdigest()


def _fsync_dir(directory: Path) -> None:
    """Persist a rename. Best-effort: not every filesystem supports it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - Drive FUSE and friends
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> tuple[int, str]:
    """temp -> flush -> fsync -> replace -> **re-verify**. Returns (bytes, sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + TEMP_SUFFIX)
    with open(temp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)
    return verify_file(path, len(payload), hashlib.sha256(payload).hexdigest())


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> tuple[int, str]:
    """Re-read from disk and confirm size and digest. Fails closed.

    Drive's FUSE layer does not guarantee POSIX durability, so a finalised
    artifact is never trusted on the strength of a successful `write()`.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointViolation(f"expected file is missing after write: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise CheckpointViolation(
            f"{path.name}: {size} bytes on disk, expected {expected_bytes}"
        )
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise CheckpointViolation(
            f"{path.name}: sha256 {digest} on disk, expected {expected_sha256}"
        )
    return size, digest


def remove_orphan_temporaries(directory: Path) -> list[str]:
    """Delete `*.tmp` left by a death mid-write. They are never valid input."""
    removed = []
    if not directory.is_dir():
        return removed
    for candidate in sorted(directory.glob(f"*{TEMP_SUFFIX}")):
        candidate.unlink()
        removed.append(candidate.name)
    return removed


# ---------------------------------------------------------------------------
# Identity and state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointIdentity:
    """Everything a resume must match. Any difference fails closed.

    Carries **no raw corpus text and no UIT-VSFC text** -- digests, counts,
    pins and versions only.
    """

    repository_head: str | None
    protocol_version: str
    chunk_schema_version: str
    corpus_dataset: str
    corpus_revision: str
    corpus_files: tuple[tuple[str, int, str], ...]
    tokenizer_checkpoint: str
    tokenizer_revision: str
    transformers_version: str | None
    max_length: int
    raw_base_policy: str
    split_seed: int
    dev_documents: int
    contamination_method: str
    contamination_excluded_count: int
    document_sequence_digest: str
    partition_assignment_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_head": self.repository_head,
            "protocol_version": self.protocol_version,
            "chunk_schema_version": self.chunk_schema_version,
            "corpus_dataset": self.corpus_dataset,
            "corpus_revision": self.corpus_revision,
            "corpus_files": [list(f) for f in self.corpus_files],
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "tokenizer_revision": self.tokenizer_revision,
            "transformers_version": self.transformers_version,
            "max_length": self.max_length,
            "raw_base_policy": self.raw_base_policy,
            "split_seed": self.split_seed,
            "dev_documents": self.dev_documents,
            "contamination_method": self.contamination_method,
            "contamination_excluded_count": self.contamination_excluded_count,
            "document_sequence_digest": self.document_sequence_digest,
            "partition_assignment_digest": self.partition_assignment_digest,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointIdentity:
        return cls(
            repository_head=raw["repository_head"],
            protocol_version=raw["protocol_version"],
            chunk_schema_version=raw["chunk_schema_version"],
            corpus_dataset=raw["corpus_dataset"],
            corpus_revision=raw["corpus_revision"],
            corpus_files=tuple(tuple(f) for f in raw["corpus_files"]),
            tokenizer_checkpoint=raw["tokenizer_checkpoint"],
            tokenizer_revision=raw["tokenizer_revision"],
            transformers_version=raw["transformers_version"],
            max_length=raw["max_length"],
            raw_base_policy=raw["raw_base_policy"],
            split_seed=raw["split_seed"],
            dev_documents=raw["dev_documents"],
            contamination_method=raw["contamination_method"],
            contamination_excluded_count=raw["contamination_excluded_count"],
            document_sequence_digest=raw["document_sequence_digest"],
            partition_assignment_digest=raw["partition_assignment_digest"],
        )

    def require_match(self, other: "CheckpointIdentity") -> None:
        mine, theirs = self.to_dict(), other.to_dict()
        for key in mine:
            if mine[key] != theirs[key]:
                raise CheckpointViolation(
                    f"checkpoint identity mismatch on {key!r}: checkpoint has "
                    f"{theirs[key]!r}, this environment has {mine[key]!r}. Resuming "
                    "would prepare a different document stream."
                )


_SHA1_HEX = re.compile(r"\A[0-9a-f]{40}\Z")


def resolve_repository_head(root: Path | None = None) -> str:
    """The **actual** Git HEAD of the executing source tree. Fails closed.

    Checkpoint identity must record the commit that produced the shards, so
    this is derived from the repository rather than accepted from the caller.
    There is deliberately **no CLI flag and no environment override**: a
    caller-claimed HEAD would let a checkpoint written by commit A resume under
    commit B while asserting it did not.

    Returns the full 40-character SHA. Raises `CheckpointViolation` when the
    repository cannot answer, when `git` is unavailable, or when the result is
    not a full SHA -- never `"unknown"`, never a branch name, never a default.
    """
    import subprocess

    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CheckpointViolation(
            f"cannot resolve the repository HEAD under {root}: {error}. Stage-6 "
            "checkpoint identity binds the commit that produced the shards, so "
            "preparation refuses to run without it."
        ) from error
    if completed.returncode != 0:
        raise CheckpointViolation(
            f"cannot resolve the repository HEAD under {root}: git exited "
            f"{completed.returncode} ({completed.stderr.strip()[:200]}). Stage-6 "
            "checkpoint identity binds the commit that produced the shards."
        )
    head = completed.stdout.strip()
    if not _SHA1_HEX.match(head):
        raise CheckpointViolation(
            f"repository HEAD {head!r} is not a full 40-character commit sha. A "
            "branch name or abbreviated revision is not an identity."
        )
    return head


def document_sequence_digest(document_ids: Sequence[str]) -> str:
    """Order-SENSITIVE digest of the ordered source-document stream."""
    digest = hashlib.sha256()
    for document_id in document_ids:
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class ShardRecord:
    name: str
    bytes: int
    sha256: str
    first_document_index: int
    next_document_index: int
    chunks: int
    train_chunks: int
    dev_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShardRecord":
        return cls(**raw)


@dataclass
class CheckpointState:
    """The committed prefix. `next_document_index` is the resume point."""

    identity: CheckpointIdentity
    shards: list[ShardRecord] = field(default_factory=list)
    next_document_index: int = 0
    total_documents: int = 0
    chunks_total: int = 0
    train_chunks: int = 0
    dev_chunks: int = 0
    last_document_id: str | None = None
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "shards": [s.to_dict() for s in self.shards],
            "next_document_index": self.next_document_index,
            "total_documents": self.total_documents,
            "chunks_total": self.chunks_total,
            "train_chunks": self.train_chunks,
            "dev_chunks": self.dev_chunks,
            "last_document_id": self.last_document_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointState":
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointViolation(
                f"checkpoint schema {raw.get('schema_version')!r} != "
                f"{CHECKPOINT_SCHEMA_VERSION!r}"
            )
        return cls(
            identity=CheckpointIdentity.from_dict(raw["identity"]),
            shards=[ShardRecord.from_dict(s) for s in raw["shards"]],
            next_document_index=raw["next_document_index"],
            total_documents=raw["total_documents"],
            chunks_total=raw["chunks_total"],
            train_chunks=raw["train_chunks"],
            dev_chunks=raw["dev_chunks"],
            last_document_id=raw["last_document_id"],
        )


# ---------------------------------------------------------------------------
# Serialisation of one prepared chunk -- the scientific payload record
# ---------------------------------------------------------------------------
def chunk_record(chunk) -> dict[str, Any]:
    """The exact record the pre-3c writer emitted. **Unchanged.**"""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "partition": chunk.partition,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "source_start": chunk.source_start,
        "source_end": chunk.source_end,
        "source_shard": chunk.source_shard,
    }


def chunk_line(chunk) -> str:
    return json.dumps(chunk_record(chunk), ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------
class PrepareCheckpoint:
    """Streaming, resumable Stage-6 writer.

    Usage::

        cp = PrepareCheckpoint(checkpoint_dir, identity, total_documents)
        state = cp.begin()                      # START or RESUME
        for index in range(state.next_document_index, total):
            chunks = chunk_document(...)
            cp.add_document(index, document_id, chunks)
        cp.commit(force=True)
        cp.finalize(output_dir, ...)

    Shards accumulate on a **local staging directory** and only completed,
    verified shards are copied to the checkpoint directory, so a Drive-mounted
    destination never sees one write per chunk.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        identity: CheckpointIdentity,
        total_documents: int,
        *,
        interval: int = CHECKPOINT_INTERVAL_DOCUMENTS,
        staging_dir: Path | None = None,
    ) -> None:
        self.root = Path(checkpoint_dir)
        self.shard_dir = self.root / SHARD_DIRNAME
        self.identity = identity
        self.total_documents = total_documents
        self.interval = max(1, int(interval))
        self._staging = Path(staging_dir) if staging_dir else None
        self._owns_staging = staging_dir is None
        self.state = CheckpointState(identity=identity, total_documents=total_documents)
        self._buffer: list[str] = []
        self._buffer_documents = 0
        self._buffer_first_index: int | None = None
        self._pending_next_index: int | None = None
        self._pending_last_id: str | None = None
        self._pending_chunks = 0
        self._pending_train = 0
        self._pending_dev = 0
        self.checkpoint_seconds = 0.0
        self.checkpoint_bytes = 0
        self.commits = 0
        self.timings = Stage6Timings()

    # -- lifecycle ---------------------------------------------------------
    @property
    def staging(self) -> Path:
        if self._staging is None:
            self._staging = Path(tempfile.mkdtemp(prefix="unmark-stage6-"))
        self._staging.mkdir(parents=True, exist_ok=True)
        return self._staging

    def begin(self) -> CheckpointState:
        """START if there is no valid checkpoint, otherwise verified RESUME."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        remove_orphan_temporaries(self.root)
        remove_orphan_temporaries(self.shard_dir)
        state_path = self.root / STATE_NAME
        if not state_path.is_file():
            self.state = CheckpointState(
                identity=self.identity, total_documents=self.total_documents
            )
            return self.state

        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise CheckpointViolation(
                f"checkpoint state is malformed: {error}. Refusing to guess a resume "
                "point; delete the checkpoint directory to start over."
            ) from error
        state = CheckpointState.from_dict(raw)
        self.identity.require_match(state.identity)
        if state.total_documents != self.total_documents:
            raise CheckpointViolation(
                f"checkpoint covers {state.total_documents} documents, this run has "
                f"{self.total_documents}"
            )
        # Every committed shard must still be exactly what was committed.
        expected_index = 0
        for shard in state.shards:
            if shard.first_document_index != expected_index:
                raise CheckpointViolation(
                    f"shard {shard.name} starts at {shard.first_document_index}, "
                    f"expected {expected_index}: the committed prefix is not contiguous"
                )
            verify_file(self.shard_dir / shard.name, shard.bytes, shard.sha256)
            expected_index = shard.next_document_index
        if expected_index != state.next_document_index:
            raise CheckpointViolation(
                f"shards cover {expected_index} documents but state claims "
                f"{state.next_document_index}"
            )
        self.state = state
        return state

    # -- writing -----------------------------------------------------------
    def add_document(self, index: int, document_id: str, chunks: Iterable[Any]) -> int:
        """Buffer one document's chunks. Commits only at the configured interval."""
        if index != self._expected_index():
            raise CheckpointViolation(
                f"document index {index} out of order; expected "
                f"{self._expected_index()}. A checkpoint prefix must be contiguous."
            )
        if self._buffer_first_index is None:
            self._buffer_first_index = index
        written = 0
        # One clock read per document, not per chunk: the instrumentation must
        # not become the cost it exists to measure.
        started = time.monotonic()
        append = self._buffer.append
        for chunk in chunks:
            append(chunk_line(chunk))
            self._pending_chunks += 1
            if chunk.partition == "train":
                self._pending_train += 1
            else:
                self._pending_dev += 1
            written += 1
        self.timings.serialization_seconds += time.monotonic() - started
        self.timings.json_records_written += written
        self.timings.chunks_processed += written
        self.timings.documents_processed += 1
        self._buffer_documents += 1
        self._pending_next_index = index + 1
        self._pending_last_id = document_id
        if self._buffer_documents >= self.interval:
            self.commit()
        return written

    def _expected_index(self) -> int:
        return (
            self._pending_next_index
            if self._pending_next_index is not None
            else self.state.next_document_index
        )

    def commit(self, force: bool = False) -> ShardRecord | None:
        """Finalise the buffered shard and advance the committed prefix.

        Payload first, verified, **then** state -- so a death between them
        leaves the previous checkpoint valid and the new shard simply unused.

        This is the **only** place that touches durable storage: it runs once
        per `interval` documents, so no fsync, hash, copy or Drive write happens
        per chunk or per document. Asserted by test.
        """
        if self._buffer_first_index is None or self._pending_next_index is None:
            return None
        if not force and self._buffer_documents < self.interval:
            return None

        started = time.monotonic()
        name = f"shard-{len(self.state.shards):06d}.jsonl"
        payload = "".join(self._buffer).encode("utf-8")

        staged = self.staging / name
        with open(staged, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        verify_file(staged, len(payload), digest)

        final = self.shard_dir / name
        if staged.resolve() != final.resolve():
            temp = final.with_name(final.name + TEMP_SUFFIX)
            shutil.copyfile(staged, temp)
            with open(temp, "rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
            _fsync_dir(final.parent)
            staged.unlink(missing_ok=True)
        verify_file(final, len(payload), digest)

        record = ShardRecord(
            name=name, bytes=len(payload), sha256=digest,
            first_document_index=self._buffer_first_index,
            next_document_index=self._pending_next_index,
            chunks=self._pending_chunks,
            train_chunks=self._pending_train,
            dev_chunks=self._pending_dev,
        )
        self.state.shards.append(record)
        self.state.next_document_index = record.next_document_index
        self.state.chunks_total += record.chunks
        self.state.train_chunks += record.train_chunks
        self.state.dev_chunks += record.dev_chunks
        self.state.last_document_id = self._pending_last_id
        self._write_state()

        self.checkpoint_bytes += len(payload)
        elapsed = time.monotonic() - started
        self.checkpoint_seconds += elapsed
        self.timings.checkpoint_commit_seconds += elapsed
        self.timings.checkpoint_bytes += len(payload)
        self.commits += 1
        self._buffer.clear()
        self._buffer_documents = 0
        self._buffer_first_index = None
        self._pending_chunks = self._pending_train = self._pending_dev = 0
        return record

    def _write_state(self) -> None:
        payload = (
            json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        atomic_write_bytes(self.root / STATE_NAME, payload)

    # -- reading back ------------------------------------------------------
    def iter_committed_lines(self) -> Iterator[str]:
        """Every committed payload line, in document order."""
        for shard in self.state.shards:
            with open(self.shard_dir / shard.name, encoding="utf-8") as handle:
                for line in handle:
                    yield line

    def cleanup_staging(self) -> None:
        if self._owns_staging and self._staging and self._staging.is_dir():
            shutil.rmtree(self._staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Streaming finalisation -- bounded memory, identical output
# ---------------------------------------------------------------------------
@dataclass
class StreamedCounts:
    """Manifest counts derived by streaming, not by holding every chunk."""

    chunks_total: int = 0
    chunks_by_partition: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "dev": 0}
    )
    parent_documents_by_partition: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "dev": 0}
    )
    parent_documents_total: int = 0
    membership_digest: str = ""


def _external_sorted_digest(
    keys: Iterable[str], workdir: Path, timings: "Stage6Timings | None" = None
) -> str:
    """sha256 over **sorted** `chunk_id\\ttab\\tpartition` lines, bounded memory.

    `chunk_membership_digest` sorts, so the digest is order-independent -- but
    sorting 28 M keys in RAM is not affordable. Blocks are sorted, spilled, then
    merged, which yields exactly the same sorted sequence and therefore exactly
    the same digest.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    spills: list[Path] = []
    block: list[str] = []

    def spill(rows: list[str]) -> None:
        """One sorted block -> one file. Blocks are `_EXTERNAL_SORT_BLOCK` keys,
        so a 27.8 M-chunk corpus spills ~14 times in total, not per key."""
        started = time.monotonic()
        rows.sort()
        path = workdir / f"keys-{len(spills):06d}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows))
            if rows:
                handle.write("\n")
        spills.append(path)
        if timings is not None:
            timings.membership_spill_seconds += time.monotonic() - started
            timings.membership_spills += 1

    for key in keys:
        block.append(key)
        if len(block) >= _EXTERNAL_SORT_BLOCK:
            spill(block)
            block = []
    if block or not spills:
        spill(block)

    handles = [open(p, encoding="utf-8") for p in spills]
    try:
        digest = hashlib.sha256()
        first = True
        for line in heapq.merge(*( (l.rstrip("\n") for l in h) for h in handles )):
            if not first:
                digest.update(b"\n")
            digest.update(line.encode("utf-8"))
            first = False
        return digest.hexdigest()
    finally:
        for handle in handles:
            handle.close()
        for path in spills:
            path.unlink(missing_ok=True)


def stream_counts(
    lines: Iterable[str], workdir: Path, timings: "Stage6Timings | None" = None
) -> StreamedCounts:
    """Derive every manifest count from the payload stream.

    Equivalent to `build_manifest`'s in-memory accounting, including the
    no-parent-spans-both-partitions invariant, which is checked here with an
    explicit `seen` set rather than assumed from contiguity.
    """
    counts = StreamedCounts()
    seen_documents: dict[str, str] = {}
    keys: list[str] = []
    spill_dir = workdir / "membership"

    def key_stream() -> Iterator[str]:
        for line in lines:
            record = json.loads(line)
            partition = record["partition"]
            document_id = record["document_id"]
            counts.chunks_total += 1
            counts.chunks_by_partition[partition] += 1
            previous = seen_documents.get(document_id)
            if previous is None:
                seen_documents[document_id] = partition
                counts.parent_documents_by_partition[partition] += 1
            elif previous != partition:
                raise CheckpointViolation(
                    f"document {document_id!r} has chunks in both partitions "
                    f"({previous} and {partition}); chunks must inherit their "
                    "parent's partition"
                )
            yield f"{record['chunk_id']}\t{partition}"

    started = time.monotonic()
    counts.membership_digest = _external_sorted_digest(key_stream(), spill_dir, timings)
    if timings is not None:
        # Accumulation net of the spill/merge time counted inside the sorter.
        timings.membership_accumulator_seconds += (
            time.monotonic() - started - timings.membership_spill_seconds
        )
    counts.parent_documents_total = len(seen_documents)
    del seen_documents, keys
    return counts


def concatenate_shards(
    shard_paths: Sequence[Path], destination: Path, block: int = 1 << 20
) -> tuple[int, str]:
    """Build the payload by streaming shards in order. temp -> fsync -> replace."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + TEMP_SUFFIX)
    digest = hashlib.sha256()
    total = 0
    with open(temp, "wb") as out:
        for path in shard_paths:
            with open(path, "rb") as handle:
                for piece in iter(lambda: handle.read(block), b""):
                    out.write(piece)
                    digest.update(piece)
                    total += len(piece)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, destination)
    _fsync_dir(destination.parent)
    return verify_file(destination, total, digest.hexdigest())


def write_completion_marker(
    checkpoint_dir: Path,
    *,
    identity: CheckpointIdentity,
    artifacts: dict[str, tuple[int, str]],
    counts: dict[str, Any],
) -> Path:
    """The LAST write. Binds final artifact hashes and the run identity.

    A directory existing proves nothing; only this marker -- written after every
    scientific artifact is on disk and verified -- means ALREADY_COMPLETE.
    """
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "artifacts": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in sorted(artifacts.items())
        },
        "counts": counts,
        "complete": True,
    }
    body = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path = Path(checkpoint_dir) / COMPLETE_NAME
    atomic_write_bytes(path, body)
    return path


def read_completion(
    checkpoint_dir: Path, output_dir: Path, identity: CheckpointIdentity
) -> dict[str, Any] | None:
    """ALREADY_COMPLETE only if the marker validates against artifacts on disk."""
    path = Path(checkpoint_dir) / COMPLETE_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("complete") is not True:
        return None
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointViolation(
            f"completion marker schema {payload.get('schema_version')!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION!r}"
        )
    identity.require_match(CheckpointIdentity.from_dict(payload["identity"]))
    for name, meta in payload["artifacts"].items():
        verify_file(Path(output_dir) / name, meta["bytes"], meta["sha256"])
    return payload
