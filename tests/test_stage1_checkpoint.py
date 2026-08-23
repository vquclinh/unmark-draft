"""Cross-runtime Stage-6 checkpointing: durability, resume, identical output.

ML-free. Stage 6 takes ~10.5 h on the real corpus, so a Colab runtime death
must not restart it at document 0 (Audit 029 §U). These tests prove that a run
interrupted at any point and resumed produces **byte-identical** scientific
artifacts to an uninterrupted run.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from unmark.stage1.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    COMPLETE_NAME,
    STATE_NAME,
    CheckpointIdentity,
    CheckpointViolation,
    PrepareCheckpoint,
    atomic_write_bytes,
    chunk_line,
    concatenate_shards,
    document_sequence_digest,
    read_completion,
    remove_orphan_temporaries,
    sha256_file,
    stream_counts,
    verify_file,
    write_completion_marker,
)
from unmark.stage1.chunking import chunk_document
from unmark.stage1.corpus import CorpusDocument, partition_documents
from unmark.stage1.manifest import (
    CHUNKS_NAME,
    MANIFEST_NAME,
    build_manifest,
    build_manifest_from_counts,
    chunk_membership_digest,
)
from unmark.stage1.protocol import (
    CHUNK_SCHEMA_VERSION,
    DEV_DOCUMENTS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    RAW_BASE_POLICY,
    SPLIT_SEED,
    STAGE1_PROTOCOL_VERSION,
)

WORDS = "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương".split()


def make_documents(n=60, seed=51733):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 160)))
        if i % 7 == 0:
            body += "\n\n" + " ".join(rng.choice(WORDS) for _ in range(30))
        out.append(CorpusDocument(f"doc-{i:04d}", body, "train.parquet", i))
    return out


def ref_len(text):
    return len(text.split()) + 2


def base_len(text):
    return len(text.split()) + 4


def identity_for(documents, partition, **overrides):
    fields = dict(
        repository_head="4c72639",
        protocol_version=STAGE1_PROTOCOL_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        corpus_dataset="undertheseanlp/UVW-2026",
        corpus_revision="a0a79294e4568137e25828bb3f2a4cde8546e1fb",
        corpus_files=(("train.parquet", 608316204, "5243" + "0" * 60),),
        tokenizer_checkpoint=ENCODER_CHECKPOINT,
        tokenizer_revision=ENCODER_REVISION,
        transformers_version="4.57.6",
        max_length=MAX_LENGTH,
        raw_base_policy=RAW_BASE_POLICY,
        split_seed=SPLIT_SEED,
        dev_documents=DEV_DOCUMENTS,
        contamination_method="exact_canonical_duplicate",
        contamination_excluded_count=0,
        document_sequence_digest=document_sequence_digest(
            [d.document_id for d in documents]
        ),
        partition_assignment_digest=partition.membership_digest,
    )
    fields.update(overrides)
    return CheckpointIdentity(**fields)


def chunk_all(documents, partition, max_length=40):
    out = []
    for document in documents:
        out.extend(chunk_document(
            document, partition.assignment[document.document_id],
            reference_length=ref_len, base_length=base_len, max_length=max_length,
        ))
    return out


@pytest.fixture(scope="module")
def fixture():
    documents = make_documents()
    partition = partition_documents(
        [d.document_id for d in documents], dev_documents=12
    )
    chunks = chunk_all(documents, partition)
    return documents, partition, chunks


# ---------------------------------------------------------------------------
# The uninterrupted oracle
# ---------------------------------------------------------------------------
def run_preparation(tmp_path, documents, partition, *, interval=7, stop_after=None,
                    checkpoint_dir=None, max_length=40):
    """Drive the writer the way the runner does. `stop_after` simulates death."""
    checkpoint_dir = checkpoint_dir or tmp_path / "cp"
    identity = identity_for(documents, partition)
    checkpoint = PrepareCheckpoint(
        checkpoint_dir, identity, len(documents), interval=interval,
        staging_dir=tmp_path / "staging",
    )
    state = checkpoint.begin()
    for index in range(state.next_document_index, len(documents)):
        if stop_after is not None and index >= stop_after:
            return checkpoint, "interrupted"
        document = documents[index]
        chunks = chunk_document(
            document, partition.assignment[document.document_id],
            reference_length=ref_len, base_length=base_len, max_length=max_length,
        )
        checkpoint.add_document(index, document.document_id, chunks)
    checkpoint.commit(force=True)
    return checkpoint, "finished"


def finalize(checkpoint, output, source, contamination, partition):
    output.mkdir(parents=True, exist_ok=True)
    paths = [checkpoint.shard_dir / s.name for s in checkpoint.state.shards]
    payload = concatenate_shards(paths, output / CHUNKS_NAME)
    with open(output / CHUNKS_NAME, encoding="utf-8") as handle:
        counts = stream_counts(handle, checkpoint.staging / "fin")
    manifest = build_manifest_from_counts(
        source=source, contamination=contamination, partition=partition,
        chunks_total=counts.chunks_total,
        chunks_by_partition=counts.chunks_by_partition,
        parent_documents_total=counts.parent_documents_total,
        parent_documents_by_partition=counts.parent_documents_by_partition,
        chunk_membership_digest=counts.membership_digest,
        overflow_count=0, base_invariance_violations=0,
    )
    body = (json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n").encode("utf-8")
    manifest_meta = atomic_write_bytes(output / MANIFEST_NAME, body)
    write_completion_marker(
        checkpoint.root, identity=checkpoint.identity,
        artifacts={CHUNKS_NAME: payload, MANIFEST_NAME: manifest_meta},
        counts=manifest.to_dict()["counts"],
    )
    return counts


SOURCE = {"revision": "a0a79294e4568137e25828bb3f2a4cde8546e1fb"}
CONTAM = {"method": "exact_canonical_duplicate", "official_test_screened": False}


def prepared_bytes(output):
    return ((output / CHUNKS_NAME).read_bytes(), (output / MANIFEST_NAME).read_bytes())


# ---------------------------------------------------------------------------
# Streaming equals the in-memory writer
# ---------------------------------------------------------------------------
def test_streamed_payload_equals_the_in_memory_writer(tmp_path, fixture):
    documents, partition, chunks = fixture
    checkpoint, status = run_preparation(tmp_path, documents, partition)
    assert status == "finished"
    finalize(checkpoint, tmp_path / "out", SOURCE, CONTAM, partition.to_dict())

    expected = "".join(chunk_line(c) for c in chunks).encode("utf-8")
    assert (tmp_path / "out" / CHUNKS_NAME).read_bytes() == expected


def test_streamed_manifest_equals_the_in_memory_manifest(tmp_path, fixture):
    documents, partition, chunks = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    counts = finalize(checkpoint, tmp_path / "out", SOURCE, CONTAM, partition.to_dict())

    in_memory = build_manifest(
        source=SOURCE, contamination=CONTAM, partition=partition.to_dict(),
        chunks=chunks, overflow_count=0, base_invariance_violations=0,
    ).to_dict()
    on_disk = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == in_memory
    assert counts.membership_digest == chunk_membership_digest(chunks)


# ---------------------------------------------------------------------------
# TASK G — interruption at every point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop_after", [0, 3, 7, 8, 14, 20, 35, 59])
def test_interrupt_and_resume_matches_the_uninterrupted_oracle(tmp_path, fixture, stop_after):
    documents, partition, chunks = fixture

    oracle_dir = tmp_path / "oracle"
    checkpoint, _ = run_preparation(oracle_dir, documents, partition)
    finalize(checkpoint, oracle_dir / "out", SOURCE, CONTAM, partition.to_dict())
    expected = prepared_bytes(oracle_dir / "out")

    work = tmp_path / "work"
    run_preparation(work, documents, partition, stop_after=stop_after)   # dies
    resumed, status = run_preparation(work, documents, partition)        # fresh runtime
    assert status == "finished"
    finalize(resumed, work / "out", SOURCE, CONTAM, partition.to_dict())

    assert prepared_bytes(work / "out") == expected, f"stop_after={stop_after}"


def test_resume_starts_exactly_at_the_committed_prefix(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition, stop_after=20)
    committed = checkpoint.state.next_document_index
    assert committed % 7 == 0, "a commit must land on the interval boundary"
    assert committed <= 20, "never commits a document that was not completed"

    resumed = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition), len(documents),
        interval=7, staging_dir=tmp_path / "staging2",
    )
    state = resumed.begin()
    assert state.next_document_index == committed
    assert state.last_document_id == documents[committed - 1].document_id


def test_no_document_is_duplicated_or_missing_after_resume(tmp_path, fixture):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=23)
    resumed, _ = run_preparation(tmp_path, documents, partition)
    finalize(resumed, tmp_path / "out", SOURCE, CONTAM, partition.to_dict())

    seen_documents, seen_chunks = [], set()
    with open(tmp_path / "out" / CHUNKS_NAME, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            assert record["chunk_id"] not in seen_chunks, "duplicate chunk id"
            seen_chunks.add(record["chunk_id"])
            if not seen_documents or seen_documents[-1] != record["document_id"]:
                seen_documents.append(record["document_id"])
    assert seen_documents == [d.document_id for d in documents], "order or coverage changed"
    assert len(seen_documents) == len(set(seen_documents)), "a document appeared twice"


def test_a_death_midway_through_an_uncommitted_shard_loses_only_that_shard(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition, stop_after=10)
    # 10 documents processed, interval 7 -> exactly one commit, 3 documents lost
    assert checkpoint.state.next_document_index == 7
    assert len(checkpoint.state.shards) == 1


def test_an_orphan_temp_is_ignored_and_removed(tmp_path, fixture):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=21)
    orphan = tmp_path / "cp" / "shards" / "shard-000003.jsonl.tmp"
    orphan.write_text("garbage that must never be accepted\n", encoding="utf-8")
    stale_state = tmp_path / "cp" / f"{STATE_NAME}.tmp"
    stale_state.write_text("{}", encoding="utf-8")

    resumed, _ = run_preparation(tmp_path, documents, partition)
    assert not orphan.exists() and not stale_state.exists()
    finalize(resumed, tmp_path / "out", SOURCE, CONTAM, partition.to_dict())
    assert "garbage" not in (tmp_path / "out" / CHUNKS_NAME).read_text(encoding="utf-8")


def test_state_written_after_payload_so_a_gap_is_safe(tmp_path, fixture):
    """A shard on disk that state.json does not list is simply unused."""
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=14)
    state = json.loads((tmp_path / "cp" / STATE_NAME).read_text(encoding="utf-8"))
    listed = {s["name"] for s in state["shards"]}
    (tmp_path / "cp" / "shards" / "shard-000099.jsonl").write_text("{}\n", encoding="utf-8")

    resumed, _ = run_preparation(tmp_path, documents, partition)
    used = {s.name for s in resumed.state.shards}
    assert "shard-000099.jsonl" not in used
    assert listed <= used


# ---------------------------------------------------------------------------
# Fail-closed identity and integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, value", [
    ("repository_head", "deadbeef"),
    ("protocol_version", "other"),
    ("chunk_schema_version", "other"),
    ("corpus_revision", "b" * 40),
    ("tokenizer_revision", "c" * 40),
    ("transformers_version", "4.0.0"),
    ("max_length", 512),
    ("raw_base_policy", "SEGMENTED"),
    ("split_seed", 1),
    ("dev_documents", 10),
    ("contamination_method", "fuzzy"),
    ("document_sequence_digest", "d" * 64),
    ("partition_assignment_digest", "e" * 64),
])
def test_resume_fails_closed_on_any_identity_change(tmp_path, fixture, field, value):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=21)
    foreign = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition, **{field: value}),
        len(documents), interval=7, staging_dir=tmp_path / "s2",
    )
    with pytest.raises(CheckpointViolation, match="identity mismatch"):
        foreign.begin()


def test_resume_fails_closed_on_a_tampered_shard(tmp_path, fixture):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=21)
    shard = sorted((tmp_path / "cp" / "shards").glob("shard-*.jsonl"))[0]
    shard.write_bytes(shard.read_bytes() + b'{"tampered": true}\n')

    resumed = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition), len(documents),
        interval=7, staging_dir=tmp_path / "s3",
    )
    with pytest.raises(CheckpointViolation, match="bytes on disk|sha256"):
        resumed.begin()


def test_resume_fails_closed_on_a_different_document_count(tmp_path, fixture):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=21)
    shorter = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition), len(documents) - 1,
        interval=7, staging_dir=tmp_path / "s4",
    )
    with pytest.raises(CheckpointViolation, match="documents"):
        shorter.begin()


def test_out_of_order_documents_are_refused(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition), len(documents),
        interval=7, staging_dir=tmp_path / "s5",
    )
    checkpoint.begin()
    checkpoint.add_document(0, documents[0].document_id, [])
    with pytest.raises(CheckpointViolation, match="out of order"):
        checkpoint.add_document(5, documents[5].document_id, [])


def test_malformed_state_is_refused_rather_than_guessed(tmp_path, fixture):
    documents, partition, _ = fixture
    (tmp_path / "cp").mkdir(parents=True)
    (tmp_path / "cp" / STATE_NAME).write_text("{not json", encoding="utf-8")
    checkpoint = PrepareCheckpoint(
        tmp_path / "cp", identity_for(documents, partition), len(documents),
        interval=7, staging_dir=tmp_path / "s6",
    )
    with pytest.raises(CheckpointViolation, match="malformed"):
        checkpoint.begin()


# ---------------------------------------------------------------------------
# Completion marker
# ---------------------------------------------------------------------------
def test_a_directory_alone_is_never_ALREADY_COMPLETE(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    output = tmp_path / "out"
    output.mkdir()
    (output / CHUNKS_NAME).write_text("", encoding="utf-8")
    assert read_completion(checkpoint.root, output, checkpoint.identity) is None


def test_completion_is_recognised_only_after_the_marker(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    output = tmp_path / "out"
    assert read_completion(checkpoint.root, output, checkpoint.identity) is None
    finalize(checkpoint, output, SOURCE, CONTAM, partition.to_dict())
    payload = read_completion(checkpoint.root, output, checkpoint.identity)
    assert payload is not None and payload["complete"] is True
    assert set(payload["artifacts"]) == {CHUNKS_NAME, MANIFEST_NAME}


def test_completion_fails_closed_if_an_artifact_changed(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    output = tmp_path / "out"
    finalize(checkpoint, output, SOURCE, CONTAM, partition.to_dict())
    (output / CHUNKS_NAME).write_bytes((output / CHUNKS_NAME).read_bytes() + b"x")
    with pytest.raises(CheckpointViolation, match="bytes on disk|sha256"):
        read_completion(checkpoint.root, output, checkpoint.identity)


def test_completion_fails_closed_for_a_foreign_identity(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    output = tmp_path / "out"
    finalize(checkpoint, output, SOURCE, CONTAM, partition.to_dict())
    with pytest.raises(CheckpointViolation, match="identity mismatch"):
        read_completion(checkpoint.root, output,
                        identity_for(documents, partition, max_length=512))


def test_finalisation_is_idempotent(tmp_path, fixture):
    documents, partition, _ = fixture
    checkpoint, _ = run_preparation(tmp_path, documents, partition)
    output = tmp_path / "out"
    finalize(checkpoint, output, SOURCE, CONTAM, partition.to_dict())
    first = prepared_bytes(output)
    finalize(checkpoint, output, SOURCE, CONTAM, partition.to_dict())
    assert prepared_bytes(output) == first


# ---------------------------------------------------------------------------
# Atomic write primitives
# ---------------------------------------------------------------------------
def test_atomic_write_verifies_from_disk(tmp_path):
    size, digest = atomic_write_bytes(tmp_path / "a.json", b"hello")
    assert size == 5 and digest == hashlib.sha256(b"hello").hexdigest()
    assert not (tmp_path / "a.json.tmp").exists()


def test_verify_file_detects_size_and_digest_drift(tmp_path):
    path = tmp_path / "b.bin"
    path.write_bytes(b"12345")
    with pytest.raises(CheckpointViolation, match="bytes on disk"):
        verify_file(path, 4, sha256_file(path))
    with pytest.raises(CheckpointViolation, match="sha256"):
        verify_file(path, 5, "0" * 64)


def test_orphan_temporary_removal(tmp_path):
    (tmp_path / "x.tmp").write_text("x")
    (tmp_path / "keep.json").write_text("{}")
    assert remove_orphan_temporaries(tmp_path) == ["x.tmp"]
    assert (tmp_path / "keep.json").exists()


def test_checkpoint_state_carries_no_corpus_text(tmp_path, fixture):
    documents, partition, _ = fixture
    run_preparation(tmp_path, documents, partition, stop_after=21)
    state = (tmp_path / "cp" / STATE_NAME).read_text(encoding="utf-8")
    for document in documents[:5]:
        assert document.content[:40] not in state
    assert CHECKPOINT_SCHEMA_VERSION in state
