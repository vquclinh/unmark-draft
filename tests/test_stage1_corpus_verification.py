"""A training run may only consume a verified prepared corpus (Audit 030 F1).

Before the hardening, `execute.load_prepared_chunks` read `chunks.jsonl` whole
and `execute_stage` recorded `manifest["counts"]["chunk_membership_digest"]` as
provenance **without ever verifying that the bytes it loaded corresponded to
that digest**. A truncated, modified, swapped or foreign payload would have
trained silently under a digest describing different data.

`verify_prepared_corpus` closes that. These tests are the rejection matrix: each
one corrupts exactly one thing and requires a fail-closed refusal.

No real corpus is used -- every fixture is a small synthetic prepared corpus
built through the same `write_completion_marker` the real Stage 6 used.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.checkpoint import (
    CheckpointIdentity,
    CheckpointViolation,
    VerifiedCorpus,
    verify_prepared_corpus,
    write_completion_marker,
)
from unmark.stage1.manifest import CHUNKS_NAME, MANIFEST_NAME, MANIFEST_SCHEMA_VERSION
from unmark.stage1.protocol import (
    CHUNK_SCHEMA_VERSION,
    CONTAMINATION_METHOD,
    CORPUS_DATASET,
    CORPUS_REVISION,
    DEV_DOCUMENTS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    RAW_BASE_POLICY,
    SPLIT_SEED,
    STAGE1_PROTOCOL_VERSION,
)

DIGEST = "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6"


def identity(**overrides) -> CheckpointIdentity:
    base = dict(
        repository_head="a" * 40,
        protocol_version=STAGE1_PROTOCOL_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        corpus_dataset=CORPUS_DATASET,
        corpus_revision=CORPUS_REVISION,
        corpus_files=(("train.parquet", 1, "b" * 64),),
        tokenizer_checkpoint=ENCODER_CHECKPOINT,
        tokenizer_revision=ENCODER_REVISION,
        transformers_version="4.57.6",
        max_length=MAX_LENGTH,
        raw_base_policy=RAW_BASE_POLICY,
        split_seed=SPLIT_SEED,
        dev_documents=DEV_DOCUMENTS,
        contamination_method=CONTAMINATION_METHOD,
        contamination_excluded_count=0,
        document_sequence_digest="c" * 64,
        partition_assignment_digest="d" * 64,
    )
    base.update(overrides)
    return CheckpointIdentity(**base)


def counts() -> dict:
    return {
        "chunks_total": 3,
        "chunks_by_partition": {"train": 2, "dev": 1},
        "parent_documents_total": 2,
        "parent_documents_by_partition": {"train": 1, "dev": 1},
        "parents_spanning_both_partitions": 0,
        "overflow_count": 0,
        "base_invariance_violations": 0,
        "chunk_membership_digest": DIGEST,
    }


def manifest_dict() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "official_test_used": False,
        "source": {"dataset": CORPUS_DATASET, "revision": CORPUS_REVISION,
                   "shard_order": ["train.parquet"], "shard_labels_are_a_split": False},
        "chunking": {"max_length": MAX_LENGTH, "split_before_chunk": True,
                     "chunks_inherit_parent_partition": True, "truncation": False,
                     "tokenizer_revision": ENCODER_REVISION,
                     "tokenizer_checkpoint": ENCODER_CHECKPOINT,
                     "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                     "on_overflow": "FAIL"},
        "partition": {"seed": SPLIT_SEED, "dev_documents": DEV_DOCUMENTS,
                      "unit": "document"},
        "contamination": {"method": CONTAMINATION_METHOD, "excluded_count": 0,
                          "official_test_screened": False},
        "counts": counts(),
    }


CHUNK_LINES = [
    {"chunk_id": "doc-0#0", "document_id": "doc-0", "partition": "train",
     "chunk_index": 0, "text": "alpha", "source_start": 0, "source_end": 5,
     "source_shard": "train.parquet"},
    {"chunk_id": "doc-0#1", "document_id": "doc-0", "partition": "train",
     "chunk_index": 1, "text": " beta", "source_start": 5, "source_end": 10,
     "source_shard": "train.parquet"},
    {"chunk_id": "doc-1#0", "document_id": "doc-1", "partition": "dev",
     "chunk_index": 0, "text": "gamma", "source_start": 0, "source_end": 5,
     "source_shard": "train.parquet"},
]


def build_corpus(tmp_path, *, manifest=None, ident=None, drop=None,
                 marker_counts=None) -> tuple[pathlib.Path, pathlib.Path]:
    """A small prepared corpus, completed exactly as Stage 6 completes one."""
    prepared = tmp_path / "prepared"
    checkpoint = tmp_path / "checkpoint"
    prepared.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CHUNK_LINES)
    chunks_bytes = payload.encode("utf-8")
    (prepared / CHUNKS_NAME).write_bytes(chunks_bytes)

    manifest_bytes = (
        json.dumps(manifest or manifest_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    (prepared / MANIFEST_NAME).write_bytes(manifest_bytes)

    artifacts = {
        CHUNKS_NAME: (len(chunks_bytes), hashlib.sha256(chunks_bytes).hexdigest()),
        MANIFEST_NAME: (len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()),
    }
    if drop is not None:
        artifacts.pop(drop)
    write_completion_marker(
        checkpoint, identity=ident or identity(), artifacts=artifacts,
        counts=marker_counts if marker_counts is not None else counts(),
    )
    return prepared, checkpoint


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def test_a_valid_completed_prepared_corpus_is_accepted(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    verified = verify_prepared_corpus(prepared, checkpoint)
    assert isinstance(verified, VerifiedCorpus)
    assert verified.chunk_membership_digest == DIGEST
    assert set(verified.artifacts) == {CHUNKS_NAME, MANIFEST_NAME}
    assert verified.counts["chunks_total"] == 3


def test_the_verified_digest_is_what_provenance_should_record(tmp_path):
    """The point of F1: the digest is a checked value, not a declaration."""
    prepared, checkpoint = build_corpus(tmp_path)
    verified = verify_prepared_corpus(prepared, checkpoint)
    assert verified.chunk_membership_digest == verified.counts["chunk_membership_digest"]
    assert verified.chunk_membership_digest == verified.manifest["counts"]["chunk_membership_digest"]


def test_a_relocated_prepared_corpus_still_verifies(tmp_path):
    """COMPLETE.json binds relative names, so the payload may move."""
    prepared, checkpoint = build_corpus(tmp_path)
    moved = tmp_path / "elsewhere" / "restored"
    moved.parent.mkdir(parents=True, exist_ok=True)
    prepared.rename(moved)
    assert verify_prepared_corpus(moved, checkpoint).chunk_membership_digest == DIGEST


# ---------------------------------------------------------------------------
# Rejection matrix -- one thing wrong per test
# ---------------------------------------------------------------------------
def test_missing_completion_marker_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    (checkpoint / "COMPLETE.json").unlink()
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "no completion marker" in str(caught.value)


def test_malformed_completion_marker_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    (checkpoint / "COMPLETE.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "malformed" in str(caught.value)


def test_an_incomplete_prepare_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    path = checkpoint / "COMPLETE.json"
    payload = json.loads(path.read_text())
    payload["complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "did not finish" in str(caught.value)


def test_one_modified_chunk_byte_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    data = bytearray((prepared / CHUNKS_NAME).read_bytes())
    data[0] = data[0] ^ 0x01          # exactly one bit
    (prepared / CHUNKS_NAME).write_bytes(bytes(data))
    with pytest.raises(CheckpointViolation):
        verify_prepared_corpus(prepared, checkpoint)


def test_a_truncated_chunk_file_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    data = (prepared / CHUNKS_NAME).read_bytes()
    (prepared / CHUNKS_NAME).write_bytes(data[:-10])
    with pytest.raises(CheckpointViolation):
        verify_prepared_corpus(prepared, checkpoint)


def test_an_appended_chunk_file_is_refused(tmp_path):
    prepared, checkpoint = build_corpus(tmp_path)
    with open(prepared / CHUNKS_NAME, "ab") as handle:
        handle.write(b'{"chunk_id": "smuggled#0"}\n')
    with pytest.raises(CheckpointViolation):
        verify_prepared_corpus(prepared, checkpoint)


@pytest.mark.parametrize("name", [CHUNKS_NAME, MANIFEST_NAME])
def test_a_missing_bound_file_is_refused(tmp_path, name):
    prepared, checkpoint = build_corpus(tmp_path)
    (prepared / name).unlink()
    with pytest.raises(CheckpointViolation):
        verify_prepared_corpus(prepared, checkpoint)


@pytest.mark.parametrize("name", [CHUNKS_NAME, MANIFEST_NAME])
def test_a_marker_that_does_not_bind_an_artifact_is_refused(tmp_path, name):
    prepared, checkpoint = build_corpus(tmp_path, drop=name)
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "does not bind" in str(caught.value)


def test_a_foreign_manifest_is_refused(tmp_path):
    """A manifest swapped for another prepare's: its hash no longer matches."""
    prepared, checkpoint = build_corpus(tmp_path)
    other = manifest_dict()
    other["counts"]["chunks_total"] = 999
    (prepared / MANIFEST_NAME).write_bytes(
        (json.dumps(other, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    )
    with pytest.raises(CheckpointViolation):
        verify_prepared_corpus(prepared, checkpoint)


def test_a_marker_whose_counts_disagree_with_the_manifest_is_refused(tmp_path):
    """Both files individually intact, but describing different prepares."""
    foreign = counts()
    foreign["chunks_total"] = 4
    prepared, checkpoint = build_corpus(tmp_path, marker_counts=foreign)
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "do not describe the same prepare" in str(caught.value)


@pytest.mark.parametrize("field,value", [
    ("protocol_version", "stage1-protocol-v0"),
    ("chunk_schema_version", "stage1-chunk-v0"),
    ("corpus_revision", "0" * 40),
    ("corpus_dataset", "someone/else"),
    ("tokenizer_revision", "0" * 40),
    ("tokenizer_checkpoint", "bert-base-uncased"),
    ("max_length", 512),
    ("raw_base_policy", "SOMETHING_ELSE"),
    ("split_seed", 1),
    ("dev_documents", 1000),
])
def test_an_identity_off_the_locked_protocol_is_refused(tmp_path, field, value):
    prepared, checkpoint = build_corpus(tmp_path, ident=identity(**{field: value}))
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert field in str(caught.value)


def test_a_marker_without_a_membership_digest_is_refused(tmp_path):
    bare = counts()
    bare.pop("chunk_membership_digest")
    bad_manifest = manifest_dict()
    bad_manifest["counts"] = bare
    prepared, checkpoint = build_corpus(tmp_path, manifest=bad_manifest, marker_counts=bare)
    with pytest.raises(CheckpointViolation) as caught:
        verify_prepared_corpus(prepared, checkpoint)
    assert "chunk_membership_digest" in str(caught.value)


# ---------------------------------------------------------------------------
# The consumer actually uses it
# ---------------------------------------------------------------------------
def test_execute_stage_records_the_verified_digest_not_a_manifest_field():
    """Structural: the provenance digest must come from `verified`."""
    import ast
    import inspect

    import unmark.stage1.execute as module

    source = inspect.getsource(module)
    assert "verified.chunk_membership_digest" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            text = ast.unparse(node)
            assert "manifest['counts']['chunk_membership_digest']" not in text, (
                "provenance is reading an unverified manifest field again"
            )


def test_every_training_command_verifies_before_anything_else():
    source = pathlib.Path("scripts/stage1_runner.py").read_text(encoding="utf-8")
    for command in ("run_lr_pilot", "run_r_phase1", "run_final_main"):
        body = source.split(f"def {command}(args)")[1].split("\ndef ")[0]
        assert "_verified_corpus(args)" in body, f"{command} does not verify the corpus"
