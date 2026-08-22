"""Corpus pin, schema, contamination screen and the document-level split.

ML-free and synthetic: no parquet, no UVW bytes, no network. `pyarrow` is only
imported by `read_shard`, which these tests deliberately do not call -- the
contracts under test are all in the pure-Python layer beneath it.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from unmark.stage1.corpus import (
    CorpusContractViolation,
    CorpusDocument,
    CorpusPin,
    ShardPin,
    _documents_from_columns,
    canonical_digest,
    check_schema,
    concatenate,
    file_sha256,
    load_pin,
    partition_documents,
    screen_contamination,
    verify_corpus_root,
)
from unmark.stage1.protocol import (
    CORPUS_DATASET,
    CORPUS_REVISION,
    CORPUS_SHARD_ORDER,
    DEV_DOCUMENTS,
    SPLIT_SEED,
)


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------
def test_the_committed_pin_matches_the_locked_protocol():
    pin = load_pin()
    assert pin.dataset == CORPUS_DATASET == "undertheseanlp/UVW-2026"
    assert pin.revision == CORPUS_REVISION == "a0a79294e4568137e25828bb3f2a4cde8546e1fb"
    assert len(pin.revision) == 40
    assert pin.concatenation_order == CORPUS_SHARD_ORDER
    assert tuple(f.name for f in pin.files) == (
        "train.parquet", "validation.parquet", "test.parquet"
    )
    for shard in pin.files:
        assert shard.bytes > 0
        assert len(shard.sha256) == 64


def test_a_moving_main_is_not_a_pin():
    with pytest.raises(CorpusContractViolation, match="full 40-character sha"):
        CorpusPin("d", "main", (ShardPin("a", 1, "x" * 64),), ("a",), "v1")


def write_shards(tmp_path, contents: dict[str, bytes]):
    for name, blob in contents.items():
        (tmp_path / name).write_bytes(blob)


def pin_for(tmp_path, contents: dict[str, bytes]) -> CorpusPin:
    return CorpusPin(
        dataset=CORPUS_DATASET,
        revision=CORPUS_REVISION,
        files=tuple(
            ShardPin(n, len(b), hashlib.sha256(b).hexdigest()) for n, b in contents.items()
        ),
        concatenation_order=tuple(contents),
        schema_version="stage1-corpus-pin-v1",
    )


@pytest.fixture
def three(tmp_path):
    contents = {
        "train.parquet": b"TRAIN-BYTES",
        "validation.parquet": b"VALIDATION-BYTES",
        "test.parquet": b"TEST-BYTES",
    }
    write_shards(tmp_path, contents)
    return tmp_path, contents, pin_for(tmp_path, contents)


def test_all_three_files_are_required(three):
    root, contents, pin = three
    assert verify_corpus_root(root, pin)["files"][0]["name"] == "train.parquet"
    (root / "validation.parquet").unlink()
    with pytest.raises(CorpusContractViolation, match="missing corpus shard validation"):
        verify_corpus_root(root, pin)


def test_a_wrong_byte_size_fails_closed(three):
    root, contents, pin = three
    (root / "test.parquet").write_bytes(b"TEST-BYTES-LONGER")
    with pytest.raises(CorpusContractViolation, match="expected .* bytes"):
        verify_corpus_root(root, pin)


def test_a_wrong_digest_fails_closed_even_at_the_right_size(three):
    root, contents, pin = three
    original = contents["train.parquet"]
    (root / "train.parquet").write_bytes(b"X" * len(original))
    with pytest.raises(CorpusContractViolation, match="sha256 mismatch"):
        verify_corpus_root(root, pin)


def test_verification_reports_the_locked_order(three):
    root, _, pin = three
    report = verify_corpus_root(root, pin)
    assert report["concatenation_order"] == list(CORPUS_SHARD_ORDER)
    assert report["shard_labels_are_a_split"] is False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_id_and_content_are_required():
    check_schema(("id", "content", "title"), "train.parquet")
    with pytest.raises(CorpusContractViolation, match="missing required column"):
        check_schema(("id", "title"), "train.parquet")


@pytest.mark.parametrize("ids, contents, message", [
    ([None, "b"], ["x", "y"], "null/empty document id"),
    (["", "b"], ["x", "y"], "null/empty document id"),
    (["a", "b"], [None, "y"], "null/empty content"),
    (["a", "b"], ["x", "   "], "null/empty content"),
])
def test_null_ids_and_text_are_rejected(ids, contents, message):
    with pytest.raises(CorpusContractViolation, match=message):
        _documents_from_columns(ids, contents, "train.parquet")


def test_duplicate_document_ids_fail_closed_and_are_reported():
    shards = {
        "train.parquet": _documents_from_columns(["a", "b"], ["x", "y"], "train.parquet"),
        "validation.parquet": _documents_from_columns(["b"], ["z"], "validation.parquet"),
        "test.parquet": _documents_from_columns(["c"], ["w"], "test.parquet"),
    }
    with pytest.raises(CorpusContractViolation) as excinfo:
        concatenate(shards)
    message = str(excinfo.value)
    assert "duplicate document id" in message
    assert "'b'" in message, "the offending id must be reported, not silently renamed"
    assert "Refusing to rename" in message


def test_concatenation_uses_the_locked_order_regardless_of_dict_order():
    shards = {
        "test.parquet": _documents_from_columns(["c"], ["w"], "test.parquet"),
        "train.parquet": _documents_from_columns(["a"], ["x"], "train.parquet"),
        "validation.parquet": _documents_from_columns(["b"], ["y"], "validation.parquet"),
    }
    assert [d.document_id for d in concatenate(shards)] == ["a", "b", "c"]


def test_a_missing_shard_is_refused():
    with pytest.raises(CorpusContractViolation, match="missing shard"):
        concatenate({"train.parquet": []})


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------
def docs(n, prefix="d"):
    return [CorpusDocument(f"{prefix}{i}", f"Câu văn số {i}", "train.parquet", i) for i in range(n)]


def test_only_already_opened_uitvsfc_sources_are_accepted():
    documents = docs(3)
    kept, report = screen_contamination(documents, {"uitvsfc_derived_train": []})
    assert len(kept) == 3 and report.excluded_count == 0

    for forbidden in ("uitvsfc_official_test", "official_test", "test"):
        with pytest.raises(CorpusContractViolation, match="contamination screen refuses"):
            screen_contamination(documents, {forbidden: ["x"]})


def test_exact_canonical_duplicates_are_excluded():
    documents = docs(5)
    kept, report = screen_contamination(
        documents, {"uitvsfc_official_validation": ["Câu văn số 2"]}
    )
    assert report.excluded_count == 1
    assert report.excluded_document_ids == ("d2",)
    assert "d2" not in {d.document_id for d in kept}


def test_screening_is_canonical_not_byte_equality():
    """`hoà` and `hòa` are the same canonical text, so both must be caught."""
    documents = [CorpusDocument("x1", "hoà bình", "train.parquet", 0)]
    kept, report = screen_contamination(documents, {"uitvsfc_derived_train": ["hòa bình"]})
    assert report.excluded_count == 1 and not kept


def test_near_duplicates_are_NOT_excluded():
    """Exact/canonical only. A fuzzy match would need a threshold, and thresholds
    are choices this pipeline does not make."""
    documents = [CorpusDocument("x1", "Câu văn số 2 và thêm chữ", "train.parquet", 0)]
    kept, report = screen_contamination(documents, {"uitvsfc_derived_train": ["Câu văn số 2"]})
    assert report.excluded_count == 0 and len(kept) == 1


def test_the_report_carries_no_uitvsfc_text_and_claims_nothing_about_TEST():
    _, report = screen_contamination(docs(3), {"uitvsfc_derived_train": ["Câu văn số 1"]})
    payload = json.dumps(report.to_dict())
    assert "Câu văn số 1" not in payload, "the screen must not persist UIT-VSFC text"
    assert report.to_dict()["official_test_screened"] is False
    assert "SEALED" in report.to_dict()["claim"]
    assert "NOT a claim of zero overlap" in report.to_dict()["claim"]


# ---------------------------------------------------------------------------
# The document split
# ---------------------------------------------------------------------------
def test_exactly_five_thousand_dev_documents():
    ids = [f"doc-{i}" for i in range(20000)]
    part = partition_documents(ids)
    assert len(part.dev) == DEV_DOCUMENTS == 5000
    assert len(part.train) == 15000
    assert not set(part.train) & set(part.dev)
    assert len(set(part.train) | set(part.dev)) == 20000


def test_the_split_uses_the_locked_seed():
    assert SPLIT_SEED == 51733
    ids = [f"doc-{i}" for i in range(8000)]
    a = partition_documents(ids, dev_documents=100)
    b = partition_documents(ids, dev_documents=100, seed=SPLIT_SEED)
    assert a.membership_digest == b.membership_digest
    c = partition_documents(ids, dev_documents=100, seed=SPLIT_SEED + 1)
    assert c.membership_digest != a.membership_digest


def test_the_split_is_order_independent():
    ids = [f"doc-{i}" for i in range(8000)]
    shuffled = ids[:]
    random.Random(1234).shuffle(shuffled)
    assert (
        partition_documents(ids, dev_documents=250).membership_digest
        == partition_documents(shuffled, dev_documents=250).membership_digest
    )


def test_the_split_refuses_duplicates_and_impossible_sizes():
    with pytest.raises(CorpusContractViolation, match="duplicate document ids"):
        partition_documents(["a", "a", "b"], dev_documents=1)
    with pytest.raises(CorpusContractViolation, match="cannot take"):
        partition_documents([f"d{i}" for i in range(10)], dev_documents=10)


def test_no_document_is_in_both_partitions():
    ids = [f"doc-{i}" for i in range(9000)]
    part = partition_documents(ids, dev_documents=1500)
    assignment = part.assignment
    assert len(assignment) == 9000
    assert set(assignment.values()) == {"train", "dev"}
