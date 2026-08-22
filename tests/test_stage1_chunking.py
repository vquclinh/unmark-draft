"""The deterministic pre-chunker, and the split-before-chunk invariant.

ML-free: the length functions are injected, so the whole contract is provable
without the real pinned tokenizer.
"""

from __future__ import annotations

import pytest

from unmark.stage1.chunking import (
    ChunkingViolation,
    PreparedChunk,
    chunk_corpus,
    chunk_document,
    verify_no_parent_spans_partitions,
)
from unmark.stage1.corpus import CorpusDocument, partition_documents
from unmark.stage1.protocol import MAX_LENGTH


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def doc(doc_id, n_words, shard="train.parquet", row=0):
    return CorpusDocument(doc_id, words(n_words), shard, row)


# mock tokenizers: whitespace tokens + 2 special tokens. The base path is
# deliberately *longer*, mirroring the real asymmetry between the two branches.
def ref_len(text: str) -> int:
    return len(text.split()) + 2


def base_len(text: str) -> int:
    return len(text.split()) + 4


def test_chunks_fit_both_tokenizer_paths():
    chunks = chunk_document(doc("a", 100), "train", reference_length=ref_len,
                            base_length=base_len, max_length=20)
    assert chunks
    for chunk in chunks:
        assert ref_len(chunk.text) <= 20
        assert base_len(chunk.text) <= 20, "the longer BASE path must also fit"
        assert chunk.reference_length <= 20 and chunk.base_length <= 20


def test_source_text_order_and_content_are_preserved():
    document = doc("a", 97)
    chunks = chunk_document(document, "train", reference_length=ref_len,
                            base_length=base_len, max_length=15)
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined == document.content, "no interior content may be lost or reordered"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_stable_chunk_ids():
    chunks = chunk_document(doc("vi-123", 40), "dev", reference_length=ref_len,
                            base_length=base_len, max_length=12)
    assert [c.chunk_id for c in chunks[:3]] == ["vi-123#0", "vi-123#1", "vi-123#2"]
    again = chunk_document(doc("vi-123", 40), "dev", reference_length=ref_len,
                           base_length=base_len, max_length=12)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_a_mismatched_chunk_id_is_refused():
    with pytest.raises(ChunkingViolation, match="does not match"):
        PreparedChunk("wrong", "d", "train", 0, "t", 3, 3, "train.parquet")


def test_no_truncation_ever_happens():
    """An indivisible span that cannot fit FAILS; it is never trimmed or dropped."""
    long_token = "x" * 400
    document = CorpusDocument("bad", f"short {long_token} tail", "test.parquet", 7)

    def ref(text):  # one token per character -- forces the single word to overflow
        return max(len(w) for w in text.split()) + 2

    with pytest.raises(ChunkingViolation) as excinfo:
        chunk_document(document, "train", reference_length=ref, base_length=ref, max_length=20)
    message = str(excinfo.value)
    assert "indivisible span" in message
    assert "bad" in message and "test.parquet" in message and "row 7" in message, (
        "failure must carry enough provenance to diagnose it"
    )
    assert "does not truncate and does not drop text" in message


def test_chunking_requires_a_partition_and_never_invents_one():
    with pytest.raises(ChunkingViolation, match="requires the parent document's partition"):
        chunk_document(doc("a", 5), "somewhere", reference_length=ref_len,
                       base_length=base_len, max_length=20)


def test_every_chunk_inherits_its_parent_partition():
    documents = [doc(f"d{i}", 60, row=i) for i in range(30)]
    partition = partition_documents([d.document_id for d in documents], dev_documents=10)
    chunks = chunk_corpus(documents, partition.assignment, reference_length=ref_len,
                          base_length=base_len, max_length=15)
    assignment = partition.assignment
    for chunk in chunks:
        assert chunk.partition == assignment[chunk.document_id]


def test_no_article_can_span_train_and_dev():
    """The invariant the whole split-before-chunk ordering exists to guarantee."""
    documents = [doc(f"d{i}", 80, row=i) for i in range(40)]
    partition = partition_documents([d.document_id for d in documents], dev_documents=12)
    chunks = chunk_corpus(documents, partition.assignment, reference_length=ref_len,
                          base_length=base_len, max_length=14)
    assert verify_no_parent_spans_partitions(chunks) == 40

    sides = {}
    for chunk in chunks:
        sides.setdefault(chunk.document_id, set()).add(chunk.partition)
    assert all(len(s) == 1 for s in sides.values())


def test_the_span_invariant_is_actually_checked_not_assumed():
    """Hand-build a violating chunk set; the verifier must catch it."""
    bad = [
        PreparedChunk("d#0", "d", "train", 0, "a", 3, 5, "train.parquet"),
        PreparedChunk("d#1", "d", "dev", 1, "b", 3, 5, "train.parquet"),
    ]
    with pytest.raises(ChunkingViolation, match="chunks in BOTH partitions"):
        verify_no_parent_spans_partitions(bad)


def test_chunking_refuses_documents_with_no_partition():
    documents = [doc("a", 10), doc("b", 10)]
    with pytest.raises(ChunkingViolation, match="split must run BEFORE chunking"):
        chunk_corpus(documents, {"a": "train"}, reference_length=ref_len,
                     base_length=base_len, max_length=20)


def test_the_chunker_performs_no_normalization():
    """It cuts; it does not canonicalise, restore or repair."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("unmark/stage1/chunking.py").read_text(encoding="utf-8"))
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    for forbidden in ("canon", "decompose", "corrupt", "normalize", "lower", "strip_to_base"):
        assert forbidden not in called, f"the chunker must not call {forbidden}()"


def test_diacritics_survive_chunking_unchanged():
    text = "Tôi đã đọc quyển sách này rồi và thấy rất hay ở đây"
    document = CorpusDocument("vi", text, "train.parquet", 0)
    chunks = chunk_document(document, "train", reference_length=ref_len,
                            base_length=base_len, max_length=8)
    assert " ".join(c.text for c in chunks) == text
    assert any("ô" in c.text or "đ" in c.text or "ấ" in c.text for c in chunks)


def test_max_length_default_is_the_locked_256():
    assert MAX_LENGTH == 256
    chunks = chunk_document(doc("a", 50), "train", reference_length=ref_len, base_length=base_len)
    assert len(chunks) == 1, "50 short words fit comfortably inside the locked 256"
