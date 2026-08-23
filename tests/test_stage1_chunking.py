"""The deterministic pre-chunker: safe boundaries, exact tiling, fail-closed.

ML-free: the length functions are injected, so the whole contract is provable
without the real pinned tokenizer.

Shaped by the real UVW-2026 defect (D-S1B-009): the locked corpus contains
maximal **non-whitespace** units far larger than `max_length` -- underscored
article titles -- so a whitespace-only cutting rule could not prepare it at all.
The corrected rule cuts at boundaries the orthography module itself reports.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.orthography import decompose
from unmark.orthography.decompose import source_letter_runs
from unmark.stage1.chunking import (
    ChunkingViolation,
    PreparedChunk,
    chunk_corpus,
    chunk_document,
    safe_cut_offsets,
    verify_no_parent_spans_partitions,
    verify_tiles_source,
)
from unmark.stage1.corpus import CorpusDocument, partition_documents
from unmark.stage1.protocol import MAX_LENGTH


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def doc(doc_id, content, shard="train.parquet", row=0):
    return CorpusDocument(doc_id, content, shard, row)


# mock tokenizers: whitespace tokens + 2 specials. The base path is deliberately
# LONGER, mirroring the real asymmetry between the two Stage-1 branches.
def ref_len(text: str) -> int:
    return len(text.split()) + 2


def base_len(text: str) -> int:
    return len(text.split()) + 4


# ~1 token per character: mirrors the real evidence (2 647 chars -> 1 707 tokens)
def char_len(text: str) -> int:
    return len(text) + 2


def cut(document, partition="train", ref=ref_len, base=base_len, max_length=256, **kw):
    return chunk_document(
        document, partition, reference_length=ref, base_length=base,
        max_length=max_length, **kw,
    )


# ---------------------------------------------------------------------------
# 1-2. Oversized non-whitespace spans -- the real defect
# ---------------------------------------------------------------------------
TITLE = "Đội_tuyển_bóng_đá_quốc_gia_Afghanistan"


def test_an_oversized_maximal_non_whitespace_span_is_subdivided():
    """The committed chunker raised here; whitespace-only cutting cannot work."""
    content = "_".join([TITLE] * 12)
    assert not any(c.isspace() for c in content), "fixture must be ONE non-whitespace unit"
    assert char_len(content) > 256

    chunks = cut(doc("a", content), ref=char_len, base=char_len)
    assert len(chunks) > 1
    assert all(c.reference_length <= 256 and c.base_length <= 256 for c in chunks)
    assert "".join(c.text for c in chunks) == content


def test_a_longer_base_path_forces_an_earlier_cut():
    content = "_".join([TITLE] * 12)
    lenient = cut(doc("a", content), ref=char_len, base=char_len)
    strict = cut(
        doc("a", content), ref=char_len, base=lambda t: 2 * len(t) + 2,
    )
    assert len(strict) > len(lenient), "the longer BASE path must cut earlier"
    for chunk in strict:
        assert chunk.base_length <= 256


def test_internal_punctuation_offers_legal_boundaries():
    for separator in ("_", "-", ".", ",", ";", ":", "/", "|"):
        content = separator.join(["Viên_Chiếu"] * 60)
        chunks = cut(doc(f"d{separator}", content), ref=char_len, base=char_len)
        assert len(chunks) > 1, separator
        assert "".join(c.text for c in chunks) == content


def test_a_long_non_vietnamese_no_whitespace_surface_is_subdividable():
    """Real evidence shape: a very long unit with legal internal boundaries."""
    content = "-".join([f"Section{i}.Sub{i}" for i in range(80)])
    assert not any(c.isspace() for c in content)
    chunks = cut(doc("mixed", content), ref=char_len, base=char_len)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == content
    assert all(c.reference_length <= 256 for c in chunks)


# ---------------------------------------------------------------------------
# 4-5. Syllable spans are never bisected; truly atomic regions fail closed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    "_".join([TITLE] * 12),
    "_".join(["Quần_đảo_Hoàng_Sa"] * 20),
    ",".join(["Viên_Chiếu"] * 60),
    "".join(f"Tôi{c}đã{c}đọc" for c in "-.,;:/|()[]") * 8,
])
def test_no_vietnamese_candidate_span_is_ever_bisected(content):
    chunks = cut(doc("x", content), ref=char_len, base=char_len)
    interior = set()
    for span in decompose(content).syllables:
        interior.update(range(span.canonical_start + 1, span.canonical_end))
    boundaries = {c.source_start for c in chunks} | {c.source_end for c in chunks}
    assert not (boundaries & interior), sorted(boundaries & interior)[:5]


def test_safe_offsets_come_from_the_orthography_module_not_a_second_parser():
    text = "Đội_tuyển_bóng"
    offsets = safe_cut_offsets(text)
    spans = decompose(text).syllables
    for span in spans:
        for inside in range(span.canonical_start + 1, span.canonical_end):
            assert inside not in offsets, f"offset {inside} is inside span {span.text!r}"
        assert span.canonical_start in offsets
        assert span.canonical_end in offsets


def test_a_genuinely_atomic_vietnamese_span_still_fails_closed():
    """One alphabetic run longer than max_length has no legal interior cut."""
    atomic = "Nghiêng" * 60  # a single maximal alphabetic run
    assert len(decompose(atomic).syllables) == 1, "fixture must be ONE span"
    with pytest.raises(ChunkingViolation) as excinfo:
        cut(doc("atomic", atomic, shard="test.parquet", row=41), ref=char_len, base=char_len)
    message = str(excinfo.value)
    assert "indivisible orthographic region" in message
    assert "atomic" in message and "test.parquet" in message and "row 41" in message
    assert "does not truncate and does not drop text" in message


def test_non_canonical_text_still_offers_safe_source_boundaries():
    """REPLACED, not weakened (Audit 029 §Z).

    This test previously asserted `safe_cut_offsets(nfd) == frozenset()` -- i.e.
    it asserted the defect. `decompose` does report canonical offsets, and using
    those on a non-canonical source really would cut in the wrong place; the old
    code was right to refuse *them*. It was wrong to conclude that the source has
    no safe boundaries, and that conclusion stopped Stage 6 at document 847 848.

    Offsets are now computed in source coordinates, so a non-canonical string
    keeps the boundaries it visibly has -- and they still address the original.
    """
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Tôi đã đọc")
    assert nfd != unicodedata.normalize("NFC", nfd)

    offsets = safe_cut_offsets(nfd)
    interior = offsets - {0, len(nfd)}
    assert interior, "a non-canonical string must not become globally indivisible"
    # Every offset indexes the SOURCE, and none splits a base from its marks.
    for offset in offsets:
        assert 0 <= offset <= len(nfd)
        if 0 < offset < len(nfd):
            assert unicodedata.combining(nfd[offset]) == 0
    # And no offset lands inside an alphabetic run.
    for start, end in source_letter_runs(nfd):
        assert not any(start < o < end for o in offsets)


# ---------------------------------------------------------------------------
# 6-10. Whitespace fidelity and exact reconstruction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    "alpha  beta   gamma",                       # multiple spaces
    "alpha\tbeta\t\tgamma",                      # tabs
    "alpha\nbeta\n\ngamma",                      # newlines
    "  leading and trailing  ",                  # leading + trailing whitespace
    "\n\tmixed \t\n whitespace \n\n",
    "Tôi đã đọc\n\nquyển sách  này\trồi",
])
def test_exact_reconstruction_and_range_tiling(content):
    chunks = cut(doc("w", content), max_length=8)
    assert "".join(c.text for c in chunks) == content, "byte-exact reconstruction"
    cursor = 0
    for chunk in chunks:
        assert chunk.source_start == cursor, "no gaps, no overlaps"
        assert chunk.text == content[chunk.source_start:chunk.source_end]
        cursor = chunk.source_end
    assert cursor == len(content), "ranges must tile the whole document"


def test_whitespace_is_never_collapsed_or_synthesised():
    content = "a  b\t\tc\n\nd"
    # max_length must exceed the mock base path's +4 special-token overhead, or
    # even a single character is unfittable and the fixture tests nothing.
    chunks = cut(doc("w", content), max_length=6)
    rebuilt = "".join(c.text for c in chunks)
    assert rebuilt == content
    assert rebuilt.count(" ") == content.count(" ")
    assert rebuilt.count("\t") == content.count("\t")
    assert rebuilt.count("\n") == content.count("\n")


def test_no_interior_content_is_lost_or_reordered():
    content = words(97)
    chunks = cut(doc("a", content), max_length=15)
    assert "".join(c.text for c in chunks) == content
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_the_tiling_verifier_actually_catches_a_gap():
    content = "abcdef"
    bad = [
        PreparedChunk("d#0", "d", "train", 0, "abc", 0, 3, 3, 3, "train.parquet"),
        PreparedChunk("d#1", "d", "train", 1, "ef", 4, 6, 2, 2, "train.parquet"),
    ]
    with pytest.raises(ChunkingViolation, match="expected 3"):
        verify_tiles_source(bad, content, "d")


def test_the_tiling_verifier_catches_short_coverage():
    good_prefix = [PreparedChunk("d#0", "d", "train", 0, "abc", 0, 3, 3, 3, "train.parquet")]
    with pytest.raises(ChunkingViolation, match="text would be lost"):
        verify_tiles_source(good_prefix, "abcdef", "d")


# ---------------------------------------------------------------------------
# 11-16. Ids, determinism, fitting, no truncation
# ---------------------------------------------------------------------------
def test_stable_chunk_ids_across_repeated_calls():
    content = words(40)
    first = cut(doc("vi-123", content), "dev", max_length=12)
    again = cut(doc("vi-123", content), "dev", max_length=12)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert first[0].chunk_id == "vi-123#0"
    assert [(c.source_start, c.source_end) for c in first] == [
        (c.source_start, c.source_end) for c in again
    ]


def test_a_mismatched_chunk_id_is_refused():
    with pytest.raises(ChunkingViolation, match="does not match"):
        PreparedChunk("wrong", "d", "train", 0, "text", 0, 4, 3, 3, "train.parquet")


def test_a_chunk_whose_text_is_not_its_slice_is_refused():
    with pytest.raises(ChunkingViolation, match="must slice, never rewrite"):
        PreparedChunk("d#0", "d", "train", 0, "rewritten", 0, 4, 3, 3, "train.parquet")


def test_document_order_does_not_change_any_document_s_chunks():
    documents = [doc(f"d{i}", words(30 + i), row=i) for i in range(12)]
    partition = partition_documents([d.document_id for d in documents], dev_documents=4)
    forward = chunk_corpus(documents, partition.assignment, reference_length=ref_len,
                           base_length=base_len, max_length=14)
    backward = chunk_corpus(list(reversed(documents)), partition.assignment,
                            reference_length=ref_len, base_length=base_len, max_length=14)
    by_id = lambda cs: {c.chunk_id: (c.text, c.partition) for c in cs}
    assert by_id(forward) == by_id(backward)


def test_every_emitted_chunk_fits_both_paths():
    content = " ".join(["_".join([TITLE] * 3)] * 6)
    chunks = cut(doc("a", content), ref=char_len, base=lambda t: len(t) + 30)
    for chunk in chunks:
        assert chunk.reference_length <= 256
        assert chunk.base_length <= 256
        assert char_len(chunk.text) <= 256


def test_no_truncation_and_no_dropping():
    content = "  " + "_".join([TITLE] * 9) + "  tail here\n"
    chunks = cut(doc("a", content), ref=char_len, base=char_len)
    assert "".join(c.text for c in chunks) == content
    assert sum(c.source_end - c.source_start for c in chunks) == len(content)


def test_diacritics_survive_chunking_unchanged():
    content = "Tôi đã đọc quyển sách này rồi và thấy rất hay ở đây"
    chunks = cut(doc("vi", content), max_length=8)
    assert "".join(c.text for c in chunks) == content
    assert any("ô" in c.text or "đ" in c.text or "ấ" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# 17. Split-before-chunk and partition inheritance
# ---------------------------------------------------------------------------
def test_chunking_requires_a_partition_and_never_invents_one():
    with pytest.raises(ChunkingViolation, match="requires the parent document's partition"):
        cut(doc("a", words(5)), "somewhere")


def test_every_chunk_inherits_its_parent_partition():
    documents = [doc(f"d{i}", words(60), row=i) for i in range(30)]
    partition = partition_documents([d.document_id for d in documents], dev_documents=10)
    chunks = chunk_corpus(documents, partition.assignment, reference_length=ref_len,
                          base_length=base_len, max_length=15)
    for chunk in chunks:
        assert chunk.partition == partition.assignment[chunk.document_id]


def test_no_article_can_span_train_and_dev():
    documents = [doc(f"d{i}", words(80), row=i) for i in range(40)]
    partition = partition_documents([d.document_id for d in documents], dev_documents=12)
    chunks = chunk_corpus(documents, partition.assignment, reference_length=ref_len,
                          base_length=base_len, max_length=14)
    assert verify_no_parent_spans_partitions(chunks) == 40
    sides = {}
    for chunk in chunks:
        sides.setdefault(chunk.document_id, set()).add(chunk.partition)
    assert all(len(s) == 1 for s in sides.values())


def test_the_span_invariant_is_actually_checked_not_assumed():
    bad = [
        PreparedChunk("d#0", "d", "train", 0, "a", 0, 1, 3, 5, "train.parquet"),
        PreparedChunk("d#1", "d", "dev", 1, "b", 1, 2, 3, 5, "train.parquet"),
    ]
    with pytest.raises(ChunkingViolation, match="chunks in BOTH partitions"):
        verify_no_parent_spans_partitions(bad)


def test_chunking_refuses_documents_with_no_partition():
    documents = [doc("a", words(10)), doc("b", words(10))]
    with pytest.raises(ChunkingViolation, match="split must run BEFORE chunking"):
        chunk_corpus(documents, {"a": "train"}, reference_length=ref_len,
                     base_length=base_len, max_length=20)


# ---------------------------------------------------------------------------
# No normalization
# ---------------------------------------------------------------------------
def test_the_chunker_normalises_nothing():
    """It may QUERY `decompose` for safe offsets, but must not rewrite text.

    `decompose` is now legitimately called -- that is the whole repair -- so the
    check is on the *rewriting* primitives, and on the structural guarantee that
    every chunk is a slice of the source.
    """
    tree = ast.parse(pathlib.Path("unmark/stage1/chunking.py").read_text(encoding="utf-8"))
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    for forbidden in ("canon", "corrupt", "normalize", "lower", "upper", "strip_to_base",
                      "recompose", "replace", "sub"):
        assert forbidden not in called, f"the chunker must not call {forbidden}()"
    # Boundaries must still come from the orthography layer. After the
    # source-coordinate repair the primitives are the offset-carrying ones;
    # `decompose` itself is no longer called, because it canonicalises first.
    assert {"source_letter_runs", "split_units_with_offsets"} <= called, (
        "safe boundaries must come from the orthography module"
    )


def test_max_length_default_is_the_locked_256():
    assert MAX_LENGTH == 256
    chunks = cut(doc("a", words(50)))
    assert len(chunks) == 1, "50 short words fit comfortably inside the locked 256"
