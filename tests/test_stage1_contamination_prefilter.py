"""The contamination screen's prefilters: zero false negatives, same decisions.

ML-free. The screen's *criterion* is unchanged -- exclusion is still decided by
`sha256(canon(x))` alone. These tests exist because the prefilters that keep a
1.1 M-document corpus off that expensive path are only acceptable if they are
**necessary conditions**: they may skip a document only after proving it cannot
match.

The reference implementation below is the pre-optimisation screen, kept **as a
test oracle only** -- there is no second production pathway.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import unicodedata as ud

import pytest

from unmark.orthography import canon
from unmark.orthography.marks import TONE_MARKS
from unmark.stage1.corpus import (
    CorpusDocument,
    _length_guard_excludes,
    _placement_insensitive_length,
    _standalone_tone_marks,
    canonical_digest,
    placement_insensitive_digest,
    screen_contamination,
)

TONES = "".join(TONE_MARKS)


def f(text: str) -> str:
    """`NFD(text)` with only the five tone marks removed."""
    return "".join(c for c in ud.normalize("NFD", text) if c not in TONES)


def reference_screen(documents, reference_texts):
    """The ORIGINAL screen, verbatim in behaviour. Test oracle only."""
    reference = set()
    for texts in reference_texts.values():
        reference.update(canonical_digest(t) for t in texts)
    kept, excluded_ids, excluded_digests = [], [], []
    for doc in documents:
        digest = canonical_digest(doc.content)
        if digest in reference:
            excluded_ids.append(doc.document_id)
            excluded_digests.append(digest)
        else:
            kept.append(doc)
    return kept, excluded_ids, excluded_digests


# ---------------------------------------------------------------------------
# The corpus of hazards every test family draws on
# ---------------------------------------------------------------------------
PLACEMENT_PAIRS = [
    ("hòa", "hoà"), ("thúy", "thuý"), ("khỏe", "khoẻ"),
    ("hòa bình", "hoà bình"), ("Thúy", "Thuý"), ("KHỎE", "KHOẺ"),
]

HAZARDS = [
    "", " ", "\t", "\n", "\r\n", "   ", "a  b\t\tc\n\nd",
    "Tôi đã đọc quyển sách này rồi", "TÔI ĐÃ ĐỌC", "Tôi Đã Đọc",
    "đường", "ĐƯỜNG", "Đ", "đ", "ăn", "âm", "êm", "ôm", "ơn", "ưu",
    "à", "á", "ã", "ả", "ạ", "ằ", "ắ", "ẵ", "ẳ", "ặ",
    "ề", "ế", "ễ", "ể", "ệ", "ồ", "ố", "ỗ", "ổ", "ộ",
    "Müller", "café", "naïve", "Ünter",
    "https://vi.wikipedia.org/wiki/Việt_Nam", "nguoi.dung@example.vn",
    "Tôi dùng Python và PyTorch", "1234567890", "!@#$%^&*()",
    "Việt Nam " * 200, "ế" * 60,
    "á̀b",          # two tone marks on one base: malformed, still handled
    "́abc",    # leading standalone tone mark
    "x̣́y",
]
HAZARDS += [a for pair in PLACEMENT_PAIRS for a in pair]
HAZARDS = [v for h in HAZARDS for v in (ud.normalize("NFC", h), ud.normalize("NFD", h))]


def random_strings(n, seed=51733, max_len=40):
    rng = random.Random(seed)
    alphabet = ("aeiouyăâêôơưAEIOUYĂÂÊÔƠƯbcdghklmnpqrstvxĐđ"
                + TONES + "̛̂̆ .,-_\t\n/@:")
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))
            for _ in range(n)]


# ---------------------------------------------------------------------------
# TASK C -- the necessary-condition lemmas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", HAZARDS)
def test_lemma_prefilter_is_invariant_under_canon(text):
    """`f(canon(x)) == f(x)` -- the whole proof rests on this."""
    assert f(canon(text)) == f(text)
    assert placement_insensitive_digest(canon(text)) == placement_insensitive_digest(text)


def test_lemma_holds_on_a_large_random_population():
    bad = [x for x in random_strings(20000) if f(canon(x)) != f(x)]
    assert not bad, bad[:3]


@pytest.mark.parametrize("a, b", PLACEMENT_PAIRS)
def test_placement_variants_survive_the_prefilter(a, b):
    """The case the prefilter must NOT reject: canon-equal, differently spelled."""
    assert canonical_digest(a) == canonical_digest(b), "fixture must be canon-equal"
    assert placement_insensitive_digest(a) == placement_insensitive_digest(b)


def test_nfc_and_nfd_forms_survive_the_prefilter():
    for text in ["Tôi đã đọc", "đường", "khỏe", "Việt Nam"]:
        nfc, nfd = ud.normalize("NFC", text), ud.normalize("NFD", text)
        assert nfc != nfd
        assert canonical_digest(nfc) == canonical_digest(nfd)
        assert placement_insensitive_digest(nfc) == placement_insensitive_digest(nfd)


def test_zero_false_negatives_over_a_large_cross_product():
    """No pair may be canon-equal while the prefilter separates them."""
    population = HAZARDS + random_strings(4000, seed=19225, max_len=18)
    by_canon: dict[str, list[str]] = {}
    for text in population:
        by_canon.setdefault(canonical_digest(text), []).append(text)
    checked = 0
    for group in by_canon.values():
        if len(group) < 2:
            continue
        digests = {placement_insensitive_digest(t) for t in group}
        assert len(digests) == 1, f"canon-equal group split by prefilter: {group[:3]}"
        checked += 1
    assert checked > 5, "the population must actually contain canon-equal groups"


# --- the length guard -------------------------------------------------------
@pytest.mark.parametrize("text", HAZARDS)
def test_lemma_length_lower_bound(text):
    """`len(f(x)) >= len(x) - standalone_tone_marks(x)`."""
    assert _placement_insensitive_length(text) >= len(text) - _standalone_tone_marks(text)


def test_length_lemma_on_a_large_random_population():
    bad = [
        x for x in random_strings(20000, seed=7)
        if _placement_insensitive_length(x) < len(x) - _standalone_tone_marks(x)
    ]
    assert not bad, bad[:3]


def test_the_length_guard_never_skips_a_possible_match():
    population = HAZARDS + random_strings(3000, seed=42, max_len=30)
    for a in population:
        for b in ("hòa", "Tôi đã đọc", "khỏe", ""):
            if canonical_digest(a) != canonical_digest(b):
                continue
            assert not _length_guard_excludes(a, _placement_insensitive_length(b)), (
                f"length guard would have skipped a genuine match: {a!r} vs {b!r}"
            )


def test_the_length_guard_actually_skips_long_documents():
    assert _length_guard_excludes("Việt Nam " * 500, 400)
    assert not _length_guard_excludes("hoà", 400)


# ---------------------------------------------------------------------------
# TASK C -- full-report equivalence against the pre-optimisation oracle
# ---------------------------------------------------------------------------
def build_corpus():
    """Deterministic corpus with every required category."""
    documents, expected = [], set()

    def add(doc_id, text, contaminated):
        documents.append(CorpusDocument(doc_id, text, "train.parquet", len(documents)))
        if contaminated:
            expected.add(doc_id)

    refs = ["Tôi đã đọc quyển sách này rồi", "hòa bình", "khỏe mạnh",
            "Giảng viên dạy dễ hiểu", "Sản phẩm rất tốt"]
    add("exact", "Tôi đã đọc quyển sách này rồi", True)          # true canonical duplicate
    add("placement", "hoà bình", True)                            # placement-only duplicate
    add("nfd", ud.normalize("NFD", "khỏe mạnh"), True)            # NFC/NFD duplicate
    add("near1", "Tôi đã đọc quyển sách này rồi.", False)         # near but not equal
    add("near2", "Tôi đã đọc quyển sách này", False)
    add("collision1", "hoa binh", False)                          # prefilter collision, no match
    add("collision2", "hòa bình!", False)
    add("collision3", "khoe manh", False)
    for i in range(40):                                           # ordinary nonmatches
        add(f"plain-{i}", f"Bài viết số {i} về lịch sử Việt Nam", False)
    for i in range(10):                                           # long nonmatches
        add(f"long-{i}", "Việt Nam là một quốc gia " * (60 + i), False)
    return documents, {"uitvsfc_derived_train": refs}, expected


def test_optimized_screen_matches_the_reference_implementation_exactly():
    documents, references, expected = build_corpus()
    kept, report = screen_contamination(documents, references)
    ref_kept, ref_ids, ref_digests = reference_screen(documents, references)

    assert list(report.excluded_document_ids) == ref_ids
    assert list(report.excluded_digests) == ref_digests
    assert [d.document_id for d in kept] == [d.document_id for d in ref_kept]
    assert set(report.excluded_document_ids) == expected
    assert report.reference_digest_count == len(
        {canonical_digest(t) for t in references["uitvsfc_derived_train"]}
    )


def test_prefilter_collisions_are_not_contamination():
    """A tier-2 match that fails the canonical check must be KEPT."""
    documents = [
        CorpusDocument("c1", "hoa binh", "train.parquet", 0),   # tones stripped
        CorpusDocument("c2", "hòa bình", "train.parquet", 1),   # the real match
    ]
    references = {"uitvsfc_derived_train": ["hòa bình"]}
    kept, report = screen_contamination(documents, references)
    assert report.excluded_document_ids == ("c2",)
    assert [d.document_id for d in kept] == ["c1"]
    # c1 reached tier 3 (same prefilter digest) but was not excluded
    assert placement_insensitive_digest("hoa binh") == placement_insensitive_digest("hòa bình")
    assert report.counters.prefilter_candidates == 2
    assert report.counters.full_canon_calls_for_corpus_candidates == 2


def test_equivalence_on_randomised_corpora():
    rng = random.Random(35422)
    pool = HAZARDS + random_strings(300, seed=999, max_len=25)
    for trial in range(25):
        refs = rng.sample(pool, 12)
        docs = [
            CorpusDocument(f"d{i}", rng.choice(pool), "train.parquet", i)
            for i in range(60)
        ]
        kept, report = screen_contamination(docs, {"uitvsfc_official_validation": refs})
        ref_kept, ref_ids, ref_digests = reference_screen(
            docs, {"uitvsfc_official_validation": refs}
        )
        assert list(report.excluded_document_ids) == ref_ids, trial
        assert list(report.excluded_digests) == ref_digests, trial
        assert [d.document_id for d in kept] == [d.document_id for d in ref_kept], trial


def test_an_empty_reference_set_excludes_nothing():
    documents, _, _ = build_corpus()
    kept, report = screen_contamination(documents, {"uitvsfc_derived_train": []})
    assert report.excluded_count == 0
    assert len(kept) == len(documents)


# ---------------------------------------------------------------------------
# TASK D -- algorithmic counters, not wall-clock
# ---------------------------------------------------------------------------
def test_full_canon_calls_scale_with_candidates_not_corpus_size():
    """The whole point: expensive work must not be O(corpus documents)."""
    references = {"uitvsfc_derived_train": ["Tôi đã đọc quyển sách này rồi"]}
    documents = [
        CorpusDocument(f"plain-{i}", f"Bài viết số {i} về lịch sử Việt Nam và thế giới",
                       "train.parquet", i)
        for i in range(5000)
    ]
    documents.append(CorpusDocument("hit", "Tôi đã đọc quyển sách này rồi", "train.parquet", 5000))

    _, report = screen_contamination(documents, references)
    counters = report.counters
    assert counters.corpus_documents_seen == 5001
    assert counters.full_canon_calls_for_corpus_candidates == counters.prefilter_candidates
    assert counters.prefilter_candidates <= 5, (
        f"{counters.prefilter_candidates} candidates from 5001 documents; the prefilter "
        "is not eliminating the expensive path"
    )
    assert counters.full_canon_calls_for_corpus_candidates < 0.01 * counters.corpus_documents_seen
    assert counters.full_canon_calls_for_reference_set == 1
    assert report.excluded_document_ids == ("hit",)


def test_the_length_guard_keeps_long_documents_off_the_nfd_path():
    references = {"uitvsfc_derived_train": ["ngắn"]}
    documents = [
        CorpusDocument(f"long-{i}", "Việt Nam là một quốc gia " * 100, "train.parquet", i)
        for i in range(500)
    ]
    _, report = screen_contamination(documents, references)
    assert report.counters.length_guard_skips == 500
    assert report.counters.cheap_prefilter_checks == 0
    assert report.counters.full_canon_calls_for_corpus_candidates == 0


def test_counters_are_reported_without_any_text():
    documents, references, _ = build_corpus()
    _, report = screen_contamination(documents, references)
    payload = report.to_dict()
    assert set(payload["counters"]) == {
        "corpus_documents_seen", "length_guard_skips", "cheap_prefilter_checks",
        "prefilter_candidates", "full_canon_calls_for_corpus_candidates",
        "opened_reference_examples", "full_canon_calls_for_reference_set",
    }
    assert all(isinstance(v, int) for v in payload["counters"].values())
    for text in references["uitvsfc_derived_train"]:
        assert text not in str(payload), "no UIT-VSFC text may reach the report"


# ---------------------------------------------------------------------------
# TASK E -- progress, and the unchanged boundary
# ---------------------------------------------------------------------------
def test_progress_callback_reports_counts_only():
    documents, references, _ = build_corpus()
    seen = []
    screen_contamination(documents, references, on_progress=lambda *a: seen.append(a))
    assert len(seen) == len(documents)
    assert seen[-1][0] == len(documents)
    assert all(isinstance(v, int) for triple in seen for v in triple)


def test_official_test_still_has_no_route():
    documents, _, _ = build_corpus()
    from unmark.stage1.corpus import CorpusContractViolation

    for forbidden in ("uitvsfc_official_test", "official_test", "test", "uitvsfc_test"):
        with pytest.raises(CorpusContractViolation, match="contamination screen refuses"):
            screen_contamination(documents, {forbidden: ["x"]})


def test_the_decision_is_still_made_by_canonical_digest():
    import ast
    import pathlib

    source = pathlib.Path("unmark/stage1/corpus.py").read_text(encoding="utf-8")
    fn = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "screen_contamination"
    )
    body = ast.unparse(fn)
    assert "canonical_digest(content)" in body
    assert "digest in reference" in body
    # the prefilters may only guard, never decide an exclusion
    assert "excluded_ids.append" in body
    decision_block = body[body.index("digest = canonical_digest(content)"):]
    assert "excluded_ids.append" in decision_block


def test_the_prefilter_strips_exactly_the_five_tone_marks():
    """A *selectivity* guard, not a correctness one -- and the distinction matters.

    Stripping MORE than the tone marks (say, letter-forming diacritics too) stays
    a valid necessary condition: it only creates additional collisions, which
    tier 3 then resolves. It is therefore **safe but slower**, and no equivalence
    test can catch it. Stripping LESS -- or using a non-canon-invariant transform
    such as NFC -- is **unsafe**, and the lemma tests above do catch that.

    This test pins the intended set so a silent widening shows up as an
    intentional change rather than as unexplained candidate growth.
    """
    from unmark.stage1.corpus import _TONE_TRANSLATION

    assert {chr(cp) for cp in _TONE_TRANSLATION} == set(TONE_MARKS)
    assert {f"U+{cp:04X}" for cp in _TONE_TRANSLATION} == {
        "U+0300", "U+0301", "U+0303", "U+0309", "U+0323"
    }
    # letter-forming diacritics must survive: they carry channel information
    for letter_mark in ("̂", "̆", "̛"):
        assert ord(letter_mark) not in _TONE_TRANSLATION
    assert placement_insensitive_digest("ăn") != placement_insensitive_digest("an")
    assert placement_insensitive_digest("đường") != placement_insensitive_digest("duong")
