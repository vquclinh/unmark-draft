"""Non-Vietnamese scripts must survive Stage-1 orthography (Audit 029 §AA).

The second real blocker was a 98-character region of 97 Hangul syllables plus
one quotation mark. It exposed **two independent defects**, and this file keeps
them independent (Task D), because fixing either alone would have left the other
live:

**A — over-broad protection.** `source_letter_runs` protected every maximal
`str.isalpha()` run, so 97 Hangul syllables became one indivisible "Vietnamese
candidate span" with a single legal cut at its end.

**B — RAW_BASE expansion.** `decompose` grouped units over `nfd(whole text)`.
Hangul NFD yields 2-3 Jamo of combining class **0**, so each Jamo became its own
unit with its own base: 98 source characters produced a 269-character base
stream and a RAW_BASE length of 271 against `max_length = 256`.

**No real corpus text appears here.** Every fixture is synthetic, constructed to
the reported shape; the real region's bytes are neither committed nor needed.
"""

from __future__ import annotations

import json
import sys
import pathlib
import unicodedata

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.orthography.canonical import canon, nfc, nfd
from unmark.orthography.decompose import (
    decompose,
    protects_a_vietnamese_candidate,
    recompose,
    source_letter_runs,
)
from unmark.orthography.units import split_units_with_offsets
from unmark.stage1.chunking import (
    ChunkingViolation,
    chunk_document,
    safe_cut_offsets,
    verify_tiles_source,
)
from unmark.stage1.corpus import CorpusDocument


# ---------------------------------------------------------------------------
# Synthetic fixtures, built to the reported shape
# ---------------------------------------------------------------------------
def hangul(count: int = 97) -> str:
    """`count` distinct precomposed Hangul syllables. Synthetic, not corpus text."""
    return "".join(chr(0xAC00 + (index * 37) % 11172) for index in range(count))


SCRIPTS = {
    "hangul": hangul(40),
    "cyrillic": "Привет мир Кириллица",
    "greek": "Ελληνικά γράμματα",
    "cjk": "日本語漢字仮名",
    "latin_non_vietnamese": "Müller naïve façade Ångström",
    "vietnamese_nfc": nfc("Tôi đã đọc một quyển sách"),
    "vietnamese_nfd": nfd("Tôi đã đọc một quyển sách"),
}


def document(text: str, doc_id: str = "doc-0") -> CorpusDocument:
    return CorpusDocument(
        document_id=doc_id, content=text, source_shard="train.parquet", source_row=0
    )


def lengths():
    """Injected length functions with a RAW_BASE path that grows with the base.

    Deliberately asymmetric: the base function measures `decompose(...).base_text`,
    so a base stream that explodes is *visible* to the chunker exactly as the
    real RAW_BASE pathway saw it.
    """
    reference = lambda t: len(canon(t)) // 3 + 2  # noqa: E731
    base = lambda t: len(decompose(t).base_text) // 3 + 2  # noqa: E731
    return reference, base


# ---------------------------------------------------------------------------
# Defect B -- RAW_BASE must not rewrite unrelated scripts
# ---------------------------------------------------------------------------
def test_hangul_is_nfc_and_nfd_expands_it():
    """The precondition the blocker rests on, stated as a fact not an assumption."""
    text = hangul(97)
    assert text == nfc(text), "fixture must be NFC"
    assert text == canon(text), "fixture must already be canonical"
    assert len(nfd(text)) > len(text), "NFD must expand Hangul"


def test_raw_base_does_not_expand_hangul_into_jamo():
    """Proposal §4.2 requires 'recomposition of the base'. 98 in, 98 out."""
    text = hangul(97)
    parts = decompose(text)
    assert len(parts.base_text) == len(text), (
        f"base stream expanded {len(text)} -> {len(parts.base_text)}"
    )
    assert parts.base_text == text, "Hangul carries no Vietnamese mark, so base == source"
    assert len(parts.units) == len(text), "one unit per Hangul syllable"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_base_stream_never_grows_beyond_its_source(name):
    """A base is a *stripping*; it must never be longer than what it strips."""
    text = SCRIPTS[name]
    assert len(decompose(text).base_text) <= len(canon(text)), name


@pytest.mark.parametrize("text", [
    "Müller", "naïve", "façade", "Ångström", "Přehled",
])
def test_marks_outside_the_vietnamese_sets_are_preserved_in_the_base(text):
    """The repository's own contract: `Müller` survives as `Müller`.

    Stated in `decompose`'s source comment and cross-checked by the G-1
    `base_signature` tests; reused here rather than reinvented. Diaeresis, ring,
    caron and cedilla are in neither Vietnamese mark set, so the base keeps them.
    """
    assert decompose(text).base_text == canon(text)


@pytest.mark.parametrize("text,expected", [
    ("señor", "senor"),      # U+0303 tilde IS the Vietnamese ngã tone mark
    ("café", "cafe"),        # U+0301 acute IS sắc
    ("règle", "regle"),      # U+0300 grave IS huyền
])
def test_marks_that_collide_with_vietnamese_are_stripped_and_that_is_UNCHANGED(text, expected):
    """A documented collision, pinned so §AA cannot be blamed for it.

    The Vietnamese tone marks are ordinary Unicode combining characters that
    other languages also use, so a Spanish `ñ` or a French `é` is stripped by the
    same rule that strips `ã` and `é` in Vietnamese. `dec` cannot distinguish
    them without language identification, which
    `unmark/linguistics/classify.py` forbids by design.

    **This behaviour is identical before and after the §AA repair** — verified
    against `1f86667` — and is recorded here so the collision is a known
    property rather than a surprise, and so a future change to it is deliberate.
    """
    assert decompose(text).base_text == expected


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_recompose_still_rebuilds_the_canonical_string(name):
    """The round-trip contract `rec(dec(x)) == canon(x)` must survive the repair."""
    text = SCRIPTS[name]
    assert recompose(decompose(text)) == canon(text), name


def test_vietnamese_base_still_strips_tone_and_letter_diacritics():
    """The repair must not make RAW_BASE a no-op for the script it exists for."""
    assert decompose("Đường").base_text == "Duong"
    assert decompose("Tôi đã đọc").base_text == "Toi da doc"
    assert decompose(nfd("hoà")).base_text == "hoa"


# ---------------------------------------------------------------------------
# Defect A -- protected span is Vietnamese-candidate, not any alphabetic run
# ---------------------------------------------------------------------------
def test_a_long_hangul_run_is_not_one_protected_span():
    text = hangul(97)
    assert source_letter_runs(text) == [], "Hangul cannot spell a Vietnamese syllable"
    interior = safe_cut_offsets(text) - {0, len(text)}
    assert len(interior) == len(text) - 1, "every syllable boundary is a legal cut"


@pytest.mark.parametrize("name,protected", [
    ("hangul", False), ("cyrillic", False), ("greek", False), ("cjk", False),
    ("latin_non_vietnamese", True), ("vietnamese_nfc", True), ("vietnamese_nfd", True),
])
def test_only_latin_script_runs_are_protected(name, protected):
    """Latin is wider than Vietnamese on purpose: over-protection only costs cuts."""
    assert bool(source_letter_runs(SCRIPTS[name])) is protected, name


def test_the_predicate_is_lexicon_free():
    """It must not consult the syllable inventory -- see the docstring's reasoning.

    `classify_candidate` answers inventory *membership* and returns
    NOT_APPLICABLE for a valid but OOV syllable, so using it here would permit a
    cut inside a genuine Vietnamese word.
    """
    import ast
    import inspect

    source = pathlib.Path(inspect.getsourcefile(protects_a_vietnamese_candidate)).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "protects_a_vietnamese_candidate"
    )
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(function) if isinstance(n, ast.Call)
    }
    for forbidden in ("classify_candidate", "is_vietnamese_candidate",
                      "contains_membership_form", "membership_form"):
        assert forbidden not in called, f"the cut predicate must not consult {forbidden}"


def test_hangul_chunks_end_to_end_with_both_pathways_within_limit():
    """The whole blocker, synthetically: it must now chunk, exactly and safely."""
    text = hangul(97) + '"'
    reference, base = lengths()
    chunks = chunk_document(document(text), "train", reference_length=reference,
                            base_length=base, max_length=16)
    assert len(chunks) > 1
    verify_tiles_source(chunks, text, "doc-0")
    assert "".join(c.text for c in chunks) == text, "byte-exact reconstruction"
    assert all(c.reference_length <= 16 and c.base_length <= 16 for c in chunks)
    boundaries = {u.start for u in split_units_with_offsets(text)} | {0, len(text)}
    assert all(c.source_start in boundaries for c in chunks)


# ---------------------------------------------------------------------------
# Task G -- Vietnamese protection is NOT weakened
# ---------------------------------------------------------------------------
VIETNAMESE = [
    "nghiêng", "khuỷu", "quyệt", "nguyễn", "hoà", "thuý", "khoẻ",
    "Đường", "đọc", "ưởng", "xoèn", "quãng",
    # Orthographically valid, deliberately obscure -- an inventory may not list
    # these, and they must be protected anyway.
    "ghiếc", "khuỵp", "nguyễnh", "thoắtl",
]


@pytest.mark.parametrize("word", VIETNAMESE)
@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_no_cut_falls_inside_a_vietnamese_candidate(word, form):
    text = unicodedata.normalize(form, word)
    runs = source_letter_runs(text)
    assert runs, f"{text!r} must be protected in {form}"
    for offset in safe_cut_offsets(text):
        for start, end in runs:
            assert not (start < offset < end), (offset, start, end, repr(text))


@pytest.mark.parametrize("word", VIETNAMESE)
def test_noncanonical_tone_placement_stays_protected(word):
    """`hòa` vs `hoà`: canon relocates the mark; protection must not depend on it."""
    for text in (word, nfd(word)):
        for offset in safe_cut_offsets(text):
            if 0 < offset < len(text):
                assert unicodedata.combining(text[offset]) == 0


def test_an_oversized_indivisible_vietnamese_candidate_still_fails_closed():
    """The narrowing must not turn fail-closed into fail-open."""
    text = "nghiêng" * 60          # one contiguous Latin run, no separator
    assert len(source_letter_runs(text)) == 1, "fixture must be one protected run"
    reference, base = lengths()
    with pytest.raises(ChunkingViolation) as caught:
        chunk_document(document(text), "train", reference_length=reference,
                       base_length=base, max_length=16)
    assert "indivisible" in str(caught.value)


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_source_is_never_normalised_or_rewritten(name):
    """Whatever the script, chunks are slices of the source, never a rewrite."""
    text = SCRIPTS[name]
    reference, base = lengths()
    try:
        chunks = chunk_document(document(text), "train", reference_length=reference,
                                base_length=base, max_length=24)
    except ChunkingViolation:
        return
    assert "".join(c.text for c in chunks) == text
    for chunk in chunks:
        assert chunk.text == text[chunk.source_start:chunk.source_end]


# ---------------------------------------------------------------------------
# Task H -- the reprobe emits metadata only, and cannot train
# ---------------------------------------------------------------------------
def test_the_blocker_probe_reports_metadata_and_never_corpus_text():
    """Every reported field must be a hash, a count or a flag -- never text."""
    from scripts.stage1_blocker_probe import build_report

    region = hangul(97) + '"'
    reference, base = lengths()
    # max_length chosen so the region splits in TWO, like the real blocker
    # (reference 100 / RAW_BASE 271 against 256), so the viable-single-cut
    # requirement is the one that actually applies.
    report = build_report(region, reference, base, max_length=20)

    assert report["status"] == "PASS", report["failures"]
    assert report["chunks"] == 2, report["chunks"]
    assert report["safe_interior_cuts"] >= 1
    assert report["viable_cuts_both_pathways"] >= 1
    assert report["reconstructs_byte_exact"] is True
    assert report["region_chars"] == len(region)

    # No field may contain a slice of the region.
    blob = json.dumps(report, ensure_ascii=False)
    for start in range(0, len(region) - 4):
        assert region[start:start + 5] not in blob, "corpus text leaked into the report"


def test_the_blocker_probe_cannot_load_an_encoder_or_step():
    import ast

    source = pathlib.Path("scripts/stage1_blocker_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    for forbidden in ("backward", "step", "zero_grad", "AdamW", "build_optimizer",
                      "AutoModel", "from_pretrained_model", "train"):
        assert forbidden not in called, f"the probe must not call {forbidden}()"
