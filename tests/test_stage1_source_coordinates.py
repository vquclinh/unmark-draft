"""Safe cut offsets must address the SOURCE, canonical or not (Audit 029 §Z).

The real blocker: Stage 6 failed closed on a 604-character, punctuation-rich
source region containing 85 alphabetic runs (longest 11 characters) that
differed from its canonical form at exactly two of its 604 positions. The old
`safe_cut_offsets` discarded **every** boundary in the region because of that
local difference, so a document that had ~600 legal cut points looked
indivisible.

**No real corpus text appears here.** Every fixture is constructed to the
*reported shape* of the real region; the real bytes are neither committed nor
needed.

Two properties are asserted, and they pull in opposite directions on purpose:

* canonical input must behave **exactly** as it did before the repair;
* non-canonical input must stop being globally unsplittable -- without
  "non-canonical" ever coming to mean "splittable anywhere".
"""

from __future__ import annotations

import random
import sys
import pathlib
import unicodedata

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.orthography.canonical import canon
from unmark.orthography.decompose import decompose, source_letter_runs
from unmark.orthography.units import split_units, split_units_with_offsets
from unmark.stage1.chunking import (
    ChunkingViolation,
    chunk_document,
    safe_cut_offsets,
    verify_tiles_source,
)
from unmark.stage1.corpus import CorpusDocument

SEPARATORS = "|_=-./"


# ---------------------------------------------------------------------------
# The pre-repair implementation, kept as a TEST ORACLE only
# ---------------------------------------------------------------------------
def canonical_coordinate_offsets(text: str, classifier=None) -> frozenset[int]:
    """Verbatim `2f6d024` `safe_cut_offsets`. Never a production pathway."""
    if not text:
        return frozenset()
    parts = decompose(text, eligibility_classifier=classifier)
    if parts.canonical_text != text:
        return frozenset()
    unsafe: set[int] = set()
    for span in parts.syllables:
        unsafe.update(range(span.canonical_start + 1, span.canonical_end))
    candidates = {unit.canonical_start for unit in parts.units}
    candidates.update((0, len(text)))
    return frozenset(candidates - unsafe)


# ---------------------------------------------------------------------------
# Fixtures built to the reported SHAPE of the real region
# ---------------------------------------------------------------------------
def punctuation_rich_body(runs: int = 84) -> str:
    """Many short alphabetic runs joined by `| _ = - . /`, no whitespace."""
    words = [f"seg{index:02d}abcdef"[: 3 + (index % 9)] for index in range(runs)]
    return "".join(w + SEPARATORS[i % len(SEPARATORS)] for i, w in enumerate(words))


def noncanonical_tail() -> str:
    """A base + combining mark whose canonical form differs from the source."""
    return "ho" + unicodedata.normalize("NFD", "ạ")


@pytest.fixture
def blocker_shaped() -> str:
    text = punctuation_rich_body() + noncanonical_tail()
    assert canon(text) != text, "fixture must be non-canonical to be the right shape"
    assert not any(c.isspace() for c in text), "fixture must contain no whitespace"
    return text


def lengths(limit: int = 8):
    """Injected length functions: a chunk fits while it is short enough."""
    return (lambda t: len(t) // 4 + 2), (lambda t: len(t) // 4 + 2)


def document(text: str, doc_id: str = "doc-0") -> CorpusDocument:
    return CorpusDocument(
        document_id=doc_id, content=text, source_shard="train.parquet", source_row=0
    )


# ---------------------------------------------------------------------------
# Task C -- canonical input is byte-for-byte unchanged
# ---------------------------------------------------------------------------
def deterministic_corpus() -> list[str]:
    seeds = [
        "Tôi đã đọc một quyển sách", "Đội_tuyển_bóng_đá_quốc_gia", "hoà bình",
        "Chincha|Alta=x-y.z/w", "Müller", "Đường", "a|b_c-d.e/f=g", "xin chào",
        "Quần_đảo_Hoàng_Sa", "ABC123", "a", "|", "  spaced  out  ", "tab\there",
        "new\nline", "đđđ", "ệệệ", "https://example.com/path?q=1&r=2",
        "mixed ĐƯỜNG and Müller", "1.2.3.4", "__init__", "α β γ", "日本語",
        "CamelCase|snake_case|kebab-case", punctuation_rich_body(12),
    ]
    alphabet = "aáàảãạăâeéèêiíoóôơuưyđĐ|_=-./ 0123456789ÀÁÂÃÈÉÊÌ"
    rng = random.Random(20260823)
    corpus = list(seeds)
    for _ in range(2_000):
        corpus.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40))))
    for form in ("NFC", "NFD"):
        corpus.extend(unicodedata.normalize(form, s) for s in seeds)
    return corpus


def is_latin_only(text: str) -> bool:
    """No letter outside the Latin script -- the population §Z's oracle covers."""
    return all(
        not ch.isalpha() or unicodedata.name(ch, "").startswith("LATIN ")
        for ch in unicodedata.normalize("NFD", text)
    )


def test_canonical_latin_input_is_identical_to_the_pre_repair_implementation():
    """The §Z repair was about non-canonical sources. Canonical Latin must not move.

    Scoped to Latin in Audit 029 §AA. The §AA repair deliberately narrows the
    protected span from "alphabetic in any script" to "Latin script", so a
    canonical *Greek* or *CJK* string legitimately gains cuts the old oracle
    refused. That divergence is the repair; asserting the old equality on it
    would be asserting the defect. Latin -- and therefore all Vietnamese -- is
    still required to be byte-for-byte identical.
    """
    canonical = [t for t in deterministic_corpus() if t and canon(t) == t and is_latin_only(t)]
    assert len(canonical) > 500, f"weak fixture set: {len(canonical)}"
    mismatches = [t for t in canonical if canonical_coordinate_offsets(t) != safe_cut_offsets(t)]
    assert mismatches == [], mismatches[:3]


def test_narrowing_only_ever_adds_cuts_and_never_removes_one():
    """The §AA change is a *relaxation*: it can offer more boundaries, never fewer.

    Stated as a superset rather than a count, so the property holds whatever the
    script mix of the fixture happens to be.
    """
    for text in deterministic_corpus():
        if not text or canon(text) != text:
            continue
        assert canonical_coordinate_offsets(text) <= safe_cut_offsets(text), repr(text)


def test_canonical_chunk_boundaries_are_unchanged():
    reference, base = lengths()
    for index, text in enumerate(t for t in deterministic_corpus() if t and canon(t) == t):
        if index > 300:
            break
        try:
            chunks = chunk_document(document(text), "train", reference_length=reference,
                                    base_length=base, max_length=16)
        except ChunkingViolation:
            continue
        assert "".join(c.text for c in chunks) == text


# ---------------------------------------------------------------------------
# Task D -- the non-canonical source contract
# ---------------------------------------------------------------------------
def test_the_defect_shape_is_reproduced_and_repaired(blocker_shaped):
    """Old: nothing is cuttable. New: many legal source boundaries."""
    old = canonical_coordinate_offsets(blocker_shaped)
    new = safe_cut_offsets(blocker_shaped)
    interior = lambda s: s - {0, len(blocker_shaped)}  # noqa: E731
    assert interior(old) == frozenset(), "the oracle must reproduce the defect"
    assert len(interior(new)) > 100, len(interior(new))


def test_returned_offsets_address_the_original_source(blocker_shaped):
    """Every offset must be a real index into `text`, not into `canon(text)`."""
    for offset in safe_cut_offsets(blocker_shaped):
        assert 0 <= offset <= len(blocker_shaped)


def test_no_cut_falls_inside_a_source_alphabetic_span(blocker_shaped):
    runs = source_letter_runs(blocker_shaped)
    assert runs, "fixture should contain alphabetic runs"
    for offset in safe_cut_offsets(blocker_shaped):
        for start, end in runs:
            assert not (start < offset < end), (offset, start, end)


def test_no_cut_falls_inside_a_character_unit(blocker_shaped):
    boundaries = {u.start for u in split_units_with_offsets(blocker_shaped)}
    boundaries.update((0, len(blocker_shaped)))
    assert safe_cut_offsets(blocker_shaped) <= boundaries


def test_a_noncanonical_document_now_chunks_with_exact_tiling(blocker_shaped):
    reference, base = lengths()
    chunks = chunk_document(document(blocker_shaped), "train", reference_length=reference,
                            base_length=base, max_length=24)
    assert len(chunks) > 1
    verify_tiles_source(chunks, blocker_shaped, "doc-0")
    assert "".join(c.text for c in chunks) == blocker_shaped
    assert all(c.reference_length <= 24 and c.base_length <= 24 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [f"doc-0#{i}" for i in range(len(chunks))]


def test_repeated_execution_is_deterministic(blocker_shaped):
    reference, base = lengths()
    runs = [
        [(c.chunk_id, c.text, c.source_start, c.source_end)
         for c in chunk_document(document(blocker_shaped), "train",
                                 reference_length=reference, base_length=base,
                                 max_length=24)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_document_order_does_not_change_the_result(blocker_shaped):
    reference, base = lengths()
    others = [document(punctuation_rich_body(10), "a"), document("xin chào", "b")]
    target = document(blocker_shaped, "target")

    def chunk_all(docs):
        out = {}
        for d in docs:
            out[d.document_id] = [
                (c.chunk_id, c.text) for c in
                chunk_document(d, "train", reference_length=reference,
                               base_length=base, max_length=24)
            ]
        return out

    forward = chunk_all([*others, target])
    backward = chunk_all([target, *reversed(others)])
    assert forward["target"] == backward["target"]


def test_an_indivisible_noncanonical_span_still_fails_closed():
    """Non-canonical must NOT come to mean 'always splittable'."""
    text = "a" * 400 + unicodedata.normalize("NFD", "ạ")
    assert canon(text) != text
    assert len(source_letter_runs(text)) == 1, "fixture must be one contiguous run"
    reference, base = lengths()
    with pytest.raises(ChunkingViolation) as caught:
        chunk_document(document(text), "train", reference_length=reference,
                       base_length=base, max_length=16)
    assert "indivisible" in str(caught.value)


# ---------------------------------------------------------------------------
# Task E -- combining-sequence safety, canonical and not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    unicodedata.normalize("NFD", "Tôi đã đọc"),
    unicodedata.normalize("NFD", "Đường|Müller_hoà"),
    "a" + "̣" + "|b" + "́" + "_c",
    "́leading-mark",
    unicodedata.normalize("NFD", "ệ") * 20,
    "x" + "̣̀" + "|y",
    punctuation_rich_body(6) + unicodedata.normalize("NFD", "ạ"),
])
def test_a_cut_never_separates_a_base_from_its_combining_marks(text):
    """Holds whether or not the source is canonical."""
    unit_starts = {u.start for u in split_units_with_offsets(text)}
    unit_starts.update((0, len(text)))
    offsets = safe_cut_offsets(text)
    assert offsets <= unit_starts
    for offset in offsets:
        if 0 < offset < len(text):
            assert unicodedata.combining(text[offset]) == 0, (offset, repr(text))


# ---------------------------------------------------------------------------
# Task F -- canonicalisation CAN change codepoint count; do not assume it cannot
# ---------------------------------------------------------------------------
def test_canonicalisation_can_change_codepoint_count():
    """Recorded as fact, not assumption: NFC composition shortens NFD input."""
    nfd_text = unicodedata.normalize("NFD", "é")
    assert len(nfd_text) == 2 and len(canon(nfd_text)) == 1


def test_safe_cuts_stay_correct_when_canon_changes_the_length():
    """The real blocker happened to be 604 -> 604. Correctness must not rely on that."""
    text = punctuation_rich_body(40) + unicodedata.normalize("NFD", "ạ")
    assert len(canon(text)) != len(text), "fixture must change length under canon"
    offsets = safe_cut_offsets(text)
    assert offsets, "a length-changing source must still offer cuts"
    for offset in offsets:
        assert 0 <= offset <= len(text)
    for start, end in source_letter_runs(text):
        assert not any(start < o < end for o in offsets)
    reference, base = lengths()
    chunks = chunk_document(document(text), "train", reference_length=reference,
                            base_length=base, max_length=24)
    assert "".join(c.text for c in chunks) == text


# ---------------------------------------------------------------------------
# The shared unitisation primitive really is shared
# ---------------------------------------------------------------------------
def test_split_units_is_a_projection_of_the_offset_version():
    """One grouping rule, two views -- so they cannot drift apart."""
    for text in deterministic_corpus()[:400]:
        nfd_text = unicodedata.normalize("NFD", text)
        projected = [(u.base, u.marks) for u in split_units_with_offsets(nfd_text)]
        assert split_units(nfd_text) == projected


def test_offset_units_tile_the_source_exactly():
    for text in deterministic_corpus()[:400]:
        units = split_units_with_offsets(text)
        assert "".join(u.text for u in units) == text
        cursor = 0
        for unit in units:
            assert unit.start == cursor
            cursor = unit.end
        assert cursor == len(text)


def test_protected_runs_agree_with_decompose_on_canonical_latin_text():
    """On Latin text the two segmentations are the same rule, as before §AA."""
    for text in deterministic_corpus():
        if not text or canon(text) != text or not is_latin_only(text):
            continue
        expected = [(s.canonical_start, s.canonical_end) for s in decompose(text).syllables]
        assert source_letter_runs(text) == expected, repr(text)


def test_protected_runs_are_contained_in_the_alphabetic_runs():
    """The general relationship after §AA: protected ⊆ alphabetic, not equal.

    `SyllableSpan` answers "where are the alphabetic runs?" for channel
    metadata. `source_letter_runs` answers "what may a cut never bisect?" and is
    narrower. Containment is the invariant that must hold for every script; the
    Hangul blocker is precisely a case where equality does not.
    """
    for text in deterministic_corpus():
        if not text or canon(text) != text:
            continue
        alphabetic = [(s.canonical_start, s.canonical_end) for s in decompose(text).syllables]
        for start, end in source_letter_runs(text):
            assert any(a <= start and end <= b for a, b in alphabetic), (
                f"protected run [{start},{end}) escapes every alphabetic run in {text!r}"
            )


def test_no_second_parser_was_introduced():
    """`source_letter_runs` must query the shared primitives, not re-implement them."""
    import ast
    import inspect

    # `unmark.orthography` re-exports the *function* `decompose`, which shadows
    # the submodule, so the defining file is located from the function itself.
    source = pathlib.Path(inspect.getsourcefile(source_letter_runs)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "source_letter_runs")
    called = {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
              for n in ast.walk(function) if isinstance(n, ast.Call)}
    assert "split_units_with_offsets" in called, called
    body = ast.dump(function)
    for forbidden in ("compile", "findall", "match", "search"):
        assert forbidden not in called, f"{forbidden} suggests a second parser"


# ---------------------------------------------------------------------------
# Task G -- what the runner's length actually measures (282 vs 283)
# ---------------------------------------------------------------------------
class ToneSensitiveTokenizer:
    """A double whose token count depends on the exact codepoint sequence.

    Real BPE does too: relocating a tone mark changes the byte sequence and can
    change how it segments. That is the whole point of the fixture.
    """

    all_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

    def get_added_vocab(self):
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    # A greedy longest-match subword vocabulary. This is what makes real BPE
    # position-sensitive: moving a tone mark changes which merges apply, so the
    # same characters in a different order can segment into a different number
    # of pieces.
    VOCABULARY = ("ho", "họ", "oa", "seg", "abc")

    def tokenize(self, text):
        import re as _re

        out = []
        for match in _re.finditer(r"\S+\n?", text):
            run, cursor = match.group(0), 0
            while cursor < len(run):
                for piece in sorted(self.VOCABULARY, key=len, reverse=True):
                    if run.startswith(piece, cursor):
                        out.append(piece)
                        cursor += len(piece)
                        break
                else:
                    out.append(run[cursor])
                    cursor += 1
        return out

    def convert_tokens_to_ids(self, tokens):
        return [len(t) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


def test_the_runner_length_measures_canon_of_the_text_not_the_raw_source():
    """The 282-vs-283 gap: two different quantities, both intended.

    `reference_length(x)` is defined on **`canon(x)`** -- the Stage-1 reference
    pathway is canonical text. A diagnostic that calls `tokenizer(x)` on the raw
    source measures something else whenever `canon(x) != x`, which is exactly
    the case for the real blocker region (its canonical form relocates a tone
    mark at the tail). Neither number is wrong; they are not the same
    measurement, and the runner's is the contract.
    """
    from unmark.stage1.lengths import (
        RunLengthComposer,
        TransformCounters,
        build_length_functions,
    )

    tokenizer = ToneSensitiveTokenizer()
    reference_length, base_length, _ = build_length_functions(tokenizer)
    oracle = RunLengthComposer(tokenizer, counters=TransformCounters())

    source = "ho" + "ọ" + "a"
    assert canon(source) != source, "fixture must be non-canonical"

    # The contract: the runner's reference length IS the authoritative length of
    # the CANONICAL text.
    assert reference_length(source) == oracle.authoritative_length(canon(source))

    # The base pathway measures its own transform, not the source either.
    assert base_length(source) == oracle.authoritative_length(
        decompose(canon(source)).base_text
    )

    # And a diagnostic run over the RAW source is a different measurement. The
    # fixture is chosen so the two genuinely disagree, which is the 282-vs-283
    # gap in miniature.
    assert oracle.authoritative_length(source) != reference_length(source)

