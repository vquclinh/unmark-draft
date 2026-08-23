"""Stage-6 length composition: exact equivalence and algorithmic reduction.

ML-free. Stage 6 fed every document through `canon`/`decompose` ~250x its own
length (Audit 029 §R). The repair memoises the orthographic transforms and
composes token counts per non-whitespace run, and must produce **bit-identical
chunk boundaries**.

Two properties are load-bearing and both are tested here:

* the **composability lemma** -- `canon` and `base_text` compose across
  whitespace-run boundaries -- a property of this repository's own code;
* the **per-chunk tokenization fact** (D-B3B1B-001) -- verified at runtime, and
  the verifier itself is tested against a deliberately non-conforming tokenizer.

Nothing here assumes token counts are monotone in text length; the
`NON_MONOTONIC` oracle family exists precisely to prove that.
"""

from __future__ import annotations

import random
import re
import unicodedata as ud

import pytest

from unmark.orthography import canon, decompose
from unmark.stage1.chunking import ChunkingViolation, chunk_document
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.corpus import CorpusDocument
from unmark.stage1.lengths import (
    DEFAULT_VERIFY_FIRST,
    ComposedTransforms,
    build_length_functions,
)

SEGMENT = re.compile(r"\s+|\S+")


# ---------------------------------------------------------------------------
# The composability lemma
# ---------------------------------------------------------------------------
def segments(text):
    return [m.group(0) for m in SEGMENT.finditer(text)]


LEMMA_CASES = [
    "", " ", "\t", "\n", "  ", "hoà bình", "hòa bình", "Tôi  đã\tđọc\nrồi",
    "  leading", "trailing  ", "\n\tmixed \t\n whitespace \n\n",
    "Việt Nam là một quốc gia", "đường ĐƯỜNG Đ đ", "Müller café naïve",
    "https://vi.wikipedia.org/wiki/Việt_Nam a@b.vn", "1234 !@#$ ...",
    "thúy thuý khỏe khoẻ", "Tôi dùng Python và PyTorch",
    "ế" * 30 + " " + "ế" * 30,
]
LEMMA_CASES += [ud.normalize("NFD", c) for c in LEMMA_CASES]


@pytest.mark.parametrize("text", LEMMA_CASES)
def test_canon_composes_across_whitespace_segments(text):
    assert "".join(canon(s) for s in segments(text)) == canon(text)


@pytest.mark.parametrize("text", LEMMA_CASES)
def test_base_text_composes_across_whitespace_segments(text):
    direct = decompose(canon(text)).base_text
    composed = "".join(decompose(canon(s)).base_text for s in segments(text))
    assert composed == direct


def test_the_lemma_holds_on_a_large_random_population():
    rng = random.Random(51733)
    alphabet = "aeiouyăâêôơưbcdghklmnpqrstvxĐđ" + "̛̣̀́̃̉̂̆" + " \t\n.,-_/"
    bad_canon, bad_base = [], []
    for _ in range(6000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        if "".join(canon(s) for s in segments(text)) != canon(text):
            bad_canon.append(text)
        direct = decompose(canon(text)).base_text
        if "".join(decompose(canon(s)).base_text for s in segments(text)) != direct:
            bad_base.append(text)
    assert not bad_canon, bad_canon[:2]
    assert not bad_base, bad_base[:2]


@pytest.mark.parametrize("text", LEMMA_CASES)
def test_composed_transforms_equal_the_direct_calls(text):
    transforms = ComposedTransforms()
    assert transforms.canonical(text) == canon(text)
    assert transforms.base(text) == decompose(canon(text)).base_text


# ---------------------------------------------------------------------------
# Tokenizer doubles -- including a deliberately NON-MONOTONIC one
# ---------------------------------------------------------------------------
class WordTokenizer:
    """Whitespace-then-BPE-per-chunk, like the pinned PhoBERT slow tokenizer."""

    def tokenize(self, text):
        out = []
        for run in text.split():
            out.extend([run[i:i + 3] for i in range(0, len(run), 3)] or [run])
        return out

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def build_inputs_with_special_tokens(self, ids):
        return ["<s>", *ids, "</s>"]


class NonMonotonicTokenizer(WordTokenizer):
    """Extending the text can REDUCE the count -- monotonicity must not be used.

    A run of exactly the sentinel length collapses to one token, so appending
    characters to a shorter run can shrink the total.
    """

    def tokenize(self, text):
        out = []
        for run in text.split():
            out.extend(["<merged>"] if len(run) == 7 else
                       [run[i:i + 3] for i in range(0, len(run), 3)] or [run])
        return out


class NonComposingTokenizer(WordTokenizer):
    """Violates D-B3B1B-001: adds a token that depends on the whole string."""

    def tokenize(self, text):
        base = super().tokenize(text)
        return base + (["<extra>"] if len(text.split()) > 2 else [])


def old_length_functions(tokenizer):
    """The pre-optimisation implementation. **Test oracle only.**"""

    def whole(text):
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        return len(tokenizer.build_inputs_with_special_tokens(list(ids)))

    return (lambda t: whole(canon(t)),
            lambda t: whole(decompose(canon(t)).base_text))


# ---------------------------------------------------------------------------
# Output equivalence oracle
# ---------------------------------------------------------------------------
def doc(doc_id, content, shard="train.parquet", row=0):
    return CorpusDocument(doc_id, content, shard, row)


TITLE = "Đội_tuyển_bóng_đá_quốc_gia_Afghanistan"

CORPUS = {
    "short-one-chunk": "Tôi đã đọc",
    "many-short-segments": " ".join(f"từ{i}" for i in range(400)),
    "many-chunks": " ".join("Việt Nam là một quốc gia".split() * 300),
    "multiple-spaces": "alpha  beta   gamma    delta",
    "tabs-newlines": "alpha\tbeta\n\ngamma\r\ndelta",
    "leading-trailing": "   padded text   ",
    "punctuation": "Tôi, đã; đọc: quyển! sách? này. rồi...",
    "vietnamese": "Giảng viên dạy dễ hiểu nhưng đề thi hơi khó so với nội dung " * 20,
    "mixed": "Tôi dùng Python và PyTorch cho Deep Learning tại Việt Nam " * 20,
    "nfc": ud.normalize("NFC", "Tôi đã đọc quyển sách này rồi " * 30),
    "nfd": ud.normalize("NFD", "Tôi đã đọc quyển sách này rồi " * 30),
    "oversized-unit": "_".join([TITLE] * 12),
    "oversized-in-text": "mở đầu " + "_".join([TITLE] * 12) + " kết thúc",
    "interior-fallback": "-".join([f"Section{i}.Sub{i}" for i in range(80)]),
    "atomic-oversized": "Nghiêng" * 60,
    "urls": "Xem https://vi.wikipedia.org/wiki/Việt_Nam và a@b.vn để biết thêm " * 15,
}


def compare(document, partition, ref_old, base_old, ref_new, base_new, max_length):
    def run(r, b):
        try:
            chunks = chunk_document(document, partition, reference_length=r,
                                    base_length=b, max_length=max_length)
            return ("ok", [(c.chunk_id, c.document_id, c.partition, c.chunk_index,
                            c.text, c.source_start, c.source_end,
                            c.reference_length, c.base_length) for c in chunks])
        except ChunkingViolation as error:
            return ("raised", str(error))

    return run(ref_old, base_old), run(ref_new, base_new)


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize("tokenizer_cls", [WordTokenizer, NonMonotonicTokenizer])
@pytest.mark.parametrize("max_length", [16, 40, 256])
def test_optimized_output_is_identical_to_the_old_implementation(
    name, tokenizer_cls, max_length
):
    document = doc(name, CORPUS[name], row=7)
    tokenizer = tokenizer_cls()
    ref_old, base_old = old_length_functions(tokenizer)
    ref_new, base_new, _ = build_length_functions(tokenizer_cls())
    old, new = compare(document, "dev", ref_old, base_old, ref_new, base_new, max_length)
    assert old == new, f"{name}/{tokenizer_cls.__name__}/{max_length}"


@pytest.mark.parametrize("tokenizer_cls", [WordTokenizer, NonMonotonicTokenizer])
def test_asymmetric_paths_both_directions(tokenizer_cls):
    """base longer than reference, and reference longer than base."""
    document = doc("asym", CORPUS["vietnamese"])
    tokenizer = tokenizer_cls()
    ref_old, base_old = old_length_functions(tokenizer)
    ref_new, base_new, _ = build_length_functions(tokenizer_cls())

    for pad_ref, pad_base in ((0, 20), (20, 0)):
        old, new = compare(
            document, "train",
            lambda t: ref_old(t) + pad_ref, lambda t: base_old(t) + pad_base,
            lambda t: ref_new(t) + pad_ref, lambda t: base_new(t) + pad_base,
            256,
        )
        assert old == new, (pad_ref, pad_base)


def test_equivalence_on_randomised_documents():
    rng = random.Random(19225)
    pieces = ["Việt", "Nam", "hoà", "hòa", "đường", "Python", "a@b.vn", "1234",
              TITLE, "Nghiêng", "café", "x" * 40]
    seps = [" ", "  ", "\t", "\n", "_", "-", ""]
    for trial in range(40):
        content = "".join(
            rng.choice(pieces) + rng.choice(seps) for _ in range(rng.randint(1, 60))
        )
        if not content.strip():
            continue
        document = doc(f"r{trial}", content, row=trial)
        tokenizer_cls = rng.choice([WordTokenizer, NonMonotonicTokenizer])
        ref_old, base_old = old_length_functions(tokenizer_cls())
        ref_new, base_new, _ = build_length_functions(tokenizer_cls())
        old, new = compare(document, "train", ref_old, base_old, ref_new, base_new,
                           rng.choice([24, 64, 256]))
        assert old == new, (trial, content[:60])


def test_violation_provenance_is_identical_too():
    document = doc("atomic", CORPUS["atomic-oversized"], shard="test.parquet", row=41)
    tokenizer = WordTokenizer()
    ref_old, base_old = old_length_functions(tokenizer)
    ref_new, base_new, _ = build_length_functions(WordTokenizer())
    old, new = compare(document, "train", ref_old, base_old, ref_new, base_new, 16)
    assert old[0] == new[0] == "raised"
    assert old[1] == new[1], "violation message and provenance must match exactly"


# ---------------------------------------------------------------------------
# The runtime verifier
# ---------------------------------------------------------------------------
def test_a_non_composing_tokenizer_now_FAILS_CLOSED():
    """Revision 3b restores composition, so a non-conforming tokenizer must raise.

    Replaces the Revision-3a test that asserted such a tokenizer was *harmless*.
    That was true only because 3a had removed composition entirely; 3b composes
    over the tokenizer's own run unit, so a tokenizer that does not decompose
    that way is a contract violation and Stage-1 must refuse to chunk.
    """
    ref, _, _ = build_length_functions(NonComposingTokenizer())
    with pytest.raises(Stage1ContractViolation, match="run composition disagreed"):
        ref("một hai ba bốn năm")


def test_the_optimized_length_equals_the_authoritative_length_by_construction():
    """For tokenizers that decompose as `PHOBERT_RUN` describes -- like the
    pinned one -- the optimised number is the authoritative number."""
    for tokenizer_cls in (WordTokenizer, PhoBERTShapedTokenizer):
        authoritative_ref, authoritative_base = old_length_functions(tokenizer_cls())
        opt_ref, opt_base, _ = build_length_functions(tokenizer_cls())
        for text in LEMMA_CASES:
            assert opt_ref(text) == authoritative_ref(text), (tokenizer_cls, text)
            assert opt_base(text) == authoritative_base(text), (tokenizer_cls, text)


def test_a_non_monotonic_tokenizer_is_still_fine_when_it_composes():
    """No monotonicity is assumed; only per-run decomposition is used."""
    authoritative_ref, _ = old_length_functions(NonMonotonicTokenizer())
    ref, _, _ = build_length_functions(NonMonotonicTokenizer())
    for text in LEMMA_CASES:
        assert ref(text) == authoritative_ref(text), repr(text)


def test_the_transform_verifier_still_fails_closed():
    """Task E: the remaining optimisation keeps a fail-closed verifier."""
    import unmark.stage1.lengths as lengths

    transforms = ComposedTransforms(verify_first=8)
    original = lengths.canon
    try:
        # a deliberately wrong per-segment transform must be caught
        lengths.canon = lambda t, *a, **k: original(t) + ("X" if t.strip() else "")
        with pytest.raises(Stage1ContractViolation, match="composed transform disagreed"):
            transforms.canonical("một hai ba")
    finally:
        lengths.canon = original


def test_the_verifier_accepts_a_correct_composition():
    ref, base, transforms = build_length_functions(WordTokenizer())
    for text in LEMMA_CASES:
        ref(text)
        base(text)
    assert transforms.counters.verifications > 0


# ---------------------------------------------------------------------------
# Algorithmic reduction -- counters, never wall-clock
# ---------------------------------------------------------------------------
def test_expensive_work_is_linear_in_document_not_quadratic_in_segments():
    content = " ".join("Việt Nam là một quốc gia".split() * 400)
    document = doc("big", content)
    ref, base, transforms = build_length_functions(WordTokenizer())
    chunk_document(document, "train", reference_length=ref, base_length=base,
                   max_length=256)
    counters = transforms.counters

    assert counters.length_queries > 100, "the fixture must exercise the greedy scan"
    # canon/decompose run on unique segments only, not on every growing prefix
    assert counters.characters_canonicalised < len(content), (
        f"canonicalised {counters.characters_canonicalised} chars for a "
        f"{len(content)}-char document"
    )
    # the growing prefix is extended, not rescanned
    assert counters.incremental_extensions > 10 * counters.full_rescans
    # segments examined stay proportional to the document, not to queries x segments
    assert counters.segments_seen < 8 * len(content.split()), (
        f"segments_seen {counters.segments_seen} is quadratic-looking for "
        f"{len(content.split())} words"
    )


def test_the_transform_memo_is_reused_across_documents():
    ref, base, transforms = build_length_functions(WordTokenizer())
    text = " ".join("Việt Nam là một quốc gia".split() * 50)
    for i in range(5):
        chunk_document(doc(f"d{i}", text), "train", reference_length=ref,
                       base_length=base, max_length=256)
    counters = transforms.counters
    assert counters.canon_cache_hits > counters.canon_calls
    assert counters.base_cache_hits > counters.decompose_calls


def test_counters_carry_no_text():
    _, _, transforms = build_length_functions(WordTokenizer())
    payload = transforms.counters.to_dict()
    assert all(isinstance(v, int) for v in payload.values())


# ===========================================================================
# Revision 3b: composition over the tokenizer's OWN run unit
# ===========================================================================
from unmark.stage1.lengths import PHOBERT_RUN, RunLengthComposer  # noqa: E402

NAIVE_RUN = re.compile(r"\S+")


class PhoBERTShapedTokenizer:
    """Faithful to `PhobertTokenizer._tokenize`: runs are ``\\S+\\n?``.

    BPE's end-of-word marker lands on the run's LAST character, so a run that
    ends in a newline costs more tokens than the same word without it. That is
    the behaviour that made naive ``\\S+`` composition wrong.
    """

    VOCAB = {"alpha", "beta", "gamma", "delta", "Tôi", "đã", "đọc", "một", "hai", "ba"}

    def bpe(self, token):
        if token in self.VOCAB:
            return token
        if token.endswith("\n") and token[:-1] in self.VOCAB:
            return f"{token[:-1]}@@ nl_a nl_b"
        return " ".join(token[i:i + 4] for i in range(0, len(token), 4)) or token

    def tokenize(self, text):
        out = []
        for run in PHOBERT_RUN.findall(text):
            out.extend(self.bpe(run).split(" "))
        return out

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def build_inputs_with_special_tokens(self, ids):
        return ["<s>", *ids, "</s>"]


NEWLINE_CASES = [
    "Tôi\nđã\nđọc", "Tôi đã đọc\n", "Tôi đã\nđọc", "một\n", "\nmột",
    "a\tb", "a  b", "a\r\nb", "  lead", "trail  ", "", " ", "\n", "\n\n",
    "x\n\n\ny", "alpha  beta\tgamma\n\ndelta   ",
]


@pytest.mark.parametrize("text", NEWLINE_CASES)
def test_the_run_unit_is_the_tokenizers_own_not_naive_non_whitespace(text):
    """`\\S+\\n?`, never `\\S+`. Token LIST equality, not just counts."""
    tokenizer = PhoBERTShapedTokenizer()
    whole = tokenizer.tokenize(text)
    exact = [t for run in PHOBERT_RUN.findall(text) for t in tokenizer.bpe(run).split(" ")]
    assert exact == whole, "composition over the tokenizer's own runs must be exact"


def test_naive_non_whitespace_composition_is_demonstrably_wrong():
    """The bb50823 defect, pinned so it cannot be reintroduced."""
    tokenizer = PhoBERTShapedTokenizer()
    offenders = []
    for text in NEWLINE_CASES:
        whole = tokenizer.tokenize(text)
        naive = [t for run in NAIVE_RUN.findall(text) for t in tokenizer.bpe(run).split(" ")]
        if naive != whole:
            offenders.append(text)
    assert offenders, "the fixture set must contain cases where naive composition fails"
    assert any("\n" in text for text in offenders)


def test_the_historical_five_versus_seven_is_reproduced_and_repaired():
    """Audit 029 §T forensics: `composed 5, exact 7` was the `\\S+` defect.

    The prefix is from the historical probe's own "whitespace" fixture.
    """
    tokenizer = PhoBERTShapedTokenizer()
    piece = "alpha  beta\tgamma\n\n"
    specials = 2

    naive = specials + sum(
        len(tokenizer.tokenize(run)) for run in NAIVE_RUN.findall(canon(piece))
    )
    exact = len(tokenizer.build_inputs_with_special_tokens(
        list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(canon(piece))))
    ))
    assert (naive, exact) == (5, 7), "must reproduce the reported numbers"

    ref, _, _ = build_length_functions(PhoBERTShapedTokenizer())
    assert ref(piece) == exact == 7, "the repaired composer must match authoritative"


@pytest.mark.parametrize("text", NEWLINE_CASES + LEMMA_CASES)
def test_optimized_equals_authoritative_on_both_pathways(text):
    tokenizer = PhoBERTShapedTokenizer()
    authoritative_ref, authoritative_base = old_length_functions(tokenizer)
    ref, base, _ = build_length_functions(PhoBERTShapedTokenizer())
    assert ref(text) == authoritative_ref(text), repr(text)
    assert base(text) == authoritative_base(text), repr(text)


def test_special_tokens_come_from_the_authoritative_api_not_a_constant():
    class ThreeSpecials(PhoBERTShapedTokenizer):
        def build_inputs_with_special_tokens(self, ids):
            return ["<s>", "<x>", *ids, "</s>"]

    ref, _, _ = build_length_functions(ThreeSpecials())
    authoritative_ref, _ = old_length_functions(ThreeSpecials())
    for text in ("Tôi đã đọc", "một\n", ""):
        assert ref(text) == authoritative_ref(text), text


def test_incremental_extension_handles_a_run_gaining_a_newline():
    """`"gamma"` -> `"gamma\\n"` changes the LAST run; it must be recomputed."""
    tokenizer = PhoBERTShapedTokenizer()
    authoritative_ref, _ = old_length_functions(tokenizer)
    ref, _, _ = build_length_functions(PhoBERTShapedTokenizer())
    growing = "alpha  beta\tgamma\n\ndelta   "
    for end in range(1, len(growing) + 1):
        piece = growing[:end]
        assert ref(piece) == authoritative_ref(piece), repr(piece)


def test_the_run_verifier_fails_closed_on_a_corrupted_run_count():
    """Task E: a deliberately wrong run counter must raise."""
    composer = RunLengthComposer(
        PhoBERTShapedTokenizer(), counters=TransformCountersFactory()
    )
    original = composer._run_tokens
    composer._run_tokens = lambda run: original(run) + 1  # corrupt the counter
    with pytest.raises(Stage1ContractViolation, match="run composition disagreed"):
        composer.length("Tôi đã đọc")


def TransformCountersFactory():
    from unmark.stage1.lengths import TransformCounters

    return TransformCounters()


def test_expensive_run_evaluations_scale_with_runs_not_prefix_lengths():
    """Task G: the algorithmic claim, asserted with counters not wall-clock."""
    content = " ".join("Việt Nam là một quốc gia".split() * 400)
    document = doc("big", content)
    ref, base, transforms = build_length_functions(PhoBERTShapedTokenizer())
    chunk_document(document, "train", reference_length=ref, base_length=base,
                   max_length=256)
    counters = transforms.counters
    words = len(content.split())

    assert counters.length_queries > 100, "the fixture must exercise the greedy scan"
    # BPE is evaluated once per DISTINCT run, not once per growing prefix
    assert counters.bpe_run_evaluations < words, (
        f"{counters.bpe_run_evaluations} BPE evaluations for {words} words"
    )
    assert counters.run_cache_hits > 10 * counters.run_cache_misses
    # the growing candidate is appended to, not rescanned
    assert counters.incremental_appends > 10 * counters.full_fallbacks
    # authoritative whole-string calls are bounded by the verification window
    assert counters.authoritative_queries <= 2 * (DEFAULT_VERIFY_FIRST + 1) + 4
