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
from unmark.stage1.lengths import ComposedTransforms, build_length_functions

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
def test_a_non_composing_tokenizer_is_caught_and_fails_closed():
    """The audited per-chunk fact is a CHECKED precondition, not a belief."""
    ref, _, _ = build_length_functions(NonComposingTokenizer())
    with pytest.raises(Stage1ContractViolation, match="per-chunk token composition"):
        ref("một hai ba bốn năm")


def test_the_verifier_accepts_a_conforming_tokenizer():
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
    # runs examined stay proportional to the document, not to queries x segments
    assert counters.runs_seen < 4 * len(content.split()), (
        f"runs_seen {counters.runs_seen} is quadratic-looking for "
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
    assert counters.run_cache_hits > counters.tokenizer_calls


def test_counters_carry_no_text():
    _, _, transforms = build_length_functions(WordTokenizer())
    payload = transforms.counters.to_dict()
    assert all(isinstance(v, int) for v in payload.values())
