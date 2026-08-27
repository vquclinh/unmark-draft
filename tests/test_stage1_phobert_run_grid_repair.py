"""The Audit-044 PhoBERT run-grid repair. **Torch-free.**

Stage-6 measured its token grid over PhoBERT's own decomposition unit,
`PHOBERT_RUN = r"\\S+\\n?"`. Stage-1's alignment used plain `r"\\S+"`, so on any
newline-bearing chunk the two built *different* token grids: BPE's end-of-word
marker lands on the newline in one and on the last letter in the other.

Audit 043 measured the consequence on the real corpus: the grids disagreed on
92 566 of a 100 000-row window, and **9** TRAIN chunks that Stage-6 admitted at
exactly 256 came out of Stage-1 over `MAX_LENGTH` — eight at 257 and one at 259
— which aborted the first real `lr-pilot` fail-closed.

The repair is that Stage-1 now uses the same unit. These tests hold the repaired
path to the invariants that makes true, using a **faithful** transcription of
`PhobertTokenizer.bpe` rather than a stub that models only token counts.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from unmark.alignment.manual import (  # noqa: E402
    _CHUNK_PATTERN,
    align_chunk,
    piece_surface,
    reconstruct_surface,
    whitespace_chunks,
)
from unmark.linguistics import make_classifier, try_load_inventory  # noqa: E402
from unmark.stage1.contracts import CorruptionRatePolicy, TruncationPolicy  # noqa: E402
from unmark.stage1.data import Stage1Example, prepare_example, project_text  # noqa: E402
from unmark.stage1.lengths import PHOBERT_RUN, build_length_functions  # noqa: E402
from unmark.stage1.protocol import CORRUPTION_SEED, MAX_LENGTH  # noqa: E402

from test_stage1_length_contract_scanner import RunUnitTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return RunUnitTokenizer()


@pytest.fixture(scope="module")
def classifier():
    return make_classifier(try_load_inventory())


@pytest.fixture(scope="module")
def stage6_base_length(tokenizer):
    _reference, base_length, _transforms = build_length_functions(tokenizer)
    return base_length


# The §I matrix. Every shape the repair had to get right.
CASES = {
    "no newline": "xin chao ban hien",
    "internal newline": "xin chao\nban hien",
    "trailing newline": "xin chao ban\n",
    "leading newline": "\nxin chao ban",
    "two consecutive newlines": "xin\n\nchao ban",
    "three consecutive newlines": "xin\n\n\nchao",
    "space before newline": "xin \nchao",
    "space after newline": "xin\n chao",
    "tab then newline": "xin\t\nchao",
    "carriage return newline": "xin\r\nchao",
    "leading whitespace": "   xin chao",
    "trailing whitespace": "xin chao   ",
    "newline after bpe-sensitive word": "abcd\nabcd abcd",
    "every word newline-terminated": "abcd\nefgh\nabcd\n",
    "single char run": "a",
    "single char run with newline": "a\n",
    "unknown-ish surface": "zzqq\nzzqq",
    "vietnamese with newline": "Tôi học tiếng Việt\nmỗi ngày",
    "mixed whitespace": "alpha  beta\tgamma\n\ndelta   ",
}

TEXTS = list(CASES.values())


def authoritative_content_ids(tokenizer, base_text):
    """The grid the model actually sees: whole-string tokenization."""
    return list(tokenizer.convert_tokens_to_ids(list(tokenizer.tokenize(base_text))))


# ---------------------------------------------------------------------------
# The unit itself
# ---------------------------------------------------------------------------
def test_stage1_and_stage6_share_one_run_unit():
    assert _CHUNK_PATTERN.pattern == PHOBERT_RUN.pattern == r"\S+\n?"


def test_the_repair_is_not_a_naive_regex_swap(tokenizer):
    """The unit change is only sound because bpe partitions characters.

    `bpe` puts `</w>` on the run's LAST character and strips it from the joined
    output, so the pieces' surfaces concatenate back to the run — newline
    included. That is what lets the surface-exact alignment accept these runs.
    """
    for run in ("gamma\n", "abcd\n", "a\n", "xin\n", "đọc\n"):
        pieces = tokenizer.bpe(run).split(" ")
        assert "".join(piece_surface(p) for p in pieces) == run, (run, pieces)


@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_chunks_reproduce_the_tokenizers_own_runs(name, text):
    assert [c.text for c in whitespace_chunks(text)] == PHOBERT_RUN.findall(text)


# ---------------------------------------------------------------------------
# Invariant 1 — Stage-1 ids ARE the authoritative whole-string ids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_content_ids_equal_the_whole_string_tokenization(name, text, tokenizer,
                                                         classifier):
    base_text, content_ids, _projections = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    assert list(content_ids) == authoritative_content_ids(tokenizer, base_text), name


def test_the_pre_repair_unit_would_break_invariant_one(tokenizer, classifier):
    """Mutation check: plain `\\S+` really does produce a different grid.

    Without this the invariant above could be passing for some unrelated
    reason. Recomputing over the old unit must disagree somewhere.
    """
    import re

    old_unit = re.compile(r"\S+")
    disagreements = 0
    for text in TEXTS:
        base_text, content_ids, _p = project_text(
            text, tokenizer, classifier, tokenizer.unk_token_id
        )
        old_ids = []
        for match in old_unit.finditer(base_text):
            tokens = tokenizer.tokenize(match.group(0))
            old_ids.extend(tokenizer.convert_tokens_to_ids(list(tokens)))
        if old_ids != list(content_ids):
            disagreements += 1
    assert disagreements > 0, (
        "the old plain-\\S+ unit must differ from the repaired grid on at least "
        "one newline-bearing case, or this suite proves nothing"
    )


# ---------------------------------------------------------------------------
# Invariant 2 — Stage-1 realised length == Stage-6 authoritative length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_realised_length_equals_stage6_authoritative(name, text, tokenizer,
                                                     classifier, stage6_base_length):
    _base, content_ids, _p = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    realised = len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    assert realised == stage6_base_length(text), name


# ---------------------------------------------------------------------------
# Invariant 3 — ids and channel metadata stay 1:1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_ids_and_projections_remain_one_to_one(name, text, tokenizer, classifier):
    _base, content_ids, projections = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    assert len(content_ids) == len(projections), name


# ---------------------------------------------------------------------------
# Invariants 4 and 5 — alignment, spans, and newline ownership
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_alignment_round_trips_and_tiles_every_run(name, text, tokenizer):
    for chunk in whitespace_chunks(text):
        tokens = tuple(tokenizer.tokenize(chunk.text))
        ids = tuple(tokenizer.convert_tokens_to_ids(list(tokens)))
        alignment = align_chunk(chunk, tokens, ids,
                                unk_token_id=tokenizer.unk_token_id)
        assert alignment.pieces, f"{name}: alignment failed on {chunk.text!r}"
        assert reconstruct_surface(tokens) == chunk.text
        # Pieces tile the run contiguously, in order, with no gap or overlap.
        cursor = chunk.start
        for piece in alignment.pieces:
            assert piece.global_start == cursor
            cursor = piece.global_end
        assert cursor == chunk.end
        rebuilt = "".join(text[p.global_start:p.global_end] for p in alignment.pieces)
        assert rebuilt == chunk.text


def test_a_newline_is_owned_once_and_only_once(tokenizer):
    """Not dropped, not duplicated, not moved to the wrong run."""
    text = "abcd\nefgh\nabcd"
    covered = []
    for chunk in whitespace_chunks(text):
        tokens = tuple(tokenizer.tokenize(chunk.text))
        ids = tuple(tokenizer.convert_tokens_to_ids(list(tokens)))
        for piece in align_chunk(chunk, tokens, ids).pieces:
            covered.append((piece.global_start, piece.global_end))
    newline_positions = {i for i, ch in enumerate(text) if ch == "\n"}
    for position in newline_positions:
        owners = [span for span in covered if span[0] <= position < span[1]]
        assert len(owners) == 1, f"newline at {position} owned by {len(owners)} pieces"


def test_non_newline_text_is_unchanged_by_the_repair(tokenizer, classifier):
    """Nothing moves for text the tokenizer would have decomposed identically."""
    import re

    for text in ("xin chao ban", "alpha beta gamma", "  spaced   out  "):
        assert re.compile(r"\S+").findall(text) == PHOBERT_RUN.findall(text)
        _b, ids, _p = project_text(text, tokenizer, classifier, tokenizer.unk_token_id)
        assert list(ids) == authoritative_content_ids(tokenizer, _b)


# ---------------------------------------------------------------------------
# Defensive shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", " ", "\n", "\n\n", "   \t\n  "])
def test_empty_and_whitespace_only_text_is_defensible(text, tokenizer, classifier):
    base_text, content_ids, projections = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    assert list(content_ids) == authoritative_content_ids(tokenizer, base_text)
    assert len(content_ids) == len(projections)
    assert whitespace_chunks(text) == () or all(c.text.strip() for c in
                                                whitespace_chunks(text))


def test_a_final_continuation_marker_still_fails_closed(tokenizer):
    """The alignment guard is untouched by the repair."""
    chunk = whitespace_chunks("abcd")[0]
    broken = ("ab@@", "cd@@")
    alignment = align_chunk(chunk, broken, (1, 2))
    assert alignment.pieces == ()
    assert alignment.failure_reason is not None


def test_a_surface_mismatch_still_fails_closed(tokenizer):
    chunk = whitespace_chunks("abcd")[0]
    alignment = align_chunk(chunk, ("zz",), (1,))
    assert alignment.pieces == ()


def test_unknown_token_ids_are_reported_not_fatal(tokenizer):
    chunk = whitespace_chunks("abcd\n")[0]
    tokens = tuple(tokenizer.tokenize(chunk.text))
    ids = tuple(tokenizer.unk_token_id for _ in tokens)
    alignment = align_chunk(chunk, tokens, ids, unk_token_id=tokenizer.unk_token_id)
    assert alignment.pieces, "an OOV surface must still align"
    assert all(p.has_unknown_token_id for p in alignment.pieces)


# ---------------------------------------------------------------------------
# The real 9-offender class, reproduced synthetically
# ---------------------------------------------------------------------------
def boundary_text(tokenizer, stage6_base_length, target):
    """A text whose Stage-6 authoritative length is exactly `target`."""
    for words in range(1, 400):
        for tail in ("", " a", " ab", " abc"):
            text = ("abcd\n" * words).rstrip("\n") + tail
            if stage6_base_length(text) == target:
                return text
    return None


def test_the_former_overflow_class_no_longer_overflows(tokenizer, classifier,
                                                       stage6_base_length):
    """Audit 043's 9 offenders: Stage-6 256, old Stage-1 257 or 259.

    Every one was newline-bearing. Post-repair, a chunk Stage-6 admits at 256
    must come out of Stage-1 at exactly 256 — never 257, never 259.
    """
    text = boundary_text(tokenizer, stage6_base_length, MAX_LENGTH)
    assert text is not None, "could not construct an exactly-256 newline-bearing text"
    assert "\n" in text

    _base, content_ids, _p = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    realised = len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    assert stage6_base_length(text) == MAX_LENGTH
    assert realised == MAX_LENGTH, f"repaired Stage-1 is {realised}, not {MAX_LENGTH}"
    assert realised <= MAX_LENGTH


@pytest.mark.parametrize("target", [MAX_LENGTH - 2, MAX_LENGTH - 1, MAX_LENGTH])
def test_the_boundary_holds_from_below(target, tokenizer, classifier,
                                       stage6_base_length):
    text = boundary_text(tokenizer, stage6_base_length, target)
    if text is None:
        pytest.skip(f"no newline-bearing text at exactly {target} under this vocab")
    _base, content_ids, _p = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    realised = len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    assert realised == target <= MAX_LENGTH


def test_a_truncation_policy_accepts_what_stage6_admitted(tokenizer, classifier,
                                                          stage6_base_length):
    """The exact gate that aborted the real run, on the repaired path."""
    truncation = TruncationPolicy(max_length=MAX_LENGTH,
                                  on_overflow=__import__(
                                      "unmark.stage1.contracts", fromlist=["x"]
                                  ).OverflowBehaviour.FAIL)
    text = boundary_text(tokenizer, stage6_base_length, MAX_LENGTH)
    assert text is not None
    _base, content_ids, _p = project_text(
        text, tokenizer, classifier, tokenizer.unk_token_id
    )
    realised = len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    assert truncation.check(realised, "base sequence") is True


# ---------------------------------------------------------------------------
# Invariant 6 — corruption cannot move the base grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", list(CASES.items()), ids=list(CASES))
def test_corruption_cannot_change_the_base_grid(name, text, tokenizer, classifier):
    """`prepare_example` asserts b(C(x)) == b(x) and equal base ids itself.

    Running it over every newline shape proves the repaired grid keeps that
    invariant rather than merely not crashing.
    """
    if not text.strip():
        pytest.skip("no content to corrupt")
    prepared = prepare_example(
        Stage1Example(text, f"doc-{abs(hash(name)) % 10000:04d}#0"),
        tokenizer,
        corruption_policy=CorruptionRatePolicy(seed=CORRUPTION_SEED),
        truncation=TruncationPolicy.unbounded(),
        visit=0,
        classifier=classifier,
        unk_token_id=tokenizer.unk_token_id,
    )
    assert prepared is not None
    assert len(prepared.base_input_ids) == len(prepared.base_special_tokens_mask)
    assert len(prepared.clean_tone_ids) == len(prepared.corrupt_tone_ids)
    assert len(prepared.clean_letter_ids) == len(prepared.corrupt_letter_ids)
