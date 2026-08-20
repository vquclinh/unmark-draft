"""Local tests for the B3B-1A manual alignment core.

Mock BPE sequences only: no transformers, no torch, no Java, no network. These
prove the alignment *logic*; whether PhoBERT's real pieces reconstruct is what
the Colab probe measures, and nothing here claims it does.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from unmark.alignment import (
    CONTINUATION_MARKER,
    AlignmentFailureReason,
    SpanAlignmentStatus,
    align_span,
    characters_for_piece,
    compare_sequences,
    piece_surface,
    pieces_for_character,
    reconstruct_surface,
    summarize_alignments,
)
from unmark.orthography import Eligibility

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "b3b1_phobert_alignment_probe.py"
VN = Eligibility.VIETNAMESE_CANDIDATE
NA = Eligibility.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Continuation-marker semantics
# ---------------------------------------------------------------------------
def test_marker_is_the_fastbpe_suffix():
    assert CONTINUATION_MARKER == "@@"


@pytest.mark.parametrize(
    "token,surface", [("ngh@@", "ngh"), ("ien", "ien"), ("@@", ""), ("a@@b", "a@@b"), ("", "")]
)
def test_piece_surface_strips_only_a_trailing_marker(token, surface):
    assert piece_surface(token) == surface


def test_reconstruct_surface_concatenates_pieces():
    assert reconstruct_surface(["ngh@@", "ien"]) == "nghien"
    assert reconstruct_surface(["toi"]) == "toi"
    assert reconstruct_surface([]) == ""


# ---------------------------------------------------------------------------
# Successful alignment
# ---------------------------------------------------------------------------
def test_single_piece_syllable():
    alignment = align_span("toi", ["toi"], [7], eligibility=VN)
    assert alignment.status is SpanAlignmentStatus.ALIGNED
    assert alignment.subword_count == 1
    piece = alignment.pieces[0]
    assert (piece.start, piece.end) == (0, 3)
    assert not piece.is_continuation
    assert alignment.carries_channels


def test_multi_piece_syllable_gets_exact_half_open_ranges():
    alignment = align_span("nghien", ["ngh@@", "ien"], [1, 2], eligibility=VN)
    assert alignment.aligned
    assert [(p.start, p.end) for p in alignment.pieces] == [(0, 3), (3, 6)]
    assert [characters_for_piece(alignment, i) for i in range(2)] == ["ngh", "ien"]
    assert alignment.pieces[0].is_continuation and not alignment.pieces[1].is_continuation


def test_three_piece_syllable():
    alignment = align_span("nghieng", ["n@@", "ghie@@", "ng"], eligibility=VN)
    assert alignment.aligned
    assert [(p.start, p.end) for p in alignment.pieces] == [(0, 1), (1, 5), (5, 7)]
    assert "".join(characters_for_piece(alignment, i) for i in range(3)) == "nghieng"


def test_repeated_substrings_get_distinct_ranges():
    """`toitoi` must not collapse both `toi` pieces onto the same range."""
    alignment = align_span("toitoi", ["toi@@", "toi"], eligibility=VN)
    assert [(p.start, p.end) for p in alignment.pieces] == [(0, 3), (3, 6)]
    assert characters_for_piece(alignment, 0) == characters_for_piece(alignment, 1) == "toi"


def test_uppercase_span_aligns():
    alignment = align_span("TOI", ["TO@@", "I"], eligibility=VN)
    assert alignment.aligned
    assert alignment.reconstructed == "TOI"


def test_character_to_piece_lookup():
    """Needed to pool per-character letter-diacritic states into subwords."""
    alignment = align_span("nghien", ["ngh@@", "ien"], eligibility=VN)
    assert pieces_for_character(alignment, 0) == [0]
    assert pieces_for_character(alignment, 2) == [0]
    assert pieces_for_character(alignment, 3) == [1]
    assert pieces_for_character(alignment, 99) == []


# ---------------------------------------------------------------------------
# Failure policy — never a silent label
# ---------------------------------------------------------------------------
def test_unknown_token_is_an_alignment_failure():
    alignment = align_span("xyz", ["<unk>"], eligibility=VN, unk_token="<unk>")
    assert alignment.status is SpanAlignmentStatus.ALIGNMENT_FAILURE
    assert alignment.failure_reason is AlignmentFailureReason.UNKNOWN_TOKEN
    assert alignment.pieces == (), "a failed alignment must expose no ranges"
    assert not alignment.carries_channels


def test_unknown_token_among_valid_pieces_still_fails():
    alignment = align_span("abcdef", ["abc@@", "<unk>"], eligibility=VN, unk_token="<unk>")
    assert alignment.failure_reason is AlignmentFailureReason.UNKNOWN_TOKEN
    assert "index(es) [1]" in alignment.detail


def test_surface_mismatch_is_an_alignment_failure():
    alignment = align_span("nghien", ["ngh@@", "iem"], eligibility=VN)
    assert alignment.failure_reason is AlignmentFailureReason.SURFACE_MISMATCH
    assert alignment.reconstructed == "nghiem"
    assert alignment.pieces == ()


def test_malformed_continuation_on_the_final_piece_fails():
    """A trailing marker means the span's tokenization is not self-contained."""
    alignment = align_span("nghien", ["ngh@@", "ien@@"], eligibility=VN)
    assert alignment.failure_reason is AlignmentFailureReason.MALFORMED_CONTINUATION


def test_no_tokens_fails():
    alignment = align_span("toi", [], eligibility=VN)
    assert alignment.failure_reason is AlignmentFailureReason.NO_TOKENS


def test_undecided_eligibility_cannot_enter_a_scientific_alignment():
    """The defect audit 011 repaired: UNDECIDED must never be labelled."""
    alignment = align_span("toi", ["toi"], eligibility=Eligibility.UNDECIDED)
    assert alignment.status is SpanAlignmentStatus.ALIGNMENT_FAILURE
    assert alignment.failure_reason is AlignmentFailureReason.UNRESOLVED_ELIGIBILITY
    assert not alignment.carries_channels
    assert "inventory" in alignment.detail


def test_undecided_may_be_aligned_only_when_explicitly_permitted():
    alignment = align_span(
        "toi", ["toi"], eligibility=Eligibility.UNDECIDED, require_resolved_eligibility=False
    )
    assert alignment.status is SpanAlignmentStatus.ALIGNED
    assert not alignment.carries_channels, "still no channels without a resolved verdict"


# ---------------------------------------------------------------------------
# Non-Vietnamese / punctuation policy
# ---------------------------------------------------------------------------
def test_non_vietnamese_span_is_not_applicable():
    alignment = align_span("Python", ["Py@@", "thon"], eligibility=NA)
    assert alignment.status is SpanAlignmentStatus.NOT_APPLICABLE
    assert not alignment.carries_channels
    assert "N/A in both orthography channels" in alignment.detail


def test_punctuation_span_is_not_applicable():
    alignment = align_span(" , ", [",", "@@"], eligibility=NA)
    assert alignment.status is SpanAlignmentStatus.NOT_APPLICABLE


def test_punctuation_adjacent_to_a_syllable_does_not_shift_ranges():
    alignment = align_span("hoc", ["hoc"], eligibility=VN)
    assert (alignment.pieces[0].start, alignment.pieces[0].end) == (0, 3)


# ---------------------------------------------------------------------------
# Full-sequence reconciliation
# ---------------------------------------------------------------------------
def test_consistent_composition():
    result = compare_sequences(["toi", "di@@", "hoc"], ["toi", "di@@", "hoc"])
    assert result["consistent"]
    assert result["unexplained_tokens"] == []


def test_special_tokens_are_excluded_from_the_comparison():
    result = compare_sequences(["<s>", "toi", "</s>"], ["toi"], special_tokens=["<s>", "</s>"])
    assert result["consistent"]


def test_inconsistent_composition_reports_unexplained_tokens():
    result = compare_sequences(["toi", "di", "hoc"], ["toi", "hoc"])
    assert not result["consistent"]
    assert result["unexplained_tokens"] == ["di"]
    assert "unaccounted for" in result["detail"]


def test_reordered_composition_is_inconsistent():
    result = compare_sequences(["a", "b"], ["b", "a"])
    assert not result["consistent"]


# ---------------------------------------------------------------------------
# Summaries and serialisation
# ---------------------------------------------------------------------------
def test_summary_counts_outcomes_and_reasons():
    alignments = [
        align_span("toi", ["toi"], eligibility=VN),
        align_span("nghien", ["ngh@@", "ien"], eligibility=VN),
        align_span("xyz", ["<unk>"], eligibility=VN, unk_token="<unk>"),
        align_span("abc", ["ab@@", "d"], eligibility=VN),
        align_span("Python", ["Python"], eligibility=NA),
    ]
    summary = summarize_alignments(alignments)
    assert summary["total"] == 5
    assert summary["aligned"] == 2
    assert summary["failed"] == 2
    assert summary["not_applicable"] == 1
    assert summary["spans_with_unknown_token"] == 1
    assert summary["surface_reconstruction_failures"] == 1
    assert summary["mean_subwords_per_span"] == pytest.approx(1.5)
    assert summary["max_subwords_per_span"] == 2


def test_summary_of_no_alignments():
    summary = summarize_alignments([])
    assert summary["alignment_rate"] is None
    assert summary["mean_subwords_per_span"] is None


def test_alignment_serialises_deterministically():
    alignment = align_span("nghien", ["ngh@@", "ien"], [1, 2], eligibility=VN)
    first = json.dumps(alignment.to_dict(), ensure_ascii=False, sort_keys=True)
    second = json.dumps(alignment.to_dict(), ensure_ascii=False, sort_keys=True)
    assert first == second
    restored = json.loads(first)
    assert restored["status"] == "ALIGNED"
    assert restored["eligibility"] == "VIETNAMESE_CANDIDATE"
    assert restored["pieces"][1]["start"] == 3
    assert restored["carries_channels"] is True


def test_failure_serialises_with_its_reason():
    payload = align_span("xyz", ["<unk>"], eligibility=VN, unk_token="<unk>").to_dict()
    assert payload["status"] == "ALIGNMENT_FAILURE"
    assert payload["failure_reason"] == "UNKNOWN_TOKEN"
    assert payload["pieces"] == []


# ---------------------------------------------------------------------------
# Probe hygiene
# ---------------------------------------------------------------------------
def test_alignment_core_imports_no_ml_library():
    banned = {"torch", "transformers", "tokenizers", "sentencepiece", "datasets"}
    tree = ast.parse((REPO_ROOT / "unmark" / "alignment" / "manual.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned


def test_alignment_probe_loads_no_model_weights():
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in name, f"probe calls {name}.from_pretrained"


def test_alignment_probe_uses_the_slow_tokenizer_as_authority():
    source = PROBE.read_text(encoding="utf-8")
    assert "use_fast=False" in source
    assert "authoritative" in source.lower()
    # The optional fast diagnostic must never be promoted to authority.
    assert '"authoritative": "slow"' in source


def test_alignment_probe_requires_a_resolved_inventory():
    """Without it every span is UNDECIDED — the bug this task repaired."""
    source = PROBE.read_text(encoding="utf-8")
    assert "load_inventory()" in source
    assert "try_load_inventory" not in source, "the probe must fail loudly, not degrade"


def test_alignment_probe_reuses_the_b3b0_probe_revision():
    source = PROBE.read_text(encoding="utf-8")
    assert "01daacda68afe13d83023d16ec647239e344a1e6" in source
    assert "not the final backbone lock" in source


def test_alignment_probe_refuses_locally_without_transformers():
    result = subprocess.run(
        [sys.executable, str(PROBE)], cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert result.returncode == 2
    assert "transformers is not installed" in result.stderr


# ---------------------------------------------------------------------------
# The eligibility integration bug (audit 011 §H)
# ---------------------------------------------------------------------------
def test_inventory_loading_is_independent_of_the_working_directory():
    """Root cause of the all-UNDECIDED artifact: a relative default manifest
    path resolved against whatever cwd a dependency had chdir()'d into."""
    import os

    from unmark.linguistics import DEFAULT_MANIFEST, clear_inventory_cache, try_load_inventory

    assert Path(DEFAULT_MANIFEST).is_absolute(), "the default manifest path must be repo-anchored"
    original = Path.cwd()
    try:
        clear_inventory_cache()
        os.chdir("/")
        assert try_load_inventory() is not None, "inventory must load regardless of cwd"
    finally:
        os.chdir(original)
        clear_inventory_cache()


def test_resolved_eligibility_propagates_into_decomposition_after_a_chdir():
    import os

    from unmark.linguistics import clear_inventory_cache, make_classifier, try_load_inventory
    from unmark.orthography import decompose

    original = Path.cwd()
    try:
        clear_inventory_cache()
        os.chdir("/")
        classifier = make_classifier(try_load_inventory())
        spans = decompose("toi dung Python", eligibility_classifier=classifier).syllables
        verdicts = [s.eligibility for s in spans]
        assert Eligibility.UNDECIDED not in verdicts
        assert verdicts == [VN, VN, NA]
    finally:
        os.chdir(original)
        clear_inventory_cache()
