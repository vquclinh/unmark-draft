"""Local tests for the B3B-1B whitespace-chunk alignment core.

Mock raw-BPE sequences only: no transformers, no torch, no Java, no network.
The mock pieces reproduce sequences the researcher observed from the real
authoritative tokenizer, so the logic is exercised against real behaviour —
but nothing here claims to validate PhoBERT. That is the Colab probe's job.
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
    AlignmentStatusB,
    Chunk,
    OrthographicRegion,
    ToneOwnership,
    align_chunk,
    characters_for_piece,
    compose,
    overlay_orthography,
    piece_surface,
    reconstruct_surface,
    summarize_chunk_alignments,
    verify_token_grid,
    whitespace_chunks,
)
from unmark.orthography import Eligibility

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "b3b1_phobert_alignment_probe.py"
VN = Eligibility.VIETNAMESE_CANDIDATE
NA = Eligibility.NOT_APPLICABLE
UNK_ID = 3


def one_chunk(text: str) -> Chunk:
    return whitespace_chunks(text)[0]


# ---------------------------------------------------------------------------
# 1-3. Whitespace chunking
# ---------------------------------------------------------------------------
def test_chunks_carry_exact_global_ranges():
    chunks = whitespace_chunks("Toi hoc nhien.")
    assert [(c.text, c.start, c.end) for c in chunks] == [
        ("Toi", 0, 3), ("hoc", 4, 7), ("nhien.", 8, 14)
    ]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_multiple_spaces_do_not_shift_ranges():
    text = "a   b"
    chunks = whitespace_chunks(text)
    assert [(c.text, c.start, c.end) for c in chunks] == [("a", 0, 1), ("b", 4, 5)]
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text


def test_tabs_and_newlines_are_whitespace():
    chunks = whitespace_chunks("a\tb\nc\r\nd")
    assert [c.text for c in chunks] == ["a", "b", "c", "d"]


def test_leading_and_trailing_whitespace():
    chunks = whitespace_chunks("   toi  ")
    assert [(c.text, c.start, c.end) for c in chunks] == [("toi", 3, 6)]


def test_empty_and_whitespace_only_text_yield_no_chunks():
    assert whitespace_chunks("") == ()
    assert whitespace_chunks("   \t\n") == ()


# ---------------------------------------------------------------------------
# 4-9. Real observed chunk shapes
# ---------------------------------------------------------------------------
def test_punctuation_attached_to_a_word():
    """Observed: "nhien." -> ["nhi@@", "en@@", "."] — NOT ["nh@@","ien"]+["."]."""
    alignment = align_chunk(one_chunk("nhien."), ["nhi@@", "en@@", "."], [1, 2, 3])
    assert alignment.aligned
    assert [(p.local_start, p.local_end) for p in alignment.pieces] == [(0, 3), (3, 5), (5, 6)]
    assert alignment.reconstructed == "nhien."


def test_leading_punctuation():
    """Observed: "(VAT" -> ["(@@", "VAT"]."""
    alignment = align_chunk(one_chunk("(VAT"), ["(@@", "VAT"], [4, 5])
    assert alignment.aligned
    assert [p.surface for p in alignment.pieces] == ["(", "VAT"]


def test_hyphenated_word():
    """Observed: "Viet-Nam" -> ["Viet@@", "-@@", "Nam"]."""
    alignment = align_chunk(one_chunk("Viet-Nam"), ["Viet@@", "-@@", "Nam"], [6, 7, 8])
    assert alignment.aligned
    assert [(p.local_start, p.local_end) for p in alignment.pieces] == [(0, 4), (4, 5), (5, 8)]


def test_acronym_with_hyphen():
    """Observed: "VNU-HCM" -> ["VN@@", "U-@@", "HCM"]."""
    alignment = align_chunk(one_chunk("VNU-HCM"), ["VN@@", "U-@@", "HCM"], [9, 10, 11])
    assert alignment.aligned
    assert "".join(p.surface for p in alignment.pieces) == "VNU-HCM"


def test_url_like_chunk_reconstructs_exactly():
    url = "https://example.edu.vn/tuyen-sinh?id=42&lang=vi"
    pieces = ["https://@@", "example.@@", "edu.@@", "vn/@@", "tuyen-@@", "sinh?@@", "id=@@", "42&@@", "lang=@@", "vi"]
    alignment = align_chunk(one_chunk(url), pieces, list(range(len(pieces))))
    assert alignment.aligned
    assert alignment.reconstructed == url
    assert alignment.pieces[-1].global_end == len(url)


def test_email_like_chunk_reconstructs_exactly():
    email = "lien.he@example.com"
    # Note `he@` + the marker: a surface may legitimately end in "@", and only
    # ONE trailing marker is stripped.
    pieces = ["lien.@@", "he@" + CONTINUATION_MARKER, "example.@@", "com"]
    alignment = align_chunk(one_chunk(email), pieces, [1, 2, 3, 4])
    assert alignment.aligned, alignment.detail
    assert alignment.reconstructed == email
    assert alignment.pieces[1].surface == "he@"


# ---------------------------------------------------------------------------
# 10-13. Reconstruction and ranges
# ---------------------------------------------------------------------------
def test_marker_and_surface_semantics():
    assert CONTINUATION_MARKER == "@@"
    assert piece_surface("nhi@@") == "nhi"
    assert piece_surface(".") == "."
    assert reconstruct_surface(["nhi@@", "en@@", "."]) == "nhien."


def test_global_ranges_translate_from_the_chunk_offset():
    chunks = whitespace_chunks("Toi hoc nhien.")
    alignment = align_chunk(chunks[2], ["nhi@@", "en@@", "."], [1, 2, 3])
    assert [(p.global_start, p.global_end) for p in alignment.pieces] == [(8, 11), (11, 13), (13, 14)]


def test_characters_for_piece_uses_global_ranges():
    text = "Toi hoc nhien."
    alignment = align_chunk(whitespace_chunks(text)[2], ["nhi@@", "en@@", "."], [1, 2, 3])
    assert [characters_for_piece(text, p) for p in alignment.pieces] == ["nhi", "en", "."]


def test_piece_ranges_are_monotonic_and_tile_the_chunk():
    alignment = align_chunk(one_chunk("nghieng"), ["n@@", "ghie@@", "ng"], [1, 2, 3])
    previous = alignment.chunk.start
    for piece in alignment.pieces:
        assert piece.global_start == previous
        previous = piece.global_end
    assert previous == alignment.chunk.end


def test_repeated_substrings_get_distinct_ranges():
    alignment = align_chunk(one_chunk("toitoi"), ["toi@@", "toi"], [1, 2])
    assert [(p.local_start, p.local_end) for p in alignment.pieces] == [(0, 3), (3, 6)]


# ---------------------------------------------------------------------------
# 14-16. Composition and the token-grid invariant
# ---------------------------------------------------------------------------
def _aligned_sentence():
    text = "Toi hoc nhien."
    chunks = whitespace_chunks(text)
    return text, [
        align_chunk(chunks[0], ["Toi"], [10]),
        align_chunk(chunks[1], ["hoc"], [11]),
        align_chunk(chunks[2], ["nhi@@", "en@@", "."], [12, 13, 14]),
    ]


def test_chunk_composition_of_tokens_and_ids():
    _text, alignments = _aligned_sentence()
    tokens, ids = compose(alignments)
    assert tokens == ("Toi", "hoc", "nhi@@", "en@@", ".")
    assert ids == (10, 11, 12, 13, 14)


def test_token_grid_verification_passes_on_agreement():
    _text, alignments = _aligned_sentence()
    tokens, ids = compose(alignments)
    result = verify_token_grid(tokens, ids, tokens, ids)
    assert result["consistent"] and result["tokens_match"] and result["ids_match"]
    assert result["unexplained_tokens"] == []


def test_token_grid_verification_fails_on_a_token_mismatch():
    """The 6/13 failure mode: composition disagrees with the authority."""
    _text, alignments = _aligned_sentence()
    tokens, ids = compose(alignments)
    authoritative = ("Toi", "hoc", "nh@@", "ien", ".")
    result = verify_token_grid(tokens, ids, authoritative, ids)
    assert not result["consistent"]
    assert not result["tokens_match"]
    assert result["unexplained_tokens"][0]["authoritative"] == "nh@@"
    assert result["unexplained_tokens"][0]["composed"] == "nhi@@"


def test_token_grid_verification_fails_on_an_id_mismatch():
    _text, alignments = _aligned_sentence()
    tokens, ids = compose(alignments)
    result = verify_token_grid(tokens, ids, tokens, (99,) + ids[1:])
    assert not result["ids_match"]
    assert not result["consistent"]


def test_length_mismatch_is_reported():
    result = verify_token_grid(("a",), (1,), ("a", "b"), (1, 2))
    assert not result["consistent"]
    assert result["authoritative_length"] == 2 and result["composed_length"] == 1


# ---------------------------------------------------------------------------
# 17-19. Unknown vocabulary id vs surface recoverability
# ---------------------------------------------------------------------------
def test_unknown_token_id_with_recoverable_surface_stays_aligned():
    """The B3B-1A defect: `khut` tokenizes to ["khut"] with id 3 (<unk>).
    The surface is exact; only the vocabulary lookup failed."""
    alignment = align_chunk(one_chunk("khut"), ["khut"], [UNK_ID], unk_token_id=UNK_ID)
    assert alignment.status is AlignmentStatusB.ALIGNED
    assert alignment.reconstructed == "khut"
    assert alignment.pieces[0].has_unknown_token_id is True
    assert alignment.unknown_id_count == 1


def test_unknown_token_id_is_reported_separately_from_status():
    alignment = align_chunk(one_chunk("khut"), ["khut"], [UNK_ID], unk_token_id=UNK_ID)
    payload = alignment.to_dict()
    assert payload["status"] == "ALIGNED"
    assert payload["failure_reason"] is None
    assert payload["unknown_id_count"] == 1
    assert payload["pieces"][0]["has_unknown_token_id"] is True


def test_unknown_token_id_is_not_a_failure_reason():
    assert not hasattr(AlignmentFailureReason, "UNKNOWN_TOKEN")
    assert "UNKNOWN_TOKEN" not in {r.value for r in AlignmentFailureReason}


def test_genuine_surface_mismatch_still_fails():
    alignment = align_chunk(one_chunk("nhien."), ["nhi@@", "em@@", "."], [1, 2, 3])
    assert alignment.status is AlignmentStatusB.ALIGNMENT_FAILURE
    assert alignment.failure_reason is AlignmentFailureReason.SURFACE_MISMATCH
    assert alignment.pieces == (), "a failed alignment must expose no ranges"


def test_malformed_continuation_fails():
    alignment = align_chunk(one_chunk("nhien"), ["nhi@@", "en@@"], [1, 2])
    assert alignment.failure_reason is AlignmentFailureReason.MALFORMED_CONTINUATION


def test_no_tokens_fails():
    assert align_chunk(one_chunk("toi"), [], []).failure_reason is AlignmentFailureReason.NO_TOKENS


# ---------------------------------------------------------------------------
# 20-22. Orthographic overlay
# ---------------------------------------------------------------------------
def test_overlay_attributes_pieces_to_orthographic_regions():
    text = "Toi hoc nhien."
    alignment = align_chunk(whitespace_chunks(text)[2], ["nhi@@", "en@@", "."], [1, 2, 3])
    regions = [
        OrthographicRegion(0, "nhien", 8, 13, VN),
        OrthographicRegion(1, ".", 13, 14, NA, is_syllable=False),
    ]
    overlays = overlay_orthography(alignment.pieces, regions)
    assert [o.tone_ownership for o in overlays] == [
        ToneOwnership.SINGLE_CANDIDATE,
        ToneOwnership.SINGLE_CANDIDATE,
        ToneOwnership.NOT_APPLICABLE,
    ]
    assert overlays[0].tone_region_index == 0
    assert overlays[0].carries_tone and not overlays[2].carries_tone


def test_piece_mixing_one_candidate_with_punctuation_keeps_the_tone():
    """A BPE piece may straddle a Vietnamese candidate and punctuation.

    Only one candidate contributes, so nothing competes for the tone label and
    the piece keeps it. The punctuation is still recorded as a contributor -- it
    is excluded from the channels, not from the evidence.
    """
    text = "nhien."
    alignment = align_chunk(one_chunk(text), ["nhien."], [1])
    regions = [
        OrthographicRegion(0, "nhien", 0, 5, VN),
        OrthographicRegion(1, ".", 5, 6, NA, is_syllable=False),
    ]
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    assert overlay.tone_ownership is ToneOwnership.SINGLE_CANDIDATE
    assert overlay.carries_tone and not overlay.is_multi_candidate
    assert overlay.tone_region_index == 0
    assert overlay.candidate_region_indices == (0,)
    assert len(overlay.contributions) == 2
    assert [c.length for c in overlay.contributions] == [5, 1]


def test_piece_spanning_two_candidates_is_ambiguous_and_never_resolved():
    """Two distinct candidates in one piece: no tone, both recorded."""
    text = "nhien-hoc"
    alignment = align_chunk(one_chunk(text), ["nhien-hoc"], [1])
    regions = [
        OrthographicRegion(0, "nhien", 0, 5, VN),
        OrthographicRegion(1, "-", 5, 6, NA, is_syllable=False),
        OrthographicRegion(2, "hoc", 6, 9, VN),
    ]
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    assert overlay.tone_ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS
    assert overlay.is_multi_candidate and not overlay.carries_tone
    assert overlay.tone_region_index is None
    assert overlay.candidate_region_indices == (0, 2)
    assert len(overlay.contributions) == 3


def test_ambiguity_is_about_source_count_not_label_agreement():
    """Two candidates that would yield the same tone are still ambiguous.

    Sharing a value is not the same as having one source. Collapsing this case
    would smuggle in a "they agree, so pick it" rule that has no principled
    extension to the disagreeing case.
    """
    alignment = align_chunk(one_chunk("hoc-hoc"), ["hoc-hoc"], [1])
    regions = [
        OrthographicRegion(0, "hoc", 0, 3, VN),
        OrthographicRegion(1, "-", 3, 4, NA, is_syllable=False),
        OrthographicRegion(2, "hoc", 4, 7, VN),
    ]
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    assert overlay.tone_ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS
    assert overlay.candidate_region_indices == (0, 2)


def test_repeated_contributions_from_one_region_stay_single_candidate():
    """Candidate counting is over DISTINCT regions, not contribution records."""
    alignment = align_chunk(one_chunk("nhien"), ["nhien"], [1])
    regions = [OrthographicRegion(0, "nhien", 0, 5, VN)]
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    assert overlay.tone_ownership is ToneOwnership.SINGLE_CANDIDATE
    assert overlay.candidate_region_indices == (0,)


def test_undecided_eligibility_cannot_silently_carry_channels():
    alignment = align_chunk(one_chunk("toi"), ["toi"], [1])
    regions = [OrthographicRegion(0, "toi", 0, 3, Eligibility.UNDECIDED)]
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    assert overlay.tone_ownership is ToneOwnership.UNRESOLVED
    assert not overlay.carries_tone
    assert "resolve the inventory" in overlay.detail


def test_non_vietnamese_region_yields_not_applicable():
    alignment = align_chunk(one_chunk("Python"), ["Py@@", "thon"], [1, 2])
    regions = [OrthographicRegion(0, "Python", 0, 6, NA)]
    overlays = overlay_orthography(alignment.pieces, regions)
    assert all(o.tone_ownership is ToneOwnership.NOT_APPLICABLE for o in overlays)


def test_overlay_records_exact_overlap_ranges():
    alignment = align_chunk(one_chunk("Viet-Nam"), ["Viet@@", "-@@", "Nam"], [1, 2, 3])
    regions = [
        OrthographicRegion(0, "Viet", 0, 4, VN),
        OrthographicRegion(1, "-", 4, 5, NA, is_syllable=False),
        OrthographicRegion(2, "Nam", 5, 8, VN),
    ]
    overlays = overlay_orthography(alignment.pieces, regions)
    assert overlays[0].contributions[0].overlap_start == 0
    assert overlays[0].contributions[0].overlap_end == 4
    assert overlays[2].tone_region_index == 2


def test_piece_with_no_overlapping_region_is_not_applicable():
    alignment = align_chunk(one_chunk("abc"), ["abc"], [1])
    assert overlay_orthography(alignment.pieces, [])[0].tone_ownership is ToneOwnership.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Summaries and serialisation
# ---------------------------------------------------------------------------
def test_summary_separates_failures_from_unknown_ids():
    alignments = [
        align_chunk(one_chunk("Toi"), ["Toi"], [1]),
        align_chunk(one_chunk("khut"), ["khut"], [UNK_ID], unk_token_id=UNK_ID),
        align_chunk(one_chunk("nhien."), ["nhi@@", "em@@", "."], [1, 2, 3]),
    ]
    summary = summarize_chunk_alignments(alignments)
    assert summary["total_chunks"] == 3
    assert summary["aligned"] == 2
    assert summary["failed"] == 1
    assert summary["surface_reconstruction_failures"] == 1
    assert summary["chunks_with_unknown_token_id"] == 1
    assert summary["unknown_token_ids"] == 1
    assert summary["range_failures"] == 0


def test_summary_of_no_chunks():
    summary = summarize_chunk_alignments([])
    assert summary["mean_subwords_per_chunk"] is None


def test_alignment_serialises_deterministically():
    alignment = align_chunk(one_chunk("nhien."), ["nhi@@", "en@@", "."], [1, 2, 3])
    first = json.dumps(alignment.to_dict(), sort_keys=True, ensure_ascii=False)
    assert first == json.dumps(alignment.to_dict(), sort_keys=True, ensure_ascii=False)
    restored = json.loads(first)
    assert restored["status"] == "ALIGNED"
    assert restored["pieces"][1]["global_start"] == 3


def test_overlay_serialises():
    alignment = align_chunk(one_chunk("nhien."), ["nhien."], [1])
    regions = [OrthographicRegion(0, "nhien", 0, 5, VN), OrthographicRegion(1, ".", 5, 6, NA)]
    payload = overlay_orthography(alignment.pieces, regions)[0].to_dict()
    assert payload["tone_ownership"] == "SINGLE_CANDIDATE"
    assert payload["candidate_region_indices"] == [0]
    assert len(payload["contributions"]) == 2


# ---------------------------------------------------------------------------
# 23-24. Hygiene
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


def test_core_does_not_retokenize_linguistic_spans():
    """The B3B-1A error: spans are orthographic metadata, not BPE boundaries."""
    source = (REPO_ROOT / "unmark" / "alignment" / "manual.py").read_text(encoding="utf-8")
    assert "align_span" not in source, "span-level retokenization must not return"
    assert "whitespace_chunks" in source
    assert "not tokenization boundaries" in source


def test_probe_loads_no_model_weights():
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in name


def test_probe_keeps_the_slow_tokenizer_authoritative():
    source = PROBE.read_text(encoding="utf-8")
    assert "use_fast=False" in source
    assert '"authoritative": "slow"' in source


def test_probe_uses_raw_bpe_tokens_not_an_id_round_trip():
    """`convert_ids_to_tokens(encode(...))` destroys an OOV surface."""
    source = PROBE.read_text(encoding="utf-8")
    assert ".tokenize(" in source, "raw pieces must come from tokenizer.tokenize"
    assert "convert_tokens_to_ids" in source


def test_probe_refuses_locally_without_transformers():
    result = subprocess.run(
        [sys.executable, str(PROBE)], cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert result.returncode == 2
    assert "transformers is not installed" in result.stderr
