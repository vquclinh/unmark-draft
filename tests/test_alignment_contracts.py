"""Local tests for the B3B-0 input-contract structures.

Mock tokenizer output only. The suite needs no network, no transformers, no
torch, no Java and no VnCoreNLP.

**These tests prove the analysis logic, not PhoBERT's behaviour.** Nothing here
establishes what the real tokenizer does; that is what the Colab probe is for.
Every token sequence below is invented for the purpose of exercising a code
path.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from unmark.alignment import (
    PROBE_CONDITIONS,
    REPO_LOCAL_HF_CACHE,
    AlignmentStatus,
    OffsetAvailability,
    PathAvailability,
    PathObservation,
    PreprocessingPath,
    SegmenterContract,
    TokenizerContract,
    TokenSpan,
    alignment_status,
    character_coverage,
    compare_paths,
    grid_invariance,
    path_summary,
    syllable_token_map,
    tokens_for_span,
    validate_offsets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "b3b0_phobert_input_probe.py"


def mock_spans(text: str, pieces: list[tuple[str, int, int]], *, specials=("<s>", "</s>"), unk="<unk>"):
    """Build spans from (token, start, end) triples, wrapped in special tokens."""
    spans = [TokenSpan(0, specials[0], 0, is_special=True)]
    for offset, (token, start, end) in enumerate(pieces, start=1):
        spans.append(
            TokenSpan(offset, token, 100 + offset, start, end, is_unknown=(token == unk))
        )
    spans.append(TokenSpan(len(pieces) + 1, specials[1], 2, is_special=True))
    return spans


# ---------------------------------------------------------------------------
# TokenSpan
# ---------------------------------------------------------------------------
def test_token_span_overlap_is_half_open():
    span = TokenSpan(0, "abc", 1, 2, 5)
    assert span.overlaps(2, 3) and span.overlaps(4, 6)
    assert not span.overlaps(5, 7), "end is exclusive"
    assert not span.overlaps(0, 2), "start is inclusive"
    assert span.overlap_length(3, 10) == 2


def test_zero_width_and_offsetless_spans_never_overlap():
    assert not TokenSpan(0, "x", 1, 3, 3).overlaps(0, 10)
    assert not TokenSpan(0, "x", 1).overlaps(0, 10)
    assert TokenSpan(0, "x", 1).length == 0


def test_special_tokens_carry_no_offsets():
    spans = mock_spans("abc", [("abc", 0, 3)])
    assert spans[0].is_special and not spans[0].has_offsets
    assert spans[-1].is_special


# ---------------------------------------------------------------------------
# Offset validation
# ---------------------------------------------------------------------------
def test_exact_offsets_are_recognised():
    text = "nghien cuu"
    spans = mock_spans(text, [("nghien", 0, 6), ("cuu", 7, 10)])
    availability, reason = validate_offsets(text, spans)
    assert availability is OffsetAvailability.NATIVE_EXACT
    assert "reproduces" in reason


def test_bpe_continuation_markers_are_tolerated():
    """`@@`, `##` and `▁` are stripped before comparing a token to its slice."""
    text = "nghiencuu"
    for token, rest in (("nghien@@", "cuu"), ("##cuu", "nghien"), ("▁cuu", "nghien")):
        if token.startswith(("##", "▁")):
            spans = mock_spans(text, [(rest, 0, 6), (token, 6, 9)])
        else:
            spans = mock_spans(text, [(token, 0, 6), (rest, 6, 9)])
        assert validate_offsets(text, spans)[0] is OffsetAvailability.NATIVE_EXACT, token


def test_missing_offsets_are_reported_as_absent():
    spans = [TokenSpan(0, "<s>", 0, is_special=True), TokenSpan(1, "abc", 1), TokenSpan(2, "def", 2)]
    availability, reason = validate_offsets("abcdef", spans)
    assert availability is OffsetAvailability.ABSENT
    assert "no offset mapping" in reason


def test_partially_missing_offsets_are_malformed():
    spans = [TokenSpan(0, "abc", 1, 0, 3), TokenSpan(1, "def", 2)]
    assert validate_offsets("abcdef", spans)[0] is OffsetAvailability.NATIVE_MALFORMED


@pytest.mark.parametrize(
    "start,end,fragment",
    [(-1, 3, "outside"), (0, 99, "outside"), (5, 2, "end < start")],
)
def test_structurally_invalid_offsets_are_malformed(start, end, fragment):
    spans = [TokenSpan(0, "abc", 1, start, end)]
    availability, reason = validate_offsets("abcdef", spans)
    assert availability is OffsetAvailability.NATIVE_MALFORMED
    assert fragment in reason


def test_overlapping_offsets_are_malformed():
    spans = mock_spans("abcdef", [("abc", 0, 3), ("bcd", 1, 4)])
    availability, reason = validate_offsets("abcdef", spans)
    assert availability is OffsetAvailability.NATIVE_MALFORMED
    assert "before the previous token ended" in reason


def test_offsets_that_do_not_reconcile_with_token_strings_are_inexact():
    """The dangerous case: offsets look fine but point at the wrong text."""
    text = "nghien cuu"
    spans = mock_spans(text, [("nghien", 0, 6), ("XXX", 7, 10)])
    availability, reason = validate_offsets(text, spans)
    assert availability is OffsetAvailability.NATIVE_INEXACT
    assert "differs from the token string" in reason


def test_no_content_tokens_is_not_probed():
    spans = [TokenSpan(0, "<s>", 0, is_special=True)]
    assert validate_offsets("abc", spans)[0] is OffsetAvailability.NOT_PROBED


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def test_full_coverage():
    text = "abcdef"
    coverage = character_coverage(text, mock_spans(text, [("abc", 0, 3), ("def", 3, 6)]))
    assert coverage["fully_covered"] and coverage["coverage_rate"] == 1.0
    assert coverage["uncovered_indices"] == []


def test_gaps_are_reported():
    text = "abc def"
    coverage = character_coverage(text, mock_spans(text, [("abc", 0, 3), ("def", 4, 7)]))
    assert not coverage["fully_covered"]
    assert coverage["uncovered_indices"] == [3]  # the space
    assert coverage["covered_characters"] == 6


def test_coverage_of_empty_text():
    coverage = character_coverage("", [])
    assert coverage["coverage_rate"] is None and not coverage["fully_covered"]


# ---------------------------------------------------------------------------
# Syllable-to-token mapping
# ---------------------------------------------------------------------------
def test_each_syllable_maps_to_its_tokens():
    text = "toi di hoc"
    spans = mock_spans(text, [("toi", 0, 3), ("di", 4, 6), ("hoc", 7, 10)])
    mapping = syllable_token_map(spans, [(0, 3), (4, 6), (7, 10)])
    assert mapping["syllable_to_tokens"] == [[1], [2], [3]]
    assert mapping["all_syllables_mapped"]
    assert mapping["subwords_per_syllable_mean"] == 1.0


def test_fragmented_syllable_reports_multiple_subwords():
    text = "nghien"
    spans = mock_spans(text, [("ngh", 0, 3), ("ien", 3, 6)])
    mapping = syllable_token_map(spans, [(0, 6)])
    assert mapping["syllable_to_tokens"] == [[1, 2]]
    assert mapping["subwords_per_syllable_max"] == 2


def test_unmapped_syllable_is_reported():
    text = "toi di"
    spans = mock_spans(text, [("toi", 0, 3)])
    mapping = syllable_token_map(spans, [(0, 3), (4, 6)])
    assert mapping["unmapped_syllables"] == [1]
    assert not mapping["all_syllables_mapped"]


def test_token_straddling_two_syllables_is_flagged():
    """A single token covering two syllables makes its tone label ambiguous."""
    text = "toidi"
    spans = mock_spans(text, [("toidi", 0, 5)])
    mapping = syllable_token_map(spans, [(0, 3), (3, 5)])
    assert mapping["has_straddling_tokens"]
    assert mapping["straddling_tokens"] == [1]


def test_tokens_for_span_ignores_special_tokens():
    text = "abc"
    spans = mock_spans(text, [("abc", 0, 3)])
    assert tokens_for_span(spans, 0, 3) == [1]


# ---------------------------------------------------------------------------
# Alignment status
# ---------------------------------------------------------------------------
def test_alignment_status_aligned():
    text = "toi di"
    spans = mock_spans(text, [("toi", 0, 3), ("di", 4, 6)])
    availability, _ = validate_offsets(text, spans)
    status, reason = alignment_status(
        availability, character_coverage(text, spans), syllable_token_map(spans, [(0, 3), (4, 6)])
    )
    assert status is AlignmentStatus.PARTIAL  # the space is uncovered
    assert "covered by no token" in reason


def test_alignment_status_unaligned_without_offsets():
    spans = [TokenSpan(0, "abc", 1)]
    status, reason = alignment_status(OffsetAvailability.ABSENT, {}, {})
    assert status is AlignmentStatus.UNALIGNED
    assert "ABSENT" in reason


def test_alignment_status_unaligned_when_malformed():
    status, reason = alignment_status(OffsetAvailability.NATIVE_MALFORMED, {}, {})
    assert status is AlignmentStatus.UNALIGNED
    assert "malformed" in reason


def test_alignment_status_partial_when_a_syllable_is_unmapped():
    status, reason = alignment_status(
        OffsetAvailability.NATIVE_EXACT,
        {"fully_covered": True},
        {"all_syllables_mapped": False, "unmapped_syllables": [2]},
    )
    assert status is AlignmentStatus.PARTIAL
    assert "reached no token" in reason


def test_alignment_status_fully_aligned():
    status, _ = alignment_status(
        OffsetAvailability.NATIVE_EXACT,
        {"fully_covered": True},
        {"all_syllables_mapped": True},
    )
    assert status is AlignmentStatus.ALIGNED


# ---------------------------------------------------------------------------
# Grid invariance
# ---------------------------------------------------------------------------
def observation(condition, base, tokenizer_input, ids, *, path=PreprocessingPath.RAW_BASE,
                availability=PathAvailability.OK, case="c", tokens=None, error=None):
    return PathObservation(
        case_id=case, condition=condition, path=path, availability=availability,
        base_text=base, tokenizer_input=tokenizer_input,
        tokens=tuple(tokens or []), token_ids=tuple(ids), error=error,
    )


def test_grid_invariance_holds_when_ids_match():
    rows = [observation(c, "toi di hoc", "toi di hoc", (1, 2, 3)) for c in PROBE_CONDITIONS]
    result = grid_invariance(rows)
    assert result["satisfies_base_grid_invariance"]
    assert result["reference_condition"] == "FULL"
    assert result["token_ids_invariant"] and result["base_text_invariant"]


def test_grid_invariance_detects_a_preprocessing_break():
    """The scenario B3B-0 exists to catch: identical base text, different
    tokenizer input because segmentation reacted to the corruption level."""
    rows = [
        observation("FULL", "toi di hoc", "toi_di hoc", (1, 2)),
        observation("P100", "toi di hoc", "toi di hoc", (3, 4, 5)),
    ]
    result = grid_invariance(rows)
    assert result["base_text_invariant"], "B2 kept the base equal"
    assert not result["tokenizer_input_invariant"]
    assert not result["token_ids_invariant"]
    assert not result["satisfies_base_grid_invariance"]
    assert result["tokenizer_input_mismatches"] == ["P100"]


def test_grid_invariance_flags_a_base_text_break_separately():
    rows = [
        observation("FULL", "toi di hoc", "toi di hoc", (1, 2, 3)),
        observation("P50", "toi di lam", "toi di lam", (1, 2, 4)),
    ]
    result = grid_invariance(rows)
    assert not result["base_text_invariant"]
    assert result["base_text_mismatches"] == ["P50"]


def test_grid_invariance_is_not_comparable_without_usable_rows():
    rows = [observation("FULL", "x", "x", (1,), availability=PathAvailability.UNAVAILABLE_SEGMENTER)]
    assert grid_invariance(rows)["comparable"] is False


def test_errored_observations_are_excluded_from_comparison():
    rows = [
        observation("FULL", "toi", "toi", (1,)),
        observation("P50", "toi", "toi", (), error="boom"),
    ]
    result = grid_invariance(rows)
    assert result["conditions_compared"] == ["FULL"]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_path_summary_counts_availability_and_unknowns():
    text = "abc"
    spans = mock_spans(text, [("<unk>", 0, 3)])
    usable = PathObservation(
        case_id="c", condition="FULL", path=PreprocessingPath.RAW_BASE,
        availability=PathAvailability.OK, tokenizer_input=text, spans=tuple(spans),
        alignment=AlignmentStatus.ALIGNED, syllable_map={"syllable_count": 1},
    )
    missing = PathObservation(
        case_id="c", condition="FULL", path=PreprocessingPath.BASE_THEN_SEGMENT,
        availability=PathAvailability.UNAVAILABLE_SEGMENTER,
    )
    summary = path_summary([usable, missing])
    assert summary["usable"] == 1 and summary["unavailable"] == 1
    assert summary["unavailable_reasons"] == ["UNAVAILABLE_SEGMENTER"]
    assert summary["total_unknown_tokens"] == 1
    assert summary["alignment_rate"] == 1.0
    assert summary["mean_fragmentation"] == 1.0


def test_fragmentation_is_none_without_syllables():
    assert PathObservation(
        case_id="c", condition="FULL", path=PreprocessingPath.RAW_BASE,
        availability=PathAvailability.OK,
    ).fragmentation() is None


def test_compare_paths_reports_facts_and_makes_no_decision():
    by_path = {
        "RAW_BASE": [observation(c, "toi", "toi", (1,)) for c in PROBE_CONDITIONS],
        "BASE_THEN_SEGMENT": [
            observation("FULL", "toi", "toi_x", (1, 2)),
            observation("P100", "toi", "toi", (3,)),
        ],
    }
    comparison = compare_paths(by_path)
    assert comparison["decision"] == "NOT_MADE"
    assert "does not choose" in comparison["decision_note"]
    assert comparison["grid_invariance"]["RAW_BASE"]["all_cases_invariant"]
    assert not comparison["grid_invariance"]["BASE_THEN_SEGMENT"]["all_cases_invariant"]


def test_observation_serialises_to_json():
    text = "abc"
    obs = PathObservation(
        case_id="c", condition="FULL", path=PreprocessingPath.RAW_BASE,
        availability=PathAvailability.OK, tokenizer_input=text,
        spans=tuple(mock_spans(text, [("abc", 0, 3)])), token_ids=(1, 2, 3),
    )
    payload = json.dumps(obs.to_dict(), ensure_ascii=False)
    restored = json.loads(payload)
    assert restored["path"] == "RAW_BASE"
    assert restored["availability"] == "OK"
    assert restored["spans"][1]["token"] == "abc"


# ---------------------------------------------------------------------------
# Mixed / punctuation / underscore behaviour of the analysis
# ---------------------------------------------------------------------------
def test_underscore_segmented_input_maps_cleanly():
    text = "nghien_cuu xu_ly"
    spans = mock_spans(text, [("nghien_cuu", 0, 10), ("xu_ly", 11, 16)])
    mapping = syllable_token_map(spans, [(0, 6), (7, 10), (11, 13), (14, 16)])
    assert mapping["has_straddling_tokens"], "one token covering two syllables must be flagged"
    assert validate_offsets(text, spans)[0] is OffsetAvailability.NATIVE_EXACT


def test_punctuation_tokens_are_handled():
    text = "toi, di."
    spans = mock_spans(text, [("toi", 0, 3), (",", 3, 4), ("di", 5, 7), (".", 7, 8)])
    assert validate_offsets(text, spans)[0] is OffsetAvailability.NATIVE_EXACT
    assert character_coverage(text, spans)["uncovered_indices"] == [4]


def test_mixed_script_tokens_are_handled():
    text = "toi dung Python"
    spans = mock_spans(text, [("toi", 0, 3), ("dung", 4, 8), ("Python", 9, 15)])
    mapping = syllable_token_map(spans, [(0, 3), (4, 8), (9, 15)])
    assert mapping["all_syllables_mapped"]


# ---------------------------------------------------------------------------
# The local package must stay ML-free, and the probe must not load weights
# ---------------------------------------------------------------------------
def test_alignment_package_imports_no_ml_library():
    banned = {"torch", "transformers", "tokenizers", "sentencepiece", "datasets", "py_vncorenlp", "vncorenlp"}
    for path in (REPO_ROOT / "unmark" / "alignment").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, f"{path.name}: {node.module}"


def test_probe_script_never_loads_model_weights():
    """Tokenizer only: no AutoModel call anywhere, at any indentation."""
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in name, f"probe calls {name}.from_pretrained"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "transformers":
            for alias in node.names:
                assert "Model" not in alias.name, f"probe imports {alias.name}"


def test_probe_script_heavy_imports_are_lazy():
    """The probe must import cleanly in the ML-free local venv so it can print
    its own instructions instead of crashing."""
    for lineno, line in enumerate(PROBE_SCRIPT.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("import transformers", "from transformers", "import py_vncorenlp")):
            assert line.startswith((" ", "\t")), f"{PROBE_SCRIPT.name}:{lineno} heavy import at module level"


def test_probe_script_uses_a_repo_local_hf_cache():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert REPO_LOCAL_HF_CACHE == ".hf-cache"
    assert "HF_HOME" in source and REPO_LOCAL_HF_CACHE in source
    assert "~/.cache" not in source


def test_probe_declares_no_policy_decision():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "does not choose" in source.lower() or "not choose a policy" in source.lower()
    assert "D-B3B0-001" in source


def test_contracts_enumerate_every_candidate_path():
    names = {p.name for p in PreprocessingPath}
    assert {
        "RAW_BASE",
        "CLEAN_SEGMENT_THEN_BASE",
        "BASE_THEN_SEGMENT",
        "PRESEGMENTED_DATASET",
        "OBSERVED_SEGMENT_THEN_BASE",
    } <= names


def test_segmenter_contract_records_pinning_risk():
    contract = SegmenterContract(available=True, name="VnCoreNLP", pinned=False)
    assert contract.to_dict()["pinned"] is False


def test_tokenizer_contract_records_the_segmentation_expectation():
    contract = TokenizerContract(
        checkpoint="vinai/phobert-base", revision=None, tokenizer_class="PhobertTokenizer",
        is_fast=False, word_segmentation_expected=True,
    )
    payload = contract.to_dict()
    assert payload["word_segmentation_expected"] is True
    assert payload["revision"] is None, "an unpinned revision must be visible, not defaulted"
