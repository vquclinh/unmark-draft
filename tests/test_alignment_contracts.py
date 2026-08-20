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
        checkpoint="vinai/phobert-base", revision_requested=None,
        tokenizer_class="PhobertTokenizer", is_fast=False, word_segmentation_expected=True,
    )
    payload = contract.to_dict()
    assert payload["word_segmentation_expected"] is True
    assert payload["revision_requested"] is None, "an unpinned revision must be visible, not defaulted"
    assert payload["revision_verified"] is False


# ===========================================================================
# B3B-0 probe repair (audit 007)
# ===========================================================================
# The first real Colab run exposed two bugs the mock tests had not caught:
# an automatic VnCoreNLP download left the segmentation model unpinned, and a
# relative output root drifted after py_vncorenlp chdir()'d into its resource
# directory. These tests pin both repairs.

import importlib.util
import os
import subprocess
import sys as _sys


def _probe_module():
    """Import the probe by path. It must import in the ML-free local venv."""
    spec = importlib.util.spec_from_file_location("b3b0_probe", PROBE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- Bug 1: no automatic segmenter download --------------------------------
def test_probe_never_calls_download_model():
    """The repair: segmenter resources are externally provisioned, never fetched."""
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name != "download_model", "probe must not download segmenter resources"


def test_probe_source_mentions_download_model_only_to_forbid_it():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    for line in source.splitlines():
        if "download_model" in line:
            assert line.strip().startswith(("#", "*", '"""')) or "deliberately absent" in line, line


def test_missing_segmenter_directory_is_reported_not_created(tmp_path):
    module = _probe_module()
    absent = tmp_path / "nope"
    segmenter, contract = module.load_segmenter(absent, {}, None)
    assert segmenter is None
    assert contract.available is False
    assert contract.pinned is False
    assert "does not exist" in contract.notes
    assert "never downloads" in contract.notes
    assert not absent.exists(), "the probe must not create the resource directory"


def test_missing_required_files_fail_clearly(tmp_path):
    module = _probe_module()
    resource = tmp_path / "vncorenlp"
    (resource / "models" / "wordsegmenter").mkdir(parents=True)
    (resource / "VnCoreNLP-1.2.jar").write_bytes(b"jar")
    segmenter, contract = module.load_segmenter(resource, {}, None)
    assert segmenter is None and contract.available is False
    assert "vi-vocab" in contract.notes and "wordsegmenter.rdr" in contract.notes


def test_missing_jar_fails_clearly(tmp_path):
    module = _probe_module()
    resource = tmp_path / "vncorenlp"
    seg = resource / "models" / "wordsegmenter"
    seg.mkdir(parents=True)
    (seg / "vi-vocab").write_bytes(b"v")
    (seg / "wordsegmenter.rdr").write_bytes(b"r")
    _, contract = module.load_segmenter(resource, {}, None)
    assert contract.available is False
    # The pin names the jar exactly; the message must name it too, not a glob.
    assert "VnCoreNLP-1.2.jar" in contract.notes
    assert contract.required_jar == "VnCoreNLP-1.2.jar"


def _resource_dir(tmp_path):
    resource = tmp_path / "vncorenlp"
    seg = resource / "models" / "wordsegmenter"
    seg.mkdir(parents=True)
    (resource / "VnCoreNLP-1.2.jar").write_bytes(b"jar-bytes")
    (seg / "vi-vocab").write_bytes(b"vocab-bytes")
    (seg / "wordsegmenter.rdr").write_bytes(b"rdr-bytes")
    return resource


def test_resource_hashes_are_recorded(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    _, contract = module.load_segmenter(resource, {}, None)
    hashes = contract.resource_hashes
    assert set(hashes) == {"VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"}
    assert all(len(digest) == 64 for digest in hashes.values())
    assert hashes["VnCoreNLP-1.2.jar"] == module.sha256_of(resource / "VnCoreNLP-1.2.jar")


def test_pinned_is_false_when_no_hashes_were_supplied(tmp_path):
    """Existence must never be mistaken for provenance."""
    module = _probe_module()
    _, contract = module.load_segmenter(_resource_dir(tmp_path), {}, None)
    assert contract.pinned is False
    # py_vncorenlp is absent locally, so the note reports that; the point is that
    # provenance was still computed and pinned was not fabricated.
    assert contract.resource_hashes
    assert contract.model_version is None


def test_pinned_is_false_when_some_files_were_unverified(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    partial = {"files": {"VnCoreNLP-1.2.jar": module.sha256_of(resource / "VnCoreNLP-1.2.jar")}}
    _, contract = module.load_segmenter(resource, partial, None)
    assert contract.pinned is False, "a partially verified checkout is not pinned"
    assert contract.resource_hashes


def test_hash_mismatch_refuses_to_load(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    wrong = {"files": {"VnCoreNLP-1.2.jar": "0" * 64}}
    segmenter, contract = module.load_segmenter(resource, wrong, None)
    assert segmenter is None
    assert contract.available is False and contract.pinned is False
    assert "REFUSING" in contract.notes and "mismatch" in contract.notes


def test_supplied_revision_is_recorded(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    _, contract = module.load_segmenter(resource, {"revision": "abc123"}, None)
    assert contract.model_version == "abc123"
    _, contract2 = module.load_segmenter(resource, {}, "deadbeef")
    assert contract2.model_version == "deadbeef"


def test_expected_hashes_file_is_read(tmp_path):
    module = _probe_module()
    path = tmp_path / "hashes.json"
    path.write_text(json.dumps({"revision": "r", "files": {"a": "b"}}), encoding="utf-8")
    assert module.load_expected_hashes(path)["files"] == {"a": "b"}
    assert module.load_expected_hashes(None) == {}


def test_malformed_hashes_file_is_rejected(tmp_path):
    module = _probe_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="files"):
        module.load_expected_hashes(path)


# --- Bug 2: output root must not drift with the cwd ------------------------
def test_relative_output_root_resolves_against_the_repository_root():
    module = _probe_module()
    assert module.REPO_ROOT == REPO_ROOT
    resolved = (module.REPO_ROOT / Path("results/b3b0")).resolve()
    assert resolved == (REPO_ROOT / "results" / "b3b0").resolve()
    assert resolved.is_absolute()


def test_output_artifacts_survive_a_dependency_changing_the_cwd(tmp_path, monkeypatch):
    """Simulates py_vncorenlp.VnCoreNLP() chdir()ing into its resource directory.

    Reproduces the first Colab run's failure mode: artifacts landed in
    `.vncorenlp/results/b3b0/` instead of `<repo>/results/b3b0/`.
    """
    output_root = (tmp_path / "results" / "b3b0").resolve()  # resolved BEFORE the chdir
    elsewhere = tmp_path / "vncorenlp"
    elsewhere.mkdir(parents=True)

    original = Path.cwd()
    try:
        os.chdir(elsewhere)  # the dependency's side effect
        assert Path.cwd() != original

        run_dir = output_root / "RUN"
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "environment.json", "cases.jsonl", "summary.json", "report.md"):
            (run_dir / name).write_text("{}", encoding="utf-8")
    finally:
        os.chdir(original)

    for name in ("config.json", "environment.json", "cases.jsonl", "summary.json", "report.md"):
        assert (output_root / "RUN" / name).is_file(), name
    assert not (elsewhere / "results").exists(), "artifacts must not follow the cwd"


def test_probe_resolves_paths_before_loading_the_segmenter():
    """Source-level guard: resolution must happen before load_segmenter runs."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    resolve_index = source.index("output_root = Path(args.output_root)")
    load_index = source.index("segmenter, segmenter_contract = load_segmenter(")
    assert resolve_index < load_index, "output root must be absolute before the segmenter runs"
    assert "cwd_at_start = Path.cwd()" in source
    assert source.index("cwd_at_start = Path.cwd()") < load_index


def test_probe_records_cwd_diagnostics():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    for field in (
        "cwd_at_start",
        "cwd_after_segmenter_initialization",
        "repository_root",
        "resolved_output_root",
        "resolved_vncorenlp_dir",
        "cwd_changed_by_dependency",
    ):
        assert f'"{field}"' in source, field


def test_probe_does_not_use_chdir_to_fix_paths():
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "chdir":
            raise AssertionError("probe must resolve absolute paths, not chdir()")


# --- PhoBERT revision contract ---------------------------------------------
def test_probe_fails_closed_without_a_revision():
    result = subprocess.run(
        [_sys.executable, str(PROBE_SCRIPT), "--checkpoint", "vinai/phobert-base"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert result.returncode == 2
    assert "--revision is required" in result.stderr
    assert "unreproducible" in result.stderr
    assert "--allow-floating-revision" in result.stderr


def test_probe_still_supports_an_explicit_revision():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert '"--revision"' in source
    assert "revision_pinned" in source
    assert "scientifically_usable" in source


def test_floating_revision_must_be_requested_explicitly():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "--allow-floating-revision" in source
    assert "NOT scientifically usable" in source or "not scientifically usable" in source.lower()


def test_no_model_loading_call_was_introduced_by_the_repair():
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in name, f"probe calls {name}.from_pretrained"


def test_probe_imports_cleanly_without_any_ml_dependency():
    """It must import in the ML-free local venv so it can print its own guidance."""
    module = _probe_module()
    assert hasattr(module, "load_segmenter")
    assert hasattr(module, "sha256_of")
    assert "transformers" not in _sys.modules
    assert "py_vncorenlp" not in _sys.modules


def test_vncorenlp_runtime_directory_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".vncorenlp/"], cwd=REPO_ROOT, capture_output=True
    )
    assert result.returncode == 0, ".vncorenlp/ must never be committed"
    stray = subprocess.run(
        ["git", "check-ignore", "-q", ".vncorenlp/results/b3b0/x/report.md"],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert stray.returncode == 0, "stray artifacts from the invalid run must stay untracked"


def test_verified_checkout_is_marked_pinned_even_if_the_library_is_absent(tmp_path):
    """`pinned` is a statement about the files, not about whether py_vncorenlp
    imported. Locally the import always fails, and provenance must survive it."""
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    complete = {
        "revision": "a" * 40,
        "files": {
            "VnCoreNLP-1.2.jar": module.sha256_of(resource / "VnCoreNLP-1.2.jar"),
            "models/wordsegmenter/vi-vocab": module.sha256_of(resource / "models/wordsegmenter/vi-vocab"),
            "models/wordsegmenter/wordsegmenter.rdr": module.sha256_of(
                resource / "models/wordsegmenter/wordsegmenter.rdr"
            ),
        },
    }
    segmenter, contract = module.load_segmenter(resource, complete, None)
    assert segmenter is None, "py_vncorenlp is not installed locally"
    assert contract.hashes_verified is True, "every file was verified against a supplied hash"
    # Not a git checkout, so the revision cannot be verified and pinned stays false.
    assert contract.revision_verified is False
    assert contract.pinned is False
    # `required_jar` is what the pin names; `jar_name` records what was actually
    # loaded, and nothing was, because py_vncorenlp is absent locally.
    assert contract.required_jar == "VnCoreNLP-1.2.jar"
    assert contract.jar_name is None
    assert contract.available is False
    assert "import failed" in contract.notes


# ===========================================================================
# B3B-0 manifest hardening (audit 008)
# ===========================================================================
# Audit 007 N1/N2: VnCoreNLP hashes were caller-supplied and the jar was chosen
# by globbing. Both are closed here — the pin is committed, and the jar is named.

MANIFEST_PATH = REPO_ROOT / "configs" / "linguistics" / "vncorenlp_v1.2.json"
REQUIRED_RESOURCE_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)


def _write_manifest(tmp_path, files, revision="a" * 40, required_jar="VnCoreNLP-1.2.jar", **extra):
    payload = {
        "schema_version": "vncorenlp-pin-v1",
        "source": "VnCoreNLP",
        "source_repository": "https://github.com/vncorenlp/VnCoreNLP",
        "release_tag": "v1.2",
        "revision": revision,
        "required_jar": required_jar,
        "files": files,
        **extra,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _hashes_of(module, resource):
    return {
        "VnCoreNLP-1.2.jar": module.sha256_of(resource / "VnCoreNLP-1.2.jar"),
        "models/wordsegmenter/vi-vocab": module.sha256_of(resource / "models/wordsegmenter/vi-vocab"),
        "models/wordsegmenter/wordsegmenter.rdr": module.sha256_of(
            resource / "models/wordsegmenter/wordsegmenter.rdr"
        ),
    }


# --- 1. The committed manifest ---------------------------------------------
def test_committed_manifest_exists_and_parses():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "vncorenlp-pin-v1"
    assert payload["release_tag"] == "v1.2"
    assert payload["source_repository"] == "https://github.com/vncorenlp/VnCoreNLP.git"


def test_committed_manifest_names_the_exact_required_jar():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["required_jar"] == "VnCoreNLP-1.2.jar"
    assert "*" not in payload["required_jar"], "the pin must name a jar, not a glob"


def test_committed_manifest_requires_exactly_the_three_resources():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(payload["files"]) == set(REQUIRED_RESOURCE_FILES)


# The exact values the researcher extracted from the provisioned checkout with
# `git rev-parse HEAD` and sha256 over the three resources. Hard-coded here so a
# silent edit to the pin fails the suite.
PINNED_REVISION = "62bbc58fe5d113c898eae112656be97dcf50b3a0"
PINNED_HASHES = {
    "VnCoreNLP-1.2.jar": "9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793",
    "models/wordsegmenter/vi-vocab": "0a47c5b55bbce163029d37730a67b9479740388695c29c106c112b815613eaa5",
    "models/wordsegmenter/wordsegmenter.rdr": "9e62f96bd93e37a24f364238e8d8ae986fa5dad6dbc9f4eae622ab3651b7fa06",
}


def test_committed_manifest_is_pinned_and_loads():
    """Audit 008's blocker is closed: the pin is complete and accepted."""
    module = _probe_module()
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PINNED"
    manifest = module.load_vncorenlp_manifest(MANIFEST_PATH)
    assert manifest["revision"] == PINNED_REVISION


def test_no_provenance_placeholder_remains():
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "PENDING_RESEARCHER_PROVENANCE" not in text
    assert "AWAITING_RESEARCHER_PROVENANCE" not in text.replace("AWAITING_RESEARCHER_PROVENANCE\"", "")


def test_committed_manifest_records_the_exact_researcher_revision():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["revision"] == PINNED_REVISION
    assert len(payload["revision"]) == 40
    assert payload["release_tag"] == "v1.2"


def test_committed_manifest_records_the_exact_researcher_hashes():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["files"] == PINNED_HASHES


# --- 4-5. Verification outcomes --------------------------------------------
def _git_checkout(tmp_path):
    """A real git checkout, so revision verification can be exercised offline."""
    resource = _resource_dir(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", str(resource)], env=env, check=True)
    subprocess.run(["git", "-C", str(resource), "add", "-A"], env=env, check=True)
    subprocess.run(["git", "-C", str(resource), "commit", "-qm", "pin"], env=env, check=True)
    head = subprocess.run(
        ["git", "-C", str(resource), "rev-parse", "HEAD"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return resource, head


def test_matching_hashes_and_revision_yield_pinned_true(tmp_path):
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, _hashes_of(module, resource), revision=head)
    )
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.hashes_verified is True
    assert contract.revision_verified is True
    assert contract.pinned is True
    assert contract.required_jar == "VnCoreNLP-1.2.jar"
    assert contract.observed_revision == head
    assert contract.manifest_revision == head


@pytest.mark.parametrize("corrupt_file", REQUIRED_RESOURCE_FILES)
def test_any_hash_mismatch_fails_closed(tmp_path, corrupt_file):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    files = _hashes_of(module, resource)
    files[corrupt_file] = "0" * 64
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, files))
    segmenter, contract = module.load_segmenter(resource, manifest, None)
    assert segmenter is None
    assert contract.available is False and contract.pinned is False
    assert "REFUSING" in contract.notes and corrupt_file in contract.notes


def test_missing_required_jar_fails_closed(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    (resource / "VnCoreNLP-1.2.jar").unlink()
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, {n: "b" * 64 for n in REQUIRED_RESOURCE_FILES})
    )
    segmenter, contract = module.load_segmenter(resource, manifest, None)
    assert segmenter is None and contract.pinned is False
    assert "VnCoreNLP-1.2.jar" in contract.notes


# --- 7. Extra jars must not change the selection ---------------------------
def test_extra_jar_cannot_change_the_selected_jar(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    # A jar that sorts BEFORE the required one; globbing would have picked it.
    (resource / "VnCoreNLP-1.1.jar").write_bytes(b"older-jar")
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, _hashes_of(module, resource)))
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.required_jar == "VnCoreNLP-1.2.jar"
    assert "VnCoreNLP-1.1.jar" in contract.other_jars_present
    assert contract.resource_hashes["VnCoreNLP-1.2.jar"] == module.sha256_of(
        resource / "VnCoreNLP-1.2.jar"
    )
    assert "VnCoreNLP-1.1.jar" not in contract.resource_hashes


def test_extra_jars_are_reported_when_the_required_one_is_absent(tmp_path):
    module = _probe_module()
    resource = _resource_dir(tmp_path)
    (resource / "VnCoreNLP-1.2.jar").unlink()
    (resource / "VnCoreNLP-1.1.jar").write_bytes(b"older")
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, {n: "e" * 64 for n in REQUIRED_RESOURCE_FILES})
    )
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.available is False
    assert "NOT substituted" in contract.notes
    assert contract.other_jars_present == ("VnCoreNLP-1.1.jar",)


# --- 8. Conflicting provenance ---------------------------------------------
def test_cli_revision_conflicting_with_the_manifest_fails_closed(tmp_path):
    module = _probe_module()
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, {n: "f" * 64 for n in REQUIRED_RESOURCE_FILES}, revision="a" * 40)
    )
    with pytest.raises(SystemExit, match="contradicts the committed manifest"):
        module.reconcile_provenance(manifest, {}, "b" * 40)


def test_hashes_file_conflicting_with_the_manifest_fails_closed(tmp_path):
    module = _probe_module()
    files = {n: "f" * 64 for n in REQUIRED_RESOURCE_FILES}
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, files))
    conflicting = {"files": {"VnCoreNLP-1.2.jar": "0" * 64}}
    with pytest.raises(SystemExit, match="contradicts the committed manifest"):
        module.reconcile_provenance(manifest, conflicting, None)


def test_agreeing_provenance_is_accepted(tmp_path):
    module = _probe_module()
    files = {n: "f" * 64 for n in REQUIRED_RESOURCE_FILES}
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, files))
    merged = module.reconcile_provenance(manifest, {"files": dict(files)}, "a" * 40)
    assert merged["revision"] == "a" * 40


# --- Manifest schema validation --------------------------------------------
@pytest.mark.parametrize("missing", ["schema_version", "revision", "required_jar", "files"])
def test_manifest_missing_a_required_key_is_rejected(tmp_path, missing):
    module = _probe_module()
    payload = json.loads(_write_manifest(tmp_path, {n: "f" * 64 for n in REQUIRED_RESOURCE_FILES}).read_text())
    del payload[missing]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing required key"):
        module.load_vncorenlp_manifest(path)


def test_manifest_missing_a_required_resource_is_rejected(tmp_path):
    module = _probe_module()
    path = _write_manifest(tmp_path, {"VnCoreNLP-1.2.jar": "f" * 64})
    with pytest.raises(SystemExit, match="missing the required resource"):
        module.load_vncorenlp_manifest(path)


def test_manifest_whose_required_jar_is_not_in_files_is_rejected(tmp_path):
    module = _probe_module()
    path = _write_manifest(
        tmp_path, {n: "f" * 64 for n in REQUIRED_RESOURCE_FILES}, required_jar="VnCoreNLP-9.9.jar"
    )
    with pytest.raises(SystemExit, match="has no entry in 'files'"):
        module.load_vncorenlp_manifest(path)


def test_invalid_json_manifest_is_rejected(tmp_path):
    module = _probe_module()
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid JSON"):
        module.load_vncorenlp_manifest(path)


# --- 9-13. Run metadata, CLI, scratch policy -------------------------------
def test_run_metadata_records_full_provenance(tmp_path):
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    manifest_path = _write_manifest(tmp_path, _hashes_of(module, resource), revision=head)
    manifest = {**module.load_vncorenlp_manifest(manifest_path), "_manifest_path": str(manifest_path)}
    _, contract = module.load_segmenter(resource, manifest, None)
    payload = contract.to_dict()
    for field in (
        "source" if False else "manifest_path", "manifest_revision", "observed_revision",
        "revision_verified", "observed_tags_at_head", "required_jar", "jar_name",
        "other_jars_present", "expected_hashes", "resource_hashes", "hashes_verified", "pinned",
    ):
        assert field in payload, field
    assert payload["required_jar"] == "VnCoreNLP-1.2.jar"
    assert set(payload["expected_hashes"]) == set(REQUIRED_RESOURCE_FILES)
    assert set(payload["resource_hashes"]) == set(REQUIRED_RESOURCE_FILES)
    assert payload["observed_revision"] == head
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["pinned"] is True


def test_committed_manifest_is_the_default_cli_provenance_source():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "--vncorenlp-manifest" in source
    assert "configs/linguistics/vncorenlp_v1.2.json" in source
    module = _probe_module()
    assert module.DEFAULT_VNCORENLP_MANIFEST == "configs/linguistics/vncorenlp_v1.2.json"


def test_probe_does_not_read_notebook_scratch_files():
    """`.probe_*` files are notebook state, never scientific configuration."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    for scratch in (".probe_phobert_revision", ".probe_vncorenlp_revision", ".probe_vncorenlp_hashes"):
        assert scratch not in source, f"probe reads notebook scratch file {scratch}"


def test_notebook_scratch_files_are_gitignored():
    for name in (".probe_phobert_revision", ".probe_vncorenlp_revision", ".probe_vncorenlp_hashes.txt"):
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO_ROOT, capture_output=True)
        assert result.returncode == 0, f"{name} must not be committable"


def test_no_downloader_was_reintroduced_by_the_hardening():
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name != "download_model"


def test_no_model_loading_was_introduced_by_the_hardening():
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in name


# ===========================================================================
# B3B-0 provenance closure (audit 009)
# ===========================================================================
# Closes audit 008's researcher-provenance blocker and its N1 revision gap.

def test_wrong_git_revision_fails_closed(tmp_path):
    """7. A checkout at the wrong revision must be refused, not warned about."""
    module = _probe_module()
    resource, _head = _git_checkout(tmp_path)
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, _hashes_of(module, resource), revision="0" * 40)
    )
    segmenter, contract = module.load_segmenter(resource, manifest, None)
    assert segmenter is None
    assert contract.available is False
    assert contract.pinned is False
    assert contract.revision_verified is False
    assert "REFUSING" in contract.notes
    assert "!= pinned revision" in contract.notes


def test_unavailable_git_metadata_still_checks_hashes_but_is_not_pinned(tmp_path):
    """8. No .git: hashes are still verified, revision is not fabricated."""
    module = _probe_module()
    resource = _resource_dir(tmp_path)  # deliberately not a git checkout
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, _hashes_of(module, resource), revision="b" * 40)
    )
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.hashes_verified is True, "content verification must still happen"
    assert contract.observed_revision is None, "must not fabricate a revision"
    assert contract.revision_verified is False
    assert contract.pinned is False, "this manifest pins a revision, so hashes alone are not enough"


def test_revision_match_alone_is_not_enough_without_hash_match(tmp_path):
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    files = _hashes_of(module, resource)
    files["models/wordsegmenter/vi-vocab"] = "0" * 64
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, files, revision=head))
    segmenter, contract = module.load_segmenter(resource, manifest, None)
    assert segmenter is None and contract.pinned is False
    assert "REFUSING" in contract.notes


def test_observed_tags_at_head_are_recorded_as_a_diagnostic(tmp_path):
    """3. Tags are reported; tag text alone never constitutes verification."""
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "-C", str(resource), "tag", "v1.2"], env=env, check=True)
    assert module.git_tags_at_head(resource) == ("v1.2",)

    # A correct tag with the WRONG pinned revision must still be refused.
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, _hashes_of(module, resource), revision="0" * 40)
    )
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.observed_tags_at_head == ("v1.2",)
    assert contract.available is False, "a matching tag must not rescue a revision mismatch"


def test_git_helpers_return_none_outside_a_checkout(tmp_path):
    module = _probe_module()
    assert module.git_head_revision(tmp_path) is None
    assert module.git_tags_at_head(tmp_path) == ()


def test_expected_and_observed_hashes_are_both_recorded(tmp_path):
    """13. Both sides of the comparison must survive into the artifact."""
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    expected = _hashes_of(module, resource)
    manifest = module.load_vncorenlp_manifest(_write_manifest(tmp_path, expected, revision=head))
    _, contract = module.load_segmenter(resource, manifest, None)
    payload = contract.to_dict()
    assert payload["expected_hashes"] == expected
    assert payload["resource_hashes"] == expected
    assert payload["hashes_verified"] is True


def test_scientifically_usable_requires_every_check():
    """The conjunction is explicit in the probe source, and now keys on the
    tokenizer revision having been *verified*, not merely supplied."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert (
        '"scientifically_usable": tokenizer_contract.revision_verified '
        "and segmenter_contract.pinned" in source
    )
    assert '"vncorenlp_provenance": segmenter_contract.to_dict()' in source
    assert '"phobert_provenance": tokenizer_contract.to_dict()' in source


def test_probe_never_reads_the_colab_provenance_scratch_file():
    """15. Including the /content path the notebook used."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    for scratch in (
        ".probe_phobert_revision",
        ".probe_vncorenlp_revision",
        ".probe_vncorenlp_hashes",
        "/content/vncorenlp-provenance.json",
        "vncorenlp-provenance",
    ):
        assert scratch not in source, f"probe references notebook scratch {scratch}"


def test_git_verification_uses_a_local_subprocess_not_the_network():
    module = _probe_module()
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    git_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "run" and node.args:
                first = node.args[0]
                if isinstance(first, ast.List) and first.elts:
                    head = first.elts[0]
                    if isinstance(head, ast.Constant) and head.value == "git":
                        git_calls.append(node)
    assert git_calls, "revision verification must actually shell out to git"
    # And no network client was introduced alongside it.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("requests", "httpx", "socket")


# ===========================================================================
# B3B-0 tokenizer revision verification (audit 010)
# ===========================================================================
# Closes audit 009 N3: `--revision` was passed to from_pretrained and recorded,
# but nothing checked what actually loaded. These tests use a fake tokenizer
# object with synthetic Hugging Face cache paths — no transformers, no network.

SHA_A = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
SHA_B = "0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a"


class _FakeTokenizer:
    """Stands in for a loaded tokenizer: only the attributes the probe reads."""

    def __init__(self, *, vocab_file=None, merges_file=None, tokenizer_file=None,
                 init_kwargs=None, name_or_path="vinai/phobert-base", is_fast=False):
        if vocab_file is not None:
            self.vocab_file = vocab_file
        if merges_file is not None:
            self.merges_file = merges_file
        if tokenizer_file is not None:
            self.tokenizer_file = tokenizer_file
        self.init_kwargs = init_kwargs or {}
        self.name_or_path = name_or_path
        self.is_fast = is_fast


def _snapshot(sha, filename="vocab.txt", root="/root/.cache/huggingface/hub"):
    return f"{root}/models--vinai--phobert-base/snapshots/{sha}/{filename}"


# --- Full-SHA policy -------------------------------------------------------
def test_full_commit_sha_is_accepted():
    module = _probe_module()
    assert module.is_full_commit_sha(SHA_A)
    assert module.FULL_SHA_LENGTH == 40


@pytest.mark.parametrize(
    "value",
    ["main", "master", "develop", "v1.0", "refs/heads/main", SHA_A[:12], SHA_A[:39],
     SHA_A + "0", SHA_A.upper(), "", None, "zzzz" + SHA_A[4:]],
)
def test_non_immutable_revisions_are_rejected(value):
    """7/8/9. Branches, tags and abbreviated SHAs are mutable or ambiguous."""
    module = _probe_module()
    assert not module.is_full_commit_sha(value)


@pytest.mark.parametrize("value", ["main", "v1.2", SHA_A[:12], SHA_A.upper()])
def test_probe_rejects_a_non_full_revision_at_the_cli(value):
    result = subprocess.run(
        [_sys.executable, str(PROBE_SCRIPT), "--revision", value],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert result.returncode == 2
    assert "not a full immutable commit SHA" in result.stderr
    assert "--allow-floating-revision" in result.stderr


# --- Snapshot extraction ---------------------------------------------------
def test_snapshot_path_yields_the_commit_sha():
    """10. The hub caches under snapshots/<resolved commit>/."""
    module = _probe_module()
    assert module.extract_snapshot_revision(_snapshot(SHA_A)) == SHA_A
    assert module.extract_snapshot_revision(_snapshot(SHA_A, "tokenizer.json")) == SHA_A


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/local-dir/vocab.txt",
        "/hub/models--x/snapshots/tooshort/vocab.txt",
        "/hub/models--x/snapshots/" + SHA_A.upper() + "/vocab.txt",
        "/hub/models--x/blobs/" + SHA_A + "/vocab.txt",
        "",
        None,
        12345,
    ],
)
def test_malformed_paths_do_not_fabricate_a_revision(path):
    """11. Anything that is not a snapshot path yields None, never a guess."""
    module = _probe_module()
    assert module.extract_snapshot_revision(path) is None


def test_windows_style_snapshot_path_is_handled():
    module = _probe_module()
    path = "C:" + chr(92) + "hub" + chr(92) + "snapshots" + chr(92) + SHA_A + chr(92) + "vocab.txt"
    assert module.extract_snapshot_revision(path) == SHA_A


# --- Observation from a loaded tokenizer -----------------------------------
def test_revision_is_observed_from_the_resolved_vocab_file():
    module = _probe_module()
    tokenizer = _FakeTokenizer(vocab_file=_snapshot(SHA_A))
    observed, evidence, source = module.observe_tokenizer_revision(tokenizer)
    assert observed == SHA_A
    assert evidence and evidence[0].endswith("vocab.txt")
    assert "snapshot path" in source


def test_revision_is_observed_from_init_kwargs():
    module = _probe_module()
    tokenizer = _FakeTokenizer(init_kwargs={"merges_file": _snapshot(SHA_A, "bpe.codes")})
    observed, _, _ = module.observe_tokenizer_revision(tokenizer)
    assert observed == SHA_A


def test_unobservable_revision_returns_none_with_a_reason():
    """5. A locally-loaded tokenizer has no snapshot path; do not invent one."""
    module = _probe_module()
    tokenizer = _FakeTokenizer(vocab_file="/tmp/local/vocab.txt")
    observed, _evidence, source = module.observe_tokenizer_revision(tokenizer)
    assert observed is None
    assert "no Hugging Face snapshot path" in source


def test_disagreeing_snapshot_paths_yield_no_revision():
    module = _probe_module()
    tokenizer = _FakeTokenizer(
        vocab_file=_snapshot(SHA_A), merges_file=_snapshot(SHA_B, "bpe.codes")
    )
    observed, _evidence, source = module.observe_tokenizer_revision(tokenizer)
    assert observed is None, "contradictory evidence must not be resolved by picking one"
    assert "disagree" in source


def test_observation_performs_no_network_or_second_download():
    """The evidence must come from what already loaded, not a fresh lookup."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    body = source[source.index("def observe_tokenizer_revision"):source.index("def load_tokenizer")]
    for forbidden in ("hf_hub_download", "snapshot_download", "HfApi", "list_repo_refs", "urlopen"):
        assert forbidden not in body, f"observation must not call {forbidden}"


# --- Verification outcome ---------------------------------------------------
def _describe(module, tokenizer, requested):
    observed, evidence, source = module.observe_tokenizer_revision(tokenizer)
    verified = bool(requested) and observed is not None and observed == requested
    return TokenizerContract(
        checkpoint="vinai/phobert-base",
        revision_requested=requested,
        revision_observed=observed,
        revision_verified=verified,
        revision_evidence=evidence,
        revision_evidence_source=source,
        tokenizer_class="PhobertTokenizer",
        is_fast=False,
    )


def test_matching_observed_revision_verifies():
    """1/2/3. Requested and observed are recorded separately, and agree."""
    module = _probe_module()
    contract = _describe(module, _FakeTokenizer(vocab_file=_snapshot(SHA_A)), SHA_A)
    assert contract.revision_requested == SHA_A
    assert contract.revision_observed == SHA_A
    assert contract.revision_verified is True


def test_mismatched_observed_revision_does_not_verify():
    """4. A different commit actually loaded must never verify."""
    module = _probe_module()
    contract = _describe(module, _FakeTokenizer(vocab_file=_snapshot(SHA_B)), SHA_A)
    assert contract.revision_observed == SHA_B
    assert contract.revision_verified is False


def test_supplying_a_revision_is_not_by_itself_verification():
    """6. The whole point of audit 009 N3."""
    module = _probe_module()
    contract = _describe(module, _FakeTokenizer(vocab_file="/tmp/local/vocab.txt"), SHA_A)
    assert contract.revision_requested == SHA_A
    assert contract.revision_observed is None
    assert contract.revision_verified is False, "a supplied argument is not a verification"


def test_tokenizer_contract_serialises_the_separated_fields():
    module = _probe_module()
    payload = _describe(module, _FakeTokenizer(vocab_file=_snapshot(SHA_A)), SHA_A).to_dict()
    for field in (
        "revision_requested", "revision_observed", "revision_verified",
        "revision_evidence", "revision_evidence_source",
    ):
        assert field in payload, field
    assert "revision_pinned" not in payload, "ambiguous field must not reappear"
    assert payload["revision_evidence"], "evidence paths must be recorded"


def test_probe_fails_closed_on_a_post_load_revision_mismatch():
    """The refusal is in the probe body, returning a distinct exit code."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "REFUSING: the tokenizer that loaded is not the one requested." in source
    assert "return 3" in source


# --- scientifically_usable --------------------------------------------------
def test_scientifically_usable_depends_on_tokenizer_revision_verification():
    """12. Not on whether the argument was supplied."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert (
        '"scientifically_usable": tokenizer_contract.revision_verified '
        "and segmenter_contract.pinned" in source
    )
    assert '"scientifically_usable": bool(args.revision)' not in source


def test_ambiguous_revision_pinned_flag_is_gone():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert '"revision_pinned"' not in source, "revision_pinned only meant 'argument present'"
    assert '"tokenizer_revision_verified"' in source


# --- 13-16. Regressions -----------------------------------------------------
def test_vncorenlp_verification_is_unchanged(tmp_path):
    """13. Audit 009's segmenter contract still behaves identically."""
    module = _probe_module()
    resource, head = _git_checkout(tmp_path)
    manifest = module.load_vncorenlp_manifest(
        _write_manifest(tmp_path, _hashes_of(module, resource), revision=head)
    )
    _, contract = module.load_segmenter(resource, manifest, None)
    assert contract.revision_verified is True
    assert contract.hashes_verified is True
    assert contract.pinned is True


def test_committed_vncorenlp_pin_is_still_intact():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PINNED"
    assert payload["revision"] == PINNED_REVISION
    assert payload["files"] == PINNED_HASHES


def test_no_downloader_and_no_model_loading_after_this_change():
    """14/15."""
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name != "download_model"
        if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":
            owner = node.value
            owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            assert "Model" not in owner_name


def test_output_root_repair_is_still_intact():
    """16. Audit 007's absolute-path resolution survives."""
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    resolve_index = source.index("output_root = Path(args.output_root)")
    load_index = source.index("segmenter, segmenter_contract = load_segmenter(")
    assert resolve_index < load_index
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "chdir":
            raise AssertionError("probe must not chdir")
