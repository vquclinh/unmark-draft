"""Offline tests for the B2 deterministic corruption engine.

No torch, no transformers, no tokenizer, no corpus, no network. Generated
material here is implementation verification, never a dataset or benchmark.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from unmark.corruption import (
    ACTIVE_ELIGIBILITY_POLICY,
    CorruptionPurpose,
    EligibilityPolicy,
    EligibilityUnresolved,
    is_resolved,
    require_resolved_eligibility,
    CONDITIONS,
    CORRUPTION_SCHEMA_VERSION,
    UNIMPLEMENTED_CONDITIONS,
    CorruptionScope,
    UnknownCondition,
    corrupt,
    corrupt_batch,
    get_condition,
    is_selected,
    text_identity,
    unit_score,
)
from unmark.orthography import (
    Eligibility,
    LetterDiacritic,
    ObservedTone,
    Tone,
    canon,
    decompose,
    strip_to_base,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every test here is implementation verification, so it opts in explicitly to the
# provisional candidate-span fallback. Under the default SCIENTIFIC purpose these
# calls raise, which is the point of the guard.
SELF_CHECK = CorruptionPurpose.SELF_CHECK

VI = "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."
TONES = "má mà mả mã mạ ma"
# A deterministically generated sequence long enough that a seed collision is
# negligible, without being probabilistic: the content is fixed.
LONG = " ".join(["má", "mà", "mả", "mã", "mạ", "phở", "cứu", "học", "tự", "ngữ"] * 8)

MIXED_TEXTS = [
    VI,
    TONES,
    "đường ăn cân ơn ưu êm ôm",
    "ĐẠI HỌC KHOA HỌC TỰ NHIÊN",
    "toi dung Python va PyTorch",
    "Năm 2026, GDP tăng 6,5% (VAT 10%)!",
    "Xem tại https://example.edu.vn/a?id=42&lang=vi",
    "Liên hệ qua lien.he@example.com nhé",
    "hôm nay tôi rất vui 😄🎉",
    "Müller façade naïve",
    unicodedata.normalize("NFD", "Tiếng Việt"),
    "2026 6,5% !!!",
    "phở",
    "",
    "   ",
]


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


# ---------------------------------------------------------------------------
# 1-2. Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize("text", MIXED_TEXTS)
def test_identical_inputs_give_identical_output(text, condition):
    a = corrupt(text, condition, seed=7, sample_id="s", purpose=SELF_CHECK)
    b = corrupt(text, condition, seed=7, sample_id="s", purpose=SELF_CHECK)
    assert a.corrupted_text == b.corrupted_text
    assert [d.selected for d in a.decisions] == [d.selected for d in b.decisions]
    assert [d.score for d in a.decisions] == [d.score for d in b.decisions]
    assert a.to_dict() == b.to_dict()


def test_output_is_identical_in_a_fresh_python_process():
    """Guards against any dependence on process state, including PYTHONHASHSEED."""
    code = (
        "import json,sys;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from unmark.corruption import corrupt, CorruptionPurpose;"
        f"r = corrupt({LONG!r}, 'P50', seed=99, sample_id='fresh',"
        " purpose=CorruptionPurpose.SELF_CHECK);"
        "print(json.dumps({'text': r.corrupted_text,"
        " 'selected': [d.selected for d in r.decisions],"
        " 'scores': [d.score for d in r.decisions]}))"
    )
    outputs = []
    for hashseed in ("0", "1", "random"):
        env = {"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))

    local = corrupt(LONG, "P50", seed=99, sample_id="fresh", purpose=SELF_CHECK)
    assert all(o == outputs[0] for o in outputs), "output varied with PYTHONHASHSEED"
    assert outputs[0]["text"] == local.corrupted_text
    assert outputs[0]["selected"] == [d.selected for d in local.decisions]


def test_no_builtin_hash_or_global_rng_in_the_corruption_package():
    """36/37. Reproducibility-critical code must not use hash() or a global RNG."""
    import ast

    for path in (REPO_ROOT / "unmark" / "corruption").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "hash", f"{path.name}: builtin hash() is process-randomised"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("random", "numpy"), f"{path.name} imports {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in ("random", "numpy"), f"{path.name}: {node.module}"


def test_scores_do_not_depend_on_neighbouring_units():
    """A sequential RNG would make unit k depend on units before it."""
    identity = text_identity("anything")
    scores = [unit_score(seed=3, sample_id="s", identity=identity, unit_index=i) for i in range(10)]
    again = [unit_score(seed=3, sample_id="s", identity=identity, unit_index=i) for i in reversed(range(10))]
    assert scores == list(reversed(again))


# ---------------------------------------------------------------------------
# 3-5. Seed, sample_id, ordering
# ---------------------------------------------------------------------------
def test_a_different_seed_changes_selection():
    a = corrupt(LONG, "P50", seed=1, sample_id="a", purpose=SELF_CHECK)
    b = corrupt(LONG, "P50", seed=2, sample_id="a", purpose=SELF_CHECK)
    assert a.candidate_units >= 60
    assert [d.selected for d in a.decisions] != [d.selected for d in b.decisions]
    assert a.corrupted_text != b.corrupted_text


def test_a_different_sample_id_changes_selection():
    a = corrupt(LONG, "P50", seed=1, sample_id="a", purpose=SELF_CHECK)
    b = corrupt(LONG, "P50", seed=1, sample_id="b", purpose=SELF_CHECK)
    assert [d.selected for d in a.decisions] != [d.selected for d in b.decisions]
    assert a.corrupted_text != b.corrupted_text


def test_dataset_reordering_does_not_change_any_sample():
    samples = [(text, f"id-{i}") for i, text in enumerate(MIXED_TEXTS)]
    forward = {r.sample_id: r.to_dict() for r in corrupt_batch(samples, "P50", seed=5, purpose=SELF_CHECK)}
    backward = {r.sample_id: r.to_dict() for r in corrupt_batch(list(reversed(samples)), "P50", seed=5, purpose=SELF_CHECK)}
    shuffled = [samples[i] for i in (3, 0, 7, 1, 9, 2)]
    partial = {r.sample_id: r.to_dict() for r in corrupt_batch(shuffled, "P50", seed=5, purpose=SELF_CHECK)}
    assert forward == backward
    for sample_id, record in partial.items():
        assert record == forward[sample_id]


def test_sample_id_not_row_index_determines_corruption():
    """Two rows with the same text but different ids corrupt differently; the
    same id at a different position corrupts identically."""
    first = corrupt(LONG, "P50", seed=5, sample_id="stable", purpose=SELF_CHECK)
    assert first.corrupted_text != corrupt(LONG, "P50", seed=5, sample_id="other", purpose=SELF_CHECK).corrupted_text
    batch = corrupt_batch([("x", "a"), (LONG, "stable")], "P50", seed=5, purpose=SELF_CHECK)
    assert batch[1].corrupted_text == first.corrupted_text


# ---------------------------------------------------------------------------
# 6. Canonical variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize("a,b", [("hòa", "hoà"), ("thúy", "thuý"), ("khỏe", "khoẻ")])
def test_placement_variants_corrupt_identically(a, b, condition):
    ra = corrupt(a, condition, seed=11, sample_id="v", purpose=SELF_CHECK)
    rb = corrupt(b, condition, seed=11, sample_id="v", purpose=SELF_CHECK)
    assert ra.canonical_clean_text == rb.canonical_clean_text
    assert ra.text_identity == rb.text_identity
    assert ra.corrupted_text == rb.corrupted_text
    assert [d.selected for d in ra.decisions] == [d.selected for d in rb.decisions]


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_nfc_and_nfd_sources_corrupt_identically(condition):
    a = corrupt(_nfd(VI), condition, seed=11, sample_id="v", purpose=SELF_CHECK)
    b = corrupt(VI, condition, seed=11, sample_id="v", purpose=SELF_CHECK)
    assert a.corrupted_text == b.corrupted_text
    assert a.canonical_clean_text == b.canonical_clean_text


def test_original_text_is_never_mutated():
    nfd_text = _nfd("Tiếng Việt")
    result = corrupt(nfd_text, "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.original_text == nfd_text
    assert result.canonical_clean_text == canon(nfd_text) != nfd_text


# ---------------------------------------------------------------------------
# 7-12. Condition semantics
# ---------------------------------------------------------------------------
def test_full_returns_canonical_clean_text_unchanged():
    result = corrupt(VI, "FULL", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.corrupted_text == canon(VI)
    assert result.selected_candidates == 0
    assert result.modified_candidates == 0
    assert result.requested_probability == 0.0
    assert result.candidate_selection_rate == 0.0
    assert result.is_unchanged


@pytest.mark.parametrize("condition,probability", [("P25", 0.25), ("P50", 0.50), ("P75", 0.75), ("P100", 1.0)])
def test_probability_conditions_record_the_requested_rate(condition, probability):
    result = corrupt(LONG, condition, seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.requested_probability == probability
    assert result.condition.scope is CorruptionScope.TONE


def test_realized_probability_tracks_the_requested_rate_on_a_large_input():
    """Not a statistical claim: LONG is fixed, so these values are constants."""
    rates = {c: corrupt(LONG, c, seed=1, sample_id="s", purpose=SELF_CHECK).candidate_selection_rate for c in ("P25", "P50", "P75", "P100")}
    assert rates["P100"] == 1.0
    assert rates["P25"] < rates["P50"] < rates["P75"] < rates["P100"]
    for condition, expected in (("P25", 0.25), ("P50", 0.50), ("P75", 0.75)):
        assert abs(rates[condition] - expected) < 0.12


def test_p100_removes_every_tone_mark_but_keeps_letter_diacritics():
    result = corrupt(VI, "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    corrupted = decompose(result.corrupted_text)
    assert all(s.observed_tone is ObservedTone.UNMARKED for s in corrupted.syllables)
    # ô and ê survive: they are letter identity, not tone.
    assert "ô" in result.corrupted_text and "ê" in result.corrupted_text
    assert LetterDiacritic.CIRCUMFLEX in corrupted.letter_channel


def test_strip_all_removes_tone_and_letter_diacritics():
    result = corrupt(VI, "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.corrupted_text == "Toi dang nghien cuu xu ly ngon ngu tu nhien."
    corrupted = decompose(result.corrupted_text)
    assert all(s.observed_tone is ObservedTone.UNMARKED for s in corrupted.syllables)
    assert set(corrupted.letter_channel) <= {LetterDiacritic.NONE, LetterDiacritic.NA}


def test_p100_and_strip_all_are_different_conditions():
    """The proposal distinguishes them; conflating them would silently change
    the headline STRIP-ALL number."""
    p100 = corrupt(VI, "P100", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text
    strip = corrupt(VI, "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text
    assert p100 != strip
    assert get_condition("P100").scope is CorruptionScope.TONE
    assert get_condition("STRIP_ALL").scope is CorruptionScope.TONE_AND_LETTER


def test_strip_all_maps_every_vietnamese_letter_to_its_base():
    result = corrupt("đường ăn cân ơn ưu êm ôm Đ", "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.corrupted_text == "duong an can on uu em om D"


def test_condition_lookup_is_forgiving_about_spelling():
    assert get_condition("p50") is get_condition("P50")
    assert get_condition("strip-all") is get_condition("STRIP_ALL")
    with pytest.raises(UnknownCondition):
        get_condition("P42")


def test_variant_condition_is_recognised_but_refused():
    assert "VARIANT" in UNIMPLEMENTED_CONDITIONS
    with pytest.raises(UnknownCondition, match="not implemented"):
        get_condition("VARIANT")


# ---------------------------------------------------------------------------
# 13-14. Degenerate unit counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["2026 6,5% !!!", "", "   ", "😄🎉", "123-456"])
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_zero_eligible_units_is_handled_without_dividing_by_zero(text, condition):
    result = corrupt(text, condition, seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.candidate_units == 0
    assert result.selected_candidates == 0
    assert result.candidate_selection_rate is None
    assert result.candidate_modification_rate is None
    assert result.corrupted_text == canon(text)


def test_one_eligible_unit():
    result = corrupt("phở", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.candidate_units == 1
    assert result.selected_candidates == 1
    assert result.candidate_selection_rate == 1.0
    assert result.corrupted_text == "phơ"


# ---------------------------------------------------------------------------
# 15-17. Tone semantics and the H4 ambiguity
# ---------------------------------------------------------------------------
def test_genuine_ngang_and_stripped_tone_both_become_unmarked():
    result = corrupt("ma má", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    genuine, stripped = result.decisions
    assert genuine.clean_observed_tone is ObservedTone.UNMARKED
    assert genuine.corrupted_observed_tone is ObservedTone.UNMARKED
    assert stripped.clean_observed_tone is ObservedTone.SAC
    assert stripped.corrupted_observed_tone is ObservedTone.UNMARKED
    assert result.corrupted_text == "ma ma"


def test_stripped_tone_is_never_relabelled_as_ngang():
    result = corrupt(TONES, "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    for decision in result.decisions:
        assert decision.corrupted_observed_tone is ObservedTone.UNMARKED
        assert decision.corrupted_observed_tone is not Tone.NGANG  # different enums entirely


def test_paired_metadata_distinguishes_the_two_origins_of_unmarked():
    """The corrupted string cannot express the difference; the metadata must."""
    result = corrupt("ma má", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    genuine, stripped = result.decisions
    assert genuine.clean_lexical_tone is Tone.NGANG and not genuine.tone_mark_removed
    assert stripped.clean_lexical_tone is Tone.SAC and stripped.tone_mark_removed


def test_h4_oracle_views_are_derivable_without_implementing_h4():
    result = corrupt("ma má mà mả mã mạ", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    genuine = result.decisions[0]
    assert genuine.oracle_tone_is_genuine_ngang and not genuine.oracle_tone_is_missing
    for decision in result.decisions[1:]:
        assert decision.oracle_tone_is_missing and not decision.oracle_tone_is_genuine_ngang


@pytest.mark.parametrize(
    "syllable,tone",
    [("má", Tone.SAC), ("mà", Tone.HUYEN), ("mả", Tone.HOI), ("mã", Tone.NGA), ("mạ", Tone.NANG)],
)
def test_all_five_marked_tones_are_stripped_and_recorded(syllable, tone):
    result = corrupt(syllable, "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    (decision,) = result.decisions
    assert decision.clean_lexical_tone is tone
    assert decision.tone_mark_removed
    assert decision.corrupted_observed_tone is ObservedTone.UNMARKED
    assert result.corrupted_text == "ma"


def test_ngang_syllables_are_invariant_under_corruption():
    """Proposal 4.3: 'a ngang syllable is invariant'. It can be selected but
    never modified, and selected != modified is reported, not hidden."""
    result = corrupt("toi di hoc", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.selected_candidates == 3
    assert result.modified_candidates == 0
    assert result.candidate_selection_rate == 1.0
    assert result.candidate_modification_rate == 0.0
    assert result.is_unchanged


# ---------------------------------------------------------------------------
# 18-20. Letter diacritics, đ/Đ, case
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("char,base", [("ă", "a"), ("â", "a"), ("ê", "e"), ("ô", "o"), ("ơ", "o"), ("ư", "u")])
def test_letter_diacritics_survive_tone_conditions_and_fall_under_strip_all(char, base):
    assert corrupt(char, "P100", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == char
    assert corrupt(char, "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == base


@pytest.mark.parametrize("char,base", [("đ", "d"), ("Đ", "D")])
def test_d_stroke_survives_tone_conditions_and_falls_under_strip_all(char, base):
    assert corrupt(char, "P100", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == char
    assert corrupt(char, "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == base


def test_letter_diacritic_removals_are_recorded():
    result = corrupt("được", "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK)
    (decision,) = result.decisions
    assert decision.tone_mark_removed
    assert "STROKE" in decision.letter_diacritics_removed
    assert len(decision.letter_diacritics_removed) >= 3  # stroke + two horns


def test_case_is_preserved_by_every_condition():
    for condition in CONDITIONS:
        result = corrupt("ĐẠI HỌC Tự Nhiên", condition, seed=1, sample_id="s", purpose=SELF_CHECK)
        assert result.corrupted_text == result.corrupted_text.replace("dai", "DAI")
        assert result.corrupted_text.split()[0].isupper()


def test_uppercase_strip_all():
    assert corrupt("ĐẠI HỌC TỰ NHIÊN", "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == "DAI HOC TU NHIEN"


# ---------------------------------------------------------------------------
# 21-30. Mixed content safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize(
    "text",
    [
        "Năm 2026, GDP tăng 6,5% (VAT 10%)!",
        "Cuộc họp lúc 14:30 ngày 19/08/2026.",
        "  spaced   out  \t text  ",
        "hôm nay tôi rất vui 😄🎉",
    ],
)
def test_punctuation_whitespace_digits_and_emoji_are_preserved(text, condition):
    result = corrupt(text, condition, seed=1, sample_id="s", purpose=SELF_CHECK)
    clean, corrupted = result.canonical_clean_text, result.corrupted_text
    assert len(clean.split(" ")) == len(corrupted.split(" "))
    for char in clean:
        if char.isdigit() or char.isspace() or char in ",.!%():/😄🎉":
            assert char in corrupted


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_urls_and_emails_keep_their_structure(condition):
    for text in ["https://example.edu.vn/a?id=42&lang=vi", "lien.he@example.com"]:
        result = corrupt(text, condition, seed=1, sample_id="s", purpose=SELF_CHECK)
        assert result.corrupted_text.count("/") == text.count("/")
        assert result.corrupted_text.count("@") == text.count("@")
        assert result.corrupted_text.count(".") == text.count(".")
        assert result.corrupted_text.count("=") == text.count("=")


def test_english_words_are_structurally_unchanged():
    result = corrupt("toi dung Python va PyTorch", "STRIP_ALL", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert "Python" in result.corrupted_text and "PyTorch" in result.corrupted_text


def test_unsupported_combining_marks_are_not_dropped():
    """Diaeresis and cedilla are not Vietnamese marks and must survive every
    condition, including STRIP_ALL."""
    for condition in CONDITIONS:
        result = corrupt("Müller façade", condition, seed=1, sample_id="s", purpose=SELF_CHECK)
        assert "ü" in result.corrupted_text
        assert "ç" in result.corrupted_text


def test_documented_collateral_effect_on_loanword_acute():
    """`é` is a Vietnamese acute codepoint, so a loanword carrying it is stripped
    like any other syllable. Recorded as a decision (D-B2-003), not a surprise."""
    assert corrupt("café", "P100", seed=1, sample_id="s", purpose=SELF_CHECK).corrupted_text == "cafe"


def test_malformed_multiple_tone_syllable_is_handled_without_crashing():
    from unmark.orthography import marks as M

    text = unicodedata.normalize("NFC", "a" + M.ACUTE + M.GRAVE)
    result = corrupt(text, "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert strip_to_base(result.canonical_clean_text) == strip_to_base(result.corrupted_text)
    assert result.corrupted_text == "a"  # both marks removed, nothing else touched


# ---------------------------------------------------------------------------
# 31-38. Result contract and base invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize("text", MIXED_TEXTS)
def test_base_invariance_holds_for_every_condition(text, condition):
    """The proposal's load-bearing invariant: b(x) == b(x̃)."""
    result = corrupt(text, condition, seed=13, sample_id="s", purpose=SELF_CHECK)
    assert strip_to_base(result.canonical_clean_text) == strip_to_base(result.corrupted_text)
    assert result.clean_decomposition.base_text == result.corrupted_decomposition.base_text


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize("text", MIXED_TEXTS)
def test_corruption_never_changes_the_syllable_count(text, condition):
    result = corrupt(text, condition, seed=13, sample_id="s", purpose=SELF_CHECK)
    assert len(result.clean_decomposition.syllables) == len(result.corrupted_decomposition.syllables)
    assert len(result.decisions) == result.candidate_units


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
@pytest.mark.parametrize("text", MIXED_TEXTS)
def test_corrupted_text_is_itself_canonical(text, condition):
    result = corrupt(text, condition, seed=13, sample_id="s", purpose=SELF_CHECK)
    assert canon(result.corrupted_text) == result.corrupted_text


def test_selected_metadata_matches_the_actual_output():
    result = corrupt(TONES, "P50", seed=4, sample_id="s", purpose=SELF_CHECK)
    for decision in result.decisions:
        span = result.corrupted_decomposition.syllables[decision.unit_index]
        if decision.tone_mark_removed:
            assert decision.selected
            assert span.observed_tone is ObservedTone.UNMARKED
        else:
            assert span.observed_tone is decision.clean_observed_tone
        if decision.modified:
            assert decision.selected


def test_realized_probability_is_selected_over_eligible():
    result = corrupt(LONG, "P50", seed=4, sample_id="s", purpose=SELF_CHECK)
    assert result.candidate_selection_rate == result.selected_candidates / result.candidate_units
    assert result.candidate_modification_rate == result.modified_candidates / result.candidate_units
    assert result.modified_candidates <= result.selected_candidates


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_schema_version_is_always_recorded(condition):
    result = corrupt(VI, condition, seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.schema_version == CORRUPTION_SCHEMA_VERSION == "b2-v1"
    assert result.to_dict()["schema_version"] == "b2-v1"


def test_changing_the_schema_version_changes_decisions():
    """So that artifacts from different algorithm versions cannot be pooled."""
    a = [is_selected(probability=0.5, seed=1, sample_id="s", identity="i", unit_index=i)[0] for i in range(80)]
    b = [
        is_selected(
            probability=0.5, seed=1, sample_id="s", identity="i", unit_index=i, schema_version="b2-vNEXT"
        )[0]
        for i in range(80)
    ]
    assert a != b


def test_result_is_json_serialisable():
    result = corrupt(VI, "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "b2-v1" in payload
    restored = json.loads(payload)
    assert restored["corrupted_text"] == result.corrupted_text
    assert restored["condition"] == "P50"
    assert len(restored["decisions"]) == result.candidate_units


def test_no_tokenizer_or_ml_import_in_the_corruption_package():
    """15/16. B2 is entirely pre-tokenization."""
    import ast

    banned = {"torch", "transformers", "tokenizers", "sentencepiece", "datasets", "safetensors", "accelerate"}
    for path in (REPO_ROOT / "unmark" / "corruption").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, f"{path.name}: {node.module}"


# ---------------------------------------------------------------------------
# Self-check script
# ---------------------------------------------------------------------------
def test_self_check_script_runs_and_writes_artifacts(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/b2_corruption_self_check.py", "--output-root", str(tmp_path), "--run-id", "T"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "T"
    for name in ("config.json", "cases.jsonl", "summary.json", "report.md"):
        assert (run_dir / name).is_file(), name
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_failures"] == 0
    assert summary["schema_version"] == "b2-v1"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "not a dataset, not a benchmark and not a training corpus" in report


def test_self_check_makes_no_natural_corpus_claim():
    source = (REPO_ROOT / "scripts" / "b2_corruption_self_check.py").read_text(encoding="utf-8")
    assert "implementation verification" in source.lower()
    assert "not a dataset" in source.lower()


def test_config_file_matches_the_implemented_conditions():
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "corruption" / "default.yaml").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == CORRUPTION_SCHEMA_VERSION
    assert set(cfg["conditions"]) == set(CONDITIONS)
    for name, block in cfg["conditions"].items():
        assert block["probability"] == CONDITIONS[name].probability
        assert block["scope"] == CONDITIONS[name].scope.value
    assert cfg["unit"]["kind"] == "candidate_syllable_span"
    assert cfg["canonicalisation"]["tone_placement"] == "MODERN"
    # The config must record the provisional eligibility state, not hide it.
    assert cfg["eligibility"]["policy"] == ACTIVE_ELIGIBILITY_POLICY.value == "UNRESOLVED"
    assert cfg["eligibility"]["provisional"] is True
    assert cfg["eligibility"]["resolved_policy_name"] == EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY.value


# ===========================================================================
# Eligibility safety (audit 004 follow-up)
# ===========================================================================
# Researcher review of audit 003 reclassified the GAP-2 interaction from a
# harmless clarification to a pre-training semantic dependency. These tests
# assert that the provisional fallback cannot become the scientific protocol.

def test_deterministic_output_is_unchanged_by_the_eligibility_refactor():
    """1. The engine is final; only the eligibility framing changed. These are
    the exact values the pre-refactor engine produced."""
    result = corrupt(VI, "P75", seed=42, sample_id="s1", purpose=SELF_CHECK)
    assert result.corrupted_text == "Tôi đang nghiên cứu xư lý ngôn ngư tự nhiên."
    assert corrupt(VI, "P100", seed=42, sample_id="s1", purpose=SELF_CHECK).corrupted_text == (
        "Tôi đang nghiên cưu xư ly ngôn ngư tư nhiên."
    )
    assert corrupt(VI, "STRIP_ALL", seed=42, sample_id="s1", purpose=SELF_CHECK).corrupted_text == (
        "Toi dang nghien cuu xu ly ngon ngu tu nhien."
    )
    assert result.candidate_units == 10
    assert result.selected_candidates == 6


def test_scores_are_unchanged_by_the_eligibility_refactor():
    identity = text_identity("Tôi đang học")
    assert unit_score(seed=42, sample_id="s1", identity=identity, unit_index=0) == pytest.approx(
        0.21396497977394088
    )


def test_undecided_eligibility_is_not_presented_as_resolved():
    """2/5. A candidate span is not a confirmed Vietnamese syllable."""
    result = corrupt("toi dung Python va PyTorch", "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert result.candidate_units == 5  # includes Python and PyTorch
    assert all(d.eligibility is Eligibility.UNDECIDED for d in result.decisions)
    assert result.provisional_eligibility is True
    assert result.eligibility_policy is EligibilityPolicy.UNRESOLVED


def test_eligible_units_refuses_to_return_a_provisional_number():
    """The substitution that would turn the fallback into the protocol."""
    result = corrupt(VI, "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    with pytest.raises(EligibilityUnresolved) as excinfo:
        result.eligible_units
    message = str(excinfo.value)
    assert "candidate" in message
    assert "GAP-2" in message
    assert "B3" in message


def test_scientific_purpose_is_the_default_and_fails_today():
    """3. The unsafe path must be the one you have to ask for."""
    with pytest.raises(EligibilityUnresolved):
        corrupt(VI, "P50", seed=1, sample_id="s")
    with pytest.raises(EligibilityUnresolved):
        corrupt(VI, "P50", seed=1, sample_id="s", purpose=CorruptionPurpose.SCIENTIFIC)
    with pytest.raises(EligibilityUnresolved):
        corrupt_batch([(VI, "a")], "P50", seed=1)


def test_guard_error_names_gap2_and_the_b3_resolution_owner():
    """4. The message must tell a reader what is blocked and who closes it."""
    with pytest.raises(EligibilityUnresolved) as excinfo:
        require_resolved_eligibility()
    message = str(excinfo.value)
    assert "GAP-2" in message
    assert "B3" in message
    assert "syllable inventory" in message
    assert "SELF_CHECK" in message
    assert "denominator" in message


def test_guard_passes_once_a_policy_is_resolved():
    """The guard is a real switch, not a permanent refusal."""
    require_resolved_eligibility(policy=EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY)
    assert is_resolved(EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY)
    assert not is_resolved(EligibilityPolicy.UNRESOLVED)
    assert ACTIVE_ELIGIBILITY_POLICY is EligibilityPolicy.UNRESOLVED


def test_metric_names_cannot_be_misread_as_a_vietnamese_syllable_rate():
    """6. No field name claims more than the engine knows."""
    result = corrupt("toi dung Python", "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    payload = result.to_dict()
    assert "candidate_units" in payload
    assert "candidate_selection_rate" in payload
    assert "eligible_units" not in payload
    assert "realized_probability" not in payload
    assert payload["provisional_eligibility"] is True
    assert payload["eligibility_policy"] == "UNRESOLVED"
    assert not hasattr(result, "realized_probability")


def test_result_metadata_states_the_fallback_and_its_owner():
    result = corrupt(VI, "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    note = result.metadata["eligibility_filter"]
    assert "PROVISIONAL" in note
    assert "GAP-2" in note
    assert "B3" in note
    assert result.metadata["unit"] == "candidate_syllable_span"
    assert result.metadata["purpose"] == "SELF_CHECK"


def test_the_documented_provisional_consequences_are_still_reproducible():
    """The two behaviours that motivated the reclassification. They are not
    bugs in the engine; they are why the denominator is not yet scientific."""
    english = corrupt("toi dung Python va PyTorch", "P50", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert english.candidate_units == 5, "English spans sit in the provisional denominator"
    loanword = corrupt("café", "P100", seed=1, sample_id="s", purpose=SELF_CHECK)
    assert loanword.corrupted_text == "cafe", "acute is a Vietnamese codepoint"


def test_self_check_artifacts_are_stamped_provisional(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/b2_corruption_self_check.py", "--output-root", str(tmp_path), "--run-id", "P"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "P" / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "P" / "summary.json").read_text(encoding="utf-8"))
    assert config["eligibility_policy"] == "UNRESOLVED"
    assert config["provisional_eligibility"] is True
    assert config["purpose"] == "SELF_CHECK"
    assert summary["provisional_eligibility"] is True
    report = (tmp_path / "P" / "report.md").read_text(encoding="utf-8")
    assert "Provisional eligibility" in report
    assert "candidate spans" in report
    assert "GAP-2" in report


def test_no_dictionary_classifier_or_tokenizer_was_added():
    """7. The fix must not smuggle in a language identifier."""
    import ast

    banned_modules = {"torch", "transformers", "tokenizers", "sentencepiece", "datasets"}
    suspicious = ("SYLLABLE_INVENTORY_DATA", "VIETNAMESE_WORDS", "WORDLIST", "is_english", "detect_language")
    for path in (REPO_ROOT / "unmark" / "corruption").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in suspicious:
            assert name not in source, f"{path.name} looks like a language classifier: {name}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned_modules, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned_modules, f"{path.name}: {node.module}"
    # The resolved policy exists as a name only; no inventory ships with it.
    assert EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY.value == "VIETNAMESE_SYLLABLE_INVENTORY"


def test_decision_log_records_the_reclassification():
    log = (REPO_ROOT / "docs" / "spec" / "decisions.md").read_text(encoding="utf-8")
    assert "D-B2-003" in log
    assert "TEMPORARY IMPLEMENTATION FALLBACK" in log
    assert "B3" in log
    assert "audit 004" in log.lower() or "004" in log
