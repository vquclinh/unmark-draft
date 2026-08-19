"""Unit tests for the B1A Vietnamese orthography core.

Offline, standard library only: no torch, no transformers, no corpus, no
network. The generated cases below are *implementation verification*, built by
construction from the Unicode mark inventory. They are not a natural-language
benchmark and must never be reported as corpus evidence.
"""

from __future__ import annotations

import itertools
import unicodedata

import pytest

from unmark.orthography import (
    Anomaly,
    Eligibility,
    LetterDiacritic,
    ObservedTone,
    Tone,
    TonePlacement,
    TonePlacementUndecided,
    base_signature,
    canon,
    canonical_differences,
    decompose,
    observed_from_lexical,
    recompose,
    strip_to_base,
)
from unmark.orthography import marks as M

# ---------------------------------------------------------------------------
# Deterministic generated material (by construction, no randomness, no seed)
# ---------------------------------------------------------------------------
BASE_VOWELS = "aeiouy"
LETTER_MARK_OPTIONS = ("", M.BREVE, M.CIRCUMFLEX, M.HORN)
TONE_MARK_OPTIONS = ("", M.ACUTE, M.GRAVE, M.HOOK_ABOVE, M.TILDE, M.DOT_BELOW)

# Every (vowel x letter mark x tone mark x case) combination the inventory
# admits. Many are not real Vietnamese letters; that is deliberate -- the point
# is exhaustive coverage of Unicode mark handling, not lexical plausibility.
GENERATED_LETTERS: list[tuple[str, str, str, str]] = [
    (vowel.upper() if upper else vowel, letter_mark, tone_mark, "upper" if upper else "lower")
    for vowel, letter_mark, tone_mark, upper in itertools.product(
        BASE_VOWELS, LETTER_MARK_OPTIONS, TONE_MARK_OPTIONS, (False, True)
    )
]

# The real Vietnamese letters carrying BOTH a letter diacritic and a tone.
VIETNAMESE_LETTER_PLUS_TONE = "ắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữự"

CURATED_TEXTS = [
    "",
    "   ",
    "\n\t ",
    "toi dang hoc",
    "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.",
    "Đường Nguyễn Huệ",
    "ĐẠI HỌC KHOA HỌC TỰ NHIÊN",
    "hoà",
    "hòa",
    "Bạn có khoẻ không?",
    "Nam 2026, GDP tang 6,5% so voi nam truoc.",
    "toi dang hoc machine learning tai VNU-HCM",
    "Lien he qua lien.he@example.com nhe",
    "Xem tai https://example.edu.vn/tuyen-sinh?id=42&lang=vi",
    "hom nay toi rat vui 😄🎉",
    "Müller façade naïve",
    "ắằẳẵặ ấầẩẫậ ếềểễệ ốồổỗộ ớờởỡợ ứừửữự",
    "đĐ ăĂ âÂ êÊ ôÔ ơƠ ưƯ",
    "Cuoc hop luc 14:30 ngay 19/08/2026.",
]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


# ---------------------------------------------------------------------------
# canon()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_canon_is_idempotent(text):
    assert canon(canon(text)) == canon(text)


@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_canon_output_is_nfc(text):
    assert canon(text) == _nfc(canon(text))


@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_canon_maps_nfc_and_nfd_to_the_same_string(text):
    assert canon(_nfc(text)) == canon(_nfd(text))


def test_canon_preserves_whitespace_case_and_punctuation():
    """canon is a reconstruction target, not a comparison form: unlike
    base_signature it must not touch spacing, case or punctuation."""
    text = "  Toi   di\thoc!!  "
    assert canon(text) == text


def test_canon_preserves_non_vietnamese_marks():
    assert canon("Müller façade") == "Müller façade"


# --- the open specification decision ---------------------------------------
def test_modern_is_the_default_canonical_placement():
    """GAP-1 closed: canon() with no override uses the locked convention."""
    from unmark.orthography import DEFAULT_TONE_PLACEMENT

    assert DEFAULT_TONE_PLACEMENT is TonePlacement.MODERN
    assert canon("hòa") == canon("hòa", TonePlacement.MODERN)


def test_traditional_placement_is_not_implemented():
    """The project needs one canonical convention, not two. Requesting the other
    must raise rather than silently fall back to something."""
    with pytest.raises(TonePlacementUndecided) as excinfo:
        canon("hoà", TonePlacement.TRADITIONAL)
    assert "not implemented" in str(excinfo.value)


def test_placement_variants_collapse_under_the_default():
    assert canon("hoà") == canon("hòa") == "hoà"
    assert base_signature("hoà") == base_signature("hòa")  # and they share a base


def test_preserve_remains_available_as_an_explicit_diagnostic_mode():
    """PRESERVE is no longer the project pathway, but it must still show what an
    input actually contained."""
    assert canon("hòa", TonePlacement.PRESERVE) == "hòa"
    assert canon("hoà", TonePlacement.PRESERVE) == "hoà"
    assert canon("hòa", TonePlacement.PRESERVE) != canon("hoà", TonePlacement.PRESERVE)


def test_canonical_differences_separates_normalisation_from_placement():
    info = canonical_differences("hòa")
    assert info["nfc_changed"] is False  # already NFC
    assert info["tone_placement_changed"] is True  # but not in nucleus placement
    assert info["already_canonical"] is False
    modern = canonical_differences("hoà")
    assert modern["tone_placement_changed"] is False
    assert modern["already_canonical"] is True


def test_canonical_differences_enumerates_rather_than_hides():
    info = canonical_differences(_nfd("Tiếng Việt"))
    assert info["already_canonical"] is False
    assert info["nfc_changed"] is True
    assert info["input_form"] == "NFD"
    assert canonical_differences("toi di hoc")["already_canonical"] is True


# ---------------------------------------------------------------------------
# Round-trip invariant: recompose(decompose(x)) == canon(x)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_round_trip_on_curated_text(text):
    assert recompose(decompose(text)) == canon(text)


@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_round_trip_holds_for_nfd_input_too(text):
    assert recompose(decompose(_nfd(text))) == canon(text)


def test_round_trip_over_every_generated_letter():
    failures = []
    for letter, letter_mark, tone_mark, _case in GENERATED_LETTERS:
        text = _nfc(letter + letter_mark + tone_mark)
        if recompose(decompose(text)) != canon(text):
            failures.append(text)
    assert not failures, f"round-trip failed for {len(failures)} generated letters: {failures[:10]}"


def test_round_trip_inside_a_syllable_context():
    """Marks in running text, not just in isolation."""
    failures = []
    for letter, letter_mark, tone_mark, _case in GENERATED_LETTERS:
        text = _nfc(f"tr{letter}{letter_mark}{tone_mark}ng oi!")
        if recompose(decompose(text)) != canon(text):
            failures.append(text)
    assert not failures, failures[:10]


@pytest.mark.parametrize("char", list(VIETNAMESE_LETTER_PLUS_TONE))
def test_round_trip_for_real_letter_plus_tone_characters(char):
    assert recompose(decompose(char)) == canon(char)
    assert recompose(decompose(char.upper())) == canon(char.upper())


def test_decomposition_is_deterministic():
    for text in CURATED_TEXTS:
        assert decompose(text) == decompose(text)


def test_recomposition_is_deterministic():
    for text in CURATED_TEXTS:
        parts = decompose(text)
        assert recompose(parts) == recompose(parts)


# ---------------------------------------------------------------------------
# Base stream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_base_text_carries_no_vietnamese_marks(text):
    base = decompose(text).base_text
    decomposed = _nfd(base)
    assert not (set(decomposed) & set(M.VIETNAMESE_MARKS)), f"residual Vietnamese mark in {base!r}"
    assert not (set(base) & set(M.D_STROKE)), f"residual d-stroke in {base!r}"


@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_base_text_agrees_with_the_g1_signature(text):
    """Cross-check between the two implementations that strip Vietnamese marks.
    base_signature additionally collapses whitespace, so compare its strict form."""
    assert decompose(text).base_text == base_signature(text, collapse_whitespace=False)


def test_strip_to_base_matches_decompose():
    for text in CURATED_TEXTS:
        assert strip_to_base(text) == decompose(text).base_text


def test_base_preserves_case():
    assert decompose("ĐẠI HỌC").base_text == "DAI HOC"
    assert decompose("đại học").base_text == "dai hoc"


def test_base_preserves_digits_punctuation_and_symbols():
    text = "Nam 2026, GDP tang 6,5% (VAT 10%)! 😄"
    assert decompose(text).base_text == text


def test_base_preserves_urls_and_emails():
    for text in ["https://example.edu.vn/a-b?id=42&lang=vi", "lien.he@example.com"]:
        assert decompose(text).base_text == text


def test_base_keeps_non_vietnamese_diacritics():
    """Diaeresis and cedilla are not Vietnamese marks and must survive stripping."""
    assert decompose("Müller façade").base_text == "Müller façade"


# ---------------------------------------------------------------------------
# Tone channel: lexical vs observable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "syllable,expected",
    [
        ("ma", ObservedTone.UNMARKED),
        ("má", ObservedTone.SAC),
        ("mà", ObservedTone.HUYEN),
        ("mả", ObservedTone.HOI),
        ("mã", ObservedTone.NGA),
        ("mạ", ObservedTone.NANG),
    ],
)
def test_all_six_tone_classes_are_observed(syllable, expected):
    (span,) = decompose(syllable).syllables
    assert span.observed_tone is expected


@pytest.mark.parametrize(
    "syllable,expected",
    [("má", Tone.SAC), ("mà", Tone.HUYEN), ("mả", Tone.HOI), ("mã", Tone.NGA), ("mạ", Tone.NANG)],
)
def test_a_visible_mark_settles_the_lexical_tone(syllable, expected):
    (span,) = decompose(syllable).syllables
    assert span.lexical_tone is expected
    assert span.tone_is_knowable


def test_unmarked_is_not_treated_as_known_ngang():
    """The core semantic requirement. An absent mark is either genuine ngang or
    a stripped tone; the string cannot say which, so lexical tone is None."""
    (span,) = decompose("ma").syllables
    assert span.observed_tone is ObservedTone.UNMARKED
    assert span.lexical_tone is None
    assert not span.tone_is_knowable


def test_clean_source_assertion_makes_ngang_knowable():
    """Only an explicit caller assertion promotes UNMARKED to lexical NGANG."""
    (span,) = decompose("ma", source_is_clean=True).syllables
    assert span.observed_tone is ObservedTone.UNMARKED
    assert span.lexical_tone is Tone.NGANG


def test_clean_source_assertion_does_not_change_the_observable_channel():
    """The deployable channel must be identical either way: what is observable
    cannot depend on what the caller happens to know."""
    for text in ["ma", "má", "Tôi đang học", "toi dang hoc"]:
        assert decompose(text).observed_tone_channel == decompose(text, source_is_clean=True).observed_tone_channel


def test_lexical_and_observed_tone_are_distinct_types():
    assert Tone.NGANG not in set(ObservedTone)
    assert ObservedTone.UNMARKED not in set(Tone)
    assert {t.name for t in Tone} - {t.name for t in ObservedTone} == {"NGANG"}
    assert {t.name for t in ObservedTone} - {t.name for t in Tone} == {"UNMARKED"}


def test_observed_from_lexical_projection_is_lossy_only_for_ngang():
    assert observed_from_lexical(Tone.NGANG) is ObservedTone.UNMARKED
    for tone in (Tone.SAC, Tone.HUYEN, Tone.HOI, Tone.NGA, Tone.NANG):
        assert observed_from_lexical(tone).name == tone.name


def test_tone_is_per_syllable_not_per_character():
    text = "Tôi đang nghiên cứu"
    parts = decompose(text)
    assert len(parts.syllables) == 4
    assert [s.base_text for s in parts.syllables] == ["Toi", "dang", "nghien", "cuu"]
    # "Tôi" and "nghiên" carry a circumflex, which is a LETTER diacritic and
    # contributes no tone; only "cứu" carries a tone mark.
    assert [s.observed_tone.name for s in parts.syllables] == ["UNMARKED", "UNMARKED", "UNMARKED", "SAC"]


def test_tone_position_is_canonical_across_placement_variants():
    """After canonicalisation the two written variants are the same string, so
    the recorded tone position agrees too."""
    (span,) = decompose("hoà").syllables
    (other,) = decompose("hòa").syllables
    assert span.tone_unit_index == other.tone_unit_index
    assert span.tone_unit_index == 2  # the nucleus `a`, not the glide `o`
    assert span.observed_tone is other.observed_tone is ObservedTone.HUYEN


def test_preserve_mode_still_exposes_the_original_tone_position():
    """The diagnostic pathway must not be canonicalised away."""
    a = decompose("hoà", placement=TonePlacement.PRESERVE).syllables[0]
    b = decompose("hòa", placement=TonePlacement.PRESERVE).syllables[0]
    assert a.tone_unit_index == 2
    assert b.tone_unit_index == 1
    assert a.observed_tone is b.observed_tone is ObservedTone.HUYEN


# ---------------------------------------------------------------------------
# Letter-diacritic channel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "char,base,state",
    [
        ("a", "a", LetterDiacritic.NONE),
        ("ă", "a", LetterDiacritic.BREVE),
        ("â", "a", LetterDiacritic.CIRCUMFLEX),
        ("e", "e", LetterDiacritic.NONE),
        ("ê", "e", LetterDiacritic.CIRCUMFLEX),
        ("o", "o", LetterDiacritic.NONE),
        ("ô", "o", LetterDiacritic.CIRCUMFLEX),
        ("ơ", "o", LetterDiacritic.HORN),
        ("u", "u", LetterDiacritic.NONE),
        ("ư", "u", LetterDiacritic.HORN),
        ("d", "d", LetterDiacritic.NONE),
        ("đ", "d", LetterDiacritic.STROKE),
        ("A", "A", LetterDiacritic.NONE),
        ("Ă", "A", LetterDiacritic.BREVE),
        ("Â", "A", LetterDiacritic.CIRCUMFLEX),
        ("Ê", "E", LetterDiacritic.CIRCUMFLEX),
        ("Ô", "O", LetterDiacritic.CIRCUMFLEX),
        ("Ơ", "O", LetterDiacritic.HORN),
        ("Ư", "U", LetterDiacritic.HORN),
        ("Đ", "D", LetterDiacritic.STROKE),
    ],
)
def test_letter_diacritic_states_and_base_characters(char, base, state):
    (unit,) = decompose(char).units
    assert unit.base_char == base
    assert unit.letter_diacritic is state


def test_na_and_none_are_not_conflated():
    """NONE = a letter that could carry a diacritic and does not.
    NA = the channel does not apply at all."""
    parts = decompose("a 1.")
    states = [u.letter_diacritic for u in parts.units]
    assert states == [LetterDiacritic.NONE, LetterDiacritic.NA, LetterDiacritic.NA, LetterDiacritic.NA]
    assert LetterDiacritic.NONE is not LetterDiacritic.NA


@pytest.mark.parametrize("char", ["đ", "Đ"])
def test_d_stroke_is_a_letter_diacritic_not_a_tone(char):
    (unit,) = decompose(char).units
    assert unit.letter_diacritic is LetterDiacritic.STROKE
    assert unit.observed_tone is ObservedTone.UNMARKED
    assert unit.has_stroke is True
    assert recompose(decompose(char)) == char


def test_letter_channel_is_per_character_within_one_syllable():
    """A syllable may carry several letter diacritics on different characters --
    the reason the channel is per character (proposal 4.3)."""
    parts = decompose("được")
    (span,) = parts.syllables
    assert span.base_text == "duoc"
    # đ=stroke, ư=horn, ợ=horn (o + horn + dot below), c=none.
    assert parts.letter_channel == (
        LetterDiacritic.STROKE,
        LetterDiacritic.HORN,
        LetterDiacritic.HORN,
        LetterDiacritic.NONE,
    )
    assert span.observed_tone is ObservedTone.NANG  # one tone for the whole syllable


def test_generated_letter_states_are_correct():
    expected_state = {
        "": LetterDiacritic.NONE,
        M.BREVE: LetterDiacritic.BREVE,
        M.CIRCUMFLEX: LetterDiacritic.CIRCUMFLEX,
        M.HORN: LetterDiacritic.HORN,
    }
    expected_tone = {
        "": ObservedTone.UNMARKED,
        M.ACUTE: ObservedTone.SAC,
        M.GRAVE: ObservedTone.HUYEN,
        M.HOOK_ABOVE: ObservedTone.HOI,
        M.TILDE: ObservedTone.NGA,
        M.DOT_BELOW: ObservedTone.NANG,
    }
    for letter, letter_mark, tone_mark, _case in GENERATED_LETTERS:
        (unit,) = decompose(_nfc(letter + letter_mark + tone_mark)).units
        assert unit.base_char == letter, (letter, letter_mark, tone_mark)
        assert unit.letter_diacritic is expected_state[letter_mark]
        assert unit.observed_tone is expected_tone[tone_mark]


@pytest.mark.parametrize("char", list(VIETNAMESE_LETTER_PLUS_TONE))
def test_real_letter_plus_tone_characters_split_into_both_channels(char):
    (unit,) = decompose(char).units
    assert unit.letter_diacritic in (
        LetterDiacritic.BREVE,
        LetterDiacritic.CIRCUMFLEX,
        LetterDiacritic.HORN,
    ), char
    assert unit.observed_tone is not ObservedTone.UNMARKED, char
    assert unit.base_char in "aeiouy", char


# ---------------------------------------------------------------------------
# NFC / NFD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", CURATED_TEXTS)
def test_nfc_and_nfd_produce_identical_decompositions(text):
    a = decompose(_nfc(text))
    b = decompose(_nfd(text))
    assert a.canonical_text == b.canonical_text
    assert a.base_text == b.base_text
    assert a.letter_channel == b.letter_channel
    assert a.observed_tone_channel == b.observed_tone_channel


def test_original_text_is_retained_alongside_the_canonical_form():
    nfd_text = _nfd("Tiếng Việt")
    parts = decompose(nfd_text)
    assert parts.original_text == nfd_text
    assert parts.canonical_text == "Tiếng Việt"
    assert parts.is_canonical is False
    assert decompose("Tiếng Việt").is_canonical is True


# ---------------------------------------------------------------------------
# Spans, offsets, eligibility
# ---------------------------------------------------------------------------
def test_unit_offsets_index_the_canonical_and_base_strings():
    for text in CURATED_TEXTS:
        parts = decompose(text)
        for unit in parts.units:
            assert parts.canonical_text[unit.canonical_start : unit.canonical_end] == unit.canonical_text
            assert parts.base_text[unit.base_start : unit.base_end] == unit.base_char


def test_syllable_offsets_index_the_canonical_and_base_strings():
    for text in CURATED_TEXTS:
        parts = decompose(text)
        for span in parts.syllables:
            assert parts.canonical_text[span.canonical_start : span.canonical_end] == span.text
            assert parts.base_text[span.base_start : span.base_end] == span.base_text


def test_spans_cover_alphabetic_runs_only():
    parts = decompose("Toi 2026 hoc, ok!")
    assert [s.base_text for s in parts.syllables] == ["Toi", "hoc", "ok"]


def test_digits_and_punctuation_form_no_span():
    assert decompose("2026 6,5% !!!").syllables == ()


def test_eligibility_is_undecided_not_guessed():
    """The proposal decides Vietnamese candidacy by matching a syllable
    inventory that does not exist in this repository. B1A must not invent one."""
    for text in ["ban", "AI", "machine", "learning", "Tôi", "PyTorch"]:
        for span in decompose(text).syllables:
            assert span.eligibility is Eligibility.UNDECIDED


def test_eligibility_does_not_depend_on_the_presence_of_marks():
    """Deciding candidacy from diacritics would break the proposal's invariance
    requirement: the rule must be a pure function of the stripped form."""
    marked = [s.eligibility for s in decompose("Tôi đang học").syllables]
    stripped = [s.eligibility for s in decompose("Toi dang hoc").syllables]
    assert marked == stripped


# ---------------------------------------------------------------------------
# Unusual and malformed input: preserve, flag, never delete
# ---------------------------------------------------------------------------
def test_multiple_tone_marks_in_one_syllable_are_flagged_not_repaired():
    text = _nfc("a" + M.ACUTE + M.GRAVE)
    parts = decompose(text)
    assert recompose(parts) == canon(text), "must still round-trip"
    (span,) = parts.syllables
    assert Anomaly.MULTIPLE_TONE_MARKS in span.anomalies
    assert span.observed_tone is ObservedTone.SAC  # first mark, deterministically


def test_two_toned_characters_in_one_syllable_are_flagged():
    parts = decompose("máà")
    (span,) = parts.syllables
    assert Anomaly.MULTIPLE_TONE_MARKS in span.anomalies
    assert recompose(parts) == canon("máà")


def test_unsupported_combining_marks_are_preserved_and_flagged():
    text = "cafë"  # diaeresis: not a Vietnamese mark
    parts = decompose(text)
    assert recompose(parts) == canon(text)
    assert Anomaly.UNSUPPORTED_COMBINING_MARK in parts.anomalies
    assert parts.base_text == canon(text), "a non-Vietnamese mark stays on the base"


def test_multiple_letter_diacritics_on_one_character_are_flagged():
    text = _nfc("o" + M.CIRCUMFLEX + M.HORN)
    parts = decompose(text)
    assert recompose(parts) == canon(text)
    assert Anomaly.MULTIPLE_LETTER_DIACRITICS in parts.anomalies


def test_combining_mark_with_no_base_character_is_preserved():
    text = M.ACUTE + "a"
    parts = decompose(text)
    assert recompose(parts) == canon(text)


def test_tone_mark_on_a_non_letter_is_flagged():
    text = _nfc("1" + M.ACUTE)
    parts = decompose(text)
    assert recompose(parts) == canon(text)
    assert Anomaly.TONE_MARK_ON_NON_LETTER in parts.anomalies
    assert parts.units[0].letter_diacritic is LetterDiacritic.NA


def test_empty_and_whitespace_only_input():
    for text in ["", " ", "\n\t "]:
        parts = decompose(text)
        assert parts.base_text == text
        assert parts.syllables == ()
        assert recompose(parts) == canon(text)


def test_emoji_are_preserved_and_form_no_span():
    parts = decompose("vui 😄🎉 qua")
    assert recompose(parts) == canon("vui 😄🎉 qua")
    assert [s.base_text for s in parts.syllables] == ["vui", "qua"]


def test_mixed_vietnamese_and_english_is_decomposed_without_language_guessing():
    parts = decompose("tôi dùng Python và PyTorch")
    assert [s.base_text for s in parts.syllables] == ["toi", "dung", "Python", "va", "PyTorch"]
    assert all(s.eligibility is Eligibility.UNDECIDED for s in parts.syllables)


def test_no_information_is_silently_deleted():
    """Every canonical character must be accounted for by exactly one unit."""
    for text in CURATED_TEXTS:
        parts = decompose(text)
        assert "".join(u.canonical_text for u in parts.units) == parts.canonical_text


# ---------------------------------------------------------------------------
# Channel views
# ---------------------------------------------------------------------------
def test_letter_channel_is_aligned_to_units():
    parts = decompose("Đường Nguyễn")
    assert len(parts.letter_channel) == len(parts.units)


def test_tone_channels_are_aligned_to_syllables():
    parts = decompose("Tôi đang học", source_is_clean=True)
    assert len(parts.observed_tone_channel) == len(parts.syllables)
    assert len(parts.lexical_tone_channel) == len(parts.syllables)
    # Tôi and đang are ngang; học carries nặng.
    assert parts.lexical_tone_channel == (Tone.NGANG, Tone.NGANG, Tone.NANG)


def test_lexical_channel_is_none_where_unknowable():
    parts = decompose("Tôi dang học")
    # Only the syllable with a visible mark has a knowable lexical tone.
    assert parts.lexical_tone_channel == (None, None, Tone.NANG)


# ---------------------------------------------------------------------------
# The orthography core must stay lightweight
# ---------------------------------------------------------------------------
def test_orthography_core_uses_only_the_standard_library():
    import ast
    import pathlib
    import sys

    allowed = set(sys.stdlib_module_names) | {"unmark"}
    for name in ("marks", "models", "canonical", "decompose"):
        path = pathlib.Path(__file__).resolve().parents[1] / "unmark" / "orthography" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in allowed, f"{name}.py imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".")[0] in allowed, f"{name}.py imports {node.module}"


def test_letter_diacritics_are_never_mistaken_for_tones():
    """Regression guard. ô, ê, ơ, ư, ă, â and đ change letter identity and carry
    no tone: a syllable containing only these is UNMARKED, not toned. Confusing
    the two channels is the easiest error to make in this module."""
    # Note "đương" not "đường": ờ = o + horn + GRAVE does carry huyền.
    for text in ["Tôi", "nghiên", "đương", "ăn", "cân", "ơn", "ưu"]:
        (span,) = decompose(text).syllables
        assert span.observed_tone is ObservedTone.UNMARKED, text
        assert span.lexical_tone is None, text
        assert any(u.letter_diacritic not in (LetterDiacritic.NONE, LetterDiacritic.NA) for u in decompose(text).units), text


def test_a_toned_syllable_with_letter_diacritics_is_still_toned():
    """The converse of the guard above: đường really does carry huyền, on a
    character that also carries a horn."""
    (span,) = decompose("đường").syllables
    assert span.observed_tone is ObservedTone.HUYEN
    assert span.lexical_tone is Tone.HUYEN
    assert span.base_text == "duong"


def test_a_letter_diacritic_and_a_tone_on_one_character_are_separated():
    """ợ = o + horn + dot below: horn goes to the letter channel, dot below to
    the tone channel, and the base is a bare o."""
    (unit,) = decompose("ợ").units
    assert unit.base_char == "o"
    assert unit.letter_diacritic is LetterDiacritic.HORN
    assert unit.observed_tone is ObservedTone.NANG


# ---------------------------------------------------------------------------
# G0 checker scaffold
# ---------------------------------------------------------------------------
def _load_g0_module():
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "g0_orthography_check.py"
    spec = importlib.util.spec_from_file_location("g0_orthography_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_g0_checker_exists_and_imports_without_heavy_dependencies():
    module = _load_g0_module()
    assert hasattr(module, "check_unit")
    assert hasattr(module, "summarize")


def test_g0_checker_never_claims_a_corpus_pass():
    """The strongest allowed conclusion is readiness, not G0 PASS."""
    module = _load_g0_module()
    assert module.STATUS_READY == "ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK"
    statuses = {v for k, v in vars(module).items() if k.startswith("STATUS_")}
    assert not any("G0_PASS" in s or s == "PASS" for s in statuses)


def test_g0_checker_round_trips_its_self_check_units():
    module = _load_g0_module()
    records = [module.check_unit(i, t, TonePlacement.PRESERVE) for i, t in enumerate(module.SELF_CHECK_UNITS)]
    assert all(r["passed"] for r in records)
    assert all(r["error"] is None for r in records)


def test_g0_summary_reports_canonical_only_differences_rather_than_hiding_them():
    module = _load_g0_module()
    units = ["Tiếng Việt", _nfd("Tiếng Việt")]
    records = [module.check_unit(i, t, TonePlacement.PRESERVE) for i, t in enumerate(units)]
    summary = module.summarize(records, TonePlacement.PRESERVE)
    assert summary["num_checked"] == 2
    assert summary["num_failed"] == 0
    assert summary["num_canonical_only_differences"] == 1  # the NFD one
    assert summary["input_forms"] == {"NFC": 1, "NFD": 1}


def test_g0_summary_records_that_variant_collapsing_is_not_implemented():
    module = _load_g0_module()
    summary = module.summarize([], TonePlacement.PRESERVE)
    assert summary["variant_collapsing_implemented"] is False
    assert summary["tone_placement"] == "PRESERVE"


def test_g0_checker_reports_a_genuine_round_trip_failure():
    """The checker must surface failures, not absorb them. Simulated with a
    deliberately broken reconstruction so the reporting path is exercised."""
    module = _load_g0_module()
    broken = {
        "index": 0, "input": "x", "passed": False, "error": None,
        "canonical_equals_input": True, "tone_placement_changed": False, "input_form": "NFC",
        "reconstructed": "y", "canonical": "x", "first_divergence_index": 0, "anomalies": [],
    }
    summary = module.summarize([broken], TonePlacement.PRESERVE)
    assert summary["num_failed"] == 1
    assert summary["failure_rate"] == 1.0
    report = module.render_report(
        {"run_id": "r", "timestamp_utc": "t", "source": "s", "status": module.STATUS_FAILURES,
         "max_samples": None, "tone_placement": "PRESERVE", "python_version": "3", "platform": "p",
         "script_version": "v"},
        summary,
        [broken],
    )
    assert "Representative failures" in report
    assert "reconstructed: y" in report


def test_module_and_function_share_a_name_by_design():
    """Documented shadowing (proposal 8.1 fixes both names): the package
    re-exports the function, so the plain `import ... as` form binds the
    function rather than the module, like `datetime.datetime`."""
    from unmark.orthography import decompose as exported

    assert callable(exported)
    from unmark.orthography.decompose import decompose as direct, recompose as direct_recompose

    assert direct is exported
    assert direct_recompose(direct("Đường")) == "Đường"


# ===========================================================================
# GAP-1: UNMARK's fixed nucleus-based canonical tone placement
# ===========================================================================
# The convention is a project canonicalisation choice for reproducibility, not
# a claim that other conventions are wrong. Tests are organised by the rule
# clause each one exercises, so a future change to the rule shows exactly which
# linguistic case it broke.

# (traditional/other placement, nucleus placement) pairs.
PLACEMENT_PAIRS = [
    # -- the pairs named in the specification decision --------------------
    ("hòa", "hoà"),
    ("hóa", "hoá"),
    ("thúy", "thuý"),
    ("thủy", "thuỷ"),
    ("khỏe", "khoẻ"),
    # -- oa / oe: `o` is a glide before a and e ---------------------------
    ("hòa", "hoà"),
    ("tòa", "toà"),
    ("lòa", "loà"),
    ("khòe", "khoè"),
    ("tòe", "toè"),
    # -- uy: `u` is a glide before y --------------------------------------
    ("tùy", "tuỳ"),
    ("hủy", "huỷ"),
    ("lũy", "luỹ"),
    ("mỹ", "mỹ"),  # single vowel: unchanged
    ("ủy", "uỷ"),  # no onset at all
    # -- qu-: the u belongs to the onset ----------------------------------
    ("qùa", "quà"),
    ("qúy", "quý"),
    ("qủa", "quả"),
    ("qùan", "quàn"),
    # -- gi-: the i belongs to the onset when another vowel follows -------
    ("gía", "giá"),
    ("gìa", "già"),
    ("gì", "gì"),  # no following vowel: i is the nucleus
]

# Syllables whose nucleus placement is already correct and must not move.
STABLE_SYLLABLES = [
    # ia / iê
    "kìa", "chìa", "tiếng", "chiều", "biển", "kiếm",
    # ua / uô
    "mùa", "chùa", "muốn", "tuổi", "buồn", "cuốn",
    # ưa / ươ
    "mưa", "mửa", "người", "được", "tường", "rượu", "mười",
    # yê
    "yếu", "yến", "chuyện", "nguyễn", "khuyến",
    # letter diacritic + tone
    "ắt", "ằng", "ấy", "ầm", "ế", "ề", "ố", "ồ", "ớ", "ờ", "ứ", "ừ",
    # plain diphthongs, no glide
    "mài", "cào", "kéo", "cúi", "hỏi", "dìu", "chạy", "máy",
    # ăn / â with coda
    "hoặc", "tuần", "thuở", "khuấy",
    # coda forces the last vowel of a plain pair
    "moóc",
]


def _canon_pairs():
    return [(old, new) for old, new in PLACEMENT_PAIRS]


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_placement_variants_collapse_to_the_nucleus_form(old, new):
    """1. old/new placement pair collapsing; 3. traditional input -> modern."""
    assert canon(old) == new
    assert canon(new) == new


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_canonical_form_is_idempotent_for_placement_pairs(old, new):
    """2. modern form idempotence."""
    assert canon(canon(old)) == canon(old)
    assert canon(canon(new)) == canon(new)


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_placement_is_identical_from_nfc_and_nfd(old, new):
    """4. NFC/NFD equivalence on placement-sensitive input."""
    assert canon(_nfc(old)) == canon(_nfd(old)) == new
    assert canon(old) == _nfc(canon(old))


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_placement_works_on_uppercase_equivalents(old, new):
    """5. uppercase equivalents."""
    assert canon(old.upper()) == new.upper()
    assert canon(old.capitalize()) == new.capitalize()


@pytest.mark.parametrize("syllable", STABLE_SYLLABLES)
def test_already_canonical_syllables_are_left_alone(syllable):
    """6/7. tone-only vowels and letter-diacritic + tone combinations."""
    assert canon(syllable) == syllable
    assert canon(syllable.upper()) == syllable.upper()


@pytest.mark.parametrize(
    "old,new",
    [("hòa", "hoà"), ("tòa", "toà"), ("hòan", "hoàn"), ("tòan", "toàn"), ("khóan", "khoán")],
)
def test_oa_cluster(old, new):
    """8. oa -- `o` is the glide, `a` the nucleus, with and without a coda."""
    assert canon(old) == new


@pytest.mark.parametrize("old,new", [("khỏe", "khoẻ"), ("tòe", "toè"), ("khóe", "khoé")])
def test_oe_cluster(old, new):
    """9. oe."""
    assert canon(old) == new


@pytest.mark.parametrize("old,new", [("thúy", "thuý"), ("thủy", "thuỷ"), ("tùy", "tuỳ"), ("hủy", "huỷ")])
def test_uy_cluster(old, new):
    """10. uy -- `u` is the glide, `y` the nucleus."""
    assert canon(old) == new


@pytest.mark.parametrize("old,new", [("qùa", "quà"), ("qúy", "quý"), ("qủa", "quả"), ("qùen", "quèn")])
def test_qu_onset(old, new):
    """11. qu- sequences: the u is part of the onset and never takes the tone."""
    assert canon(old) == new


def test_qu_onset_does_not_swallow_a_diacritic_nucleus():
    for syllable in ["quốc", "quyết", "quần", "quế"]:
        assert canon(syllable) == syllable


@pytest.mark.parametrize("syllable", ["kìa", "chìa", "tiếng", "chiều", "biết", "kiểm"])
def test_ia_and_ie_clusters(syllable):
    """12. ia / iê."""
    assert canon(syllable) == syllable


@pytest.mark.parametrize("syllable", ["mùa", "chùa", "muốn", "tuổi", "buồn"])
def test_ua_and_uo_clusters(syllable):
    """13. ua / uô -- `u` is NOT a glide before `a`, unlike before `y`."""
    assert canon(syllable) == syllable


@pytest.mark.parametrize("syllable", ["mưa", "mửa", "người", "được", "tưởng", "rượu", "mười"])
def test_ua_horn_and_uo_horn_clusters(syllable):
    """14. ưa / ươ -- with two diacritic vowels the tone goes on the second."""
    assert canon(syllable) == syllable


def test_uo_horn_pair_places_the_tone_on_the_second_vowel():
    from unmark.orthography import LetterDiacritic

    parts = decompose("được")
    toned = parts.units[parts.syllables[0].tone_unit_index]
    assert toned.base_char == "o"
    assert toned.letter_diacritic is LetterDiacritic.HORN


@pytest.mark.parametrize("syllable", ["yếu", "yến", "chuyện", "nguyễn", "khuyên"])
def test_ye_cluster(syllable):
    """15. yê."""
    assert canon(syllable) == syllable


def test_punctuation_around_a_syllable_is_untouched():
    """16. punctuation surrounding the syllable."""
    assert canon('"hòa",') == '"hoà",'
    assert canon("(thúy)!") == "(thuý)!"
    assert canon("...khỏe?") == "...khoẻ?"
    assert canon("hòa-bình") == "hoà-bình"


def test_mixed_vietnamese_and_english_only_moves_vietnamese_tones():
    """17. mixed Vietnamese/English text."""
    assert canon("hòa với machine learning") == "hoà với machine learning"
    assert canon("Python và PyTorch") == "Python và PyTorch"
    assert canon("toi dung Python") == "toi dung Python"


def test_urls_and_emails_are_preserved():
    """18. URLs / e-mail preservation."""
    for text in [
        "https://example.edu.vn/tuyen-sinh?id=42&lang=vi",
        "lien.he@example.com",
        "Xem tai https://example.com/a_b-c.html nhe",
        "Gui toi user.name+tag@example.co.uk",
    ]:
        assert canon(text) == text


def test_malformed_multiple_tone_input_is_left_alone_and_still_flagged():
    """19. malformed multiple-tone input is not repaired."""
    text = _nfc("a" + M.ACUTE + M.GRAVE)
    assert canon(text) == _nfc(text)
    parts = decompose(text)
    assert Anomaly.MULTIPLE_TONE_MARKS in parts.anomalies
    assert recompose(parts) == canon(text)

    two_chars = "máà"
    assert canon(two_chars) == _nfc(two_chars)
    assert Anomaly.MULTIPLE_TONE_MARKS in decompose(two_chars).anomalies


def test_tone_mark_on_a_non_letter_is_not_relocated():
    text = _nfc("1" + M.ACUTE)
    assert canon(text) == _nfc(text)
    assert Anomaly.TONE_MARK_ON_NON_LETTER in decompose(text).anomalies


def test_unsupported_combining_marks_survive_canonicalisation():
    text = "cafë"
    assert canon(text) == _nfc(text)
    assert Anomaly.UNSUPPORTED_COMBINING_MARK in decompose(text).anomalies


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_decomposition_channels_agree_across_placement_variants(old, new):
    """20. All four channel-relevant views must be identical for the variants."""
    a = decompose(old)
    b = decompose(new)
    assert a.base_text == b.base_text
    assert a.letter_channel == b.letter_channel
    assert a.observed_tone_channel == b.observed_tone_channel
    assert a.lexical_tone_channel == b.lexical_tone_channel
    assert a.canonical_text == b.canonical_text


@pytest.mark.parametrize("old,new", _canon_pairs())
def test_round_trip_holds_for_both_placement_variants(old, new):
    for text in (old, new, old.upper(), _nfd(old)):
        assert recompose(decompose(text)) == canon(text)


# --- structural invariance of the placement step ---------------------------
@pytest.mark.parametrize("text", CURATED_TEXTS + [old for old, _ in PLACEMENT_PAIRS])
def test_canonicalisation_never_changes_the_lexical_base(text):
    """Only tone position may move: letters, case, digits and spacing survive."""
    assert base_signature(text, collapse_whitespace=False) == base_signature(
        canon(text), collapse_whitespace=False
    )


@pytest.mark.parametrize("text", CURATED_TEXTS + [old for old, _ in PLACEMENT_PAIRS])
def test_canonicalisation_preserves_the_letter_diacritic_channel(text):
    """ơ, ư, â, ê must never be altered to relocate a tone."""
    assert decompose(text).letter_channel == decompose(canon(text)).letter_channel


def test_canonicalisation_preserves_whitespace_and_length_of_the_base():
    text = "  hòa   thúy\tkhỏe  "
    assert canon(text) == "  hoà   thuý\tkhoẻ  "


def test_tone_never_crosses_a_consonant():
    """Relocation is bounded to the vowel cluster that already holds the tone."""
    parts_before = decompose("hòa", placement=TonePlacement.PRESERVE)
    parts_after = decompose("hòa")
    before = parts_before.syllables[0]
    after = parts_after.syllables[0]
    assert before.base_text == after.base_text == "hoa"
    assert before.tone_unit_index == 1 and after.tone_unit_index == 2


# --- the rule is a rule, not a lookup table -------------------------------
def test_placement_is_rule_based_not_a_hard_coded_example_table():
    """Syllables that appear in no test list and in no source file must still
    canonicalise correctly, which a lookup table could not do."""
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "unmark" / "orthography" / "placement.py"
    body = source.read_text(encoding="utf-8")

    unseen = [("phòa", "phoà"), ("nhòe", "nhoè"), ("chúy", "chuý"), ("nhũy", "nhuỹ"), ("khóa", "khoá")]
    for old, new in unseen:
        assert canon(old) == new, old
        assert old not in body and new not in body, f"{old}/{new} must not be hard-coded"


def test_nucleus_index_is_exposed_for_inspection():
    from unmark.orthography import find_nucleus_index

    assert callable(find_nucleus_index)


def test_g0_checker_uses_the_locked_canonical_convention():
    module = _load_g0_module()
    from unmark.orthography import DEFAULT_TONE_PLACEMENT

    assert DEFAULT_TONE_PLACEMENT is TonePlacement.MODERN
    records = [
        module.check_unit(i, t, DEFAULT_TONE_PLACEMENT) for i, t in enumerate(module.SELF_CHECK_UNITS)
    ]
    assert all(r["passed"] for r in records)
    summary = module.summarize(records, DEFAULT_TONE_PLACEMENT)
    assert summary["tone_placement"] == "MODERN"
    assert summary["variant_collapsing_implemented"] is True


def test_g0_self_check_units_include_placement_variants():
    module = _load_g0_module()
    assert "hòa" in module.SELF_CHECK_UNITS and "hoà" in module.SELF_CHECK_UNITS


def test_g0_summary_separates_normalisation_from_placement_collapsing():
    module = _load_g0_module()
    units = ["hoà", "hòa", _nfd("Tiếng Việt")]
    records = [module.check_unit(i, t, TonePlacement.MODERN) for i, t in enumerate(units)]
    summary = module.summarize(records, TonePlacement.MODERN)
    assert summary["num_failed"] == 0
    assert summary["num_tone_placement_collapsed"] == 1  # only "hòa"
    assert summary["num_canonical_only_differences"] == 2  # "hòa" and the NFD one
