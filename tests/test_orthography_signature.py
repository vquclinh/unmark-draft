"""Unit tests for the Vietnamese base signature.

Pure standard library plus pytest: no torch, no transformers, no checkpoint.
These run inside the lightweight local `.venv` (`requirements/dev.txt`).
"""

from __future__ import annotations

import unicodedata

import pytest

from unmark.orthography import (
    base_signature,
    first_divergence,
    rewrite_signature,
    strip_vietnamese_diacritics,
    word_diff,
)


# ---------------------------------------------------------------------------
# base_signature: whole sentences
# ---------------------------------------------------------------------------
def test_base_signature_strips_full_vietnamese_sentence():
    assert base_signature("Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.") == (
        "Toi dang nghien cuu xu ly ngon ngu tu nhien."
    )


def test_base_signature_is_idempotent():
    text = "Hôm nay thời tiết ở Thành phố Hồ Chí Minh rất đẹp."
    once = base_signature(text)
    assert base_signature(once) == once


def test_base_signature_of_plain_ascii_is_unchanged():
    text = "toi dang nghien cuu xu ly ngon ngu tu nhien"
    assert base_signature(text) == text


def test_base_signature_is_strip_plus_whitespace_normalisation():
    """The two helpers must not drift apart: base_signature is defined as
    strip_vietnamese_diacritics plus the documented whitespace step."""
    text = "  Đường   Nguyễn Huệ  "
    assert base_signature(text, collapse_whitespace=False) == strip_vietnamese_diacritics(text)
    assert base_signature(text) == " ".join(strip_vietnamese_diacritics(text).split())


# ---------------------------------------------------------------------------
# NFC / NFD equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["Tiếng Việt", "Đường Nguyễn Huệ", "ưỡn ẵm ộp", "Hồ Chí Minh", "cà phê sữa đá"],
)
def test_nfc_and_nfd_spellings_share_a_signature(text):
    nfc_form = unicodedata.normalize("NFC", text)
    nfd_form = unicodedata.normalize("NFD", text)
    assert nfc_form != nfd_form, "test string must actually have a decomposed form"
    assert base_signature(nfc_form) == base_signature(nfd_form)


def test_nfd_input_is_not_left_decomposed():
    """The signature must be an NFC string, otherwise comparing an NFC input
    against an NFD output would fail for the wrong reason."""
    sig = base_signature(unicodedata.normalize("NFD", "cà phê"))
    assert sig == unicodedata.normalize("NFC", sig)


# ---------------------------------------------------------------------------
# tone marks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "toned,base",
    [
        ("a à á ả ã ạ", "a a a a a a"),
        ("e è é ẻ ẽ ẹ", "e e e e e e"),
        ("i ì í ỉ ĩ ị", "i i i i i i"),
        ("o ò ó ỏ õ ọ", "o o o o o o"),
        ("u ù ú ủ ũ ụ", "u u u u u u"),
        ("y ỳ ý ỷ ỹ ỵ", "y y y y y y"),
    ],
)
def test_all_six_tones_reduce_to_the_bare_vowel(toned, base):
    assert base_signature(toned) == base


def test_tone_marks_on_modified_letters_are_stripped_together():
    # ế = e + circumflex + acute; ự = u + horn + dot below
    assert base_signature("ế ự ẫ ặ ỗ ợ") == "e u a a o o"


def test_tone_marks_are_stripped_in_uppercase_too():
    assert base_signature("TIẾNG VIỆT") == "TIENG VIET"


# ---------------------------------------------------------------------------
# letter diacritics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "letter,base",
    [
        ("ă", "a"), ("â", "a"), ("ê", "e"), ("ô", "o"), ("ơ", "o"), ("ư", "u"),
        ("Ă", "A"), ("Â", "A"), ("Ê", "E"), ("Ô", "O"), ("Ơ", "O"), ("Ư", "U"),
    ],
)
def test_letter_diacritics_reduce_to_the_base_letter(letter, base):
    assert base_signature(letter) == base


# ---------------------------------------------------------------------------
# d with stroke
# ---------------------------------------------------------------------------
def test_d_stroke_maps_to_plain_d_preserving_case():
    assert base_signature("đ") == "d"
    assert base_signature("Đ") == "D"
    assert base_signature("Đường Đinh Tiên Hoàng") == "Duong Dinh Tien Hoang"
    assert base_signature("đại học") == "dai hoc"


def test_d_stroke_is_case_sensitive():
    assert base_signature("Đ") != "d"


# ---------------------------------------------------------------------------
# everything else is preserved
# ---------------------------------------------------------------------------
def test_punctuation_digits_and_case_are_preserved():
    assert base_signature("Năm 2026, GDP tăng 6,5% so với năm trước.") == (
        "Nam 2026, GDP tang 6,5% so voi nam truoc."
    )


@pytest.mark.parametrize(
    "text",
    [
        "Ban co chac khong? Toi nghi la khong!",
        "Cuoc hop bat dau luc 14:30 ngay 19/08/2026.",
        "Gia ve la 250.000 dong (da bao gom thue VAT 10%).",
        "https://example.edu.vn/tuyen-sinh?id=42&lang=vi",
        "lien.he@example.com",
        "\U0001F604 \U0001F389 ☔",
        "machine learning, Python, PyTorch, VNU-HCM",
    ],
)
def test_non_vietnamese_content_passes_through_untouched(text):
    assert base_signature(text) == text


def test_non_vietnamese_combining_marks_are_not_stripped():
    """Diaeresis and cedilla are not Vietnamese diacritics and must survive."""
    assert base_signature("Müller") == "Müller"
    assert base_signature("façade") == "façade"


def test_documented_collateral_stripping_of_shared_marks():
    """Documented limitation: acute and tilde are stripped wherever they occur,
    because the rule is codepoint-based. Recorded as a test so the behaviour is
    a decision rather than a surprise."""
    assert base_signature("café") == "cafe"
    assert base_signature("mañana") == "manana"


# ---------------------------------------------------------------------------
# whitespace
# ---------------------------------------------------------------------------
def test_whitespace_is_collapsed_by_default():
    assert base_signature("  toi   dang\thoc\n") == "toi dang hoc"


def test_strict_mode_preserves_whitespace_differences():
    assert base_signature("toi  dang hoc", collapse_whitespace=False) != base_signature(
        "toi dang hoc", collapse_whitespace=False
    )
    assert base_signature("toi  dang hoc") == base_signature("toi dang hoc")


def test_non_breaking_space_is_collapsed_like_ordinary_space():
    assert base_signature("toi\u00a0dang hoc") == "toi dang hoc"
    assert base_signature("toi\u00a0dang hoc", collapse_whitespace=False) != "toi dang hoc"


def test_empty_and_whitespace_only_input():
    assert base_signature("") == ""
    assert base_signature("   ") == ""
    assert base_signature("   ", collapse_whitespace=False) == "   "


# ---------------------------------------------------------------------------
# the diagnostic must still catch real rewrites
# ---------------------------------------------------------------------------
def test_lexical_rewrites_are_not_normalised_away():
    """The point of the diagnostic: mark changes are invisible, word changes are not."""
    assert base_signature("toi di hoc") == base_signature("tôi đi học")  # marks only
    assert base_signature("toi di hoc") != base_signature("tôi đi làm")  # word replaced
    assert base_signature("toi di hoc") != base_signature("tôi đi học nhé")  # word inserted
    assert base_signature("toi di hoc") != base_signature("tôi học")  # word deleted


# ---------------------------------------------------------------------------
# diff helpers
# ---------------------------------------------------------------------------
def test_word_diff_is_empty_for_identical_strings():
    assert word_diff("toi di hoc", "toi di hoc") == []


def test_word_diff_reports_the_replaced_word():
    changes = word_diff("toi di hoc", "toi di lam")
    assert len(changes) == 1
    assert changes[0]["op"] == "replace"
    assert changes[0]["input_words"] == ["hoc"]
    assert changes[0]["output_words"] == ["lam"]


def test_word_diff_reports_insertion_and_deletion():
    assert word_diff("a b", "a b c")[0]["op"] == "insert"
    assert word_diff("a b c", "a b")[0]["op"] == "delete"


def test_first_divergence():
    assert first_divergence("abc", "abc") is None
    assert first_divergence("abc", "abd") == 2
    assert first_divergence("abc", "abcd") == 3


# ---------------------------------------------------------------------------
# rewrite_signature: the engineering lexical-preservation diagnostic
# ---------------------------------------------------------------------------
# Strict base_signature answers "what did the restorer change at all?".
# rewrite_signature answers the narrower question the G-1 engineering check
# needs: "did the restorer rewrite any word?". It tolerates exactly two
# formatting differences -- letter case, and punctuation terminating the whole
# string -- and must remain blind to nothing else.


def _kept(a: str, b: str) -> bool:
    """Would the engineering lexical check treat a -> b as preserved?"""
    return rewrite_signature(a) == rewrite_signature(b)


# --- the two cases named in audit 001 -------------------------------------
def test_audit_001_false_failure_case_is_now_lexically_preserved():
    src = "hom nay thoi tiet o thanh pho ho chi minh rat dep"
    out = "Hôm nay thời tiết ở Thành phố Hồ Chí Minh rất đẹp."
    assert base_signature(src) != base_signature(out)  # strict still reports it
    assert _kept(src, out)  # engineering check does not


def test_audit_001_genuine_rewrite_is_still_caught():
    assert not _kept("toi dang hoc AI", "Tôi đang nghiên cứu AI.")


# --- tolerated formatting --------------------------------------------------
def test_sentence_initial_capitalisation_is_tolerated():
    assert _kept("toi di hoc", "Tôi đi học")


def test_proper_noun_capitalisation_is_tolerated():
    assert _kept("nguyen viet anh dang lam viec tai ha noi", "Nguyễn Viết Anh đang làm việc tại Hà Nội")


def test_all_caps_input_is_tolerated():
    assert _kept("TOI DI HOC", "Tôi đi học")


def test_final_period_is_tolerated():
    assert _kept("toi di hoc", "Tôi đi học.")


def test_final_question_mark_is_tolerated():
    assert _kept("ban co khoe khong", "Bạn có khoẻ không?")


def test_final_exclamation_mark_is_tolerated():
    assert _kept("chuc mung sinh nhat", "Chúc mừng sinh nhật!")


def test_final_ellipsis_is_tolerated_both_spellings():
    assert _kept("toi khong biet", "Tôi không biết…")
    assert _kept("toi khong biet", "Tôi không biết...")


def test_repeated_terminal_punctuation_is_tolerated():
    assert _kept("that tuyet", "Thật tuyệt!!!")
    assert _kept("that tuyet", "Thật tuyệt?!")


def test_capitalisation_and_final_punctuation_together_are_tolerated():
    assert _kept("hom nay troi dep", "Hôm nay trời đẹp.")
    assert _kept("ban ten la gi", "Bạn tên là gì?")


# --- genuine lexical change is NOT tolerated ------------------------------
def test_word_replacement_is_detected():
    assert not _kept("toi di hoc", "Tôi đi làm.")


def test_word_insertion_is_detected():
    assert not _kept("toi di hoc", "Tôi đi học ngay.")


def test_word_deletion_is_detected():
    assert not _kept("toi di hoc som", "Tôi đi học.")


def test_single_character_lexical_change_is_detected():
    assert not _kept("toi di hoc", "Tôi đi hoct.")


def test_word_reordering_is_detected():
    assert not _kept("toi di hoc", "Học đi tôi.")


# --- internal punctuation is preserved -------------------------------------
def test_internal_punctuation_removal_is_detected():
    assert not _kept("Ban co chac khong? Toi nghi la khong!", "Bạn có chắc không Tôi nghĩ là không!")


def test_internal_punctuation_addition_is_detected():
    assert not _kept("toi di hoc roi ve nha", "Tôi đi học, rồi về nhà.")


def test_only_trailing_punctuation_is_ignored_not_all_punctuation():
    # The comma sits inside the string and must survive.
    assert not _kept("Nam 2026 GDP tang", "Năm 2026, GDP tăng.")


def test_hyphen_inside_a_token_is_preserved():
    assert not _kept("VNU HCM", "VNU-HCM")


# --- URLs, e-mail, digits --------------------------------------------------
def test_url_is_preserved_not_punctuation_stripped():
    src = "Tai lieu o http://example.com/tai-lieu?id=42&lang=vi"
    assert _kept(src, "Tài liệu ở http://example.com/tai-lieu?id=42&lang=vi")
    assert not _kept(src, "Tài liệu ở http://example.com/tai-lieu?id=43&lang=vi")
    assert not _kept(src, "Tài liệu ở http://example.com/tailieu?id=42&lang=vi")


def test_url_internal_dots_survive_terminal_stripping():
    assert "example.com" in rewrite_signature("Xem https://example.com")
    assert not _kept("Xem https://example.com", "Xem https://examplecom")


def test_trailing_stop_after_a_url_is_still_tolerated():
    assert _kept("Xem https://example.com", "Xem https://example.com.")


def test_email_structure_is_preserved():
    src = "Lien he qua lien.he@example.com nhe"
    assert _kept(src, "Liên hệ qua lien.he@example.com nhé")
    assert not _kept(src, "Liên hệ qua lienhe@example.com nhé")
    assert not _kept(src, "Liên hệ qua lien.he@example.org nhé")


def test_digits_are_preserved():
    assert _kept("nam 2026 gdp tang 6,5%", "Năm 2026 GDP tăng 6,5%")
    assert not _kept("nam 2026 gdp tang 6,5%", "Năm 2027 GDP tăng 6,5%")
    assert not _kept("nam 2026 gdp tang 6,5%", "Năm 2026 GDP tăng 6,6%")


def test_percent_and_currency_symbols_are_preserved():
    assert not _kept("thue VAT 10%", "Thuế VAT 10")
    assert not _kept("gia 250.000 dong", "Giá 250.000 đồng nhé")


def test_time_and_date_separators_are_preserved():
    assert _kept("cuoc hop luc 14:30 ngay 19/08/2026", "Cuộc họp lúc 14:30 ngày 19/08/2026")
    assert not _kept("cuoc hop luc 14:30 ngay 19/08/2026", "Cuộc họp lúc 14h30 ngày 19/08/2026")


# --- shared semantics with base_signature ---------------------------------
def test_rewrite_signature_reuses_the_diacritic_semantics():
    """It must be base_signature plus casefold plus terminal stripping, not a
    second, independently drifting implementation."""
    for text in ["Đường Nguyễn Huệ", "tiếng Việt", "ĐẠI HỌC", "cà phê"]:
        assert rewrite_signature(text) == base_signature(text).casefold()


def test_rewrite_signature_is_idempotent():
    for text in ["Hôm nay trời đẹp.", "Bạn khoẻ không?", "toi di hoc"]:
        once = rewrite_signature(text)
        assert rewrite_signature(once) == once


def test_rewrite_signature_still_ignores_diacritics_and_whitespace():
    assert _kept("toi   di\thoc", "Tôi đi học")
    assert _kept("Đường", "duong")


def test_rewrite_signature_handles_empty_and_punctuation_only_input():
    assert rewrite_signature("") == ""
    assert rewrite_signature("   ") == ""
    assert rewrite_signature("...") == ""
    assert rewrite_signature("?!") == ""
