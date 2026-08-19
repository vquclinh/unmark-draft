"""Vietnamese orthographic utilities (pure Python, no ML dependencies)."""

from unmark.orthography.signature import (
    D_STROKE,
    LETTER_MARKS,
    TERMINAL_PUNCTUATION,
    TONE_MARKS,
    VIETNAMESE_MARKS,
    base_signature,
    first_divergence,
    nfc,
    rewrite_signature,
    strip_vietnamese_diacritics,
    word_diff,
)

__all__ = [
    "D_STROKE",
    "LETTER_MARKS",
    "TERMINAL_PUNCTUATION",
    "TONE_MARKS",
    "VIETNAMESE_MARKS",
    "base_signature",
    "first_divergence",
    "nfc",
    "rewrite_signature",
    "strip_vietnamese_diacritics",
    "word_diff",
]
