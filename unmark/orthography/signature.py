"""Vietnamese base signature: the comparison form used by the G-1 diagnostic.

A diacritic restorer is supposed to *add marks*, not to *rewrite words*. The
base signature makes that testable: two strings share a base signature exactly
when they differ only in Vietnamese diacritics (plus documented whitespace
normalisation). If a restorer's input and output signatures diverge, it changed
the lexical base -- which at this gate is the serious failure mode, not a
mis-chosen tone.

Pure standard library on purpose: this module must import with nothing beyond
`requirements/dev.txt`.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any

# The five Vietnamese tone marks, as Unicode combining characters. `ngang`
# (the level tone) is written with no mark at all and therefore has no
# codepoint -- that absence is the premise of the whole UNMARK project.
TONE_MARKS = (
    "\u0300"  # grave      - huyen
    "\u0301"  # acute      - sac
    "\u0303"  # tilde      - nga
    "\u0309"  # hook above - hoi
    "\u0323"  # dot below  - nang
)

# Letter diacritics. These form distinct letters (a/ă/â, e/ê, o/ô/ơ, u/ư) and
# belong to the letter identity rather than to the tone.
LETTER_MARKS = (
    "\u0302"  # circumflex - a-hat, e-hat, o-hat
    "\u0306"  # breve      - a-breve
    "\u031B"  # horn       - o-horn, u-horn
)

VIETNAMESE_MARKS = TONE_MARKS + LETTER_MARKS

# `d` with stroke has no canonical decomposition, so it needs an explicit map.
D_STROKE = {"đ": "d", "Đ": "D"}

_D_STROKE_TABLE = str.maketrans(D_STROKE)
_MARK_TABLE = {ord(ch): None for ch in VIETNAMESE_MARKS}


def nfc(text: str) -> str:
    """Unicode NFC normalisation (the canonical form used throughout G-1)."""
    return unicodedata.normalize("NFC", text)


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese tone marks and letter diacritics, nothing else.

    Order of operations:

    1. NFC-normalise, so NFC and NFD spellings of the same string agree;
    2. map ``đ``/``Đ`` to ``d``/``D`` (no canonical decomposition exists);
    3. NFD-decompose, then delete the five tone marks and the three letter
       diacritics (circumflex, breve, horn);
    4. NFC-recompose, so the result is a canonical string that can be compared
       against another NFC string by ordinary equality.

    Whitespace, case, digits, punctuation, symbols, emoji, URLs and e-mail
    addresses are untouched. Non-Vietnamese combining marks (diaeresis,
    cedilla, ...) are *not* stripped.

    Documented limitation: step 3 is codepoint-based, so a non-Vietnamese letter
    carrying one of the same marks is also stripped (``café`` -> ``cafe``,
    ``mañana`` -> ``manana``). Since the function is applied identically to a
    restorer's input and output, this never manufactures a false match between
    two different lexical bases; it only means the diagnostic is blind to those
    marks inside loanwords.
    """
    text = nfc(text)
    text = text.translate(_D_STROKE_TABLE)
    text = unicodedata.normalize("NFD", text)
    text = text.translate(_MARK_TABLE)
    return nfc(text)


def base_signature(text: str, *, collapse_whitespace: bool = True) -> str:
    """Diacritic-free comparison form of ``text``.

    This is :func:`strip_vietnamese_diacritics` plus one optional, explicitly
    documented normalisation: runs of Unicode whitespace collapse to a single
    space and the ends are stripped. That step is a genuine normalisation, so
    callers that need to see raw spacing differences pass
    ``collapse_whitespace=False``; the G-1 runner records both forms and flags
    whitespace-only differences rather than hiding them.
    """
    text = strip_vietnamese_diacritics(text)
    if collapse_whitespace:
        text = " ".join(text.split())
    return text


def first_divergence(a: str, b: str) -> int | None:
    """Index of the first differing character, or ``None`` if ``a == b``."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def word_diff(a: str, b: str) -> list[dict[str, Any]]:
    """Whitespace-token diff, used to inspect a base-signature mismatch.

    Returns one entry per non-equal opcode, so a reader can see exactly which
    words the model replaced, inserted or deleted.
    """
    a_words = a.split()
    b_words = b.split()
    matcher = SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    changes: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        changes.append(
            {
                "op": op,
                "input_index": i1,
                "output_index": j1,
                "input_words": a_words[i1:i2],
                "output_words": b_words[j1:j2],
            }
        )
    return changes


# ---------------------------------------------------------------------------
# Engineering rewrite signature
# ---------------------------------------------------------------------------
# `base_signature` is deliberately strict: it preserves case and punctuation, so
# it reports *any* change beyond the diacritics themselves. That strictness is
# what makes it a good record of what the restorer actually did, and it stays
# unchanged.
#
# It is the wrong instrument for one specific decision: the G-1 engineering
# check "did the model rewrite the lexical base?". A restorer trained on
# Wikipedia and news routinely capitalises a lowercase sentence and adds a final
# stop. Those are formatting changes, not lexical rewrites, and scoring them as
# rewrites would fail the gate for the wrong reason.
#
# `rewrite_signature` answers only that narrower question. It tolerates exactly
# two harmless formatting differences -- letter case, and punctuation that
# terminates the whole string -- and nothing else. Word substitutions,
# insertions and deletions, internal punctuation, digits, URLs and e-mail
# addresses all still show up as differences.

# Terminal sentence punctuation. Only a run of these at the very end of the
# string is ignored; the same characters inside the string are significant.
TERMINAL_PUNCTUATION = ".!?…"


def _strip_terminal_punctuation(text: str) -> str:
    """Drop a trailing run of sentence-final punctuation and spacing.

    Only the end of the whole string is affected, so internal punctuation --
    the `?` in "Ban co chac khong? Toi nghi la khong!", the dots inside
    "example.com" or "250.000" -- is untouched.
    """
    return text.rstrip(TERMINAL_PUNCTUATION + " \t\r\n")


def rewrite_signature(text: str) -> str:
    """Comparison form for the G-1 *engineering* lexical-rewrite check.

    Built on :func:`base_signature`, so the Vietnamese diacritic semantics are
    shared rather than reimplemented, then relaxed in exactly two ways:

    1. ``str.casefold`` -- sentence-initial and proper-noun capitalisation
       introduced by a restorer is not a lexical rewrite;
    2. a trailing run of ``.``, ``!``, ``?`` or ``…`` is removed -- adding a
       final stop to an unpunctuated input is not a lexical rewrite either.

    Everything else survives, deliberately: internal punctuation, digits, URL
    and e-mail structure, and of course the words themselves. Two strings share
    a rewrite signature only if they contain the same sequence of lexical
    tokens, up to diacritics, case and a final stop.

    This is *not* a replacement for :func:`base_signature`. Both are computed
    and both are recorded; they answer different questions.
    """
    return _strip_terminal_punctuation(base_signature(text).casefold())
