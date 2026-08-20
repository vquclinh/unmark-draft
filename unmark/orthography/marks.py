"""Vietnamese mark inventories and channel state sets.

Single source of truth for *which* Unicode combining characters this project
treats as Vietnamese, and for the typed states of the tone and letter-diacritic
channels. Pure standard library; nothing here imports torch or transformers.

Proposal reference (v1.3 section 4.2): "separation of combining marks into tone
marks (U+0300, U+0301, U+0303, U+0309, U+0323) and letter-forming marks
(U+0302, U+0306, U+031B, plus the đ stroke)". Those eight codepoints plus the
stroke are the entire Vietnamese inventory; every other combining mark is
deliberately *not* Vietnamese and is preserved untouched (section 4.3 treats
non-Vietnamese material as `N/A`).
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Combining marks
# ---------------------------------------------------------------------------
# Tone marks. Five of the six Vietnamese tones are written with a mark; the
# sixth, `ngang`, is written with no mark at all and therefore has no codepoint.
# That absence is the premise of the whole UNMARK project (proposal 1.2).
GRAVE = "\u0300"  # huyền
ACUTE = "\u0301"  # sắc
TILDE = "\u0303"  # ngã
HOOK_ABOVE = "\u0309"  # hỏi
DOT_BELOW = "\u0323"  # nặng

TONE_MARKS = GRAVE + ACUTE + TILDE + HOOK_ABOVE + DOT_BELOW

# Letter-forming marks. These change letter identity (a/ă/â, e/ê, o/ô/ơ, u/ư)
# and are not tones (proposal 1.1).
CIRCUMFLEX = "\u0302"  # â, ê, ô
BREVE = "\u0306"  # ă
HORN = "\u031B"  # ơ, ư

LETTER_MARKS = CIRCUMFLEX + BREVE + HORN

VIETNAMESE_MARKS = TONE_MARKS + LETTER_MARKS

# `d` with stroke has no canonical decomposition, so it is handled by an
# explicit map rather than as a combining mark.
D_STROKE = {"đ": "d", "Đ": "D"}  # đ -> d, Đ -> D
D_STROKE_INVERSE = {base: stroked for stroked, base in D_STROKE.items()}


# ---------------------------------------------------------------------------
# Tone channel: three deliberately separate concepts
# ---------------------------------------------------------------------------
class Tone(Enum):
    """LEXICAL tone: the true tone of a syllable, knowable only from clean text.

    Six values, `ngang` included. This is *not* what a deployed system observes;
    see :class:`ObservedTone`. A decomposition only fills this in when the caller
    asserts the source text is clean, or when a visible mark settles it.
    """

    NGANG = "NGANG"
    SAC = "SAC"
    HUYEN = "HUYEN"
    HOI = "HOI"
    NGA = "NGA"
    NANG = "NANG"


class ObservedTone(Enum):
    """DEPLOYABLE tone: what can be read off the string at inference time.

    `UNMARKED` means exactly "no visible tone mark". It does **not** mean
    `Tone.NGANG`. Proposal 1.2 and 4.3: a syllable with no mark is either a
    genuine `ngang` or a syllable whose mark was lost, and the orthography
    cannot tell which. Conflating the two is the error UNMARK exists to avoid,
    so the two enums are kept separate at the type level rather than by
    convention.
    """

    UNMARKED = "UNMARKED"
    SAC = "SAC"
    HUYEN = "HUYEN"
    HOI = "HOI"
    NGA = "NGA"
    NANG = "NANG"


# The five marked tones map one-to-one between the two enums; `ngang` and
# `UNMARKED` deliberately have no counterpart in the other.
TONE_MARK_TO_TONE: dict[str, Tone] = {
    ACUTE: Tone.SAC,
    GRAVE: Tone.HUYEN,
    HOOK_ABOVE: Tone.HOI,
    TILDE: Tone.NGA,
    DOT_BELOW: Tone.NANG,
}

TONE_MARK_TO_OBSERVED: dict[str, ObservedTone] = {
    ACUTE: ObservedTone.SAC,
    GRAVE: ObservedTone.HUYEN,
    HOOK_ABOVE: ObservedTone.HOI,
    TILDE: ObservedTone.NGA,
    DOT_BELOW: ObservedTone.NANG,
}

TONE_TO_MARK: dict[Tone, str] = {tone: mark for mark, tone in TONE_MARK_TO_TONE.items()}


def observed_from_lexical(tone: Tone) -> ObservedTone:
    """Project a lexical tone onto what would be observable.

    Total and lossy in one direction only: `NGANG` becomes `UNMARKED`, which is
    exactly the information loss the project studies. There is deliberately no
    inverse function.
    """
    if tone is Tone.NGANG:
        return ObservedTone.UNMARKED
    return ObservedTone[tone.name]


# ---------------------------------------------------------------------------
# Letter-diacritic channel
# ---------------------------------------------------------------------------
class LetterDiacritic(Enum):
    """Per-character letter-diacritic state (proposal 4.3: character level).

    `NONE` and `NA` are different and must not be conflated:

    * `NONE` -- a letter that *could* carry a Vietnamese letter diacritic and
      does not (the `a` in `ban`).
    * `NA` -- the channel does not apply at all: space, punctuation, digit,
      symbol, emoji.
    """

    NONE = "NONE"
    BREVE = "BREVE"
    CIRCUMFLEX = "CIRCUMFLEX"
    HORN = "HORN"
    STROKE = "STROKE"
    NA = "NA"


LETTER_MARK_TO_STATE: dict[str, LetterDiacritic] = {
    BREVE: LetterDiacritic.BREVE,
    CIRCUMFLEX: LetterDiacritic.CIRCUMFLEX,
    HORN: LetterDiacritic.HORN,
}

STATE_TO_LETTER_MARK: dict[LetterDiacritic, str] = {
    state: mark for mark, state in LETTER_MARK_TO_STATE.items()
}


# ---------------------------------------------------------------------------
# Vietnamese-candidate eligibility
# ---------------------------------------------------------------------------
class Eligibility(Enum):
    """Whether an orthographic span is a Vietnamese candidate.

    Proposal 4.3: "an alphabetic span is treated as a Vietnamese candidate if it
    matches the Vietnamese syllable inventory after stripping; otherwise both
    channels are `N/A`". B3A resolves this against a pinned inventory
    (`unmark.linguistics`); GAP-2 is closed.

    The rule is a pure function of the *stripped* form, so it assigns identical
    labels to clean and corrupted input and cannot break grid invariance. Using
    the presence of diacritics to decide would destroy that, so nothing in this
    project ever does.
    """

    VIETNAMESE_CANDIDATE = "VIETNAMESE_CANDIDATE"
    """Resolved: the stripped form is in the pinned Vietnamese syllable
    inventory. Both channels apply."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """Resolved: not Vietnamese. Digits, punctuation, symbols and emoji are
    trivially N/A; so is an alphabetic span whose stripped form is not in the
    inventory. Both channels are `N/A` (proposal 4.3)."""

    UNDECIDED = "UNDECIDED"
    """**Unresolvable**, not "unknown-but-probably-not": the inventory is not
    available, so no classification was attempted. Never used for a span that
    was actually checked."""


class Anomaly(Enum):
    """Deviations from well-formed Vietnamese orthography, recorded not repaired.

    Nothing here changes the text or the reconstruction; these flags exist so a
    reader can find unusual input instead of it being silently absorbed.
    """

    MULTIPLE_TONE_MARKS = "MULTIPLE_TONE_MARKS"
    MULTIPLE_LETTER_DIACRITICS = "MULTIPLE_LETTER_DIACRITICS"
    UNSUPPORTED_COMBINING_MARK = "UNSUPPORTED_COMBINING_MARK"
    TONE_MARK_ON_NON_LETTER = "TONE_MARK_ON_NON_LETTER"
