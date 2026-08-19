"""Grouping a string into base-character + combining-mark units.

Shared by `placement` (which relocates tone marks) and `decompose` (which
classifies them), so the two cannot disagree about where one letter ends and
the next begins. Pure standard library.

A "unit" is one base codepoint together with the combining marks that follow
it in NFD order -- what a reader sees as a single letter. Units exist because a
Vietnamese letter is one codepoint in NFC and up to three in NFD.
"""

from __future__ import annotations

import unicodedata


def split_units(text_nfd: str) -> list[tuple[str, tuple[str, ...]]]:
    """Group an NFD string into `(base codepoint, combining marks)` pairs.

    A leading combining mark with no base -- malformed but possible input -- is
    kept as its own unit with an empty base, so nothing is ever discarded.
    """
    units: list[tuple[str, tuple[str, ...]]] = []
    marks: list[str] = []
    base: str | None = None
    for ch in text_nfd:
        if unicodedata.combining(ch):
            if base is None:
                # Combining mark with nothing to attach to: preserve verbatim.
                units.append(("", (ch,)))
                continue
            marks.append(ch)
        else:
            if base is not None:
                units.append((base, tuple(marks)))
            base, marks = ch, []
    if base is not None:
        units.append((base, tuple(marks)))
    return units


def join_units(units: list[tuple[str, tuple[str, ...]]]) -> str:
    """Inverse of :func:`split_units`, before normalisation.

    Callers normalise the result themselves; NFC re-applies canonical ordering,
    so marks need not be appended in canonical order.
    """
    return "".join(base + "".join(marks) for base, marks in units)
