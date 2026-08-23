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
from dataclasses import dataclass


@dataclass(frozen=True)
class OffsetUnit:
    """One character unit, with the half-open range it occupies **in the string
    it was grouped from**.

    `split_units` throws offsets away because `decompose` only ever needed
    canonical ones. Chunking needs offsets into the *original* source, which may
    be non-canonically spelled, so the grouping is exposed here with its
    coordinates rather than re-derived somewhere else. See
    :func:`split_units_with_offsets`.
    """

    start: int
    end: int
    text: str
    """The unit exactly as it appears in the grouped string -- `text[start:end]`."""

    @property
    def base(self) -> str:
        """The base codepoint, or `""` for a stray leading combining mark."""
        return "" if not self.text or unicodedata.combining(self.text[0]) else self.text[0]

    @property
    def marks(self) -> tuple[str, ...]:
        return tuple(self.text[0 if not self.base else 1:])


def split_units_with_offsets(text: str) -> list[OffsetUnit]:
    """Group `text` into character units, **keeping each unit's offsets**.

    This is the single implementation of the grouping rule; `split_units` is a
    thin projection of it, so the two cannot disagree about where one letter
    ends and the next begins -- which is the whole reason `split_units` exists.

    `text` is grouped **as given**. Passing an NFD string reproduces
    `split_units` exactly; passing a raw source string yields units addressed in
    *source* coordinates, which is what a caller needs when it must not
    normalise its input.
    """
    units: list[OffsetUnit] = []
    start: int | None = None
    for index, ch in enumerate(text):
        if unicodedata.combining(ch):
            if start is None:
                # Combining mark with nothing to attach to: its own unit,
                # preserved verbatim rather than discarded.
                units.append(OffsetUnit(index, index + 1, ch))
                continue
        else:
            if start is not None:
                units.append(OffsetUnit(start, index, text[start:index]))
            start = index
    if start is not None:
        units.append(OffsetUnit(start, len(text), text[start:]))
    return units


def split_units(text_nfd: str) -> list[tuple[str, tuple[str, ...]]]:
    """Group an NFD string into `(base codepoint, combining marks)` pairs.

    A leading combining mark with no base -- malformed but possible input -- is
    kept as its own unit with an empty base, so nothing is ever discarded.
    """
    return [(unit.base, unit.marks) for unit in split_units_with_offsets(text_nfd)]


def join_units(units: list[tuple[str, tuple[str, ...]]]) -> str:
    """Inverse of :func:`split_units`, before normalisation.

    Callers normalise the result themselves; NFC re-applies canonical ordering,
    so marks need not be appended in canonical order.
    """
    return "".join(base + "".join(marks) for base, marks in units)
