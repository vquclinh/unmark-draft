"""Canonical form for Vietnamese text.

The G0 invariant (proposal v1.3 section 4.2) is::

    rec(dec(x)) = canon(x)

"where `canon` applies Unicode NFC and a fixed tone-placement rule".

Both halves are now fixed. The pipeline is::

    input  ->  Unicode normalisation  ->  fixed tone placement  ->  NFC canonical text

**UNMARK's fixed canonical tone-placement convention is nucleus-based**
(`TonePlacement.MODERN`): the tone mark sits on the vowel nucleus, so `hòa`
canonicalises to `hoà`, `thúy` to `thuý`, `khỏe` to `khoẻ`. The rule itself is
in `placement.py`.

This is a **project canonicalisation convention adopted for reproducibility**.
Competing tone-placement conventions exist in real Vietnamese text and are not
errors; nothing here claims this convention is the sole official or universally
preferred Vietnamese orthography. It is fixed so that `canon` is deterministic,
so placement variants collapse to one form, and so experiments are comparable.
The decision is recorded in `docs/spec/orthography.md`.

`TonePlacement.PRESERVE` remains available as an **explicit diagnostic mode**
that leaves every tone mark where it was found. It is no longer the default and
is not the project pathway: use it to inspect what an input actually contained,
for example when reporting how far a corpus is from canonical form.

`TonePlacement.TRADITIONAL` is not implemented -- the project does not need it,
and implementing an unused second convention would add a spelling standard
nobody asked for. Requesting it raises.

`canon` deliberately does **not** normalise whitespace, case or punctuation.
That distinguishes it from `signature.base_signature`, which collapses
whitespace because it is a comparison form for the G-1 diagnostic rather than a
reconstruction target.
"""

from __future__ import annotations

import unicodedata
from enum import Enum

from unmark.orthography.placement import apply_modern_placement


class TonePlacement(Enum):
    """Tone-mark placement conventions."""

    MODERN = "MODERN"
    """Nucleus-based (`hoà`, `thuý`, `khoẻ`). UNMARK's canonical convention."""

    PRESERVE = "PRESERVE"
    """Leave each tone mark where it was found. Diagnostic mode; NFC only."""

    TRADITIONAL = "TRADITIONAL"
    """Non-nucleus placement (`hòa`, `thúy`, `khỏe`). Not implemented."""


DEFAULT_TONE_PLACEMENT = TonePlacement.MODERN


class TonePlacementNotImplemented(NotImplementedError):
    """Raised for a placement convention this project does not implement."""


# Retained under the former name so existing callers keep working.
TonePlacementUndecided = TonePlacementNotImplemented


_NOT_IMPLEMENTED_MESSAGE = """\
Tone-placement convention {name} is not implemented.

UNMARK's canonical convention is TonePlacement.MODERN (nucleus-based): the tone
mark sits on the vowel nucleus, so hoa+huyen canonicalises to `hoà`, not `hòa`.
It is the default for canon() and is recorded in docs/spec/orthography.md.

TonePlacement.PRESERVE is also available, as an explicit diagnostic mode that
leaves tone marks exactly where the input put them.

TRADITIONAL is deliberately absent: the project has no use for a second
canonical convention, and adding one would introduce a spelling standard no
part of the design needs. If a future comparison requires it, implement it in
placement.py and record the decision alongside the MODERN one.
"""


def canon(text: str, placement: TonePlacement = DEFAULT_TONE_PLACEMENT) -> str:
    """Return the canonical form of `text`.

    By default this applies UNMARK's fixed nucleus-based tone placement and
    returns NFC, so `canon("hòa") == canon("hoà") == "hoà"`. Whitespace, case,
    punctuation, digits and non-Vietnamese text are untouched; only the position
    of a tone mark within its own syllable may change.

    With `placement=TonePlacement.PRESERVE` the tone-placement step is skipped
    and the result is Unicode NFC alone -- a diagnostic view of what the input
    actually contained.

    `canon` is idempotent under both.
    """
    if placement is TonePlacement.PRESERVE:
        return unicodedata.normalize("NFC", text)
    if placement is TonePlacement.MODERN:
        return apply_modern_placement(text)
    raise TonePlacementNotImplemented(_NOT_IMPLEMENTED_MESSAGE.format(name=placement.name))


def nfc(text: str) -> str:
    """Unicode NFC. Named separately from `canon` so the two stay distinguishable
    once a placement rule is adopted and `canon` becomes more than NFC."""
    return unicodedata.normalize("NFC", text)


def nfd(text: str) -> str:
    """Unicode NFD, used internally to separate combining marks."""
    return unicodedata.normalize("NFD", text)


def canonical_differences(text: str, placement: TonePlacement = DEFAULT_TONE_PLACEMENT) -> dict[str, object]:
    """Describe how `text` differs from its canonical form.

    The proposal requires that "every difference between x and canon(x) is
    enumerable and logged. No silent loss is tolerated." This returns that
    enumeration for one string; `scripts/g0_orthography_check.py` aggregates it.
    """
    canonical = canon(text, placement)
    nfc_only = nfc(text)
    return {
        "already_canonical": text == canonical,
        "nfc_changed": text != nfc_only,
        # True when the tone-placement step alone changed the string, i.e. the
        # input used a different placement convention. Reported separately so a
        # G0 run can distinguish Unicode normalisation from variant collapsing.
        "tone_placement_changed": nfc_only != canonical,
        "input_len": len(text),
        "canonical_len": len(canonical),
        "input_form": _unicode_form(text),
        "placement": placement.name,
    }


def _unicode_form(text: str) -> str:
    """Best-effort label for which normalisation form `text` is already in."""
    is_nfc = text == unicodedata.normalize("NFC", text)
    is_nfd = text == unicodedata.normalize("NFD", text)
    if is_nfc and is_nfd:
        return "NFC=NFD"  # no composable/decomposable characters present
    if is_nfc:
        return "NFC"
    if is_nfd:
        return "NFD"
    return "MIXED"
