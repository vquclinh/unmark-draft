"""Corruption conditions, taken verbatim from proposal v1.3 section 6.3.

| Condition   | Proposal wording                                          |
|-------------|-----------------------------------------------------------|
| `FULL`      | Fully diacritized (upper bound)                           |
| `P25/50/75` | Tone marks removed from 25/50/75% of syllables            |
| `P100`      | All tone marks removed                                    |
| `STRIP_ALL` | Tone **and** letter diacritics removed (real typing)      |
| `VARIANT`   | Tone-placement variants (hoà/hòa) and NFC/NFD forms       |

Two things the table settles, and which this module encodes rather than guesses:

* the sampling unit is the **syllable**, not the character or the word;
* `P100` and `STRIP_ALL` are **not** the same. `P100` removes every *tone mark*
  and leaves `ă â ê ô ơ ư đ` intact; `STRIP_ALL` additionally removes the
  letter-forming diacritics. A `P100` sentence is still recognisably Vietnamese
  letters without tones; a `STRIP_ALL` sentence is what someone types on a
  keyboard with no IME.

`VARIANT` is recognised but not implemented in B2; see `docs/spec/decisions.md`
(D-B2-005). It needs `TonePlacement.TRADITIONAL`, which B1A deliberately did not
implement, so producing it would mean adopting a second spelling convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CorruptionScope(Enum):
    """Which orthographic information a condition removes."""

    NONE = "NONE"
    """Nothing is removed; the canonical clean text is returned."""

    TONE = "TONE"
    """Tone marks only. Letter diacritics (ă â ê ô ơ ư đ) survive."""

    TONE_AND_LETTER = "TONE_AND_LETTER"
    """Tone marks and Vietnamese letter-forming diacritics."""


@dataclass(frozen=True)
class CorruptionCondition:
    """One named condition: what it removes, and from what fraction of syllables."""

    name: str
    scope: CorruptionScope
    probability: float
    description: str

    @property
    def is_deterministic(self) -> bool:
        """True when no per-unit sampling is involved (p is 0 or 1)."""
        return self.probability in (0.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"{self.name}: probability must be in [0, 1], got {self.probability}")
        if self.scope is CorruptionScope.NONE and self.probability != 0.0:
            raise ValueError(f"{self.name}: scope NONE requires probability 0.0")


FULL = CorruptionCondition("FULL", CorruptionScope.NONE, 0.0, "Fully diacritized (upper bound)")
P25 = CorruptionCondition("P25", CorruptionScope.TONE, 0.25, "Tone marks removed from 25% of syllables")
P50 = CorruptionCondition("P50", CorruptionScope.TONE, 0.50, "Tone marks removed from 50% of syllables")
P75 = CorruptionCondition("P75", CorruptionScope.TONE, 0.75, "Tone marks removed from 75% of syllables")
P100 = CorruptionCondition("P100", CorruptionScope.TONE, 1.0, "All tone marks removed")
STRIP_ALL = CorruptionCondition(
    "STRIP_ALL", CorruptionScope.TONE_AND_LETTER, 1.0, "Tone and letter diacritics removed (real typing behaviour)"
)

CONDITIONS: dict[str, CorruptionCondition] = {
    c.name: c for c in (FULL, P25, P50, P75, P100, STRIP_ALL)
}

# Recognised in proposal 6.3 but not implemented in B2 (D-B2-005).
UNIMPLEMENTED_CONDITIONS = {
    "VARIANT": (
        "Requires TonePlacement.TRADITIONAL to emit the non-canonical placement, "
        "which B1A deliberately did not implement (docs/spec/orthography.md D-001). "
        "Implementing half of it (NFD forms only) would misrepresent the condition."
    )
}


class UnknownCondition(KeyError):
    """Raised for a condition name that is not implemented."""


def get_condition(name: str | CorruptionCondition) -> CorruptionCondition:
    """Look up a condition by name, case-insensitively; `P50` and `p50` both work."""
    if isinstance(name, CorruptionCondition):
        return name
    key = str(name).upper().replace("-", "_")
    if key in CONDITIONS:
        return CONDITIONS[key]
    if key in UNIMPLEMENTED_CONDITIONS:
        raise UnknownCondition(f"condition {key!r} is not implemented in B2: {UNIMPLEMENTED_CONDITIONS[key]}")
    raise UnknownCondition(f"unknown condition {key!r}; known: {sorted(CONDITIONS)}")
