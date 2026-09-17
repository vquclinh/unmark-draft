"""Corruption-condition vocabulary used by ViUnMark reproduction interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CorruptionScope(Enum):
    NONE = "NONE"
    TONE = "TONE"
    TONE_AND_LETTER = "TONE_AND_LETTER"


@dataclass(frozen=True)
class CorruptionCondition:
    name: str
    scope: CorruptionScope
    probability: float
    description: str

    @property
    def is_deterministic(self) -> bool:
        return self.probability in (0.0, 1.0)


FULL = CorruptionCondition("FULL", CorruptionScope.NONE, 0.0, "Fully diacritized")
P25 = CorruptionCondition("P25", CorruptionScope.TONE, 0.25, "Tone marks removed from 25% of syllables")
P50 = CorruptionCondition("P50", CorruptionScope.TONE, 0.50, "Tone marks removed from 50% of syllables")
P75 = CorruptionCondition("P75", CorruptionScope.TONE, 0.75, "Tone marks removed from 75% of syllables")
P100 = CorruptionCondition("P100", CorruptionScope.TONE, 1.0, "All tone marks removed")
STRIP_ALL = CorruptionCondition("STRIP_ALL", CorruptionScope.TONE_AND_LETTER, 1.0, "Tone and letter diacritics removed")
CONDITIONS = {c.name: c for c in (FULL, P25, P50, P75, P100, STRIP_ALL)}


def get_condition(name: str | CorruptionCondition) -> CorruptionCondition:
    if isinstance(name, CorruptionCondition):
        return name
    return CONDITIONS[str(name).upper().replace("-", "_")]
