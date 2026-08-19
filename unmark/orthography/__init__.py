"""Vietnamese orthographic utilities (pure Python, no ML dependencies).

Two layers live here, with different purposes:

* **Orthography core (B1A)** -- `marks`, `units`, `placement`, `canonical`,
  `decompose`, `models`. The deterministic `canon` / `decompose` / `recompose`
  that UNMARK's base, tone and letter-diacritic channels are built from. `canon`
  applies UNMARK's fixed nucleus-based tone-placement convention.
* **G-1 signatures** -- `signature`. Comparison forms used by the RESTORE
  smoke test. They collapse whitespace and are not reconstruction targets.

Both share one mark inventory, defined in `marks`.
"""

from unmark.orthography.canonical import (
    DEFAULT_TONE_PLACEMENT,
    TonePlacement,
    TonePlacementNotImplemented,
    TonePlacementUndecided,
    canon,
    canonical_differences,
    nfd,
)
from unmark.orthography.placement import apply_modern_placement, find_nucleus_index
from unmark.orthography.decompose import decompose, recompose, strip_to_base
from unmark.orthography.marks import (
    D_STROKE,
    LETTER_MARKS,
    TONE_MARKS,
    VIETNAMESE_MARKS,
    Anomaly,
    Eligibility,
    LetterDiacritic,
    ObservedTone,
    Tone,
    observed_from_lexical,
)
from unmark.orthography.models import CharacterUnit, DecomposedText, SyllableSpan
from unmark.orthography.signature import (
    TERMINAL_PUNCTUATION,
    base_signature,
    first_divergence,
    nfc,
    rewrite_signature,
    strip_vietnamese_diacritics,
    word_diff,
)

__all__ = [
    # marks and channel states
    "D_STROKE",
    "LETTER_MARKS",
    "TERMINAL_PUNCTUATION",
    "TONE_MARKS",
    "VIETNAMESE_MARKS",
    "Anomaly",
    "Eligibility",
    "LetterDiacritic",
    "ObservedTone",
    "Tone",
    "observed_from_lexical",
    # canonical form
    "DEFAULT_TONE_PLACEMENT",
    "TonePlacement",
    "TonePlacementNotImplemented",
    "TonePlacementUndecided",
    "apply_modern_placement",
    "canon",
    "canonical_differences",
    "nfc",
    "nfd",
    # decomposition
    "CharacterUnit",
    "DecomposedText",
    "SyllableSpan",
    "decompose",
    "find_nucleus_index",
    "recompose",
    "strip_to_base",
    # G-1 signatures
    "base_signature",
    "first_divergence",
    "rewrite_signature",
    "strip_vietnamese_diacritics",
    "word_diff",
]
