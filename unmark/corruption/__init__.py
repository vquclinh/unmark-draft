"""Deterministic orthographic corruption (B2).

Generates the clean/corrupted pairs used by UNMARK stage-1 training and by every
evaluation corruption condition. Pure standard library plus the B1A orthography
core: no tokenizer, no model, no word list, no network.

**The deterministic engine is final. The eligibility policy is not.** Corruption
probabilities are meant to apply to eligible Vietnamese syllables (proposal 4.3,
6.3), but the syllable inventory that rule needs does not exist yet (GAP-2). B2
therefore scores *candidate spans* -- every maximal alphabetic run -- and
`corrupt()` refuses by default so that provisional counts cannot become the
scientific protocol by accident. See `unmark/corruption/eligibility.py`.
"""

from unmark.corruption.conditions import (
    CONDITIONS,
    FULL,
    P25,
    P50,
    P75,
    P100,
    STRIP_ALL,
    UNIMPLEMENTED_CONDITIONS,
    CorruptionCondition,
    CorruptionScope,
    UnknownCondition,
    get_condition,
)
from unmark.corruption.corrupt import corrupt, corrupt_batch
from unmark.corruption.eligibility import (
    active_eligibility_policy,
    CorruptionPurpose,
    EligibilityPolicy,
    EligibilityUnresolved,
    is_resolved,
    require_resolved_eligibility,
)
from unmark.corruption.deterministic import (
    CORRUPTION_SCHEMA_VERSION,
    is_selected,
    text_identity,
    unit_score,
)
from unmark.corruption.models import CorruptionResult, UnitDecision

__all__ = [
    "active_eligibility_policy",
    "CONDITIONS",
    "CORRUPTION_SCHEMA_VERSION",
    "FULL",
    "P25",
    "P50",
    "P75",
    "P100",
    "STRIP_ALL",
    "UNIMPLEMENTED_CONDITIONS",
    "CorruptionCondition",
    "CorruptionPurpose",
    "CorruptionResult",
    "CorruptionScope",
    "EligibilityPolicy",
    "EligibilityUnresolved",
    "UnitDecision",
    "UnknownCondition",
    "corrupt",
    "corrupt_batch",
    "get_condition",
    "is_resolved",
    "is_selected",
    "require_resolved_eligibility",
    "text_identity",
    "unit_score",
]
