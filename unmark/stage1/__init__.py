"""Stage-1 self-supervised representation alignment (proposal §4.6).

**This module is torch-free and must stay that way** -- it is imported by the
ML-free local environment. Only the pure-data contracts and the deterministic
data path are re-exported.

The objective imports torch and is **deliberately not re-exported**::

    from unmark.stage1.objective import Stage1Objective, representation_distance

`unmark.stage1.data` imports torch lazily, inside `collate_stage1_batch` only,
so preparation and padding are usable and testable without it.

**Nothing here trains.** There is no optimizer, no scheduler, no training loop
and no checkpointing; training additionally requires the repository-wide
PRE-TRAIN audit, which has not happened.
"""

from unmark.stage1.contracts import (
    LOCKED_STAGE1_VALUES,
    OPEN_STAGE1_VALUES,
    STAGE1_SCHEMA_VERSION,
    BaseInvarianceViolation,
    CorruptionRatePolicy,
    ObjectiveWeights,
    OverflowBehaviour,
    SCIENTIFIC_REQUIRED_VALUES,
    Stage1Branch,
    Stage1ContractViolation,
    Stage1Purpose,
    Stage1RunConfig,
    TruncationPolicy,
    UnresolvedStage1Value,
    require_resolved,
)
from unmark.stage1.data import (
    PreparedStage1Example,
    Stage1Example,
    padded_stage1_batch,
    prepare_example,
    project_text,
)

__all__ = [
    "BaseInvarianceViolation",
    "CorruptionRatePolicy",
    "LOCKED_STAGE1_VALUES",
    "OPEN_STAGE1_VALUES",
    "ObjectiveWeights",
    "OverflowBehaviour",
    "SCIENTIFIC_REQUIRED_VALUES",
    "Stage1Purpose",
    "Stage1RunConfig",
    "PreparedStage1Example",
    "STAGE1_SCHEMA_VERSION",
    "Stage1Branch",
    "Stage1ContractViolation",
    "Stage1Example",
    "TruncationPolicy",
    "UnresolvedStage1Value",
    "padded_stage1_batch",
    "prepare_example",
    "project_text",
    "require_resolved",
]
