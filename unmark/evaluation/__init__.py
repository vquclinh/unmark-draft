"""Downstream (Stage-2) evaluation support.

**This module is torch-free and must stay that way** -- it is imported by the
ML-free local environment. Only the pure contracts and metrics are re-exported.

The pathway plumbing imports torch and is **deliberately not re-exported**::

    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

Scope: the minimum for the **Vanilla vs Base-only** fail-fast diagnostic that
§4.5 identifies as "exactly hypothesis H1, and exactly what G1 measures".
**No head trainer, no optimizer, no dataset, and no RESTORE or ALIGN.**
"""

from unmark.evaluation.contracts import (
    LOCKED_EVALUATION_VALUES,
    OPEN_EVALUATION_VALUES,
    SCIENTIFIC_REQUIRED_VALUES,
    STAGE1_POOLING_DOES_NOT_TRANSFER,
    EvaluationContractViolation,
    EvaluationPurpose,
    EvaluationRunConfig,
    HeadConfig,
    Split,
    SplitLeakage,
    SystemPathway,
    TaskExample,
    TaskSplit,
    UnresolvedEvaluationValue,
    assert_disjoint_splits,
    require_resolved,
)
from unmark.evaluation.metrics import (
    GRR_FORMULA,
    ClassScore,
    UndefinedGRR,
    accuracy,
    gap_recovery_rate,
    is_grr_defined,
    macro_f1,
    per_class_scores,
)

__all__ = [
    "ClassScore",
    "EvaluationContractViolation",
    "EvaluationPurpose",
    "EvaluationRunConfig",
    "GRR_FORMULA",
    "HeadConfig",
    "LOCKED_EVALUATION_VALUES",
    "OPEN_EVALUATION_VALUES",
    "SCIENTIFIC_REQUIRED_VALUES",
    "STAGE1_POOLING_DOES_NOT_TRANSFER",
    "Split",
    "SplitLeakage",
    "SystemPathway",
    "TaskExample",
    "TaskSplit",
    "UndefinedGRR",
    "UnresolvedEvaluationValue",
    "accuracy",
    "assert_disjoint_splits",
    "gap_recovery_rate",
    "is_grr_defined",
    "macro_f1",
    "per_class_scores",
    "require_resolved",
]
