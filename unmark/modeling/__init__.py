"""Contracts and neural implementation for the UNMARK adapter.

**This module is torch-free and must stay that way.** It is imported by the
deterministic pipeline and by the ML-free local environment, so it re-exports
only the pure-data contracts (B4A).

The B4B neural modules are **deliberately not re-exported**, because importing
them pulls in torch. Import them explicitly, from an environment that has it::

    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.collate import build_example, collate_examples
    from unmark.modeling.pooling import masked_mean_non_special

`unmark.modeling.collate` is a special case: only its final tensor-packing step
imports torch, and it does so lazily, so `build_example` and `padded_batch` are
usable and testable without it.
"""

from unmark.modeling.config import (
    FUSION_INPUT_MULTIPLIER,
    FUSION_KIND,
    LAYERNORM_POSITION,
    PARAMETER_FORMULA,
    POSITION_EMBEDDINGS_SOURCE,
    AdapterConfig,
    ParameterCount,
    fusion_equation,
    h4_equalized,
)
from unmark.modeling.contracts import (
    APPLICABLE_LETTER_LABELS,
    FUSION_IS_CONVEX,
    GATE_INIT_BIAS,
    GATE_INIT_TARGET,
    GATE_INIT_WEIGHT,
    GATE_IS_PROJECTION,
    GATE_TRANSFORM,
    GATE_ZERO_IS_ATTAINABLE,
    GATE_ZERO_IS_WIRING_TEST_ONLY,
    GATE_ZERO_RECOVERS,
    LETTER_EMPTY_IS_ZERO_VECTOR,
    LETTER_LABEL_IDS,
    LETTER_NA_SENTINEL,
    LETTER_TABLE_ROWS,
    LOCKED_GATE_INIT,
    LOCKED_LETTER_EMPTY_TREATMENT,
    LOCKED_TONE_NA_TREATMENT,
    MARKED_TONE_LABELS,
    OBSERVABLE_TONE_IDS,
    STAGE1_POOLING,
    STAGE1_ZERO_CONTENT_POLICY,
    TONE_NA_IS_ZERO_VECTOR,
    TONE_NA_SENTINEL,
    TONE_SLOT_A_ID,
    TONE_SLOT_B_ID,
    TONE_TABLE_ROWS,
    GateContract,
    GateInit,
    LetterChannelContract,
    LetterEmptyTreatment,
    LockedContractViolation,
    Stage1PoolingContract,
    Stage1PoolingError,
    ToneChannelContract,
    ToneNaTreatment,
    TonePolicy,
    UnresolvedAdapterContract,
    logit,
    sigmoid,
)

__all__ = [
    "APPLICABLE_LETTER_LABELS",
    "AdapterConfig",
    "FUSION_INPUT_MULTIPLIER",
    "FUSION_IS_CONVEX",
    "FUSION_KIND",
    "GATE_INIT_BIAS",
    "GATE_INIT_TARGET",
    "GATE_INIT_WEIGHT",
    "GATE_IS_PROJECTION",
    "GATE_TRANSFORM",
    "GATE_ZERO_IS_ATTAINABLE",
    "GATE_ZERO_IS_WIRING_TEST_ONLY",
    "GATE_ZERO_RECOVERS",
    "GateContract",
    "GateInit",
    "LAYERNORM_POSITION",
    "LETTER_EMPTY_IS_ZERO_VECTOR",
    "LETTER_LABEL_IDS",
    "LETTER_NA_SENTINEL",
    "LETTER_TABLE_ROWS",
    "LOCKED_GATE_INIT",
    "LOCKED_LETTER_EMPTY_TREATMENT",
    "LOCKED_TONE_NA_TREATMENT",
    "LetterChannelContract",
    "LetterEmptyTreatment",
    "LockedContractViolation",
    "MARKED_TONE_LABELS",
    "OBSERVABLE_TONE_IDS",
    "PARAMETER_FORMULA",
    "POSITION_EMBEDDINGS_SOURCE",
    "ParameterCount",
    "STAGE1_POOLING",
    "STAGE1_ZERO_CONTENT_POLICY",
    "Stage1PoolingContract",
    "Stage1PoolingError",
    "TONE_NA_IS_ZERO_VECTOR",
    "TONE_NA_SENTINEL",
    "TONE_SLOT_A_ID",
    "TONE_SLOT_B_ID",
    "TONE_TABLE_ROWS",
    "ToneChannelContract",
    "ToneNaTreatment",
    "TonePolicy",
    "UnresolvedAdapterContract",
    "fusion_equation",
    "h4_equalized",
    "logit",
    "sigmoid",
]
