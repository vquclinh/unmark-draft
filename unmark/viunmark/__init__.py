"""ViUnMark: robust Vietnamese classification under missing diacritics.

**Torch-free at import.** Functions that need torch import it when called.

The proposed system fuses four branches. Each branch is the mean of its heads'
logits:

    ViUnMark-Gate  -> Gate Robust Readout            (MASKED_MEAN, weighted CE)
    ViUnMark-Scale -> Scale Unweighted Readout        (CONCAT,      unweighted CE)
    ViUnMark-Scale -> Scale Weighted Readout          (CONCAT,      weighted CE)
    PhoBERT        -> PhoBERT Readout                 (CONCAT,      unweighted CE)

    scale    = 0.5 * scale_unweighted + 0.5 * scale_weighted
    adapted  = 0.5 * gate             + 0.5 * scale
    viunmark = 0.5 * adapted          + 0.5 * phobert       -> one calibration

Modules:

    config       method configuration (no dataset-specific values)
    readout      FIRST_TOKEN / MASKED_MEAN / CONCAT
    heads        robust MLP head
    losses       class-weight rule
    training     recovered head-training recipe (initialisation, optimizer, budget,
                 six-condition augmentation, batch and dropout seeds, selection)
    inputs       corruption protocol (seed required) and cross-pathway input identity
    fusion       within-branch ensembling, raw fusion, single calibration
    system       ViUnMarkGate, ViUnMarkScale, ViUnMark, AdaptedOnlyFusion
    provenance   recorded specs and the UIT-VSFC reproduction record
    diagnostics  analyses under descriptive names

Calibration values are dataset-specific. A default `ViUnMarkConfig` carries no
calibration; reproduction values come only from `provenance`.
"""

from unmark.viunmark.config import (
    CONCAT_ORDER,
    FINAL_BRANCH_STREAMS,
    SIX_CONDITIONS,
    AdaptedOnlyFusionConfig,
    BranchReadoutConfig,
    CalibrationConfig,
    GateReadoutConfig,
    LogitStream,
    LossKind,
    Pathway,
    PhoBERTReadoutConfig,
    ReadoutKind,
    RobustMLPHeadConfig,
    ScaleUnweightedReadoutConfig,
    ScaleWeightedReadoutConfig,
    TrainingDistribution,
    ViUnMarkConfig,
    ViUnMarkContractError,
    ViUnMarkGateConfig,
    ViUnMarkScaleConfig,
)
from unmark.viunmark.fusion import (
    BranchLogits,
    CalibratedLogits,
    FusedLogits,
    HeadLogits,
    adapted_only_fusion,
    apply_system_calibration,
    ensemble_branch,
    expanded_adapted_only_weights,
    expanded_per_head_weights,
    expanded_viunmark_weights,
    fuse_adapted_only_raw,
    fuse_scale,
    fuse_viunmark_raw,
    viunmark_fusion,
)
from unmark.viunmark.heads import (
    build_robust_mlp_head,
    build_robust_mlp_optimizer,
    initialize_robust_mlp_head,
    robust_mlp_layers,
)
from unmark.viunmark.inputs import (
    FULL_CONDITION_API_SEED,
    CorruptionProtocol,
    CorruptionRealizationKey,
    require_cross_pathway_input_identity,
)
from unmark.viunmark.losses import class_weights_for, sqrt_inverse_frequency_weights
from unmark.viunmark.readout import (
    concat_first_token_masked_mean,
    first_token,
    first_token_and_masked_mean,
    masked_mean,
    readout,
)
from unmark.viunmark.system import AdaptedOnlyFusion, ViUnMark, ViUnMarkGate, ViUnMarkScale
from unmark.viunmark.training import (
    ROBUST_MLP_OPTIMIZER_POLICY,
    ROBUST_MLP_TRAINING_BUDGET,
    BoundaryScore,
    HeadTrainingBudget,
    RobustMLPOptimizerPolicy,
    augmented_labels,
    augmented_row_order,
    cycle_seed,
    dropout_seed,
    select_checkpoint_boundary,
)

__all__ = [
    "AdaptedOnlyFusion",
    "BoundaryScore",
    "CorruptionProtocol",
    "CorruptionRealizationKey",
    "FULL_CONDITION_API_SEED",
    "HeadTrainingBudget",
    "ROBUST_MLP_OPTIMIZER_POLICY",
    "ROBUST_MLP_TRAINING_BUDGET",
    "RobustMLPOptimizerPolicy",
    "augmented_labels",
    "augmented_row_order",
    "build_robust_mlp_optimizer",
    "cycle_seed",
    "dropout_seed",
    "initialize_robust_mlp_head",
    "require_cross_pathway_input_identity",
    "select_checkpoint_boundary",
    "AdaptedOnlyFusionConfig",
    "BranchLogits",
    "BranchReadoutConfig",
    "CONCAT_ORDER",
    "CalibratedLogits",
    "CalibrationConfig",
    "FINAL_BRANCH_STREAMS",
    "FusedLogits",
    "GateReadoutConfig",
    "HeadLogits",
    "LogitStream",
    "LossKind",
    "Pathway",
    "PhoBERTReadoutConfig",
    "ReadoutKind",
    "RobustMLPHeadConfig",
    "SIX_CONDITIONS",
    "ScaleUnweightedReadoutConfig",
    "ScaleWeightedReadoutConfig",
    "TrainingDistribution",
    "ViUnMark",
    "ViUnMarkConfig",
    "ViUnMarkContractError",
    "ViUnMarkGate",
    "ViUnMarkGateConfig",
    "ViUnMarkScale",
    "ViUnMarkScaleConfig",
    "adapted_only_fusion",
    "apply_system_calibration",
    "build_robust_mlp_head",
    "class_weights_for",
    "concat_first_token_masked_mean",
    "ensemble_branch",
    "expanded_adapted_only_weights",
    "expanded_per_head_weights",
    "expanded_viunmark_weights",
    "first_token",
    "first_token_and_masked_mean",
    "fuse_adapted_only_raw",
    "fuse_scale",
    "fuse_viunmark_raw",
    "masked_mean",
    "readout",
    "robust_mlp_layers",
    "sqrt_inverse_frequency_weights",
    "viunmark_fusion",
]
