"""Public ViUnMark method configuration.

This module holds reusable method facts only. UIT-VSFC class counts,
checkpoint digests and calibration values live in provenance records, not in
generic method defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class ViUnMarkContractError(ValueError):
    """Raised when a public ViUnMark contract is violated."""


SIX_CONDITIONS = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")
PHOBERT_CHECKPOINT = "vinai/phobert-base"
PHOBERT_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
UVW_2026_DATASET = "undertheseanlp/UVW-2026"
UVW_2026_REVISION = "a0a79294e4568137e25828bb3f2a4cde8546e1fb"
HIDDEN_SIZE = 768
TONE_TABLE_ROWS = 7
LETTER_TABLE_ROWS = 5
GATE_INIT_TARGET = 0.01
GATE_INIT_WEIGHT = 0.0
GATE_INIT_BIAS = math.log(GATE_INIT_TARGET / (1.0 - GATE_INIT_TARGET))
SCALE_EPSILON = 1e-8
ADAPTER_TRAINABLE_PARAMETERS = 3_551_232
STAGE1_OBJECTIVE_ID = "align-clean-pooled-v1"


class ReadoutKind(Enum):
    FIRST_TOKEN = "FIRST_TOKEN"
    MASKED_MEAN = "MASKED_MEAN"
    CONCAT = "CONCAT"

    def feature_dim(self, hidden_size: int) -> int:
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ViUnMarkContractError(f"hidden_size must be a positive int, got {hidden_size!r}")
        return 2 * hidden_size if self is ReadoutKind.CONCAT else hidden_size


CONCAT_ORDER = (ReadoutKind.FIRST_TOKEN, ReadoutKind.MASKED_MEAN)


class LossKind(Enum):
    UNWEIGHTED_CROSS_ENTROPY = "UNWEIGHTED_CROSS_ENTROPY"
    SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY = "SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY"


class TrainingDistribution(Enum):
    AUGMENTED_SIX_CONDITIONS = "AUGMENTED_SIX_CONDITIONS"

    @property
    def conditions(self) -> tuple[str, ...]:
        return SIX_CONDITIONS


class Pathway(Enum):
    VIUNMARK_GATE = "ViUnMark-Gate"
    VIUNMARK_SCALE = "ViUnMark-Scale"
    PHOBERT = "PhoBERT"


class LogitStream(Enum):
    GATE_ROBUST_READOUT = "gate_robust_readout"
    SCALE_UNWEIGHTED_READOUT = "scale_unweighted_readout"
    SCALE_WEIGHTED_READOUT = "scale_weighted_readout"
    PHOBERT_READOUT = "phobert_readout"
    GATE_MATCHED_RECIPE_READOUT = "gate_matched_recipe_readout"


FINAL_BRANCH_STREAMS = (
    LogitStream.GATE_ROBUST_READOUT,
    LogitStream.SCALE_UNWEIGHTED_READOUT,
    LogitStream.SCALE_WEIGHTED_READOUT,
    LogitStream.PHOBERT_READOUT,
)


@dataclass(frozen=True)
class _AdaptedPathwayConfig:
    FUSION_ID: ClassVar[str]
    PATHWAY: ClassVar[Pathway]
    FUSION_EQUATION: ClassVar[str]

    encoder_checkpoint: str = PHOBERT_CHECKPOINT
    encoder_revision: str = PHOBERT_REVISION
    hidden_size: int = HIDDEN_SIZE
    tone_rows: int = TONE_TABLE_ROWS
    letter_rows: int = LETTER_TABLE_ROWS
    gate_weight_initial: float = GATE_INIT_WEIGHT
    gate_initial_value: float = GATE_INIT_TARGET
    objective_id: str = STAGE1_OBJECTIVE_ID
    corpus_dataset: str = UVW_2026_DATASET
    corpus_revision: str = UVW_2026_REVISION
    precision: str = "fp32"
    encoder_frozen: bool = True

    def __post_init__(self) -> None:
        expected = {
            "encoder_checkpoint": PHOBERT_CHECKPOINT,
            "encoder_revision": PHOBERT_REVISION,
            "hidden_size": HIDDEN_SIZE,
            "tone_rows": TONE_TABLE_ROWS,
            "letter_rows": LETTER_TABLE_ROWS,
            "gate_weight_initial": GATE_INIT_WEIGHT,
            "gate_initial_value": GATE_INIT_TARGET,
            "objective_id": STAGE1_OBJECTIVE_ID,
            "corpus_dataset": UVW_2026_DATASET,
            "corpus_revision": UVW_2026_REVISION,
            "precision": "fp32",
            "encoder_frozen": True,
        }
        drift = {k: getattr(self, k) for k, v in expected.items() if getattr(self, k) != v}
        if drift:
            raise ViUnMarkContractError(f"{type(self).__name__} differs from the method: {drift}")

    @property
    def fusion_id(self) -> str:
        return self.FUSION_ID

    @property
    def pathway(self) -> Pathway:
        return self.PATHWAY

    @property
    def fusion_equation(self) -> str:
        return self.FUSION_EQUATION

    @property
    def adapter_trainable_parameters(self) -> int:
        return ADAPTER_TRAINABLE_PARAMETERS


@dataclass(frozen=True)
class ViUnMarkGateConfig(_AdaptedPathwayConfig):
    FUSION_ID: ClassVar[str] = "historical-fusion-v1"
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    FUSION_EQUATION: ClassVar[str] = "z = g * f + (1 - g) * e"


@dataclass(frozen=True)
class ViUnMarkScaleConfig(_AdaptedPathwayConfig):
    FUSION_ID: ClassVar[str] = "scale-calibrated-fusion-v1"
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    FUSION_EQUATION: ClassVar[str] = (
        "scale = ||e||_2 / max(||f||_2, 1e-8); z = g * (scale * f) + (1 - g) * e"
    )
    scale_epsilon: float = SCALE_EPSILON

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale_epsilon != SCALE_EPSILON:
            raise ViUnMarkContractError(f"scale_epsilon must be {SCALE_EPSILON}")


ROBUST_MLP_HIDDEN_DIM = 256
ROBUST_MLP_DROPOUT = 0.1
ROBUST_MLP_LAYERS = ("LayerNorm", "Linear", "GELU", "Dropout", "Linear")


def _positive_int(value: object, what: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ViUnMarkContractError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class RobustMLPHeadConfig:
    input_dim: int
    num_labels: int

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, "input_dim")
        _positive_int(self.num_labels, "num_labels", 2)

    @property
    def hidden_dim(self) -> int:
        return ROBUST_MLP_HIDDEN_DIM

    @property
    def dropout(self) -> float:
        return ROBUST_MLP_DROPOUT

    @property
    def parameter_count(self) -> int:
        d, h, n = self.input_dim, ROBUST_MLP_HIDDEN_DIM, self.num_labels
        return 2 * d + (h * d + h) + (h * n + n)


@dataclass(frozen=True)
class BranchReadoutConfig:
    STREAM: ClassVar[LogitStream]
    PATHWAY: ClassVar[Pathway]
    READOUT: ClassVar[ReadoutKind]
    LOSS: ClassVar[LossKind]
    TRAINING_DISTRIBUTION: ClassVar[TrainingDistribution] = TrainingDistribution.AUGMENTED_SIX_CONDITIONS

    num_labels: int
    hidden_size: int = HIDDEN_SIZE

    def __post_init__(self) -> None:
        if type(self) is BranchReadoutConfig:
            raise ViUnMarkContractError("BranchReadoutConfig is abstract")
        _positive_int(self.num_labels, "num_labels", 2)
        if self.hidden_size != HIDDEN_SIZE:
            raise ViUnMarkContractError(f"hidden_size must be {HIDDEN_SIZE}")

    @property
    def stream(self) -> LogitStream:
        return self.STREAM

    @property
    def pathway(self) -> Pathway:
        return self.PATHWAY

    @property
    def readout(self) -> ReadoutKind:
        return self.READOUT

    @property
    def loss(self) -> LossKind:
        return self.LOSS

    @property
    def training_distribution(self) -> TrainingDistribution:
        return self.TRAINING_DISTRIBUTION

    @property
    def input_dim(self) -> int:
        return self.READOUT.feature_dim(self.hidden_size)

    @property
    def head(self) -> RobustMLPHeadConfig:
        return RobustMLPHeadConfig(self.input_dim, self.num_labels)


@dataclass(frozen=True)
class GateReadoutConfig(BranchReadoutConfig):
    STREAM: ClassVar[LogitStream] = LogitStream.GATE_ROBUST_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.MASKED_MEAN
    LOSS: ClassVar[LossKind] = LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY


@dataclass(frozen=True)
class ScaleUnweightedReadoutConfig(BranchReadoutConfig):
    STREAM: ClassVar[LogitStream] = LogitStream.SCALE_UNWEIGHTED_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


@dataclass(frozen=True)
class ScaleWeightedReadoutConfig(BranchReadoutConfig):
    STREAM: ClassVar[LogitStream] = LogitStream.SCALE_WEIGHTED_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY


@dataclass(frozen=True)
class PhoBERTReadoutConfig(BranchReadoutConfig):
    STREAM: ClassVar[LogitStream] = LogitStream.PHOBERT_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.PHOBERT
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


@dataclass(frozen=True)
class GateMatchedRecipeReadoutConfig(BranchReadoutConfig):
    STREAM: ClassVar[LogitStream] = LogitStream.GATE_MATCHED_RECIPE_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


READOUT_CONFIG_BY_STREAM = {
    c.STREAM: c
    for c in (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
        PhoBERTReadoutConfig,
        GateMatchedRecipeReadoutConfig,
    )
}


@dataclass(frozen=True)
class CalibrationConfig:
    class_index: int | None
    additive_logit_bias: float = 0.0
    label_name: str | None = None

    def __post_init__(self) -> None:
        if self.class_index is None:
            if self.additive_logit_bias != 0.0 or self.label_name is not None:
                raise ViUnMarkContractError("identity calibration carries no bias or label")
            return
        _positive_int(self.class_index, "class_index", 0)
        if isinstance(self.additive_logit_bias, bool) or not isinstance(
            self.additive_logit_bias, (int, float)
        ) or not math.isfinite(self.additive_logit_bias):
            raise ViUnMarkContractError("additive_logit_bias must be finite")

    @classmethod
    def identity(cls) -> "CalibrationConfig":
        return cls(None)

    @property
    def is_identity(self) -> bool:
        return self.class_index is None or self.additive_logit_bias == 0.0

    def bias_vector(self, num_labels: int) -> tuple[float, ...]:
        _positive_int(num_labels, "num_labels", 2)
        vector = [0.0] * num_labels
        if self.class_index is not None:
            if self.class_index >= num_labels:
                raise ViUnMarkContractError("class_index outside output width")
            vector[self.class_index] = float(self.additive_logit_bias)
        return tuple(vector)


def _head_seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ViUnMarkContractError("head_seeds must be a non-empty tuple")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise ViUnMarkContractError("head_seeds must be ints")
    if len(set(value)) != len(value):
        raise ViUnMarkContractError("head_seeds contain duplicates")
    return value


@dataclass(frozen=True)
class _FusionSystemConfig:
    num_labels: int
    head_seeds: tuple[int, ...]
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig.identity)
    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]]

    def __post_init__(self) -> None:
        _positive_int(self.num_labels, "num_labels", 2)
        _head_seeds(self.head_seeds)
        if not isinstance(self.calibration, CalibrationConfig):
            raise ViUnMarkContractError("calibration must be a CalibrationConfig")
        self.calibration.bias_vector(self.num_labels)

    @property
    def heads_per_branch(self) -> int:
        return len(self.head_seeds)

    @property
    def head_count(self) -> int:
        return len(self.BRANCHES) * len(self.head_seeds)

    @property
    def branches(self) -> tuple[BranchReadoutConfig, ...]:
        return tuple(branch(self.num_labels) for branch in self.BRANCHES)

    @property
    def streams(self) -> tuple[LogitStream, ...]:
        return tuple(branch.STREAM for branch in self.BRANCHES)

    @property
    def stage2_parameter_count(self) -> int:
        return sum(branch.head.parameter_count for branch in self.branches) * len(self.head_seeds)


@dataclass(frozen=True)
class AdaptedOnlyFusionConfig(_FusionSystemConfig):
    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]] = (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
    )
    IS_DIAGNOSTIC: ClassVar[bool] = True


@dataclass(frozen=True)
class ViUnMarkConfig(_FusionSystemConfig):
    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]] = (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
        PhoBERTReadoutConfig,
    )
    IS_DIAGNOSTIC: ClassVar[bool] = False
