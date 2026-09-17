"""ViUnMark method configuration. **Torch-free.**

This module holds the *reusable method*: the two adapted pathways, the readout
and loss vocabulary, the robust MLP head shape, the four branch recipes, and the
system configurations. It holds **no dataset-specific value**. Class counts,
selected heads, and calibration numbers for a particular dataset belong to that
dataset's reproduction record (see `unmark.viunmark.provenance`), never to a
default here.

Everything in this module is a frozen dataclass or an enum, so a configuration
cannot be mutated after it has been validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from unmark.corruption import CONDITIONS
from unmark.modeling.contracts import (
    FUSION_SCALE_EPSILON,
    GATE_INIT_TARGET,
    GATE_INIT_WEIGHT,
    HISTORICAL_FUSION_ID,
    LETTER_TABLE_ROWS,
    SCALE_CALIBRATED_FUSION_ID,
    TONE_TABLE_ROWS,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORPUS_DATASET,
    CORPUS_REVISION,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    HIDDEN_SIZE,
    HISTORICAL_OBJECTIVE_ID,
)


class ViUnMarkContractError(ValueError):
    """Raised when a ViUnMark configuration or input violates the method contract."""


# ---------------------------------------------------------------------------
# Corruption conditions
# ---------------------------------------------------------------------------
SIX_CONDITIONS: tuple[str, ...] = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")
"""The six authoritative corruption conditions, in their canonical order.

Taken from `unmark.corruption` and checked against it at import time, so the
method cannot silently drift from the corruption implementation it relies on.
"""

if tuple(CONDITIONS) != SIX_CONDITIONS:  # pragma: no cover - import guard
    raise AssertionError(
        f"unmark.corruption defines {tuple(CONDITIONS)}, but ViUnMark is built on "
        f"{SIX_CONDITIONS}. The corruption framework is not changed by ViUnMark."
    )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class ReadoutKind(Enum):
    """How a sequence of hidden states becomes one feature vector."""

    FIRST_TOKEN = "FIRST_TOKEN"
    """`hidden[:, 0, :]` -- the `<s>` position."""

    MASKED_MEAN = "MASKED_MEAN"
    """Mean over positions that are neither padding nor tokenizer special tokens."""

    CONCAT = "CONCAT"
    """`concat([FIRST_TOKEN, MASKED_MEAN], dim=-1)`, from one encoder forward."""

    def feature_dim(self, hidden_size: int) -> int:
        """Width of the feature vector for an encoder of `hidden_size`."""
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ViUnMarkContractError(f"hidden_size must be a positive int, got {hidden_size!r}")
        return 2 * hidden_size if self is ReadoutKind.CONCAT else hidden_size


CONCAT_ORDER: tuple[ReadoutKind, ReadoutKind] = (ReadoutKind.FIRST_TOKEN, ReadoutKind.MASKED_MEAN)
"""The order of the two halves of a `CONCAT` feature. **Exactly `[FT ; MM]`.**"""


class LossKind(Enum):
    """The two cross-entropy losses the branch heads are trained with."""

    UNWEIGHTED_CROSS_ENTROPY = "UNWEIGHTED_CROSS_ENTROPY"
    SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY = "SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY"
    """Class weights `1/sqrt(count)`, normalised to mean 1. See `losses`."""


class TrainingDistribution(Enum):
    """Which inputs a branch head is trained on."""

    AUGMENTED_SIX_CONDITIONS = "AUGMENTED_SIX_CONDITIONS"
    """Training examples under all six corruption conditions."""

    @property
    def conditions(self) -> tuple[str, ...]:
        return SIX_CONDITIONS


class Pathway(Enum):
    """The three encoder pathways a branch can read from."""

    VIUNMARK_GATE = "ViUnMark-Gate"
    VIUNMARK_SCALE = "ViUnMark-Scale"
    PHOBERT = "PhoBERT"


class LogitStream(Enum):
    """One trained readout recipe, identified by what it is rather than by a run label."""

    GATE_ROBUST_READOUT = "gate_robust_readout"
    SCALE_UNWEIGHTED_READOUT = "scale_unweighted_readout"
    SCALE_WEIGHTED_READOUT = "scale_weighted_readout"
    PHOBERT_READOUT = "phobert_readout"
    GATE_MATCHED_RECIPE_READOUT = "gate_matched_recipe_readout"
    """Diagnostic only: the Gate pathway under the matched downstream recipe."""


FINAL_BRANCH_STREAMS: tuple[LogitStream, ...] = (
    LogitStream.GATE_ROBUST_READOUT,
    LogitStream.SCALE_UNWEIGHTED_READOUT,
    LogitStream.SCALE_WEIGHTED_READOUT,
    LogitStream.PHOBERT_READOUT,
)
"""The four branches of the final ViUnMark system."""


# ---------------------------------------------------------------------------
# Stage-I pathway identities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _AdaptedPathwayConfig:
    """Shared, frozen identity of a Stage-I adapted pathway.

    Every field has the value the method fixes. Construction validates each one,
    so an object claiming to be ViUnMark-Gate or ViUnMark-Scale cannot describe a
    different encoder, a different table size, or a different objective.
    """

    FUSION_ID: ClassVar[str]
    PATHWAY: ClassVar[Pathway]
    FUSION_EQUATION: ClassVar[str]

    encoder_checkpoint: str = ENCODER_CHECKPOINT
    encoder_revision: str = ENCODER_REVISION
    hidden_size: int = HIDDEN_SIZE
    tone_rows: int = TONE_TABLE_ROWS
    letter_rows: int = LETTER_TABLE_ROWS
    gate_weight_initial: float = GATE_INIT_WEIGHT
    gate_initial_value: float = GATE_INIT_TARGET
    objective_id: str = HISTORICAL_OBJECTIVE_ID
    corpus_dataset: str = CORPUS_DATASET
    corpus_revision: str = CORPUS_REVISION
    precision: str = "fp32"
    encoder_frozen: bool = True

    def __post_init__(self) -> None:
        expected = {
            "encoder_checkpoint": ENCODER_CHECKPOINT,
            "encoder_revision": ENCODER_REVISION,
            "hidden_size": HIDDEN_SIZE,
            "tone_rows": TONE_TABLE_ROWS,
            "letter_rows": LETTER_TABLE_ROWS,
            "gate_weight_initial": GATE_INIT_WEIGHT,
            "gate_initial_value": GATE_INIT_TARGET,
            "objective_id": HISTORICAL_OBJECTIVE_ID,
            "corpus_dataset": CORPUS_DATASET,
            "corpus_revision": CORPUS_REVISION,
            "precision": "fp32",
            "encoder_frozen": True,
        }
        drift = {
            name: getattr(self, name) for name, want in expected.items()
            if getattr(self, name) != want
        }
        if drift:
            raise ViUnMarkContractError(
                f"{type(self).__name__} is a frozen Stage-I identity; these fields differ "
                f"from the method: {drift}"
            )

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
    """ViUnMark-Gate: gated fusion of the fused channels and the base embedding."""

    FUSION_ID: ClassVar[str] = HISTORICAL_FUSION_ID
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    FUSION_EQUATION: ClassVar[str] = "z = g * f + (1 - g) * e"


@dataclass(frozen=True)
class ViUnMarkScaleConfig(_AdaptedPathwayConfig):
    """ViUnMark-Scale: `f` is rescaled to `||e||` before the gated fusion."""

    FUSION_ID: ClassVar[str] = SCALE_CALIBRATED_FUSION_ID
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    FUSION_EQUATION: ClassVar[str] = (
        "scale = ||e||_2 / max(||f||_2, 1e-8); z = g * (scale * f) + (1 - g) * e"
    )
    scale_epsilon: float = FUSION_SCALE_EPSILON

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale_epsilon != FUSION_SCALE_EPSILON:
            raise ViUnMarkContractError(
                f"ViUnMark-Scale's norm floor is fixed at {FUSION_SCALE_EPSILON}, got "
                f"{self.scale_epsilon!r}"
            )


# ---------------------------------------------------------------------------
# Robust MLP head
# ---------------------------------------------------------------------------
ROBUST_MLP_HIDDEN_DIM = 256
ROBUST_MLP_DROPOUT = 0.1
ROBUST_MLP_LAYERS: tuple[str, ...] = ("LayerNorm", "Linear", "GELU", "Dropout", "Linear")


def _positive_int(value: object, what: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ViUnMarkContractError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class RobustMLPHeadConfig:
    """`LayerNorm(d) -> Linear(d, 256) -> GELU -> Dropout(0.1) -> Linear(256, n)`."""

    input_dim: int
    num_labels: int

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, "input_dim")
        _positive_int(self.num_labels, "num_labels", minimum=2)

    @property
    def hidden_dim(self) -> int:
        return ROBUST_MLP_HIDDEN_DIM

    @property
    def dropout(self) -> float:
        return ROBUST_MLP_DROPOUT

    @property
    def parameter_count(self) -> int:
        """`2d + (256d + 256) + (256n + n)` trainable parameters."""
        d, h, n = self.input_dim, ROBUST_MLP_HIDDEN_DIM, self.num_labels
        return 2 * d + (h * d + h) + (h * n + n)


# ---------------------------------------------------------------------------
# Branch readout recipes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BranchReadoutConfig:
    """One branch recipe: pathway, readout, loss, training distribution, head.

    The recipe is fixed per subclass. Only `num_labels`, which belongs to the
    task, is supplied by the caller.
    """

    STREAM: ClassVar[LogitStream]
    PATHWAY: ClassVar[Pathway]
    READOUT: ClassVar[ReadoutKind]
    LOSS: ClassVar[LossKind]
    TRAINING_DISTRIBUTION: ClassVar[TrainingDistribution] = (
        TrainingDistribution.AUGMENTED_SIX_CONDITIONS
    )

    num_labels: int
    hidden_size: int = HIDDEN_SIZE

    def __post_init__(self) -> None:
        if type(self) is BranchReadoutConfig:
            raise ViUnMarkContractError(
                "BranchReadoutConfig is abstract; use one of the named branch recipes"
            )
        _positive_int(self.num_labels, "num_labels", minimum=2)
        if self.hidden_size != HIDDEN_SIZE:
            raise ViUnMarkContractError(
                f"branch readouts read the {HIDDEN_SIZE}-dimensional encoder, got "
                f"hidden_size={self.hidden_size!r}"
            )

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
        return RobustMLPHeadConfig(input_dim=self.input_dim, num_labels=self.num_labels)


@dataclass(frozen=True)
class GateReadoutConfig(BranchReadoutConfig):
    """Gate Robust Readout: ViUnMark-Gate, MASKED_MEAN, sqrt-inverse-frequency CE."""

    STREAM: ClassVar[LogitStream] = LogitStream.GATE_ROBUST_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.MASKED_MEAN
    LOSS: ClassVar[LossKind] = LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY


@dataclass(frozen=True)
class ScaleUnweightedReadoutConfig(BranchReadoutConfig):
    """Scale Unweighted Readout: ViUnMark-Scale, CONCAT, unweighted CE."""

    STREAM: ClassVar[LogitStream] = LogitStream.SCALE_UNWEIGHTED_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


@dataclass(frozen=True)
class ScaleWeightedReadoutConfig(BranchReadoutConfig):
    """Scale Weighted Readout: ViUnMark-Scale, CONCAT, sqrt-inverse-frequency CE."""

    STREAM: ClassVar[LogitStream] = LogitStream.SCALE_WEIGHTED_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_SCALE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY


@dataclass(frozen=True)
class PhoBERTReadoutConfig(BranchReadoutConfig):
    """PhoBERT Readout: native PhoBERT, CONCAT, unweighted CE."""

    STREAM: ClassVar[LogitStream] = LogitStream.PHOBERT_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.PHOBERT
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


@dataclass(frozen=True)
class GateMatchedRecipeReadoutConfig(BranchReadoutConfig):
    """Diagnostic only: ViUnMark-Gate under the matched downstream recipe."""

    STREAM: ClassVar[LogitStream] = LogitStream.GATE_MATCHED_RECIPE_READOUT
    PATHWAY: ClassVar[Pathway] = Pathway.VIUNMARK_GATE
    READOUT: ClassVar[ReadoutKind] = ReadoutKind.CONCAT
    LOSS: ClassVar[LossKind] = LossKind.UNWEIGHTED_CROSS_ENTROPY


READOUT_CONFIG_BY_STREAM: dict[LogitStream, type[BranchReadoutConfig]] = {
    config.STREAM: config
    for config in (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
        PhoBERTReadoutConfig,
        GateMatchedRecipeReadoutConfig,
    )
}


# ---------------------------------------------------------------------------
# Calibration -- always dataset-specific
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    """One additive logit bias on at most one class. **Dataset-specific.**

    `class_index=None` is the identity: no bias at all. There is no method-level
    calibration value; a dataset's reproduction record states its own.
    """

    class_index: int | None
    additive_logit_bias: float = 0.0
    label_name: str | None = None

    def __post_init__(self) -> None:
        if self.class_index is None:
            if self.additive_logit_bias != 0.0 or self.label_name is not None:
                raise ViUnMarkContractError(
                    "a calibration with no class_index is the identity and carries no bias "
                    f"and no label; got bias={self.additive_logit_bias!r}, "
                    f"label_name={self.label_name!r}"
                )
            return
        _positive_int(self.class_index, "class_index", minimum=0)
        if isinstance(self.additive_logit_bias, bool) or not isinstance(
            self.additive_logit_bias, (int, float)
        ) or not math.isfinite(self.additive_logit_bias):
            raise ViUnMarkContractError(
                f"additive_logit_bias must be a finite number, got {self.additive_logit_bias!r}"
            )

    @classmethod
    def identity(cls) -> "CalibrationConfig":
        return cls(class_index=None)

    @property
    def is_identity(self) -> bool:
        return self.class_index is None or self.additive_logit_bias == 0.0

    def bias_vector(self, num_labels: int) -> tuple[float, ...]:
        """The per-class additive bias for a `num_labels`-way output."""
        _positive_int(num_labels, "num_labels", minimum=2)
        vector = [0.0] * num_labels
        if self.class_index is not None:
            if self.class_index >= num_labels:
                raise ViUnMarkContractError(
                    f"class_index {self.class_index} is outside a {num_labels}-way output"
                )
            vector[self.class_index] = float(self.additive_logit_bias)
        return tuple(vector)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def _head_seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ViUnMarkContractError(
            f"head_seeds must be a non-empty tuple of ints, got {value!r}"
        )
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ViUnMarkContractError(f"head seed {seed!r} is not an int")
    if len(set(value)) != len(value):
        raise ViUnMarkContractError(f"head_seeds contain duplicates: {value!r}")
    return value


@dataclass(frozen=True)
class _FusionSystemConfig:
    """Shared shape of the two fusion systems."""

    num_labels: int
    head_seeds: tuple[int, ...]
    """One trained head per seed, in every branch. Branch logits are the MEAN over
    these heads; no seed is ever selected over another."""
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig.identity)

    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]]

    def __post_init__(self) -> None:
        _positive_int(self.num_labels, "num_labels", minimum=2)
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
        return tuple(branch(num_labels=self.num_labels) for branch in self.BRANCHES)

    @property
    def streams(self) -> tuple[LogitStream, ...]:
        return tuple(branch.STREAM for branch in self.BRANCHES)

    @property
    def stage2_parameter_count(self) -> int:
        return sum(b.head.parameter_count for b in self.branches) * len(self.head_seeds)


@dataclass(frozen=True)
class AdaptedOnlyFusionConfig(_FusionSystemConfig):
    """Adapted-Only Fusion. **Diagnostic / ablation, not the proposed system.**"""

    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]] = (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
    )
    IS_DIAGNOSTIC: ClassVar[bool] = True


@dataclass(frozen=True)
class ViUnMarkConfig(_FusionSystemConfig):
    """ViUnMark, the proposed system: four branches, hierarchical raw-logit fusion."""

    BRANCHES: ClassVar[tuple[type[BranchReadoutConfig], ...]] = (
        GateReadoutConfig,
        ScaleUnweightedReadoutConfig,
        ScaleWeightedReadoutConfig,
        PhoBERTReadoutConfig,
    )
    IS_DIAGNOSTIC: ClassVar[bool] = False


__all__ = [
    "AdaptedOnlyFusionConfig",
    "BranchReadoutConfig",
    "CONCAT_ORDER",
    "CalibrationConfig",
    "FINAL_BRANCH_STREAMS",
    "GateMatchedRecipeReadoutConfig",
    "GateReadoutConfig",
    "LogitStream",
    "LossKind",
    "Pathway",
    "PhoBERTReadoutConfig",
    "READOUT_CONFIG_BY_STREAM",
    "ROBUST_MLP_DROPOUT",
    "ROBUST_MLP_HIDDEN_DIM",
    "ROBUST_MLP_LAYERS",
    "ReadoutKind",
    "RobustMLPHeadConfig",
    "SIX_CONDITIONS",
    "ScaleUnweightedReadoutConfig",
    "ScaleWeightedReadoutConfig",
    "TrainingDistribution",
    "ViUnMarkConfig",
    "ViUnMarkContractError",
    "ViUnMarkGateConfig",
    "ViUnMarkScaleConfig",
]
