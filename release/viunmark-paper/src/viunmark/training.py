"""Recovered public training policy facts for robust readouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from viunmark.config import SIX_CONDITIONS, ViUnMarkContractError

RECIPE_EVIDENCE_SCOPE = (
    "gate_robust_readout",
    "scale_unweighted_readout",
    "scale_weighted_readout",
)
PHOBERT_READOUT_POLICY_CONFIRMED = (
    "head_architecture",
    "initialization",
    "optimizer_name",
    "learning_rate",
    "matrix_weight_decay",
    "bias_and_vector_weight_decay",
    "budget",
    "six_condition_training_order",
    "seeds",
    "best_seed_selection",
    "precision",
    "checkpoint_selection",
    "selected_candidate",
    "deployment",
)
PHOBERT_READOUT_NOT_ESTABLISHED = (
    "adamw_betas",
    "adamw_eps",
    "augmented_concatenation_implementation",
    "cycle_seed",
    "dropout_seed",
    "cross_entropy_reduction",
    "initialization_rng_stream",
    "partial_final_batch",
    "state_dict_layout",
)
TRAINING_DTYPE = "float32"
TRAINING_AMP = False
TRAINING_TF32 = False
CROSS_ENTROPY_REDUCTION_RECOVERED = False
LAYERNORM_WEIGHT_INIT = 1.0
LAYERNORM_BIAS_INIT = 0.0
LINEAR_WEIGHT_INIT = "xavier_uniform"
LINEAR_BIAS_INIT = 0.0


@dataclass(frozen=True)
class RobustMLPOptimizerPolicy:
    name: str = "AdamW"
    learning_rate: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    matrix_weight_decay: float = 0.01
    non_matrix_weight_decay: float = 0.0

    def weight_decay_for(self, parameter_ndim: int) -> float:
        if isinstance(parameter_ndim, bool) or not isinstance(parameter_ndim, int) or parameter_ndim < 1:
            raise ViUnMarkContractError("parameter_ndim must be a positive int")
        return self.matrix_weight_decay if parameter_ndim >= 2 else self.non_matrix_weight_decay


ROBUST_MLP_OPTIMIZER_POLICY = RobustMLPOptimizerPolicy()


@dataclass(frozen=True)
class HeadTrainingBudget:
    batch_size: int
    max_optimizer_updates: int
    selection_boundaries: int

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_optimizer_updates", "selection_boundaries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ViUnMarkContractError(f"{name} must be a positive int")
        if self.max_optimizer_updates % self.selection_boundaries:
            raise ViUnMarkContractError("updates must divide evenly into selection boundaries")

    @property
    def updates_per_boundary(self) -> int:
        return self.max_optimizer_updates // self.selection_boundaries

    def boundary_update(self, boundary: int) -> int:
        if isinstance(boundary, bool) or not isinstance(boundary, int) or not 1 <= boundary <= self.selection_boundaries:
            raise ViUnMarkContractError("boundary out of range")
        return boundary * self.updates_per_boundary


ROBUST_MLP_TRAINING_BUDGET = HeadTrainingBudget(128, 2160, 30)


def augmented_row_order(num_rows: int) -> tuple[tuple[str, int], ...]:
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 1:
        raise ViUnMarkContractError("num_rows must be a positive int")
    return tuple((condition, row) for condition in SIX_CONDITIONS for row in range(num_rows))


def augmented_labels(labels: Sequence[int]) -> tuple[int, ...]:
    labels = tuple(labels)
    if not labels:
        raise ViUnMarkContractError("cannot augment empty labels")
    return labels * len(SIX_CONDITIONS)


def cycle_seed(seed: int, cycle_index: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ViUnMarkContractError("seed must be a non-negative int")
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 1:
        raise ViUnMarkContractError("cycle_index must be >= 1")
    return seed * 1000 + cycle_index


def dropout_seed(seed: int, optimizer_update: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ViUnMarkContractError("seed must be a non-negative int")
    if isinstance(optimizer_update, bool) or not isinstance(optimizer_update, int) or optimizer_update < 0:
        raise ViUnMarkContractError("optimizer_update must be >= 0")
    return seed * 100000 + optimizer_update


@dataclass(frozen=True)
class BoundaryScore:
    boundary: int
    macro_f1_by_condition: Mapping[str, float] = field(repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.boundary, bool) or not isinstance(self.boundary, int) or self.boundary < 1:
            raise ViUnMarkContractError("boundary must be positive")
        if set(self.macro_f1_by_condition) != set(SIX_CONDITIONS):
            raise ViUnMarkContractError("boundary scores must cover exactly the six conditions")
        for value in self.macro_f1_by_condition.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ViUnMarkContractError("Macro-F1 values must be finite")

    @property
    def six_condition_mean(self) -> float:
        return sum(self.macro_f1_by_condition[c] for c in SIX_CONDITIONS) / len(SIX_CONDITIONS)

    @property
    def worst_condition(self) -> float:
        return min(self.macro_f1_by_condition[c] for c in SIX_CONDITIONS)

    @property
    def full(self) -> float:
        return self.macro_f1_by_condition["FULL"]


def select_checkpoint_boundary(scores: Sequence[BoundaryScore]) -> BoundaryScore:
    if not scores:
        raise ViUnMarkContractError("cannot select from no boundaries")
    boundaries = [score.boundary for score in scores]
    if len(set(boundaries)) != len(boundaries):
        raise ViUnMarkContractError("duplicate selection boundaries")
    return min(scores, key=lambda s: (-s.six_condition_mean, -s.worst_condition, -s.full, s.boundary))
