"""Robust MLP head training recipe. **Torch-free.**

The recipe below was recovered for the adapted-pathway heads (Gate Robust
Readout, Scale Unweighted Readout, Scale Weighted Readout). It is exposed as
policy, not as dataset data. Nothing in this module holds a row count, a class
count, a corruption seed, or a selected boundary of any dataset.

    initialisation   LayerNorm weight 1, bias 0; Linear Xavier-uniform weight, bias 0
    optimizer        AdamW, lr 0.01, betas (0.9, 0.999), eps 1e-8,
                     weight decay 0.01 on matrix weights, 0 on biases, vectors, LayerNorm
    budget           batch 128, 2160 optimizer updates, 30 selection boundaries (72 updates each)
    training set     condition-major concatenation FULL, P25, P50, P75, P100, STRIP_ALL;
                     each condition keeps the training split's row order; labels repeat per condition
    batch stream     cycle_index from 1; cycle_seed = seed * 1000 + cycle_index;
                     cycles continue until the update budget is exhausted
    dropout          dropout_seed = seed * 100000 + optimizer_update
    selection        per head, on protocol-dev: six-condition Macro-F1 mean, then worst-condition
                     Macro-F1, then FULL Macro-F1, then the earlier boundary
    precision        float32, no AMP, no TF32

Not recovered, and therefore not encoded: the cross-entropy reduction of these
heads (`CROSS_ENTROPY_REDUCTION_RECOVERED`).

This recipe is recovered at protocol level. It is not a bit-exact record of the
historical execution: the within-cycle permutation, partial-batch handling, the
initialisation random stream and the head-file layout are not recorded.

**PhoBERT Readout heads.** The native protocol confirms a subset of this recipe,
listed in `PHOBERT_READOUT_POLICY_CONFIRMED`. It does not establish the fields in
`PHOBERT_READOUT_NOT_ESTABLISHED`, among them the AdamW betas and eps, the
cycle-seed and dropout-seed rules, and the augmented-concatenation
implementation. Those Gate/Scale facts are never promoted to the PhoBERT
Readout heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from unmark.viunmark.config import SIX_CONDITIONS, ViUnMarkContractError

RECIPE_EVIDENCE_SCOPE: tuple[str, ...] = (
    "gate_robust_readout",
    "scale_unweighted_readout",
    "scale_weighted_readout",
)
"""The branch streams the FULL recovered training recipe is recorded for."""

PHOBERT_READOUT_POLICY_CONFIRMED: tuple[str, ...] = (
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
"""What the frozen native PhoBERT protocol directly confirms for the PhoBERT Readout heads."""

PHOBERT_READOUT_NOT_ESTABLISHED: tuple[str, ...] = (
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
"""What that protocol does NOT establish. Never filled in from the Gate/Scale recipe."""

# ---------------------------------------------------------------------------
# Precision and loss
# ---------------------------------------------------------------------------
TRAINING_DTYPE = "float32"
TRAINING_AMP = False
TRAINING_TF32 = False

CROSS_ENTROPY_REDUCTION_RECOVERED = False
"""The reduction of the heads' cross-entropy is NOT recovered.

The runner that trained these heads is not in the repository. Every
cross-entropy call the repository does contain trains a single linear head
without class weights, so none of them can stand in for it. A trainer built on
this recipe must choose and record its reduction explicitly.
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
LAYERNORM_WEIGHT_INIT = 1.0
LAYERNORM_BIAS_INIT = 0.0
LINEAR_WEIGHT_INIT = "xavier_uniform"
LINEAR_BIAS_INIT = 0.0


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobustMLPOptimizerPolicy:
    """AdamW with decoupled weight decay on matrix weights only."""

    name: str = "AdamW"
    learning_rate: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    matrix_weight_decay: float = 0.01
    non_matrix_weight_decay: float = 0.0

    def weight_decay_for(self, parameter_ndim: int) -> float:
        """Matrix weights (ndim >= 2) decay; biases, vectors and LayerNorm do not."""
        if isinstance(parameter_ndim, bool) or not isinstance(parameter_ndim, int) or (
            parameter_ndim < 1
        ):
            raise ViUnMarkContractError(f"parameter ndim must be an int >= 1, got {parameter_ndim!r}")
        return self.matrix_weight_decay if parameter_ndim >= 2 else self.non_matrix_weight_decay


ROBUST_MLP_OPTIMIZER_POLICY = RobustMLPOptimizerPolicy()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadTrainingBudget:
    """A fixed optimizer-update budget, evaluated at evenly spaced boundaries."""

    batch_size: int
    max_optimizer_updates: int
    selection_boundaries: int

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_optimizer_updates", "selection_boundaries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ViUnMarkContractError(f"{name} must be a positive int, got {value!r}")
        if self.max_optimizer_updates % self.selection_boundaries:
            raise ViUnMarkContractError(
                f"{self.max_optimizer_updates} updates do not divide into "
                f"{self.selection_boundaries} equal selection intervals"
            )

    @property
    def updates_per_boundary(self) -> int:
        return self.max_optimizer_updates // self.selection_boundaries

    def boundary_update(self, boundary: int) -> int:
        """The optimizer update at which 1-based `boundary` is evaluated."""
        if isinstance(boundary, bool) or not isinstance(boundary, int) or not (
            1 <= boundary <= self.selection_boundaries
        ):
            raise ViUnMarkContractError(
                f"boundary must be in 1..{self.selection_boundaries}, got {boundary!r}"
            )
        return boundary * self.updates_per_boundary


ROBUST_MLP_TRAINING_BUDGET = HeadTrainingBudget(
    batch_size=128, max_optimizer_updates=2160, selection_boundaries=30
)


# ---------------------------------------------------------------------------
# Six-condition augmented training set
# ---------------------------------------------------------------------------
def augmented_row_order(num_rows: int) -> tuple[tuple[str, int], ...]:
    """`(condition, row_index)` in condition-major order.

    All rows under `FULL` in their original order, then all rows under `P25`, and so
    on through `STRIP_ALL`. The training split's row order is preserved inside
    every condition.
    """
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 1:
        raise ViUnMarkContractError(f"num_rows must be a positive int, got {num_rows!r}")
    return tuple((condition, row) for condition in SIX_CONDITIONS for row in range(num_rows))


def augmented_labels(labels: Sequence[int]) -> tuple[int, ...]:
    """The training labels repeated once per condition, in condition-major order."""
    labels = tuple(labels)
    if not labels:
        raise ViUnMarkContractError("cannot augment an empty label vector")
    return labels * len(SIX_CONDITIONS)


# ---------------------------------------------------------------------------
# Deterministic streams
# ---------------------------------------------------------------------------
def _int(value: object, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ViUnMarkContractError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def cycle_seed(seed: int, cycle_index: int) -> int:
    """`seed * 1000 + cycle_index`. Cycles are numbered from 1."""
    _int(seed, "seed", 0)
    _int(cycle_index, "cycle_index", 1)
    return seed * 1000 + cycle_index


def dropout_seed(seed: int, optimizer_update: int) -> int:
    """`seed * 100000 + optimizer_update`."""
    _int(seed, "seed", 0)
    _int(optimizer_update, "optimizer_update", 0)
    return seed * 100000 + optimizer_update


# ---------------------------------------------------------------------------
# Checkpoint selection within one head
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryScore:
    """Protocol-dev Macro-F1 of one head at one selection boundary, per condition."""

    boundary: int
    macro_f1_by_condition: Mapping[str, float] = field(repr=False)

    def __post_init__(self) -> None:
        _int(self.boundary, "boundary", 1)
        if set(self.macro_f1_by_condition) != set(SIX_CONDITIONS):
            raise ViUnMarkContractError(
                f"boundary {self.boundary} must score exactly {list(SIX_CONDITIONS)}, got "
                f"{sorted(self.macro_f1_by_condition)}"
            )
        for condition, value in self.macro_f1_by_condition.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                not math.isfinite(value)
            ):
                raise ViUnMarkContractError(f"{condition} Macro-F1 must be finite, got {value!r}")

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
    """Highest six-condition mean, then highest worst-condition, then highest FULL,
    then the EARLIER boundary. A total order; iteration order cannot decide it."""
    if not scores:
        raise ViUnMarkContractError("cannot select a checkpoint from no boundaries")
    boundaries = [score.boundary for score in scores]
    if len(set(boundaries)) != len(boundaries):
        raise ViUnMarkContractError(f"duplicate selection boundaries: {sorted(boundaries)}")
    return min(
        scores,
        key=lambda s: (-s.six_condition_mean, -s.worst_condition, -s.full, s.boundary),
    )


__all__ = [
    "BoundaryScore",
    "CROSS_ENTROPY_REDUCTION_RECOVERED",
    "HeadTrainingBudget",
    "PHOBERT_READOUT_NOT_ESTABLISHED",
    "PHOBERT_READOUT_POLICY_CONFIRMED",
    "LAYERNORM_BIAS_INIT",
    "LAYERNORM_WEIGHT_INIT",
    "LINEAR_BIAS_INIT",
    "LINEAR_WEIGHT_INIT",
    "RECIPE_EVIDENCE_SCOPE",
    "ROBUST_MLP_OPTIMIZER_POLICY",
    "ROBUST_MLP_TRAINING_BUDGET",
    "RobustMLPOptimizerPolicy",
    "TRAINING_AMP",
    "TRAINING_DTYPE",
    "TRAINING_TF32",
    "augmented_labels",
    "augmented_row_order",
    "cycle_seed",
    "dropout_seed",
    "select_checkpoint_boundary",
]
