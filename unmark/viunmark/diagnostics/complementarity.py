"""Matched-Recipe Complementarity Analysis. **Torch-free.**

Asks whether plain averaging explains complementarity when every branch is
trained with the SAME downstream recipe:

    readout               CONCAT [FT ; MM]
    loss                  UNWEIGHTED_CROSS_ENTROPY
    head                  robust MLP
    training distribution AUGMENTED_SIX_CONDITIONS
    heads per branch      5
    calibration           none

A branch satisfies the recipe when its readout configuration has exactly these
settings. `matched_recipe_streams` derives that set from the configurations, so
it cannot drift from them. `matched_recipe_average` averages RAW branch
ensembles with equal weight. Which branches enter the average is the caller's
explicit choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from unmark.viunmark.config import (
    READOUT_CONFIG_BY_STREAM,
    CalibrationConfig,
    LogitStream,
    LossKind,
    ReadoutKind,
    TrainingDistribution,
    ViUnMarkContractError,
)
from unmark.viunmark.fusion import BranchLogits, Matrix

MATCHED_RECIPE_READOUT = ReadoutKind.CONCAT
MATCHED_RECIPE_LOSS = LossKind.UNWEIGHTED_CROSS_ENTROPY
MATCHED_RECIPE_TRAINING_DISTRIBUTION = TrainingDistribution.AUGMENTED_SIX_CONDITIONS
MATCHED_RECIPE_HEADS_PER_BRANCH = 5
MATCHED_RECIPE_CALIBRATION = CalibrationConfig.identity()


def matched_recipe_streams() -> tuple[LogitStream, ...]:
    """Every branch stream whose recipe is the matched recipe."""
    return tuple(
        stream
        for stream, config in READOUT_CONFIG_BY_STREAM.items()
        if config.READOUT is MATCHED_RECIPE_READOUT
        and config.LOSS is MATCHED_RECIPE_LOSS
        and config.TRAINING_DISTRIBUTION is MATCHED_RECIPE_TRAINING_DISTRIBUTION
    )


@dataclass(frozen=True)
class MatchedRecipeAverage:
    streams: tuple[LogitStream, ...]
    head_seeds: tuple[int, ...]
    row_ids: tuple[str, ...]
    values: Matrix


def matched_recipe_average(branches: Sequence[BranchLogits]) -> MatchedRecipeAverage:
    """Equal-weight mean of two or more matched-recipe RAW branch ensembles."""
    if len(branches) < 2:
        raise ViUnMarkContractError("a complementarity average needs at least two branches")
    allowed = set(matched_recipe_streams())
    for branch in branches:
        if type(branch) is not BranchLogits:
            raise ViUnMarkContractError(
                f"matched_recipe_average takes RAW branch ensembles, got {type(branch).__name__}"
            )
        if branch.stream not in allowed:
            raise ViUnMarkContractError(
                f"{branch.stream.value} is not trained with the matched recipe"
            )
        if len(branch.head_seeds) != MATCHED_RECIPE_HEADS_PER_BRANCH:
            raise ViUnMarkContractError(
                f"{branch.stream.value} averages {len(branch.head_seeds)} heads; the matched "
                f"recipe uses {MATCHED_RECIPE_HEADS_PER_BRANCH}"
            )
    pathways = [READOUT_CONFIG_BY_STREAM[b.stream].PATHWAY for b in branches]
    if len(set(pathways)) != len(pathways):
        raise ViUnMarkContractError("each branch in the average must read a different pathway")
    first = branches[0]
    for branch in branches[1:]:
        if branch.row_ids != first.row_ids or branch.num_labels != first.num_labels:
            raise ViUnMarkContractError("branches cover different rows or class counts")
        if set(branch.head_seeds) != set(first.head_seeds):
            raise ViUnMarkContractError("branches were ensembled over different head seeds")
    weight = 1.0 / len(branches)
    values = tuple(
        tuple(sum(b.values[r][c] for b in branches) * weight for c in range(first.num_labels))
        for r in range(len(first.row_ids))
    )
    return MatchedRecipeAverage(
        streams=tuple(b.stream for b in branches),
        head_seeds=first.head_seeds,
        row_ids=first.row_ids,
        values=values,
    )


__all__ = [
    "MATCHED_RECIPE_CALIBRATION",
    "MATCHED_RECIPE_HEADS_PER_BRANCH",
    "MATCHED_RECIPE_LOSS",
    "MATCHED_RECIPE_READOUT",
    "MATCHED_RECIPE_TRAINING_DISTRIBUTION",
    "MatchedRecipeAverage",
    "matched_recipe_average",
    "matched_recipe_streams",
]
