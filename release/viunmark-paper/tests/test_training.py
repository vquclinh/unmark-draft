import pytest

from viunmark.training import (
    CROSS_ENTROPY_REDUCTION_RECOVERED,
    RECIPE_EVIDENCE_SCOPE,
    ROBUST_MLP_TRAINING_BUDGET,
    BoundaryScore,
    augmented_labels,
    augmented_row_order,
    cycle_seed,
    dropout_seed,
    select_checkpoint_boundary,
)

CONDITIONS = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")


def score(boundary, value=0.5, **updates):
    data = {c: value for c in CONDITIONS}
    data.update(updates)
    return BoundaryScore(boundary, data)


def test_training_policy_facts():
    assert RECIPE_EVIDENCE_SCOPE == (
        "gate_robust_readout",
        "scale_unweighted_readout",
        "scale_weighted_readout",
    )
    assert CROSS_ENTROPY_REDUCTION_RECOVERED is False
    assert ROBUST_MLP_TRAINING_BUDGET.updates_per_boundary == 72
    assert augmented_row_order(2) == tuple((c, r) for c in CONDITIONS for r in range(2))
    assert augmented_labels([0, 1]) == (0, 1) * 6
    assert cycle_seed(53148, 1) == 53148001
    assert dropout_seed(9428, 2160) == 942802160


def test_checkpoint_selection_tie_breaks():
    assert select_checkpoint_boundary([score(5), score(2)]).boundary == 2
    assert select_checkpoint_boundary([score(1, FULL=0.9, P25=0.1), score(2, FULL=0.5, P25=0.5)]).boundary == 2
    with pytest.raises(Exception):
        select_checkpoint_boundary([score(1), score(1)])
