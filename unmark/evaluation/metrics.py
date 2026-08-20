"""Downstream metrics (proposal §6.5). **Pure Python — no torch, no sklearn.**

§6.5 specifies exactly:

* **Task level.** "Macro-F1 and accuracy per task and condition."
* **Gap Recovery Rate**, the headline number::

      GRR = (S_system - S_FLOOR) / (S_UPPER - S_FLOOR)

Implemented without a numerics dependency so they are deterministic, exactly
testable, and usable in the ML-free environment.

The §6.5 "Representation level" and "Cost" metrics are **not** implemented here:
they are not needed for the Vanilla-vs-Base-only diagnostic, and stubbing them
would suggest coverage this module does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from unmark.evaluation.contracts import EvaluationContractViolation


def _validate(predictions: Sequence[int], labels: Sequence[int]) -> None:
    if len(predictions) != len(labels):
        raise EvaluationContractViolation(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) differ in length"
        )
    if not predictions:
        raise EvaluationContractViolation("cannot score an empty prediction set")


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Fraction of exactly correct predictions, in `[0, 1]`."""
    _validate(predictions, labels)
    return sum(1 for p, y in zip(predictions, labels) if p == y) / len(labels)


@dataclass(frozen=True)
class ClassScore:
    """Per-class precision, recall and F1, with the counts they came from."""

    label: int
    support: int
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def per_class_scores(
    predictions: Sequence[int], labels: Sequence[int], num_labels: int | None = None
) -> tuple[ClassScore, ...]:
    """One `ClassScore` per class.

    The class set is the union of labels seen in `labels` and `predictions`
    unless `num_labels` is given. Including predicted-but-absent classes matters:
    a model that invents a class it was never shown should be penalised, not
    silently ignored.
    """
    _validate(predictions, labels)
    classes = (
        range(num_labels) if num_labels is not None else sorted(set(labels) | set(predictions))
    )
    scores = []
    for label in classes:
        tp = sum(1 for p, y in zip(predictions, labels) if p == label and y == label)
        fp = sum(1 for p, y in zip(predictions, labels) if p == label and y != label)
        fn = sum(1 for p, y in zip(predictions, labels) if p != label and y == label)
        scores.append(
            ClassScore(
                label=label,
                support=sum(1 for y in labels if y == label),
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
            )
        )
    return tuple(scores)


def macro_f1(
    predictions: Sequence[int], labels: Sequence[int], num_labels: int | None = None
) -> float:
    """Unweighted mean of per-class F1 (§6.5).

    Macro, so every class counts equally regardless of support -- which is the
    point on the imbalanced tasks §6.2 names.
    """
    scores = per_class_scores(predictions, labels, num_labels)
    return sum(s.f1 for s in scores) / len(scores)


# ---------------------------------------------------------------------------
# Gap Recovery Rate
# ---------------------------------------------------------------------------
GRR_FORMULA = "(S_system - S_FLOOR) / (S_UPPER - S_FLOOR)"
"""Proposal §6.5, verbatim.

§6.4 fixes the anchors: `UPPER` is "Clean input, unmodified model" and `FLOOR` is
"Corrupted input, unmodified model". **Both anchors are the VANILLA pathway** --
`S_UPPER` is vanilla at `FULL`, `S_FLOOR` is vanilla at the same corruption
condition the system is being scored at.
"""


class UndefinedGRR(ValueError):
    """Raised when `S_UPPER == S_FLOOR`, so the denominator is zero.

    §6.5 defines **no** epsilon, clamp or fallback for this case, so none is
    invented. A zero denominator means corruption cost the unmodified model
    nothing at this condition -- there is no gap, and "the fraction of the gap
    recovered" is not a meaningful quantity rather than a large one.
    """


def gap_recovery_rate(
    score_system: float, score_floor: float, score_upper: float
) -> float:
    """`GRR = (S_system - S_FLOOR) / (S_UPPER - S_FLOOR)` (§6.5).

    Args:
        score_system: the system's score at corruption condition `c`.
        score_floor: `FLOOR` -- **vanilla** at the same condition `c`.
        score_upper: `UPPER` -- **vanilla** at `FULL`.

    Returns:
        The recovered fraction. **Not clamped.** `0.0` means the system matched
        the floor and recovered nothing; `1.0` means it reached the upper bound;
        above `1.0` means it beat clean vanilla, and below `0.0` means it did
        worse than the corrupted unmodified model. §6.5 prescribes no clamping,
        and clamping would hide exactly those two informative outcomes.

    Raises:
        UndefinedGRR: when `score_upper == score_floor`.
    """
    denominator = score_upper - score_floor
    if denominator == 0:
        raise UndefinedGRR(
            f"S_UPPER == S_FLOOR == {score_upper}: the denominator is zero, so GRR is "
            "undefined. §6.5 prescribes no epsilon or fallback, and none is invented "
            "here (grr_degenerate_denominator_policy is OPEN)."
        )
    return (score_system - score_floor) / denominator


def is_grr_defined(score_floor: float, score_upper: float) -> bool:
    """Whether GRR can be computed for these anchors."""
    return score_upper != score_floor
