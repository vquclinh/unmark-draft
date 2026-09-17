"""Class weights for the branch losses. **Torch-free.**

`SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY` weights each class by::

    raw_weight[c] = 1 / sqrt(train_count[c])
    weight[c]     = raw_weight[c] / mean(raw_weight)

The RULE is the method. The counts, and so the numbers, come from whichever
training split a head is trained on. No weight vector is a default here.

Only the class-weight rule lives in this module. How the cross-entropy call
reduces over a batch and whether it smooths labels are head-training details
that this layer does not fix.
"""

from __future__ import annotations

import math
import struct
from typing import Sequence

from unmark.viunmark.config import LossKind, ViUnMarkContractError

WEIGHT_ARITHMETIC = ("float64", "float32")


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def class_counts_from_labels(labels: Sequence[int], num_labels: int) -> tuple[int, ...]:
    """Per-class counts over `range(num_labels)`. Refuses out-of-range labels."""
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels < 2:
        raise ViUnMarkContractError(f"num_labels must be an int >= 2, got {num_labels!r}")
    counts = [0] * num_labels
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_labels:
            raise ViUnMarkContractError(f"label {label!r} is outside range({num_labels})")
        counts[label] += 1
    return tuple(counts)


def sqrt_inverse_frequency_weights(
    class_counts: Sequence[int], *, arithmetic: str = "float64"
) -> tuple[float, ...]:
    """The normalised inverse-square-root-frequency class weights.

    Args:
        class_counts: training count per class, in class-index order. Every class
            must be present: a zero count has no defined weight, and quietly
            substituting one would change the loss.
        arithmetic: `"float64"` (default) or `"float32"`. The rule is the same;
            `"float32"` evaluates each step at single precision, which is how a
            weight tensor computed in FP32 training code comes out.

    The result has mean exactly 1 up to the chosen precision, and it does not
    change if every count is multiplied by the same factor.
    """
    if arithmetic not in WEIGHT_ARITHMETIC:
        raise ViUnMarkContractError(
            f"arithmetic must be one of {WEIGHT_ARITHMETIC}, got {arithmetic!r}"
        )
    counts = tuple(class_counts)
    if len(counts) < 2:
        raise ViUnMarkContractError("class weights need at least two classes")
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ViUnMarkContractError(
                f"every class count must be a positive int, got {counts!r}"
            )
    if arithmetic == "float64":
        raw = [1.0 / math.sqrt(count) for count in counts]
        mean = sum(raw) / len(raw)
        return tuple(value / mean for value in raw)

    raw32 = [_f32(1.0 / _f32(math.sqrt(_f32(float(count))))) for count in counts]
    total = _f32(0.0)
    for value in raw32:
        total = _f32(total + value)
    mean32 = _f32(total / len(raw32))
    return tuple(_f32(value / mean32) for value in raw32)


def class_weights_for(
    loss: LossKind, class_counts: Sequence[int] | None, *, arithmetic: str = "float64"
) -> tuple[float, ...] | None:
    """Class weights a loss uses: `None` for unweighted, the rule for weighted.

    The weighted loss refuses to run without counts rather than falling back to
    uniform weights or to another dataset's vector.
    """
    if not isinstance(loss, LossKind):
        raise ViUnMarkContractError(f"loss must be a LossKind, got {loss!r}")
    if loss is LossKind.UNWEIGHTED_CROSS_ENTROPY:
        if class_counts is not None:
            raise ViUnMarkContractError(
                "UNWEIGHTED_CROSS_ENTROPY takes no class counts; passing them suggests the "
                "wrong loss was chosen"
            )
        return None
    if class_counts is None:
        raise ViUnMarkContractError(
            "SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY needs the training split's class counts; "
            "there is no default weight vector"
        )
    return sqrt_inverse_frequency_weights(class_counts, arithmetic=arithmetic)


__all__ = [
    "WEIGHT_ARITHMETIC",
    "class_counts_from_labels",
    "class_weights_for",
    "sqrt_inverse_frequency_weights",
]
