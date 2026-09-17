"""Loss helpers for ViUnMark readouts."""

from __future__ import annotations

import math
import struct
from typing import Sequence

from viunmark.config import LossKind, ViUnMarkContractError


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def class_counts_from_labels(labels: Sequence[int], num_labels: int) -> tuple[int, ...]:
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels < 2:
        raise ViUnMarkContractError("num_labels must be an int >= 2")
    counts = [0] * num_labels
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_labels:
            raise ViUnMarkContractError(f"label {label!r} is outside range({num_labels})")
        counts[label] += 1
    return tuple(counts)


def sqrt_inverse_frequency_weights(
    class_counts: Sequence[int], *, arithmetic: str = "float64"
) -> tuple[float, ...]:
    if arithmetic not in ("float64", "float32"):
        raise ViUnMarkContractError("arithmetic must be float64 or float32")
    counts = tuple(class_counts)
    if len(counts) < 2 or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in counts):
        raise ViUnMarkContractError("every class count must be a positive int")
    if arithmetic == "float64":
        raw = [1.0 / math.sqrt(c) for c in counts]
        mean = sum(raw) / len(raw)
        return tuple(v / mean for v in raw)
    raw32 = [_f32(1.0 / _f32(math.sqrt(_f32(float(c))))) for c in counts]
    total = _f32(0.0)
    for value in raw32:
        total = _f32(total + value)
    mean32 = _f32(total / len(raw32))
    return tuple(_f32(v / mean32) for v in raw32)


def class_weights_for(
    loss: LossKind, class_counts: Sequence[int] | None, *, arithmetic: str = "float64"
) -> tuple[float, ...] | None:
    if not isinstance(loss, LossKind):
        raise ViUnMarkContractError("loss must be a LossKind")
    if loss is LossKind.UNWEIGHTED_CROSS_ENTROPY:
        if class_counts is not None:
            raise ViUnMarkContractError("unweighted CE takes no class counts")
        return None
    if class_counts is None:
        raise ViUnMarkContractError("weighted CE requires training-split class counts")
    return sqrt_inverse_frequency_weights(class_counts, arithmetic=arithmetic)
