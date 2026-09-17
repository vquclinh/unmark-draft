"""Native-adapted decision geometry helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

from viunmark.config import ViUnMarkContractError
from viunmark.fusion import argmax_rows

CENTERED_LOGIT_COSINE_IMPLEMENTED = False


def _rows(values: Any, what: str) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    rows = [[float(v) for v in row] for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ViUnMarkContractError(f"{what} is empty or ragged")
    return rows


def cosine_distances(native: Any, adapted: Any) -> tuple[float, ...]:
    left, right = _rows(native, "native"), _rows(adapted, "adapted")
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ViUnMarkContractError("native and adapted features must have the same shape")
    out = []
    for index, (a, b) in enumerate(zip(left, right)):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0.0 or nb == 0.0:
            raise ViUnMarkContractError(f"row {index} has a zero-norm feature")
        out.append(1.0 - sum(x * y for x, y in zip(a, b)) / (na * nb))
    return tuple(out)


def prediction_agreement(native_logits: Any, adapted_logits: Any) -> float:
    left, right = _rows(native_logits, "native_logits"), _rows(adapted_logits, "adapted_logits")
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ViUnMarkContractError("native and adapted logits must have the same shape")
    native_pred = argmax_rows(tuple(map(tuple, left)))
    adapted_pred = argmax_rows(tuple(map(tuple, right)))
    return sum(a == b for a, b in zip(native_pred, adapted_pred)) / len(native_pred)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ViUnMarkContractError("cannot average no values")
    return sum(values) / len(values)
