"""Native-Adapted Decision Geometry. **Torch-free.**

Asks whether the adapted pathway keeps the decision geometry native PhoBERT
presents to a classifier. Recorded quantities of this analysis:

* representation cosine distance between native and adapted features;
* agreement of one native head's predictions on native vs adapted features;
* a centered-logit cosine.

Implemented here: the first two, per example, since their definitions leave no
free choice. Rows are aligned by position; callers pass matching orders.

Not implemented, deliberately: the centered-logit cosine. Which axis the logits
are centred over and how per-example values are aggregated are not recorded.
Per-example outputs are returned so no aggregation is imposed silently.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from unmark.viunmark.config import ViUnMarkContractError
from unmark.viunmark.fusion import argmax_rows

CENTERED_LOGIT_COSINE_IMPLEMENTED = False


def _rows(values: Any, what: str) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    rows = [[float(v) for v in row] for row in values]
    if not rows or not rows[0]:
        raise ViUnMarkContractError(f"{what} is empty")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ViUnMarkContractError(f"{what} is not rectangular")
    return rows


def cosine_distances(native: Any, adapted: Any) -> tuple[float, ...]:
    """Per-example `1 - cos(native_i, adapted_i)`.

    A zero-norm row has no direction, so its cosine is undefined. It is refused
    rather than assigned a value.
    """
    left, right = _rows(native, "native"), _rows(adapted, "adapted")
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ViUnMarkContractError("native and adapted features must have the same shape")
    distances = []
    for index, (a, b) in enumerate(zip(left, right)):
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            raise ViUnMarkContractError(f"row {index} has a zero-norm feature; cosine is undefined")
        distances.append(1.0 - sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b))
    return tuple(distances)


def prediction_agreement(native_logits: Any, adapted_logits: Any) -> float:
    """Fraction of examples where one head predicts the same class on both inputs.

    Pass the SAME head's logits on the native features and on the adapted
    features. The first maximal logit wins a tie.
    """
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


__all__ = [
    "CENTERED_LOGIT_COSINE_IMPLEMENTED",
    "cosine_distances",
    "mean",
    "prediction_agreement",
]
