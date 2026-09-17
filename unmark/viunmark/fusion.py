"""ViUnMark logit ensembling, fusion and calibration. **Pure functions, torch-free.**

The system is a fixed graph over RAW logits::

    branch_raw      = mean(logits of the branch's heads)          # one head per head seed
    scale_raw       = 0.5 * scale_unweighted_raw + 0.5 * scale_weighted_raw
    adapted_raw     = 0.5 * gate_raw             + 0.5 * scale_raw
    viunmark_raw    = 0.5 * adapted_raw          + 0.5 * phobert_raw

    adapted_only    = adapted_raw  + calibration     (diagnostic)
    viunmark        = viunmark_raw + calibration     (proposed system)

Three types carry the logits, and the functions accept only the right one:

* `HeadLogits` -- one trained head's raw output;
* `BranchLogits` -- the within-branch mean over a branch's heads;
* `FusedLogits` -- an uncalibrated intermediate or system output;
* `CalibratedLogits` -- a system output after its single calibration.

Fusion takes `BranchLogits` only. A `CalibratedLogits` can never be fused, and
`apply_system_calibration` takes only an uncalibrated system output. Parent
biases therefore cannot be stacked by passing objects around.

Logits are held as tuples of tuples of floats. Tensors are accepted through
`.tolist()`. Nothing here loads a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from unmark.viunmark.config import (
    CalibrationConfig,
    LogitStream,
    ViUnMarkContractError,
)

Matrix = tuple[tuple[float, ...], ...]

# ---------------------------------------------------------------------------
# The fusion graph -- declared once, consumed by both the functions and tests
# ---------------------------------------------------------------------------
SCALE_FUSION_WEIGHTS: Mapping[LogitStream, float] = {
    LogitStream.SCALE_UNWEIGHTED_READOUT: 0.5,
    LogitStream.SCALE_WEIGHTED_READOUT: 0.5,
}
ADAPTED_ONLY_GATE_WEIGHT = 0.5
ADAPTED_ONLY_SCALE_WEIGHT = 0.5
VIUNMARK_ADAPTED_WEIGHT = 0.5
VIUNMARK_PHOBERT_WEIGHT = 0.5

WITHIN_BRANCH_LOGIT_ENSEMBLE = "MEAN"
BEST_SEED_SELECTION = False
PARENT_BIASES_STACKED = False


def expanded_adapted_only_weights() -> dict[LogitStream, float]:
    """Branch weights of `adapted_raw`, expanded through the graph."""
    return {
        LogitStream.GATE_ROBUST_READOUT: ADAPTED_ONLY_GATE_WEIGHT,
        **{
            stream: ADAPTED_ONLY_SCALE_WEIGHT * weight
            for stream, weight in SCALE_FUSION_WEIGHTS.items()
        },
    }


def expanded_viunmark_weights() -> dict[LogitStream, float]:
    """Branch weights of `viunmark_raw`, expanded through the graph."""
    return {
        LogitStream.PHOBERT_READOUT: VIUNMARK_PHOBERT_WEIGHT,
        **{
            stream: VIUNMARK_ADAPTED_WEIGHT * weight
            for stream, weight in expanded_adapted_only_weights().items()
        },
    }


def expanded_per_head_weights(
    branch_weights: Mapping[LogitStream, float], heads_per_branch: int
) -> dict[LogitStream, float]:
    """Weight of ONE head in each branch, given equal-weight within-branch means."""
    if isinstance(heads_per_branch, bool) or not isinstance(heads_per_branch, int) or (
        heads_per_branch < 1
    ):
        raise ViUnMarkContractError(
            f"heads_per_branch must be a positive int, got {heads_per_branch!r}"
        )
    return {stream: weight / heads_per_branch for stream, weight in branch_weights.items()}


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
class FusionNode(Enum):
    """An uncalibrated node of the fusion graph."""

    SCALE_RAW = "scale_raw"
    ADAPTED_ONLY_RAW = "adapted_only_raw"
    VIUNMARK_RAW = "viunmark_raw"


class SystemKind(Enum):
    """A calibrated system output."""

    ADAPTED_ONLY_FUSION = "adapted_only_fusion"
    VIUNMARK = "viunmark"


_SYSTEM_FOR_NODE = {
    FusionNode.ADAPTED_ONLY_RAW: SystemKind.ADAPTED_ONLY_FUSION,
    FusionNode.VIUNMARK_RAW: SystemKind.VIUNMARK,
}

_LOGIT_CONTAINERS: tuple[type, ...] = ()  # filled after the classes exist


def _matrix(values: Any, what: str) -> Matrix:
    """Coerce to a rectangular, finite, at-least-2-column float matrix."""
    if isinstance(values, _LOGIT_CONTAINERS):
        raise ViUnMarkContractError(
            f"{what}: got a {type(values).__name__}; pass raw numbers only where a new "
            "logit container is being built, never an existing container"
        )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ViUnMarkContractError(f"{what}: logits must be a matrix, got {type(values).__name__}")
    rows = []
    width = None
    for index, row in enumerate(values):
        if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
            raise ViUnMarkContractError(f"{what}: row {index} is not a sequence")
        cells = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                raise ViUnMarkContractError(f"{what}: row {index} holds non-numeric {cell!r}")
            if not math.isfinite(cell):
                raise ViUnMarkContractError(f"{what}: row {index} holds non-finite {cell!r}")
            cells.append(float(cell))
        if width is None:
            width = len(cells)
        if len(cells) != width:
            raise ViUnMarkContractError(f"{what}: row {index} has {len(cells)} columns, not {width}")
        rows.append(tuple(cells))
    if not rows:
        raise ViUnMarkContractError(f"{what}: logits have no rows")
    if width is None or width < 2:
        raise ViUnMarkContractError(f"{what}: logits need at least 2 classes")
    return tuple(rows)


def _row_ids(value: Any, count: int, what: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(v, str) and v for v in value):
        raise ViUnMarkContractError(f"{what}: row_ids must be a tuple of non-empty strings")
    if len(set(value)) != len(value):
        raise ViUnMarkContractError(f"{what}: row_ids contain duplicates")
    if len(value) != count:
        raise ViUnMarkContractError(
            f"{what}: {len(value)} row_ids for {count} logit rows"
        )
    return value


def _seed(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ViUnMarkContractError(f"{what}: seed must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class HeadLogits:
    """RAW logits of ONE trained head, for one branch and one head seed."""

    stream: LogitStream
    seed: int
    row_ids: tuple[str, ...]
    values: Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.stream, LogitStream):
            raise ViUnMarkContractError("stream must be a LogitStream")
        _seed(self.seed, "HeadLogits")
        object.__setattr__(self, "values", _matrix(self.values, "HeadLogits"))
        _row_ids(self.row_ids, len(self.values), "HeadLogits")

    @property
    def num_labels(self) -> int:
        return len(self.values[0])


@dataclass(frozen=True)
class BranchLogits:
    """RAW branch logits: the arithmetic mean of the branch's head logits.

    Build it with `ensemble_branch`, which checks the head set. `head_seeds` names
    exactly the heads the mean covers.
    """

    stream: LogitStream
    head_seeds: tuple[int, ...]
    row_ids: tuple[str, ...]
    values: Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.stream, LogitStream):
            raise ViUnMarkContractError("stream must be a LogitStream")
        if not isinstance(self.head_seeds, tuple) or not self.head_seeds:
            raise ViUnMarkContractError("BranchLogits: head_seeds must be a non-empty tuple")
        for seed in self.head_seeds:
            _seed(seed, "BranchLogits")
        if len(set(self.head_seeds)) != len(self.head_seeds):
            raise ViUnMarkContractError("BranchLogits: head_seeds contain duplicates")
        object.__setattr__(self, "values", _matrix(self.values, "BranchLogits"))
        _row_ids(self.row_ids, len(self.values), "BranchLogits")

    @property
    def num_labels(self) -> int:
        return len(self.values[0])


@dataclass(frozen=True)
class FusedLogits:
    """An UNCALIBRATED node of the fusion graph."""

    node: FusionNode
    head_seeds: tuple[int, ...]
    row_ids: tuple[str, ...]
    values: Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.node, FusionNode):
            raise ViUnMarkContractError("node must be a FusionNode")
        object.__setattr__(self, "values", _matrix(self.values, "FusedLogits"))
        _row_ids(self.row_ids, len(self.values), "FusedLogits")

    @property
    def num_labels(self) -> int:
        return len(self.values[0])


@dataclass(frozen=True)
class CalibratedLogits:
    """A system output after its ONE calibration. Terminal: never fused again."""

    system: SystemKind
    head_seeds: tuple[int, ...]
    row_ids: tuple[str, ...]
    values: Matrix
    calibration: CalibrationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.system, SystemKind):
            raise ViUnMarkContractError("system must be a SystemKind")
        if not isinstance(self.calibration, CalibrationConfig):
            raise ViUnMarkContractError("calibration must be a CalibrationConfig")
        object.__setattr__(self, "values", _matrix(self.values, "CalibratedLogits"))
        _row_ids(self.row_ids, len(self.values), "CalibratedLogits")

    @property
    def num_labels(self) -> int:
        return len(self.values[0])

    def predictions(self) -> tuple[int, ...]:
        return argmax_rows(self.values)


_LOGIT_CONTAINERS = (HeadLogits, BranchLogits, FusedLogits, CalibratedLogits)


def argmax_rows(values: Matrix) -> tuple[int, ...]:
    """Index of the largest logit per row; the FIRST maximum wins a tie."""
    return tuple(max(range(len(row)), key=lambda c, r=row: (r[c], -c)) for row in values)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def _linear(terms: Sequence[tuple[float, Matrix]]) -> Matrix:
    rows, cols = len(terms[0][1]), len(terms[0][1][0])
    return tuple(
        tuple(sum(weight * matrix[r][c] for weight, matrix in terms) for c in range(cols))
        for r in range(rows)
    )


def _require_type(value: Any, expected: type, what: str) -> None:
    if type(value) is not expected:
        raise ViUnMarkContractError(
            f"{what} must be {expected.__name__}, got {type(value).__name__}. Fusion consumes "
            "RAW within-branch means only: a single head, a fused node or a calibrated "
            "output is refused here."
        )


def _require_aligned(parts: Sequence[Any], what: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    first = parts[0]
    for other in parts[1:]:
        if other.row_ids != first.row_ids:
            raise ViUnMarkContractError(
                f"{what}: inputs cover different rows or a different row order"
            )
        if other.num_labels != first.num_labels:
            raise ViUnMarkContractError(f"{what}: inputs have different numbers of classes")
        if set(other.head_seeds) != set(first.head_seeds) or len(other.head_seeds) != len(
            first.head_seeds
        ):
            raise ViUnMarkContractError(
                f"{what}: branches were ensembled over different head seed sets "
                f"({sorted(first.head_seeds)} vs {sorted(other.head_seeds)}); every branch "
                "must average the same heads"
            )
    return first.head_seeds, first.row_ids


# ---------------------------------------------------------------------------
# Within-branch ensemble
# ---------------------------------------------------------------------------
def ensemble_branch(
    heads: Sequence[HeadLogits], *, expected_seeds: Sequence[int]
) -> BranchLogits:
    """Average the logits of a branch's heads. **Every expected head, no selection.**

    Refuses a missing head, an extra head, a duplicated seed, heads from more than
    one branch, and heads that disagree on rows or classes. The mean is equal-weight
    over exactly `expected_seeds`; there is no argument that could pick a subset.
    """
    expected = tuple(expected_seeds)
    for seed in expected:
        _seed(seed, "expected_seeds")
    if not expected or len(set(expected)) != len(expected):
        raise ViUnMarkContractError(
            f"expected_seeds must be non-empty and unique, got {expected!r}"
        )
    if not heads:
        raise ViUnMarkContractError("ensemble_branch: no heads supplied")
    for head in heads:
        _require_type(head, HeadLogits, "ensemble_branch input")
    streams = {head.stream for head in heads}
    if len(streams) != 1:
        raise ViUnMarkContractError(
            f"ensemble_branch: heads come from several branches {sorted(s.value for s in streams)}"
        )
    seeds = [head.seed for head in heads]
    if len(set(seeds)) != len(seeds):
        raise ViUnMarkContractError(f"ensemble_branch: duplicated head seed in {seeds}")
    if set(seeds) != set(expected):
        missing = sorted(set(expected) - set(seeds))
        extra = sorted(set(seeds) - set(expected))
        raise ViUnMarkContractError(
            f"ensemble_branch: head seeds do not match the branch's head set "
            f"(missing {missing}, unexpected {extra}). Every head is averaged; none is "
            "selected or dropped."
        )
    first = heads[0]
    for head in heads[1:]:
        if head.row_ids != first.row_ids:
            raise ViUnMarkContractError("ensemble_branch: heads cover different rows or order")
        if head.num_labels != first.num_labels:
            raise ViUnMarkContractError("ensemble_branch: heads have different class counts")
    ordered = sorted(heads, key=lambda head: expected.index(head.seed))
    weight = 1.0 / len(ordered)
    mean = tuple(
        tuple(
            sum(head.values[r][c] for head in ordered) * weight
            for c in range(first.num_labels)
        )
        for r in range(len(first.row_ids))
    )
    return BranchLogits(
        stream=first.stream, head_seeds=expected, row_ids=first.row_ids, values=mean
    )


# ---------------------------------------------------------------------------
# Hierarchical raw fusion
# ---------------------------------------------------------------------------
def _require_stream(branch: BranchLogits, stream: LogitStream, what: str) -> None:
    _require_type(branch, BranchLogits, what)
    if branch.stream is not stream:
        raise ViUnMarkContractError(
            f"{what} must be the {stream.value} branch, got {branch.stream.value}"
        )


def fuse_scale(unweighted: BranchLogits, weighted: BranchLogits) -> FusedLogits:
    """`scale_raw = 0.5 * scale_unweighted_raw + 0.5 * scale_weighted_raw`."""
    _require_stream(unweighted, LogitStream.SCALE_UNWEIGHTED_READOUT, "unweighted")
    _require_stream(weighted, LogitStream.SCALE_WEIGHTED_READOUT, "weighted")
    seeds, rows = _require_aligned((unweighted, weighted), "fuse_scale")
    values = _linear((
        (SCALE_FUSION_WEIGHTS[LogitStream.SCALE_UNWEIGHTED_READOUT], unweighted.values),
        (SCALE_FUSION_WEIGHTS[LogitStream.SCALE_WEIGHTED_READOUT], weighted.values),
    ))
    return FusedLogits(node=FusionNode.SCALE_RAW, head_seeds=seeds, row_ids=rows, values=values)


def fuse_adapted_only_raw(
    gate: BranchLogits, scale_unweighted: BranchLogits, scale_weighted: BranchLogits
) -> FusedLogits:
    """`adapted_raw = 0.5 * gate_raw + 0.5 * scale_raw`."""
    _require_stream(gate, LogitStream.GATE_ROBUST_READOUT, "gate")
    scale = fuse_scale(scale_unweighted, scale_weighted)
    seeds, rows = _require_aligned((gate, scale_unweighted, scale_weighted), "fuse_adapted_only_raw")
    values = _linear((
        (ADAPTED_ONLY_GATE_WEIGHT, gate.values),
        (ADAPTED_ONLY_SCALE_WEIGHT, scale.values),
    ))
    return FusedLogits(
        node=FusionNode.ADAPTED_ONLY_RAW, head_seeds=seeds, row_ids=rows, values=values
    )


def fuse_viunmark_raw(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
) -> FusedLogits:
    """`viunmark_raw = 0.5 * adapted_raw + 0.5 * phobert_raw`, from RAW branches.

    `adapted_raw` is recomputed here from the three RAW adapted branches. There is
    no parameter through which a calibrated Adapted-Only output, or a calibrated
    standalone PhoBERT output, could enter.
    """
    _require_stream(phobert, LogitStream.PHOBERT_READOUT, "phobert")
    adapted = fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted)
    seeds, rows = _require_aligned(
        (gate, scale_unweighted, scale_weighted, phobert), "fuse_viunmark_raw"
    )
    values = _linear((
        (VIUNMARK_ADAPTED_WEIGHT, adapted.values),
        (VIUNMARK_PHOBERT_WEIGHT, phobert.values),
    ))
    return FusedLogits(node=FusionNode.VIUNMARK_RAW, head_seeds=seeds, row_ids=rows, values=values)


# ---------------------------------------------------------------------------
# Calibration -- once, at the end
# ---------------------------------------------------------------------------
def apply_system_calibration(raw: FusedLogits, calibration: CalibrationConfig) -> CalibratedLogits:
    """Add the system's single calibration bias to an UNCALIBRATED system output.

    Only `ADAPTED_ONLY_RAW` and `VIUNMARK_RAW` are system outputs. A branch, a
    head, the internal `SCALE_RAW` node, or an already calibrated output is
    refused, so no bias can be applied twice or to a parent.
    """
    _require_type(raw, FusedLogits, "apply_system_calibration input")
    if raw.node not in _SYSTEM_FOR_NODE:
        raise ViUnMarkContractError(
            f"{raw.node.value} is an internal fusion node, not a system output; it is never "
            "calibrated"
        )
    if not isinstance(calibration, CalibrationConfig):
        raise ViUnMarkContractError("calibration must be a CalibrationConfig")
    bias = calibration.bias_vector(raw.num_labels)
    values = tuple(tuple(v + b for v, b in zip(row, bias)) for row in raw.values)
    return CalibratedLogits(
        system=_SYSTEM_FOR_NODE[raw.node],
        head_seeds=raw.head_seeds,
        row_ids=raw.row_ids,
        values=values,
        calibration=calibration,
    )


def adapted_only_fusion(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    calibration: CalibrationConfig,
) -> CalibratedLogits:
    """Adapted-Only Fusion (diagnostic): raw fusion, then its own calibration."""
    return apply_system_calibration(
        fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted), calibration
    )


def viunmark_fusion(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
    calibration: CalibrationConfig,
) -> CalibratedLogits:
    """ViUnMark: raw fusion of the four RAW branches, then ONE calibration."""
    return apply_system_calibration(
        fuse_viunmark_raw(gate, scale_unweighted, scale_weighted, phobert), calibration
    )


__all__ = [
    "ADAPTED_ONLY_GATE_WEIGHT",
    "ADAPTED_ONLY_SCALE_WEIGHT",
    "BEST_SEED_SELECTION",
    "BranchLogits",
    "CalibratedLogits",
    "FusedLogits",
    "FusionNode",
    "HeadLogits",
    "PARENT_BIASES_STACKED",
    "SCALE_FUSION_WEIGHTS",
    "SystemKind",
    "VIUNMARK_ADAPTED_WEIGHT",
    "VIUNMARK_PHOBERT_WEIGHT",
    "WITHIN_BRANCH_LOGIT_ENSEMBLE",
    "adapted_only_fusion",
    "apply_system_calibration",
    "argmax_rows",
    "ensemble_branch",
    "expanded_adapted_only_weights",
    "expanded_per_head_weights",
    "expanded_viunmark_weights",
    "fuse_adapted_only_raw",
    "fuse_scale",
    "fuse_viunmark_raw",
    "viunmark_fusion",
]
