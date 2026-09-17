"""ViUnMark logit ensembling, fusion and calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from viunmark.config import CalibrationConfig, LogitStream, ViUnMarkContractError

Matrix = tuple[tuple[float, ...], ...]

SCALE_FUSION_WEIGHTS = {
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
    return {
        LogitStream.GATE_ROBUST_READOUT: ADAPTED_ONLY_GATE_WEIGHT,
        LogitStream.SCALE_UNWEIGHTED_READOUT: 0.25,
        LogitStream.SCALE_WEIGHTED_READOUT: 0.25,
    }


def expanded_viunmark_weights() -> dict[LogitStream, float]:
    return {
        LogitStream.PHOBERT_READOUT: VIUNMARK_PHOBERT_WEIGHT,
        LogitStream.GATE_ROBUST_READOUT: 0.25,
        LogitStream.SCALE_UNWEIGHTED_READOUT: 0.125,
        LogitStream.SCALE_WEIGHTED_READOUT: 0.125,
    }


def expanded_per_head_weights(
    branch_weights: Mapping[LogitStream, float], heads_per_branch: int
) -> dict[LogitStream, float]:
    if isinstance(heads_per_branch, bool) or not isinstance(heads_per_branch, int) or heads_per_branch < 1:
        raise ViUnMarkContractError("heads_per_branch must be a positive int")
    return {stream: weight / heads_per_branch for stream, weight in branch_weights.items()}


class FusionNode(Enum):
    SCALE_RAW = "scale_raw"
    ADAPTED_ONLY_RAW = "adapted_only_raw"
    VIUNMARK_RAW = "viunmark_raw"


class SystemKind(Enum):
    ADAPTED_ONLY_FUSION = "adapted_only_fusion"
    VIUNMARK = "viunmark"


_SYSTEM_FOR_NODE = {
    FusionNode.ADAPTED_ONLY_RAW: SystemKind.ADAPTED_ONLY_FUSION,
    FusionNode.VIUNMARK_RAW: SystemKind.VIUNMARK,
}
_LOGIT_CONTAINERS: tuple[type, ...] = ()


def _matrix(values: Any, what: str) -> Matrix:
    if isinstance(values, _LOGIT_CONTAINERS):
        raise ViUnMarkContractError(f"{what}: pass raw numbers, not {type(values).__name__}")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ViUnMarkContractError(f"{what}: logits must be a matrix")
    rows: list[tuple[float, ...]] = []
    width = None
    for index, row in enumerate(values):
        if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
            raise ViUnMarkContractError(f"{what}: row {index} is not a sequence")
        cells = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)) or not math.isfinite(cell):
                raise ViUnMarkContractError(f"{what}: row {index} has invalid cell {cell!r}")
            cells.append(float(cell))
        if width is None:
            width = len(cells)
        if len(cells) != width:
            raise ViUnMarkContractError(f"{what}: ragged logits")
        rows.append(tuple(cells))
    if not rows or width is None or width < 2:
        raise ViUnMarkContractError(f"{what}: logits need rows and at least two classes")
    return tuple(rows)


def _row_ids(value: Any, count: int, what: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(v, str) and v for v in value):
        raise ViUnMarkContractError(f"{what}: row_ids must be a tuple of non-empty strings")
    if len(value) != count or len(set(value)) != len(value):
        raise ViUnMarkContractError(f"{what}: row_ids count or uniqueness violation")
    return value


def _seed(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ViUnMarkContractError(f"{what}: seed must be an int")
    return value


@dataclass(frozen=True)
class HeadLogits:
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
    stream: LogitStream
    head_seeds: tuple[int, ...]
    row_ids: tuple[str, ...]
    values: Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.stream, LogitStream):
            raise ViUnMarkContractError("stream must be a LogitStream")
        if not isinstance(self.head_seeds, tuple) or not self.head_seeds:
            raise ViUnMarkContractError("head_seeds must be a non-empty tuple")
        if len(set(self.head_seeds)) != len(self.head_seeds):
            raise ViUnMarkContractError("head_seeds contain duplicates")
        for seed in self.head_seeds:
            _seed(seed, "BranchLogits")
        object.__setattr__(self, "values", _matrix(self.values, "BranchLogits"))
        _row_ids(self.row_ids, len(self.values), "BranchLogits")

    @property
    def num_labels(self) -> int:
        return len(self.values[0])


@dataclass(frozen=True)
class FusedLogits:
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
    return tuple(max(range(len(row)), key=lambda c, r=row: (r[c], -c)) for row in values)


def _linear(terms: Sequence[tuple[float, Matrix]]) -> Matrix:
    rows, cols = len(terms[0][1]), len(terms[0][1][0])
    return tuple(
        tuple(sum(weight * matrix[r][c] for weight, matrix in terms) for c in range(cols))
        for r in range(rows)
    )


def _require_aligned(parts: Sequence[Any], what: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    first = parts[0]
    for other in parts[1:]:
        if other.row_ids != first.row_ids or other.num_labels != first.num_labels:
            raise ViUnMarkContractError(f"{what}: inputs are not row/class aligned")
        if set(other.head_seeds) != set(first.head_seeds) or len(other.head_seeds) != len(first.head_seeds):
            raise ViUnMarkContractError(f"{what}: branches use different head seed sets")
    return first.head_seeds, first.row_ids


def ensemble_branch(heads: Sequence[HeadLogits], *, expected_seeds: Sequence[int]) -> BranchLogits:
    expected = tuple(expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ViUnMarkContractError("expected_seeds must be non-empty and unique")
    if not heads:
        raise ViUnMarkContractError("ensemble_branch: no heads supplied")
    if any(type(head) is not HeadLogits for head in heads):
        raise ViUnMarkContractError("ensemble_branch takes HeadLogits")
    streams = {head.stream for head in heads}
    if len(streams) != 1:
        raise ViUnMarkContractError("ensemble_branch: heads come from several branches")
    seeds = [head.seed for head in heads]
    if len(set(seeds)) != len(seeds):
        raise ViUnMarkContractError("ensemble_branch: duplicated head seed")
    if set(seeds) != set(expected):
        raise ViUnMarkContractError("ensemble_branch: missing or unexpected head seed")
    first = heads[0]
    for head in heads[1:]:
        if head.row_ids != first.row_ids or head.num_labels != first.num_labels:
            raise ViUnMarkContractError("ensemble_branch: heads are not aligned")
    ordered = sorted(heads, key=lambda head: expected.index(head.seed))
    weight = 1.0 / len(ordered)
    values = tuple(
        tuple(sum(head.values[r][c] for head in ordered) * weight for c in range(first.num_labels))
        for r in range(len(first.row_ids))
    )
    return BranchLogits(first.stream, expected, first.row_ids, values)


def _require_stream(branch: BranchLogits, stream: LogitStream, what: str) -> None:
    if type(branch) is not BranchLogits or branch.stream is not stream:
        got = getattr(getattr(branch, "stream", None), "value", type(branch).__name__)
        raise ViUnMarkContractError(f"{what} must be {stream.value}, got {got}")


def fuse_scale(unweighted: BranchLogits, weighted: BranchLogits) -> FusedLogits:
    _require_stream(unweighted, LogitStream.SCALE_UNWEIGHTED_READOUT, "unweighted")
    _require_stream(weighted, LogitStream.SCALE_WEIGHTED_READOUT, "weighted")
    seeds, rows = _require_aligned((unweighted, weighted), "fuse_scale")
    values = _linear(((0.5, unweighted.values), (0.5, weighted.values)))
    return FusedLogits(FusionNode.SCALE_RAW, seeds, rows, values)


def fuse_adapted_only_raw(
    gate: BranchLogits, scale_unweighted: BranchLogits, scale_weighted: BranchLogits
) -> FusedLogits:
    _require_stream(gate, LogitStream.GATE_ROBUST_READOUT, "gate")
    scale = fuse_scale(scale_unweighted, scale_weighted)
    seeds, rows = _require_aligned((gate, scale_unweighted, scale_weighted), "fuse_adapted_only_raw")
    values = _linear(((ADAPTED_ONLY_GATE_WEIGHT, gate.values), (ADAPTED_ONLY_SCALE_WEIGHT, scale.values)))
    return FusedLogits(FusionNode.ADAPTED_ONLY_RAW, seeds, rows, values)


def fuse_viunmark_raw(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
) -> FusedLogits:
    _require_stream(phobert, LogitStream.PHOBERT_READOUT, "phobert")
    adapted = fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted)
    seeds, rows = _require_aligned((gate, scale_unweighted, scale_weighted, phobert), "fuse_viunmark_raw")
    values = _linear(((VIUNMARK_ADAPTED_WEIGHT, adapted.values), (VIUNMARK_PHOBERT_WEIGHT, phobert.values)))
    return FusedLogits(FusionNode.VIUNMARK_RAW, seeds, rows, values)


def apply_system_calibration(raw: FusedLogits, calibration: CalibrationConfig) -> CalibratedLogits:
    if type(raw) is not FusedLogits:
        raise ViUnMarkContractError("calibration applies only to uncalibrated fused logits")
    if raw.node not in _SYSTEM_FOR_NODE:
        raise ViUnMarkContractError("internal fusion nodes are not calibrated")
    if not isinstance(calibration, CalibrationConfig):
        raise ViUnMarkContractError("calibration must be a CalibrationConfig")
    bias = calibration.bias_vector(raw.num_labels)
    values = tuple(tuple(v + b for v, b in zip(row, bias)) for row in raw.values)
    return CalibratedLogits(_SYSTEM_FOR_NODE[raw.node], raw.head_seeds, raw.row_ids, values, calibration)


def adapted_only_fusion(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    calibration: CalibrationConfig,
) -> CalibratedLogits:
    return apply_system_calibration(fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted), calibration)


def viunmark_fusion(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
    calibration: CalibrationConfig,
) -> CalibratedLogits:
    return apply_system_calibration(fuse_viunmark_raw(gate, scale_unweighted, scale_weighted, phobert), calibration)
