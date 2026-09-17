"""Robustness gain factorization over final-system nodes."""

from __future__ import annotations

from enum import Enum

from viunmark.config import CalibrationConfig
from viunmark.fusion import (
    BranchLogits,
    CalibratedLogits,
    FusedLogits,
    apply_system_calibration,
    fuse_adapted_only_raw,
    fuse_viunmark_raw,
)


class RobustnessFactor(Enum):
    COMMON_MATCHED_PAIR = "common_matched_pair"
    GATE_SPECIALIZATION = "gate_specialization"
    SCALE_DUAL_READOUT = "scale_dual_readout"
    ADAPTED_ONLY_RAW = "adapted_only_raw"
    ADAPTED_ONLY_CALIBRATED = "adapted_only_calibrated"
    ADD_NATIVE_PHOBERT = "add_native_phobert"
    VIUNMARK_CALIBRATED = "viunmark_calibrated"


FACTORIZATION_ORDER = tuple(RobustnessFactor)
UNRESOLVED_FACTORS = FACTORIZATION_ORDER[:3]
COMPUTABLE_FACTORS = FACTORIZATION_ORDER[3:]


def final_system_factors(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
    *,
    adapted_only_calibration: CalibrationConfig,
    viunmark_calibration: CalibrationConfig,
) -> dict[RobustnessFactor, FusedLogits | CalibratedLogits]:
    adapted_raw = fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted)
    viunmark_raw = fuse_viunmark_raw(gate, scale_unweighted, scale_weighted, phobert)
    return {
        RobustnessFactor.ADAPTED_ONLY_RAW: adapted_raw,
        RobustnessFactor.ADAPTED_ONLY_CALIBRATED: apply_system_calibration(adapted_raw, adapted_only_calibration),
        RobustnessFactor.ADD_NATIVE_PHOBERT: viunmark_raw,
        RobustnessFactor.VIUNMARK_CALIBRATED: apply_system_calibration(viunmark_raw, viunmark_calibration),
    }
