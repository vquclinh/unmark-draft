"""Robustness Gain Factorization. **Torch-free.**

Attributes the robustness gain to successive steps, in this fixed order:

    1. common matched pair
    2. Gate specialization
    3. Scale dual readout
    4. Adapted-Only Fusion, raw
    5. Adapted-Only Fusion, calibrated
    6. add native PhoBERT
    7. ViUnMark, calibrated

Implemented here: steps 4-7, which are nodes of the final system itself.

* Step 6 is the uncalibrated ViUnMark fusion: PhoBERT is added to the RAW adapted
  fusion.
* Step 7 calibrates that raw output once.

The step-5 bias is never carried into steps 6-7, because parent biases do not stack.

Not implemented, deliberately: steps 1-3. Which branches form the "common matched
pair", and exactly what each of the next two steps substitutes into it, are not
recorded precisely enough to compute them without guessing.
"""

from __future__ import annotations

from enum import Enum

from unmark.viunmark.config import CalibrationConfig
from unmark.viunmark.fusion import (
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


FACTORIZATION_ORDER: tuple[RobustnessFactor, ...] = tuple(RobustnessFactor)
UNRESOLVED_FACTORS: tuple[RobustnessFactor, ...] = (
    RobustnessFactor.COMMON_MATCHED_PAIR,
    RobustnessFactor.GATE_SPECIALIZATION,
    RobustnessFactor.SCALE_DUAL_READOUT,
)
COMPUTABLE_FACTORS: tuple[RobustnessFactor, ...] = tuple(
    factor for factor in FACTORIZATION_ORDER if factor not in UNRESOLVED_FACTORS
)


def final_system_factors(
    gate: BranchLogits,
    scale_unweighted: BranchLogits,
    scale_weighted: BranchLogits,
    phobert: BranchLogits,
    *,
    adapted_only_calibration: CalibrationConfig,
    viunmark_calibration: CalibrationConfig,
) -> dict[RobustnessFactor, FusedLogits | CalibratedLogits]:
    """Steps 4-7 from the four RAW branch ensembles."""
    adapted_raw = fuse_adapted_only_raw(gate, scale_unweighted, scale_weighted)
    viunmark_raw = fuse_viunmark_raw(gate, scale_unweighted, scale_weighted, phobert)
    return {
        RobustnessFactor.ADAPTED_ONLY_RAW: adapted_raw,
        RobustnessFactor.ADAPTED_ONLY_CALIBRATED: apply_system_calibration(
            adapted_raw, adapted_only_calibration
        ),
        RobustnessFactor.ADD_NATIVE_PHOBERT: viunmark_raw,
        RobustnessFactor.VIUNMARK_CALIBRATED: apply_system_calibration(
            viunmark_raw, viunmark_calibration
        ),
    }


__all__ = [
    "COMPUTABLE_FACTORS",
    "FACTORIZATION_ORDER",
    "RobustnessFactor",
    "UNRESOLVED_FACTORS",
    "final_system_factors",
]
