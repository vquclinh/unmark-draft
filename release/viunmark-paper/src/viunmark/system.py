"""Public system handles."""

from __future__ import annotations

from typing import Mapping, Sequence

from viunmark.adapters import build_gate_adapter, build_scale_adapter
from viunmark.config import AdaptedOnlyFusionConfig, LogitStream, ViUnMarkConfig, ViUnMarkContractError
from viunmark.fusion import (
    BranchLogits,
    CalibratedLogits,
    HeadLogits,
    adapted_only_fusion,
    ensemble_branch,
    viunmark_fusion,
)


class ViUnMarkGate:
    """ViUnMark-Gate adapter factory."""

    @staticmethod
    def build_adapter():
        return build_gate_adapter()


class ViUnMarkScale:
    """ViUnMark-Scale adapter factory."""

    @staticmethod
    def build_adapter():
        return build_scale_adapter()


def _branch_ensembles(
    head_logits: Mapping[LogitStream, Sequence[HeadLogits]],
    streams: Sequence[LogitStream],
    head_seeds: Sequence[int],
) -> dict[LogitStream, BranchLogits]:
    if set(head_logits) != set(streams):
        raise ViUnMarkContractError("head logits do not cover exactly the configured streams")
    return {stream: ensemble_branch(list(head_logits[stream]), expected_seeds=head_seeds) for stream in streams}


class ViUnMark:
    """The proposed system: four branch means, fixed raw-logit fusion, one calibration."""

    def __init__(self, config: ViUnMarkConfig) -> None:
        if not isinstance(config, ViUnMarkConfig):
            raise ViUnMarkContractError("config must be a ViUnMarkConfig")
        self.config = config

    def branch_ensembles(self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]) -> dict[LogitStream, BranchLogits]:
        return _branch_ensembles(head_logits, self.config.streams, self.config.head_seeds)

    def predict_logits(self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]) -> CalibratedLogits:
        branches = self.branch_ensembles(head_logits)
        return viunmark_fusion(
            branches[LogitStream.GATE_ROBUST_READOUT],
            branches[LogitStream.SCALE_UNWEIGHTED_READOUT],
            branches[LogitStream.SCALE_WEIGHTED_READOUT],
            branches[LogitStream.PHOBERT_READOUT],
            self.config.calibration,
        )


class AdaptedOnlyFusion:
    """Diagnostic system over adapted branches only."""

    def __init__(self, config: AdaptedOnlyFusionConfig) -> None:
        if not isinstance(config, AdaptedOnlyFusionConfig):
            raise ViUnMarkContractError("config must be an AdaptedOnlyFusionConfig")
        self.config = config

    def predict_logits(self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]) -> CalibratedLogits:
        branches = _branch_ensembles(head_logits, self.config.streams, self.config.head_seeds)
        return adapted_only_fusion(
            branches[LogitStream.GATE_ROBUST_READOUT],
            branches[LogitStream.SCALE_UNWEIGHTED_READOUT],
            branches[LogitStream.SCALE_WEIGHTED_READOUT],
            self.config.calibration,
        )
