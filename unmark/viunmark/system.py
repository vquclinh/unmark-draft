"""ViUnMark pathways and systems.

`ViUnMarkGate` and `ViUnMarkScale` are thin, public-facing handles over the
repository's accepted Stage-I adapter. They do not re-implement it: building
goes through `unmark.stage1.initialisation.fresh_adapter`, and loading goes
through `unmark.stage1.reconstruct`, which reads the fusion rule a checkpoint
records and loads its tensors with `strict=True`. Loading refuses a checkpoint
whose recorded fusion is the other pathway's. The two adapters have identical
tensor shapes, so without that check the weights would load silently into the
wrong equation.

`ViUnMark` and `AdaptedOnlyFusion` turn per-head RAW logits into a system output:
average each branch's heads, fuse the branches, then calibrate once. Both are
pure over logits; neither loads a model.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.reconstruct import reconstruct_adapter, require_loadable_as
from unmark.viunmark.config import (
    AdaptedOnlyFusionConfig,
    LogitStream,
    ViUnMarkConfig,
    ViUnMarkContractError,
    ViUnMarkGateConfig,
    ViUnMarkScaleConfig,
    _AdaptedPathwayConfig,
)
from unmark.viunmark.fusion import (
    BranchLogits,
    CalibratedLogits,
    HeadLogits,
    adapted_only_fusion,
    ensemble_branch,
    viunmark_fusion,
)


# ---------------------------------------------------------------------------
# Stage-I pathways
# ---------------------------------------------------------------------------
class _AdaptedPathway:
    CONFIG: ClassVar[_AdaptedPathwayConfig]

    @classmethod
    def config(cls) -> _AdaptedPathwayConfig:
        return cls.CONFIG

    @classmethod
    def build_adapter(cls, init_seed: int) -> Any:
        """A fresh adapter for this pathway, initialised on CPU from `init_seed`."""
        from unmark.stage1.initialisation import fresh_adapter

        if isinstance(init_seed, bool) or not isinstance(init_seed, int):
            raise ViUnMarkContractError(f"init_seed must be an int, got {init_seed!r}")
        config = cls.CONFIG
        return fresh_adapter(config.hidden_size, init_seed, config.fusion_id)

    @classmethod
    def load_adapter(cls, payload: Mapping[str, Any]) -> Any:
        """Rebuild this pathway's adapter from a checkpoint payload. Fails closed.

        Refuses unless the payload's recorded fusion is this pathway's, then builds
        through the recorded fusion and loads with `strict=True`.
        """
        config = cls.CONFIG
        try:
            require_loadable_as(payload, config.fusion_id)
        except Stage1ContractViolation as error:
            raise ViUnMarkContractError(
                f"this checkpoint is not a {config.pathway.value} adapter: {error}"
            ) from error
        adapter = reconstruct_adapter(payload, config.hidden_size)
        built = getattr(getattr(adapter, "config", None), "fusion_id", None)
        if built != config.fusion_id:
            raise ViUnMarkContractError(
                f"reconstructed adapter implements {built!r}, expected {config.fusion_id!r}"
            )
        return adapter


class ViUnMarkGate(_AdaptedPathway):
    """ViUnMark-Gate: `z = g * f + (1 - g) * e`."""

    CONFIG: ClassVar[ViUnMarkGateConfig] = ViUnMarkGateConfig()


class ViUnMarkScale(_AdaptedPathway):
    """ViUnMark-Scale: `z = g * (scale * f) + (1 - g) * e`, `scale = ||e|| / max(||f||, 1e-8)`."""

    CONFIG: ClassVar[ViUnMarkScaleConfig] = ViUnMarkScaleConfig()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def _branch_ensembles(
    head_logits: Mapping[LogitStream, Sequence[HeadLogits]],
    streams: Sequence[LogitStream],
    head_seeds: Sequence[int],
) -> dict[LogitStream, BranchLogits]:
    supplied = set(head_logits)
    expected = set(streams)
    if supplied != expected:
        raise ViUnMarkContractError(
            f"expected head logits for exactly {sorted(s.value for s in expected)}, got "
            f"{sorted(s.value for s in supplied)}"
        )
    ensembles = {}
    for stream in streams:
        heads = list(head_logits[stream])
        wrong = sorted({h.stream.value for h in heads if h.stream is not stream})
        if wrong:
            raise ViUnMarkContractError(
                f"heads filed under {stream.value} belong to {wrong}"
            )
        ensembles[stream] = ensemble_branch(heads, expected_seeds=head_seeds)
    return ensembles


class ViUnMark:
    """The proposed system: a fixed-weight fusion of four branch ensembles.

    One output per call. The heads inside each branch are averaged first; the
    system is never evaluated head by head or seed by seed.
    """

    def __init__(self, config: ViUnMarkConfig) -> None:
        if not isinstance(config, ViUnMarkConfig):
            raise ViUnMarkContractError("config must be a ViUnMarkConfig")
        self.config = config

    def branch_ensembles(
        self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]
    ) -> dict[LogitStream, BranchLogits]:
        return _branch_ensembles(head_logits, self.config.streams, self.config.head_seeds)

    def predict_logits(
        self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]
    ) -> CalibratedLogits:
        branches = self.branch_ensembles(head_logits)
        return viunmark_fusion(
            branches[LogitStream.GATE_ROBUST_READOUT],
            branches[LogitStream.SCALE_UNWEIGHTED_READOUT],
            branches[LogitStream.SCALE_WEIGHTED_READOUT],
            branches[LogitStream.PHOBERT_READOUT],
            self.config.calibration,
        )


class AdaptedOnlyFusion:
    """Adapted-Only Fusion. **Diagnostic / ablation, not the proposed system.**"""

    def __init__(self, config: AdaptedOnlyFusionConfig) -> None:
        if not isinstance(config, AdaptedOnlyFusionConfig):
            raise ViUnMarkContractError("config must be an AdaptedOnlyFusionConfig")
        self.config = config

    def predict_logits(
        self, head_logits: Mapping[LogitStream, Sequence[HeadLogits]]
    ) -> CalibratedLogits:
        branches = _branch_ensembles(head_logits, self.config.streams, self.config.head_seeds)
        return adapted_only_fusion(
            branches[LogitStream.GATE_ROBUST_READOUT],
            branches[LogitStream.SCALE_UNWEIGHTED_READOUT],
            branches[LogitStream.SCALE_WEIGHTED_READOUT],
            self.config.calibration,
        )


__all__ = ["AdaptedOnlyFusion", "ViUnMark", "ViUnMarkGate", "ViUnMarkScale"]
