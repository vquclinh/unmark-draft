"""Machine-readable provenance loaders for the publication export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from viunmark.config import (
    FINAL_BRANCH_STREAMS,
    SIX_CONDITIONS,
    AdaptedOnlyFusionConfig,
    CalibrationConfig,
    LogitStream,
    ViUnMarkConfig,
    ViUnMarkContractError,
)
from viunmark.losses import sqrt_inverse_frequency_weights

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_SYSTEM_SPEC_PATH = REPO_ROOT / "artifacts/historical/configs/viunmark-final-system-v1.json"
HISTORICAL_ALIAS_SPEC_PATH = REPO_ROOT / "artifacts/provenance/historical_aliases.json"
REPRODUCTION_MANIFEST_PATH = REPO_ROOT / "artifacts/provenance/reproduction_manifest.json"


def _load(path: Path, schema_version: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ViUnMarkContractError(f"missing provenance file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if schema_version is not None and payload.get("schema_version") != schema_version:
        raise ViUnMarkContractError(f"{path.name} schema is {payload.get('schema_version')!r}")
    return payload


def load_final_system_spec() -> dict[str, Any]:
    return _load(FINAL_SYSTEM_SPEC_PATH, "viunmark-final-system-v1")


def load_historical_aliases() -> dict[str, Any]:
    payload = _load(HISTORICAL_ALIAS_SPEC_PATH, "viunmark-historical-aliases-v1")
    if payload.get("public_api") is not False:
        raise ViUnMarkContractError("historical aliases must not be public API")
    return payload


def load_reproduction_manifest() -> dict[str, Any]:
    return _load(REPRODUCTION_MANIFEST_PATH, "viunmark-reproduction-manifest-v1")


@dataclass(frozen=True)
class SelectedHead:
    stream: LogitStream
    seed: int
    selected_boundary: int
    sha256: str


@dataclass(frozen=True)
class UitVsfcReproduction:
    label_names: tuple[str, ...]
    head_seeds: tuple[int, ...]
    protocol_train_class_counts: tuple[int, ...]
    weight_vector: tuple[float, ...]
    selected_heads: Mapping[LogitStream, tuple[SelectedHead, ...]]
    adapted_only_calibration: CalibrationConfig
    viunmark_calibration: CalibrationConfig
    artifact_lineage: Mapping[str, str]
    protocol_lineage: Mapping[str, str]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    @property
    def head_count(self) -> int:
        return sum(len(heads) for heads in self.selected_heads.values())

    def viunmark_config(self) -> ViUnMarkConfig:
        return ViUnMarkConfig(self.num_labels, self.head_seeds, self.viunmark_calibration)

    def adapted_only_fusion_config(self) -> AdaptedOnlyFusionConfig:
        return AdaptedOnlyFusionConfig(self.num_labels, self.head_seeds, self.adapted_only_calibration)


def _calibration(block: Mapping[str, Any], labels: tuple[str, ...]) -> CalibrationConfig:
    cal = CalibrationConfig(block["class_index"], block["additive_logit_bias"], block["label_name"])
    if labels[cal.class_index] != cal.label_name:
        raise ViUnMarkContractError("calibration label/index mismatch")
    return cal


def load_uit_vsfc_reproduction() -> UitVsfcReproduction:
    spec = load_final_system_spec()
    record = spec["uit_vsfc_reproduction"]
    labels = tuple(record["label_names"])
    head_seeds = tuple(record["head_seeds"])
    if tuple(record["corruption"]["conditions"]) != SIX_CONDITIONS:
        raise ViUnMarkContractError("condition order drifted")
    counts_block = record["weighted_loss"]["protocol_train_class_counts"]
    counts = tuple(counts_block[name] for name in labels)
    weights = tuple(record["weighted_loss"]["weight_vector"])
    if weights != sqrt_inverse_frequency_weights(counts, arithmetic="float32"):
        raise ViUnMarkContractError("recorded UIT-VSFC weights do not match the reusable rule")
    selected: dict[LogitStream, tuple[SelectedHead, ...]] = {}
    for stream in FINAL_BRANCH_STREAMS:
        heads = tuple(
            SelectedHead(stream, item["seed"], item["selected_boundary"], item["sha256"])
            for item in record["selected_heads"][stream.value]
        )
        if tuple(h.seed for h in heads) != head_seeds:
            raise ViUnMarkContractError(f"{stream.value} heads do not cover the shared seed set")
        selected[stream] = heads
    cal = record["calibration"]
    reproduction = UitVsfcReproduction(
        label_names=labels,
        head_seeds=head_seeds,
        protocol_train_class_counts=counts,
        weight_vector=weights,
        selected_heads=selected,
        adapted_only_calibration=_calibration(cal["adapted_only_fusion"], labels),
        viunmark_calibration=_calibration(cal["viunmark"], labels),
        artifact_lineage=dict(record["artifact_lineage"]),
        protocol_lineage=dict(record["protocol_lineage"]),
    )
    if reproduction.head_count != 20 or reproduction.viunmark_config().head_count != 20:
        raise ViUnMarkContractError("ViUnMark must be a 20-head ensemble")
    standalone = cal["phobert_readout_standalone_development_stage"]
    if standalone["inherited_by_viunmark"] is not False:
        raise ViUnMarkContractError("standalone PhoBERT calibration must not be inherited")
    return reproduction
