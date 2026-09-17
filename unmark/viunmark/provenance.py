"""Recorded provenance: the final-system spec and dataset reproduction records.

Two committed files are read here:

* `docs/spec/viunmark-final-system-v1.json` holds the method, including the
  recovered head-training recipe, and the UIT-VSFC reproduction record:
  checkpoint and head identities, class counts, corruption identity, calibration
  values, and artifact and protocol lineage.
* `docs/spec/viunmark-historical-aliases-v1.json` maps old research labels to
  descriptive names. It is provenance only, never an API.

Everything dataset-specific is exposed under an explicit `uit_vsfc_` name, so
reusable code cannot inherit a UIT-VSFC value by accident. The loader also checks
the record against the method definitions in `config`, `training` and `inputs`.
The spec and the code therefore cannot drift apart silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from unmark.viunmark.config import (
    FINAL_BRANCH_STREAMS,
    SIX_CONDITIONS,
    AdaptedOnlyFusionConfig,
    CalibrationConfig,
    LogitStream,
    ViUnMarkConfig,
    ViUnMarkContractError,
    ViUnMarkGateConfig,
    ViUnMarkScaleConfig,
)
from unmark.viunmark.inputs import FULL_CONDITION_API_SEED, CorruptionProtocol
from unmark.viunmark.losses import sqrt_inverse_frequency_weights
from unmark.viunmark.training import (
    LAYERNORM_BIAS_INIT,
    LAYERNORM_WEIGHT_INIT,
    LINEAR_BIAS_INIT,
    LINEAR_WEIGHT_INIT,
    PHOBERT_READOUT_NOT_ESTABLISHED,
    PHOBERT_READOUT_POLICY_CONFIRMED,
    RECIPE_EVIDENCE_SCOPE,
    ROBUST_MLP_OPTIMIZER_POLICY,
    ROBUST_MLP_TRAINING_BUDGET,
    TRAINING_AMP,
    TRAINING_DTYPE,
    TRAINING_TF32,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FINAL_SYSTEM_SPEC_PATH = REPOSITORY_ROOT / "docs/spec/viunmark-final-system-v1.json"
HISTORICAL_ALIAS_SPEC_PATH = REPOSITORY_ROOT / "docs/spec/viunmark-historical-aliases-v1.json"
FINAL_SYSTEM_SPEC_VERSION = "viunmark-final-system-v1"
HISTORICAL_ALIAS_SPEC_VERSION = "viunmark-historical-aliases-v1"

_PATHWAY_CONFIGS = {"viunmark_gate": ViUnMarkGateConfig, "viunmark_scale": ViUnMarkScaleConfig}


def _load(path: Path, version: str) -> dict[str, Any]:
    if not path.is_file():
        raise ViUnMarkContractError(f"missing spec: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != version:
        raise ViUnMarkContractError(
            f"{path.name} is {payload.get('schema_version')!r}, expected {version!r}"
        )
    return payload


def load_final_system_spec() -> dict[str, Any]:
    """The final-system spec, schema-checked."""
    return _load(FINAL_SYSTEM_SPEC_PATH, FINAL_SYSTEM_SPEC_VERSION)


def load_historical_aliases() -> dict[str, Any]:
    """The historical alias spec. **Provenance only; not public API.**"""
    payload = _load(HISTORICAL_ALIAS_SPEC_PATH, HISTORICAL_ALIAS_SPEC_VERSION)
    if payload.get("public_api") is not False:
        raise ViUnMarkContractError("the historical alias spec must declare public_api=false")
    return payload


@dataclass(frozen=True)
class SelectedHead:
    """One recorded head: its seed, recorded selection boundary, and file digest."""

    stream: LogitStream
    seed: int
    selected_boundary: int
    sha256: str


@dataclass(frozen=True)
class StageOneCheckpoint:
    """A recorded Stage-I checkpoint. Provenance, not a requirement for retraining."""

    pathway: str
    sha256: str
    selected_update: int
    fusion_id: str
    source_repository_head: str


@dataclass(frozen=True)
class UitVsfcReproduction:
    """The recorded UIT-VSFC reproduction facts for the final system.

    **Not a method default.** Another dataset trains its own heads, counts its own
    classes, freezes its own corruption seed, and decides its own calibration.
    """

    label_names: tuple[str, ...]
    head_seeds: tuple[int, ...]
    stage1_checkpoints: Mapping[str, StageOneCheckpoint]
    corruption: Mapping[str, Any]
    representation_bank: Mapping[str, Any]
    protocol_train_class_counts: tuple[int, ...]
    weight_vector: tuple[float, ...]
    protocol_train_rows: int
    augmented_rows: int
    selected_heads: Mapping[LogitStream, tuple[SelectedHead, ...]]
    total_stage2_parameters: int
    adapted_only_calibration: CalibrationConfig
    viunmark_calibration: CalibrationConfig
    phobert_standalone_development_calibration: CalibrationConfig
    phobert_standalone_calibration_inherited: bool
    stage2_head_training: Mapping[str, Any]
    system_selection: Mapping[str, Any]
    artifact_lineage: Mapping[str, str]
    protocol_lineage: Mapping[str, str]
    development_stages: Mapping[str, Any]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    @property
    def head_count(self) -> int:
        return sum(len(heads) for heads in self.selected_heads.values())

    @property
    def stage1_checkpoint_sha256(self) -> dict[str, str]:
        return {name: c.sha256 for name, c in self.stage1_checkpoints.items()}

    @property
    def scientific_corruption_seed(self) -> int:
        return self.corruption["scientific_corruption_seed"]

    def corruption_protocol(self) -> CorruptionProtocol:
        """The UIT-VSFC corruption protocol. Explicitly UIT-VSFC's, never a default."""
        return CorruptionProtocol(scientific_corruption_seed=self.scientific_corruption_seed)

    def viunmark_config(self) -> ViUnMarkConfig:
        return ViUnMarkConfig(
            num_labels=self.num_labels,
            head_seeds=self.head_seeds,
            calibration=self.viunmark_calibration,
        )

    def adapted_only_fusion_config(self) -> AdaptedOnlyFusionConfig:
        return AdaptedOnlyFusionConfig(
            num_labels=self.num_labels,
            head_seeds=self.head_seeds,
            calibration=self.adapted_only_calibration,
        )


def _calibration(block: Mapping[str, Any], label_names: tuple[str, ...]) -> CalibrationConfig:
    calibration = CalibrationConfig(
        class_index=block["class_index"],
        additive_logit_bias=block["additive_logit_bias"],
        label_name=block["label_name"],
    )
    if label_names[calibration.class_index] != calibration.label_name:
        raise ViUnMarkContractError(
            f"calibration names class {calibration.class_index} "
            f"{calibration.label_name!r}, but that index is {label_names[calibration.class_index]!r}"
        )
    return calibration


def require_recipe_matches_code(method: Mapping[str, Any]) -> None:
    """The spec's recovered training recipe must equal the policy the code exposes."""
    recipe = method["robust_mlp_training_recipe"]
    if tuple(recipe["evidence_scope"]) != RECIPE_EVIDENCE_SCOPE:
        raise ViUnMarkContractError("recipe evidence scope disagrees with the code")
    budget = recipe["budget"]
    code_budget = ROBUST_MLP_TRAINING_BUDGET
    if (budget["batch_size"], budget["max_optimizer_updates"], budget["selection_boundaries"],
            budget["updates_per_boundary"]) != (
            code_budget.batch_size, code_budget.max_optimizer_updates,
            code_budget.selection_boundaries, code_budget.updates_per_boundary):
        raise ViUnMarkContractError(f"recipe budget {budget} disagrees with the code")
    if budget["max_optimizer_updates"] != budget["selection_boundaries"] * budget[
        "updates_per_boundary"
    ]:
        raise ViUnMarkContractError("recipe budget is not boundaries x updates_per_boundary")
    optimizer = recipe["optimizer"]
    policy = ROBUST_MLP_OPTIMIZER_POLICY
    if (optimizer["name"], optimizer["learning_rate"], tuple(optimizer["betas"]), optimizer["eps"],
            optimizer["weight_decay"]["matrix_weights"],
            optimizer["weight_decay"]["biases_vectors_layernorm"]) != (
            policy.name, policy.learning_rate, policy.betas, policy.eps,
            policy.matrix_weight_decay, policy.non_matrix_weight_decay):
        raise ViUnMarkContractError(f"recipe optimizer {optimizer} disagrees with the code")
    if tuple(recipe["augmented_training_set"]["condition_order"]) != SIX_CONDITIONS:
        raise ViUnMarkContractError("recipe condition order disagrees with the code")
    if recipe["cross_entropy_reduction"]["status"] != "UNRESOLVED":
        raise ViUnMarkContractError(
            "the spec claims a cross-entropy reduction the code does not recover"
        )


_UNESTABLISHED_CLAIM_KEYS = frozenset({
    "betas", "adamw_betas", "eps", "adamw_eps", "cycle_seed", "batch_stream", "dropout_seed",
    "cross_entropy_reduction", "reduction", "augmented_training_set",
    "augmented_concatenation_implementation", "row_order_within_condition",
    "initialization_rng_stream", "partial_final_batch", "state_dict_layout",
})


def _keys_deep(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = set(value)
        for item in value.values():
            found |= _keys_deep(item)
        return found
    if isinstance(value, list):
        return set().union(*(_keys_deep(item) for item in value)) if value else set()
    return set()


def require_phobert_record_matches_code(method: Mapping[str, Any]) -> None:
    """The native PhoBERT protocol record: exactly what it confirms, nothing it does not.

    Its confirmed fields must equal `PHOBERT_READOUT_POLICY_CONFIRMED` and agree with
    the code's policy constants. It must list exactly `PHOBERT_READOUT_NOT_ESTABLISHED`
    as not established, and it may not claim any of those anywhere inside its
    confirmed fields. Fields recovered only for the adapted-pathway heads therefore
    cannot be written into this record.
    """
    record = method["phobert_readout_training"]
    confirmed = record["confirmed"]
    if tuple(confirmed) != PHOBERT_READOUT_POLICY_CONFIRMED:
        raise ViUnMarkContractError(
            f"PhoBERT readout record confirms {list(confirmed)}, the code recognises "
            f"{list(PHOBERT_READOUT_POLICY_CONFIRMED)}"
        )
    if tuple(record["not_established_by_protocol"]) != PHOBERT_READOUT_NOT_ESTABLISHED:
        raise ViUnMarkContractError("PhoBERT readout not-established list disagrees with the code")
    claimed = _keys_deep(confirmed) & _UNESTABLISHED_CLAIM_KEYS
    if claimed:
        raise ViUnMarkContractError(
            f"the native PhoBERT protocol does not establish {sorted(claimed)}; the record may "
            "not claim them"
        )
    if (record["policy_status"], record["bit_exact_retraining"]) != (
        "SUBSTANTIALLY_RECOVERED", "NOT_RECOVERED"
    ):
        raise ViUnMarkContractError("PhoBERT readout recovery status is misstated")

    policy, budget = ROBUST_MLP_OPTIMIZER_POLICY, ROBUST_MLP_TRAINING_BUDGET
    expected = {
        "head_architecture": method["robust_mlp_head"]["layers"],
        "initialization": {
            "LayerNorm": {"weight": LAYERNORM_WEIGHT_INIT, "bias": LAYERNORM_BIAS_INIT},
            "Linear": {"weight": LINEAR_WEIGHT_INIT, "bias": LINEAR_BIAS_INIT},
        },
        "optimizer_name": policy.name,
        "learning_rate": policy.learning_rate,
        "matrix_weight_decay": policy.matrix_weight_decay,
        "bias_and_vector_weight_decay": policy.non_matrix_weight_decay,
        "budget": {
            "batch_size": budget.batch_size,
            "max_optimizer_updates": budget.max_optimizer_updates,
            "selection_boundaries": budget.selection_boundaries,
            "updates_per_boundary": budget.updates_per_boundary,
        },
        "best_seed_selection": False,
        "precision": {"dtype": TRAINING_DTYPE, "amp": TRAINING_AMP, "tf32": TRAINING_TF32},
    }
    for name, want in expected.items():
        if confirmed[name] != want:
            raise ViUnMarkContractError(
                f"PhoBERT readout record {name}={confirmed[name]!r} disagrees with {want!r}"
            )
    if tuple(confirmed["six_condition_training_order"]["condition_order"]) != SIX_CONDITIONS:
        raise ViUnMarkContractError("PhoBERT readout condition order disagrees with the code")
    selection = confirmed["checkpoint_selection"]
    adapted_selection = method["robust_mlp_training_recipe"]["checkpoint_selection"]
    if (selection["primary"], selection["tie_breaks"]) != (
        adapted_selection["primary"], adapted_selection["tie_breaks"]
    ):
        raise ViUnMarkContractError("PhoBERT readout checkpoint selection rule is misstated")
    candidate = confirmed["selected_candidate"]
    phobert_branch = method["branches"]["phobert_readout"]
    if (candidate["readout"], candidate["input_dim"], candidate["loss"]) != (
        phobert_branch["readout"], phobert_branch["input_dim"], phobert_branch["loss"]
    ):
        raise ViUnMarkContractError("PhoBERT selected candidate disagrees with the branch recipe")


def require_development_stages(stages: Mapping[str, Any], protocol_lineage: Mapping[str, str]) -> None:
    """The two recovered Stage-II development stages, as recorded."""
    screen = stages["stage2_readout_and_augmentation_screen"]
    candidates = screen["candidates"]
    if [c["index"] for c in candidates] != [1, 2, 3, 4, 5, 6]:
        raise ViUnMarkContractError("the readout and augmentation screen has six candidates")
    grid = {(c["readout"], c["training_distribution"]) for c in candidates}
    readouts = {"MASKED_MEAN", "L2_MASKED_MEAN", "CONCAT"}
    distributions = {"CLEAN_ONLY", "AUGMENTED_SIX_CONDITIONS"}
    if grid != {(r, d) for r in readouts for d in distributions}:
        raise ViUnMarkContractError("the screen candidates are not the readout x distribution grid")
    if screen["best_seed_selection"] is not False:
        raise ViUnMarkContractError("the screen performs no best-seed selection")
    balance = stages["stage2_class_balance_loss_optimization"]
    if balance["promoted_screen_candidates"] != [4, 6] or balance["focal_gamma"] != 2.0:
        raise ViUnMarkContractError("class-balance stage bases or focal gamma are misstated")
    for stage in (screen, balance):
        if stage["protocol"] not in protocol_lineage:
            raise ViUnMarkContractError(f"{stage['protocol']} has no protocol lineage digest")


def load_uit_vsfc_reproduction() -> UitVsfcReproduction:
    """Load and cross-check the UIT-VSFC reproduction record. Fails closed."""
    spec = load_final_system_spec()
    method = spec["method"]
    record = spec["uit_vsfc_reproduction"]
    require_recipe_matches_code(method)
    require_phobert_record_matches_code(method)

    label_names = tuple(record["label_names"])
    head_seeds = tuple(record["head_seeds"])

    # --- Stage-I ------------------------------------------------------------
    checkpoints = {}
    for name, block in record["stage1_checkpoints"].items():
        checkpoint = StageOneCheckpoint(
            pathway=name,
            sha256=block["sha256"],
            selected_update=block["selected_update"],
            fusion_id=block["fusion_id"],
            source_repository_head=block["source_repository_head"],
        )
        if checkpoint.fusion_id != _PATHWAY_CONFIGS[name]().fusion_id:
            raise ViUnMarkContractError(
                f"{name} checkpoint records fusion {checkpoint.fusion_id!r}, the method "
                f"defines {_PATHWAY_CONFIGS[name]().fusion_id!r}"
            )
        if method["stage1"]["pathways"][name]["fusion_id"] != checkpoint.fusion_id:
            raise ViUnMarkContractError(f"{name} fusion differs between method and record")
        checkpoints[name] = checkpoint

    # --- corruption identity and representation bank -------------------------
    corruption = dict(record["corruption"])
    if tuple(corruption["conditions"]) != SIX_CONDITIONS:
        raise ViUnMarkContractError("recorded corruption conditions differ from the method")
    placeholder = corruption["full_condition_api_seed"]
    if placeholder["scientific"] is not False or placeholder["value"] != FULL_CONDITION_API_SEED:
        raise ViUnMarkContractError(
            "the FULL-condition API seed must be the non-scientific placeholder "
            f"{FULL_CONDITION_API_SEED}"
        )
    CorruptionProtocol(scientific_corruption_seed=corruption["scientific_corruption_seed"])
    bank = dict(record["representation_bank"])
    if bank["max_length"] != corruption["max_length"]:
        raise ViUnMarkContractError("representation bank and corruption disagree on max_length")
    for flag in ("same_encoder_forward", "base_grid_invariance", "cross_pathway_input_identity"):
        if bank[flag] is not True:
            raise ViUnMarkContractError(f"representation bank must record {flag}=true")

    # --- weighted loss and augmented rows -----------------------------------
    counts_block = record["weighted_loss"]["protocol_train_class_counts"]
    counts = tuple(counts_block[name] for name in label_names)
    recorded_vector = tuple(record["weighted_loss"]["weight_vector"])
    recomputed = sqrt_inverse_frequency_weights(counts, arithmetic="float32")
    if recorded_vector != recomputed:
        raise ViUnMarkContractError(
            f"recorded weight vector {recorded_vector} is not the rule applied to the "
            f"recorded counts {counts} (float32 gives {recomputed})"
        )
    rows = record["augmented_training_set"]["protocol_train_rows"]
    augmented = record["augmented_training_set"]["augmented_rows"]
    if rows != sum(counts) or augmented != rows * len(SIX_CONDITIONS):
        raise ViUnMarkContractError(
            f"recorded rows {rows} / augmented {augmented} disagree with counts {counts} and "
            f"{len(SIX_CONDITIONS)} conditions"
        )

    # --- the twenty heads -----------------------------------------------------
    boundaries = method["robust_mlp_training_recipe"]["budget"]["selection_boundaries"]
    selected: dict[LogitStream, tuple[SelectedHead, ...]] = {}
    for stream in FINAL_BRANCH_STREAMS:
        heads = tuple(
            SelectedHead(stream, row["seed"], row["selected_boundary"], row["sha256"])
            for row in record["selected_heads"][stream.value]
        )
        if tuple(h.seed for h in heads) != head_seeds:
            raise ViUnMarkContractError(
                f"{stream.value} heads cover seeds {[h.seed for h in heads]}, not {head_seeds}"
            )
        for head in heads:
            if not 1 <= head.selected_boundary <= boundaries:
                raise ViUnMarkContractError(
                    f"{stream.value} seed {head.seed} records boundary "
                    f"{head.selected_boundary}, outside 1..{boundaries}"
                )
        selected[stream] = heads
    digests = [h.sha256 for heads in selected.values() for h in heads]
    if len(set(digests)) != len(digests):
        raise ViUnMarkContractError("two recorded heads share one sha256")

    # --- lineage ----------------------------------------------------------------
    artifact_lineage = dict(record["artifact_lineage"])
    protocol_lineage = dict(record["protocol_lineage"])
    shared = set(artifact_lineage.values()) & set(protocol_lineage.values())
    if shared:
        raise ViUnMarkContractError(
            f"a protocol digest is also recorded as an artifact digest: {sorted(shared)}"
        )
    development_stages = dict(record["development_stages"])
    require_development_stages(development_stages, protocol_lineage)
    phobert_heads = record["stage2_head_training"]["phobert_readout_heads"]
    if tuple(phobert_heads["seeds"]) != head_seeds or phobert_heads["best_seed_selection"] is not False:
        raise ViUnMarkContractError("PhoBERT readout heads must use the five head seeds, no selection")

    calibration = record["calibration"]
    standalone = calibration["phobert_readout_standalone_development_stage"]
    reproduction = UitVsfcReproduction(
        label_names=label_names,
        head_seeds=head_seeds,
        stage1_checkpoints=checkpoints,
        corruption=corruption,
        representation_bank=bank,
        protocol_train_class_counts=counts,
        weight_vector=recorded_vector,
        protocol_train_rows=rows,
        augmented_rows=augmented,
        selected_heads=selected,
        total_stage2_parameters=record["total_stage2_parameters"],
        adapted_only_calibration=_calibration(calibration["adapted_only_fusion"], label_names),
        viunmark_calibration=_calibration(calibration["viunmark"], label_names),
        phobert_standalone_development_calibration=_calibration(standalone, label_names),
        phobert_standalone_calibration_inherited=standalone["inherited_by_viunmark"],
        stage2_head_training=dict(record["stage2_head_training"]),
        system_selection=dict(record["system_selection"]),
        artifact_lineage=artifact_lineage,
        protocol_lineage=protocol_lineage,
        development_stages=development_stages,
    )

    config = reproduction.viunmark_config()
    if reproduction.head_count != record["head_counts"]["total"] or (
        reproduction.head_count != config.head_count
    ):
        raise ViUnMarkContractError(
            f"recorded head count {record['head_counts']['total']} disagrees with "
            f"{reproduction.head_count} recorded heads and the method's {config.head_count}"
        )
    if reproduction.total_stage2_parameters != config.stage2_parameter_count:
        raise ViUnMarkContractError(
            f"recorded {reproduction.total_stage2_parameters} Stage-II parameters, the "
            f"architecture gives {config.stage2_parameter_count}"
        )
    if reproduction.phobert_standalone_calibration_inherited is not False:
        raise ViUnMarkContractError(
            "the standalone PhoBERT calibration must not be inherited by ViUnMark"
        )
    adapted_streams = tuple(record["stage2_head_training"]["adapted_pathway_heads"]["streams"])
    if adapted_streams != RECIPE_EVIDENCE_SCOPE:
        raise ViUnMarkContractError(
            "the recovered training recipe must stay scoped to the adapted-pathway heads"
        )
    return reproduction


__all__ = [
    "FINAL_SYSTEM_SPEC_PATH",
    "HISTORICAL_ALIAS_SPEC_PATH",
    "SelectedHead",
    "StageOneCheckpoint",
    "UitVsfcReproduction",
    "load_final_system_spec",
    "load_historical_aliases",
    "load_uit_vsfc_reproduction",
    "require_development_stages",
    "require_phobert_record_matches_code",
    "require_recipe_matches_code",
]
