"""Recorded ViUnMark identities: the UIT-VSFC reproduction record, exactly.

Every digest, boundary, count and calibration below is a literal copied from the
recovered final-system definition, and each is checked against the committed
spec and the loader that reads it. Where the repository already holds an
identity (the two Stage-I checkpoints), this migration test also checks the
public record against those historical research modules. Torch-free.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.viunmark.provenance as provenance  # noqa: E402
from unmark.viunmark.config import (  # noqa: E402
    FINAL_BRANCH_STREAMS,
    CalibrationConfig,
    LogitStream,
    ViUnMarkContractError,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
SEEDS = (53148, 59945, 42941, 720, 9428)

STAGE1 = {
    "viunmark_gate": "6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91",
    "viunmark_scale": "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685",
}

HEADS = {
    LogitStream.GATE_ROBUST_READOUT: [
        (53148, 13, "8d068ec69f4c67b9117197ef52141c97038844ca2a8db65066cbb107b3ccfca3"),
        (59945, 14, "cdf52d615d7bc5f0bcf64af9ba51a98b9313753657abee1aff5e7b16706435db"),
        (42941, 14, "00e7a00c293c565ee8ecff0f8d862126733df76729e261a4430a9a584c8d89f8"),
        (720, 12, "0e8ce86edad10199a1e6b670b792d3442a2528d5a371388eefca9612157ccdc4"),
        (9428, 20, "c6409f9d74c38619e956dd5698e6c968c7eef139261d5473ffb7bf9319725cc9"),
    ],
    LogitStream.SCALE_UNWEIGHTED_READOUT: [
        (53148, 22, "be5448e72285cf8563fb9235f570c281f149d52290d3974e3c82ee23fdc8bb7b"),
        (59945, 24, "9eace3bb3b22568154eda12aef02b2727bc2f5c33fb8f71d1f7154c546e5d291"),
        (42941, 14, "877defe241573f728e3a659a03ff5e6d39e04bc9de2a9876f85f81b32e07edae"),
        (720, 27, "484818e703a9084ba8594addec629a8c0ba7d5588dce897ee75eec3c4d4ae291"),
        (9428, 26, "c1db40e71518a044c81b9f556b166b8fdba8a05040e05c2189ba23724ba55c7f"),
    ],
    LogitStream.SCALE_WEIGHTED_READOUT: [
        (53148, 19, "481471e02a92c99b7a78c714ed80986c35ad9d7d74d97f81170ebc8a9b021971"),
        (59945, 15, "ba75892a9e03e81fd096ba168e5b0ebf563f77916d5a514d8e70e4b9c3a7901b"),
        (42941, 25, "dc04eeb9972b97b3ec8f849d320fd38ec5821d5c36179890a7e051849aaaace7"),
        (720, 30, "942a7380aa9f86a7df10ca210a5787aa0d6cc5c3862534fecd78997858644189"),
        (9428, 22, "c9066d857f3a0ff3ca7bd9a6b20dfac5c179e79db147ffa3ac31ca7e90e165d2"),
    ],
    LogitStream.PHOBERT_READOUT: [
        (53148, 17, "94f8238f391f498bab69449ed0f623281315068d4575a481bb6951c0cdbd2431"),
        (59945, 24, "e2b7315ec6ac623e253f355126a9ba1a7908f5d47f437f7ed1736b8ac04c0ab8"),
        (42941, 12, "5f4503709e2574ea5af6f400a23d7506462d930d22db8bc4990c37830d76d868"),
        (720, 26, "c13f5905c78d8620f959690bc990d347acb57881693c6167107d3abfc25b9bc3"),
        (9428, 27, "84451b7a34128a17dad9f16eb3e7d06beec16cea113b1cdfac3d9d9e703d115b"),
    ],
}

LINEAGE = {
    "stage1_gate_checkpoint":
        "6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91",
    "stage1_scale_checkpoint":
        "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685",
    "adapted_robust_readout_optimization_final":
        "8f78de272bd0f5046cbb76053148c74febc6af280dd632bae9a8cd2aa9e2835c",
    "adapted_only_fusion_final":
        "d520e792df6df1284411fb397a9b15bcd035406868b8b94e1884b54f53fbd5f4",
    "phobert_robust_readout_selection_final":
        "d6eeecbf9f8c13cb0cdc89d47c2a1d122d4c2b9b6e9887563dad6f4aa9a0530e",
    "viunmark_protocol":
        "f1d3887307ae5ad36abdb7ea307d89af2488933d6e8b1fe25dd01b388089e121",
    "viunmark_final":
        "0f0801b19945ba60537865a32663eed3468f0b4c7968511ddd7d4608a164911a",
    "viunmark_official_validation_evidence":
        "a2d9db0e964b48a0084ac2db4173cc585489c5ca04d166516a0cc7f109015a34",
    "viunmark_official_test_evidence":
        "7f996fdaadbacd075d407dc4aeea529c824f3df9d476f6f928bad16bfe1269d2",
}


@pytest.fixture(scope="module")
def reproduction():
    return provenance.load_uit_vsfc_reproduction()


# ---------------------------------------------------------------------------
# Stage-I identities
# ---------------------------------------------------------------------------
def test_stage1_checkpoint_identities(reproduction):
    assert dict(reproduction.stage1_checkpoint_sha256) == STAGE1


def test_stage1_identities_agree_with_the_historical_research_modules():
    """Migration check: the public record names the checkpoints the research code verifies."""
    from unmark.evaluation.stage2_scf_pathway import V2_SCF_CHECKPOINT  # historical alias
    from unmark.stage1.finalists import FINALIST_A  # historical alias of ViUnMark-Gate

    assert FINALIST_A.checkpoint_sha256 == STAGE1["viunmark_gate"]
    assert V2_SCF_CHECKPOINT.checkpoint_sha256 == STAGE1["viunmark_scale"]


def test_spec_pathway_fusions_match_the_method():
    spec = provenance.load_final_system_spec()
    pathways = spec["method"]["stage1"]["pathways"]
    assert pathways["viunmark_gate"]["fusion_id"] == "historical-fusion-v1"
    assert pathways["viunmark_scale"]["fusion_id"] == "scale-calibrated-fusion-v1"


# ---------------------------------------------------------------------------
# The twenty selected heads
# ---------------------------------------------------------------------------
def test_exactly_twenty_heads_five_per_branch(reproduction):
    assert set(reproduction.selected_heads) == set(FINAL_BRANCH_STREAMS)
    for stream in FINAL_BRANCH_STREAMS:
        assert len(reproduction.selected_heads[stream]) == 5
    assert reproduction.head_count == 20
    assert reproduction.viunmark_config().head_count == 20


def test_every_branch_uses_exactly_the_five_recorded_seeds(reproduction):
    assert reproduction.head_seeds == SEEDS
    assert set(reproduction.head_seeds) == {53148, 59945, 42941, 720, 9428}
    for stream in FINAL_BRANCH_STREAMS:
        assert tuple(h.seed for h in reproduction.selected_heads[stream]) == SEEDS


@pytest.mark.parametrize("stream", FINAL_BRANCH_STREAMS)
def test_head_seed_boundary_and_digest_identities(reproduction, stream):
    recorded = [(h.seed, h.selected_boundary, h.sha256) for h in reproduction.selected_heads[stream]]
    assert recorded == HEADS[stream]


def test_head_digests_are_distinct_and_well_formed(reproduction):
    digests = [h.sha256 for heads in reproduction.selected_heads.values() for h in heads]
    assert len(digests) == len(set(digests)) == 20
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in digests)


def test_total_stage2_parameters(reproduction):
    assert reproduction.total_stage2_parameters == 6955580
    assert reproduction.viunmark_config().stage2_parameter_count == 6955580


# ---------------------------------------------------------------------------
# Weighted loss, calibration, training and selection facts
# ---------------------------------------------------------------------------
def test_weighted_loss_reproduction_values(reproduction):
    assert reproduction.label_names == ("negative", "neutral", "positive")
    assert reproduction.protocol_train_class_counts == (4259, 366, 4514)
    assert reproduction.weight_vector == (0.5573523044586182, 1.9012669324874878, 0.5413808226585388)
    assert reproduction.weight_vector == pytest.approx((0.5573523045, 1.9012669325, 0.5413808227),
                                                       abs=1e-10)


def test_protocol_train_counts_agree_with_the_historical_split_tests():
    """The recorded counts are the ones the committed split contract already pins."""
    text = (REPO / "tests/test_preg1_split.py").read_text(encoding="utf-8")
    assert '"negative": 4259, "neutral": 366, "positive": 4514' in text


def test_reproduction_calibrations(reproduction):
    assert reproduction.viunmark_calibration == CalibrationConfig(
        class_index=1, additive_logit_bias=1.25, label_name="neutral")
    assert reproduction.adapted_only_calibration == CalibrationConfig(
        class_index=1, additive_logit_bias=0.75, label_name="neutral")
    assert reproduction.phobert_standalone_development_calibration == CalibrationConfig(
        class_index=1, additive_logit_bias=0.75, label_name="neutral")
    assert reproduction.phobert_standalone_calibration_inherited is False
    assert reproduction.viunmark_config().calibration.bias_vector(3) == (0.0, 1.25, 0.0)
    assert reproduction.adapted_only_fusion_config().calibration.bias_vector(3) == (0.0, 0.75, 0.0)


def test_stage2_head_training_facts(reproduction):
    recipe = provenance.load_final_system_spec()["method"]["robust_mlp_training_recipe"]
    assert recipe["budget"]["batch_size"] == 128
    assert recipe["optimizer"]["learning_rate"] == 0.01
    assert recipe["budget"]["max_optimizer_updates"] == 2160
    assert recipe["budget"]["selection_boundaries"] == 30
    assert recipe["precision"] == {"dtype": "float32", "amp": False, "tf32": False}
    adapted = reproduction.stage2_head_training["adapted_pathway_heads"]
    assert adapted["best_seed_selection"] is False
    assert adapted["architecture_confirmed_by_frozen_artifact"] is True
    assert adapted["recipe"] == "method.robust_mlp_training_recipe"
    assert reproduction.stage2_head_training["phobert_readout_heads"]["best_seed_selection"] is False


def test_system_selection_facts(reproduction):
    selection = reproduction.system_selection
    assert selection["scientific_status"] == "posthoc-exploratory"
    assert selection["selection_data"] == "protocol-dev only"
    assert selection["primary_criterion"] == "six-condition Macro-F1 mean"
    assert selection["tie_breaks"] == [
        "worst-condition Macro-F1",
        "FULL Macro-F1",
        "smaller absolute class-1 bias",
        "fewer deployed/selected heads",
    ]
    assert all(v is False for v in selection["official_validation"].values())
    assert set(selection["official_test"]) == {
        "training", "selection", "tuning", "recalibration", "post_test_retuning"}
    assert all(v is False for v in selection["official_test"].values())


def test_artifact_lineage(reproduction):
    assert dict(reproduction.artifact_lineage) == LINEAGE


def test_spec_status_records_recovery_not_selection():
    status = provenance.load_final_system_spec()["status"]
    assert status["recovered_not_newly_selected"] is True
    for flag in ("new_model_selection", "new_hyperparameter_search", "new_training",
                 "validation_read", "test_read", "test_tuning", "test_recalibration",
                 "post_test_retuning"):
        assert status[flag] is False, flag


def test_spec_records_the_twenty_head_ensemble_semantics():
    spec = provenance.load_final_system_spec()
    ensemble = spec["method"]["head_ensemble"]
    assert ensemble["within_branch_logit_ensemble"] == "MEAN"
    assert ensemble["best_seed_selection"] is False
    assert ensemble["fusion_happens_after_within_branch_mean"] is True
    assert ensemble["final_system_replicate_count"] == 1
    record = spec["uit_vsfc_reproduction"]
    assert record["heads_per_branch"] == 5
    assert record["final_branches"] == 4
    assert record["final_head_count"] == 20
    assert record["final_system_is_20_head_ensemble"] is True
    assert spec["method"]["calibration"]["parent_biases_stacked"] is False


def test_spec_method_section_agrees_with_the_code():
    from unmark.viunmark.fusion import expanded_per_head_weights, expanded_viunmark_weights

    spec = provenance.load_final_system_spec()["method"]
    expanded = {s.value: w for s, w in expanded_viunmark_weights().items()}
    assert spec["fusion_graph"]["expanded_weights"]["viunmark_raw"] == expanded
    per_head = {s.value: w for s, w in expanded_per_head_weights(expanded_viunmark_weights(), 5).items()}
    assert spec["fusion_graph"]["expanded_per_head_weights_with_five_heads_per_branch"][
        "viunmark_raw"] == pytest.approx(per_head, abs=1e-15)
    assert spec["robust_mlp_head"]["parameters_num_labels_3"] == {
        "input_dim_768": 199171, "input_dim_1536": 397315}
    assert spec["readouts"]["CONCAT"]["order"] == ["FIRST_TOKEN", "MASKED_MEAN"]
    assert spec["training_distribution"]["AUGMENTED_SIX_CONDITIONS"]["conditions"] == [
        "FULL", "P25", "P50", "P75", "P100", "STRIP_ALL"]


def test_calibration_values_live_only_in_the_reproduction_section():
    method = json.dumps(provenance.load_final_system_spec()["method"])
    assert "1.25" not in method and "0.75" not in method


# ---------------------------------------------------------------------------
# The loader fails closed on a drifted record
# ---------------------------------------------------------------------------
def write_spec(tmp_path, monkeypatch, mutate):
    spec = copy.deepcopy(provenance.load_final_system_spec())
    mutate(spec)
    path = tmp_path / "viunmark-final-system-v1.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(provenance, "FINAL_SYSTEM_SPEC_PATH", path)


def test_loader_refuses_a_nineteen_head_record(tmp_path, monkeypatch):
    def drop(spec):
        spec["uit_vsfc_reproduction"]["selected_heads"]["phobert_readout"].pop()
    write_spec(tmp_path, monkeypatch, drop)
    with pytest.raises(ViUnMarkContractError):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_branch_with_a_different_seed(tmp_path, monkeypatch):
    def swap(spec):
        spec["uit_vsfc_reproduction"]["selected_heads"]["gate_robust_readout"][0]["seed"] = 1
    write_spec(tmp_path, monkeypatch, swap)
    with pytest.raises(ViUnMarkContractError, match="seeds"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_weight_vector_that_is_not_the_rule(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["weighted_loss"]["weight_vector"][1] = 1.9
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="rule"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_an_inherited_standalone_calibration(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["calibration"][
            "phobert_readout_standalone_development_stage"]["inherited_by_viunmark"] = True
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="inherited"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_calibration_naming_the_wrong_label(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["calibration"]["viunmark"]["label_name"] = "positive"
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_boundary_outside_the_recorded_range(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["selected_heads"]["scale_weighted_readout"][3][
            "selected_boundary"] = 31
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="boundary"):
        provenance.load_uit_vsfc_reproduction()


# ---------------------------------------------------------------------------
# Evidence amendment: corruption identity, representation bank, Stage-I
# provenance, protocol lineage, recipe scope
# ---------------------------------------------------------------------------
PROTOCOL_LINEAGE = {
    "stage2_optimization_parent_freeze":
        "f07fa83b38063b81a237eed68f59d2ddda45cd46ff283b7398587af486a8c900",
    "stage2_representation_bank_final":
        "0044ba7bda8caa2aabbfbcb2a30fe3bff224d988c5806417c76822715917da50",
    "stage2_readout_and_augmentation_screen_protocol":
        "3fbb1b3dc581df123ca3f0f40bd293f3964aba17995cd5b2385e0eae923169e8",
    "stage2_class_balance_loss_optimization_protocol":
        "c802a1b277c6fd48b7d76588737a7abf9e6058c3421d9f0e94adea2c5ea7a4b2",
    "adapted_robust_readout_optimization_protocol":
        "b39bb40df5495479e3ba55ddb25ca45e19990deb8be6230b8f0b65b45eb91689",
    "phobert_robust_readout_protocol":
        "e2b6dcff0cb131d5bd10b51eded6c4fe35141bdc04328503018c662d07738cab",
}


def test_historical_scientific_corruption_seed(reproduction):
    assert reproduction.scientific_corruption_seed == 19225
    assert reproduction.corruption_protocol().scientific_corruption_seed == 19225
    assert reproduction.corruption["evidence"] == [
        "stage2_optimization_parent_freeze", "stage2_representation_bank_final"]


def test_historical_seed_agrees_with_the_research_measurement_seed():
    """Migration check: the recovered seed is the frozen Stage-2 measurement seed."""
    from unmark.evaluation.stage2_head_campaign import STAGE2_MEASUREMENT_CORRUPTION_SEED

    assert STAGE2_MEASUREMENT_CORRUPTION_SEED == 19225


def test_full_condition_api_seed_is_a_non_scientific_placeholder(reproduction):
    placeholder = reproduction.corruption["full_condition_api_seed"]
    assert placeholder["value"] == 0
    assert placeholder["scientific"] is False


def test_historical_corruption_identity(reproduction):
    from unmark.stage1.finalists import INVENTORY_SHA256, INVENTORY_SOURCE_REVISION

    corruption = reproduction.corruption
    assert corruption["conditions"] == ["FULL", "P25", "P50", "P75", "P100", "STRIP_ALL"]
    assert corruption["eligibility_policy"] == "VIETNAMESE_SYLLABLE_INVENTORY"
    assert corruption["inventory_sha256"] == (
        "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2") == INVENTORY_SHA256
    assert corruption["inventory_source_revision"] == (
        "135a4d9716e49a981624474156d6f247b9b46f6a") == INVENTORY_SOURCE_REVISION
    assert corruption["max_length"] == 256


def test_historical_representation_bank(reproduction):
    bank = reproduction.representation_bank
    assert bank["dtype"] == "torch.float32"
    assert bank["hidden_size"] == 768
    assert bank["max_length"] == 256
    assert "stage2_first_token_representation" in bank["first_token_source"]
    assert "masked_mean_non_special" in bank["masked_mean_source"]
    assert "collate_stage2_unmark_batch" in bank["special_tokens_mask_source"]
    assert bank["same_encoder_forward"] is True
    assert bank["base_grid_invariance"] is True
    assert bank["cross_pathway_input_identity"] is True


def test_representation_bank_sources_exist_in_the_repository():
    from unmark.evaluation import stage2_dual_finalist
    from unmark.modeling import contracts  # noqa: F401  (torch-free import check)

    assert callable(stage2_dual_finalist.stage2_first_token_representation)
    assert callable(stage2_dual_finalist.collate_stage2_unmark_batch)
    source = (REPO / "unmark/modeling/pooling.py").read_text(encoding="utf-8")
    assert "def masked_mean_non_special(" in source


def test_stage1_selected_updates_and_source_heads(reproduction):
    gate = reproduction.stage1_checkpoints["viunmark_gate"]
    scale = reproduction.stage1_checkpoints["viunmark_scale"]
    assert (gate.selected_update, gate.fusion_id, gate.source_repository_head) == (
        3500, "historical-fusion-v1", "7773c77b1df92a6e685dac13c49765ce974f84d8")
    assert (scale.selected_update, scale.fusion_id, scale.source_repository_head) == (
        8000, "scale-calibrated-fusion-v1", "8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526")


def test_stage1_provenance_agrees_with_the_historical_research_modules(reproduction):
    from unmark.evaluation.stage2_scf_pathway import V2_SCF_CHECKPOINT  # historical alias
    from unmark.stage1.finalists import FINALIST_A  # historical alias of ViUnMark-Gate

    gate = reproduction.stage1_checkpoints["viunmark_gate"]
    scale = reproduction.stage1_checkpoints["viunmark_scale"]
    assert (FINALIST_A.update, FINALIST_A.source_repository_head) == (
        gate.selected_update, gate.source_repository_head)
    assert (V2_SCF_CHECKPOINT.update, V2_SCF_CHECKPOINT.source_repository_head) == (
        scale.selected_update, scale.source_repository_head)


def test_augmented_training_rows(reproduction):
    assert reproduction.protocol_train_rows == 9139
    assert reproduction.augmented_rows == 54834 == 9139 * 6


def test_protocol_lineage(reproduction):
    assert dict(reproduction.protocol_lineage) == PROTOCOL_LINEAGE


def test_protocol_and_artifact_digests_are_kept_distinct(reproduction):
    assert not set(reproduction.protocol_lineage.values()) & set(
        reproduction.artifact_lineage.values())
    assert (reproduction.protocol_lineage["adapted_robust_readout_optimization_protocol"]
            != reproduction.artifact_lineage["adapted_robust_readout_optimization_final"])


def test_recovery_status_block():
    recovery = provenance.load_final_system_spec()["status"]["recovery"]
    assert recovery == {
        "p0": "RESOLVED",
        "final_inference_definition": "RECOVERED",
        "adapted_readout_training_policy": "RECOVERED_AT_PROTOCOL_LEVEL",
        "phobert_readout_training_policy": "SUBSTANTIALLY_RECOVERED",
        "historical_bit_exact_retraining": "NOT_FULLY_RECOVERED",
        "phobert_bit_exact_retraining": "NOT_RECOVERED",
        "cross_entropy_reduction": "UNRESOLVED",
        "scientific_corruption_seed": "RECOVERED",
        "cross_pathway_input_identity": "REQUIRED",
        "weighted_ce_rule": "RECOVERED",
        "stage2_development_stage_semantics": "RECOVERED",
        "adapted_readout_policy_components": {
            "mlp_initialization": "RECOVERED",
            "optimizer_policy": "RECOVERED",
            "augmented_train_construction": "RECOVERED",
            "deterministic_batch_stream": "RECOVERED",
            "dropout_seed_policy": "RECOVERED",
            "checkpoint_selection": "RECOVERED",
        },
        "distinction": "method/protocol recovery is not bit-exact historical execution recovery",
    }


def test_adapted_recipe_is_not_promoted_to_the_phobert_readout():
    """Gate/Scale-only facts stay out of the PhoBERT record; the full recipe stays scoped."""
    spec = provenance.load_final_system_spec()
    recipe = spec["method"]["robust_mlp_training_recipe"]
    assert "phobert_readout" not in recipe["evidence_scope"]
    phobert = spec["method"]["phobert_readout_training"]
    for gate_scale_only in ("adamw_betas", "adamw_eps", "cycle_seed", "dropout_seed",
                            "augmented_concatenation_implementation"):
        assert gate_scale_only in phobert["not_established_by_protocol"], gate_scale_only
        assert gate_scale_only not in phobert["confirmed"], gate_scale_only
    record = spec["uit_vsfc_reproduction"]["stage2_head_training"]
    assert record["phobert_readout_heads"]["recipe"].endswith("(substantially recovered)")


def test_seed_and_inventory_values_live_only_in_the_reproduction_section():
    method = json.dumps(provenance.load_final_system_spec()["method"])
    for literal in ("19225", "78eeb840", "9139", "54834"):
        assert literal not in method, literal


def test_loader_refuses_promoting_the_recipe_to_the_phobert_readout(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["stage2_head_training"]["adapted_pathway_heads"][
            "streams"].append("phobert_readout")
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="scoped"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_scientific_full_placeholder(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["corruption"]["full_condition_api_seed"]["scientific"] = True
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="placeholder"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_protocol_digest_reused_as_an_artifact_digest(tmp_path, monkeypatch):
    def edit(spec):
        record = spec["uit_vsfc_reproduction"]
        record["protocol_lineage"]["adapted_robust_readout_optimization_protocol"] = record[
            "artifact_lineage"]["adapted_robust_readout_optimization_final"]
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="protocol digest"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_recipe_that_disagrees_with_the_code(tmp_path, monkeypatch):
    def edit(spec):
        spec["method"]["robust_mlp_training_recipe"]["budget"]["updates_per_boundary"] = 70
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="budget"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_a_claimed_cross_entropy_reduction(tmp_path, monkeypatch):
    def edit(spec):
        spec["method"]["robust_mlp_training_recipe"]["cross_entropy_reduction"]["status"] = "mean"
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="reduction"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_refuses_augmented_rows_that_are_not_six_times_the_training_rows(
    tmp_path, monkeypatch
):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["augmented_training_set"]["augmented_rows"] = 54833
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="augmented"):
        provenance.load_uit_vsfc_reproduction()


# ---------------------------------------------------------------------------
# Final review correction: native PhoBERT protocol record and development stages
# ---------------------------------------------------------------------------
NATIVE_NOT_ESTABLISHED = [
    "adamw_betas", "adamw_eps", "augmented_concatenation_implementation", "cycle_seed",
    "dropout_seed", "cross_entropy_reduction", "initialization_rng_stream",
    "partial_final_batch", "state_dict_layout",
]


def phobert_record():
    return provenance.load_final_system_spec()["method"]["phobert_readout_training"]


def test_native_protocol_record_is_the_phobert_readout_protocol_digest(reproduction):
    record = phobert_record()
    assert record["evidence"] == "phobert_robust_readout_protocol"
    assert reproduction.protocol_lineage["phobert_robust_readout_protocol"] == (
        "e2b6dcff0cb131d5bd10b51eded6c4fe35141bdc04328503018c662d07738cab")
    assert record["policy_status"] == "SUBSTANTIALLY_RECOVERED"
    assert record["bit_exact_retraining"] == "NOT_RECOVERED"


def test_native_protocol_record_confirms_head_and_initialisation():
    confirmed = phobert_record()["confirmed"]
    assert confirmed["head_architecture"] == [
        "LayerNorm(input_dim)", "Linear(input_dim, 256)", "GELU", "Dropout(p=0.1)",
        "Linear(256, num_labels)"]
    assert confirmed["initialization"] == {
        "LayerNorm": {"weight": 1.0, "bias": 0.0},
        "Linear": {"weight": "xavier_uniform", "bias": 0.0},
    }


def test_native_protocol_record_confirms_only_the_stated_optimizer_portion():
    confirmed = phobert_record()["confirmed"]
    assert confirmed["optimizer_name"] == "AdamW"
    assert confirmed["learning_rate"] == 0.01
    assert confirmed["matrix_weight_decay"] == 0.01
    assert confirmed["bias_and_vector_weight_decay"] == 0.0
    flat = json.dumps(confirmed)
    for absent in ("betas", "eps", "0.999", "1e-08"):
        assert absent not in flat, absent


def test_native_protocol_record_confirms_training_budget_order_and_precision(reproduction):
    confirmed = phobert_record()["confirmed"]
    assert confirmed["budget"] == {
        "batch_size": 128, "max_optimizer_updates": 2160,
        "selection_boundaries": 30, "updates_per_boundary": 72}
    assert confirmed["budget"]["max_optimizer_updates"] == 30 * 72
    assert confirmed["six_condition_training_order"] == {
        "mode": "six-condition augmented",
        "condition_order": ["FULL", "P25", "P50", "P75", "P100", "STRIP_ALL"]}
    assert confirmed["best_seed_selection"] is False
    assert confirmed["precision"] == {"dtype": "float32", "amp": False, "tf32": False}
    heads = reproduction.stage2_head_training["phobert_readout_heads"]
    assert heads["seeds"] == [53148, 59945, 42941, 720, 9428]
    assert heads["best_seed_selection"] is False
    assert heads["deployment"] == "uniform mean logits over all five selected heads"


def test_native_protocol_record_confirms_selection_candidate_and_deployment():
    confirmed = phobert_record()["confirmed"]
    assert confirmed["checkpoint_selection"] == {
        "scope": "per seed",
        "data": "protocol-dev",
        "primary": "six-condition Macro-F1 mean",
        "tie_breaks": ["worst-condition Macro-F1", "FULL Macro-F1", "earlier selection boundary"],
    }
    assert confirmed["selected_candidate"] == {
        "readout": "CONCAT", "concat_order": ["FIRST_TOKEN", "MASKED_MEAN"], "input_dim": 1536,
        "loss": "UNWEIGHTED_CROSS_ENTROPY", "head": "robust_mlp_head"}
    assert confirmed["deployment"] == "uniform mean logits over all five selected heads"


def test_native_protocol_record_does_not_establish_the_unsupported_fields():
    record = phobert_record()
    assert record["not_established_by_protocol"] == NATIVE_NOT_ESTABLISHED
    flat_keys = json.dumps(record["confirmed"])
    for claim in ("cycle_seed", "dropout_seed", "reduction", "augmented_training_set",
                  "row_order_within_condition", "betas", "eps"):
        assert f'"{claim}"' not in flat_keys, claim


def test_native_record_constants_match_the_code():
    from unmark.viunmark.training import (
        PHOBERT_READOUT_NOT_ESTABLISHED,
        PHOBERT_READOUT_POLICY_CONFIRMED,
    )

    assert tuple(phobert_record()["confirmed"]) == PHOBERT_READOUT_POLICY_CONFIRMED
    assert list(PHOBERT_READOUT_NOT_ESTABLISHED) == NATIVE_NOT_ESTABLISHED
    assert not set(PHOBERT_READOUT_POLICY_CONFIRMED) & set(PHOBERT_READOUT_NOT_ESTABLISHED)


@pytest.mark.parametrize("path, value", [
    (("optimizer_betas",), [0.9, 0.999]),
    (("budget", "betas"), [0.9, 0.999]),
    (("eps",), 1e-8),
    (("cycle_seed",), "seed * 1000 + cycle_index"),
    (("six_condition_training_order", "dropout_seed"), "seed * 100000 + optimizer_update"),
    (("cross_entropy_reduction",), "mean"),
    (("six_condition_training_order", "augmented_training_set"), {"concatenation": "condition-major"}),
    (("six_condition_training_order", "row_order_within_condition"), "preserved"),
])
def test_loader_rejects_native_claims_the_protocol_does_not_establish(
    tmp_path, monkeypatch, path, value
):
    def edit(spec):
        target = spec["method"]["phobert_readout_training"]["confirmed"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError):
        provenance.load_uit_vsfc_reproduction()


def test_loader_rejects_a_misstated_native_optimizer(tmp_path, monkeypatch):
    def edit(spec):
        spec["method"]["phobert_readout_training"]["confirmed"]["learning_rate"] = 0.001
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="learning_rate"):
        provenance.load_uit_vsfc_reproduction()


def test_loader_rejects_an_overclaimed_native_status(tmp_path, monkeypatch):
    def edit(spec):
        spec["method"]["phobert_readout_training"]["bit_exact_retraining"] = "RECOVERED"
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="status"):
        provenance.load_uit_vsfc_reproduction()


def test_readout_and_augmentation_screen_semantics(reproduction):
    screen = reproduction.development_stages["stage2_readout_and_augmentation_screen"]
    assert screen["protocol"] == "stage2_readout_and_augmentation_screen_protocol"
    assert [(c["index"], c["readout"], c["training_distribution"]) for c in screen["candidates"]] == [
        (1, "MASKED_MEAN", "CLEAN_ONLY"),
        (2, "L2_MASKED_MEAN", "CLEAN_ONLY"),
        (3, "CONCAT", "CLEAN_ONLY"),
        (4, "MASKED_MEAN", "AUGMENTED_SIX_CONDITIONS"),
        (5, "L2_MASKED_MEAN", "AUGMENTED_SIX_CONDITIONS"),
        (6, "CONCAT", "AUGMENTED_SIX_CONDITIONS"),
    ]
    assert screen["concat_order"] == ["FIRST_TOKEN", "MASKED_MEAN"]
    assert screen["head"] == "Linear(input_dim, 3, bias=True)"
    assert screen["loss"] == "UNWEIGHTED_CROSS_ENTROPY"
    assert screen["candidate_ranking"] == "across all five seeds"
    assert screen["best_seed_selection"] is False


def test_class_balance_loss_optimization_semantics(reproduction):
    balance = reproduction.development_stages["stage2_class_balance_loss_optimization"]
    assert balance["protocol"] == "stage2_class_balance_loss_optimization_protocol"
    assert balance["promoted_screen_candidates"] == [4, 6]
    assert balance["loss_variants"] == [
        "UNWEIGHTED_CROSS_ENTROPY", "SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY", "FOCAL"]
    assert balance["focal_gamma"] == 2.0
    assert balance["calibration"] == "the existing shared class-1 calibration protocol"


def test_historical_aliases_pin_the_development_stage_semantics():
    aliases = {a["historical_id"]: a for a in provenance.load_historical_aliases()["aliases"]}
    assert aliases["OPT1"]["descriptive_name"] == "Stage-II Readout and Augmentation Screen"
    assert aliases["OPT2"]["descriptive_name"] == "Stage-II Class-Balance Loss Optimization"
    for stage in ("OPT1", "OPT2"):
        assert aliases[stage]["public_api"] is False and aliases[stage]["python_name"] is None
    assert "Linear(input_dim, 3, bias=True)" in aliases["OPT1"]["note"]
    assert "no best-seed selection" in aliases["OPT1"]["note"]
    assert "gamma 2.0" in aliases["OPT2"]["note"] and "R4 and R6" in aliases["OPT2"]["note"]
    expected = {
        "R1": "screen candidate 1: MASKED_MEAN, CLEAN_ONLY",
        "R2": "screen candidate 2: L2_MASKED_MEAN, CLEAN_ONLY",
        "R3": "screen candidate 3: [FIRST_TOKEN;MASKED_MEAN], CLEAN_ONLY",
        "R4": "screen candidate 4: MASKED_MEAN, AUGMENTED_6COND",
        "R5": "screen candidate 5: L2_MASKED_MEAN, AUGMENTED_6COND",
        "R6": "screen candidate 6: [FIRST_TOKEN;MASKED_MEAN], AUGMENTED_6COND",
        "F": "FOCAL_GAMMA_2",
    }
    for historical, descriptive in expected.items():
        assert aliases[historical]["descriptive_name"] == descriptive
        assert aliases[historical]["public_api"] is False


def test_no_public_api_exposes_the_screened_readouts_or_focal_loss():
    import unmark.viunmark as viunmark
    from unmark.viunmark.config import LossKind, ReadoutKind

    import re

    exported = " ".join(viunmark.__all__)
    assert re.search(r"\bOPT\d", exported) is None
    assert "Focal" not in exported and "FOCAL" not in exported
    assert "L2_MASKED_MEAN" not in {k.name for k in ReadoutKind}
    assert "FOCAL" not in {k.name for k in LossKind}


def test_loader_rejects_a_misstated_screen(tmp_path, monkeypatch):
    def edit(spec):
        spec["uit_vsfc_reproduction"]["development_stages"][
            "stage2_class_balance_loss_optimization"]["promoted_screen_candidates"] = [4, 5]
    write_spec(tmp_path, monkeypatch, edit)
    with pytest.raises(ViUnMarkContractError, match="class-balance"):
        provenance.load_uit_vsfc_reproduction()
