"""Contracts for the post-diagnostic PhoNER Stage-II optimization funnel."""

from __future__ import annotations

import ast
import json
import importlib
import pathlib
import types

import pytest

from unmark.cross_task import phoner_stage2_optimization as opt
from unmark.cross_task import phoner_transfer as phoner

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = importlib.import_module("scripts.cross_task.run_phoner_stage2_optimization")

try:  # pragma: no cover - environment dependent
    import torch

    TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(not TORCH, reason="torch not installed")


def test_protocol_is_post_diagnostic_funnel_with_test_disabled():
    cfg = opt.OptimizedPhoNERStage2Config()
    payload = cfg.to_dict()
    assert payload["namespace"] == "phoner_stage2_optimized"
    assert payload["post_diagnostic_development"] is True
    assert payload["test_enabled"] is False
    assert payload["encoder_checkpoint"] == "vinai/phobert-base"
    assert payload["encoder_revision"] == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert payload["gate_checkpoint_sha256"] == phoner.PHONER_GATE_SHA256
    assert payload["scale_checkpoint_sha256"] == phoner.PHONER_SCALE_SHA256
    assert payload["negative_controls"]["crf_used"] is False
    assert payload["negative_controls"]["sentiment_bias_or_calibration_used"] is False
    assert payload["negative_controls"]["best_seed_selection_used"] is False


def test_stage_boundaries_are_the_fixed_research_funnel():
    assert [stage.value for stage in opt.OPTIMIZED_STAGE_ORDER] == [
        "protocol",
        "protocol-amend",
        "build-bank",
        "d1-train",
        "d1-select",
        "d2-decode",
        "d3-train",
        "d3-select",
        "d4-analyze",
        "sys1-evaluate",
        "sys2-1-train",
        "sys2-1-select",
        "sys2-2-evaluate",
        "comp-d1",
        "comp-d2",
        "freeze-final",
    ]


def test_d1_candidate_family_is_exact_and_scale_only():
    family = opt.d1_candidate_family()
    assert [recipe.recipe_id for recipe in family] == ["D1-A-NER", "D1-B-NER", "D1-C-NER"]
    assert [recipe.pathway for recipe in family] == [phoner.PhoNERPathway.VIUNMARK_SCALE] * 3
    assert family[0].train_conditions == ("FULL",)
    assert family[0].loss_weighting is opt.LossWeighting.UNWEIGHTED
    assert family[1].train_conditions == phoner.SIX_CONDITIONS
    assert family[1].loss_weighting is opt.LossWeighting.UNWEIGHTED
    assert family[2].train_conditions == phoner.SIX_CONDITIONS
    assert family[2].loss_weighting is opt.LossWeighting.SQRT_INVERSE_FREQUENCY


def test_weighted_ce_formula_is_deterministic_sqrt_inverse_frequency():
    weights = opt.first_subtoken_label_weights([0, 0, 0, 0, 1, 1, 2], num_labels=3)
    raw = [1 / 2, 1 / (2 ** 0.5), 1]
    mean = sum(raw) / 3
    assert weights == pytest.approx(tuple(value / mean for value in raw), abs=1e-12)
    with pytest.raises(phoner.PhoNERContractViolation):
        opt.first_subtoken_label_weights([0, 0, 1], num_labels=3)


def test_d2_decoder_family_has_no_learned_parameters():
    assert opt.d2_candidate_family() == (
        opt.DecodePolicy.ARGMAX_IOB2_REPAIR,
        opt.DecodePolicy.HARD_BIO_VITERBI,
    )
    assert opt.legal_bio_transition(None, "O")
    assert opt.legal_bio_transition(None, "B-PER")
    assert not opt.legal_bio_transition(None, "I-PER")
    assert not opt.legal_bio_transition("O", "I-PER")
    assert opt.legal_bio_transition("B-PER", "I-PER")
    assert not opt.legal_bio_transition("B-ORG", "I-PER")
    assert opt.legal_bio_transition("I-ORG", "O")
    assert opt.legal_bio_transition("I-ORG", "B-PER")


def test_hard_bio_viterbi_forbids_illegal_i_transitions():
    id_to_label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG"}
    logits = [
        [0.0, 2.0, 9.0, 0.0, 0.0],  # I-PER would win by argmax but is illegal at start.
        [0.0, 0.0, 8.0, 0.0, 9.0],  # I-ORG after B-PER is illegal.
    ]
    decoded = opt.hard_bio_constrained_viterbi(logits, id_to_label)
    assert decoded == (1, 2)
    assert opt.is_valid_bio_sequence([id_to_label[index] for index in decoded])


def test_d3_matched_recipe_applies_d1_readout_to_all_pathways():
    selected = opt.d1_candidate_family()[2]
    recipes = opt.matched_d3_recipes(selected)
    assert [recipe.pathway for recipe in recipes] == [
        phoner.PhoNERPathway.PHOBERT_NATIVE,
        phoner.PhoNERPathway.VIUNMARK_GATE,
        phoner.PhoNERPathway.VIUNMARK_SCALE,
    ]
    assert {recipe.train_conditions for recipe in recipes} == {phoner.SIX_CONDITIONS}
    assert {recipe.loss_weighting for recipe in recipes} == {opt.LossWeighting.SQRT_INVERSE_FREQUENCY}


def test_d4_stage_is_no_training_in_runner(tmp_path):
    cfg = opt.OptimizedPhoNERStage2Config()
    args = types.SimpleNamespace(output_root=tmp_path)
    RUNNER.stage_protocol(args, cfg)
    RUNNER.write_json(
        RUNNER.amended_protocol_path(tmp_path, cfg),
        {
            "schema_version": opt.PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA,
            "config_digest": opt.stable_digest(cfg.to_dict()),
            "test_enabled": False,
            "expected_training_schedule_closure_for_current_corpus": opt.training_schedule_closure(5027),
        },
    )
    RUNNER.write_json(
        RUNNER.representation_bank_manifest_path(tmp_path, cfg),
        {"schema_version": opt.PHONER_STAGE2_OPT_BANK_SCHEMA},
    )
    RUNNER.write_json(
        RUNNER.training_schedule_closure_path(tmp_path, cfg),
        opt.training_schedule_closure(5027),
    )
    RUNNER.stage_d1_train(args, cfg)
    RUNNER.stage_d1_select(args, cfg)
    RUNNER.stage_d2_decode(args, cfg)
    RUNNER.stage_d3_train(args, cfg)
    RUNNER.stage_d3_select(args, cfg)
    RUNNER.stage_d4_analyze(args, cfg)
    payload = RUNNER.read_json(RUNNER.stage_artifact_path(tmp_path, cfg, opt.Stage2Stage.D4_ANALYZE, "d4_complementarity.json"))
    assert payload["training_performed"] is False


def test_d4_entity_key_includes_sample_id_to_prevent_cross_sentence_collision():
    first = opt.d4_entity_key("sample-a", 0, 2, "PER")
    second = opt.d4_entity_key("sample-b", 0, 2, "PER")
    assert first != second
    assert first == ("sample-a", 0, 2, "PER")


@requires_torch
def test_mlp_head_architecture_shape_and_dropout():
    head = opt.build_optimized_mlp_head(7, seed=53148)
    modules = list(head)
    assert isinstance(modules[0], torch.nn.LayerNorm)
    assert isinstance(modules[1], torch.nn.Linear)
    assert modules[1].in_features == 768
    assert modules[1].out_features == 256
    assert modules[3].p == pytest.approx(0.1)
    assert isinstance(modules[4], torch.nn.Linear)
    assert modules[4].out_features == 7


@requires_torch
def test_exactly_five_head_branch_ensemble_mean():
    logits = {seed: torch.tensor([float(index)]) for index, seed in enumerate(phoner.PHONER_FINAL_SEEDS)}
    assert opt.mean_five_head_logits(logits).item() == pytest.approx(2.0)
    incomplete = dict(list(logits.items())[:4])
    with pytest.raises(phoner.PhoNERContractViolation):
        opt.mean_five_head_logits(incomplete)


def test_sys1_beta_grid_and_sys2_2_gamma_grid_are_fixed():
    sys1 = opt.sys1_beta_grid()
    assert [candidate.weights for candidate in sys1] == [(0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
    assert all(candidate.branches == ("GATE_RAW5", "SCALE_RAW5") for candidate in sys1)
    assert all(candidate.decode_once_after_fusion for candidate in sys1)
    sys2 = opt.sys2_2_gamma_grid()
    assert [candidate.weights for candidate in sys2] == [(0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
    assert all(candidate.branches == ("NATIVE_RAW5", "ADAPTED_SYS1_RAW") for candidate in sys2)
    assert all(candidate.decode_once_after_fusion for candidate in sys2)


def test_sys2_1_weighted_native_candidate_is_rule_gated_by_d1():
    base = opt.sys2_1_native_candidates(d1_weighted_survives=False)
    expanded = opt.sys2_1_native_candidates(d1_weighted_survives=True)
    assert [recipe.recipe_id for recipe in base] == [
        "SYS2-1-NER-NATIVE-FULL-UNWEIGHTED",
        "SYS2-1-NER-NATIVE-AUG6-UNWEIGHTED",
    ]
    assert len(expanded) == 3
    assert expanded[-1].loss_weighting is opt.LossWeighting.SQRT_INVERSE_FREQUENCY


def test_no_best_seed_selection_and_five_seed_aggregate_summary():
    rows = []
    for seed_offset, seed in enumerate(phoner.PHONER_FINAL_SEEDS):
        for condition in phoner.SIX_CONDITIONS:
            value = 0.5 + 0.01 * seed_offset
            rows.append(
                {
                    "candidate_id": "D1-B-NER",
                    "seed": seed,
                    "condition": condition,
                    "entity_precision": value,
                    "entity_recall": value,
                    "entity_micro_f1": value,
                    "entity_true_positive": 1,
                    "entity_false_positive": 0,
                    "entity_false_negative": 0,
                }
            )
    summary = opt.summarize_optimized_candidate_scores(rows)
    robust = summary["robustness"]["D1-B-NER"]
    assert robust["seeds"] == sorted(phoner.PHONER_FINAL_SEEDS)
    assert robust["all_6_f1_mean"] == pytest.approx(0.52)
    assert robust["all_6_f1_sample_sd"] > 0
    assert "no best-seed selection" in summary["primary_policy"]


def test_fusion_selection_uses_deployed_ensemble_rows_not_per_seed_averages():
    ensemble_rows = [
        {"condition": condition, "entity_micro_f1": 0.5}
        for condition in phoner.SIX_CONDITIONS
    ]
    assert opt.ensemble_fusion_selection_tuple(ensemble_rows, default_rank=1) == (0.5, 0.5, 0.5, 1)
    per_seed_rows = [dict(ensemble_rows[0], seed=phoner.PHONER_FINAL_SEEDS[0])]
    with pytest.raises(phoner.PhoNERContractViolation, match="deployed ensemble"):
        opt.ensemble_fusion_selection_tuple(per_seed_rows)


def test_aggregate_selection_prefers_mean_all6_then_worst_then_full_then_simplicity():
    weak = {"all_6_f1_mean": 0.7, "worst_condition_f1_mean": 0.4, "FULL_f1_mean": 0.9}
    robust = {"all_6_f1_mean": 0.7, "worst_condition_f1_mean": 0.5, "FULL_f1_mean": 0.6}
    assert opt.aggregate_selection_tuple(robust, branch_count=2) > opt.aggregate_selection_tuple(weak, branch_count=1)
    assert opt.aggregate_selection_tuple(robust, branch_count=1) > opt.aggregate_selection_tuple(robust, branch_count=2)


def test_representation_bank_identity_excludes_test_and_fails_on_mismatch():
    sample_digest = opt.sample_ids_digest(["a", "b"])
    identity = opt.RepresentationBankIdentity(
        split="train",
        dataset_sha256="dataset",
        sample_ids_sha256=sample_digest,
        pathway=phoner.PhoNERPathway.VIUNMARK_SCALE,
        condition="P50",
        stage1_checkpoint_sha256=phoner.PHONER_SCALE_SHA256,
    )
    opt.verify_representation_bank_identity(identity, identity.to_dict())
    altered = identity.to_dict()
    altered["condition"] = "P75"
    with pytest.raises(phoner.PhoNERContractViolation):
        opt.verify_representation_bank_identity(identity, altered)
    with pytest.raises(phoner.PhoNERContractViolation):
        opt.RepresentationBankIdentity(
            split="test",
            dataset_sha256="dataset",
            sample_ids_sha256=sample_digest,
            pathway=phoner.PhoNERPathway.PHOBERT_NATIVE,
            condition="FULL",
            stage1_checkpoint_sha256=None,
        )


def test_training_budget_is_distribution_specific_five_complete_passes():
    full = opt.derive_training_budget(5027, ("FULL",))
    assert full.distribution is opt.TrainingDistribution.FULL
    assert full.examples_per_pass == 5027
    assert full.updates_per_complete_distribution_pass == 315
    assert "updates_per_complete_aug6_pass" not in full.to_dict()
    assert full.complete_distribution_passes == 5
    assert full.max_optimizer_updates == 1575
    assert len(full.boundary_updates) == 30
    assert list(full.boundary_updates) == sorted(set(full.boundary_updates))
    assert full.boundary_updates[-1] == 1575

    aug6 = opt.derive_training_budget(5027, phoner.SIX_CONDITIONS)
    assert aug6.distribution is opt.TrainingDistribution.AUG6
    assert aug6.examples_per_pass == 30162
    assert aug6.updates_per_complete_distribution_pass == 1886
    assert aug6.complete_distribution_passes == 5
    assert aug6.max_optimizer_updates == 9430
    assert len(aug6.boundary_updates) == 30
    assert list(aug6.boundary_updates) == sorted(set(aug6.boundary_updates))
    assert aug6.boundary_updates[-1] == 9430


def test_training_schedule_closure_fails_closed_on_unexpected_chunk_count():
    closure = opt.training_schedule_closure(5027)
    assert closure["schema_version"] == opt.PHONER_STAGE2_OPT_TRAINING_SCHEDULE_SCHEMA
    assert closure["FULL"]["max_optimizer_updates"] == 1575
    assert closure["AUG6"]["max_optimizer_updates"] == 9430
    assert closure["FULL"]["boundary_updates"][-1] == 1575
    assert closure["AUG6"]["boundary_updates"][-1] == 9430
    with pytest.raises(phoner.PhoNERContractViolation):
        opt.training_schedule_closure(5028)


def test_protocol_contract_sections_are_materialized():
    cfg = opt.OptimizedPhoNERStage2Config()
    payload = cfg.to_dict()
    assert payload["representation_bank_contract"]["splits"] == ["train", "dev"]
    assert payload["representation_bank_contract"]["test_excluded"] is True
    assert payload["representation_bank_contract"]["dtype"] == "float32"
    assert payload["d2_hard_bio_contract"]["path_score"] == "sum of raw emission logits only"
    assert payload["d2_hard_bio_contract"]["decoder_checkpoint"] is False
    assert payload["sys2_1_admission_contract"]["later_ad_hoc_admission"] is False
    assert payload["scientific_head_reuse_contract"]["fail_closed_on_identity_mismatch"] is True
    assert payload["d4_entity_contract"]["tuple_form"] == "(sample_id, word_start, word_end, entity_type)"
    assert payload["fusion_selection_contract"]["not_input"] == "mean of five independent per-seed F1 values"


def test_exact_head_identity_requires_byte_for_byte_reuse():
    budget = opt.derive_training_budget(5027, phoner.SIX_CONDITIONS)
    identity = opt.ScientificHeadIdentity(
        pathway=phoner.PhoNERPathway.VIUNMARK_SCALE,
        representation_identity_digest="repr",
        representation_bank_digest="bank",
        head_architecture=opt.MLP_ARCHITECTURE,
        training_distribution=opt.TrainingDistribution.AUG6,
        train_conditions=phoner.SIX_CONDITIONS,
        loss_definition="unweighted CE",
        loss_weights_digest=None,
        optimizer_hyperparameters_digest=opt.stable_digest({"optimizer": "AdamW"}),
        training_budget_digest=opt.stable_digest(budget.to_dict()),
        seed=phoner.PHONER_FINAL_SEEDS[0],
        checkpoint_selection_semantics=opt.OptimizedTrainingPolicy().per_head_selection,
    )
    closed = {
        "scientific_head_identity": identity.to_dict(),
        "scientific_head_identity_digest": identity.digest,
        "checkpoint_sha256": "abc123",
    }
    assert opt.require_exact_head_reuse(identity, closed) == "abc123"

    mismatched = opt.ScientificHeadIdentity(
        pathway=phoner.PhoNERPathway.VIUNMARK_SCALE,
        representation_identity_digest="repr",
        representation_bank_digest="different-bank",
        head_architecture=opt.MLP_ARCHITECTURE,
        training_distribution=opt.TrainingDistribution.AUG6,
        train_conditions=phoner.SIX_CONDITIONS,
        loss_definition="unweighted CE",
        loss_weights_digest=None,
        optimizer_hyperparameters_digest=opt.stable_digest({"optimizer": "AdamW"}),
        training_budget_digest=opt.stable_digest(budget.to_dict()),
        seed=phoner.PHONER_FINAL_SEEDS[0],
        checkpoint_selection_semantics=opt.OptimizedTrainingPolicy().per_head_selection,
    )
    with pytest.raises(phoner.PhoNERContractViolation, match="does not match"):
        opt.require_exact_head_reuse(mismatched, closed)


def test_runner_writes_only_optimized_namespace_and_not_old_diagnostic_targets(tmp_path):
    cfg = opt.OptimizedPhoNERStage2Config()
    assert RUNNER.namespace_root(tmp_path, cfg) == tmp_path / "phoner_stage2_optimized"
    paths = [
        RUNNER.protocol_path(tmp_path, cfg),
        RUNNER.amended_protocol_path(tmp_path, cfg),
        RUNNER.representation_bank_manifest_path(tmp_path, cfg),
        RUNNER.training_schedule_closure_path(tmp_path, cfg),
        RUNNER.final_freeze_path(tmp_path, cfg),
    ]
    for path in paths:
        assert path.is_relative_to(tmp_path / "phoner_stage2_optimized")
    forbidden = {
        "frozen_protocol.json",
        "frozen_probe_heads.json",
        "train_dev_results.json",
        "dev_evaluate_results.json",
    }
    assert not forbidden & {path.name for path in paths}


def _write_phoner_conll(path: pathlib.Path, rows: int) -> None:
    path.write_text(("a O\n\n" * rows), encoding="utf-8")


def test_protocol_amendment_preserves_v2_and_binds_head_train_dev_and_parent(monkeypatch, tmp_path):
    cfg = opt.OptimizedPhoNERStage2Config()
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_phoner_conll(data_root / "train_word.conll", 5027)
    _write_phoner_conll(data_root / "dev_word.conll", 2000)
    args = types.SimpleNamespace(output_root=tmp_path, data_root=data_root)
    RUNNER.stage_protocol(args, cfg)
    v2_path = RUNNER.protocol_path(tmp_path, cfg)
    v2_before = v2_path.read_bytes()
    RUNNER.write_json(tmp_path / "frozen_protocol.json", {"schema_version": "audit076", "protocol_digest": "parent"})
    monkeypatch.setattr(RUNNER, "require_clean_execution_tree", lambda: ("HEAD123", True))

    RUNNER.stage_protocol_amend(args, cfg)

    assert v2_path.read_bytes() == v2_before
    amended = RUNNER.read_json(RUNNER.amended_protocol_path(tmp_path, cfg))
    assert amended["schema_version"] == opt.PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA
    assert amended["supersedes"]["sha256"] == RUNNER.sha256_bytes(v2_before)
    assert amended["execution_repository_head"] == "HEAD123"
    assert amended["clean_execution_tree"] is True
    assert amended["dataset_provenance"]["splits"]["train"]["row_count"] == 5027
    assert amended["dataset_provenance"]["splits"]["dev"]["row_count"] == 2000
    assert amended["dataset_provenance"]["test_read"] is False
    assert amended["audit076_parent_identity"]["protocol_digest"] == "parent"
    assert amended["no_new_campaign_scientific_results_observed_between_v2_and_v3"] is True

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        RUNNER.stage_protocol_amend(args, cfg)


def test_build_bank_and_d1_train_require_amended_protocol_and_schedule(tmp_path):
    cfg = opt.OptimizedPhoNERStage2Config()
    args = types.SimpleNamespace(output_root=tmp_path)
    RUNNER.stage_protocol(args, cfg)
    with pytest.raises(SystemExit, match="required optimized artifact is missing"):
        RUNNER.stage_build_bank(args, cfg)

    RUNNER.write_json(
        RUNNER.amended_protocol_path(tmp_path, cfg),
        {
            "schema_version": opt.PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA,
            "config_digest": opt.stable_digest(cfg.to_dict()),
            "test_enabled": False,
            "expected_training_schedule_closure_for_current_corpus": opt.training_schedule_closure(5027),
        },
    )
    with pytest.raises(SystemExit, match="build-bank is fail-closed"):
        RUNNER.stage_build_bank(args, cfg)
    RUNNER.write_json(
        RUNNER.representation_bank_manifest_path(tmp_path, cfg),
        {"schema_version": opt.PHONER_STAGE2_OPT_BANK_SCHEMA},
    )
    RUNNER.write_json(
        RUNNER.training_schedule_closure_path(tmp_path, cfg),
        opt.training_schedule_closure(5027),
    )
    RUNNER.stage_d1_train(args, cfg)


def test_test_stages_are_disabled(tmp_path):
    args = types.SimpleNamespace(output_root=tmp_path)
    cfg = opt.OptimizedPhoNERStage2Config()
    with pytest.raises(SystemExit, match="TEST stages are disabled"):
        RUNNER.run_stage("test-predict", args, cfg)
    with pytest.raises(SystemExit, match="TEST stages are disabled"):
        RUNNER.run_stage("test-score", args, cfg)


def test_runner_exposes_new_stages_without_direct_fusion_campaign_or_personal_paths():
    source = (REPO / "scripts/cross_task/run_phoner_stage2_optimization.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert [stage.value for stage in RUNNER.Stage2Stage] == [
        stage.value for stage in opt.Stage2Stage
    ]
    assert "stage.value for stage in Stage2Stage" in source
    assert "FUSE_NATIVE_GATE_075_025" not in source
    assert "dev_evaluate_results.json" not in source
    assert "frozen_probe_heads.json" not in source
    assert "frozen_probe_heads.json" not in source
    assert "train_dev_results.json" not in source
    assert "dev_evaluate_results.json" not in source
    assert "/content/drive" not in source
    assert "MyDrive" not in source
