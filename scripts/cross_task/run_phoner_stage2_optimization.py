#!/usr/bin/env python
"""Post-diagnostic PhoNER Stage-II optimization funnel runner.

The runner is deliberately separate from the frozen Audit 076 diagnostic and
writes only under `phoner_stage2_optimized/`. TEST stages remain disabled until
SYS2-2 and the baseline protocols are frozen in a later task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from unmark.cross_task.phoner_stage2_optimization import (
    OPTIMIZED_STAGE_ORDER,
    PHONER_STAGE2_OPT_BANK_SCHEMA,
    PHONER_STAGE2_OPT_FINAL_SCHEMA,
    PHONER_STAGE2_OPT_NAMESPACE,
    PHONER_STAGE2_OPT_PROTOCOL_V2_SCHEMA,
    PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA,
    PHONER_STAGE2_OPT_RESULT_SCHEMA,
    PHONER_STAGE2_OPT_TRAINING_SCHEDULE_SCHEMA,
    DecodePolicy,
    OptimizedPhoNERStage2Config,
    RepresentationBankIdentity,
    Stage2Stage,
    d1_candidate_family,
    d2_candidate_family,
    derive_training_budget,
    d2_transition_contract,
    d4_entity_contract,
    fusion_selection_contract,
    head_reuse_contract,
    matched_d3_recipes,
    representation_bank_contract,
    stable_digest,
    sys1_beta_grid,
    sys2_1_admission_contract,
    sys2_1_native_candidates,
    sys2_2_gamma_grid,
    training_schedule_closure,
)
from unmark.cross_task.phoner_transfer import (
    PHONER_FINAL_SEEDS,
    PHONER_GATE_SHA256,
    PHONER_SCALE_SHA256,
    PhoNERContractViolation,
    PhoNERPathway,
    PhoNERSplit,
    find_phoner_split_file,
    parse_phoner_split,
    sha256_file,
)
from unmark.viunmark.config import SIX_CONDITIONS

OFFICIAL_AUDITED_TRAIN_ROWS = 5027
OFFICIAL_AUDITED_DEV_ROWS = 2000
TEST_STAGES = {Stage2Stage.TEST_PREDICT.value, Stage2Stage.TEST_SCORE.value}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing optimized artifact: {path}")
    write_json(path, payload)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"required optimized artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def namespace_root(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return Path(output_root) / config.namespace


def stage_root(output_root: str | Path, config: OptimizedPhoNERStage2Config, stage: Stage2Stage | str) -> Path:
    value = stage.value if isinstance(stage, Stage2Stage) else str(stage)
    return namespace_root(output_root, config) / value.replace("-", "_")


def protocol_path(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return namespace_root(output_root, config) / "umbrella_post_diagnostic_protocol.json"


def amended_protocol_path(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return namespace_root(output_root, config) / "umbrella_post_diagnostic_protocol_v3_amendment.json"


def audit076_frozen_protocol_path(output_root: str | Path) -> Path:
    return Path(output_root) / "frozen_protocol.json"


def stage_artifact_path(
    output_root: str | Path,
    config: OptimizedPhoNERStage2Config,
    stage: Stage2Stage,
    name: str,
) -> Path:
    return stage_root(output_root, config, stage) / name


def representation_bank_manifest_path(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return stage_artifact_path(output_root, config, Stage2Stage.BUILD_BANK, "representation_bank_manifest.json")


def training_schedule_closure_path(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return stage_artifact_path(output_root, config, Stage2Stage.BUILD_BANK, "training_schedule_closure.json")


def final_freeze_path(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Path:
    return stage_artifact_path(output_root, config, Stage2Stage.FREEZE_FINAL, "final_freeze.json")


def require_test_disabled(stage: str) -> None:
    if stage in TEST_STAGES:
        raise SystemExit("PhoNER optimized Stage-II TEST stages are disabled; TEST remains sealed")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def repository_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository_root(),
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def require_clean_execution_tree() -> tuple[str, bool]:
    head = repository_head()
    status = repository_status_short()
    if status.strip():
        raise SystemExit("protocol amendment requires a clean execution tree; git status --short:\n" + status.rstrip())
    return head, True


def require_protocol(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> Mapping[str, Any]:
    payload = read_json(amended_protocol_path(output_root, config))
    if payload.get("schema_version") != PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA:
        raise SystemExit("build-bank requires the v3 amended/final protocol artifact")
    if payload.get("config_digest") != stable_digest(config.to_dict()):
        raise SystemExit("optimized protocol digest mismatch")
    if payload.get("test_enabled") is not False:
        raise SystemExit("optimized protocol must keep TEST disabled")
    return payload


def require_artifact(path: Path, *, schema: str | None = None) -> Mapping[str, Any]:
    payload = read_json(path)
    if schema is not None and payload.get("schema_version") != schema:
        raise SystemExit(f"wrong schema in {path}")
    return payload


def official_expected_budget(config: OptimizedPhoNERStage2Config) -> dict[str, Any]:
    return {
        "FULL": derive_training_budget(OFFICIAL_AUDITED_TRAIN_ROWS, ("FULL",), config.policy).to_dict(),
        "AUG6": derive_training_budget(OFFICIAL_AUDITED_TRAIN_ROWS, SIX_CONDITIONS, config.policy).to_dict(),
    }


def train_dev_dataset_provenance(data_root: str | Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for split in (PhoNERSplit.TRAIN, PhoNERSplit.DEV):
        path, file_format = find_phoner_split_file(data_root, split)
        examples = parse_phoner_split(data_root, split, include_labels=True, stage="dataset-audit")
        rows[split.value] = {
            "split": split.value,
            "path_name": path.name,
            "format": file_format,
            "sha256": sha256_file(path),
            "row_count": len(examples),
            "gold_labels_read": True,
        }
    if rows["train"]["row_count"] != OFFICIAL_AUDITED_TRAIN_ROWS:
        raise SystemExit(f"unexpected PhoNER TRAIN row count: {rows['train']['row_count']}")
    if rows["dev"]["row_count"] != OFFICIAL_AUDITED_DEV_ROWS:
        raise SystemExit(f"unexpected PhoNER DEV row count: {rows['dev']['row_count']}")
    return {
        "dataset_id": "PhoNER_COVID19",
        "representation": "word",
        "source": "official PhoNER_COVID19 word-level local release supplied via DATA_ROOT",
        "source_revision": "external DATA_ROOT; no dataset content committed",
        "splits": rows,
        "test_read": False,
    }


def parent_v2_identity(output_root: str | Path, config: OptimizedPhoNERStage2Config) -> dict[str, Any]:
    path = protocol_path(output_root, config)
    data = path.read_bytes()
    payload = json.loads(data.decode("utf-8"))
    if payload.get("schema_version") != PHONER_STAGE2_OPT_PROTOCOL_V2_SCHEMA:
        raise SystemExit("v2 umbrella protocol parent has wrong schema")
    return {
        "path": str(path.relative_to(Path(output_root))),
        "schema_version": payload.get("schema_version"),
        "sha256": sha256_bytes(data),
    }


def audit076_parent_identity(output_root: str | Path) -> dict[str, Any]:
    path = audit076_frozen_protocol_path(output_root)
    if not path.exists():
        raise SystemExit("Audit-076 frozen_protocol.json parent identity is required before protocol amendment")
    data = path.read_bytes()
    payload = json.loads(data.decode("utf-8"))
    return {
        "path": path.name,
        "sha256": sha256_bytes(data),
        "schema_version": payload.get("schema_version"),
        "protocol_digest": payload.get("protocol_digest"),
        "config_digest": stable_digest(payload.get("config", {})) if isinstance(payload.get("config"), Mapping) else None,
    }


def stage_layout(config: OptimizedPhoNERStage2Config) -> dict[str, Any]:
    return {
        "namespace": config.namespace,
        "stage_order": [stage.value for stage in OPTIMIZED_STAGE_ORDER],
        "immutable_artifact_policy": "each stage writes its own namespaced artifact once",
        "test_stages": sorted(TEST_STAGES),
        "test_stage_policy": "disabled/fail-closed until SYS2-2 and baselines are frozen",
        "diagnostic_artifact_policy": "Audit 076 artifacts are read-only historical inputs and are not targets",
    }


def planned_representation_banks(config: OptimizedPhoNERStage2Config) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev"):
        for pathway in config.pathways:
            for condition in config.conditions:
                sha = None
                if pathway is PhoNERPathway.VIUNMARK_GATE:
                    sha = PHONER_GATE_SHA256
                elif pathway is PhoNERPathway.VIUNMARK_SCALE:
                    sha = PHONER_SCALE_SHA256
                rows.append(
                    RepresentationBankIdentity(
                        split=split,
                        dataset_sha256="FILLED_BY_BUILD_BANK_STAGE",
                        sample_ids_sha256="FILLED_BY_BUILD_BANK_STAGE",
                        pathway=pathway,
                        condition=condition,
                        stage1_checkpoint_sha256=sha,
                    ).to_dict()
                )
    return rows


def stage_protocol(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    payload = {
        "schema_version": PHONER_STAGE2_OPT_PROTOCOL_V2_SCHEMA,
        "config": config.to_dict(),
        "config_digest": stable_digest(config.to_dict()),
        "stage_layout": stage_layout(config),
        "official_train_rows": OFFICIAL_AUDITED_TRAIN_ROWS,
        "official_dev_rows": OFFICIAL_AUDITED_DEV_ROWS,
        "expected_distribution_specific_budgets_if_no_train_chunking": official_expected_budget(config),
        "post_diagnostic_warning": (
            "This is post-diagnostic development after corrupted DEV observation in Audit 076; "
            "it is not a pre-observation final protocol freeze."
        ),
        "test_enabled": False,
        "test_read": False,
        "test_scored": False,
        "stage1_retraining": False,
        "uit_vsfc_retuning": False,
    }
    write_json_once(protocol_path(args.output_root, config), payload)
    for stage in OPTIMIZED_STAGE_ORDER:
        stage_root(args.output_root, config, stage).mkdir(parents=True, exist_ok=True)


def stage_protocol_amend(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    output_root = Path(args.output_root)
    head, clean = require_clean_execution_tree()
    v2_parent = parent_v2_identity(output_root, config)
    dataset = train_dev_dataset_provenance(args.data_root)
    audit076_parent = audit076_parent_identity(output_root)
    payload = {
        "schema_version": PHONER_STAGE2_OPT_PROTOCOL_V3_SCHEMA,
        "supersedes": v2_parent,
        "amendment_reason": (
            "Strengthen execution, dataset, parent, representation-bank, schedule, "
            "decoder, SYS2-1, head-reuse, D4, and fusion-selection provenance before build-bank."
        ),
        "no_new_campaign_scientific_results_observed_between_v2_and_v3": True,
        "execution_repository_head": head,
        "clean_execution_tree": clean,
        "config": config.to_dict(),
        "config_digest": stable_digest(config.to_dict()),
        "dataset_provenance": dataset,
        "audit076_parent_identity": audit076_parent,
        "stage_layout": stage_layout(config),
        "official_train_rows": OFFICIAL_AUDITED_TRAIN_ROWS,
        "official_dev_rows": OFFICIAL_AUDITED_DEV_ROWS,
        "expected_distribution_specific_budgets_if_no_train_chunking": official_expected_budget(config),
        "representation_bank_contract": representation_bank_contract(config),
        "training_schedule_closure_required_before_training": True,
        "expected_training_schedule_closure_for_current_corpus": training_schedule_closure(
            OFFICIAL_AUDITED_TRAIN_ROWS, policy=config.policy
        ),
        "d2_hard_bio_contract": d2_transition_contract(),
        "sys2_1_admission_contract": sys2_1_admission_contract(),
        "scientific_head_reuse_contract": head_reuse_contract(),
        "d4_entity_contract": d4_entity_contract(),
        "fusion_selection_contract": fusion_selection_contract(),
        "test_enabled": False,
        "test_read": False,
        "test_scored": False,
        "stage1_retraining": False,
        "uit_vsfc_retuning": False,
    }
    write_json_once(amended_protocol_path(output_root, config), payload)


def stage_build_bank(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    protocol = require_protocol(args.output_root, config)
    closure = protocol.get("expected_training_schedule_closure_for_current_corpus")
    if not isinstance(closure, Mapping) or closure.get("schema_version") != PHONER_STAGE2_OPT_TRAINING_SCHEDULE_SCHEMA:
        raise SystemExit("amended protocol is missing training schedule closure")
    raise SystemExit(
        "build-bank is fail-closed in this contract audit; a later real bank materializer must "
        "write actual bank files, bank SHA256 values, stream digests, and the training schedule closure"
    )


def stage_d1_train(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_protocol(args.output_root, config)
    require_artifact(representation_bank_manifest_path(args.output_root, config), schema=PHONER_STAGE2_OPT_BANK_SCHEMA)
    require_artifact(
        training_schedule_closure_path(args.output_root, config),
        schema=PHONER_STAGE2_OPT_TRAINING_SCHEDULE_SCHEMA,
    )
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D1_TRAIN.value,
        "candidates": [recipe.to_dict() for recipe in d1_candidate_family()],
        "pathway": PhoNERPathway.VIUNMARK_SCALE.value,
        "seeds": list(PHONER_FINAL_SEEDS),
        "budget_derivation": (
            "derive max updates independently for each candidate as five complete passes "
            "over that candidate's frozen training distribution"
        ),
        "status": "training artifact placeholder; real execution must write selected head checkpoints in this stage namespace",
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D1_TRAIN, "d1_train_plan.json"), payload)


def stage_d1_select(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D1_TRAIN, "d1_train_plan.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D1_SELECT.value,
        "selection": list(config.policy.recipe_selection),
        "no_best_seed_selection": True,
        "post_diagnostic_development": True,
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D1_SELECT, "d1_selection.json"), payload)


def stage_d2_decode(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D1_SELECT, "d1_selection.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D2_DECODE.value,
        "decoder_candidates": [policy.value for policy in d2_candidate_family()],
        "learned_transition_parameters": 0,
        "checkpoint_created": False,
        "same_logits_as_d1_selected_stream": True,
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D2_DECODE, "d2_decoding_selection.json"), payload)


def stage_d3_train(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D2_DECODE, "d2_decoding_selection.json"))
    d1_default = d1_candidate_family()[1]
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D3_TRAIN.value,
        "matched_recipe_template": "actual selected D1 recipe artifact is authoritative at execution time",
        "static_recipe_shape": [recipe.to_dict() for recipe in matched_d3_recipes(d1_default)],
        "pathways": [pathway.value for pathway in config.pathways],
        "seeds": list(PHONER_FINAL_SEEDS),
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D3_TRAIN, "d3_train_plan.json"), payload)


def stage_d3_select(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D3_TRAIN, "d3_train_plan.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D3_SELECT.value,
        "controlled_representation_comparison": True,
        "selection": list(config.policy.recipe_selection),
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D3_SELECT, "d3_selection.json"), payload)


def stage_d4_analyze(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D3_SELECT, "d3_selection.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.D4_ANALYZE.value,
        "training_performed": False,
        "analyses": [
            "exact-entity correct sets",
            "Native-only/Gate-only/Scale-only correct entities",
            "pairwise correctness overlap and disagreement",
            "boundary-error counts",
            "entity-type-error counts",
            "pathway logit/prediction disagreement where well-defined",
        ],
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.D4_ANALYZE, "d4_complementarity.json"), payload)


def stage_sys1(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.D4_ANALYZE, "d4_complementarity.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.SYS1_EVALUATE.value,
        "branch_rule": "Gate_raw and Scale_raw are each arithmetic means of exactly five head logits",
        "beta_grid": [candidate.to_dict() for candidate in sys1_beta_grid()],
        "sentiment_bias_or_calibration_used": False,
        "decode_once_after_fusion": True,
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.SYS1_EVALUATE, "sys1_adapted_fusion.json"), payload)


def stage_sys2_1_train(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.SYS1_EVALUATE, "sys1_adapted_fusion.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.SYS2_1_TRAIN.value,
        "candidate_rule": (
            "Native weighted candidate is admitted only if the closed D1 selection artifact "
            "keeps the weighted readout alive under the five-seed aggregate rule"
        ),
        "minimum_candidates": [recipe.to_dict() for recipe in sys2_1_native_candidates(d1_weighted_survives=False)],
        "weighted_survives_candidates": [recipe.to_dict() for recipe in sys2_1_native_candidates(d1_weighted_survives=True)],
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_1_TRAIN, "sys2_1_native_plan.json"), payload)


def stage_sys2_1_select(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_1_TRAIN, "sys2_1_native_plan.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.SYS2_1_SELECT.value,
        "native_branch_rule": "arithmetic mean raw token logits of exactly five selected Native heads",
        "selection": list(config.policy.recipe_selection),
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_1_SELECT, "sys2_1_native_selection.json"), payload)


def stage_sys2_2(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_1_SELECT, "sys2_1_native_selection.json"))
    payload = {
        "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
        "stage": Stage2Stage.SYS2_2_EVALUATE.value,
        "gamma_grid": [candidate.to_dict() for candidate in sys2_2_gamma_grid()],
        "parent_decoder_stacking": False,
        "decode_once_after_fusion": True,
    }
    write_json_once(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_2_EVALUATE, "sys2_2_final_viunmark.json"), payload)


def stage_comp(args: argparse.Namespace, config: OptimizedPhoNERStage2Config, stage: Stage2Stage) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.SYS2_2_EVALUATE, "sys2_2_final_viunmark.json"))
    analyses = {
        Stage2Stage.COMP_D1: "matched-recipe complementarity analysis",
        Stage2Stage.COMP_D2: "factorize gains from linear diagnostic through SYS2-2",
    }
    write_json_once(
        stage_artifact_path(args.output_root, config, stage, f"{stage.value}.json"),
        {
            "schema_version": PHONER_STAGE2_OPT_RESULT_SCHEMA,
            "stage": stage.value,
            "analysis": analyses[stage],
            "test_used": False,
        },
    )


def stage_freeze_final(args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.COMP_D1, "comp-d1.json"))
    require_artifact(stage_artifact_path(args.output_root, config, Stage2Stage.COMP_D2, "comp-d2.json"))
    write_json_once(
        final_freeze_path(args.output_root, config),
        {
            "schema_version": PHONER_STAGE2_OPT_FINAL_SCHEMA,
            "config_digest": stable_digest(config.to_dict()),
            "post_diagnostic_development": True,
            "test_read": False,
            "test_scored": False,
            "stage1_retraining": False,
            "uit_vsfc_retuning": False,
        },
    )


def run_stage(stage: str, args: argparse.Namespace, config: OptimizedPhoNERStage2Config) -> None:
    require_test_disabled(stage)
    try:
        resolved = Stage2Stage(stage)
    except ValueError as error:
        raise SystemExit(f"unknown optimized Stage-II stage: {stage}") from error
    if resolved is Stage2Stage.PROTOCOL:
        stage_protocol(args, config)
    elif resolved is Stage2Stage.PROTOCOL_AMEND:
        stage_protocol_amend(args, config)
    elif resolved is Stage2Stage.BUILD_BANK:
        stage_build_bank(args, config)
    elif resolved is Stage2Stage.D1_TRAIN:
        stage_d1_train(args, config)
    elif resolved is Stage2Stage.D1_SELECT:
        stage_d1_select(args, config)
    elif resolved is Stage2Stage.D2_DECODE:
        stage_d2_decode(args, config)
    elif resolved is Stage2Stage.D3_TRAIN:
        stage_d3_train(args, config)
    elif resolved is Stage2Stage.D3_SELECT:
        stage_d3_select(args, config)
    elif resolved is Stage2Stage.D4_ANALYZE:
        stage_d4_analyze(args, config)
    elif resolved is Stage2Stage.SYS1_EVALUATE:
        stage_sys1(args, config)
    elif resolved is Stage2Stage.SYS2_1_TRAIN:
        stage_sys2_1_train(args, config)
    elif resolved is Stage2Stage.SYS2_1_SELECT:
        stage_sys2_1_select(args, config)
    elif resolved is Stage2Stage.SYS2_2_EVALUATE:
        stage_sys2_2(args, config)
    elif resolved in {Stage2Stage.COMP_D1, Stage2Stage.COMP_D2}:
        stage_comp(args, config, resolved)
    elif resolved is Stage2Stage.FREEZE_FINAL:
        stage_freeze_final(args, config)
    else:
        raise SystemExit(f"stage is not executable in this campaign: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the post-diagnostic PhoNER Stage-II optimization funnel.")
    parser.add_argument("--stage", required=True, choices=[stage.value for stage in Stage2Stage])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--scale-checkpoint", default=None)
    args = parser.parse_args()
    run_stage(args.stage, args, OptimizedPhoNERStage2Config())


if __name__ == "__main__":
    main()
