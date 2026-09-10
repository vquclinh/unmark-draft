#!/usr/bin/env python3
"""RESTORE Stage-2 baseline runner.

The scientific command is:

    python -B scripts/baselines/restore_stage2.py \
      --drive-root /content/drive/MyDrive/UNMARK/UNMARK-BACKUP \
      --run-all \
      --expected-head <FULL_COMMITTED_IMPLEMENTATION_HEAD>

There is no official TEST path or role. Official validation is read only after
the five RESTORE heads have been frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from unmark.baselines.restore.cache import (  # noqa: E402
    RestorePhaseRegistry,
    RestoreRepresentationCache,
    RestoreTextCache,
    file_sha256,
    read_json,
    write_json,
)
from unmark.baselines.restore.config import (  # noqa: E402
    RESTORE_CONDITIONS,
    RESTORE_CORRUPTION_SEED,
    RESTORE_DEGRADED_CONDITIONS,
    RESTORE_GENERATION_CONFIG,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_OFFICIAL_TEST_READ,
    RESTORE_PHASES,
    RESTORE_RESTORER_BATCH_SIZE,
    RESTORE_STAGE2_HEAD_SEEDS,
    RESTORE_TOKENIZER_CONTRACT,
    RESTORE_UNMARK_A_EVIDENCE_RELATIVE,
    RESTORE_UNMARK_A_EVIDENCE_SHA256,
    RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE,
    RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
    require_full_commit_sha,
    require_restore_namespace,
    restore_artifact_root,
    restore_runtime_cache_root,
)
from unmark.baselines.restore.model import (  # noqa: E402
    load_frozen_restorer,
    restorer_contract_manifest,
    run_batched_decode_smoke,
    verify_local_restore_artifacts,
    verify_restore_hub_metadata,
)
from unmark.baselines.restore.stage2 import (  # noqa: E402
    RESTORE_SCORE_UNITS,
    aggregate_diagnostic_reports,
    aggregate_restore_scores,
    authenticate_read_only_evidence,
    build_condition_streams,
    build_final_evidence,
    compute_grr,
    diagnostics_for_condition,
    extract_condition_metrics,
    extract_or_load_restore_representations,
    load_restore_phobert_components,
    load_restore_protocol_splits,
    load_restore_validation_split,
    minimal_comparison,
    representation_key_from_text_cache,
    RestoreHeadStore,
    restore_text_cache,
    restore_text_request_for_split,
    restore_text_request_for_stream,
    score_restore_condition,
    train_or_load_restore_heads,
)
from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import Preg1Role  # noqa: E402


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def require_clean_git_head(expected_head: str) -> str:
    expected = require_full_commit_sha(expected_head, name="--expected-head")
    actual = _git(["rev-parse", "HEAD"])
    if actual != expected:
        raise EvaluationContractViolation(
            f"HEAD must equal --expected-head {expected}, got {actual}"
        )
    status = _git(["status", "--short"])
    if status:
        raise EvaluationContractViolation(
            f"git tree must be clean before RESTORE execution, got:\n{status}"
        )
    return actual


def default_derived_train(drive_root: Path) -> Path:
    return drive_root / "runtime-input-cache" / "uit-vsfc" / "derived-train.csv"


def default_official_validation(drive_root: Path) -> Path:
    return drive_root / "runtime-input-cache" / "uit-vsfc" / "official-validation.csv"


def default_split_dir(drive_root: Path) -> Path:
    return (
        drive_root.parent
        / "preg1-uit-vsfc-internal-split"
        / "preg1-split-v1-66f4522a-7bd5d189"
    )


class RestoreRunner:
    def __init__(self, args: argparse.Namespace, *, execution_head: str) -> None:
        self.args = args
        self.execution_head = execution_head
        self.drive_root = Path(args.drive_root)
        self.artifact_root = restore_artifact_root(self.drive_root, execution_head)
        require_restore_namespace(self.artifact_root, self.drive_root)
        self.cache_root = restore_runtime_cache_root(self.drive_root)
        self.phase_registry = RestorePhaseRegistry(self.artifact_root / "phase-state")
        self._restorer = None
        self._phobert = None

    @property
    def derived_train_path(self) -> Path:
        return Path(self.args.derived_train) if self.args.derived_train else default_derived_train(self.drive_root)

    @property
    def official_validation_path(self) -> Path:
        return (
            Path(self.args.official_validation)
            if self.args.official_validation
            else default_official_validation(self.drive_root)
        )

    @property
    def split_dir(self) -> Path:
        return Path(self.args.split_dir) if self.args.split_dir else default_split_dir(self.drive_root)

    def phase_identity(self, phase: str) -> dict[str, Any]:
        return {
            "phase": phase,
            "repository_head": self.execution_head,
            "restore_model_id": RESTORE_MODEL_ID,
            "restore_model_revision": RESTORE_MODEL_REVISION,
            "artifact_root": str(self.artifact_root),
            "restore_batch_size": self.args.restore_batch_size,
            "phobert_batch_size": self.args.phobert_batch_size,
            "generation_config": RESTORE_GENERATION_CONFIG.to_dict(),
            "tokenizer_contract": RESTORE_TOKENIZER_CONTRACT.to_dict(),
        }

    def run_phase(self, phase: str, action) -> dict[str, Any]:
        return self.phase_registry.run_once(phase, self.phase_identity(phase), action)

    def ensure_restorer(self):
        if self._restorer is None:
            self._restorer = load_frozen_restorer(
                cache_dir=self.cache_root / "hf",
                device=self.args.device,
            )
        return self._restorer

    def ensure_phobert(self):
        if self._phobert is None:
            self._phobert = load_restore_phobert_components(
                cache_dir=self.cache_root / "hf",
                device=self.args.device,
            )
        return self._phobert

    def protocol_splits(self):
        return load_restore_protocol_splits(
            derived_train=self.derived_train_path,
            split_dir=self.split_dir,
            text_column=self.args.text_column,
            label_column=self.args.label_column,
            id_column=self.args.id_column,
        )

    def heads_are_frozen(self) -> bool:
        head_root = self.artifact_root / "heads"
        for seed in RESTORE_STAGE2_HEAD_SEEDS:
            store = RestoreHeadStore(head_root / f"seed-{seed}")
            if not store.exists():
                return False
            store.read_artifact(expected_seed=seed)
        return True

    def validation_split(self):
        return load_restore_validation_split(
            official_validation=self.official_validation_path,
            text_column=self.args.text_column,
            label_column=self.args.label_column,
            id_column=self.args.id_column,
            heads_frozen=self.heads_are_frozen(),
        )

    def model_preflight(self) -> dict[str, Any]:
        def action():
            hub = verify_restore_hub_metadata()
            restorer = self.ensure_restorer()
            smoke = run_batched_decode_smoke(
                restorer,
                batch_size=self.args.restore_batch_size,
            )
            snapshot = None
            local_hashes = None
            try:
                from huggingface_hub import snapshot_download

                snapshot = Path(
                    snapshot_download(
                        repo_id=RESTORE_MODEL_ID,
                        revision=RESTORE_MODEL_REVISION,
                        cache_dir=str(self.cache_root / "hf"),
                    )
                )
                local_hashes = verify_local_restore_artifacts(snapshot)
            except ImportError:
                local_hashes = {"skipped": "huggingface_hub not importable"}
            import transformers

            payload = {
                "hub_metadata": hub.to_dict(),
                "local_artifacts": local_hashes,
                "snapshot_dir": str(snapshot) if snapshot is not None else None,
                "restorer_contract": restorer_contract_manifest(
                    transformers_version=transformers.__version__
                ),
                "batch_smoke": smoke,
                "restore_batch_size": self.args.restore_batch_size,
            }
            out = self.artifact_root / "model" / "restore-model-preflight.json"
            write_json(out, payload)
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("RESTORE_MODEL_PREFLIGHT", action)

    def clean_text(self) -> dict[str, Any]:
        def action():
            restorer = self.ensure_restorer()
            splits = self.protocol_splits()
            manifests = {}
            for role in (Preg1Role.PROTOCOL_TRAIN, Preg1Role.PROTOCOL_DEV):
                split = splits[role]
                cache = RestoreTextCache(self.artifact_root / "clean-restored-text" / role.value)
                request = restore_text_request_for_split(
                    split,
                    repository_head=self.execution_head,
                    restore_batch_size=self.args.restore_batch_size,
                )
                restore_text_cache(
                    cache=cache,
                    request=request,
                    restorer=restorer,
                    sample_ids=split.sample_ids,
                    texts=split.texts,
                    batch_size=self.args.restore_batch_size,
                    extra_manifest={"input_role": role.value},
                )
                manifests[role.value] = {
                    "path": str(cache.manifest_path),
                    "sha256": file_sha256(cache.manifest_path),
                }
            out = self.artifact_root / "clean-restored-text" / "clean-text-index.json"
            write_json(out, manifests)
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("RESTORE_CLEAN_TEXT", action)

    def clean_representations(self) -> dict[str, Any]:
        def action():
            splits = self.protocol_splits()
            tokenizer, encoder = self.ensure_phobert()
            manifests = {}
            for role in (Preg1Role.PROTOCOL_TRAIN, Preg1Role.PROTOCOL_DEV):
                split = splits[role]
                text_cache = RestoreTextCache(self.artifact_root / "clean-restored-text" / role.value)
                request = restore_text_request_for_split(
                    split,
                    repository_head=self.execution_head,
                    restore_batch_size=self.args.restore_batch_size,
                )
                records = text_cache.load(request)
                restored = [record.restored_text for record in records]
                key = representation_key_from_text_cache(
                    repository_head=self.execution_head,
                    text_manifest_path=text_cache.manifest_path,
                    role=role,
                    condition="FULL",
                    corruption_seed=None,
                    sample_ids=split.sample_ids,
                    labels=split.labels,
                    restored_texts=restored,
                )
                rep_cache = RestoreRepresentationCache(
                    self.artifact_root / "clean-representations" / role.value
                )
                extract_or_load_restore_representations(
                    cache=rep_cache,
                    key=key,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    restored_texts=restored,
                    batch_size=self.args.phobert_batch_size,
                )
                manifests[role.value] = {
                    "path": str(rep_cache.manifest_path),
                    "sha256": file_sha256(rep_cache.manifest_path),
                }
            out = self.artifact_root / "clean-representations" / "clean-representation-index.json"
            write_json(out, manifests)
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("RESTORE_CLEAN_REPRESENTATIONS", action)

    def train_heads(self) -> dict[str, Any]:
        def action():
            splits = self.protocol_splits()
            loaded = {}
            for role in (Preg1Role.PROTOCOL_TRAIN, Preg1Role.PROTOCOL_DEV):
                split = splits[role]
                text_cache = RestoreTextCache(self.artifact_root / "clean-restored-text" / role.value)
                records = text_cache.load(
                    restore_text_request_for_split(
                        split,
                        repository_head=self.execution_head,
                        restore_batch_size=self.args.restore_batch_size,
                    )
                )
                key = representation_key_from_text_cache(
                    repository_head=self.execution_head,
                    text_manifest_path=text_cache.manifest_path,
                    role=role,
                    condition="FULL",
                    corruption_seed=None,
                    sample_ids=split.sample_ids,
                    labels=split.labels,
                    restored_texts=[r.restored_text for r in records],
                )
                loaded[role] = RestoreRepresentationCache(
                    self.artifact_root / "clean-representations" / role.value
                ).load(key)
            artifacts = train_or_load_restore_heads(
                head_root=self.artifact_root / "heads",
                repository_head=self.execution_head,
                train=loaded[Preg1Role.PROTOCOL_TRAIN],
                train_labels=splits[Preg1Role.PROTOCOL_TRAIN].labels,
                dev=loaded[Preg1Role.PROTOCOL_DEV],
                dev_labels=splits[Preg1Role.PROTOCOL_DEV].labels,
            )
            index = {
                str(seed): {
                    "path": str(self.artifact_root / "heads" / f"seed-{seed}" / "restore-head-artifact.json"),
                    "sha256": file_sha256(
                        self.artifact_root / "heads" / f"seed-{seed}" / "restore-head-artifact.json"
                    ),
                }
                for seed in artifacts
            }
            out = self.artifact_root / "heads" / "restore-head-index.json"
            write_json(out, index)
            return {"path": str(out), "sha256": file_sha256(out), "heads": len(index)}

        return self.run_phase("RESTORE_FIVE_HEADS", action)

    def validation_identity(self) -> dict[str, Any]:
        def action():
            validation = self.validation_split()
            payload = {
                "source_sha256": validation.source_sha256,
                "rows": len(validation.sample_ids),
                "ordered_id_digest": validation.ordered_id_digest,
                "label_digest": validation.label_digest,
                "class_counts": {
                    str(label): count
                    for label, count in sorted(Counter(validation.labels).items())
                },
                "heads_frozen_before_read": True,
            }
            out = self.artifact_root / "measurement" / "validation-identity.json"
            write_json(out, payload)
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("VALIDATION_IDENTITY_CHECK", action)

    def validation_text(self) -> dict[str, Any]:
        def action():
            restorer = self.ensure_restorer()
            validation = self.validation_split()
            streams = build_condition_streams(validation, corruption_seed=RESTORE_CORRUPTION_SEED)
            manifests = {}
            for condition in RESTORE_CONDITIONS:
                stream = streams[condition]
                cache = RestoreTextCache(self.artifact_root / "validation-restored-text" / condition)
                request = restore_text_request_for_stream(
                    stream,
                    repository_head=self.execution_head,
                    restore_batch_size=self.args.restore_batch_size,
                )
                restore_text_cache(
                    cache=cache,
                    request=request,
                    restorer=restorer,
                    sample_ids=stream.sample_ids,
                    texts=stream.observed_texts,
                    batch_size=self.args.restore_batch_size,
                    extra_manifest={
                        "condition": condition,
                        "corruption_seed": RESTORE_CORRUPTION_SEED,
                        "changed_row_count": stream.changed_row_count,
                    },
                )
                manifests[condition] = {
                    "path": str(cache.manifest_path),
                    "sha256": file_sha256(cache.manifest_path),
                    "changed_row_count": stream.changed_row_count,
                }
            out = self.artifact_root / "validation-restored-text" / "validation-text-index.json"
            write_json(out, manifests)
            return {"path": str(out), "sha256": file_sha256(out), "conditions": len(manifests)}

        return self.run_phase("RESTORE_VALIDATION_TEXT", action)

    def validation_representations(self) -> dict[str, Any]:
        def action():
            validation = self.validation_split()
            streams = build_condition_streams(validation, corruption_seed=RESTORE_CORRUPTION_SEED)
            tokenizer, encoder = self.ensure_phobert()
            manifests = {}
            for condition in RESTORE_CONDITIONS:
                stream = streams[condition]
                text_cache = RestoreTextCache(self.artifact_root / "validation-restored-text" / condition)
                records = text_cache.load(
                    restore_text_request_for_stream(
                        stream,
                        repository_head=self.execution_head,
                        restore_batch_size=self.args.restore_batch_size,
                    )
                )
                restored = [record.restored_text for record in records]
                key = representation_key_from_text_cache(
                    repository_head=self.execution_head,
                    text_manifest_path=text_cache.manifest_path,
                    role=Preg1Role.OFFICIAL_VALIDATION,
                    condition=condition,
                    corruption_seed=RESTORE_CORRUPTION_SEED,
                    sample_ids=stream.sample_ids,
                    labels=stream.labels,
                    restored_texts=restored,
                )
                rep_cache = RestoreRepresentationCache(
                    self.artifact_root / "validation-representations" / condition
                )
                extract_or_load_restore_representations(
                    cache=rep_cache,
                    key=key,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    restored_texts=restored,
                    batch_size=self.args.phobert_batch_size,
                )
                manifests[condition] = {
                    "path": str(rep_cache.manifest_path),
                    "sha256": file_sha256(rep_cache.manifest_path),
                }
            out = self.artifact_root / "validation-representations" / "validation-representation-index.json"
            write_json(out, manifests)
            return {"path": str(out), "sha256": file_sha256(out), "conditions": len(manifests)}

        return self.run_phase("RESTORE_VALIDATION_REPRESENTATIONS", action)

    def measurement(self) -> dict[str, Any]:
        def action():
            validation = self.validation_split()
            streams = build_condition_streams(validation, corruption_seed=RESTORE_CORRUPTION_SEED)
            scores = []

            for seed in RESTORE_STAGE2_HEAD_SEEDS:
                head = RestoreHeadStore(self.artifact_root / "heads" / f"seed-{seed}").load_head(
                    expected_seed=seed
                )
                for condition in RESTORE_CONDITIONS:
                    stream = streams[condition]
                    text_cache = RestoreTextCache(self.artifact_root / "validation-restored-text" / condition)
                    records = text_cache.load(
                        restore_text_request_for_stream(
                            stream,
                            repository_head=self.execution_head,
                            restore_batch_size=self.args.restore_batch_size,
                        )
                    )
                    key = representation_key_from_text_cache(
                        repository_head=self.execution_head,
                        text_manifest_path=text_cache.manifest_path,
                        role=Preg1Role.OFFICIAL_VALIDATION,
                        condition=condition,
                        corruption_seed=RESTORE_CORRUPTION_SEED,
                        sample_ids=stream.sample_ids,
                        labels=stream.labels,
                        restored_texts=[r.restored_text for r in records],
                    )
                    reps = RestoreRepresentationCache(
                        self.artifact_root / "validation-representations" / condition
                    ).load(key)
                    scores.append(
                        score_restore_condition(
                            head,
                            reps,
                            validation.labels,
                            seed=seed,
                        )
                    )
            aggregate = aggregate_restore_scores(scores)
            out = self.artifact_root / "measurement" / "restore-measurement.json"
            write_json(out, aggregate)
            return {"path": str(out), "sha256": file_sha256(out), "score_units": len(scores)}

        return self.run_phase("RESTORE_MEASUREMENT", action)

    def diagnostics(self) -> dict[str, Any]:
        def action():
            validation = self.validation_split()
            streams = build_condition_streams(validation, corruption_seed=RESTORE_CORRUPTION_SEED)
            reports = []
            for condition in RESTORE_CONDITIONS:
                stream = streams[condition]
                cache = RestoreTextCache(self.artifact_root / "validation-restored-text" / condition)
                records = cache.load(
                    restore_text_request_for_stream(
                        stream,
                        repository_head=self.execution_head,
                        restore_batch_size=self.args.restore_batch_size,
                    )
                )
                reports.append(diagnostics_for_condition(records, stream))
            aggregate = aggregate_diagnostic_reports(reports)
            out = self.artifact_root / "diagnostics" / "restore-diagnostics.json"
            write_json(out, aggregate)
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("RESTORE_DIAGNOSTICS", action)

    def grr(self) -> dict[str, Any]:
        def action():
            measurement = read_json(self.artifact_root / "measurement" / "restore-measurement.json")
            vanilla_path = self.drive_root / RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE
            vanilla = authenticate_read_only_evidence(
                vanilla_path,
                RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
            )
            grr = compute_grr(restore_aggregate=measurement, vanilla_evidence=vanilla)
            comparison = None
            unmark_path = self.drive_root / RESTORE_UNMARK_A_EVIDENCE_RELATIVE
            if unmark_path.is_file() and file_sha256(unmark_path) == RESTORE_UNMARK_A_EVIDENCE_SHA256:
                unmark = read_json(unmark_path)
                comparison = minimal_comparison(
                    vanilla=extract_condition_metrics(vanilla),
                    restore=extract_condition_metrics({"conditions": measurement["conditions"]}),
                    unmark_a=extract_condition_metrics(unmark),
                )
                write_json(self.artifact_root / "comparison" / "restore-contextual-comparison.json", comparison)
            out = self.artifact_root / "grr" / "restore-grr.json"
            write_json(out, {"grr": grr, "comparison": comparison})
            return {"path": str(out), "sha256": file_sha256(out)}

        return self.run_phase("RESTORE_GRR", action)

    def final_closeout(self) -> dict[str, Any]:
        def action():
            model_preflight = read_json(self.artifact_root / "model" / "restore-model-preflight.json")
            clean_text = read_json(self.artifact_root / "clean-restored-text" / "clean-text-index.json")
            clean_reps = read_json(self.artifact_root / "clean-representations" / "clean-representation-index.json")
            heads = {
                seed: read_json(self.artifact_root / "heads" / f"seed-{seed}" / "restore-head-artifact.json")
                for seed in RESTORE_STAGE2_HEAD_SEEDS
            }
            validation_identity = read_json(self.artifact_root / "measurement" / "validation-identity.json")
            validation_text = read_json(
                self.artifact_root / "validation-restored-text" / "validation-text-index.json"
            )
            validation_reps = read_json(
                self.artifact_root / "validation-representations" / "validation-representation-index.json"
            )
            measurement = read_json(self.artifact_root / "measurement" / "restore-measurement.json")
            diagnostics = read_json(self.artifact_root / "diagnostics" / "restore-diagnostics.json")
            grr_payload = read_json(self.artifact_root / "grr" / "restore-grr.json")
            evidence = build_final_evidence(
                repository_head=self.execution_head,
                model_preflight=model_preflight,
                clean_text_manifests=clean_text,
                clean_representation_manifests=clean_reps,
                head_artifacts=heads,
                validation_identity=validation_identity,
                validation_text_manifests=validation_text,
                validation_representation_manifests=validation_reps,
                measurement=measurement,
                diagnostics=diagnostics,
                grr=grr_payload["grr"],
                comparison=grr_payload.get("comparison"),
            )
            out = self.artifact_root / "evidence" / "restore-final-evidence.json"
            write_json(out, evidence)
            return {
                "path": str(out),
                "sha256": file_sha256(out),
                "score_units": measurement["score_units"],
                "official_test_read": False,
            }

        return self.run_phase("RESTORE_FINAL_CLOSEOUT", action)

    def run_all(self) -> dict[str, Any]:
        for phase in RESTORE_PHASES:
            if phase == "RESTORE_MODEL_PREFLIGHT":
                self.model_preflight()
            elif phase == "RESTORE_CLEAN_TEXT":
                self.clean_text()
            elif phase == "RESTORE_CLEAN_REPRESENTATIONS":
                self.clean_representations()
            elif phase == "RESTORE_FIVE_HEADS":
                self.train_heads()
            elif phase == "VALIDATION_IDENTITY_CHECK":
                self.validation_identity()
            elif phase == "RESTORE_VALIDATION_TEXT":
                self.validation_text()
            elif phase == "RESTORE_VALIDATION_REPRESENTATIONS":
                self.validation_representations()
            elif phase == "RESTORE_MEASUREMENT":
                self.measurement()
            elif phase == "RESTORE_DIAGNOSTICS":
                self.diagnostics()
            elif phase == "RESTORE_GRR":
                self.grr()
            elif phase == "RESTORE_FINAL_CLOSEOUT":
                return self.final_closeout()
        raise EvaluationContractViolation("RESTORE phase list did not close out")


def print_success(result: Mapping[str, Any]) -> None:
    payload = result.get("payload", result)
    path = payload["path"]
    sha = payload["sha256"]
    print()
    print("RESTORE_BASELINE=PASS")
    print()
    print(f"RESTORE_MODEL={RESTORE_MODEL_ID}")
    print(f"RESTORE_MODEL_REVISION={RESTORE_MODEL_REVISION}")
    print("RESTORE_MODEL_FROZEN=YES")
    print()
    print("RESTORE_CLEAN_TEXT=PASS")
    print("RESTORE_CLEAN_REPRESENTATIONS=PASS")
    print(f"RESTORE_STAGE2_HEADS={len(RESTORE_STAGE2_HEAD_SEEDS)}_OF_{len(RESTORE_STAGE2_HEAD_SEEDS)}")
    print()
    print(f"RESTORE_VALIDATION_TEXT_CONDITIONS={len(RESTORE_CONDITIONS)}_OF_{len(RESTORE_CONDITIONS)}")
    print(f"RESTORE_VALIDATION_REPRESENTATIONS={len(RESTORE_CONDITIONS)}_OF_{len(RESTORE_CONDITIONS)}")
    print(f"RESTORE_MEASUREMENT_UNITS={RESTORE_SCORE_UNITS}_OF_{RESTORE_SCORE_UNITS}")
    print()
    print("RESTORE_DIAGNOSTICS=PASS")
    print("RESTORE_GRR=PASS")
    print()
    print("CORRUPTED_LABEL_TRAINING=NO")
    print("BEST_SEED_SELECTION=NO")
    print("OFFICIAL_VALIDATION_PREVIOUSLY_SEEN_BY_AUTHORS=YES")
    print("OFFICIAL_TEST_READ=NO")
    print()
    print(f"FINAL_RESTORE_EVIDENCE={path}")
    print(f"FINAL_RESTORE_EVIDENCE_SHA256={sha}")
    print()
    print("RESULTS_AVAILABLE_FOR_AUTHOR_REVIEW=YES")
    print("HARD_STOP=YES")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--run-all", action="store_true", help="execute all RESTORE phases")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--derived-train", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--official-validation", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--restore-batch-size", type=int, default=RESTORE_RESTORER_BATCH_SIZE)
    parser.add_argument("--phobert-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_all:
        parser.error("--run-all is required for the RESTORE baseline executor")
    try:
        execution_head = require_clean_git_head(args.expected_head)
        runner = RestoreRunner(args, execution_head=execution_head)
        result = runner.run_all()
    except EvaluationContractViolation as error:
        print(f"RESTORE_REFUSED: {error}", file=sys.stderr)
        return 2
    print_success(result)
    if RESTORE_OFFICIAL_TEST_READ:
        print("RESTORE_REFUSED: official TEST read flag drifted", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
