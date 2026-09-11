"""Stage-2 RESTORE baseline orchestration primitives.

This module implements the separate RESTORE pathway. It reuses neutral PREG1
head/metric/corruption primitives, but it does not import the UNMARK adapter or
Stage-1 machinery.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from unmark.baselines.restore.cache import (
    RestoreBoundRepresentations,
    RestoreRepresentationCache,
    RestoreRepresentationKey,
    RestoreTextCache,
    RestoreTextCacheRequest,
    RestoreTextRecord,
    file_sha256,
    label_digest,
    read_json,
    semantic_text_digest,
    tensor_semantic_sha256,
    write_json,
)
from unmark.baselines.restore.config import (
    RESTORE_BASELINE_SCHEMA_VERSION,
    RESTORE_BEST_SEED_SELECTION,
    RESTORE_CANONICAL_INPUT_SEMANTICS,
    RESTORE_CONDITIONS,
    RESTORE_CORRUPTED_LABEL_TRAINING,
    RESTORE_CORRUPTION_SEED,
    RESTORE_DEGRADED_CONDITIONS,
    RESTORE_DERIVED_TRAIN_ROWS,
    RESTORE_DERIVED_TRAIN_SHA256,
    RESTORE_EXPECTED_CHANGED_ROW_COUNTS,
    RESTORE_FULL_BYPASS,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_OFFICIAL_VALIDATION_PREVIOUSLY_SEEN_BY_AUTHORS,
    RESTORE_ONE_RESTORE_PATHWAY,
    RESTORE_PHOBERT_CHECKPOINT,
    RESTORE_PHOBERT_DTYPE,
    RESTORE_PHOBERT_EVAL_MODE,
    RESTORE_PHOBERT_FROZEN,
    RESTORE_PHOBERT_HIDDEN_SIZE,
    RESTORE_PHOBERT_POOLING,
    RESTORE_PHOBERT_REVISION,
    RESTORE_PROTOCOL_DEV_LABEL_DIGEST,
    RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST,
    RESTORE_PROTOCOL_DEV_ROWS,
    RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST,
    RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST,
    RESTORE_PROTOCOL_TRAIN_ROWS,
    RESTORE_PROTOCOL_VERSION,
    RESTORE_RESTORER_BATCH_SIZE,
    RESTORE_STAGE2_HEAD_ARCHITECTURE,
    RESTORE_STAGE2_HEAD_BATCH_SIZE,
    RESTORE_STAGE2_HEAD_EARLY_STOPPING,
    RESTORE_STAGE2_HEAD_EPOCHS,
    RESTORE_STAGE2_HEAD_LR,
    RESTORE_STAGE2_HEAD_SEEDS,
    RESTORE_STAGE2_MAX_LENGTH,
    RESTORE_STAGE2_PADDING,
    RESTORE_STAGE2_TRUNCATION,
    RESTORE_VALIDATION_CLASS_COUNTS,
    RESTORE_VALIDATION_LABEL_DIGEST,
    RESTORE_VALIDATION_ORDERED_ID_DIGEST,
    RESTORE_VALIDATION_ROWS,
    RESTORE_VALIDATION_SHA256,
    RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256,
    RESTORE_UNMARK_A_EVIDENCE_SHA256,
    require_restore_conditions,
    require_restore_head_seeds,
    require_validation_class_counts,
)
from unmark.baselines.restore.diagnostics import (
    RestoreDiagnosticRow,
    aggregate_diagnostic_reports,
    restore_diagnostics,
)
from unmark.baselines.restore.evidence import (
    authenticate_read_only_evidence,
    compute_grr,
    extract_condition_metrics,
    minimal_comparison,
)
from unmark.corruption import CorruptionPurpose, EligibilityPolicy, corrupt
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.metrics import (
    per_class_scores,
)
from unmark.evaluation.preg1_head import (
    EpochScore,
    Preg1Role,
    build_head,
    build_optimizer,
    deterministic_batches,
    ordered_id_digest,
    require_full_schedule,
    score_predictions,
    select_checkpoint,
)
from unmark.evaluation.preg1_split import load_derived_pool
from unmark.orthography import canon


RESTORE_STAGE2_TRAINING_ROLE = Preg1Role.PROTOCOL_TRAIN
RESTORE_STAGE2_SELECTION_ROLE = Preg1Role.PROTOCOL_DEV
RESTORE_STAGE2_MEASUREMENT_ROLE = Preg1Role.OFFICIAL_VALIDATION
RESTORE_SCORE_UNITS = len(RESTORE_STAGE2_HEAD_SEEDS) * len(RESTORE_CONDITIONS)


def canonical_clean_texts(texts: Sequence[str]) -> tuple[str, ...]:
    """Shared neutral RESTORE Stage-2 input normalization."""

    return tuple(canon(text) for text in texts)


@dataclass(frozen=True)
class RestoreSplit:
    role: Preg1Role
    sample_ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: tuple[int, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.texts) or len(self.sample_ids) != len(self.labels):
            raise EvaluationContractViolation("RESTORE split fields differ in length")
        if not self.sample_ids:
            raise EvaluationContractViolation("RESTORE split is empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise EvaluationContractViolation("RESTORE split has duplicate sample ids")

    @property
    def ordered_id_digest(self) -> str:
        return ordered_id_digest(self.sample_ids)

    @property
    def label_digest(self) -> str:
        return label_digest(self.labels)

    @property
    def text_digest(self) -> str:
        return semantic_text_digest(self.texts)

    @property
    def canonical_texts(self) -> tuple[str, ...]:
        return canonical_clean_texts(self.texts)

    @property
    def canonical_text_digest(self) -> str:
        return semantic_text_digest(self.canonical_texts)


@dataclass(frozen=True)
class RestoreConditionStream:
    condition: str
    corruption_seed: int | None
    sample_ids: tuple[str, ...]
    observed_texts: tuple[str, ...]
    clean_gold_texts: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.condition not in RESTORE_CONDITIONS:
            raise EvaluationContractViolation(f"unknown RESTORE condition {self.condition!r}")
        if (
            len(self.sample_ids) != len(self.observed_texts)
            or len(self.sample_ids) != len(self.clean_gold_texts)
            or len(self.sample_ids) != len(self.labels)
        ):
            raise EvaluationContractViolation("RESTORE condition stream fields differ in length")
        if tuple(canon(text) for text in self.clean_gold_texts) != self.clean_gold_texts:
            raise EvaluationContractViolation("RESTORE clean reference text must already be canonical")
        if self.condition == "FULL":
            if self.corruption_seed is not None:
                raise EvaluationContractViolation("RESTORE FULL must not carry a corruption seed")
            if self.observed_texts != self.clean_gold_texts:
                raise EvaluationContractViolation("RESTORE FULL observed text must equal canonical clean text")
        elif self.corruption_seed != RESTORE_CORRUPTION_SEED:
            raise EvaluationContractViolation(
                f"RESTORE degraded conditions must use corruption seed {RESTORE_CORRUPTION_SEED}"
            )

    @property
    def ordered_id_digest(self) -> str:
        return ordered_id_digest(self.sample_ids)

    @property
    def label_digest(self) -> str:
        return label_digest(self.labels)

    @property
    def input_text_digest(self) -> str:
        return semantic_text_digest(self.observed_texts)

    @property
    def clean_gold_text_digest(self) -> str:
        return semantic_text_digest(self.clean_gold_texts)

    @property
    def changed_row_count(self) -> int:
        return sum(
            1 for observed, clean in zip(self.observed_texts, self.clean_gold_texts)
            if observed != clean
        )


def _label_index(label: Any) -> int:
    from unmark.evaluation.preg1_protocol import LABEL_MAPPING

    text = str(label).strip()
    if text in LABEL_MAPPING:
        return LABEL_MAPPING[text]
    inverse = {str(index): index for index in LABEL_MAPPING.values()}
    if text in inverse:
        return inverse[text]
    raise EvaluationContractViolation(f"unknown label {label!r}")


def _read_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise EvaluationContractViolation(f"missing membership file {path}")
    ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not ids:
        raise EvaluationContractViolation(f"membership file {path} is empty")
    return ids


def load_restore_protocol_splits(
    *,
    derived_train: str | Path,
    split_dir: str | Path,
    text_column: str,
    label_column: str,
    id_column: str,
) -> dict[Preg1Role, RestoreSplit]:
    """Load and verify the frozen PREG1 protocol-train/dev membership."""

    pool = load_derived_pool(
        derived_train,
        text_column,
        label_column,
        id_column,
        expected_sha256=RESTORE_DERIVED_TRAIN_SHA256,
        expected_rows=RESTORE_DERIVED_TRAIN_ROWS,
    )
    root = Path(split_dir)
    train_ids = _read_ids(root / "protocol-train.ids.txt")
    dev_ids = _read_ids(root / "protocol-dev.ids.txt")
    if set(train_ids) & set(dev_ids):
        raise EvaluationContractViolation("protocol-train and protocol-dev overlap")
    by_id = {sample_id: (text, label) for sample_id, text, label in pool.records}
    if set(train_ids) | set(dev_ids) != set(by_id):
        raise EvaluationContractViolation("PREG1 split membership is not the derived TRAIN pool")

    def split(role: Preg1Role, ids: tuple[str, ...]) -> RestoreSplit:
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise EvaluationContractViolation(f"membership ids missing from pool: {missing[:5]}")
        return RestoreSplit(
            role=role,
            sample_ids=ids,
            texts=tuple(by_id[sample_id][0] for sample_id in ids),
            labels=tuple(_label_index(by_id[sample_id][1]) for sample_id in ids),
            source_sha256=pool.source_sha256,
        )

    result = {
        Preg1Role.PROTOCOL_TRAIN: split(Preg1Role.PROTOCOL_TRAIN, train_ids),
        Preg1Role.PROTOCOL_DEV: split(Preg1Role.PROTOCOL_DEV, dev_ids),
    }
    require_restore_protocol_split_identities(result)
    return result


def require_restore_protocol_split_identities(splits: Mapping[Preg1Role, RestoreSplit]) -> None:
    train = splits.get(Preg1Role.PROTOCOL_TRAIN)
    dev = splits.get(Preg1Role.PROTOCOL_DEV)
    if train is None or dev is None:
        raise EvaluationContractViolation("RESTORE requires protocol-train and protocol-dev")
    expected = {
        Preg1Role.PROTOCOL_TRAIN: (
            RESTORE_PROTOCOL_TRAIN_ROWS,
            RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST,
            RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST,
        ),
        Preg1Role.PROTOCOL_DEV: (
            RESTORE_PROTOCOL_DEV_ROWS,
            RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST,
            RESTORE_PROTOCOL_DEV_LABEL_DIGEST,
        ),
    }
    for role, split in splits.items():
        count, ids_digest, labels_digest = expected[role]
        if len(split.sample_ids) != count:
            raise EvaluationContractViolation(f"{role.value} row count drifted")
        if split.ordered_id_digest != ids_digest:
            raise EvaluationContractViolation(f"{role.value} ordered-id digest drifted")
        if split.label_digest != labels_digest:
            raise EvaluationContractViolation(f"{role.value} label digest drifted")


def load_restore_validation_split(
    *,
    official_validation: str | Path,
    text_column: str,
    label_column: str,
    id_column: str,
    heads_frozen: bool,
) -> RestoreSplit:
    """Load official validation only after the RESTORE heads are frozen."""

    if not heads_frozen:
        raise EvaluationContractViolation(
            "official validation cannot be read before the five RESTORE heads freeze"
        )
    validation = load_derived_pool(
        official_validation,
        text_column,
        label_column,
        id_column,
        expected_sha256=RESTORE_VALIDATION_SHA256,
        expected_rows=RESTORE_VALIDATION_ROWS,
        expected_label_counts={
            "negative": RESTORE_VALIDATION_CLASS_COUNTS[0],
            "neutral": RESTORE_VALIDATION_CLASS_COUNTS[1],
            "positive": RESTORE_VALIDATION_CLASS_COUNTS[2],
        },
    )
    split = RestoreSplit(
        role=Preg1Role.OFFICIAL_VALIDATION,
        sample_ids=tuple(sample_id for sample_id, _, _ in validation.records),
        texts=tuple(text for _, text, _ in validation.records),
        labels=tuple(_label_index(label) for _, _, label in validation.records),
        source_sha256=validation.source_sha256,
    )
    counts = dict(Counter(split.labels))
    require_validation_class_counts(counts)
    if split.ordered_id_digest != RESTORE_VALIDATION_ORDERED_ID_DIGEST:
        raise EvaluationContractViolation("official validation ordered-id digest drifted")
    if split.label_digest != RESTORE_VALIDATION_LABEL_DIGEST:
        raise EvaluationContractViolation("official validation label digest drifted")
    return split


def restore_text_request_for_split(
    split: RestoreSplit,
    *,
    repository_head: str,
    condition: str = "FULL",
    corruption_seed: int | None = None,
    input_texts: Sequence[str] | None = None,
    restore_batch_size: int = RESTORE_RESTORER_BATCH_SIZE,
) -> RestoreTextCacheRequest:
    texts = tuple(split.canonical_texts if input_texts is None else input_texts)
    if condition == "FULL":
        if corruption_seed is not None:
            raise EvaluationContractViolation("RESTORE clean split request cannot carry a corruption seed")
        if texts != split.canonical_texts:
            raise EvaluationContractViolation("RESTORE clean split inputs must be canonical")
    return RestoreTextCacheRequest(
        repository_head=repository_head,
        role=split.role.value,
        condition=condition,
        corruption_seed=corruption_seed,
        input_row_count=len(split.sample_ids),
        ordered_id_digest=split.ordered_id_digest,
        input_text_digest=semantic_text_digest(texts),
        restore_batch_size=restore_batch_size,
    )


def restore_text_request_for_stream(
    stream: RestoreConditionStream,
    *,
    repository_head: str,
    restore_batch_size: int = RESTORE_RESTORER_BATCH_SIZE,
) -> RestoreTextCacheRequest:
    return RestoreTextCacheRequest(
        repository_head=repository_head,
        role=Preg1Role.OFFICIAL_VALIDATION.value,
        condition=stream.condition,
        corruption_seed=stream.corruption_seed,
        input_row_count=len(stream.sample_ids),
        ordered_id_digest=stream.ordered_id_digest,
        input_text_digest=stream.input_text_digest,
        restore_batch_size=restore_batch_size,
    )


def restore_records(
    restorer: Any,
    *,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
) -> tuple[RestoreTextRecord, ...]:
    """Run the one RESTORE pathway. No condition reaches the restorer."""

    if batch_size <= 0:
        raise EvaluationContractViolation("RESTORE batch size must be positive")
    if len(sample_ids) != len(texts):
        raise EvaluationContractViolation("RESTORE ids and texts differ in length")
    records: list[RestoreTextRecord] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        outputs = restorer.restore_batch(batch_texts)
        if len(outputs) != len(batch_texts):
            raise EvaluationContractViolation("RESTORE restorer returned wrong batch length")
        for sample_id, observed, restored in zip(
            sample_ids[start : start + batch_size],
            batch_texts,
            outputs,
        ):
            records.append(
                RestoreTextRecord(
                    sample_id=str(sample_id),
                    input_text=str(observed),
                    restored_text=str(restored),
                )
            )
    return tuple(records)


def restore_text_cache(
    *,
    cache: RestoreTextCache,
    request: RestoreTextCacheRequest,
    restorer: Any,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    batch_size: int,
    extra_manifest: Mapping[str, Any] | None = None,
) -> tuple[RestoreTextRecord, ...]:
    if request.restore_batch_size != batch_size:
        raise EvaluationContractViolation("RESTORE text cache batch size drifted")
    if cache.exists():
        return cache.load(request)
    records = restore_records(
        restorer,
        sample_ids=sample_ids,
        texts=texts,
        batch_size=batch_size,
    )
    return cache.save(request, records, extra_manifest=extra_manifest)


def build_condition_streams(
    validation: RestoreSplit,
    *,
    corruption_seed: int = RESTORE_CORRUPTION_SEED,
    corruption_purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> dict[str, RestoreConditionStream]:
    """Build canonical-clean validation streams before the common RESTORE path."""

    require_restore_conditions(RESTORE_CONDITIONS)
    if corruption_seed != RESTORE_CORRUPTION_SEED:
        raise EvaluationContractViolation(
            f"RESTORE degraded corruption seed must be {RESTORE_CORRUPTION_SEED}"
        )
    canonical = validation.canonical_texts
    streams: dict[str, RestoreConditionStream] = {}
    for condition in RESTORE_CONDITIONS:
        stream_seed = None if condition == "FULL" else corruption_seed
        if condition == "FULL":
            observed = canonical
        else:
            observed = tuple(
                corrupt(
                    text,
                    condition,
                    seed=corruption_seed,
                    sample_id=sample_id,
                    purpose=corruption_purpose,
                    eligibility_policy=eligibility_policy,
                ).corrupted_text
                for sample_id, text in zip(validation.sample_ids, canonical)
            )
        stream = RestoreConditionStream(
            condition=condition,
            corruption_seed=stream_seed,
            sample_ids=validation.sample_ids,
            observed_texts=observed,
            clean_gold_texts=canonical,
            labels=validation.labels,
        )
        streams[condition] = stream
    require_changed_row_counts(streams)
    return streams


def require_changed_row_counts(streams: Mapping[str, RestoreConditionStream]) -> None:
    actual = {condition: streams[condition].changed_row_count for condition in RESTORE_CONDITIONS}
    if actual != RESTORE_EXPECTED_CHANGED_ROW_COUNTS:
        raise EvaluationContractViolation(
            f"RESTORE corruption changed-row counts drifted: expected "
            f"{RESTORE_EXPECTED_CHANGED_ROW_COUNTS}, got {actual}"
        )


def load_restore_phobert_components(
    *,
    cache_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        RESTORE_PHOBERT_CHECKPOINT,
        revision=RESTORE_PHOBERT_REVISION,
        use_fast=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    encoder = AutoModel.from_pretrained(
        RESTORE_PHOBERT_CHECKPOINT,
        revision=RESTORE_PHOBERT_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    encoder.to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    setattr(encoder, "_restore_requested_revision", RESTORE_PHOBERT_REVISION)
    require_restore_phobert_backbone(encoder)
    return tokenizer, encoder


def require_restore_phobert_backbone(encoder: Any) -> None:
    hidden = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if hidden != RESTORE_PHOBERT_HIDDEN_SIZE:
        raise EvaluationContractViolation(
            f"RESTORE PhoBERT hidden size must be 768, got {hidden!r}"
        )
    revision = None
    for holder in (encoder, getattr(encoder, "config", None)):
        for attribute in ("_restore_requested_revision", "_commit_hash", "revision", "commit_hash"):
            value = getattr(holder, attribute, None)
            if isinstance(value, str) and value:
                revision = value
                break
        if revision:
            break
    if revision != RESTORE_PHOBERT_REVISION:
        raise EvaluationContractViolation(
            f"RESTORE PhoBERT revision must be {RESTORE_PHOBERT_REVISION}, got {revision!r}"
        )
    if getattr(encoder, "training", True):
        raise EvaluationContractViolation("RESTORE PhoBERT encoder must be eval")
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if trainable:
        raise EvaluationContractViolation(
            f"RESTORE PhoBERT has trainable parameter(s): {trainable[:5]}"
        )
    dtypes = {
        str(parameter.dtype)
        for _, parameter in encoder.named_parameters()
        if getattr(parameter.dtype, "is_floating_point", False)
    }
    if dtypes and dtypes != {RESTORE_PHOBERT_DTYPE}:
        raise EvaluationContractViolation(
            f"RESTORE PhoBERT must be FP32, got {sorted(dtypes)}"
        )


def encode_restore_texts(tokenizer: Any, texts: Sequence[str]) -> tuple[Any, Any]:
    encoded = tokenizer(
        list(texts),
        padding=RESTORE_STAGE2_PADDING,
        truncation=RESTORE_STAGE2_TRUNCATION,
        max_length=RESTORE_STAGE2_MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def representation_key_from_text_cache(
    *,
    repository_head: str,
    text_manifest_path: str | Path,
    role: Preg1Role,
    condition: str,
    corruption_seed: int | None,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    restored_texts: Sequence[str],
) -> RestoreRepresentationKey:
    manifest = read_json(text_manifest_path)
    if semantic_text_digest(restored_texts) != manifest["output_text_digest"]:
        raise EvaluationContractViolation(
            "RESTORE restored texts do not match the text-cache manifest"
        )
    return RestoreRepresentationKey(
        repository_head=repository_head,
        role=role.value,
        condition=condition,
        corruption_seed=corruption_seed,
        text_cache_manifest_sha256=file_sha256(text_manifest_path),
        text_output_digest=manifest["output_text_digest"],
        ordered_id_digest=ordered_id_digest(sample_ids),
        label_digest=label_digest(labels),
        count=len(sample_ids),
    )


def extract_or_load_restore_representations(
    *,
    cache: RestoreRepresentationCache,
    key: RestoreRepresentationKey,
    tokenizer: Any,
    encoder: Any,
    restored_texts: Sequence[str],
    batch_size: int,
) -> RestoreBoundRepresentations:
    if cache.exists():
        return cache.load(key)
    if batch_size <= 0:
        raise EvaluationContractViolation("PhoBERT batch size must be positive")
    import torch
    from unmark.evaluation.preg1_head import extract_representations

    chunks = []
    for start in range(0, len(restored_texts), batch_size):
        batch = restored_texts[start : start + batch_size]
        input_ids, attention_mask = encode_restore_texts(tokenizer, batch)
        device = next(encoder.parameters()).device
        chunks.append(
            extract_representations(
                encoder,
                input_ids.to(device),
                attention_mask.to(device),
            ).cpu()
        )
    values = torch.cat(chunks, dim=0)
    if tuple(values.shape) != (key.count, key.hidden_size):
        raise EvaluationContractViolation(
            f"RESTORE representation extraction produced {tuple(values.shape)}"
        )
    return cache.save(key, values)


@dataclass(frozen=True)
class RestoreHeadRun:
    seed: int
    scores: tuple[EpochScore, ...]
    selected: EpochScore
    train_key: RestoreRepresentationKey
    dev_key: RestoreRepresentationKey
    initial_head_semantic_sha256: str
    selected_state_semantic_sha256: str
    selected_state_raw_sha256: str
    history_digest: str
    selected_state: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
            "protocol_version": RESTORE_PROTOCOL_VERSION,
            "seed": self.seed,
            "head_architecture": RESTORE_STAGE2_HEAD_ARCHITECTURE,
            "optimizer": "AdamW",
            "learning_rate": RESTORE_STAGE2_HEAD_LR,
            "batch_size": RESTORE_STAGE2_HEAD_BATCH_SIZE,
            "epochs": RESTORE_STAGE2_HEAD_EPOCHS,
            "early_stopping": RESTORE_STAGE2_HEAD_EARLY_STOPPING,
            "selected_epoch": self.selected.epoch,
            "selected_clean_dev_macro_f1": self.selected.macro_f1,
            "selected_clean_dev_accuracy": self.selected.accuracy,
            "history": [score.to_dict() for score in self.scores],
            "history_digest": self.history_digest,
            "initial_head_semantic_sha256": self.initial_head_semantic_sha256,
            "selected_state_semantic_sha256": self.selected_state_semantic_sha256,
            "selected_state_raw_sha256": self.selected_state_raw_sha256,
            "train_cache_key": self.train_key.to_dict(),
            "protocol_dev_cache_key": self.dev_key.to_dict(),
            "selection_role": Preg1Role.PROTOCOL_DEV.value,
            "selection_condition": "FULL",
            "official_validation_used_for_selection": False,
            "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
        }


def _state_semantic_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_raw_sha256(state: Mapping[str, Any]) -> str:
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(dict(state), path)
        return file_sha256(path)
    finally:
        if path.exists():
            path.unlink()


def history_digest(scores: Sequence[EpochScore]) -> str:
    payload = [score.to_dict() for score in scores]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def require_restore_training_roles(
    train: RestoreBoundRepresentations,
    dev: RestoreBoundRepresentations,
) -> None:
    train.require_role(Preg1Role.PROTOCOL_TRAIN, "RESTORE head training")
    dev.require_role(Preg1Role.PROTOCOL_DEV, "RESTORE head checkpoint selection")
    if train.condition != "FULL" or dev.condition != "FULL":
        raise EvaluationContractViolation("RESTORE heads train/select on clean FULL only")
    train.require_same_geometry(dev)


def train_restore_head(
    train: RestoreBoundRepresentations,
    train_labels: Sequence[int],
    dev: RestoreBoundRepresentations,
    dev_labels: Sequence[int],
    *,
    seed: int,
    learning_rate: float = RESTORE_STAGE2_HEAD_LR,
    epochs: int = RESTORE_STAGE2_HEAD_EPOCHS,
    batch_size: int = RESTORE_STAGE2_HEAD_BATCH_SIZE,
) -> RestoreHeadRun:
    """Train one RESTORE linear head for all 30 epochs."""

    import torch
    from torch import nn

    require_restore_training_roles(train, dev)
    require_restore_head_seeds(RESTORE_STAGE2_HEAD_SEEDS)
    if seed not in RESTORE_STAGE2_HEAD_SEEDS:
        raise EvaluationContractViolation(f"RESTORE head seed {seed!r} is not frozen")
    if learning_rate != RESTORE_STAGE2_HEAD_LR:
        raise EvaluationContractViolation("RESTORE head LR must be 0.01")
    if epochs != RESTORE_STAGE2_HEAD_EPOCHS:
        raise EvaluationContractViolation("RESTORE heads must run exactly 30 epochs")
    if batch_size != RESTORE_STAGE2_HEAD_BATCH_SIZE:
        raise EvaluationContractViolation("RESTORE head batch size must be 128")
    if train.key.label_digest != label_digest(train_labels):
        raise EvaluationContractViolation("RESTORE train labels do not match cache key")
    if dev.key.label_digest != label_digest(dev_labels):
        raise EvaluationContractViolation("RESTORE dev labels do not match cache key")

    head = build_head(RESTORE_PHOBERT_HIDDEN_SIZE, seed)
    initial_sha = _state_semantic_sha256(head.state_dict())
    optimizer = build_optimizer(head, learning_rate)
    head_parameter_ids = {id(p) for p in head.parameters()}
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimized != head_parameter_ids:
        raise EvaluationContractViolation("RESTORE optimizer must contain exactly head parameters")
    loss_fn = nn.CrossEntropyLoss()
    train_y = torch.as_tensor([int(v) for v in train_labels], dtype=torch.long)
    dev_y = torch.as_tensor([int(v) for v in dev_labels], dtype=torch.long)

    scores: list[EpochScore] = []
    snapshots: dict[int, dict[str, Any]] = {}
    for epoch in range(1, epochs + 1):
        head.train()
        for batch in deterministic_batches(len(train_labels), seed * 1000 + epoch, batch_size):
            index = torch.as_tensor(batch, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(train.values[index]), train_y[index])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            predictions = head(dev.values).argmax(dim=1).tolist()
        f1, acc = score_predictions(predictions, dev_y.tolist())
        scores.append(EpochScore(epoch=epoch, macro_f1=f1, accuracy=acc))
        snapshots[epoch] = {name: value.detach().clone() for name, value in head.state_dict().items()}
    require_full_schedule(scores, epochs)
    selected = select_checkpoint(scores)
    selected_state = snapshots[selected.epoch]
    return RestoreHeadRun(
        seed=seed,
        scores=tuple(scores),
        selected=selected,
        train_key=train.key,
        dev_key=dev.key,
        initial_head_semantic_sha256=initial_sha,
        selected_state_semantic_sha256=_state_semantic_sha256(selected_state),
        selected_state_raw_sha256=_state_raw_sha256(selected_state),
        history_digest=history_digest(scores),
        selected_state=selected_state,
    )


RESTORE_HEAD_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "protocol_version",
    "repository_head",
    "restore_model_id",
    "restore_model_revision",
    "seed",
    "head_architecture",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "early_stopping",
    "selected_epoch",
    "selected_clean_dev_macro_f1",
    "selected_clean_dev_accuracy",
    "history",
    "history_digest",
    "initial_head_semantic_sha256",
    "selected_state_semantic_sha256",
    "selected_state_raw_sha256",
    "train_cache_key",
    "protocol_dev_cache_key",
    "selection_role",
    "selection_condition",
    "official_validation_used_for_selection",
    "corrupted_label_training",
    "best_seed_selection",
)


def build_restore_head_artifact(
    run: RestoreHeadRun,
    *,
    repository_head: str,
) -> dict[str, Any]:
    artifact = run.to_dict()
    artifact.update(
        {
            "repository_head": repository_head,
            "restore_model_id": RESTORE_MODEL_ID,
            "restore_model_revision": RESTORE_MODEL_REVISION,
            "corrupted_label_training": RESTORE_CORRUPTED_LABEL_TRAINING,
        }
    )
    validate_restore_head_artifact(artifact, expected_seed=run.seed)
    return artifact


def validate_restore_head_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
) -> None:
    missing = [name for name in RESTORE_HEAD_ARTIFACT_FIELDS if name not in artifact]
    if missing:
        raise EvaluationContractViolation(f"RESTORE head artifact missing {missing}")
    unknown = sorted(set(artifact) - set(RESTORE_HEAD_ARTIFACT_FIELDS))
    if unknown:
        raise EvaluationContractViolation(f"RESTORE head artifact has unknown fields {unknown}")
    if artifact["schema_version"] != RESTORE_BASELINE_SCHEMA_VERSION:
        raise EvaluationContractViolation("RESTORE head artifact schema drifted")
    if artifact["protocol_version"] != RESTORE_PROTOCOL_VERSION:
        raise EvaluationContractViolation("RESTORE head protocol drifted")
    if artifact["restore_model_id"] != RESTORE_MODEL_ID:
        raise EvaluationContractViolation("RESTORE head artifact binds wrong restorer")
    if artifact["restore_model_revision"] != RESTORE_MODEL_REVISION:
        raise EvaluationContractViolation("RESTORE head artifact binds wrong restorer revision")
    if artifact["seed"] != expected_seed:
        raise EvaluationContractViolation("RESTORE head artifact seed mismatch")
    if artifact["seed"] not in RESTORE_STAGE2_HEAD_SEEDS:
        raise EvaluationContractViolation("RESTORE head artifact uses an unfrozen seed")
    if artifact["learning_rate"] != RESTORE_STAGE2_HEAD_LR:
        raise EvaluationContractViolation("RESTORE head LR drifted")
    if artifact["epochs"] != RESTORE_STAGE2_HEAD_EPOCHS or artifact["early_stopping"] is not False:
        raise EvaluationContractViolation("RESTORE head epoch budget or early stopping drifted")
    if len(artifact["history"]) != RESTORE_STAGE2_HEAD_EPOCHS:
        raise EvaluationContractViolation("RESTORE head history must contain 30 epochs")
    require_full_schedule(
        [
            EpochScore(
                epoch=int(item["epoch"]),
                macro_f1=float(item["macro_f1"]),
                accuracy=float(item["accuracy"]),
            )
            for item in artifact["history"]
        ],
        RESTORE_STAGE2_HEAD_EPOCHS,
    )
    if artifact["selection_role"] != Preg1Role.PROTOCOL_DEV.value:
        raise EvaluationContractViolation("RESTORE head selected on wrong role")
    if artifact["selection_condition"] != "FULL":
        raise EvaluationContractViolation("RESTORE head selected on a non-clean condition")
    if artifact["official_validation_used_for_selection"] is not False:
        raise EvaluationContractViolation("RESTORE head used official validation for selection")
    if artifact["corrupted_label_training"] is not False:
        raise EvaluationContractViolation("RESTORE head used corrupted-label training")
    if artifact["best_seed_selection"] is not False:
        raise EvaluationContractViolation("RESTORE head artifact performs seed selection")
    train_key = RestoreRepresentationKey.from_dict(artifact["train_cache_key"])
    dev_key = RestoreRepresentationKey.from_dict(artifact["protocol_dev_cache_key"])
    if train_key.role != Preg1Role.PROTOCOL_TRAIN.value or dev_key.role != Preg1Role.PROTOCOL_DEV.value:
        raise EvaluationContractViolation("RESTORE head cache roles drifted")
    if train_key.condition != "FULL" or dev_key.condition != "FULL":
        raise EvaluationContractViolation("RESTORE head caches must be clean FULL")


class RestoreHeadStore:
    ARTIFACT_NAME = "restore-head-artifact.json"
    STATE_NAME = "restore-selected-head.pt"
    IN_PROGRESS_NAME = "RESTORE_HEAD.inprogress"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def artifact_path(self) -> Path:
        return self.directory / self.ARTIFACT_NAME

    @property
    def state_path(self) -> Path:
        return self.directory / self.STATE_NAME

    @property
    def in_progress_path(self) -> Path:
        return self.directory / self.IN_PROGRESS_NAME

    def exists(self) -> bool:
        return self.artifact_path.is_file() and self.state_path.is_file()

    def read_artifact(self, *, expected_seed: int) -> dict[str, Any]:
        if self.in_progress_path.exists():
            raise EvaluationContractViolation("RESTORE head store has in-progress marker")
        if not self.exists():
            raise EvaluationContractViolation(f"RESTORE head store incomplete at {self.directory}")
        artifact = read_json(self.artifact_path)
        validate_restore_head_artifact(artifact, expected_seed=expected_seed)
        if file_sha256(self.state_path) != artifact["selected_state_raw_sha256"]:
            raise EvaluationContractViolation("RESTORE selected head raw SHA drifted")
        return artifact

    def require_writable(self, *, expected_seed: int) -> None:
        if self.exists():
            self.read_artifact(expected_seed=expected_seed)
            raise EvaluationContractViolation("completed RESTORE head artifacts are immutable")
        if self.in_progress_path.exists():
            raise EvaluationContractViolation("RESTORE head in-progress marker must be inspected")

    def commit(self, artifact: Mapping[str, Any], selected_state: Mapping[str, Any]) -> None:
        import torch

        artifact = dict(artifact)
        seed = int(artifact["seed"])
        self.require_writable(expected_seed=seed)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.in_progress_path, {"seed": seed})
        temp = self.state_path.with_name(self.state_path.name + ".tmp")
        torch.save(dict(selected_state), temp)
        temp.replace(self.state_path)
        artifact["selected_state_raw_sha256"] = file_sha256(self.state_path)
        validate_restore_head_artifact(artifact, expected_seed=seed)
        write_json(self.artifact_path, dict(artifact))
        self.in_progress_path.unlink()

    def load_head(self, *, expected_seed: int) -> Any:
        import torch

        artifact = self.read_artifact(expected_seed=expected_seed)
        head = build_head(RESTORE_PHOBERT_HIDDEN_SIZE, expected_seed)
        state = torch.load(self.state_path, map_location="cpu")
        if _state_semantic_sha256(state) != artifact["selected_state_semantic_sha256"]:
            raise EvaluationContractViolation("RESTORE selected head semantic SHA drifted")
        head.load_state_dict(state)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        return head


def restore_head_schedule() -> tuple[int, ...]:
    return require_restore_head_seeds(RESTORE_STAGE2_HEAD_SEEDS)


def train_or_load_restore_heads(
    *,
    head_root: str | Path,
    repository_head: str,
    train: RestoreBoundRepresentations,
    train_labels: Sequence[int],
    dev: RestoreBoundRepresentations,
    dev_labels: Sequence[int],
) -> dict[int, dict[str, Any]]:
    artifacts: dict[int, dict[str, Any]] = {}
    for seed in restore_head_schedule():
        store = RestoreHeadStore(Path(head_root) / f"seed-{seed}")
        if store.exists():
            artifacts[seed] = store.read_artifact(expected_seed=seed)
            continue
        run = train_restore_head(
            train,
            train_labels,
            dev,
            dev_labels,
            seed=seed,
        )
        artifact = build_restore_head_artifact(run, repository_head=repository_head)
        store.commit(artifact, run.selected_state)
        artifacts[seed] = artifact
    if sorted(artifacts) != sorted(RESTORE_STAGE2_HEAD_SEEDS):
        raise EvaluationContractViolation("RESTORE did not freeze exactly five heads")
    return artifacts


@dataclass(frozen=True)
class RestoreConditionScore:
    seed: int
    condition: str
    macro_f1: float
    accuracy: float
    per_class_f1: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "condition": self.condition,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "per_class_f1": list(self.per_class_f1),
        }


def score_restore_condition(
    head: Any,
    representations: RestoreBoundRepresentations,
    labels: Sequence[int],
    *,
    seed: int,
) -> RestoreConditionScore:
    import torch

    representations.require_role(Preg1Role.OFFICIAL_VALIDATION, "RESTORE measurement")
    if representations.key.label_digest != label_digest(labels):
        raise EvaluationContractViolation("RESTORE measurement labels do not match representation key")
    if seed not in RESTORE_STAGE2_HEAD_SEEDS:
        raise EvaluationContractViolation("RESTORE measurement seed is not frozen")
    head.eval()
    with torch.no_grad():
        predictions = head(representations.values).argmax(dim=1).tolist()
    labels = [int(v) for v in labels]
    macro, acc = score_predictions(predictions, labels)
    per_class = per_class_scores(predictions, labels, num_labels=3)
    return RestoreConditionScore(
        seed=seed,
        condition=representations.condition,
        macro_f1=macro,
        accuracy=acc,
        per_class_f1=tuple(score.f1 for score in per_class),
    )


def sample_stdev(values: Sequence[float]) -> float:
    if not values:
        raise EvaluationContractViolation("cannot aggregate no values")
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def aggregate_restore_scores(scores: Sequence[RestoreConditionScore]) -> dict[str, Any]:
    require_restore_conditions(RESTORE_CONDITIONS)
    require_restore_head_seeds(RESTORE_STAGE2_HEAD_SEEDS)
    if len(scores) != RESTORE_SCORE_UNITS:
        raise EvaluationContractViolation(
            f"RESTORE measurement requires {RESTORE_SCORE_UNITS} score units, got {len(scores)}"
        )
    seen = {(score.seed, score.condition) for score in scores}
    expected = {(seed, condition) for seed in RESTORE_STAGE2_HEAD_SEEDS for condition in RESTORE_CONDITIONS}
    if seen != expected:
        raise EvaluationContractViolation("RESTORE measurement seed/condition grid is incomplete")

    by_condition: dict[str, Any] = {}
    for condition in RESTORE_CONDITIONS:
        rows = [score for score in scores if score.condition == condition]
        macro_values = [score.macro_f1 for score in rows]
        acc_values = [score.accuracy for score in rows]
        by_condition[condition] = {
            "macro_f1_mean": statistics.fmean(macro_values),
            "macro_f1_sample_std": sample_stdev(macro_values),
            "accuracy_mean": statistics.fmean(acc_values),
            "accuracy_sample_std": sample_stdev(acc_values),
            "per_class_f1_mean": [
                statistics.fmean(score.per_class_f1[index] for score in rows)
                for index in range(3)
            ],
        }
    degraded_macro = [by_condition[c]["macro_f1_mean"] for c in RESTORE_DEGRADED_CONDITIONS]
    degraded_acc = [by_condition[c]["accuracy_mean"] for c in RESTORE_DEGRADED_CONDITIONS]
    return {
        "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
        "protocol_version": RESTORE_PROTOCOL_VERSION,
        "model_id": RESTORE_MODEL_ID,
        "model_revision": RESTORE_MODEL_REVISION,
        "score_units": len(scores),
        "expected_score_units": RESTORE_SCORE_UNITS,
        "seeds": list(RESTORE_STAGE2_HEAD_SEEDS),
        "conditions": by_condition,
        "per_seed": [score.to_dict() for score in sorted(scores, key=lambda s: (s.seed, s.condition))],
        "degraded_equal_weight": {
            "macro_f1_mean": statistics.fmean(degraded_macro),
            "accuracy_mean": statistics.fmean(degraded_acc),
        },
        "sample_sd_aggregation": True,
        "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
    }


def diagnostics_for_condition(
    records: Sequence[RestoreTextRecord],
    stream: RestoreConditionStream,
) -> dict[str, Any]:
    rows = [
        RestoreDiagnosticRow(
            sample_id=record.sample_id,
            observed_text=record.input_text,
            restored_text=record.restored_text,
            clean_gold_text=gold,
        )
        for record, gold in zip(records, stream.clean_gold_texts)
    ]
    return restore_diagnostics(rows, condition=stream.condition)


def build_final_evidence(
    *,
    repository_head: str,
    model_preflight: Mapping[str, Any],
    clean_text_manifests: Mapping[str, Any],
    clean_representation_manifests: Mapping[str, Any],
    head_artifacts: Mapping[int, Mapping[str, Any]],
    validation_identity: Mapping[str, Any],
    validation_text_manifests: Mapping[str, Any],
    validation_representation_manifests: Mapping[str, Any],
    measurement: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    grr: Mapping[str, Any],
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
        "protocol_version": RESTORE_PROTOCOL_VERSION,
        "repository_head": repository_head,
        "restore_model_id": RESTORE_MODEL_ID,
        "restore_model_revision": RESTORE_MODEL_REVISION,
        "model_preflight": dict(model_preflight),
        "clean_text": dict(clean_text_manifests),
        "clean_representations": dict(clean_representation_manifests),
        "heads": {str(seed): dict(artifact) for seed, artifact in head_artifacts.items()},
        "validation_identity": dict(validation_identity),
        "validation_text": dict(validation_text_manifests),
        "validation_representations": dict(validation_representation_manifests),
        "measurement": dict(measurement),
        "diagnostics": dict(diagnostics),
        "grr": dict(grr),
        "comparison": dict(comparison) if comparison is not None else None,
        "boundaries": {
            "restore_uses_unmark_adapter": False,
            "restore_uses_base_tone_letter_channels": False,
            "restore_uses_unmark_stage1_checkpoint": False,
            "restore_trains_restorer": False,
            "restore_restorer_frozen": True,
            "corrupted_label_training": RESTORE_CORRUPTED_LABEL_TRAINING,
            "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
            "official_validation_previously_seen_by_authors": (
                RESTORE_OFFICIAL_VALIDATION_PREVIOUSLY_SEEN_BY_AUTHORS
            ),
            "official_test_read": False,
            "canonical_input_semantics": RESTORE_CANONICAL_INPUT_SEMANTICS,
            "hard_stop": True,
            "condition_aware_routing": False,
            "full_bypass": RESTORE_FULL_BYPASS,
            "one_restore_pathway": RESTORE_ONE_RESTORE_PATHWAY,
        },
    }
    return payload


def read_csv_restore_split(
    path: str | Path,
    *,
    role: Preg1Role,
    text_column: str,
    label_column: str,
    id_column: str,
    expected_sha256: str | None = None,
) -> RestoreSplit:
    """Small helper for synthetic tests and non-PREG1 callers."""

    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise EvaluationContractViolation(f"{source} SHA256 mismatch")
    records: list[tuple[str, str, int]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for column in (text_column, label_column, id_column):
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise EvaluationContractViolation(f"missing column {column!r}")
        for row in reader:
            records.append(
                (
                    str(row[id_column]),
                    row[text_column],
                    _label_index(row[label_column]),
                )
            )
    return RestoreSplit(
        role=role,
        sample_ids=tuple(sample_id for sample_id, _, _ in records),
        texts=tuple(text for _, text, _ in records),
        labels=tuple(label for _, _, label in records),
        source_sha256=file_sha256(source),
    )


__all__ = [
    "RESTORE_SCORE_UNITS",
    "RESTORE_STAGE2_MEASUREMENT_ROLE",
    "RESTORE_STAGE2_SELECTION_ROLE",
    "RESTORE_STAGE2_TRAINING_ROLE",
    "RestoreConditionScore",
    "RestoreConditionStream",
    "RestoreHeadRun",
    "RestoreHeadStore",
    "RestoreSplit",
    "aggregate_diagnostic_reports",
    "aggregate_restore_scores",
    "authenticate_read_only_evidence",
    "build_condition_streams",
    "build_final_evidence",
    "build_restore_head_artifact",
    "canonical_clean_texts",
    "compute_grr",
    "diagnostics_for_condition",
    "extract_condition_metrics",
    "extract_or_load_restore_representations",
    "load_restore_phobert_components",
    "load_restore_protocol_splits",
    "load_restore_validation_split",
    "minimal_comparison",
    "read_csv_restore_split",
    "representation_key_from_text_cache",
    "require_changed_row_counts",
    "require_restore_phobert_backbone",
    "require_restore_protocol_split_identities",
    "restore_head_schedule",
    "restore_records",
    "restore_text_cache",
    "restore_text_request_for_split",
    "restore_text_request_for_stream",
    "score_restore_condition",
    "train_or_load_restore_heads",
    "train_restore_head",
    "validate_restore_head_artifact",
]
