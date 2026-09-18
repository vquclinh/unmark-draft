"""ViUnMark Cross-Task NER Transfer Probe contracts.

This module implements the scientific plumbing for a frozen-token-probe
experiment on the official PhoNER_COVID19 word-level release. It deliberately is
not the final sentiment ViUnMark system: no sentiment readout, no class weights,
no calibration, no pooling fusion and no 20-head ensemble appear here.

Torch and Transformers are imported lazily by functions that need them. The
dataset, corruption, alignment, metric and seal contracts remain testable in the
standard lightweight repository environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from unmark.corruption import CorruptionPurpose, EligibilityPolicy, corrupt
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.modeling.collate import build_example
from unmark.orthography import Eligibility, canon, decompose
from unmark.stage1.data import project_text
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE, MAX_LENGTH
from unmark.viunmark.config import SIX_CONDITIONS
from unmark.viunmark.inputs import CorruptionProtocol, CorruptionRealizationKey

PHONER_TASK_ID = "PhoNER_COVID19"
PHONER_REPRESENTATION = "word"
PHONER_CROSS_TASK_PROTOCOL_VERSION = "phoner-cross-task-transfer-probe-v1"
PHONER_EXPERIMENT_NAME = "ViUnMark Cross-Task NER Transfer Probe"
PHONER_CORRUPTION_SEED = 19225
PHONER_FINAL_SEEDS: tuple[int, ...] = (53148, 59945, 42941, 720, 9428)
PHONER_SMOKE_SEED = 53148
PHONER_IGNORE_INDEX = -100
PHONER_GATE_SHA256 = "6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91"
PHONER_SCALE_SHA256 = "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685"
PHONER_DATASET_TERMS = (
    "Official PhoNER_COVID19 terms permit research/educational use and prohibit "
    "redistribution of original or modified dataset content."
)

PROBE_ARCHITECTURE = f"Linear({HIDDEN_SIZE}, num_ner_labels)"
PROBE_HEAD_HIDDEN_LAYERS = 0
PROBE_DROPOUT = 0.0
SENTIMENT_CALIBRATION_USED = False
SENTIMENT_MLP_USED = False
MM_CAT_POOLING_USED = False
UIT_VSFC_CLASS_WEIGHTS_USED = False
FINAL_SENTIMENT_FUSION_HEADS = 0
CORRUPTION_PARITY_CONCLUSION = (
    "PhoNER uses the same sample-level ViUnMark corruption semantics as the "
    "downstream measurement path: one call to corrupt(text, condition, seed, "
    "sample_id) per complete sample/condition, not independent per-word draws. "
    "The word sequence is joined with single spaces as a reversible word-level "
    "sentence representation, corrupted once, and split back onto the original "
    "word slots."
)

NEW_CROSS_TASK_PROBE_PROTOCOL_SOURCE = (
    "NEW CROSS-TASK PROBE PROTOCOL: repository inspection found no generic "
    "frozen token-level NER probe recipe. These values are prospective for the "
    "PhoNER transfer probe and are not historical ViUnMark/UIt-VSFC settings."
)


class PhoNERContractViolation(ValueError):
    """Raised when the PhoNER cross-task contract is violated."""


class PhoNERTestSealViolation(PhoNERContractViolation):
    """Raised before any forbidden TEST access occurs."""


class PhoNERPathway(Enum):
    """The three frozen encoders/pathways compared by the probe."""

    PHOBERT_NATIVE = "PHOBERT_NATIVE"
    VIUNMARK_GATE = "VIUNMARK_GATE"
    VIUNMARK_SCALE = "VIUNMARK_SCALE"


class PhoNERSplit(Enum):
    """Official PhoNER split identities."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


_STAGE_ALIASES = {
    "audit": "dataset-audit",
    "dataset-audit": "dataset-audit",
    "smoke": "smoke-train",
    "smoke-train": "smoke-train",
    "train-dev": "train-dev",
    "dev-evaluate": "dev-evaluate",
    "freeze": "freeze-protocol",
    "freeze-protocol": "freeze-protocol",
    "test-predict": "test-predict",
    "test-score": "test-score",
}

_TEST_TEXT_STAGES = frozenset({"test-predict", "test-score"})
_TEST_GOLD_STAGES = frozenset({"test-score"})


def canonical_stage(stage: str) -> str:
    try:
        return _STAGE_ALIASES[stage]
    except KeyError as error:
        raise PhoNERContractViolation(
            f"unknown stage {stage!r}; expected one of {sorted(_STAGE_ALIASES)}"
        ) from error


def require_stage_access(stage: str, *, split: str | PhoNERSplit, gold: bool) -> None:
    """Fail before opening a file if a stage would break the TRAIN/DEV/TEST seal."""

    resolved_stage = canonical_stage(stage)
    split_value = split.value if isinstance(split, PhoNERSplit) else str(split)
    if split_value != PhoNERSplit.TEST.value:
        return
    if gold and resolved_stage not in _TEST_GOLD_STAGES:
        raise PhoNERTestSealViolation(
            f"stage {resolved_stage!r} may not read PhoNER TEST gold labels"
        )
    if not gold and resolved_stage not in _TEST_TEXT_STAGES:
        raise PhoNERTestSealViolation(
            f"stage {resolved_stage!r} may not open PhoNER TEST contents"
        )


@dataclass(frozen=True)
class FrozenTokenProbeTrainingPolicy:
    """Prospective policy for the new cross-task token probe.

    The policy is intentionally simple and shared across all three pathways.
    It is recorded as new protocol, not recovered historical ViUnMark protocol.
    """

    source: str = NEW_CROSS_TASK_PROBE_PROTOCOL_SOURCE
    architecture: str = PROBE_ARCHITECTURE
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_optimizer_updates: int = 1000
    eval_every_updates: int = 100
    loss: str = "cross_entropy(ignore_index=-100, unweighted)"
    scheduler: str = "none"
    warmup_updates: int = 0
    gradient_clipping: str = "none"
    amp: str = "disabled"
    dataloader_shuffle_policy: str = (
        "deterministic per-update reshuffle of TRAIN/FULL chunks only; "
        "shuffle seed = run_seed * 100000 + optimizer_update"
    )
    epoch_rollover_behavior: str = (
        "cycle through the shuffled training chunks until max_optimizer_updates; "
        "reshuffle at each pass with the derived update seed"
    )
    checkpoint_tie_breaking: str = (
        "highest DEV/FULL entity micro F1, then lower validation loss if recorded, "
        "then earlier optimizer update"
    )
    deterministic_settings: str = (
        "seed Python random and torch CPU/CUDA RNGs per run; no AMP; no scheduler; "
        "device-level deterministic kernels are requested where practical by the runner"
    )
    selection_metric: str = "PhoNER DEV/FULL exact entity-level micro F1"
    initialization: str = "torch.nn.Linear default initialization under the run seed"
    precision: str = "float32; no AMP requested by this protocol"
    final_seeds: tuple[int, ...] = PHONER_FINAL_SEEDS
    smoke_seed: int = PHONER_SMOKE_SEED

    def __post_init__(self) -> None:
        if self.final_seeds != PHONER_FINAL_SEEDS:
            raise PhoNERContractViolation("final seed set is fixed for this protocol")
        if self.max_optimizer_updates % self.eval_every_updates:
            raise PhoNERContractViolation("max updates must be divisible by eval interval")
        if self.architecture != PROBE_ARCHITECTURE:
            raise PhoNERContractViolation("all pathways must use the same linear token probe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_optimizer_updates": self.max_optimizer_updates,
            "eval_every_updates": self.eval_every_updates,
            "loss": self.loss,
            "scheduler": self.scheduler,
            "warmup_updates": self.warmup_updates,
            "gradient_clipping": self.gradient_clipping,
            "amp": self.amp,
            "dataloader_shuffle_policy": self.dataloader_shuffle_policy,
            "epoch_rollover_behavior": self.epoch_rollover_behavior,
            "checkpoint_tie_breaking": self.checkpoint_tie_breaking,
            "deterministic_settings": self.deterministic_settings,
            "selection_metric": self.selection_metric,
            "initialization": self.initialization,
            "precision": self.precision,
            "final_seeds": list(self.final_seeds),
            "smoke_seed": self.smoke_seed,
        }


@dataclass(frozen=True)
class PhoNERCrossTaskConfig:
    """Dataset and protocol identity for the transfer probe."""

    dataset_id: str = PHONER_TASK_ID
    representation: str = PHONER_REPRESENTATION
    experiment_name: str = PHONER_EXPERIMENT_NAME
    protocol_version: str = PHONER_CROSS_TASK_PROTOCOL_VERSION
    encoder_checkpoint: str = ENCODER_CHECKPOINT
    encoder_revision: str = ENCODER_REVISION
    gate_checkpoint_sha256: str = PHONER_GATE_SHA256
    scale_checkpoint_sha256: str = PHONER_SCALE_SHA256
    hidden_size: int = HIDDEN_SIZE
    max_length: int = MAX_LENGTH
    corruption_seed: int = PHONER_CORRUPTION_SEED
    conditions: tuple[str, ...] = SIX_CONDITIONS
    pathways: tuple[PhoNERPathway, ...] = (
        PhoNERPathway.PHOBERT_NATIVE,
        PhoNERPathway.VIUNMARK_GATE,
        PhoNERPathway.VIUNMARK_SCALE,
    )
    probe_policy: FrozenTokenProbeTrainingPolicy = field(
        default_factory=FrozenTokenProbeTrainingPolicy
    )

    def __post_init__(self) -> None:
        if self.dataset_id != PHONER_TASK_ID or self.representation != PHONER_REPRESENTATION:
            raise PhoNERContractViolation("this protocol is only for PhoNER_COVID19 word-level")
        if tuple(self.conditions) != SIX_CONDITIONS:
            raise PhoNERContractViolation(f"condition grid must be {list(SIX_CONDITIONS)}")
        if self.corruption_seed != PHONER_CORRUPTION_SEED:
            raise PhoNERContractViolation("PhoNER corruption seed is fixed at 19225")
        if self.encoder_checkpoint != ENCODER_CHECKPOINT or self.encoder_revision != ENCODER_REVISION:
            raise PhoNERContractViolation("PhoBERT tokenizer/encoder identity drifted")

    @property
    def corruption_protocol(self) -> CorruptionProtocol:
        return CorruptionProtocol(scientific_corruption_seed=self.corruption_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "representation": self.representation,
            "experiment_name": self.experiment_name,
            "protocol_version": self.protocol_version,
            "encoder_checkpoint": self.encoder_checkpoint,
            "encoder_revision": self.encoder_revision,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
            "scale_checkpoint_sha256": self.scale_checkpoint_sha256,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "corruption_seed": self.corruption_seed,
            "conditions": list(self.conditions),
            "pathways": [p.value for p in self.pathways],
            "probe_policy": self.probe_policy.to_dict(),
            "negative_controls": {
                "sentiment_mlp_used": SENTIMENT_MLP_USED,
                "mm_cat_pooling_used": MM_CAT_POOLING_USED,
                "uit_vsfc_class_weights_used": UIT_VSFC_CLASS_WEIGHTS_USED,
                "sentiment_calibration_used": SENTIMENT_CALIBRATION_USED,
                "final_sentiment_fusion_heads": FINAL_SENTIMENT_FUSION_HEADS,
            },
        }


@dataclass(frozen=True)
class PhoNERExample:
    """One word-level PhoNER sentence."""

    sample_id: str
    split: PhoNERSplit
    words: tuple[str, ...]
    labels: tuple[str, ...] | None
    source_record_index: int
    source_sample_id: str | None = None
    word_start: int = 0
    word_end: int | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise PhoNERContractViolation("sample_id must be non-empty")
        if not self.words:
            raise PhoNERContractViolation("PhoNER examples must contain at least one word")
        if self.labels is not None and len(self.words) != len(self.labels):
            raise PhoNERContractViolation(
                f"word/label length mismatch in {self.sample_id}: "
                f"{len(self.words)} words vs {len(self.labels)} labels"
            )
        if self.word_end is not None and self.word_end - self.word_start != len(self.words):
            raise PhoNERContractViolation("chunk word span does not match word count")


@dataclass(frozen=True)
class PhoNERSplitFileIdentity:
    split: str
    path: str | None
    format: str | None
    sha256: str
    row_count: int | str
    gold_labels_read: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "path": self.path,
            "format": self.format,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "gold_labels_read": self.gold_labels_read,
        }


@dataclass(frozen=True)
class PhoNERDatasetIdentity:
    dataset_id: str
    representation: str
    source_provenance: str
    expected_schema: Mapping[str, Any]
    split_files: Mapping[str, PhoNERSplitFileIdentity]
    label_inventory_source: str
    entity_label_inventory: tuple[str, ...]
    stable_sample_id_policy: str
    dataset_terms: str
    train_dev_coverage_audit: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "representation": self.representation,
            "source_provenance": self.source_provenance,
            "expected_schema": dict(self.expected_schema),
            "split_files": {k: v.to_dict() for k, v in self.split_files.items()},
            "label_inventory_source": self.label_inventory_source,
            "entity_label_inventory": list(self.entity_label_inventory),
            "stable_sample_id_policy": self.stable_sample_id_policy,
            "dataset_terms": self.dataset_terms,
            "train_dev_coverage_audit": self.train_dev_coverage_audit,
        }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_checkpoint_sha256(path: str | Path, expected_sha256: str, *, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise PhoNERContractViolation(
            f"{label} checkpoint sha256 mismatch: got {observed}, expected {expected_sha256}"
        )
    return observed


def require_phobert_identity(tokenizer: Any | None = None, encoder: Any | None = None) -> None:
    for name, object_ in (("tokenizer", tokenizer), ("encoder", encoder)):
        if object_ is None:
            continue
        path = getattr(object_, "name_or_path", None)
        if path is not None and path != ENCODER_CHECKPOINT:
            raise PhoNERContractViolation(
                f"PhoBERT {name} checkpoint is {path!r}, expected {ENCODER_CHECKPOINT!r}"
            )
        revision = None
        for holder in (object_, getattr(object_, "config", None)):
            for attr in ("_commit_hash", "revision", "commit_hash", "_unmark_requested_revision"):
                value = getattr(holder, attr, None)
                if isinstance(value, str) and value:
                    revision = value
                    break
            if revision:
                break
        if revision is not None and revision != ENCODER_REVISION:
            raise PhoNERContractViolation(
                f"PhoBERT {name} revision is {revision!r}, expected {ENCODER_REVISION!r}"
            )


def _split_from_value(value: str | PhoNERSplit) -> PhoNERSplit:
    if isinstance(value, PhoNERSplit):
        return value
    try:
        return PhoNERSplit(str(value))
    except ValueError as error:
        raise PhoNERContractViolation(f"unknown PhoNER split {value!r}") from error


def find_phoner_split_file(data_root: str | Path, split: str | PhoNERSplit) -> tuple[Path, str]:
    root = Path(data_root)
    split_value = _split_from_value(split).value
    candidates = (
        (root / f"{split_value}_word.conll", "conll"),
        (root / f"{split_value}_word.json", "json"),
    )
    existing = [(path, fmt) for path, fmt in candidates if path.exists()]
    if len(existing) != 1:
        names = ", ".join(str(path) for path, _ in candidates)
        raise PhoNERContractViolation(
            f"expected exactly one PhoNER {split_value} word-level file among {names}; "
            f"found {len(existing)}"
        )
    return existing[0]


def _stable_sample_id(split: PhoNERSplit, source_record_index: int, words: Sequence[str]) -> str:
    payload = {
        "dataset_id": PHONER_TASK_ID,
        "representation": PHONER_REPRESENTATION,
        "split": split.value,
        "source_record_index": source_record_index,
        "words": list(words),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        f"{PHONER_TASK_ID}:{PHONER_REPRESENTATION}:{split.value}:"
        f"{source_record_index:08d}:{digest[:24]}"
    )


def _dedupe_sample_ids(examples: Iterable[tuple[tuple[str, ...], tuple[str, ...] | None]], split: PhoNERSplit) -> list[PhoNERExample]:
    seen: set[str] = set()
    out: list[PhoNERExample] = []
    for source_record_index, (words, labels) in enumerate(examples):
        sample_id = _stable_sample_id(split, source_record_index, words)
        if sample_id in seen:
            raise PhoNERContractViolation(f"duplicate sample_id {sample_id!r} within {split.value}")
        seen.add(sample_id)
        out.append(
            PhoNERExample(
                sample_id=sample_id,
                split=split,
                words=words,
                labels=labels,
                source_record_index=source_record_index,
                source_sample_id=sample_id,
                word_start=0,
                word_end=len(words),
            )
        )
    return out


def _parse_conll(path: Path, split: PhoNERSplit, *, include_labels: bool) -> list[PhoNERExample]:
    rows: list[tuple[tuple[str, ...], tuple[str, ...] | None]] = []
    words: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if words:
                    rows.append((tuple(words), tuple(labels) if include_labels else None))
                    words, labels = [], []
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 and include_labels:
                raise PhoNERContractViolation(f"expected at least word and label columns in {path}")
            words.append(parts[0])
            if include_labels:
                labels.append(parts[-1])
        if words:
            rows.append((tuple(words), tuple(labels) if include_labels else None))
    return _dedupe_sample_ids(rows, split)


def _records_from_json(payload: Any, split: PhoNERSplit) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in (split.value, "data", "sentences", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise PhoNERContractViolation("PhoNER JSON must be a list or contain data/sentences/examples")


def _json_words_labels(record: Any, *, include_labels: bool) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if not isinstance(record, Mapping):
        raise PhoNERContractViolation("PhoNER JSON records must be objects")
    words = None
    for key in ("words", "tokens", "token", "sentence"):
        value = record.get(key)
        if isinstance(value, list):
            words = tuple(str(item) for item in value)
            break
    if not words:
        raise PhoNERContractViolation("PhoNER JSON record has no words/tokens list")
    if not include_labels:
        return words, None
    labels = None
    for key in ("labels", "tags", "ner_tags", "ner", "entities"):
        value = record.get(key)
        if isinstance(value, list):
            labels = tuple(str(item) for item in value)
            break
    if labels is None:
        raise PhoNERContractViolation("PhoNER JSON record has no labels/tags list")
    return words, labels


def _parse_json(path: Path, split: PhoNERSplit, *, include_labels: bool) -> list[PhoNERExample]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [_json_words_labels(record, include_labels=include_labels) for record in _records_from_json(payload, split)]
    return _dedupe_sample_ids(rows, split)


def parse_phoner_split(
    data_root: str | Path,
    split: str | PhoNERSplit,
    *,
    include_labels: bool = True,
    stage: str = "dataset-audit",
) -> tuple[PhoNERExample, ...]:
    """Parse one official PhoNER word-level split.

    TEST gold labels are refused unless `stage` is `test-score`; TEST text is
    refused before `test-predict`.
    """

    resolved = _split_from_value(split)
    require_stage_access(stage, split=resolved, gold=include_labels)
    path, fmt = find_phoner_split_file(data_root, resolved)
    if fmt == "conll":
        examples = _parse_conll(path, resolved, include_labels=include_labels)
    else:
        examples = _parse_json(path, resolved, include_labels=include_labels)
    return tuple(examples)


def label_inventory_from_train(train_examples: Sequence[PhoNERExample]) -> tuple[str, ...]:
    labels: set[str] = set()
    for example in train_examples:
        if example.labels is None:
            raise PhoNERContractViolation("TRAIN label inventory requires TRAIN gold labels")
        labels.update(example.labels)
    if "O" not in labels:
        raise PhoNERContractViolation("PhoNER TRAIN label inventory must include O")
    return tuple(["O"] + sorted(label for label in labels if label != "O"))


def _content_token_count(words: Sequence[str], tokenizer: Any) -> int:
    return sum(len(_tokens_for_word(tokenizer, word)) for word in words)


def _model_length(words: Sequence[str], tokenizer: Any) -> int:
    content = [0] * _content_token_count(words, tokenizer)
    return len(tokenizer.build_inputs_with_special_tokens(content))


def _word_content_ranges(words: Sequence[str], tokenizer: Any) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        width = len(_tokens_for_word(tokenizer, word))
        ranges.append((cursor, cursor + width))
        cursor += width
    return tuple(ranges)


def _naive_covered_word_count(words: Sequence[str], tokenizer: Any, *, max_length: int) -> int:
    full = tokenizer.build_inputs_with_special_tokens([0] * _content_token_count(words, tokenizer))
    if len(full) <= max_length:
        return len(words)
    special_count = len(full) - _content_token_count(words, tokenizer)
    keep_content = max(0, max_length - special_count)
    covered = 0
    for start, end in _word_content_ranges(words, tokenizer):
        if end <= keep_content:
            covered += 1
    return covered


def _boundary_is_entity_safe(labels: Sequence[str] | None, boundary: int) -> bool:
    if labels is None or boundary <= 0 or boundary >= len(labels):
        return True
    next_label = labels[boundary]
    return not next_label.startswith("I-")


def _entity_spans(labels: Sequence[str] | None) -> frozenset[tuple[int, int, str]]:
    return frozenset() if labels is None else bio_entities(labels)


def _affected_entities_under_naive_truncation(
    example: PhoNERExample,
    condition: str,
    tokenizer: Any,
    config: PhoNERCrossTaskConfig,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> int:
    corrupted = corrupt_phoner_example(example, condition, config=config, purpose=purpose)
    covered = _naive_covered_word_count(corrupted.words, tokenizer, max_length=config.max_length)
    return sum(1 for start, end, _ in _entity_spans(example.labels) if end > covered)


@dataclass(frozen=True)
class PhoNERWordChunk:
    sample_id: str
    source_sample_id: str
    word_start: int
    word_end: int

    def slice(self, example: PhoNERExample) -> PhoNERExample:
        if self.source_sample_id != example.sample_id:
            raise PhoNERContractViolation("chunk source does not match example")
        words = example.words[self.word_start:self.word_end]
        labels = None if example.labels is None else example.labels[self.word_start:self.word_end]
        return PhoNERExample(
            sample_id=self.sample_id,
            split=example.split,
            words=words,
            labels=labels,
            source_record_index=example.source_record_index,
            source_sample_id=example.sample_id,
            word_start=self.word_start,
            word_end=self.word_end,
        )


def condition_invariant_chunks_for_example(
    example: PhoNERExample,
    tokenizer: Any,
    config: PhoNERCrossTaskConfig | None = None,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> tuple[PhoNERWordChunk, ...]:
    """Derive word chunks that fit all six conditions and never split entities."""

    cfg = config or PhoNERCrossTaskConfig()
    by_condition = {
        condition: corrupt_phoner_example(example, condition, config=cfg, purpose=purpose).words
        for condition in cfg.conditions
    }
    chunks: list[PhoNERWordChunk] = []
    start = 0
    n = len(example.words)
    while start < n:
        best: int | None = None
        for end in range(start + 1, n + 1):
            if not _boundary_is_entity_safe(example.labels, end):
                continue
            candidate_ok = all(
                _model_length(words[start:end], tokenizer) <= cfg.max_length
                for words in by_condition.values()
            )
            if candidate_ok:
                best = end
        if best is None:
            raise PhoNERContractViolation(
                f"cannot create a condition-invariant max_length={cfg.max_length} chunk "
                f"for {example.sample_id!r} starting at word {start}; a single entity or "
                "word span is too long under at least one condition"
            )
        chunks.append(
            PhoNERWordChunk(
                sample_id=f"{example.sample_id}:words-{start}-{best}",
                source_sample_id=example.sample_id,
                word_start=start,
                word_end=best,
            )
        )
        start = best
    return tuple(chunks)


def materialize_condition_invariant_chunks(
    examples: Sequence[PhoNERExample],
    tokenizer: Any,
    config: PhoNERCrossTaskConfig | None = None,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> tuple[PhoNERExample, ...]:
    chunks: list[PhoNERExample] = []
    for example in examples:
        for chunk in condition_invariant_chunks_for_example(
            example, tokenizer, config, purpose=purpose
        ):
            chunks.append(chunk.slice(example))
    return tuple(chunks)


def audit_condition_invariant_coverage(
    examples: Sequence[PhoNERExample],
    tokenizer: Any,
    config: PhoNERCrossTaskConfig | None = None,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> dict[str, Any]:
    cfg = config or PhoNERCrossTaskConfig()
    max_lengths = {condition: 0 for condition in cfg.conditions}
    overlength = {condition: 0 for condition in cfg.conditions}
    affected = {condition: 0 for condition in cfg.conditions}
    chunks_by_sample: dict[str, list[dict[str, int | str]]] = {}
    original_entities = 0
    chunk_entities = 0
    original_words = 0
    chunk_words = 0
    for example in examples:
        original_words += len(example.words)
        original_entities += len(_entity_spans(example.labels))
        for condition in cfg.conditions:
            corrupted = corrupt_phoner_example(example, condition, config=cfg, purpose=purpose)
            length = _model_length(corrupted.words, tokenizer)
            max_lengths[condition] = max(max_lengths[condition], length)
            if length > cfg.max_length:
                overlength[condition] += 1
            affected[condition] += _affected_entities_under_naive_truncation(
                example, condition, tokenizer, cfg, purpose=purpose
            )
        sample_chunks = condition_invariant_chunks_for_example(
            example, tokenizer, cfg, purpose=purpose
        )
        chunks_by_sample[example.sample_id] = [
            {"sample_id": chunk.sample_id, "word_start": chunk.word_start, "word_end": chunk.word_end}
            for chunk in sample_chunks
        ]
        for chunk in sample_chunks:
            materialized = chunk.slice(example)
            chunk_words += len(materialized.words)
            chunk_entities += len(_entity_spans(materialized.labels))
            for condition in cfg.conditions:
                corrupted = corrupt_phoner_example(materialized, condition, config=cfg, purpose=purpose)
                length = _model_length(corrupted.words, tokenizer)
                if length > cfg.max_length:
                    raise PhoNERContractViolation(
                        f"chunk {materialized.sample_id!r} exceeds max_length under {condition}"
                    )
    identical = original_words == chunk_words and original_entities == chunk_entities
    return {
        "max_subword_length_by_condition": max_lengths,
        "overlength_samples_by_condition": overlength,
        "gold_entities_affected_by_naive_truncation": affected,
        "final_chunking_coverage_policy": (
            "condition-invariant word-level chunks; boundaries are shared across all "
            "six conditions, never split BIO entity continuations, and each chunk fits "
            f"the pinned max_length={cfg.max_length}"
        ),
        "evaluated_word_entity_coverage_identical_across_conditions": identical,
        "original_word_count": original_words,
        "chunk_word_count": chunk_words,
        "original_entity_count": original_entities,
        "chunk_entity_count": chunk_entities,
        "chunks_by_sample": chunks_by_sample,
    }


def audit_train_dev_condition_invariant_coverage(
    splits: Mapping[str, Sequence[PhoNERExample]],
    tokenizer: Any,
    config: PhoNERCrossTaskConfig | None = None,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> dict[str, Any]:
    cfg = config or PhoNERCrossTaskConfig()
    allowed = {"train", "dev"}
    unexpected = sorted(set(splits) - allowed)
    if unexpected:
        raise PhoNERTestSealViolation(f"coverage audit may not inspect splits {unexpected}")
    return {
        split: audit_condition_invariant_coverage(examples, tokenizer, cfg, purpose=purpose)
        for split, examples in splits.items()
    }


def build_phoner_dataset_identity(
    data_root: str | Path,
    *,
    stage: str = "dataset-audit",
    allow_test_hash: bool = False,
    tokenizer: Any | None = None,
    coverage_purpose: CorruptionPurpose = CorruptionPurpose.SELF_CHECK,
) -> PhoNERDatasetIdentity:
    """Build a dataset identity without opening TEST unless explicitly allowed."""

    train = parse_phoner_split(data_root, PhoNERSplit.TRAIN, include_labels=True, stage=stage)
    dev = parse_phoner_split(data_root, PhoNERSplit.DEV, include_labels=True, stage=stage)
    inventory = label_inventory_from_train(train)
    split_files: dict[str, PhoNERSplitFileIdentity] = {}
    for split, examples, read_gold in (
        (PhoNERSplit.TRAIN, train, True),
        (PhoNERSplit.DEV, dev, True),
    ):
        path, fmt = find_phoner_split_file(data_root, split)
        split_files[split.value] = PhoNERSplitFileIdentity(
            split=split.value,
            path=str(path),
            format=fmt,
            sha256=sha256_file(path),
            row_count=len(examples),
            gold_labels_read=read_gold,
        )

    test_path, test_fmt = find_phoner_split_file(data_root, PhoNERSplit.TEST)
    if allow_test_hash:
        require_stage_access(stage, split=PhoNERSplit.TEST, gold=False)
        split_files["test"] = PhoNERSplitFileIdentity(
            split="test",
            path=str(test_path),
            format=test_fmt,
            sha256=sha256_file(test_path),
            row_count="UNCOUNTED_TEST_CONTENTS_READ_FOR_HASH_ONLY",
            gold_labels_read=False,
        )
    else:
        split_files["test"] = PhoNERSplitFileIdentity(
            split="test",
            path=str(test_path),
            format=test_fmt,
            sha256="SEALED_UNREAD",
            row_count="SEALED_UNREAD",
            gold_labels_read=False,
        )

    return PhoNERDatasetIdentity(
        dataset_id=PHONER_TASK_ID,
        representation=PHONER_REPRESENTATION,
        source_provenance=(
            "Official PhoNER_COVID19 word-level release supplied externally via DATA_ROOT; "
            "raw files are not downloaded or committed by this repository."
        ),
        expected_schema={
            "conll": "blank-line-separated sentences; first column word, final column BIO/entity label",
            "json": "records with words/tokens and labels/tags lists",
            "official_files": [
                "train_word.conll or train_word.json",
                "dev_word.conll or dev_word.json",
                "test_word.conll or test_word.json",
            ],
        },
        split_files=split_files,
        label_inventory_source="TRAIN only",
        entity_label_inventory=inventory,
        stable_sample_id_policy=(
            "dataset_id + representation + split + deterministic source_record_index + "
            "sha256(word sequence), deliberately excluding gold labels so text-only "
            "test-predict IDs match labeled test-score IDs"
        ),
        dataset_terms=PHONER_DATASET_TERMS,
        train_dev_coverage_audit=(
            None
            if tokenizer is None
            else audit_train_dev_condition_invariant_coverage(
                {"train": train, "dev": dev}, tokenizer, PhoNERCrossTaskConfig(), purpose=coverage_purpose
            )
        ),
    )


def _join_words_for_corruption(words: Sequence[str]) -> str:
    for word in words:
        if not isinstance(word, str) or not word:
            raise PhoNERContractViolation("PhoNER word surfaces must be non-empty strings")
        if word != word.strip() or any(ch.isspace() for ch in word):
            raise PhoNERContractViolation(
                f"word surface {word!r} is not reversible under single-space joining"
            )
    return " ".join(words)


def _split_corrupted_sentence(corrupted_text: str, *, expected_words: int, sample_id: str) -> tuple[str, ...]:
    pieces = corrupted_text.split(" ")
    if len(pieces) != expected_words or any(piece == "" for piece in pieces):
        raise PhoNERContractViolation(
            f"sample-level corruption changed the word-slot structure for {sample_id!r}: "
            f"expected {expected_words} slots, got {len(pieces)}"
        )
    return tuple(pieces)


def corrupt_phoner_example(
    example: PhoNERExample,
    condition: str,
    *,
    config: PhoNERCrossTaskConfig | None = None,
    purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> PhoNERExample:
    """Corrupt the complete sample once and map surfaces back to word slots.

    This matches the authoritative ViUnMark downstream semantics:
    `corrupt(text, condition, seed, sample_id)` is called once for the whole
    sample. Per-word independent draws are deliberately not used.
    """

    cfg = config or PhoNERCrossTaskConfig()
    seed = cfg.corruption_protocol.api_seed_for(condition)
    clean_sentence = _join_words_for_corruption(example.words)
    corruption = corrupt(
        clean_sentence,
        condition,
        seed=seed,
        sample_id=example.sample_id,
        purpose=purpose,
        eligibility_policy=eligibility_policy,
    )
    corrupted_words = _split_corrupted_sentence(
        corruption.corrupted_text, expected_words=len(example.words), sample_id=example.sample_id
    )
    if len(corrupted_words) != len(example.words):
        raise PhoNERContractViolation("corruption changed word count")
    return PhoNERExample(
        sample_id=example.sample_id,
        split=example.split,
        words=corrupted_words,
        labels=example.labels,
        source_record_index=example.source_record_index,
        source_sample_id=example.source_sample_id or example.sample_id,
        word_start=example.word_start,
        word_end=example.word_end if example.word_end is not None else example.word_start + len(example.words),
    )


def cross_pathway_corrupted_inputs(
    example: PhoNERExample,
    condition: str,
    *,
    config: PhoNERCrossTaskConfig | None = None,
    purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> dict[PhoNERPathway, tuple[str, ...]]:
    """Return the one shared corrupted word sequence each pathway must consume."""

    cfg = config or PhoNERCrossTaskConfig()
    corrupted = corrupt_phoner_example(
        example,
        condition,
        config=cfg,
        purpose=purpose,
        eligibility_policy=eligibility_policy,
    )
    key = CorruptionRealizationKey(
        dataset=cfg.dataset_id,
        role=example.split.value,
        sample_id=example.sample_id,
        condition=condition,
        scientific_corruption_seed=cfg.corruption_seed,
    )
    joined = " ".join(corrupted.words)
    realizations = {
        pathway: {key: joined}
        for pathway in cfg.pathways
        if pathway is not PhoNERPathway.PHOBERT_NATIVE
    }
    # Reuse the existing ViUnMark identity checker for Gate/Scale.
    from unmark.viunmark.config import Pathway as ViUnMarkPathway
    from unmark.viunmark.inputs import require_cross_pathway_input_identity

    require_cross_pathway_input_identity(
        {
            ViUnMarkPathway.VIUNMARK_GATE: realizations[PhoNERPathway.VIUNMARK_GATE],
            ViUnMarkPathway.VIUNMARK_SCALE: realizations[PhoNERPathway.VIUNMARK_SCALE],
        }
    )
    return {pathway: corrupted.words for pathway in cfg.pathways}


@dataclass(frozen=True)
class TokenizedNERExample:
    sample_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    label_ids: tuple[int, ...]
    word_ids: tuple[int | None, ...]
    words: tuple[str, ...]
    labels: tuple[str, ...] | None
    truncated: bool
    tone_ids: tuple[int, ...] | None = None
    tone_mask: tuple[bool, ...] | None = None
    letter_ids: tuple[tuple[int, ...], ...] | None = None


def _tokens_for_word(tokenizer: Any, word: str) -> list[str]:
    pieces = list(tokenizer.tokenize(word))
    if pieces:
        return pieces
    unk = getattr(tokenizer, "unk_token", None)
    return [unk] if unk is not None else [word]


def _labels_for_content_tokens(
    words: Sequence[str],
    labels: Sequence[str] | None,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
) -> tuple[list[int], list[int | None], list[int]]:
    content_ids: list[int] = []
    content_label_ids: list[int] = []
    content_word_ids: list[int | None] = []
    for word_index, word in enumerate(words):
        pieces = _tokens_for_word(tokenizer, word)
        ids = list(tokenizer.convert_tokens_to_ids(pieces))
        content_ids.extend(ids)
        for piece_index, _ in enumerate(ids):
            content_word_ids.append(word_index)
            if labels is None:
                content_label_ids.append(PHONER_IGNORE_INDEX)
            elif piece_index == 0:
                try:
                    content_label_ids.append(label_to_id[labels[word_index]])
                except KeyError as error:
                    raise PhoNERContractViolation(
                        f"label {labels[word_index]!r} is absent from TRAIN-derived inventory"
                    ) from error
            else:
                content_label_ids.append(PHONER_IGNORE_INDEX)
    return content_ids, content_word_ids, content_label_ids


def _with_special(tokenizer: Any, content_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    input_ids = list(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    mask = list(
        tokenizer.get_special_tokens_mask(list(content_ids), already_has_special_tokens=False)
    )
    if len(input_ids) != len(mask):
        raise PhoNERContractViolation("tokenizer special mask length mismatch")
    return input_ids, mask


def _truncate_content(
    tokenizer: Any,
    content_ids: list[int],
    content_word_ids: list[int | None],
    content_label_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int | None], list[int], bool]:
    full, _ = _with_special(tokenizer, content_ids)
    if len(full) <= max_length:
        return content_ids, content_word_ids, content_label_ids, False
    special_count = len(full) - len(content_ids)
    keep = max_length - special_count
    if keep < 0:
        raise PhoNERContractViolation(
            f"max_length={max_length} cannot hold tokenizer special tokens"
        )
    return content_ids[:keep], content_word_ids[:keep], content_label_ids[:keep], True


def align_word_labels(
    words: Sequence[str],
    labels: Sequence[str] | None,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    *,
    max_length: int = MAX_LENGTH,
) -> TokenizedNERExample:
    """Align word-level BIO labels to PhoBERT subwords.

    First subtoken of a word receives the word label; remaining subtokens,
    special tokens and padding positions receive `PHONER_IGNORE_INDEX`.
    """

    if labels is not None and len(words) != len(labels):
        raise PhoNERContractViolation("word/label count mismatch")
    content_ids, content_word_ids, content_label_ids = _labels_for_content_tokens(
        words, labels, tokenizer, label_to_id
    )
    content_ids, content_word_ids, content_label_ids, truncated = _truncate_content(
        tokenizer, content_ids, content_word_ids, content_label_ids, max_length
    )
    input_ids, special_mask = _with_special(tokenizer, content_ids)
    label_iter = iter(content_label_ids)
    word_iter = iter(content_word_ids)
    aligned_labels: list[int] = []
    aligned_word_ids: list[int | None] = []
    for special in special_mask:
        if special:
            aligned_labels.append(PHONER_IGNORE_INDEX)
            aligned_word_ids.append(None)
        else:
            aligned_labels.append(next(label_iter))
            aligned_word_ids.append(next(word_iter))
    return TokenizedNERExample(
        sample_id="",
        input_ids=tuple(input_ids),
        attention_mask=tuple(1 for _ in input_ids),
        special_tokens_mask=tuple(special_mask),
        label_ids=tuple(aligned_labels),
        word_ids=tuple(aligned_word_ids),
        words=tuple(words),
        labels=tuple(labels) if labels is not None else None,
        truncated=truncated,
    )


def _base_words(words: Sequence[str]) -> tuple[str, ...]:
    return tuple(decompose(canon(word)).base_text for word in words)


def encode_native_example(
    example: PhoNERExample,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    *,
    max_length: int = MAX_LENGTH,
) -> TokenizedNERExample:
    aligned = align_word_labels(example.words, example.labels, tokenizer, label_to_id, max_length=max_length)
    return TokenizedNERExample(**{**aligned.__dict__, "sample_id": example.sample_id})


def encode_adapted_example(
    example: PhoNERExample,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    *,
    classifier: Callable[[str], Eligibility] | None = None,
    unk_token_id: int | None = None,
    max_length: int = MAX_LENGTH,
) -> TokenizedNERExample:
    """Encode a corrupted word sequence for Gate/Scale on the base grid."""

    aligned = align_word_labels(_base_words(example.words), example.labels, tokenizer, label_to_id, max_length=max_length)
    base_text, content_ids, projections = project_text(
        " ".join(example.words), tokenizer, classifier, unk_token_id
    )
    content_ids = list(content_ids)
    projections = list(projections)
    if aligned.truncated:
        content_slots = sum(1 for value in aligned.special_tokens_mask if not value)
        content_ids = content_ids[:content_slots]
        projections = projections[:content_slots]
    if tuple(content_ids) != tuple(
        identifier
        for identifier, special in zip(aligned.input_ids, aligned.special_tokens_mask)
        if not special
    ):
        raise PhoNERContractViolation(
            "adapted base-grid tokenization does not match word-level label alignment"
        )
    encoded = build_example(aligned.input_ids, aligned.special_tokens_mask, projections)
    return TokenizedNERExample(
        sample_id=example.sample_id,
        input_ids=tuple(encoded.input_ids),
        attention_mask=tuple(1 for _ in encoded.input_ids),
        special_tokens_mask=tuple(encoded.special_tokens_mask),
        label_ids=aligned.label_ids,
        word_ids=aligned.word_ids,
        words=example.words,
        labels=example.labels,
        truncated=aligned.truncated,
        tone_ids=tuple(encoded.tone_ids),
        tone_mask=tuple(encoded.tone_mask),
        letter_ids=tuple(tuple(row) for row in encoded.letter_ids),
    )


def _pad_rows(examples: Sequence[TokenizedNERExample], *, pad_token_id: int) -> dict[str, Any]:
    if not examples:
        raise PhoNERContractViolation("cannot collate an empty NER batch")
    width = max(len(item.input_ids) for item in examples)
    depth = max(
        (len(row) for item in examples if item.letter_ids is not None for row in item.letter_ids),
        default=1,
    )
    depth = max(depth, 1)
    rows: dict[str, list[Any]] = {
        "input_ids": [],
        "attention_mask": [],
        "special_tokens_mask": [],
        "label_ids": [],
        "tone_ids": [],
        "tone_mask": [],
        "letter_ids": [],
        "letter_mask": [],
        "sample_ids": [item.sample_id for item in examples],
        "words": [list(item.words) for item in examples],
    }
    from unmark.modeling.contracts import LETTER_NA_SENTINEL, TONE_NA_SENTINEL

    for item in examples:
        pad = width - len(item.input_ids)
        rows["input_ids"].append(list(item.input_ids) + [pad_token_id] * pad)
        rows["attention_mask"].append(list(item.attention_mask) + [0] * pad)
        rows["special_tokens_mask"].append(list(item.special_tokens_mask) + [1] * pad)
        rows["label_ids"].append(list(item.label_ids) + [PHONER_IGNORE_INDEX] * pad)
        if item.tone_ids is not None and item.tone_mask is not None and item.letter_ids is not None:
            rows["tone_ids"].append(list(item.tone_ids) + [TONE_NA_SENTINEL] * pad)
            rows["tone_mask"].append(list(item.tone_mask) + [False] * pad)
            padded_letters = [list(row) for row in item.letter_ids] + [[] for _ in range(pad)]
            rows["letter_ids"].append(
                [row + [LETTER_NA_SENTINEL] * (depth - len(row)) for row in padded_letters]
            )
            rows["letter_mask"].append(
                [[True] * len(row) + [False] * (depth - len(row)) for row in padded_letters]
            )
    return rows


def collate_tokenized_ner_batch(
    examples: Sequence[TokenizedNERExample], *, pad_token_id: int
) -> dict[str, Any]:
    import torch

    rows = _pad_rows(examples, pad_token_id=pad_token_id)
    tensors: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "special_tokens_mask", "label_ids"):
        tensors[key] = torch.tensor(rows[key], dtype=torch.long)
    if rows["tone_ids"]:
        tensors["tone_ids"] = torch.tensor(rows["tone_ids"], dtype=torch.long)
        tensors["tone_mask"] = torch.tensor(rows["tone_mask"], dtype=torch.bool)
        tensors["letter_ids"] = torch.tensor(rows["letter_ids"], dtype=torch.long)
        tensors["letter_mask"] = torch.tensor(rows["letter_mask"], dtype=torch.bool)
    tensors["sample_ids"] = rows["sample_ids"]
    tensors["words"] = rows["words"]
    return tensors


class FrozenTokenProbe:
    """Factory for the shared linear token classifier.

    Returns a `torch.nn.Module` with exactly one `Linear(hidden_dim, num_labels)`
    classifier. The wrapper keeps this module importable without torch.
    """

    def __new__(cls, hidden_dim: int, num_labels: int) -> Any:
        import torch

        class _FrozenTokenProbe(torch.nn.Module):
            def __init__(self, hidden_dim: int, num_labels: int) -> None:
                super().__init__()
                self.classifier = torch.nn.Linear(hidden_dim, num_labels)
                self.architecture = f"Linear({hidden_dim}, {num_labels})"
                self.hidden_dim = hidden_dim
                self.num_labels = num_labels

            def forward(self, hidden_states: Any) -> Any:
                return self.classifier(hidden_states)

        _FrozenTokenProbe.__name__ = "FrozenTokenProbe"
        return _FrozenTokenProbe(hidden_dim, num_labels)


def require_frozen_module(module: Any, name: str) -> None:
    if getattr(module, "training", True):
        raise EvaluationContractViolation(f"{name} must be in eval mode")
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise EvaluationContractViolation(f"{name} parameter(s) require grad: {trainable[:5]}")


def freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def require_identical_probe_architecture(probes: Mapping[PhoNERPathway, Any]) -> None:
    shapes = {
        pathway: tuple(probe.classifier.weight.shape)
        for pathway, probe in probes.items()
    }
    if len(set(shapes.values())) != 1:
        raise PhoNERContractViolation(f"probe architectures differ across pathways: {shapes}")
    for pathway, probe in probes.items():
        children = list(probe.children())
        if len(children) != 1 or children[0] is not probe.classifier:
            raise PhoNERContractViolation(f"{pathway.value} probe is not a single Linear layer")


def bio_entities(labels: Sequence[str]) -> frozenset[tuple[int, int, str]]:
    """Convert BIO/IOB2 labels to exact `(start, end_exclusive, type)` spans.

    Compatibility policy: invalid predicted transitions are repaired the same
    way as common CoNLL/seqeval-compatible chunk extraction. A sentence-initial
    `I-X`, `O -> I-X`, or `B-Y/I-Y -> I-X` with `X != Y` starts a new `X`
    entity at that position. Exact span and entity type must match to count as a
    true positive.
    """

    entities: set[tuple[int, int, str]] = set()
    start: int | None = None
    entity_type: str | None = None

    def close(at: int) -> None:
        nonlocal start, entity_type
        if start is not None and entity_type is not None:
            entities.add((start, at, entity_type))
        start = None
        entity_type = None

    for index, label in enumerate(labels):
        if label == "O" or not label:
            close(index)
            continue
        if "-" not in label:
            close(index)
            start, entity_type = index, label
            continue
        prefix, kind = label.split("-", 1)
        if prefix == "B":
            close(index)
            start, entity_type = index, kind
        elif prefix == "I":
            if start is None or entity_type != kind:
                close(index)
                start, entity_type = index, kind
        else:
            close(index)
            start, entity_type = index, kind
    close(len(labels))
    return frozenset(entities)


@dataclass(frozen=True)
class EntityF1:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return 0.0 if denom == 0 else self.true_positive / denom

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return 0.0 if denom == 0 else self.true_positive / denom

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 0.0 if denom == 0.0 else 2 * self.precision * self.recall / denom

    @classmethod
    def from_sequences(
        cls, gold: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]
    ) -> "EntityF1":
        if len(gold) != len(predicted):
            raise PhoNERContractViolation("gold/predicted sentence count mismatch")
        tp = fp = fn = 0
        for gold_seq, pred_seq in zip(gold, predicted):
            if len(gold_seq) != len(pred_seq):
                raise PhoNERContractViolation("gold/predicted label length mismatch")
            g = bio_entities(gold_seq)
            p = bio_entities(pred_seq)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        return cls(tp, fp, fn)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "entity_true_positive": self.true_positive,
            "entity_false_positive": self.false_positive,
            "entity_false_negative": self.false_negative,
            "entity_precision": self.precision,
            "entity_recall": self.recall,
            "entity_micro_f1": self.f1,
        }


def corruption_summary_metrics(condition_f1: Mapping[str, float]) -> dict[str, float]:
    missing = [condition for condition in SIX_CONDITIONS if condition not in condition_f1]
    if missing:
        raise PhoNERContractViolation(f"missing condition F1 values for {missing}")
    full = float(condition_f1["FULL"])
    corrupt_conditions = [c for c in SIX_CONDITIONS if c != "FULL"]
    summary = {
        f"{condition}_f1": float(condition_f1[condition])
        for condition in SIX_CONDITIONS
    }
    summary["corrupt_avg_f1"] = sum(condition_f1[c] for c in corrupt_conditions) / len(corrupt_conditions)
    summary["all_6_f1"] = sum(condition_f1[c] for c in SIX_CONDITIONS) / len(SIX_CONDITIONS)
    summary["full_to_strip_absolute_drop"] = full - float(condition_f1["STRIP_ALL"])
    for condition in SIX_CONDITIONS:
        summary[f"{condition}_robustness_retention"] = (
            math.nan if full == 0.0 else float(condition_f1[condition]) / full
        )
    return summary


def mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise PhoNERContractViolation("cannot summarize no values")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize_five_seed_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-seed entity metrics without ensembling logits or picking a seed."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    by_seed: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        pathway = str(row["pathway"])
        seed = int(row["seed"])
        condition = str(row["condition"])
        grouped.setdefault((pathway, condition), []).append(row)
        by_seed.setdefault((pathway, seed), {})[condition] = float(row["entity_micro_f1"])

    condition_summary: dict[str, Any] = {}
    for (pathway, condition), group in sorted(grouped.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in group))
        if seeds != tuple(sorted(PHONER_FINAL_SEEDS)):
            raise PhoNERContractViolation(
                f"{pathway}/{condition} does not cover the frozen five seeds: {seeds}"
            )
        f1_mean, f1_sd = mean_and_sample_sd([float(row["entity_micro_f1"]) for row in group])
        p_mean, p_sd = mean_and_sample_sd([float(row["entity_precision"]) for row in group])
        r_mean, r_sd = mean_and_sample_sd([float(row["entity_recall"]) for row in group])
        condition_summary[f"{pathway}:{condition}"] = {
            "pathway": pathway,
            "condition": condition,
            "seeds": list(seeds),
            "entity_micro_f1_mean": f1_mean,
            "entity_micro_f1_sample_sd": f1_sd,
            "entity_precision_mean": p_mean,
            "entity_precision_sample_sd": p_sd,
            "entity_recall_mean": r_mean,
            "entity_recall_sample_sd": r_sd,
            "per_seed": [dict(row) for row in sorted(group, key=lambda r: int(r["seed"]))],
        }

    robustness_by_seed = []
    for (pathway, seed), values in sorted(by_seed.items()):
        robustness_by_seed.append({"pathway": pathway, "seed": seed, **corruption_summary_metrics(values)})
    robustness_grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in robustness_by_seed:
        robustness_grouped.setdefault(str(row["pathway"]), []).append(row)
    robustness_summary = {}
    for pathway, group in sorted(robustness_grouped.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in group))
        if seeds != tuple(sorted(PHONER_FINAL_SEEDS)):
            raise PhoNERContractViolation(
                f"{pathway} robustness summary does not cover five seeds: {seeds}"
            )
        metrics = {}
        for key in ("corrupt_avg_f1", "all_6_f1", "full_to_strip_absolute_drop"):
            mean, sd = mean_and_sample_sd([float(row[key]) for row in group])
            metrics[f"{key}_mean"] = mean
            metrics[f"{key}_sample_sd"] = sd
        robustness_summary[pathway] = {
            "seeds": list(seeds),
            **metrics,
            "per_seed": [dict(row) for row in sorted(group, key=lambda r: int(r["seed"]))],
        }
    return {
        "primary_policy": (
            "mean and sample standard deviation across five independently trained "
            "token probes; no logit ensembling and no best-seed selection"
        ),
        "conditions": condition_summary,
        "robustness": robustness_summary,
    }


def ids_to_word_predictions(
    predicted_ids: Sequence[int],
    word_ids: Sequence[int | None],
    id_to_label: Mapping[int, str],
    *,
    num_words: int,
) -> tuple[str, ...]:
    labels = ["O"] * num_words
    seen: set[int] = set()
    for pred, word_id in zip(predicted_ids, word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        labels[word_id] = id_to_label[int(pred)]
    return tuple(labels)


def protocol_digest(config: PhoNERCrossTaskConfig, dataset_identity: PhoNERDatasetIdentity | None = None) -> str:
    payload = {"config": config.to_dict()}
    if dataset_identity is not None:
        payload["dataset_identity"] = dataset_identity.to_dict()
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "FINAL_SENTIMENT_FUSION_HEADS",
    "MM_CAT_POOLING_USED",
    "NEW_CROSS_TASK_PROBE_PROTOCOL_SOURCE",
    "PHONER_CORRUPTION_SEED",
    "PHONER_CROSS_TASK_PROTOCOL_VERSION",
    "PHONER_DATASET_TERMS",
    "PHONER_EXPERIMENT_NAME",
    "PHONER_FINAL_SEEDS",
    "PHONER_GATE_SHA256",
    "PHONER_IGNORE_INDEX",
    "PHONER_REPRESENTATION",
    "PHONER_SCALE_SHA256",
    "PHONER_TASK_ID",
    "PROBE_ARCHITECTURE",
    "SENTIMENT_CALIBRATION_USED",
    "SENTIMENT_MLP_USED",
    "UIT_VSFC_CLASS_WEIGHTS_USED",
    "CORRUPTION_PARITY_CONCLUSION",
    "EntityF1",
    "FrozenTokenProbe",
    "FrozenTokenProbeTrainingPolicy",
    "PhoNERWordChunk",
    "PhoNERCrossTaskConfig",
    "PhoNERContractViolation",
    "PhoNERDatasetIdentity",
    "PhoNERExample",
    "PhoNERPathway",
    "PhoNERSplit",
    "PhoNERTestSealViolation",
    "TokenizedNERExample",
    "align_word_labels",
    "audit_condition_invariant_coverage",
    "audit_train_dev_condition_invariant_coverage",
    "bio_entities",
    "build_phoner_dataset_identity",
    "canonical_stage",
    "collate_tokenized_ner_batch",
    "corrupt_phoner_example",
    "corruption_summary_metrics",
    "cross_pathway_corrupted_inputs",
    "encode_adapted_example",
    "encode_native_example",
    "find_phoner_split_file",
    "freeze_module",
    "ids_to_word_predictions",
    "label_inventory_from_train",
    "materialize_condition_invariant_chunks",
    "mean_and_sample_sd",
    "parse_phoner_split",
    "protocol_digest",
    "require_checkpoint_sha256",
    "require_frozen_module",
    "require_identical_probe_architecture",
    "require_phobert_identity",
    "require_stage_access",
    "sha256_file",
    "summarize_five_seed_scores",
]
