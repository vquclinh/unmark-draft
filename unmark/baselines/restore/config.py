"""Frozen protocol constants for the RESTORE Stage-2 baseline.

This module is deliberately torch-free and does not import any UNMARK adapter or
Stage-1 module. It pins the scientific identities the runner must enforce before
any real RESTORE execution can produce evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    BATCH_SIZE,
    DERIVED_TRAIN_CSV_SHA256,
    DERIVED_TRAIN_SIZE,
    DERIVED_VALIDATION_CSV_SHA256,
    EPOCHS,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_NUM_LABELS,
    PRIMARY_TASK,
    PUBLISHED_LABEL_COUNTS,
    PUBLISHED_SPLIT_SIZES,
    TRUNCATION,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
)

RESTORE_PROTOCOL_VERSION = "restore-baseline-protocol-v1"
RESTORE_BASELINE_SCHEMA_VERSION = "restore-stage2-baseline-v1"
RESTORE_CANONICAL_INPUT_SEMANTICS = (
    "source CSV text -> unmark.orthography.canon(text) -> condition corruption if degraded"
)

RESTORE_MODEL_ID = "nrl-ai/vn-diacritic-vit5-base"
RESTORE_MODEL_REVISION = "7ec0710193721ac3321b3bb3741ec92d8b43cad3"
RESTORE_MODEL_LICENSE = "apache-2.0"
RESTORE_MODEL_ARCHITECTURE = "T5ForConditionalGeneration"
RESTORE_MODEL_PARAMETER_COUNT = 225_950_976
RESTORE_MODEL_CONFIG_SHA256 = (
    "ffd95f81f1665bd7dba68bccb1cc88f10787f9abe355af8a373f2d0f2eac5430"
)
RESTORE_MODEL_GENERATION_CONFIG_SHA256 = (
    "eaaf16b98e47131e3ce155054140429e36c4ed6a23c6ff32c2946ef66ce121ff"
)
RESTORE_TOKENIZER_ARTIFACT_SHA256S: dict[str, str] = {
    "special_tokens_map.json": (
        "1bf246679a41f9204ef9085d6ebe314cb3ec1c916db54e2a3be6e49c9c839453"
    ),
    "spiece.model": (
        "59986b62f9f0b90edafb9b073ea7b93d21114a5841219a1ea2399ade73f729c6"
    ),
    "tokenizer.json": (
        "637fb80d3a85ec307efea089e10a42e1fa22fdbd19abe612e38cb771b78de8fd"
    ),
    "tokenizer_config.json": (
        "10fe5aebb65886903db4ddb439d2192e28fc44db7cd87f988377a5d3f913cb97"
    ),
}
RESTORE_MODEL_WEIGHT_SHA256 = (
    "0dc3361b608903048dfd00fd8dcb3d6174a6cfd4025394504c722ca2487bf52d"
)
RESTORE_MODEL_WEIGHT_FILENAME = "model.safetensors"
RESTORE_RUNTIME_TRANSFORMERS_VERSION = "4.57.6"

RESTORE_RESTORER_BATCH_SIZE = 8
"""Conservative operational default for Colab L4; recorded, never tuned on scores."""

RESTORE_PUBLIC_SMOKE_EXAMPLES: tuple[str, ...] = (
    "Toi yeu Viet Nam",
    "Hop dong nay duoc lap ngay 14/3/2025",
    "Sinh vien khong hai long voi thai do phuc vu.",
    "Nha truong can cai thien chat luong phong hoc.",
)


@dataclass(frozen=True)
class RestoreTokenizerContract:
    """Exact RESTORE tokenizer encode/decode contract.

    The RESTORE wrapper does not rely on Transformers tokenizer defaults for any
    output-affecting behavior. These values preserve the intended direct
    AutoTokenizer/T5 tokenizer behavior while making every version-sensitive
    knob visible in runtime provenance.
    """

    use_fast: bool = True
    legacy: bool = True
    input_return_tensors: str = "pt"
    input_padding: bool = True
    input_truncation: bool = True
    input_max_length: int = 256
    input_add_special_tokens: bool = True
    input_return_attention_mask: bool = True
    decode_skip_special_tokens: bool = True
    decode_clean_up_tokenization_spaces: bool = False
    transformers_version: str = RESTORE_RUNTIME_TRANSFORMERS_VERSION

    def __post_init__(self) -> None:
        checks = {
            "use_fast": self.use_fast is True,
            "legacy": self.legacy is True,
            "input_return_tensors": self.input_return_tensors == "pt",
            "input_padding": self.input_padding is True,
            "input_truncation": self.input_truncation is True,
            "input_max_length": self.input_max_length == 256,
            "input_add_special_tokens": self.input_add_special_tokens is True,
            "input_return_attention_mask": self.input_return_attention_mask is True,
            "decode_skip_special_tokens": self.decode_skip_special_tokens is True,
            "decode_clean_up_tokenization_spaces": (
                self.decode_clean_up_tokenization_spaces is False
            ),
            "transformers_version": (
                self.transformers_version == RESTORE_RUNTIME_TRANSFORMERS_VERSION
            ),
        }
        drifted = [name for name, ok in checks.items() if not ok]
        if drifted:
            raise EvaluationContractViolation(
                f"RESTORE tokenizer contract drifted on {drifted}"
            )

    def to_load_kwargs(self) -> dict[str, Any]:
        return {
            "use_fast": self.use_fast,
            "legacy": self.legacy,
        }

    def to_encode_kwargs(self) -> dict[str, Any]:
        return {
            "return_tensors": self.input_return_tensors,
            "padding": self.input_padding,
            "truncation": self.input_truncation,
            "max_length": self.input_max_length,
            "add_special_tokens": self.input_add_special_tokens,
            "return_attention_mask": self.input_return_attention_mask,
        }

    def to_decode_kwargs(self) -> dict[str, Any]:
        return {
            "skip_special_tokens": self.decode_skip_special_tokens,
            "clean_up_tokenization_spaces": self.decode_clean_up_tokenization_spaces,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "load": self.to_load_kwargs(),
            "input": self.to_encode_kwargs(),
            "decode": self.to_decode_kwargs(),
            "transformers_version": self.transformers_version,
            "version_sensitive_defaults_overridden": True,
        }


RESTORE_TOKENIZER_CONTRACT = RestoreTokenizerContract()


@dataclass(frozen=True)
class RestoreGenerationConfig:
    """The deterministic direct-use generation contract for RESTORE."""

    max_length: int = 256
    do_sample: bool = False
    num_beams: int = 1
    temperature: None = None
    top_k: None = None
    top_p: None = None

    def __post_init__(self) -> None:
        if self.max_length != 256:
            raise EvaluationContractViolation(
                f"RESTORE max_length is frozen to 256, got {self.max_length!r}"
            )
        if self.do_sample is not False:
            raise EvaluationContractViolation("RESTORE generation must not sample")
        if self.num_beams != 1:
            raise EvaluationContractViolation(
                f"RESTORE generation is greedy num_beams=1, got {self.num_beams!r}"
            )
        if self.temperature is not None or self.top_k is not None or self.top_p is not None:
            raise EvaluationContractViolation(
                "RESTORE forbids temperature/top-k/top-p sampling controls"
            )

    def to_generate_kwargs(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "decoder": "greedy",
        }


RESTORE_GENERATION_CONFIG = RestoreGenerationConfig()
RESTORE_HF_GENERATION_CONFIG_FIELDS: dict[str, Any] = {
    "_from_model_config": True,
    "decoder_start_token_id": 0,
    "eos_token_id": [1],
    "pad_token_id": 0,
    "transformers_version": "4.57.6",
}
RESTORE_HF_MODEL_CONFIG_GENERATION_FIELDS: dict[str, Any] = {
    "decoder_start_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "use_cache": True,
}

RESTORE_USES_UNMARK_ADAPTER = False
RESTORE_USES_BASE_TONE_LETTER_CHANNELS = False
RESTORE_USES_UNMARK_STAGE1_CHECKPOINT = False
RESTORE_TRAINS_RESTORER = False
RESTORE_RESTORER_FROZEN = True
RESTORE_CONDITION_AWARE_ROUTING = False
RESTORE_FULL_BYPASS = False
RESTORE_ONE_RESTORE_PATHWAY = True
RESTORE_CORRUPTED_LABEL_TRAINING = False
RESTORE_BEST_SEED_SELECTION = False
RESTORE_OFFICIAL_VALIDATION_PREVIOUSLY_SEEN_BY_AUTHORS = True
RESTORE_OFFICIAL_TEST_READ = False
RESTORE_HARD_STOP = True

RESTORE_PHOBERT_CHECKPOINT = ENCODER_CHECKPOINT
RESTORE_PHOBERT_REVISION = ENCODER_REVISION
RESTORE_PHOBERT_FROZEN = True
RESTORE_PHOBERT_EVAL_MODE = True
RESTORE_PHOBERT_DTYPE = "torch.float32"
RESTORE_PHOBERT_POOLING = "FIRST_TOKEN"
RESTORE_PHOBERT_HIDDEN_SIZE = 768

RESTORE_STAGE2_MAX_LENGTH = MAX_LENGTH
RESTORE_STAGE2_TRUNCATION = TRUNCATION
RESTORE_STAGE2_PADDING = PADDING
RESTORE_STAGE2_HEAD_SEEDS: tuple[int, ...] = tuple(MEASUREMENT_SEEDS)
RESTORE_STAGE2_HEAD_LR = 0.01
RESTORE_STAGE2_HEAD_BATCH_SIZE = BATCH_SIZE
RESTORE_STAGE2_HEAD_EPOCHS = EPOCHS
RESTORE_STAGE2_HEAD_EARLY_STOPPING = False
RESTORE_STAGE2_HEAD_ARCHITECTURE = "Linear(768, 3, bias=True)"

RESTORE_CONDITIONS: tuple[str, ...] = (
    "FULL",
    "P25",
    "P50",
    "P75",
    "P100",
    "STRIP_ALL",
)
RESTORE_DEGRADED_CONDITIONS: tuple[str, ...] = tuple(
    c for c in RESTORE_CONDITIONS if c != "FULL"
)
RESTORE_CORRUPTION_SEED = 19225
RESTORE_EXPECTED_CHANGED_ROW_COUNTS: dict[str, int] = {
    "FULL": 0,
    "P25": 1268,
    "P50": 1513,
    "P75": 1564,
    "P100": 1576,
    "STRIP_ALL": 1579,
}

RESTORE_DERIVED_TRAIN_SHA256 = DERIVED_TRAIN_CSV_SHA256
RESTORE_DERIVED_TRAIN_ROWS = DERIVED_TRAIN_SIZE
RESTORE_PROTOCOL_TRAIN_ROWS = 9139
RESTORE_PROTOCOL_DEV_ROWS = 2285
RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST = (
    "2cad022dd4aabbc875e388030e453ba2479f1046bea884309398b79f7d878cbd"
)
RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST = (
    "1c1377b2cd8c8016765229079fb469c2ead5c6ec85229d986f4344fa817f6128"
)
RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST = (
    "63192edf811f8249404597103dc5ba4f48bb7843d242a592d13764013c095860"
)
RESTORE_PROTOCOL_DEV_LABEL_DIGEST = (
    "546817741453edcb968c4d5de3530e435c5de9374995c1665260e92aea07ca1c"
)

RESTORE_VALIDATION_SHA256 = DERIVED_VALIDATION_CSV_SHA256
RESTORE_VALIDATION_ROWS = PUBLISHED_SPLIT_SIZES["validation"]
RESTORE_VALIDATION_CLASS_COUNTS = {0: 705, 1: 73, 2: 805}
RESTORE_VALIDATION_ORDERED_ID_DIGEST = (
    "825714672bcf27543825b85ac72c3fea2875647f55a117c41cc42ecc8e305d1a"
)
RESTORE_VALIDATION_LABEL_DIGEST = (
    "70d5ed20e31dea2cacddc7cdb737ef3d0d1f980699b7e3c6d6aff0806711852b"
)

RESTORE_PHASES: tuple[str, ...] = (
    "RESTORE_MODEL_PREFLIGHT",
    "RESTORE_CLEAN_TEXT",
    "RESTORE_CLEAN_REPRESENTATIONS",
    "RESTORE_FIVE_HEADS",
    "VALIDATION_IDENTITY_CHECK",
    "RESTORE_VALIDATION_TEXT",
    "RESTORE_VALIDATION_REPRESENTATIONS",
    "RESTORE_MEASUREMENT",
    "RESTORE_DIAGNOSTICS",
    "RESTORE_GRR",
    "RESTORE_FINAL_CLOSEOUT",
)

RESTORE_FORBIDDEN_ARTIFACT_NAMESPACES: tuple[str, ...] = (
    "stage2-measurement",
    "stage2-training",
    "stage2-vanilla-anchor",
)

RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE = (
    "stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/"
    "evidence/vanilla-upper-floor-final-v1.json"
)
RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256 = (
    "d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862"
)
RESTORE_UNMARK_A_EVIDENCE_RELATIVE = (
    "stage2-measurement/cb78114eb0fa/audit061-8fa991e5/"
    "corrected-official-validation-v3/stage2-corrected-measurement-v3-final.json"
)
RESTORE_UNMARK_A_EVIDENCE_SHA256 = (
    "8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2"
)

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHORT_SHA_RE = re.compile(r"^[0-9a-f]{7,39}$")


def require_full_commit_sha(value: str, *, name: str = "commit sha") -> str:
    if not isinstance(value, str):
        raise EvaluationContractViolation(f"{name} must be a full 40-character SHA")
    lowered = value.strip().lower()
    if _FULL_SHA_RE.fullmatch(lowered):
        return lowered
    if _SHORT_SHA_RE.fullmatch(lowered):
        raise EvaluationContractViolation(
            f"{name} must be a full 40-character SHA, not short {value!r}"
        )
    raise EvaluationContractViolation(f"{name} must be a full lowercase hex SHA, got {value!r}")


def require_immutable_model_revision(revision: str) -> str:
    """Refuse branch names and short SHAs for the external RESTORE model."""

    if revision in {"main", "latest", "master"}:
        raise EvaluationContractViolation(
            f"RESTORE model revision must be immutable, got branch name {revision!r}"
        )
    return require_full_commit_sha(revision, name="RESTORE model revision")


def require_restore_model_identity(model_id: str, revision: str) -> None:
    if model_id != RESTORE_MODEL_ID:
        raise EvaluationContractViolation(
            f"RESTORE model id is frozen to {RESTORE_MODEL_ID!r}, got {model_id!r}"
        )
    resolved = require_immutable_model_revision(revision)
    if resolved != RESTORE_MODEL_REVISION:
        raise EvaluationContractViolation(
            f"RESTORE model revision is frozen to {RESTORE_MODEL_REVISION}, got {resolved}"
        )


def require_generation_config(config: RestoreGenerationConfig) -> None:
    if config != RESTORE_GENERATION_CONFIG:
        raise EvaluationContractViolation(
            "RESTORE generation config must equal the frozen greedy direct-use config"
        )


def require_tokenizer_contract(config: RestoreTokenizerContract) -> None:
    if config != RESTORE_TOKENIZER_CONTRACT:
        raise EvaluationContractViolation(
            "RESTORE tokenizer contract must equal the frozen direct-use contract"
        )


def require_restore_conditions(conditions: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    actual = tuple(conditions)
    if actual != RESTORE_CONDITIONS:
        raise EvaluationContractViolation(
            f"RESTORE conditions must be exactly {list(RESTORE_CONDITIONS)}, got {list(actual)}"
        )
    return actual


def require_restore_head_seeds(seeds: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    actual = tuple(int(seed) for seed in seeds)
    if actual != RESTORE_STAGE2_HEAD_SEEDS:
        raise EvaluationContractViolation(
            f"RESTORE head seeds must be exactly {list(RESTORE_STAGE2_HEAD_SEEDS)}, "
            f"got {list(actual)}"
        )
    return actual


def require_validation_class_counts(counts: Mapping[int, int]) -> None:
    expected = dict(RESTORE_VALIDATION_CLASS_COUNTS)
    if dict(counts) != expected:
        raise EvaluationContractViolation(
            f"official validation class counts must be {expected}, got {dict(counts)}"
        )


def restore_artifact_root(drive_root: str | Path, execution_head: str) -> Path:
    """Dedicated durable namespace for RESTORE artifacts."""

    head = require_full_commit_sha(execution_head, name="execution head")
    return (
        Path(drive_root)
        / "stage2-baselines"
        / "restore"
        / head[:12]
        / "audit064-restore-v1"
    )


def restore_runtime_cache_root(drive_root: str | Path) -> Path:
    return Path(drive_root) / "runtime-input-cache" / "baselines" / "restore"


def require_restore_namespace(path: str | Path, drive_root: str | Path) -> None:
    """Lexically require RESTORE outputs under the baseline namespace."""

    root = Path(drive_root).absolute()
    target = Path(path).absolute()
    restore_root = root / "stage2-baselines" / "restore"
    try:
        target.relative_to(restore_root)
    except ValueError as error:
        raise EvaluationContractViolation(
            f"RESTORE artifact path {target} is outside {restore_root}"
        ) from error
    parts = set(target.parts)
    forbidden = sorted(parts & set(RESTORE_FORBIDDEN_ARTIFACT_NAMESPACES))
    if forbidden:
        raise EvaluationContractViolation(
            f"RESTORE artifact path collides with historical namespace(s): {forbidden}"
        )


def preg1_identity_expectations() -> dict[str, Any]:
    return {
        "derived_train_sha256": RESTORE_DERIVED_TRAIN_SHA256,
        "derived_train_rows": RESTORE_DERIVED_TRAIN_ROWS,
        "protocol_train": {
            "rows": RESTORE_PROTOCOL_TRAIN_ROWS,
            "ordered_id_digest": RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST,
            "label_digest": RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST,
        },
        "protocol_dev": {
            "rows": RESTORE_PROTOCOL_DEV_ROWS,
            "ordered_id_digest": RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST,
            "label_digest": RESTORE_PROTOCOL_DEV_LABEL_DIGEST,
        },
    }


def validation_identity_expectations() -> dict[str, Any]:
    return {
        "raw_sha256": RESTORE_VALIDATION_SHA256,
        "rows": RESTORE_VALIDATION_ROWS,
        "class_counts": dict(RESTORE_VALIDATION_CLASS_COUNTS),
        "ordered_id_digest": RESTORE_VALIDATION_ORDERED_ID_DIGEST,
        "label_digest": RESTORE_VALIDATION_LABEL_DIGEST,
    }


def protocol_dataset_block() -> dict[str, Any]:
    return {
        "dataset": PRIMARY_DATASET,
        "dataset_version": PRIMARY_DATASET_VERSION,
        "task": PRIMARY_TASK,
        "num_labels": PRIMARY_NUM_LABELS,
        "published_validation_label_counts": dict(PUBLISHED_LABEL_COUNTS["validation"]),
    }
