"""RESTORE-specific immutable artifact and resume helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from unmark.baselines.restore.config import (
    RESTORE_BASELINE_SCHEMA_VERSION,
    RESTORE_GENERATION_CONFIG,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_PHOBERT_CHECKPOINT,
    RESTORE_PHOBERT_DTYPE,
    RESTORE_PHOBERT_HIDDEN_SIZE,
    RESTORE_PHOBERT_POOLING,
    RESTORE_PHOBERT_REVISION,
    RESTORE_PROTOCOL_VERSION,
    RESTORE_RESTORER_BATCH_SIZE,
    RESTORE_STAGE2_MAX_LENGTH,
    RESTORE_STAGE2_PADDING,
    RESTORE_STAGE2_TRUNCATION,
    RESTORE_TOKENIZER_CONTRACT,
    RESTORE_TOKENIZER_ARTIFACT_SHA256S,
)
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_head import Preg1Role, ordered_id_digest


def canonical_json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_bytes(data)
    temp.replace(target)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationContractViolation(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationContractViolation(f"{path} is not a JSON object")
    return payload


def semantic_text_digest(texts: Sequence[str]) -> str:
    """Order-sensitive semantic digest over text strings, without raw text leakage."""

    for text in texts:
        if not isinstance(text, str):
            raise EvaluationContractViolation("text digest input must be strings")
    return sha256_bytes(canonical_json_bytes(list(texts)))


def label_digest(labels: Sequence[int]) -> str:
    if not labels:
        raise EvaluationContractViolation("cannot digest an empty label vector")
    return hashlib.sha256(
        "\n".join(str(int(v)) for v in labels).encode("utf-8")
    ).hexdigest()


def require_ids_preserved(records: Sequence["RestoreTextRecord"], expected_ids: Sequence[str]) -> None:
    actual = [record.sample_id for record in records]
    if actual != list(expected_ids):
        raise EvaluationContractViolation(
            "RESTORE text cache row order or sample ids changed"
        )


@dataclass(frozen=True)
class RestoreTextRecord:
    sample_id: str
    input_text: str
    restored_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise EvaluationContractViolation("RESTORE text record has empty sample_id")
        if not isinstance(self.input_text, str) or not isinstance(self.restored_text, str):
            raise EvaluationContractViolation("RESTORE text record values must be strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "input_text": self.input_text,
            "restored_text": self.restored_text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RestoreTextRecord":
        try:
            return cls(
                sample_id=str(payload["sample_id"]),
                input_text=payload["input_text"],
                restored_text=payload["restored_text"],
            )
        except KeyError as error:
            raise EvaluationContractViolation(
                f"RESTORE text record is missing {error.args[0]!r}"
            ) from error


@dataclass(frozen=True)
class RestoreTextCacheRequest:
    """Input identity for one RESTORE text cache before outputs are known."""

    repository_head: str
    role: str
    condition: str
    corruption_seed: int | None
    input_row_count: int
    ordered_id_digest: str
    input_text_digest: str
    restore_batch_size: int = RESTORE_RESTORER_BATCH_SIZE
    generation_config: Mapping[str, Any] = field(
        default_factory=lambda: RESTORE_GENERATION_CONFIG.to_dict()
    )
    tokenizer_contract: Mapping[str, Any] = field(
        default_factory=lambda: RESTORE_TOKENIZER_CONTRACT.to_dict()
    )
    tokenizer_artifact_sha256s: Mapping[str, str] = field(
        default_factory=lambda: dict(RESTORE_TOKENIZER_ARTIFACT_SHA256S)
    )
    schema_version: str = RESTORE_BASELINE_SCHEMA_VERSION
    protocol_version: str = RESTORE_PROTOCOL_VERSION
    restore_model_id: str = RESTORE_MODEL_ID
    restore_model_revision: str = RESTORE_MODEL_REVISION

    def __post_init__(self) -> None:
        Preg1Role(self.role)
        if self.input_row_count <= 0:
            raise EvaluationContractViolation("RESTORE text cache row count must be positive")
        for name in ("repository_head", "condition", "ordered_id_digest", "input_text_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvaluationContractViolation(f"{name} must be a non-empty string")
        if self.restore_model_id != RESTORE_MODEL_ID:
            raise EvaluationContractViolation("RESTORE text cache binds the wrong model id")
        if self.restore_model_revision != RESTORE_MODEL_REVISION:
            raise EvaluationContractViolation("RESTORE text cache binds the wrong model revision")
        if dict(self.generation_config) != RESTORE_GENERATION_CONFIG.to_dict():
            raise EvaluationContractViolation("RESTORE text cache generation config drifted")
        if dict(self.tokenizer_contract) != RESTORE_TOKENIZER_CONTRACT.to_dict():
            raise EvaluationContractViolation("RESTORE text cache tokenizer contract drifted")
        if dict(self.tokenizer_artifact_sha256s) != dict(RESTORE_TOKENIZER_ARTIFACT_SHA256S):
            raise EvaluationContractViolation("RESTORE text cache tokenizer artifact drifted")
        if (
            isinstance(self.restore_batch_size, bool)
            or not isinstance(self.restore_batch_size, int)
            or self.restore_batch_size <= 0
        ):
            raise EvaluationContractViolation("RESTORE text cache batch size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "repository_head": self.repository_head,
            "role": self.role,
            "condition": self.condition,
            "corruption_seed": self.corruption_seed,
            "input_row_count": self.input_row_count,
            "ordered_id_digest": self.ordered_id_digest,
            "input_text_digest": self.input_text_digest,
            "restore_batch_size": self.restore_batch_size,
            "restore_model_id": self.restore_model_id,
            "restore_model_revision": self.restore_model_revision,
            "generation_config": dict(self.generation_config),
            "tokenizer_contract": dict(self.tokenizer_contract),
            "tokenizer_artifact_sha256s": dict(self.tokenizer_artifact_sha256s),
        }

    def require_compatible_manifest(self, manifest: Mapping[str, Any]) -> None:
        expected = self.to_dict()
        actual = manifest.get("request")
        if not isinstance(actual, Mapping):
            raise EvaluationContractViolation("RESTORE text manifest has no request identity")
        differences = [
            name for name in expected if expected[name] != actual.get(name)
        ]
        if differences:
            detail = ", ".join(
                f"{name}: cached={actual.get(name)!r} wanted={expected[name]!r}"
                for name in differences
            )
            raise EvaluationContractViolation(
                f"RESTORE text cache is incompatible: {detail}"
            )


class RestoreTextCache:
    """Immutable JSONL text cache plus machine-readable manifest."""

    JSONL_NAME = "restored-text.jsonl"
    MANIFEST_NAME = "restored-text-manifest.json"
    IN_PROGRESS_NAME = "RESTORE_TEXT.inprogress"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def jsonl_path(self) -> Path:
        return self.directory / self.JSONL_NAME

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST_NAME

    @property
    def in_progress_path(self) -> Path:
        return self.directory / self.IN_PROGRESS_NAME

    def exists(self) -> bool:
        return self.jsonl_path.is_file() and self.manifest_path.is_file()

    def _require_no_incomplete_transaction(self) -> None:
        if self.in_progress_path.exists():
            raise EvaluationContractViolation(
                f"{self.directory} contains {self.IN_PROGRESS_NAME}; inspect the incomplete "
                "RESTORE text transaction before reusing this cache"
            )
        if self.jsonl_path.exists() != self.manifest_path.exists():
            raise EvaluationContractViolation(
                f"{self.directory} contains a partial RESTORE text cache"
            )

    def read_manifest(self) -> dict[str, Any]:
        self._require_no_incomplete_transaction()
        if not self.manifest_path.is_file():
            raise EvaluationContractViolation(f"RESTORE text manifest missing at {self.manifest_path}")
        return read_json(self.manifest_path)

    def load(self, request: RestoreTextCacheRequest) -> tuple[RestoreTextRecord, ...]:
        manifest = self.read_manifest()
        request.require_compatible_manifest(manifest)
        records = []
        for line_number, line in enumerate(self.jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationContractViolation(
                    f"RESTORE text JSONL line {line_number} is malformed: {error}"
                ) from error
            records.append(RestoreTextRecord.from_dict(payload))
        if len(records) != request.input_row_count:
            raise EvaluationContractViolation(
                f"RESTORE text cache has {len(records)} rows, expected {request.input_row_count}"
            )
        if ordered_id_digest([r.sample_id for r in records]) != request.ordered_id_digest:
            raise EvaluationContractViolation("RESTORE text cache ordered-id digest drifted")
        if semantic_text_digest([r.input_text for r in records]) != request.input_text_digest:
            raise EvaluationContractViolation("RESTORE text cache input-text digest drifted")
        output_digest = semantic_text_digest([r.restored_text for r in records])
        if manifest.get("output_text_digest") != output_digest:
            raise EvaluationContractViolation("RESTORE text cache output-text digest drifted")
        if file_sha256(self.jsonl_path) != manifest.get("jsonl_sha256"):
            raise EvaluationContractViolation("RESTORE text JSONL raw SHA256 drifted")
        return tuple(records)

    def save(
        self,
        request: RestoreTextCacheRequest,
        records: Sequence[RestoreTextRecord],
        *,
        extra_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[RestoreTextRecord, ...]:
        if self.exists():
            return self.load(request)
        self._require_no_incomplete_transaction()
        if len(records) != request.input_row_count:
            raise EvaluationContractViolation(
                f"RESTORE text cache expected {request.input_row_count} rows, got {len(records)}"
            )
        if ordered_id_digest([r.sample_id for r in records]) != request.ordered_id_digest:
            raise EvaluationContractViolation("RESTORE text cache would change row order or ids")
        if semantic_text_digest([r.input_text for r in records]) != request.input_text_digest:
            raise EvaluationContractViolation("RESTORE text cache input digest mismatch")

        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            self.in_progress_path,
            {"request": request.to_dict(), "resume_contract": "write_once_visible_transaction"},
        )
        body = "".join(
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        atomic_write_bytes(self.jsonl_path, body.encode("utf-8"))
        manifest = {
            "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
            "protocol_version": RESTORE_PROTOCOL_VERSION,
            "request": request.to_dict(),
            "row_count": len(records),
            "ordered_id_digest": ordered_id_digest([r.sample_id for r in records]),
            "input_text_digest": request.input_text_digest,
            "output_text_digest": semantic_text_digest([r.restored_text for r in records]),
            "jsonl_sha256": file_sha256(self.jsonl_path),
            "condition_aware_routing": False,
            "full_bypass": False,
            "one_restore_pathway": True,
        }
        if extra_manifest:
            manifest["extra"] = dict(extra_manifest)
        write_json(self.manifest_path, manifest)
        self.in_progress_path.unlink()
        return self.load(request)


@dataclass(frozen=True)
class RestoreRepresentationKey:
    """Provenance for one RESTORE PhoBERT representation tensor."""

    repository_head: str
    role: str
    condition: str
    corruption_seed: int | None
    text_cache_manifest_sha256: str
    text_output_digest: str
    ordered_id_digest: str
    label_digest: str
    count: int
    schema_version: str = RESTORE_BASELINE_SCHEMA_VERSION
    protocol_version: str = RESTORE_PROTOCOL_VERSION
    representation_namespace: str = "RESTORE"
    restore_model_id: str = RESTORE_MODEL_ID
    restore_model_revision: str = RESTORE_MODEL_REVISION
    phobert_checkpoint: str = RESTORE_PHOBERT_CHECKPOINT
    phobert_revision: str = RESTORE_PHOBERT_REVISION
    max_length: int = RESTORE_STAGE2_MAX_LENGTH
    truncation: bool = RESTORE_STAGE2_TRUNCATION
    padding: str = RESTORE_STAGE2_PADDING
    pooling: str = RESTORE_PHOBERT_POOLING
    dtype: str = RESTORE_PHOBERT_DTYPE
    hidden_size: int = RESTORE_PHOBERT_HIDDEN_SIZE

    def __post_init__(self) -> None:
        Preg1Role(self.role)
        if self.representation_namespace != "RESTORE":
            raise EvaluationContractViolation("RESTORE representation namespace drifted")
        if self.restore_model_id != RESTORE_MODEL_ID:
            raise EvaluationContractViolation("RESTORE representation binds wrong restorer")
        if self.restore_model_revision != RESTORE_MODEL_REVISION:
            raise EvaluationContractViolation("RESTORE representation binds wrong restorer revision")
        if self.phobert_checkpoint != RESTORE_PHOBERT_CHECKPOINT:
            raise EvaluationContractViolation("RESTORE representation binds wrong PhoBERT checkpoint")
        if self.phobert_revision != RESTORE_PHOBERT_REVISION:
            raise EvaluationContractViolation("RESTORE representation binds wrong PhoBERT revision")
        if self.max_length != RESTORE_STAGE2_MAX_LENGTH or self.truncation is not True:
            raise EvaluationContractViolation("RESTORE tokenization semantics drifted")
        if self.padding != RESTORE_STAGE2_PADDING:
            raise EvaluationContractViolation("RESTORE padding semantics drifted")
        if self.pooling != RESTORE_PHOBERT_POOLING:
            raise EvaluationContractViolation("RESTORE pooling must be FIRST_TOKEN")
        if self.dtype != RESTORE_PHOBERT_DTYPE:
            raise EvaluationContractViolation("RESTORE representations must be torch.float32")
        if self.hidden_size != RESTORE_PHOBERT_HIDDEN_SIZE:
            raise EvaluationContractViolation("RESTORE representations must be 768-dimensional")
        if self.count <= 0:
            raise EvaluationContractViolation("RESTORE representation count must be positive")
        for name in (
            "repository_head",
            "text_cache_manifest_sha256",
            "text_output_digest",
            "ordered_id_digest",
            "label_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvaluationContractViolation(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "repository_head": self.repository_head,
            "role": self.role,
            "condition": self.condition,
            "corruption_seed": self.corruption_seed,
            "representation_namespace": self.representation_namespace,
            "restore_model_id": self.restore_model_id,
            "restore_model_revision": self.restore_model_revision,
            "text_cache_manifest_sha256": self.text_cache_manifest_sha256,
            "text_output_digest": self.text_output_digest,
            "phobert_checkpoint": self.phobert_checkpoint,
            "phobert_revision": self.phobert_revision,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "pooling": self.pooling,
            "ordered_id_digest": self.ordered_id_digest,
            "label_digest": self.label_digest,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "count": self.count,
            "representation_shape": [self.count, self.hidden_size],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RestoreRepresentationKey":
        try:
            return cls(
                schema_version=payload["schema_version"],
                protocol_version=payload["protocol_version"],
                repository_head=payload["repository_head"],
                role=payload["role"],
                condition=payload["condition"],
                corruption_seed=payload["corruption_seed"],
                representation_namespace=payload["representation_namespace"],
                restore_model_id=payload["restore_model_id"],
                restore_model_revision=payload["restore_model_revision"],
                text_cache_manifest_sha256=payload["text_cache_manifest_sha256"],
                text_output_digest=payload["text_output_digest"],
                phobert_checkpoint=payload["phobert_checkpoint"],
                phobert_revision=payload["phobert_revision"],
                max_length=payload["max_length"],
                truncation=payload["truncation"],
                padding=payload["padding"],
                pooling=payload["pooling"],
                ordered_id_digest=payload["ordered_id_digest"],
                label_digest=payload["label_digest"],
                dtype=payload["dtype"],
                hidden_size=payload["hidden_size"],
                count=payload["count"],
            )
        except (KeyError, ValueError) as error:
            raise EvaluationContractViolation(
                f"RESTORE representation key is malformed: {error}"
            ) from error

    def require_compatible(self, other: "RestoreRepresentationKey") -> None:
        mine, theirs = self.to_dict(), other.to_dict()
        differences = [name for name in mine if mine[name] != theirs[name]]
        if differences:
            detail = ", ".join(
                f"{name}: cached={theirs[name]!r} wanted={mine[name]!r}"
                for name in differences
            )
            raise EvaluationContractViolation(
                f"RESTORE representation cache is incompatible: {detail}"
            )


@dataclass(frozen=True)
class RestoreBoundRepresentations:
    values: Any
    key: RestoreRepresentationKey

    def __post_init__(self) -> None:
        shape = tuple(getattr(self.values, "shape", ()))
        if shape != (self.key.count, self.key.hidden_size):
            raise EvaluationContractViolation(
                f"RESTORE representation shape {shape} contradicts key "
                f"{(self.key.count, self.key.hidden_size)}"
            )
        dtype = str(getattr(self.values, "dtype", ""))
        if dtype != self.key.dtype:
            raise EvaluationContractViolation(
                f"RESTORE representation dtype {dtype!r} contradicts {self.key.dtype!r}"
            )
        if getattr(self.values, "requires_grad", False):
            raise EvaluationContractViolation("RESTORE representations must be detached")

    @property
    def role(self) -> Preg1Role:
        return Preg1Role(self.key.role)

    @property
    def condition(self) -> str:
        return self.key.condition

    def require_role(self, expected: Preg1Role, what: str) -> None:
        if self.role is not expected:
            raise EvaluationContractViolation(
                f"{what} requires {expected.value}, got {self.role.value}"
            )

    def require_same_geometry(self, other: "RestoreBoundRepresentations") -> None:
        for field_name in (
            "representation_namespace",
            "restore_model_id",
            "restore_model_revision",
            "phobert_checkpoint",
            "phobert_revision",
            "max_length",
            "truncation",
            "padding",
            "pooling",
            "dtype",
            "hidden_size",
        ):
            if getattr(self.key, field_name) != getattr(other.key, field_name):
                raise EvaluationContractViolation(
                    f"RESTORE representation geometry mismatch on {field_name}"
                )


def tensor_semantic_sha256(tensor: Any) -> str:
    import torch

    if not torch.is_tensor(tensor):
        raise EvaluationContractViolation("expected a torch tensor")
    if tensor.dtype is not torch.float32:
        raise EvaluationContractViolation("RESTORE tensor semantic SHA requires FP32")
    if getattr(tensor, "requires_grad", False):
        raise EvaluationContractViolation("RESTORE tensor semantic SHA requires detached tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise EvaluationContractViolation("RESTORE tensor contains NaN or Inf")
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def require_fp32_matrix(tensor: Any, key: RestoreRepresentationKey) -> None:
    import torch

    if not torch.is_tensor(tensor):
        raise EvaluationContractViolation("RESTORE representation is not a torch tensor")
    if tensor.dtype is not torch.float32:
        raise EvaluationContractViolation(f"RESTORE representation dtype is {tensor.dtype}")
    if tuple(tensor.shape) != (key.count, key.hidden_size):
        raise EvaluationContractViolation(
            f"RESTORE representation shape {tuple(tensor.shape)} != "
            f"{(key.count, key.hidden_size)}"
        )
    if tensor.requires_grad:
        raise EvaluationContractViolation("RESTORE representation requires grad")
    if not bool(torch.isfinite(tensor).all()):
        raise EvaluationContractViolation("RESTORE representation contains NaN or Inf")


class RestoreRepresentationCache:
    METADATA_NAME = "restore-representation-key.json"
    TENSOR_NAME = "restore-representations.pt"
    MANIFEST_NAME = "restore-representation-manifest.json"
    IN_PROGRESS_NAME = "RESTORE_REPRESENTATION.inprogress"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def key_path(self) -> Path:
        return self.directory / self.METADATA_NAME

    @property
    def tensor_path(self) -> Path:
        return self.directory / self.TENSOR_NAME

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST_NAME

    @property
    def in_progress_path(self) -> Path:
        return self.directory / self.IN_PROGRESS_NAME

    def exists(self) -> bool:
        return self.key_path.is_file() and self.tensor_path.is_file() and self.manifest_path.is_file()

    def _require_no_incomplete_transaction(self) -> None:
        if self.in_progress_path.exists():
            raise EvaluationContractViolation(
                f"{self.directory} contains {self.IN_PROGRESS_NAME}; inspect incomplete "
                "RESTORE representation transaction"
            )
        complete_bits = [self.key_path.exists(), self.tensor_path.exists(), self.manifest_path.exists()]
        if any(complete_bits) and not all(complete_bits):
            raise EvaluationContractViolation(
                f"{self.directory} contains a partial RESTORE representation cache"
            )

    def read_key(self) -> RestoreRepresentationKey:
        self._require_no_incomplete_transaction()
        return RestoreRepresentationKey.from_dict(read_json(self.key_path))

    def load(self, key: RestoreRepresentationKey) -> RestoreBoundRepresentations:
        import torch

        key.require_compatible(self.read_key())
        tensor = torch.load(self.tensor_path, map_location="cpu")
        require_fp32_matrix(tensor, key)
        manifest = read_json(self.manifest_path)
        if manifest.get("semantic_tensor_sha256") != tensor_semantic_sha256(tensor):
            raise EvaluationContractViolation("RESTORE representation semantic SHA drifted")
        if manifest.get("raw_tensor_sha256") != file_sha256(self.tensor_path):
            raise EvaluationContractViolation("RESTORE representation raw SHA drifted")
        return RestoreBoundRepresentations(values=tensor, key=key)

    def save(self, key: RestoreRepresentationKey, tensor: Any) -> RestoreBoundRepresentations:
        import torch

        if self.exists():
            return self.load(key)
        self._require_no_incomplete_transaction()
        require_fp32_matrix(tensor, key)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.in_progress_path, {"key": key.to_dict()})
        write_json(self.key_path, key.to_dict())
        temp = self.tensor_path.with_name(self.tensor_path.name + ".tmp")
        torch.save(tensor.detach().cpu(), temp)
        temp.replace(self.tensor_path)
        manifest = {
            "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
            "protocol_version": RESTORE_PROTOCOL_VERSION,
            "key": key.to_dict(),
            "semantic_tensor_sha256": tensor_semantic_sha256(tensor),
            "raw_tensor_sha256": file_sha256(self.tensor_path),
            "torch_dtype": str(tensor.dtype),
            "shape": [int(v) for v in tensor.shape],
        }
        write_json(self.manifest_path, manifest)
        self.in_progress_path.unlink()
        return self.load(key)


class RestorePhaseRegistry:
    """Durable phase-level resume markers for the RESTORE runner."""

    COMPLETE_SUFFIX = ".complete.json"
    IN_PROGRESS_SUFFIX = ".inprogress"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def complete_path(self, phase: str) -> Path:
        return self.directory / f"{phase}{self.COMPLETE_SUFFIX}"

    def in_progress_path(self, phase: str) -> Path:
        return self.directory / f"{phase}{self.IN_PROGRESS_SUFFIX}"

    def phase_identity_digest(self, identity: Mapping[str, Any]) -> str:
        return sha256_bytes(canonical_json_bytes(dict(identity)))

    def is_complete(self, phase: str, identity: Mapping[str, Any]) -> bool:
        path = self.complete_path(phase)
        if not path.is_file():
            return False
        recorded = read_json(path)
        expected = self.phase_identity_digest(identity)
        actual = recorded.get("identity_digest")
        if actual != expected:
            raise EvaluationContractViolation(
                f"completed RESTORE phase {phase} has identity {actual!r}, expected {expected!r}"
            )
        return True

    def start(self, phase: str, identity: Mapping[str, Any]) -> None:
        if self.is_complete(phase, identity):
            return
        marker = self.in_progress_path(phase)
        if marker.exists():
            recorded = read_json(marker)
            expected = self.phase_identity_digest(identity)
            if recorded.get("identity_digest") != expected:
                raise EvaluationContractViolation(
                    f"RESTORE phase {phase} has incompatible in-progress marker"
                )
            raise EvaluationContractViolation(
                f"RESTORE phase {phase} has an in-progress marker; inspect before resume"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            marker,
            {
                "phase": phase,
                "identity": dict(identity),
                "identity_digest": self.phase_identity_digest(identity),
            },
        )

    def complete(
        self,
        phase: str,
        identity: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        expected = self.phase_identity_digest(identity)
        marker = self.in_progress_path(phase)
        if marker.exists():
            recorded = read_json(marker)
            if recorded.get("identity_digest") != expected:
                raise EvaluationContractViolation(
                    f"RESTORE phase {phase} in-progress identity changed before completion"
                )
        complete = {
            "phase": phase,
            "identity": dict(identity),
            "identity_digest": expected,
            "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
        }
        if payload:
            complete["payload"] = dict(payload)
        write_json(self.complete_path(phase), complete)
        if marker.exists():
            marker.unlink()

    def run_once(
        self,
        phase: str,
        identity: Mapping[str, Any],
        action,
    ) -> dict[str, Any]:
        if self.is_complete(phase, identity):
            return read_json(self.complete_path(phase))
        self.start(phase, identity)
        payload = action()
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise EvaluationContractViolation(f"RESTORE phase {phase} action returned non-mapping")
        self.complete(phase, identity, payload)
        return read_json(self.complete_path(phase))


def completed_artifact_sha256s(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path)): file_sha256(path) for path in paths}
