"""Direct frozen external restorer for the RESTORE baseline."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.baselines.restore.cache import file_sha256
from unmark.baselines.restore.config import (
    RESTORE_GENERATION_CONFIG,
    RESTORE_HF_GENERATION_CONFIG_FIELDS,
    RESTORE_MODEL_ARCHITECTURE,
    RESTORE_MODEL_CONFIG_SHA256,
    RESTORE_MODEL_GENERATION_CONFIG_SHA256,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_LICENSE,
    RESTORE_MODEL_PARAMETER_COUNT,
    RESTORE_MODEL_REVISION,
    RESTORE_MODEL_WEIGHT_FILENAME,
    RESTORE_MODEL_WEIGHT_SHA256,
    RESTORE_PUBLIC_SMOKE_EXAMPLES,
    RESTORE_RUNTIME_TRANSFORMERS_VERSION,
    RESTORE_TOKENIZER_CONTRACT,
    RESTORE_TOKENIZER_ARTIFACT_SHA256S,
    RestoreGenerationConfig,
    RestoreTokenizerContract,
    require_generation_config,
    require_restore_model_identity,
    require_tokenizer_contract,
)
from unmark.evaluation.contracts import EvaluationContractViolation


RESTORE_DIRECT_MODEL_API = ("AutoTokenizer", "AutoModelForSeq2SeqLM")
RESTORE_FALLBACKS_ENABLED = False
RESTORE_RULE_BASED_FALLBACK = False
RESTORE_LLM_FALLBACK = False
RESTORE_SPELL_CORRECTION_FALLBACK = False


@dataclass(frozen=True)
class RestoreHubMetadata:
    model_id: str
    revision: str
    license: str
    architecture: str
    parameter_count: int
    weight_sha256: str
    siblings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "weight_sha256": self.weight_sha256,
            "siblings": list(self.siblings),
        }


def _json_request(url: str, *, data: bytes | None = None) -> Any:
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise EvaluationContractViolation(
            f"could not verify RESTORE Hub metadata at {url}: {error}"
        ) from error


def verify_restore_hub_metadata() -> RestoreHubMetadata:
    """Programmatically verify the pinned Hub revision and file metadata.

    This is metadata-only. It does not load transformers and does not download
    the safetensors weight blob.
    """

    model_url = f"https://huggingface.co/api/models/{RESTORE_MODEL_ID}/revision/{RESTORE_MODEL_REVISION}"
    metadata = _json_request(model_url)
    if metadata.get("id") != RESTORE_MODEL_ID or metadata.get("sha") != RESTORE_MODEL_REVISION:
        raise EvaluationContractViolation(
            "RESTORE Hub metadata did not resolve to the pinned model revision"
        )
    card = metadata.get("cardData") or {}
    config = metadata.get("config") or {}
    architecture = (config.get("architectures") or [None])[0]
    params = ((metadata.get("safetensors") or {}).get("parameters") or {}).get("F32")
    if card.get("license") != RESTORE_MODEL_LICENSE:
        raise EvaluationContractViolation(
            f"RESTORE model license drifted to {card.get('license')!r}"
        )
    if architecture != RESTORE_MODEL_ARCHITECTURE:
        raise EvaluationContractViolation(
            f"RESTORE model architecture drifted to {architecture!r}"
        )
    if int(params) != RESTORE_MODEL_PARAMETER_COUNT:
        raise EvaluationContractViolation(
            f"RESTORE model parameter count drifted to {params!r}"
        )

    paths = [
        "config.json",
        "generation_config.json",
        *sorted(RESTORE_TOKENIZER_ARTIFACT_SHA256S),
        RESTORE_MODEL_WEIGHT_FILENAME,
    ]
    paths_url = f"https://huggingface.co/api/models/{RESTORE_MODEL_ID}/paths-info/{RESTORE_MODEL_REVISION}"
    files = _json_request(paths_url, data=json.dumps({"paths": paths, "expand": True}).encode("utf-8"))
    if not isinstance(files, list):
        raise EvaluationContractViolation("RESTORE Hub paths-info response is not a list")
    by_path = {entry.get("path"): entry for entry in files if isinstance(entry, Mapping)}
    missing = sorted(set(paths) - set(by_path))
    if missing:
        raise EvaluationContractViolation(f"RESTORE Hub metadata is missing files: {missing}")
    lfs = by_path[RESTORE_MODEL_WEIGHT_FILENAME].get("lfs") or {}
    if lfs.get("oid") != RESTORE_MODEL_WEIGHT_SHA256:
        raise EvaluationContractViolation(
            f"RESTORE model weight SHA256 drifted to {lfs.get('oid')!r}"
        )
    if int(lfs.get("size", 0)) <= 0:
        raise EvaluationContractViolation("RESTORE model weight metadata has no positive size")

    require_restore_model_identity(metadata["id"], metadata["sha"])
    return RestoreHubMetadata(
        model_id=metadata["id"],
        revision=metadata["sha"],
        license=card["license"],
        architecture=architecture,
        parameter_count=int(params),
        weight_sha256=lfs["oid"],
        siblings=tuple(sorted(entry["rfilename"] for entry in metadata.get("siblings", []))),
    )


def _parameter_report(module: Any) -> tuple[int, list[str], set[str]]:
    count = 0
    trainable: list[str] = []
    dtypes: set[str] = set()
    for name, parameter in module.named_parameters():
        count += int(parameter.numel())
        if parameter.requires_grad:
            trainable.append(name)
        dtype = getattr(parameter, "dtype", None)
        if dtype is not None:
            dtypes.add(str(dtype))
    return count, trainable, dtypes


class FrozenDiacriticRestorer:
    """Thin direct wrapper over a frozen seq2seq diacritic-restoration model."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        generation_config: RestoreGenerationConfig = RESTORE_GENERATION_CONFIG,
        tokenizer_contract: RestoreTokenizerContract = RESTORE_TOKENIZER_CONTRACT,
        expected_parameter_count: int | None = RESTORE_MODEL_PARAMETER_COUNT,
    ) -> None:
        require_generation_config(generation_config)
        require_tokenizer_contract(tokenizer_contract)
        self.tokenizer = tokenizer
        self.model = model
        self.generation_config = generation_config
        self.tokenizer_contract = tokenizer_contract
        self.expected_parameter_count = expected_parameter_count
        self.require_frozen()

    def require_frozen(self) -> None:
        count, trainable, dtypes = _parameter_report(self.model)
        if trainable:
            raise EvaluationContractViolation(
                f"RESTORE restorer has trainable parameter(s): {trainable[:5]}"
            )
        if bool(getattr(self.model, "training", True)):
            raise EvaluationContractViolation("RESTORE restorer must be in eval mode")
        if self.expected_parameter_count is not None and count != self.expected_parameter_count:
            raise EvaluationContractViolation(
                f"RESTORE restorer parameter count {count} != {self.expected_parameter_count}"
            )
        floating = {dtype for dtype in dtypes if "float" in dtype}
        if floating and floating != {"torch.float32"}:
            raise EvaluationContractViolation(
                f"RESTORE restorer must run FP32, got parameter dtype(s) {sorted(floating)}"
            )

    def restore_text(self, text: str) -> str:
        return self.restore_batch([text])[0]

    def restore_batch(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise EvaluationContractViolation("restore_batch expects a sequence, not one string")
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise EvaluationContractViolation("RESTORE inputs must be strings")
        self.require_frozen()

        import torch

        encoded = self.tokenizer(list(texts), **self.tokenizer_contract.to_encode_kwargs())
        device = next(self.model.parameters()).device
        moved = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **moved,
                **self.generation_config.to_generate_kwargs(),
            )
        decoded = self.tokenizer.batch_decode(
            output_ids,
            **self.tokenizer_contract.to_decode_kwargs(),
        )
        if len(decoded) != len(texts):
            raise EvaluationContractViolation(
                f"RESTORE decoded {len(decoded)} rows for {len(texts)} inputs"
            )
        return [str(value) for value in decoded]


def _freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def load_frozen_restorer(
    *,
    cache_dir: str | Path | None = None,
    device: str = "auto",
) -> FrozenDiacriticRestorer:
    """Load the one pinned RESTORE model directly through Transformers."""

    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    require_restore_model_identity(RESTORE_MODEL_ID, RESTORE_MODEL_REVISION)
    if transformers.__version__ != RESTORE_RUNTIME_TRANSFORMERS_VERSION:
        raise EvaluationContractViolation(
            "RESTORE runtime requires transformers "
            f"{RESTORE_RUNTIME_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        RESTORE_MODEL_ID,
        revision=RESTORE_MODEL_REVISION,
        **RESTORE_TOKENIZER_CONTRACT.to_load_kwargs(),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        RESTORE_MODEL_ID,
        revision=RESTORE_MODEL_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.to(device)
    _freeze_module(model)
    return FrozenDiacriticRestorer(tokenizer, model)


def verify_local_restore_artifacts(snapshot_dir: str | Path) -> dict[str, Any]:
    """Hash local RESTORE model artifacts after the runtime has cached them."""

    root = Path(snapshot_dir)
    expected = {
        "config.json": RESTORE_MODEL_CONFIG_SHA256,
        "generation_config.json": RESTORE_MODEL_GENERATION_CONFIG_SHA256,
        **RESTORE_TOKENIZER_ARTIFACT_SHA256S,
        RESTORE_MODEL_WEIGHT_FILENAME: RESTORE_MODEL_WEIGHT_SHA256,
    }
    observed: dict[str, dict[str, Any]] = {}
    for filename, expected_sha in expected.items():
        path = root / filename
        if not path.is_file():
            raise EvaluationContractViolation(f"RESTORE local artifact missing: {path}")
        actual = file_sha256(path)
        if actual != expected_sha:
            raise EvaluationContractViolation(
                f"RESTORE artifact {filename} SHA256 mismatch: expected {expected_sha}, got {actual}"
            )
        observed[filename] = {"sha256": actual, "bytes": path.stat().st_size}
    return {
        "model_id": RESTORE_MODEL_ID,
        "revision": RESTORE_MODEL_REVISION,
        "artifacts": observed,
    }


def run_batched_decode_smoke(
    restorer: FrozenDiacriticRestorer,
    *,
    batch_size: int,
    examples: Sequence[str] = RESTORE_PUBLIC_SMOKE_EXAMPLES,
) -> dict[str, Any]:
    """Require single-example greedy decode to equal batched greedy decode."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise EvaluationContractViolation(f"RESTORE batch size must be positive, got {batch_size!r}")
    examples = tuple(examples)
    singles = [restorer.restore_text(text) for text in examples]
    batched: list[str] = []
    for start in range(0, len(examples), batch_size):
        batched.extend(restorer.restore_batch(examples[start : start + batch_size]))
    if singles != batched:
        raise EvaluationContractViolation(
            "RESTORE batched greedy decode differs from single-example greedy decode"
        )
    return {
        "examples": len(examples),
        "batch_size": batch_size,
        "single_equals_batched": True,
        "generation_config": restorer.generation_config.to_dict(),
        "tokenizer_contract": restorer.tokenizer_contract.to_dict(),
    }


def restorer_contract_manifest(*, transformers_version: str | None = None) -> dict[str, Any]:
    return {
        "direct_model_api": list(RESTORE_DIRECT_MODEL_API),
        "pipeline_used": False,
        "fallbacks_enabled": RESTORE_FALLBACKS_ENABLED,
        "rule_based_fallback": RESTORE_RULE_BASED_FALLBACK,
        "llm_fallback": RESTORE_LLM_FALLBACK,
        "spell_correction_fallback": RESTORE_SPELL_CORRECTION_FALLBACK,
        "model_id": RESTORE_MODEL_ID,
        "revision": RESTORE_MODEL_REVISION,
        "generation_config": RESTORE_GENERATION_CONFIG.to_dict(),
        "tokenizer_contract": RESTORE_TOKENIZER_CONTRACT.to_dict(),
        "hf_generation_config_fields": dict(RESTORE_HF_GENERATION_CONFIG_FIELDS),
        "required_transformers_version": RESTORE_RUNTIME_TRANSFORMERS_VERSION,
        "trainable_parameters": 0,
        "restorer_trained": False,
        "restorer_frozen": True,
        "transformers_version": transformers_version,
    }
