"""Stage-2 frozen UNMARK dual-finalist infrastructure.

This module implements only the acceptance-time plumbing Audit 049 left open:

* two frozen pathway identities, ``UNMARK-A`` and ``UNMARK-B``;
* read-only binding of an operator-supplied Stage-1 checkpoint to the matching
  frozen finalist;
* downstream UNMARK input construction over the base stream ``T(b(x))`` for the
  six frozen corruption conditions;
* first-token ``<s>`` representation extraction through a frozen encoder and a
  frozen adapter.

It deliberately does not train a head, read a dataset, inspect measurement-dev,
open TEST, select between A/B, restore text, replace the tokenizer, add a word
segmenter, or create an ensemble.

Torch and transformers are imported lazily inside the functions that need them,
so the structural contracts remain testable in the ML-free local environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from unmark.corruption import (
    CorruptionCondition,
    CorruptionPurpose,
    EligibilityPolicy,
    UnknownCondition,
    corrupt,
    get_condition,
)
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    PADDING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_NUM_LABELS,
    PRIMARY_TASK,
    TRUNCATION,
)
from unmark.modeling.collate import EncodedExample, build_example
from unmark.modeling.contracts import LETTER_NA_SENTINEL, TONE_NA_SENTINEL
from unmark.orthography import Eligibility, canon
from unmark.stage1.data import project_text
from unmark.stage1.finalists import (
    ADAPTER_STATE_KEYS,
    ADAPTER_TENSOR_COUNT,
    FINALIST_A,
    FINALIST_B,
    FINALISTS,
    PRECISION,
    FinalistFreezeViolation,
    FinalistIdentity,
    finalist_for,
    validate_finalists,
    verify_finalist_checkpoint,
)
from unmark.stage1.protocol import ADAPTER_TRAINABLE_PARAMETERS, HIDDEN_SIZE
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor


STAGE2_DUAL_FINALIST_INFRA_SCHEMA_VERSION = "stage2-dual-finalist-infra-v1"

STAGE2_UNMARK_CONDITIONS: tuple[str, ...] = (
    "FULL",
    "P25",
    "P50",
    "P75",
    "P100",
    "STRIP_ALL",
)
STAGE2_EXCLUDED_CONDITIONS: tuple[str, ...] = ("VARIANT",)
STAGE2_FORWARD_TENSOR_KEYS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "tone_ids",
    "tone_mask",
    "letter_ids",
    "letter_mask",
)
STAGE2_FIRST_TOKEN_POOLING = "FIRST_TOKEN"
STAGE2_REPRESENTATION_DTYPE = "torch.float32"
STAGE2_DATASET = PRIMARY_DATASET
STAGE2_DATASET_VERSION = PRIMARY_DATASET_VERSION
STAGE2_TASK = PRIMARY_TASK
STAGE2_NUM_LABELS = PRIMARY_NUM_LABELS
STAGE2_HEAD_TRAINING_IMPLEMENTED = False
STAGE2_A_B_SELECTION_IMPLEMENTED = False
STAGE2_MEASUREMENT_DEV_SELECTION_IMPLEMENTED = False
STAGE2_OFFICIAL_TEST_REACHABLE = False
STAGE2_TRAINING_STARTED = False

STAGE2_TOKENIZATION = (
    f"checkpoint={ENCODER_CHECKPOINT}, revision={ENCODER_REVISION}, "
    f"max_length={MAX_LENGTH}, truncation={TRUNCATION}, padding={PADDING!r}, "
    "base stream T(b(x)), no restoration, no tokenizer replacement, no word segmentation"
)


class Stage2UnmarkArm(Enum):
    """Exactly the two Audit-049 UNMARK arms.

    The enum is intentionally separate from ``SystemPathway``. That older enum
    remains the pre-G1 VANILLA/BASE_ONLY diagnostic contract; Stage-2 UNMARK has
    checkpoint-bound adapter identities rather than text-only pathway labels.
    """

    UNMARK_A = "UNMARK-A"
    UNMARK_B = "UNMARK-B"

    @property
    def finalist_key(self) -> str:
        if self is Stage2UnmarkArm.UNMARK_A:
            return "A"
        if self is Stage2UnmarkArm.UNMARK_B:
            return "B"
        raise EvaluationContractViolation(f"unknown Stage-2 UNMARK arm: {self!r}")


STAGE2_UNMARK_ARM_NAMES: tuple[str, ...] = tuple(arm.value for arm in Stage2UnmarkArm)


def require_stage2_unmark_arm(value: str | Stage2UnmarkArm) -> Stage2UnmarkArm:
    """Parse a pathway identity, refusing anything outside the frozen pair."""

    if isinstance(value, Stage2UnmarkArm):
        return value
    for arm in Stage2UnmarkArm:
        if value == arm.value:
            return arm
    raise EvaluationContractViolation(
        f"unknown Stage-2 UNMARK arm {value!r}; frozen identities are "
        f"{list(STAGE2_UNMARK_ARM_NAMES)}"
    )


def finalist_for_arm(arm: str | Stage2UnmarkArm) -> FinalistIdentity:
    """Return the frozen Stage-1 finalist identity bound to a Stage-2 arm."""

    resolved = require_stage2_unmark_arm(arm)
    return finalist_for(resolved.finalist_key)


def require_dual_finalist_state() -> None:
    """Fail closed if the Stage-1 finalist universe or arm universe drifts."""

    validate_finalists(FINALISTS)
    if STAGE2_UNMARK_ARM_NAMES != ("UNMARK-A", "UNMARK-B"):
        raise EvaluationContractViolation(
            f"Stage-2 UNMARK arms drifted to {STAGE2_UNMARK_ARM_NAMES}; Audit 049 "
            "freezes exactly UNMARK-A and UNMARK-B"
        )
    if finalist_for_arm(Stage2UnmarkArm.UNMARK_A) != FINALIST_A:
        raise EvaluationContractViolation("UNMARK-A is no longer bound to finalist A")
    if finalist_for_arm(Stage2UnmarkArm.UNMARK_B) != FINALIST_B:
        raise EvaluationContractViolation("UNMARK-B is no longer bound to finalist B")


@dataclass(frozen=True)
class Stage2FinalistBinding:
    """The verified checkpoint identity carried by a frozen UNMARK pathway."""

    arm: Stage2UnmarkArm
    finalist_key: str
    source_stage: str
    run_seed: int
    update: int
    checkpoint_sha256: str
    checkpoint_path: str
    evidence: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE2_DUAL_FINALIST_INFRA_SCHEMA_VERSION,
            "arm": self.arm.value,
            "finalist_key": self.finalist_key,
            "source_stage": self.source_stage,
            "run_seed": self.run_seed,
            "update": self.update,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_path": self.checkpoint_path,
            "stage1_evidence_kind": self.evidence.get("kind"),
        }


def bind_verified_checkpoint(
    arm: str | Stage2UnmarkArm,
    checkpoint_path: str | Path,
    evidence: Mapping[str, Any],
) -> Stage2FinalistBinding:
    """Bind verifier evidence to the expected Stage-2 arm.

    ``verify_finalist_checkpoint`` is still the authoritative verifier. This
    wrapper only checks that the returned evidence names the finalist that the
    requested Stage-2 arm is allowed to use.
    """

    resolved = require_stage2_unmark_arm(arm)
    finalist = finalist_for_arm(resolved)
    required = {
        "kind",
        "finalist_key",
        "source_stage",
        "run_seed",
        "update",
        "checkpoint_sha256",
        "checkpoint_schema_version",
        "adapter_tensor_count",
        "adapter_trainable_parameters",
        "adapter_dtype",
        "all_finite",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise EvaluationContractViolation(
            f"checkpoint evidence for {resolved.value} is missing {missing}"
        )

    expected: dict[str, Any] = {
        "kind": "stage1_finalist_checkpoint_evidence",
        "finalist_key": finalist.key,
        "source_stage": finalist.source_stage,
        "run_seed": finalist.run_seed,
        "update": finalist.update,
        "checkpoint_sha256": finalist.checkpoint_sha256,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": ADAPTER_TENSOR_COUNT,
        "adapter_trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        "adapter_dtype": PRECISION,
        "all_finite": True,
    }
    differences = [
        name for name, wanted in expected.items() if evidence.get(name) != wanted
    ]
    if differences:
        detail = ", ".join(
            f"{name}: got {evidence.get(name)!r}, expected {expected[name]!r}"
            for name in differences
        )
        raise EvaluationContractViolation(
            f"checkpoint evidence does not bind to {resolved.value}: {detail}"
        )

    return Stage2FinalistBinding(
        arm=resolved,
        finalist_key=finalist.key,
        source_stage=finalist.source_stage,
        run_seed=finalist.run_seed,
        update=finalist.update,
        checkpoint_sha256=finalist.checkpoint_sha256,
        checkpoint_path=str(Path(checkpoint_path)),
        evidence=evidence,
    )


@dataclass(frozen=True)
class Stage2UnmarkInput:
    """One downstream example encoded on the corruption-invariant base grid."""

    sample_id: str
    condition: str
    canonical_text: str
    corrupted_text: str
    base_text: str
    input_ids: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    tone_ids: tuple[int, ...]
    tone_mask: tuple[bool, ...]
    letter_ids: tuple[tuple[int, ...], ...]
    corruption_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = STAGE2_DUAL_FINALIST_INFRA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise EvaluationContractViolation("sample_id must be a non-empty stable identity")
        if self.condition not in STAGE2_UNMARK_CONDITIONS:
            raise EvaluationContractViolation(
                f"unsupported Stage-2 corruption condition {self.condition!r}"
            )
        lengths = {
            "input_ids": len(self.input_ids),
            "special_tokens_mask": len(self.special_tokens_mask),
            "tone_ids": len(self.tone_ids),
            "tone_mask": len(self.tone_mask),
            "letter_ids": len(self.letter_ids),
        }
        if len(set(lengths.values())) != 1:
            raise EvaluationContractViolation(f"UNMARK input fields disagree on length: {lengths}")
        if not self.input_ids:
            raise EvaluationContractViolation("UNMARK input cannot be empty")

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def base_token_grid(self) -> tuple[int, ...]:
        return self.input_ids

    @property
    def channels(self) -> tuple[Any, ...]:
        return (self.tone_ids, self.tone_mask, self.letter_ids)

    def encoded(self) -> EncodedExample:
        return EncodedExample(
            input_ids=list(self.input_ids),
            special_tokens_mask=list(self.special_tokens_mask),
            tone_ids=list(self.tone_ids),
            tone_mask=list(self.tone_mask),
            letter_ids=[list(row) for row in self.letter_ids],
        )


def require_stage2_condition(condition: str | CorruptionCondition) -> CorruptionCondition:
    """Return one of the six frozen conditions; ``VARIANT`` remains unreachable."""

    try:
        resolved = get_condition(condition)
    except UnknownCondition as error:
        raise EvaluationContractViolation(str(error)) from error
    if resolved.name not in STAGE2_UNMARK_CONDITIONS:
        raise EvaluationContractViolation(
            f"condition {resolved.name!r} is not in the frozen Stage-2 grid "
            f"{list(STAGE2_UNMARK_CONDITIONS)}"
        )
    return resolved


def apply_stage2_corruption(
    text: str,
    condition: str | CorruptionCondition,
    *,
    seed: int,
    sample_id: str,
    purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> Any:
    """Apply the repository's deterministic corruption API for Stage-2.

    Randomness is keyed by ``sample_id`` through ``unmark.corruption.corrupt``.
    The default purpose is scientific; tests may explicitly pass ``SELF_CHECK``
    to avoid requiring the external inventory.
    """

    resolved = require_stage2_condition(condition)
    return corrupt(
        text,
        resolved,
        seed=seed,
        sample_id=sample_id,
        purpose=purpose,
        eligibility_policy=eligibility_policy,
    )


def _with_special_tokens(tokenizer: Any, content_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    ids = list(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    mask = list(
        tokenizer.get_special_tokens_mask(
            list(content_ids), already_has_special_tokens=False
        )
    )
    if len(ids) != len(mask):
        raise EvaluationContractViolation(
            "tokenizer special-token mask length does not match input ids"
        )
    content_slots = sum(1 for value in mask if not value)
    if content_slots != len(content_ids):
        raise EvaluationContractViolation(
            "tokenizer special-token mask does not expose one content slot per base token"
        )
    return ids, mask


def _truncate_content_to_max_length(
    tokenizer: Any,
    content_ids: Sequence[int],
    projections: Sequence[Any],
    *,
    max_length: int,
) -> tuple[list[int], list[Any], bool]:
    """Apply the frozen downstream single-sequence truncation to content tokens."""

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise EvaluationContractViolation(f"max_length must be a positive int, got {max_length!r}")
    full_ids, _ = _with_special_tokens(tokenizer, content_ids)
    if len(full_ids) <= max_length:
        return list(content_ids), list(projections), False

    special_count = len(full_ids) - len(content_ids)
    keep = max_length - special_count
    if keep < 0:
        raise EvaluationContractViolation(
            f"max_length={max_length} cannot hold the tokenizer's {special_count} "
            "single-sequence special tokens"
        )
    return list(content_ids[:keep]), list(projections[:keep]), True


def prepare_stage2_unmark_input(
    *,
    text: str,
    sample_id: str,
    tokenizer: Any,
    condition: str | CorruptionCondition,
    corruption_seed: int,
    classifier: Callable[[str], Eligibility] | None = None,
    unk_token_id: int | None = None,
    max_length: int = MAX_LENGTH,
    corruption_purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> Stage2UnmarkInput:
    """Prepare one downstream example for one frozen corruption condition.

    The grid is built from the clean base stream ``b(canon(x))`` and then reused
    for the corrupted side-channel state. Any corruption that changes the base
    text, token ids, or projection count raises instead of being repaired.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise EvaluationContractViolation("sample_id must be a non-empty stable identity")
    canonical = canon(text)
    corruption = apply_stage2_corruption(
        canonical,
        condition,
        seed=corruption_seed,
        sample_id=sample_id,
        purpose=corruption_purpose,
        eligibility_policy=eligibility_policy,
    )

    clean_base, clean_content_ids, clean_projections = project_text(
        canonical, tokenizer, classifier, unk_token_id
    )
    corrupt_base, corrupt_content_ids, corrupt_projections = project_text(
        corruption.corrupted_text, tokenizer, classifier, unk_token_id
    )

    if clean_base != corrupt_base:
        raise EvaluationContractViolation(
            f"b(C(x)) != b(x) for sample {sample_id!r}: "
            f"{corrupt_base!r} vs {clean_base!r}"
        )
    if list(clean_content_ids) != list(corrupt_content_ids):
        raise EvaluationContractViolation(
            f"base token ids differ after {corruption.condition.name} for sample {sample_id!r}"
        )
    if len(clean_projections) != len(corrupt_projections):
        raise EvaluationContractViolation(
            f"projection counts differ after {corruption.condition.name} for sample {sample_id!r}: "
            f"{len(corrupt_projections)} vs {len(clean_projections)}"
        )

    content_ids, projections, truncated = _truncate_content_to_max_length(
        tokenizer, clean_content_ids, corrupt_projections, max_length=max_length
    )
    input_ids, special_tokens_mask = _with_special_tokens(tokenizer, content_ids)
    encoded = build_example(input_ids, special_tokens_mask, projections)

    return Stage2UnmarkInput(
        sample_id=sample_id,
        condition=corruption.condition.name,
        canonical_text=canonical,
        corrupted_text=corruption.corrupted_text,
        base_text=clean_base,
        input_ids=tuple(encoded.input_ids),
        special_tokens_mask=tuple(encoded.special_tokens_mask),
        tone_ids=tuple(encoded.tone_ids),
        tone_mask=tuple(encoded.tone_mask),
        letter_ids=tuple(tuple(row) for row in encoded.letter_ids),
        corruption_metadata={
            "condition_scope": corruption.condition.scope.value,
            "condition_probability": corruption.condition.probability,
            "corruption_seed": corruption_seed,
            "corruption_purpose": corruption_purpose.name,
            "eligibility_policy": corruption.eligibility_policy.name,
            "text_identity": corruption.text_identity,
            "truncated": truncated,
            "max_length": max_length,
            "content_tokens_before_truncation": len(clean_content_ids),
            "content_tokens_after_truncation": len(content_ids),
        },
    )


def require_same_base_grid(inputs: Sequence[Stage2UnmarkInput]) -> None:
    """Assert every condition for one sample sits on the same base-token grid."""

    if not inputs:
        raise EvaluationContractViolation("cannot verify an empty Stage-2 condition grid")
    sample_ids = {item.sample_id for item in inputs}
    if len(sample_ids) != 1:
        raise EvaluationContractViolation(
            f"base-grid comparison is for one sample, got sample ids {sorted(sample_ids)}"
        )
    first = inputs[0]
    for item in inputs[1:]:
        if item.base_text != first.base_text:
            raise EvaluationContractViolation(
                f"base text changed across conditions for sample {first.sample_id!r}"
            )
        if item.input_ids != first.input_ids:
            raise EvaluationContractViolation(
                f"base token ids changed across conditions for sample {first.sample_id!r}"
            )
        if item.special_tokens_mask != first.special_tokens_mask:
            raise EvaluationContractViolation(
                f"base special-token mask changed across conditions for sample {first.sample_id!r}"
            )


def prepare_stage2_unmark_condition_grid(
    *,
    text: str,
    sample_id: str,
    tokenizer: Any,
    corruption_seed: int,
    conditions: Sequence[str | CorruptionCondition] = STAGE2_UNMARK_CONDITIONS,
    classifier: Callable[[str], Eligibility] | None = None,
    unk_token_id: int | None = None,
    max_length: int = MAX_LENGTH,
    corruption_purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
    eligibility_policy: EligibilityPolicy | None = None,
) -> tuple[Stage2UnmarkInput, ...]:
    """Prepare all requested frozen conditions and prove base-grid invariance."""

    prepared = tuple(
        prepare_stage2_unmark_input(
            text=text,
            sample_id=sample_id,
            tokenizer=tokenizer,
            condition=condition,
            corruption_seed=corruption_seed,
            classifier=classifier,
            unk_token_id=unk_token_id,
            max_length=max_length,
            corruption_purpose=corruption_purpose,
            eligibility_policy=eligibility_policy,
        )
        for condition in conditions
    )
    require_same_base_grid(prepared)
    return prepared


def _pad_one(input_: Stage2UnmarkInput, *, width: int, depth: int, pad_token_id: int) -> dict[str, Any]:
    if input_.length > width:
        raise EvaluationContractViolation(
            f"prepared input length {input_.length} exceeds fixed max_length={width}"
        )
    pad = width - input_.length
    padded_letters = [list(row) for row in input_.letter_ids] + [[] for _ in range(pad)]
    return {
        "input_ids": list(input_.input_ids) + [pad_token_id] * pad,
        "attention_mask": [1] * input_.length + [0] * pad,
        "special_tokens_mask": list(input_.special_tokens_mask) + [1] * pad,
        "tone_ids": list(input_.tone_ids) + [TONE_NA_SENTINEL] * pad,
        "tone_mask": list(input_.tone_mask) + [False] * pad,
        "letter_ids": [
            list(row) + [LETTER_NA_SENTINEL] * (depth - len(row))
            for row in padded_letters
        ],
        "letter_mask": [
            [True] * len(row) + [False] * (depth - len(row))
            for row in padded_letters
        ],
    }


def collate_stage2_unmark_batch(
    inputs: Sequence[Stage2UnmarkInput],
    *,
    pad_token_id: int,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Pad Stage-2 UNMARK inputs to the frozen downstream max length.

    Returns tensors plus sample/condition metadata. Imports torch lazily.
    """

    if not inputs:
        raise EvaluationContractViolation("cannot collate an empty Stage-2 UNMARK batch")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise EvaluationContractViolation(f"max_length must be a positive int, got {max_length!r}")

    depth = max((len(row) for item in inputs for row in item.letter_ids), default=0)
    depth = max(depth, 1)
    rows = [_pad_one(item, width=max_length, depth=depth, pad_token_id=pad_token_id) for item in inputs]

    import torch

    tensor_fields = {
        "input_ids": torch.long,
        "attention_mask": torch.long,
        "special_tokens_mask": torch.long,
        "tone_ids": torch.long,
        "tone_mask": torch.bool,
        "letter_ids": torch.long,
        "letter_mask": torch.bool,
    }
    batch: dict[str, Any] = {
        key: torch.tensor([row[key] for row in rows], dtype=dtype)
        for key, dtype in tensor_fields.items()
    }
    batch["sample_ids"] = [item.sample_id for item in inputs]
    batch["conditions"] = [item.condition for item in inputs]
    return batch


@dataclass
class FrozenUnmarkPathway:
    """A frozen encoder plus a frozen Stage-1 adapter bound to one UNMARK arm."""

    arm: Stage2UnmarkArm
    checkpoint_binding: Stage2FinalistBinding
    encoder: Any
    adapter: Any
    hidden_size: int = HIDDEN_SIZE

    @property
    def finalist(self) -> FinalistIdentity:
        return finalist_for_arm(self.arm)

    @property
    def checkpoint_sha256(self) -> str:
        return self.checkpoint_binding.checkpoint_sha256

    def require_frozen(self, *, check_values: bool = False) -> None:
        require_frozen_unmark_pathway(self, check_values=check_values)


def load_stage2_phobert_components(*, cache_dir: str | Path | None = None) -> tuple[Any, Any]:
    """Load the pinned downstream tokenizer and encoder, then freeze the encoder.

    This function is not called by tests and performs no work at import time.
    """

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT,
        revision=ENCODER_REVISION,
        use_fast=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    encoder = AutoModel.from_pretrained(
        ENCODER_CHECKPOINT,
        revision=ENCODER_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    setattr(encoder, "_unmark_requested_revision", ENCODER_REVISION)
    _freeze_module(encoder)
    return tokenizer, encoder


def _freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _require_single_parameter_device(module: Any, module_name: str) -> Any:
    devices: dict[str, list[str]] = {}
    first_device = None
    for name, parameter in module.named_parameters():
        device = parameter.device
        devices.setdefault(str(device), []).append(name)
        if first_device is None:
            first_device = device
    if first_device is None:
        raise EvaluationContractViolation(f"{module_name} exposes no parameters")
    if len(devices) != 1:
        detail = ", ".join(
            f"{device}: {names[:5]}" for device, names in sorted(devices.items())
        )
        raise EvaluationContractViolation(
            f"{module_name} parameters are on multiple devices: {detail}"
        )
    if getattr(first_device, "type", None) == "meta":
        raise EvaluationContractViolation(
            f"{module_name} parameters are on the meta device; a real CPU/GPU "
            "device is required for Stage-2 representations"
        )
    return first_device


def _require_pathway_device(pathway: FrozenUnmarkPathway) -> Any:
    encoder_device = _require_single_parameter_device(pathway.encoder, "encoder")
    adapter_device = _require_single_parameter_device(pathway.adapter, "adapter")
    if adapter_device != encoder_device:
        raise EvaluationContractViolation(
            f"encoder and adapter devices differ: encoder={encoder_device}, "
            f"adapter={adapter_device}"
        )
    return encoder_device


def _stage2_forward_tensors_on_device(
    batch: Mapping[str, Any],
    *,
    device: Any,
) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key in STAGE2_FORWARD_TENSOR_KEYS:
        if key not in batch:
            raise EvaluationContractViolation(f"Stage-2 batch is missing {key!r}")
        value = batch[key]
        if not torch.is_tensor(value):
            raise EvaluationContractViolation(
                f"Stage-2 batch field {key!r} is {type(value).__name__}, not a tensor"
            )
        moved_value = value.to(device=device)
        if moved_value.dtype != value.dtype:
            raise EvaluationContractViolation(
                f"moving {key!r} to {device} changed dtype from {value.dtype} "
                f"to {moved_value.dtype}"
            )
        if tuple(moved_value.shape) != tuple(value.shape):
            raise EvaluationContractViolation(
                f"moving {key!r} to {device} changed shape from "
                f"{tuple(value.shape)} to {tuple(moved_value.shape)}"
            )
        if moved_value.device != device:
            raise EvaluationContractViolation(
                f"moving {key!r} to {device} produced tensor on {moved_value.device}"
            )
        moved[key] = moved_value
    return moved


def _observed_encoder_revision(encoder: Any) -> str | None:
    for holder in (encoder, getattr(encoder, "config", None)):
        for attribute in ("_commit_hash", "revision", "commit_hash", "_unmark_requested_revision"):
            value = getattr(holder, attribute, None)
            if isinstance(value, str) and value:
                return value
    return None


def require_stage2_encoder_identity(encoder: Any) -> None:
    """Require the frozen Stage-2 encoder identity and pinned revision."""

    from unmark.modeling.adapter import detect_checkpoint, resolve_position_profile

    checkpoint = detect_checkpoint(encoder)
    if checkpoint != ENCODER_CHECKPOINT:
        raise EvaluationContractViolation(
            f"Stage-2 encoder checkpoint is {checkpoint!r}, expected {ENCODER_CHECKPOINT!r}"
        )
    revision = _observed_encoder_revision(encoder)
    if revision != ENCODER_REVISION:
        raise EvaluationContractViolation(
            f"Stage-2 encoder revision is {revision!r}, expected {ENCODER_REVISION!r}. "
            "Use load_stage2_phobert_components or pass an encoder that exposes the "
            "pinned revision."
        )
    resolve_position_profile(encoder)


def _load_adapter_state(checkpoint_path: str | Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise EvaluationContractViolation(
            f"{checkpoint_path} did not deserialise to a checkpoint mapping"
        )
    adapter_state = payload.get("adapter_state")
    if not isinstance(adapter_state, Mapping):
        raise EvaluationContractViolation(f"{checkpoint_path} carries no adapter_state mapping")
    return adapter_state


def load_frozen_unmark_pathway(
    arm: str | Stage2UnmarkArm,
    checkpoint_path: str | Path,
    *,
    encoder: Any | None = None,
    inventory: Any = None,
    cache_dir: str | Path | None = None,
) -> FrozenUnmarkPathway:
    """Load one frozen UNMARK arm from an explicit Stage-1 checkpoint path."""

    require_dual_finalist_state()
    resolved = require_stage2_unmark_arm(arm)
    finalist = finalist_for_arm(resolved)
    if encoder is None:
        _, encoder = load_stage2_phobert_components(cache_dir=cache_dir)
    require_stage2_encoder_identity(encoder)
    encoder_device = _require_single_parameter_device(encoder, "encoder")

    try:
        evidence = verify_finalist_checkpoint(
            checkpoint_path, finalist, inventory=inventory
        )
    except FinalistFreezeViolation:
        raise
    binding = bind_verified_checkpoint(resolved, checkpoint_path, evidence)

    from unmark.modeling.adapter import OrthographyInputAdapter
    from unmark.modeling.config import AdapterConfig

    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=HIDDEN_SIZE))
    adapter.load_state_dict(_load_adapter_state(checkpoint_path), strict=True)
    adapter.to(device=encoder_device)
    _freeze_module(encoder)
    _freeze_module(adapter)

    pathway = FrozenUnmarkPathway(
        arm=resolved,
        checkpoint_binding=binding,
        encoder=encoder,
        adapter=adapter,
    )
    pathway.require_frozen(check_values=True)
    return pathway


def require_frozen_unmark_pathway(
    pathway: FrozenUnmarkPathway,
    *,
    check_values: bool = False,
) -> None:
    """Every encoder/adapter parameter frozen, FP32, and eval, or fail."""

    import torch

    require_stage2_encoder_identity(pathway.encoder)
    if not isinstance(pathway.arm, Stage2UnmarkArm):
        raise EvaluationContractViolation(f"unknown Stage-2 arm on pathway: {pathway.arm!r}")
    expected_binding = bind_verified_checkpoint(
        pathway.arm,
        pathway.checkpoint_binding.checkpoint_path,
        pathway.checkpoint_binding.evidence,
    )
    if expected_binding.to_dict() != pathway.checkpoint_binding.to_dict():
        raise EvaluationContractViolation("pathway checkpoint binding drifted from finalist identity")

    hidden = getattr(getattr(pathway.encoder, "config", None), "hidden_size", None)
    if hidden != HIDDEN_SIZE or pathway.hidden_size != HIDDEN_SIZE:
        raise EvaluationContractViolation(
            f"Stage-2 UNMARK hidden size must be {HIDDEN_SIZE}, got encoder={hidden!r} "
            f"pathway={pathway.hidden_size!r}"
        )
    _require_pathway_device(pathway)

    for module_name, module in (("encoder", pathway.encoder), ("adapter", pathway.adapter)):
        if getattr(module, "training", True):
            raise EvaluationContractViolation(f"{module_name} must be in eval mode")
        trainable = [
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        ]
        if trainable:
            raise EvaluationContractViolation(
                f"{module_name} parameter(s) require grad: {trainable[:5]}"
            )
        for name, parameter in module.named_parameters():
            if parameter.dtype.is_floating_point and parameter.dtype is not torch.float32:
                raise EvaluationContractViolation(
                    f"{module_name}.{name} dtype is {parameter.dtype}, expected torch.float32"
                )
            if check_values and parameter.dtype.is_floating_point and not bool(
                torch.isfinite(parameter).all()
            ):
                raise EvaluationContractViolation(
                    f"{module_name}.{name} contains NaN or Inf"
                )

    adapter_keys = tuple(sorted(pathway.adapter.state_dict()))
    if adapter_keys != ADAPTER_STATE_KEYS:
        raise EvaluationContractViolation(
            f"adapter state keys drifted: {list(adapter_keys)} != {list(ADAPTER_STATE_KEYS)}"
        )
    adapter_parameters = sum(int(p.numel()) for p in pathway.adapter.parameters())
    if adapter_parameters != ADAPTER_TRAINABLE_PARAMETERS:
        raise EvaluationContractViolation(
            f"adapter has {adapter_parameters} parameters, expected {ADAPTER_TRAINABLE_PARAMETERS}"
        )


def stage2_first_token_representation(hidden_states: "Tensor") -> "Tensor":
    """Return position 0 from ``[batch, length, 768]`` hidden states."""

    if hidden_states.dim() != 3:
        raise EvaluationContractViolation(
            f"expected [batch, length, hidden] hidden states, got {tuple(hidden_states.shape)}"
        )
    if int(hidden_states.shape[-1]) != HIDDEN_SIZE:
        raise EvaluationContractViolation(
            f"Stage-2 representations must be 768-dimensional, got {hidden_states.shape[-1]}"
        )
    return hidden_states[:, 0, :].contiguous()


def extract_stage2_unmark_representations(
    pathway: FrozenUnmarkPathway,
    batch: Mapping[str, "Tensor"],
    **encoder_kwargs: Any,
) -> "Tensor":
    """Frozen UNMARK first-token representations, detached FP32 ``[B, 768]``."""

    import torch
    from unmark.modeling.adapter import authoritative_position_ids, base_word_embeddings

    pathway.require_frozen()
    device = _require_pathway_device(pathway)
    forward_batch = _stage2_forward_tensors_on_device(batch, device=device)
    with torch.no_grad():
        input_ids = forward_batch["input_ids"]
        z = pathway.adapter(
            base_word_embeddings(pathway.encoder, input_ids),
            forward_batch["tone_ids"],
            forward_batch["tone_mask"],
            forward_batch["letter_ids"],
            forward_batch["letter_mask"],
        )
        outputs = pathway.encoder(
            inputs_embeds=z,
            attention_mask=forward_batch["attention_mask"],
            position_ids=authoritative_position_ids(pathway.encoder, input_ids),
            **encoder_kwargs,
        )
        hidden = getattr(outputs, "last_hidden_state", outputs)
        pooled = stage2_first_token_representation(hidden)
    return pooled.detach().to(torch.float32)


def stage2_head_trainable_parameters(pathway: FrozenUnmarkPathway, head: Any) -> tuple[Any, ...]:
    """Return future head parameters only; encoder and adapter can never leak in."""

    pathway.require_frozen()
    frozen_ids = {
        id(parameter)
        for module in (pathway.encoder, pathway.adapter)
        for parameter in module.parameters()
    }
    selected = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    leaked = [parameter for parameter in selected if id(parameter) in frozen_ids]
    if leaked:
        raise EvaluationContractViolation(
            "head parameter enumeration includes frozen encoder/adapter parameter objects"
        )
    if not selected:
        raise EvaluationContractViolation("future Stage-2 head has no trainable parameters")
    return selected


__all__ = [
    "STAGE2_A_B_SELECTION_IMPLEMENTED",
    "STAGE2_DATASET",
    "STAGE2_DATASET_VERSION",
    "STAGE2_DUAL_FINALIST_INFRA_SCHEMA_VERSION",
    "STAGE2_EXCLUDED_CONDITIONS",
    "STAGE2_FIRST_TOKEN_POOLING",
    "STAGE2_HEAD_TRAINING_IMPLEMENTED",
    "STAGE2_MEASUREMENT_DEV_SELECTION_IMPLEMENTED",
    "STAGE2_OFFICIAL_TEST_REACHABLE",
    "STAGE2_REPRESENTATION_DTYPE",
    "STAGE2_NUM_LABELS",
    "STAGE2_TASK",
    "STAGE2_TOKENIZATION",
    "STAGE2_TRAINING_STARTED",
    "STAGE2_UNMARK_ARM_NAMES",
    "STAGE2_UNMARK_CONDITIONS",
    "FrozenUnmarkPathway",
    "Stage2FinalistBinding",
    "Stage2UnmarkArm",
    "Stage2UnmarkInput",
    "apply_stage2_corruption",
    "bind_verified_checkpoint",
    "collate_stage2_unmark_batch",
    "extract_stage2_unmark_representations",
    "finalist_for_arm",
    "load_frozen_unmark_pathway",
    "load_stage2_phobert_components",
    "prepare_stage2_unmark_condition_grid",
    "prepare_stage2_unmark_input",
    "require_dual_finalist_state",
    "require_frozen_unmark_pathway",
    "require_same_base_grid",
    "require_stage2_condition",
    "require_stage2_encoder_identity",
    "require_stage2_unmark_arm",
    "stage2_first_token_representation",
    "stage2_head_trainable_parameters",
]
