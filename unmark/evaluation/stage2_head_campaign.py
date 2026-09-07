"""Stage-2 dual-finalist head campaign: cache, head runner, artifacts, reporting.

Audit 051. Audit 049 froze the protocol; Audit 050 accepted the frozen UNMARK
runtime path. This module adds the machinery that sits *around* that path:

* an immutable, arm-bound representation cache;
* a head-training runner that works **entirely over cached representations**;
* checkpoint selection on clean `FULL` protocol-dev;
* fail-closed head artifacts;
* a 2-arm x 5-seed campaign plan;
* a measurement/reporting path over a frozen selected head.

**There is no A/B selection here, and there is no place to add one.** Audit 049
adopted option (c): both finalists are carried and separately reported, and the
downstream task may not choose between them. Accordingly this module defines no
winner rule, no ranking-for-selection helper and no "best arm" state, and
:func:`aggregate_stage2_campaign` refuses a report that is missing either arm.

**What is reused rather than re-implemented.** The scientific primitives all come
from the closed pre-G1 protocol: :func:`build_head`, :func:`build_optimizer`,
:func:`deterministic_batches`, :class:`EpochScore`, :func:`select_checkpoint`,
:func:`require_full_schedule`, :func:`score_predictions`, and the
:class:`Preg1Role` enum -- which is reused deliberately because it has no
``OFFICIAL_TEST`` member, so official TEST cannot be *named* by this module
either. The frozen forward pass comes from
:func:`extract_stage2_unmark_representations` (Audit 050) and is never duplicated.

**One value is deliberately not defaulted.** The Stage-2 measurement corruption
seed is *not* pinned by `docs/spec/stage2-dual-finalist-protocol.json`, which
fixes the determinism mechanism but no seed value. It is therefore a required
argument with no default: :func:`stage2_measurement_extraction_plan` fails closed
without it. Choosing it is a scientific decision for the author, not an
implementation detail to be invented here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.metrics import per_class_scores
from unmark.evaluation.preg1_head import (
    EpochScore,
    Preg1Role,
    build_head,
    build_optimizer,
    deterministic_batches,
    ordered_id_digest,
    require_full_schedule,
    require_protocol_settings,
    score_predictions,
    select_checkpoint,
)
from unmark.evaluation.preg1_protocol import (
    BATCH_SIZE,
    EPOCHS,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_NUM_LABELS,
    PRIMARY_TASK,
    TRUNCATION,
)
from unmark.evaluation.stage2_dual_finalist import (
    STAGE2_FIRST_TOKEN_POOLING,
    extract_stage2_unmark_representations,
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_CONDITIONS,
    Stage2UnmarkArm,
    finalist_for_arm,
    require_dual_finalist_state,
    require_stage2_condition,
    require_stage2_unmark_arm,
)
from unmark.stage1.checkpoint import atomic_write_bytes
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE

STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION = "stage2-head-campaign-v1"

STAGE2_PROTOCOL_SPEC_PATH = (
    Path(__file__).resolve().parents[2] / "docs/spec/stage2-dual-finalist-protocol.json"
)
STAGE2_PROTOCOL_VERSION = "stage2-dual-finalist-protocol-v1"

def require_frozen_protocol_spec() -> dict[str, Any]:
    """Load the committed Stage-2 freeze and refuse a drifted protocol version.

    The runner must not silently outlive the protocol it implements: if
    `docs/spec/stage2-dual-finalist-protocol.json` is ever re-versioned, every
    cache key this module writes would still claim the old version.
    """
    if not STAGE2_PROTOCOL_SPEC_PATH.is_file():
        raise EvaluationContractViolation(
            f"frozen Stage-2 protocol spec is missing: {STAGE2_PROTOCOL_SPEC_PATH}"
        )
    payload = json.loads(STAGE2_PROTOCOL_SPEC_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STAGE2_PROTOCOL_VERSION:
        raise EvaluationContractViolation(
            f"frozen protocol spec is {payload.get('schema_version')!r} but this runner "
            f"implements {STAGE2_PROTOCOL_VERSION!r}"
        )
    return payload


STAGE2_HEAD_LEARNING_RATE = 0.01
"""Frozen by D-S2-001, inherited from the closed pre-G1 diagnostic. Not tunable."""

STAGE2_CLEAN_CONDITION = "FULL"
"""The only condition that may reach head training or checkpoint selection."""

STAGE2_DEGRADED_CONDITIONS: tuple[str, ...] = tuple(
    c for c in STAGE2_UNMARK_CONDITIONS if c != STAGE2_CLEAN_CONDITION
)
"""`P25, P50, P75, P100, STRIP_ALL` -- the equal-weight reporting summary set."""

STAGE2_CAMPAIGN_ARM_COUNT = 2
STAGE2_CAMPAIGN_SEEDS: tuple[int, ...] = tuple(MEASUREMENT_SEEDS)
STAGE2_CAMPAIGN_RUN_COUNT = STAGE2_CAMPAIGN_ARM_COUNT * len(STAGE2_CAMPAIGN_SEEDS)

# --- selection-safety declarations, asserted by tests ------------------------
STAGE2_AB_WINNER_RULE = None
STAGE2_AB_TIE_BREAK = None
STAGE2_AB_SELECTION_IMPLEMENTED = False
STAGE2_MEASUREMENT_MAY_SELECT = False
STAGE2_OFFICIAL_TEST_ROLE_EXISTS = False
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED = False
"""The measurement corruption seed is NOT pinned by the frozen protocol.

Recorded as a fact rather than resolved by a default: see the module docstring.
"""


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
STAGE2_TRAINING_ROLE = Preg1Role.PROTOCOL_TRAIN
STAGE2_SELECTION_ROLE = Preg1Role.PROTOCOL_DEV
STAGE2_MEASUREMENT_ROLE = Preg1Role.OFFICIAL_VALIDATION


def require_role(role: Preg1Role, expected: Preg1Role, what: str) -> None:
    """Fail closed unless a bound role is exactly the one this step permits."""
    if role is not expected:
        raise EvaluationContractViolation(
            f"{what} requires role {expected.value!r}, got {role.value!r}. Roles are "
            "read from cache provenance; there is no argument that declares one."
        )


def require_clean_condition(condition: str, what: str) -> None:
    """Head training and checkpoint selection see clean `FULL` only."""
    if condition != STAGE2_CLEAN_CONDITION:
        raise EvaluationContractViolation(
            f"{what} may only use the clean {STAGE2_CLEAN_CONDITION!r} condition, got "
            f"{condition!r}. A corrupted protocol-dev score must never influence head "
            "checkpoint selection: that would tune the head for robustness and break "
            "the 'trained on clean, then frozen' contract."
        )


def label_digest(labels: Sequence[int]) -> str:
    """Digest of labels **in order**, so a re-ordered label vector is caught."""
    if not labels:
        raise EvaluationContractViolation("cannot digest an empty label vector")
    return hashlib.sha256("\n".join(str(int(v)) for v in labels).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Representation cache
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage2RepresentationKey:
    """Everything a cached Stage-2 representation tensor is bound to.

    Modelled on `preg1_head.RepresentationKey` and deliberately *wider*: a
    Stage-2 tensor additionally depends on which arm produced it, which frozen
    Stage-1 checkpoint that arm carries, which corruption condition and seed the
    text went through, and which commit and protocol version were in force.
    Reusing one arm's vectors as the other would make the two arms look
    identical -- a plausible-looking non-result -- so comparison is exact and a
    mismatch is an error, never a recomputation.

    **No raw text.** Sample identity travels as an ordered-id digest.
    """

    repository_head: str
    arm: str
    finalist_checkpoint_sha256: str
    backbone_checkpoint: str
    backbone_revision: str
    protocol_version: str
    dataset: str
    dataset_version: str
    task: str
    role: str
    condition: str
    corruption_seed: int | None
    pooling: str
    max_length: int
    truncation: bool
    padding: str
    ordered_id_digest: str
    label_digest: str
    dtype: str
    hidden_size: int
    count: int
    schema_version: str = STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_stage2_unmark_arm(self.arm)
        require_stage2_condition(self.condition)
        Preg1Role(self.role)
        if self.condition == STAGE2_CLEAN_CONDITION and self.corruption_seed is not None:
            raise EvaluationContractViolation(
                f"{STAGE2_CLEAN_CONDITION} is the clean condition and must carry no "
                f"corruption seed, got {self.corruption_seed!r}"
            )
        if self.condition != STAGE2_CLEAN_CONDITION and self.corruption_seed is None:
            raise EvaluationContractViolation(
                f"condition {self.condition!r} is corrupted and must bind the corruption "
                "seed that produced it; the seed is not defaulted anywhere"
            )
        if self.pooling != STAGE2_FIRST_TOKEN_POOLING:
            raise EvaluationContractViolation(
                f"Stage-2 pooling is frozen to {STAGE2_FIRST_TOKEN_POOLING!r} (D-S2-001), "
                f"got {self.pooling!r}"
            )
        if self.hidden_size != HIDDEN_SIZE:
            raise EvaluationContractViolation(
                f"Stage-2 representations are {HIDDEN_SIZE}-dimensional, got {self.hidden_size}"
            )
        if self.dtype != STAGE2_REPRESENTATION_DTYPE:
            raise EvaluationContractViolation(
                f"Stage-2 representations are {STAGE2_REPRESENTATION_DTYPE} (no AMP), "
                f"got {self.dtype!r}"
            )
        if self.count <= 0:
            raise EvaluationContractViolation(f"cache count must be positive, got {self.count}")
        for name in ("repository_head", "finalist_checkpoint_sha256",
                     "ordered_id_digest", "label_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationContractViolation(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository_head": self.repository_head,
            "arm": self.arm,
            "finalist_checkpoint_sha256": self.finalist_checkpoint_sha256,
            "backbone_checkpoint": self.backbone_checkpoint,
            "backbone_revision": self.backbone_revision,
            "protocol_version": self.protocol_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "role": self.role,
            "condition": self.condition,
            "corruption_seed": self.corruption_seed,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "ordered_id_digest": self.ordered_id_digest,
            "label_digest": self.label_digest,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "count": self.count,
            "representation_shape": [self.count, self.hidden_size],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage2RepresentationKey":
        try:
            return cls(
                repository_head=payload["repository_head"],
                arm=payload["arm"],
                finalist_checkpoint_sha256=payload["finalist_checkpoint_sha256"],
                backbone_checkpoint=payload["backbone_checkpoint"],
                backbone_revision=payload["backbone_revision"],
                protocol_version=payload["protocol_version"],
                dataset=payload["dataset"],
                dataset_version=payload["dataset_version"],
                task=payload["task"],
                role=payload["role"],
                condition=payload["condition"],
                corruption_seed=payload["corruption_seed"],
                pooling=payload["pooling"],
                max_length=payload["max_length"],
                truncation=payload["truncation"],
                padding=payload["padding"],
                ordered_id_digest=payload["ordered_id_digest"],
                label_digest=payload["label_digest"],
                dtype=payload["dtype"],
                hidden_size=payload["hidden_size"],
                count=payload["count"],
                schema_version=payload["schema_version"],
            )
        except (KeyError, ValueError) as error:
            raise EvaluationContractViolation(
                f"Stage-2 cache metadata is malformed or from an unknown schema: {error}"
            ) from error

    def require_compatible(self, other: "Stage2RepresentationKey") -> None:
        """Exact match on every field, or fail. No tolerance, no coercion."""
        mine, theirs = self.to_dict(), other.to_dict()
        differences = [name for name in mine if mine[name] != theirs[name]]
        if differences:
            detail = ", ".join(
                f"{name}: cached={theirs[name]!r} wanted={mine[name]!r}"
                for name in differences
            )
            raise EvaluationContractViolation(
                f"Stage-2 representation cache is incompatible on {len(differences)} "
                f"field(s): {detail}. Refusing to reuse it -- a cache reused across arms, "
                "commits, protocols, roles or conditions produces a silent, "
                "plausible-looking result."
            )


@dataclass(frozen=True)
class Stage2BoundRepresentations:
    """A representation tensor **and** the provenance saying what it is.

    Role, arm and condition are properties of the tensor, read from the key that
    was validated when it was produced or loaded. There is no separate argument
    anywhere by which a caller can declare them, so a measurement tensor cannot
    be handed to checkpoint selection under a `PROTOCOL_DEV` label.
    """

    values: Any
    key: Stage2RepresentationKey

    def __post_init__(self) -> None:
        shape = tuple(getattr(self.values, "shape", ()))
        if shape != (self.key.count, self.key.hidden_size):
            raise EvaluationContractViolation(
                f"representation shape {shape} contradicts its key's "
                f"{(self.key.count, self.key.hidden_size)}"
            )
        dtype = str(getattr(self.values, "dtype", ""))
        if dtype != self.key.dtype:
            raise EvaluationContractViolation(
                f"representation dtype {dtype!r} contradicts its key's {self.key.dtype!r}"
            )

    @property
    def role(self) -> Preg1Role:
        return Preg1Role(self.key.role)

    @property
    def arm(self) -> Stage2UnmarkArm:
        return require_stage2_unmark_arm(self.key.arm)

    @property
    def condition(self) -> str:
        return self.key.condition

    def require_role(self, expected: Preg1Role, what: str) -> None:
        require_role(self.role, expected, what)

    def require_same_arm(self, other: "Stage2BoundRepresentations") -> None:
        if self.key.arm != other.key.arm:
            raise EvaluationContractViolation(
                f"cross-arm use: {self.key.arm!r} with {other.key.arm!r}. A head must be "
                "trained and selected on one arm's representations only."
            )
        if self.key.finalist_checkpoint_sha256 != other.key.finalist_checkpoint_sha256:
            raise EvaluationContractViolation(
                "same arm name but different finalist checkpoint sha256"
            )


class Stage2RepresentationCache:
    """Immutable fail-closed cache for frozen Stage-2 UNMARK representations.

    Stores `[N, 768]` FP32 first-token vectors beside their key. The tensor file
    is an experiment resource and is **never committed to git** (`*.pt` is
    git-ignored). Metadata is written atomically with the repository's own
    `atomic_write_bytes`; the tensor is written to a temporary path and renamed,
    so a crash mid-write leaves the previous cache intact rather than a
    truncated file that a later run would load.
    """

    METADATA_NAME = "stage2-representation-key.json"
    TENSOR_NAME = "stage2-representations.pt"
    TEMP_SUFFIX = ".tmp"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def metadata_path(self) -> Path:
        return self.directory / self.METADATA_NAME

    @property
    def tensor_path(self) -> Path:
        return self.directory / self.TENSOR_NAME

    def exists(self) -> bool:
        return self.metadata_path.is_file() and self.tensor_path.is_file()

    def read_key(self) -> Stage2RepresentationKey:
        if not self.metadata_path.is_file():
            raise EvaluationContractViolation(f"no Stage-2 cache metadata at {self.metadata_path}")
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EvaluationContractViolation(
                f"Stage-2 cache metadata is not valid JSON: {error}"
            ) from error
        return Stage2RepresentationKey.from_dict(payload)

    def save(self, key: Stage2RepresentationKey, representations: Any) -> None:
        """Write once. An existing cache may only be rewritten byte-identically.

        Immutability is the point: a completed cache is evidence. Re-saving the
        same key is tolerated (idempotent re-extraction); re-saving a *different*
        key into the same directory is refused rather than silently overwriting
        another arm's or another commit's vectors.
        """
        import torch

        _require_fp32_matrix(representations, key)
        if self.exists():
            key.require_compatible(self.read_key())
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            self.metadata_path,
            (json.dumps(key.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        temp = self.tensor_path.with_name(self.tensor_path.name + self.TEMP_SUFFIX)
        torch.save(representations, temp)
        temp.replace(self.tensor_path)

    def load(self, key: Stage2RepresentationKey) -> Stage2BoundRepresentations:
        """Load only if the stored key matches **exactly**."""
        import torch

        key.require_compatible(self.read_key())
        tensor = torch.load(self.tensor_path, map_location="cpu")
        _require_fp32_matrix(tensor, key)
        return Stage2BoundRepresentations(values=tensor, key=key)


def _require_fp32_matrix(tensor: Any, key: Stage2RepresentationKey) -> None:
    shape = tuple(getattr(tensor, "shape", ()))
    if shape != (key.count, key.hidden_size):
        raise EvaluationContractViolation(
            f"representation tensor {shape} does not match key {(key.count, key.hidden_size)}"
        )
    dtype = str(getattr(tensor, "dtype", ""))
    if dtype != key.dtype:
        raise EvaluationContractViolation(
            f"representation dtype {dtype!r} is not the frozen {key.dtype!r}"
        )


# ---------------------------------------------------------------------------
# Extraction plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage2ExtractionRequest:
    """One (arm, role, condition) extraction the campaign needs."""

    arm: str
    role: str
    condition: str
    corruption_seed: int | None
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "role": self.role,
            "condition": self.condition,
            "corruption_seed": self.corruption_seed,
            "purpose": self.purpose,
        }


def stage2_training_extraction_plan() -> tuple[Stage2ExtractionRequest, ...]:
    """Clean `FULL` protocol-train and protocol-dev, for both arms. Four items."""
    require_frozen_protocol_spec()
    require_dual_finalist_state()
    requests = []
    for arm in Stage2UnmarkArm:
        requests.append(Stage2ExtractionRequest(
            arm=arm.value, role=STAGE2_TRAINING_ROLE.value,
            condition=STAGE2_CLEAN_CONDITION, corruption_seed=None,
            purpose="head_training"))
        requests.append(Stage2ExtractionRequest(
            arm=arm.value, role=STAGE2_SELECTION_ROLE.value,
            condition=STAGE2_CLEAN_CONDITION, corruption_seed=None,
            purpose="head_checkpoint_selection"))
    return tuple(requests)


def stage2_measurement_extraction_plan(
    *, corruption_seed: int
) -> tuple[Stage2ExtractionRequest, ...]:
    """measurement-dev x six conditions x two arms. Twelve items.

    `corruption_seed` has **no default**. The frozen protocol
    (`docs/spec/stage2-dual-finalist-protocol.json`) pins the determinism
    mechanism -- keyed by `sample_id` through `unmark.corruption.corrupt` -- but
    pins no seed value. Inventing one here would be an unrecorded scientific
    choice, so the caller must supply it and it is bound into every cache key.
    """
    if isinstance(corruption_seed, bool) or not isinstance(corruption_seed, int):
        raise EvaluationContractViolation(
            "the Stage-2 measurement corruption seed must be an explicit integer; it is "
            "not pinned by the frozen protocol and must not acquire a default"
        )
    require_frozen_protocol_spec()
    require_dual_finalist_state()
    requests = []
    for arm in Stage2UnmarkArm:
        for condition in STAGE2_UNMARK_CONDITIONS:
            requests.append(Stage2ExtractionRequest(
                arm=arm.value,
                role=STAGE2_MEASUREMENT_ROLE.value,
                condition=condition,
                corruption_seed=None if condition == STAGE2_CLEAN_CONDITION else corruption_seed,
                purpose="measurement_reporting"))
    return tuple(requests)


def stage2_representation_key_for(
    request: Stage2ExtractionRequest,
    *,
    repository_head: str,
    ordered_ids: Sequence[str],
    labels: Sequence[int],
) -> Stage2RepresentationKey:
    """Build the cache key a request's tensor must be stored under."""
    finalist = finalist_for_arm(request.arm)
    return Stage2RepresentationKey(
        repository_head=repository_head,
        arm=request.arm,
        finalist_checkpoint_sha256=finalist.checkpoint_sha256,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=STAGE2_PROTOCOL_VERSION,
        dataset=PRIMARY_DATASET,
        dataset_version=PRIMARY_DATASET_VERSION,
        task=PRIMARY_TASK,
        role=request.role,
        condition=request.condition,
        corruption_seed=request.corruption_seed,
        pooling=STAGE2_FIRST_TOKEN_POOLING,
        max_length=MAX_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING,
        ordered_id_digest=ordered_id_digest(list(ordered_ids)),
        label_digest=label_digest(labels),
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=len(ordered_ids),
    )


def extract_and_cache_stage2_representations(
    pathway: Any,
    batches: Sequence[Mapping[str, Any]],
    key: Stage2RepresentationKey,
    cache: Stage2RepresentationCache,
) -> Stage2BoundRepresentations:
    """Run the **accepted Audit-050 pathway** over batches and cache the result.

    This is a driver, not a second forward pass: the encoder/adapter call is
    :func:`extract_stage2_unmark_representations` and is never re-implemented
    here. Its job is to concatenate per-batch `[b, 768]` outputs in order, check
    the row count against the key, and store the tensor under a key that binds
    the arm, checkpoint, commit, role, condition and corruption seed.

    The arm the pathway actually carries is checked against the key's arm, so a
    pathway loaded from one finalist cannot write the other's cache.
    """
    # Identity guards run BEFORE torch is imported: a cross-arm write should be
    # refused on any machine, including one with no torch installed.
    binding = getattr(pathway, "binding", None)
    if binding is not None:
        if binding.arm.value != key.arm:
            raise EvaluationContractViolation(
                f"pathway carries arm {binding.arm.value!r} but the cache key is "
                f"{key.arm!r}; one arm may never write the other's cache"
            )
        if binding.checkpoint_sha256 != key.finalist_checkpoint_sha256:
            raise EvaluationContractViolation(
                "pathway checkpoint sha256 does not match the cache key"
            )
    if not batches:
        raise EvaluationContractViolation("no batches supplied for extraction")

    import torch

    pooled = [extract_stage2_unmark_representations(pathway, batch) for batch in batches]
    values = torch.cat(pooled, dim=0)
    if int(values.shape[0]) != key.count:
        raise EvaluationContractViolation(
            f"extracted {int(values.shape[0])} rows but the key declares {key.count}"
        )
    cache.save(key, values)
    return cache.load(key)


# ---------------------------------------------------------------------------
# Head training over cached representations
# ---------------------------------------------------------------------------
def require_head_only_optimizer(optimizer: Any, head: Any) -> None:
    """Exactly the head's parameters are optimised. Nothing else can enter.

    Checked by parameter **object identity**, not by name or count: a frozen
    encoder or adapter tensor that reached a param group would train silently and
    every downstream number would still look reasonable.
    """
    head_ids = {id(p) for p in head.parameters()}
    optimised = [p for group in optimizer.param_groups for p in group["params"]]
    if not optimised:
        raise EvaluationContractViolation("optimizer has no parameters")
    foreign = [p for p in optimised if id(p) not in head_ids]
    if foreign:
        raise EvaluationContractViolation(
            f"{len(foreign)} optimiser parameter(s) do not belong to the head. Only "
            "classification-head parameters may train; encoder and adapter are frozen."
        )
    missing = head_ids - {id(p) for p in optimised}
    if missing:
        raise EvaluationContractViolation(
            f"{len(missing)} head parameter(s) are absent from the optimiser"
        )


@dataclass(frozen=True)
class Stage2HeadRun:
    """One completed head run: identity, full history, and the selected epoch."""

    arm: str
    seed: int
    learning_rate: float
    epochs: int
    scores: tuple[EpochScore, ...]
    selected: EpochScore
    train_key: Stage2RepresentationKey
    dev_key: Stage2RepresentationKey
    initial_head_fingerprint: str
    selected_head_state_sha256: str
    history_digest: str
    selected_head_state: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
            "arm": self.arm,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "selected_epoch": self.selected.epoch,
            "selected_macro_f1": self.selected.macro_f1,
            "selected_accuracy": self.selected.accuracy,
            "history": [s.to_dict() for s in self.scores],
            "history_digest": self.history_digest,
            "initial_head_fingerprint": self.initial_head_fingerprint,
            "selected_head_state_sha256": self.selected_head_state_sha256,
            "train_cache_key": self.train_key.to_dict(),
            "protocol_dev_cache_key": self.dev_key.to_dict(),
        }


def _state_digest(state: Mapping[str, Any]) -> str:
    """Order-independent digest of a head state dict, via exact float repr."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        values = state[name].detach().reshape(-1).tolist()
        digest.update(",".join(repr(float(v)) for v in values).encode("utf-8"))
    return digest.hexdigest()


def history_digest(scores: Sequence[EpochScore]) -> str:
    payload = json.dumps([s.to_dict() for s in scores], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_stage2_head(
    train: Stage2BoundRepresentations,
    train_labels: Sequence[int],
    dev: Stage2BoundRepresentations,
    dev_labels: Sequence[int],
    *,
    seed: int,
    learning_rate: float = STAGE2_HEAD_LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Stage2HeadRun:
    """Train one head for **all** `epochs` over cached representations.

    No encoder and no adapter are touched: this function takes tensors, never a
    pathway, so a cached-epoch loop cannot accidentally re-run PhoBERT. Every
    epoch executes; there is no early stopping. Selection happens afterwards,
    from the returned scores, on clean `FULL` protocol-dev only.

    The epoch loop is the same shape as `preg1_head.train_head` and uses the same
    locked primitives; it is written once here because the Stage-2 role types are
    arm-bound and cannot be expressed by the pre-G1 `RepresentationKey`. A
    regression test asserts the two produce identical scores on identical inputs.
    """
    import torch
    from torch import nn

    require_protocol_settings()
    require_dual_finalist_state()

    train.require_role(STAGE2_TRAINING_ROLE, "Stage-2 head training")
    dev.require_role(STAGE2_SELECTION_ROLE, "Stage-2 head checkpoint selection")
    require_clean_condition(train.condition, "Stage-2 head training")
    require_clean_condition(dev.condition, "Stage-2 head checkpoint selection")
    train.require_same_arm(dev)

    if learning_rate != STAGE2_HEAD_LEARNING_RATE:
        raise EvaluationContractViolation(
            f"Stage-2 head LR is frozen at {STAGE2_HEAD_LEARNING_RATE} (D-S2-001); "
            f"got {learning_rate!r}. There is no Stage-2 LR pilot."
        )
    if epochs != EPOCHS:
        raise EvaluationContractViolation(
            f"Stage-2 epoch budget is frozen at {EPOCHS}; got {epochs!r}"
        )
    if seed not in STAGE2_CAMPAIGN_SEEDS:
        raise EvaluationContractViolation(
            f"seed {seed!r} is not one of the frozen measurement seeds "
            f"{list(STAGE2_CAMPAIGN_SEEDS)}"
        )

    features, dev_features = train.values, dev.values
    if features.shape[0] != len(train_labels):
        raise EvaluationContractViolation("train features and labels differ in length")
    if dev_features.shape[0] != len(dev_labels):
        raise EvaluationContractViolation("dev features and labels differ in length")
    if train.key.label_digest != label_digest(train_labels):
        raise EvaluationContractViolation("train labels do not match the cached label digest")
    if dev.key.label_digest != label_digest(dev_labels):
        raise EvaluationContractViolation("dev labels do not match the cached label digest")

    head = build_head(int(features.shape[1]), seed)
    initial_fingerprint = _state_digest(head.state_dict())
    optimizer = build_optimizer(head, learning_rate)
    require_head_only_optimizer(optimizer, head)
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
            loss = loss_fn(head(features[index]), train_y[index])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            predictions = head(dev_features).argmax(dim=1).tolist()
        f1, acc = score_predictions(predictions, dev_y.tolist())
        scores.append(EpochScore(epoch=epoch, macro_f1=f1, accuracy=acc))
        snapshots[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}

    require_full_schedule(scores, epochs)
    selected = select_checkpoint(scores)
    selected_state = snapshots[selected.epoch]
    return Stage2HeadRun(
        arm=train.key.arm,
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs,
        scores=tuple(scores),
        selected=selected,
        train_key=train.key,
        dev_key=dev.key,
        initial_head_fingerprint=initial_fingerprint,
        selected_head_state_sha256=_state_digest(selected_state),
        history_digest=history_digest(scores),
        selected_head_state=selected_state,
    )


# ---------------------------------------------------------------------------
# Head artifact
# ---------------------------------------------------------------------------
STAGE2_HEAD_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version", "arm", "finalist_checkpoint_sha256", "repository_head",
    "protocol_version", "seed", "initial_head_fingerprint", "head_architecture",
    "optimizer", "learning_rate", "epochs", "early_stopping", "selected_epoch",
    "selected_macro_f1", "selected_accuracy", "selected_head_state_sha256",
    "train_cache_key", "protocol_dev_cache_key", "history_digest", "precision",
    "selection_role", "selection_condition", "measurement_used_for_selection",
    "ab_selection_performed",
)


def build_stage2_head_artifact(run: Stage2HeadRun, *, repository_head: str) -> dict[str, Any]:
    """The immutable record of one completed head run. Partial binding is refused."""
    finalist = finalist_for_arm(run.arm)
    if run.train_key.repository_head != repository_head:
        raise EvaluationContractViolation(
            "run caches were produced under a different repository head"
        )
    artifact = {
        "schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
        "arm": run.arm,
        "finalist_checkpoint_sha256": finalist.checkpoint_sha256,
        "repository_head": repository_head,
        "protocol_version": STAGE2_PROTOCOL_VERSION,
        "seed": run.seed,
        "initial_head_fingerprint": run.initial_head_fingerprint,
        "head_architecture": f"Linear({HIDDEN_SIZE}, {PRIMARY_NUM_LABELS}, bias=True)",
        "optimizer": "AdamW",
        "learning_rate": run.learning_rate,
        "epochs": run.epochs,
        "early_stopping": False,
        "selected_epoch": run.selected.epoch,
        "selected_macro_f1": run.selected.macro_f1,
        "selected_accuracy": run.selected.accuracy,
        "selected_head_state_sha256": run.selected_head_state_sha256,
        "train_cache_key": run.train_key.to_dict(),
        "protocol_dev_cache_key": run.dev_key.to_dict(),
        "history_digest": run.history_digest,
        "precision": STAGE2_REPRESENTATION_DTYPE,
        "selection_role": STAGE2_SELECTION_ROLE.value,
        "selection_condition": STAGE2_CLEAN_CONDITION,
        "measurement_used_for_selection": False,
        "ab_selection_performed": False,
    }
    validate_stage2_head_artifact(artifact, expected_arm=run.arm, expected_seed=run.seed)
    return artifact


def validate_stage2_head_artifact(
    artifact: Mapping[str, Any], *, expected_arm: str, expected_seed: int
) -> None:
    """Fail closed unless the artifact fully and consistently binds one run."""
    if not isinstance(artifact, Mapping):
        raise EvaluationContractViolation("head artifact is not a JSON object")
    missing = [f for f in STAGE2_HEAD_ARTIFACT_FIELDS if f not in artifact]
    if missing:
        raise EvaluationContractViolation(
            f"head artifact is missing {missing}; partial binding is refused"
        )
    unknown = sorted(set(artifact) - set(STAGE2_HEAD_ARTIFACT_FIELDS))
    if unknown:
        raise EvaluationContractViolation(
            f"head artifact carries unknown field(s) {unknown}; the schema is closed"
        )
    if artifact["schema_version"] != STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION:
        raise EvaluationContractViolation("head artifact schema version is wrong")
    arm = require_stage2_unmark_arm(artifact["arm"])
    if arm.value != expected_arm:
        raise EvaluationContractViolation(
            f"head artifact is arm {arm.value!r}, expected {expected_arm!r}. An "
            "artifact from one arm may never load as the other."
        )
    if artifact["seed"] != expected_seed:
        raise EvaluationContractViolation(
            f"head artifact seed {artifact['seed']!r} != expected {expected_seed!r}"
        )
    finalist = finalist_for_arm(arm)
    if artifact["finalist_checkpoint_sha256"] != finalist.checkpoint_sha256:
        raise EvaluationContractViolation(
            f"head artifact binds a checkpoint sha that is not arm {arm.value}'s"
        )
    if artifact["learning_rate"] != STAGE2_HEAD_LEARNING_RATE:
        raise EvaluationContractViolation("head artifact LR is not the frozen 0.01")
    if artifact["epochs"] != EPOCHS or artifact["early_stopping"] is not False:
        raise EvaluationContractViolation("head artifact budget or stopping rule drifted")
    if artifact["selection_role"] != STAGE2_SELECTION_ROLE.value:
        raise EvaluationContractViolation("head artifact was not selected on protocol-dev")
    if artifact["selection_condition"] != STAGE2_CLEAN_CONDITION:
        raise EvaluationContractViolation("head artifact was not selected on clean FULL")
    for flag in ("measurement_used_for_selection", "ab_selection_performed"):
        if artifact[flag] is not False:
            raise EvaluationContractViolation(f"head artifact {flag} must be false")
    for name, expected_role, expected_condition in (
        ("train_cache_key", STAGE2_TRAINING_ROLE, STAGE2_CLEAN_CONDITION),
        ("protocol_dev_cache_key", STAGE2_SELECTION_ROLE, STAGE2_CLEAN_CONDITION),
    ):
        key = Stage2RepresentationKey.from_dict(artifact[name])
        if key.role != expected_role.value:
            raise EvaluationContractViolation(f"{name} has role {key.role!r}")
        if key.condition != expected_condition:
            raise EvaluationContractViolation(f"{name} is not the clean condition")
        if key.arm != arm.value:
            raise EvaluationContractViolation(f"{name} belongs to a different arm")
        if key.repository_head != artifact["repository_head"]:
            raise EvaluationContractViolation(f"{name} was produced under another commit")


# ---------------------------------------------------------------------------
# Run store: completed runs are immutable; partial runs never resume
# ---------------------------------------------------------------------------
STAGE2_RESUME_CONTRACT = "atomic_rerun_no_midrun_resume"
"""A head run is atomic: complete, or absent.

**Why not mid-epoch resume.** One run is 30 epochs of a linear head over cached
`[N, 768]` FP32 vectors -- minutes, not hours -- and it is fully deterministic:
head init is reseeded from `seed`, batch order is a pure function of
`(seed, epoch)`, and no encoder or adapter runs. A restart therefore reproduces
the identical run bit-for-bit, so persisting optimiser state would add a resume
surface that could reattach a partial run to a drifted identity while buying no
scientific value. The smallest fail-closed alternative is chosen instead:
completed output is immutable, and anything incomplete is discarded and rerun
from scratch under a re-verified identity.
"""


class Stage2HeadRunStore:
    """Immutable per-(arm, seed) output directory with a fail-closed rerun rule."""

    ARTIFACT_NAME = "stage2-head-artifact.json"
    STATE_NAME = "stage2-selected-head.pt"
    IN_PROGRESS_NAME = "IN_PROGRESS"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def artifact_path(self) -> Path:
        return self.directory / self.ARTIFACT_NAME

    def is_complete(self) -> bool:
        return self.artifact_path.is_file()

    def read_artifact(self, *, expected_arm: str, expected_seed: int) -> dict[str, Any]:
        if not self.is_complete():
            raise EvaluationContractViolation(f"no completed head run at {self.directory}")
        payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        validate_stage2_head_artifact(
            payload, expected_arm=expected_arm, expected_seed=expected_seed
        )
        return payload

    def require_writable(self, *, expected_arm: str, expected_seed: int) -> None:
        """Refuse to overwrite completed evidence; permit a clean rerun otherwise."""
        if self.is_complete():
            raise EvaluationContractViolation(
                f"{self.directory} already holds a completed head run for arm "
                f"{expected_arm} seed {expected_seed}. Completed runs are immutable: "
                "delete it deliberately, or write elsewhere. Nothing here overwrites "
                "scientific evidence."
            )
        partial = self.directory / self.IN_PROGRESS_NAME
        if partial.is_file():
            recorded = json.loads(partial.read_text(encoding="utf-8"))
            if (recorded.get("arm"), recorded.get("seed")) != (expected_arm, expected_seed):
                raise EvaluationContractViolation(
                    f"{self.directory} holds a partial run for "
                    f"{recorded.get('arm')!r}/{recorded.get('seed')!r}, not "
                    f"{expected_arm!r}/{expected_seed!r}. "
                    f"Resume contract is {STAGE2_RESUME_CONTRACT!r}: a partial run is "
                    "discarded and rerun, never adopted under a different identity."
                )

    def mark_in_progress(self, *, arm: str, seed: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            self.directory / self.IN_PROGRESS_NAME,
            (json.dumps({"arm": arm, "seed": seed,
                         "resume_contract": STAGE2_RESUME_CONTRACT}, sort_keys=True)
             + "\n").encode("utf-8"),
        )

    def commit(self, artifact: Mapping[str, Any], selected_head_state: Mapping[str, Any]) -> None:
        """Publish a completed run, then clear the in-progress marker."""
        import torch

        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.directory / (self.STATE_NAME + ".tmp")
        torch.save(dict(selected_head_state), temp)
        temp.replace(self.directory / self.STATE_NAME)
        atomic_write_bytes(
            self.artifact_path,
            (json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        marker = self.directory / self.IN_PROGRESS_NAME
        if marker.is_file():
            marker.unlink()


# ---------------------------------------------------------------------------
# Campaign plan -- exactly two arms x five seeds, paired
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage2CampaignRun:
    arm: str
    seed: int


def stage2_campaign_plan() -> tuple[Stage2CampaignRun, ...]:
    """Exactly `2 x 5 = 10` runs. No third arm, no extra seed, no dropped seed."""
    require_dual_finalist_state()
    plan = tuple(
        Stage2CampaignRun(arm=arm.value, seed=seed)
        for seed in STAGE2_CAMPAIGN_SEEDS
        for arm in Stage2UnmarkArm
    )
    require_paired_campaign_plan(plan)
    return plan


def require_paired_campaign_plan(plan: Sequence[Stage2CampaignRun]) -> None:
    """Every seed must appear once for each arm; matched pairs stay paired."""
    if len(plan) != STAGE2_CAMPAIGN_RUN_COUNT:
        raise EvaluationContractViolation(
            f"the Stage-2 campaign is exactly {STAGE2_CAMPAIGN_RUN_COUNT} runs "
            f"({STAGE2_CAMPAIGN_ARM_COUNT} arms x {len(STAGE2_CAMPAIGN_SEEDS)} seeds), "
            f"got {len(plan)}"
        )
    seen: dict[int, set[str]] = {}
    for run in plan:
        require_stage2_unmark_arm(run.arm)
        if run.seed not in STAGE2_CAMPAIGN_SEEDS:
            raise EvaluationContractViolation(f"seed {run.seed} is not a frozen seed")
        arms = seen.setdefault(run.seed, set())
        if run.arm in arms:
            raise EvaluationContractViolation(f"duplicate run for {run.arm} seed {run.seed}")
        arms.add(run.arm)
    if sorted(seen) != sorted(STAGE2_CAMPAIGN_SEEDS):
        raise EvaluationContractViolation("campaign seeds drifted from the frozen five")
    for seed, arms in seen.items():
        if arms != set(a.value for a in Stage2UnmarkArm):
            raise EvaluationContractViolation(
                f"seed {seed} is not paired across both arms: {sorted(arms)}. A matched "
                "A/B pair may never be broken."
            )


# ---------------------------------------------------------------------------
# Measurement and reporting -- descriptive only
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage2ConditionScore:
    arm: str
    seed: int
    condition: str
    macro_f1: float
    accuracy: float
    per_class_f1: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "seed": self.seed, "condition": self.condition,
            "macro_f1": self.macro_f1, "accuracy": self.accuracy,
            "per_class_f1": list(self.per_class_f1),
        }


def measure_stage2_head(
    head: Any,
    measurement: Stage2BoundRepresentations,
    measurement_labels: Sequence[int],
    *,
    seed: int,
) -> Stage2ConditionScore:
    """Score a **frozen** selected head on measurement-dev. Reporting only.

    `measurement-dev` may change nothing: this returns numbers and holds no
    reference to any selection state. The role is read from the tensor's key, so
    a protocol-dev tensor cannot be scored here as if it were measurement, and a
    measurement tensor cannot reach checkpoint selection.
    """
    import torch

    measurement.require_role(STAGE2_MEASUREMENT_ROLE, "Stage-2 measurement")
    require_stage2_condition(measurement.condition)
    if measurement.key.label_digest != label_digest(measurement_labels):
        raise EvaluationContractViolation("measurement labels do not match the cached digest")
    head.eval()
    with torch.no_grad():
        predictions = head(measurement.values).argmax(dim=1).tolist()
    labels = [int(v) for v in measurement_labels]
    f1, acc = score_predictions(predictions, labels)
    classes = per_class_scores(predictions, labels, num_labels=PRIMARY_NUM_LABELS)
    return Stage2ConditionScore(
        arm=measurement.key.arm, seed=seed, condition=measurement.condition,
        macro_f1=f1, accuracy=acc,
        per_class_f1=tuple(c.f1 for c in classes),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    import statistics
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def aggregate_stage2_campaign(
    scores: Sequence[Stage2ConditionScore],
) -> dict[str, Any]:
    """Descriptive campaign report. **Both arms are required; neither is ranked.**

    Returns per-arm, per-condition mean and sd over the five seeds, the two
    frozen robustness summaries, and descriptive per-seed paired `B - A` deltas.
    It returns **no winner, no ranking and no recommendation**, and there is no
    argument by which one could be requested.
    """
    require_dual_finalist_state()
    arms = {s.arm for s in scores}
    expected_arms = {a.value for a in Stage2UnmarkArm}
    if arms != expected_arms:
        raise EvaluationContractViolation(
            f"a Stage-2 report must contain both arms {sorted(expected_arms)}, got "
            f"{sorted(arms)}. Audit 049 forbids dropping an arm; a single-arm report "
            "would be exactly that."
        )

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(expected_arms):
        conditions: dict[str, Any] = {}
        for condition in STAGE2_UNMARK_CONDITIONS:
            rows = [s for s in scores if s.arm == arm and s.condition == condition]
            if not rows:
                continue
            if sorted(r.seed for r in rows) != sorted(STAGE2_CAMPAIGN_SEEDS):
                raise EvaluationContractViolation(
                    f"{arm} {condition} does not cover the frozen five seeds; a seed may "
                    "never be dropped because a result looks bad"
                )
            conditions[condition] = {
                "macro_f1_mean": _mean([r.macro_f1 for r in rows]),
                "macro_f1_std": _stdev([r.macro_f1 for r in rows]),
                "accuracy_mean": _mean([r.accuracy for r in rows]),
                "accuracy_std": _stdev([r.accuracy for r in rows]),
                "per_class_f1_mean": [
                    _mean([r.per_class_f1[i] for r in rows]) for i in range(PRIMARY_NUM_LABELS)
                ],
            }
        summaries: dict[str, Any] = {}
        if STRIP := conditions.get("STRIP_ALL"):
            summaries["strip_all_macro_f1_mean"] = STRIP["macro_f1_mean"]
        degraded = [conditions[c]["macro_f1_mean"] for c in STAGE2_DEGRADED_CONDITIONS
                    if c in conditions]
        if len(degraded) == len(STAGE2_DEGRADED_CONDITIONS):
            summaries["degraded_equal_weight_macro_f1_mean"] = _mean(degraded)
        per_arm[arm] = {"conditions": conditions, "robustness_summaries": summaries}

    paired: dict[str, Any] = {}
    for condition in STAGE2_UNMARK_CONDITIONS:
        deltas = []
        for seed in STAGE2_CAMPAIGN_SEEDS:
            a = [s for s in scores
                 if s.arm == Stage2UnmarkArm.UNMARK_A.value
                 and s.seed == seed and s.condition == condition]
            b = [s for s in scores
                 if s.arm == Stage2UnmarkArm.UNMARK_B.value
                 and s.seed == seed and s.condition == condition]
            if a and b:
                deltas.append({"seed": seed,
                               "macro_f1_delta_b_minus_a": b[0].macro_f1 - a[0].macro_f1,
                               "accuracy_delta_b_minus_a": b[0].accuracy - a[0].accuracy})
        if deltas:
            paired[condition] = deltas

    return {
        "schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
        "arms": per_arm,
        "paired_deltas_b_minus_a": paired,
        "note": (
            "Descriptive report. Audit 049 option (c): both arms are carried and "
            "separately reported. These summaries must not select A or B."
        ),
        "ab_selection_performed": False,
        "winner": None,
    }


__all__ = [
    "STAGE2_AB_SELECTION_IMPLEMENTED",
    "STAGE2_AB_TIE_BREAK",
    "STAGE2_AB_WINNER_RULE",
    "STAGE2_CAMPAIGN_ARM_COUNT",
    "STAGE2_CAMPAIGN_RUN_COUNT",
    "STAGE2_CAMPAIGN_SEEDS",
    "STAGE2_CLEAN_CONDITION",
    "STAGE2_DEGRADED_CONDITIONS",
    "STAGE2_HEAD_ARTIFACT_FIELDS",
    "STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION",
    "STAGE2_HEAD_LEARNING_RATE",
    "STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED",
    "STAGE2_MEASUREMENT_MAY_SELECT",
    "STAGE2_MEASUREMENT_ROLE",
    "STAGE2_OFFICIAL_TEST_ROLE_EXISTS",
    "STAGE2_PROTOCOL_VERSION",
    "STAGE2_RESUME_CONTRACT",
    "require_frozen_protocol_spec",
    "STAGE2_SELECTION_ROLE",
    "STAGE2_TRAINING_ROLE",
    "Stage2BoundRepresentations",
    "Stage2CampaignRun",
    "Stage2ConditionScore",
    "Stage2ExtractionRequest",
    "Stage2HeadRun",
    "Stage2HeadRunStore",
    "Stage2RepresentationCache",
    "Stage2RepresentationKey",
    "aggregate_stage2_campaign",
    "build_stage2_head_artifact",
    "extract_and_cache_stage2_representations",
    "history_digest",
    "label_digest",
    "measure_stage2_head",
    "require_clean_condition",
    "require_head_only_optimizer",
    "require_paired_campaign_plan",
    "require_role",
    "stage2_campaign_plan",
    "stage2_measurement_extraction_plan",
    "stage2_representation_key_for",
    "stage2_training_extraction_plan",
    "train_stage2_head",
    "validate_stage2_head_artifact",
]
