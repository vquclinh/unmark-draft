"""Post-hoc **V2-SCF** Stage-2 clean campaign: cache, head runner, artifacts, plan.

Audit 071. `stage2_scf_pathway` supplies the frozen forward; this module is the
machinery around it -- an immutable representation cache, a head runner that
works entirely over cached tensors, checkpoint selection on clean protocol-dev,
fail-closed head artifacts, and a plan of **exactly five runs**, one per frozen
seed.

**Additive, and separate by construction.** Nothing here touches the historical
`stage2_head_campaign` / `stage2_campaign` modules. The V2-SCF cache has its own
key type, its own closed schema and its own on-disk filenames, so a historical
A/B cache directory cannot be read as a V2-SCF one and a V2-SCF directory cannot
be read as an arm's. The head artifact schema is likewise disjoint: it carries
`pathway_id`, never `arm`, and `validate_scf_head_artifact` rejects unknown
fields.

**What is reused rather than re-implemented.** Every scientific primitive comes
from the closed pre-G1 protocol, exactly as the historical Stage-2 runner uses
them: :func:`build_head`, :func:`build_optimizer`, :func:`deterministic_batches`,
:class:`EpochScore`, :func:`select_checkpoint`, :func:`require_full_schedule`,
:func:`score_predictions`, :func:`ordered_id_digest`, :func:`label_digest`, and
the :class:`Preg1Role` enum -- reused deliberately because it has **no
`OFFICIAL_TEST` member**, so official TEST cannot be named by this module either.

**The epoch loop is written once here** for the same reason the historical one
was written once in `stage2_head_campaign`: the V2-SCF role types are pathway
bound and cannot be expressed by either the pre-G1 or the arm-bound Stage-2 key.
A torch regression test asserts that this loop and `train_stage2_head` produce
**identical** scores on identical inputs, so "the same protocol" is checked
rather than claimed.

**Measurement is not here.** This module trains and selects on CLEAN
protocol-train / protocol-dev only. Reading official validation lives in
`stage2_scf_measurement`, behind an explicit gate that refuses until all five
clean heads are complete.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_CONDITIONS,
    require_stage2_condition,
)
from unmark.evaluation.stage2_head_campaign import (
    STAGE2_CLEAN_CONDITION,
    STAGE2_DEGRADED_CONDITIONS,
    STAGE2_HEAD_LEARNING_RATE,
    STAGE2_MEASUREMENT_CORRUPTION_SEED,
    STAGE2_MEASUREMENT_ROLE,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    label_digest,
    require_batch_provenance,
    require_clean_condition,
    require_head_only_optimizer,
    require_role,
)
from unmark.evaluation.stage2_scf_pathway import (
    V2_SCF_CHECKPOINT,
    V2_SCF_PATHWAY_ID,
    V2_SCF_POSTHOC_PROTOCOL_VERSION,
    FrozenScfPathway,
    ScfCheckpointIdentity,
    ScfPathwayViolation,
    extract_scf_representations,
    require_frozen_scf_protocol_spec,
    require_scf_pathway_id,
)
from unmark.stage1.checkpoint import atomic_write_bytes
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE

# ---------------------------------------------------------------------------
# Campaign shape -- exactly five seeds, no arms
# ---------------------------------------------------------------------------
V2_SCF_CAMPAIGN_SCHEMA_VERSION = "stage2-v2-scf-campaign-v1"

SCF_CAMPAIGN_SEEDS: tuple[int, ...] = tuple(MEASUREMENT_SEEDS)
"""The frozen five, reused unchanged: `53148, 59945, 42941, 720, 9428`.

Not re-derived and not extended. Reusing the historical Stage-2 seeds is what
lets a reader compare a V2-SCF head against a historical one at matched
initialisation without any new seed having been chosen after results existed.
"""

SCF_CAMPAIGN_RUN_COUNT = len(SCF_CAMPAIGN_SEEDS)

# --- selection-safety declarations, asserted by tests ------------------------
SCF_BEST_SEED_RULE = None
SCF_WINNER_RULE = None
SCF_TIE_BREAK = None
SCF_BEST_SEED_SELECTION_IMPLEMENTED = False
SCF_RANKS_AGAINST_HISTORICAL_ARMS = False
SCF_MEASUREMENT_MAY_SELECT = False
SCF_OFFICIAL_TEST_ROLE_EXISTS = False
SCF_TRAINING_ROLE = STAGE2_TRAINING_ROLE
SCF_SELECTION_ROLE = STAGE2_SELECTION_ROLE
SCF_MEASUREMENT_ROLE = STAGE2_MEASUREMENT_ROLE
"""Roles are the historical ones, reused. `Preg1Role` has no `OFFICIAL_TEST`
member, so official TEST cannot be named by this module at all."""


# ---------------------------------------------------------------------------
# Representation cache
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScfRepresentationKey:
    """Everything a cached V2-SCF representation tensor is bound to.

    Wider than the historical `Stage2RepresentationKey` in exactly the places
    that matter for a post-hoc candidate, and deliberately disjoint from it:

    * `pathway_id` where the historical key carries `arm`. A historical payload
      therefore cannot be parsed as this key and this key cannot be parsed as one
      -- checked, because `from_dict` refuses unknown and missing fields;
    * `fusion_id` -- the load-bearing field. The adapter tensors are
      shape-compatible across fusions, so a cache that did not record which
      equation produced it could not be told apart from a historical one;
    * `stage1_source_repository_head` **and** `stage2_repository_head`. The
      Stage-1 commit that trained the adapter and the Stage-2 commit that
      extracted the vectors are different commits and both are identity;
    * `stage1_checkpoint_sha256` and `stage1_selected_update`, so the cache names
      the one checkpoint it came from rather than the run.

    **No raw text.** Sample identity travels as an ordered-id digest.
    """

    stage2_repository_head: str
    pathway_id: str
    stage1_stage: str
    stage1_source_repository_head: str
    stage1_checkpoint_sha256: str
    stage1_selected_update: int
    fusion_id: str
    objective_id: str
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
    schema_version: str = V2_SCF_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_scf_pathway_id(self.pathway_id)
        require_stage2_condition(self.condition)
        Preg1Role(self.role)
        if self.fusion_id != V2_SCF_CHECKPOINT.fusion_id:
            raise ScfPathwayViolation(
                f"a V2-SCF cache must bind fusion {V2_SCF_CHECKPOINT.fusion_id!r}, got "
                f"{self.fusion_id!r}. The fusion is what distinguishes these vectors from "
                "a historical arm's; a cache that records the wrong one is not this "
                "campaign's."
            )
        if self.objective_id != V2_SCF_CHECKPOINT.objective_id:
            raise ScfPathwayViolation(
                f"a V2-SCF cache must bind objective {V2_SCF_CHECKPOINT.objective_id!r}, "
                f"got {self.objective_id!r}"
            )
        if self.stage1_stage != V2_SCF_CHECKPOINT.stage:
            raise ScfPathwayViolation(
                f"a V2-SCF cache must bind Stage-1 stage {V2_SCF_CHECKPOINT.stage!r}, got "
                f"{self.stage1_stage!r}"
            )
        if self.stage1_source_repository_head != V2_SCF_CHECKPOINT.source_repository_head:
            raise ScfPathwayViolation(
                f"a V2-SCF cache must bind Stage-1 source HEAD "
                f"{V2_SCF_CHECKPOINT.source_repository_head!r}, got "
                f"{self.stage1_source_repository_head!r}"
            )
        if self.stage1_checkpoint_sha256 != V2_SCF_CHECKPOINT.checkpoint_sha256:
            raise ScfPathwayViolation(
                "a V2-SCF cache must bind the frozen checkpoint digest "
                f"{V2_SCF_CHECKPOINT.checkpoint_sha256!r}, got "
                f"{self.stage1_checkpoint_sha256!r}"
            )
        if self.stage1_selected_update != V2_SCF_CHECKPOINT.update:
            raise ScfPathwayViolation(
                f"a V2-SCF cache must bind the frozen selected update "
                f"{V2_SCF_CHECKPOINT.update!r}, got {self.stage1_selected_update!r}"
            )
        if self.condition == STAGE2_CLEAN_CONDITION and self.corruption_seed is not None:
            raise ScfPathwayViolation(
                f"{STAGE2_CLEAN_CONDITION} is the clean condition and must carry no "
                f"corruption seed, got {self.corruption_seed!r}"
            )
        if self.condition != STAGE2_CLEAN_CONDITION and self.corruption_seed is None:
            raise ScfPathwayViolation(
                f"condition {self.condition!r} is corrupted and must bind the corruption "
                "seed that produced it; the seed is not defaulted anywhere"
            )
        if (
            self.role == SCF_MEASUREMENT_ROLE.value
            and self.condition != STAGE2_CLEAN_CONDITION
            and self.corruption_seed != STAGE2_MEASUREMENT_CORRUPTION_SEED
        ):
            raise ScfPathwayViolation(
                f"a V2-SCF measurement cache for degraded condition {self.condition!r} "
                f"must bind the frozen seed {STAGE2_MEASUREMENT_CORRUPTION_SEED} "
                f"(D-S2-002, reused unchanged), got {self.corruption_seed!r}"
            )
        if self.pooling != STAGE2_FIRST_TOKEN_POOLING:
            raise ScfPathwayViolation(
                f"V2-SCF pooling is frozen to {STAGE2_FIRST_TOKEN_POOLING!r}, got "
                f"{self.pooling!r}"
            )
        if self.hidden_size != HIDDEN_SIZE:
            raise ScfPathwayViolation(
                f"V2-SCF representations are {HIDDEN_SIZE}-dimensional, got "
                f"{self.hidden_size}"
            )
        if self.dtype != STAGE2_REPRESENTATION_DTYPE:
            raise ScfPathwayViolation(
                f"V2-SCF representations are {STAGE2_REPRESENTATION_DTYPE} (no AMP), got "
                f"{self.dtype!r}"
            )
        if self.count <= 0:
            raise ScfPathwayViolation(f"cache count must be positive, got {self.count}")
        for name in (
            "stage2_repository_head",
            "stage1_checkpoint_sha256",
            "ordered_id_digest",
            "label_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ScfPathwayViolation(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage2_repository_head": self.stage2_repository_head,
            "pathway_id": self.pathway_id,
            "stage1_stage": self.stage1_stage,
            "stage1_source_repository_head": self.stage1_source_repository_head,
            "stage1_checkpoint_sha256": self.stage1_checkpoint_sha256,
            "stage1_selected_update": self.stage1_selected_update,
            "fusion_id": self.fusion_id,
            "objective_id": self.objective_id,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScfRepresentationKey":
        """Parse a stored key. **Closed schema, in both directions.**

        A historical A/B key carries `arm` and `finalist_checkpoint_sha256` and
        carries no `pathway_id` or `fusion_id`, so it fails here on both counts.
        That is the point: the two artifact families must be mutually unreadable,
        not merely conventionally distinct.
        """
        if not isinstance(payload, Mapping):
            raise ScfPathwayViolation("V2-SCF cache metadata is not a JSON object")
        known = set(cls.__dataclass_fields__) | {"representation_shape"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ScfPathwayViolation(
                f"V2-SCF cache metadata carries unknown field(s) {unknown}; the schema is "
                "closed. A historical UNMARK-A/B cache is not a V2-SCF cache and may not "
                "be adopted as one."
            )
        missing = sorted(set(cls.__dataclass_fields__) - set(payload))
        if missing:
            raise ScfPathwayViolation(
                f"V2-SCF cache metadata is missing {missing}; partial binding is refused"
            )
        try:
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        except (TypeError, ValueError) as error:
            raise ScfPathwayViolation(
                f"V2-SCF cache metadata is malformed: {error}"
            ) from error

    def require_compatible(self, other: "ScfRepresentationKey") -> None:
        """Exact match on every field, or fail. No tolerance, no coercion."""
        mine, theirs = self.to_dict(), other.to_dict()
        differences = [name for name in mine if mine[name] != theirs[name]]
        if differences:
            detail = ", ".join(
                f"{name}: cached={theirs[name]!r} wanted={mine[name]!r}"
                for name in differences
            )
            raise ScfPathwayViolation(
                f"V2-SCF representation cache is incompatible on {len(differences)} "
                f"field(s): {detail}. Refusing to reuse it -- a cache reused across "
                "checkpoints, fusions, commits, roles or conditions produces a silent, "
                "plausible-looking result."
            )


@dataclass(frozen=True)
class ScfBoundRepresentations:
    """A representation tensor **and** the provenance saying what it is.

    Role and condition are properties of the tensor, read from the key that was
    validated when it was produced or loaded. There is no argument anywhere by
    which a caller can declare them, so a measurement tensor cannot be handed to
    checkpoint selection under a `PROTOCOL_DEV` label.
    """

    values: Any
    key: ScfRepresentationKey

    def __post_init__(self) -> None:
        shape = tuple(getattr(self.values, "shape", ()))
        if shape != (self.key.count, self.key.hidden_size):
            raise ScfPathwayViolation(
                f"representation shape {shape} contradicts its key's "
                f"{(self.key.count, self.key.hidden_size)}"
            )
        dtype = str(getattr(self.values, "dtype", ""))
        if dtype != self.key.dtype:
            raise ScfPathwayViolation(
                f"representation dtype {dtype!r} contradicts its key's {self.key.dtype!r}"
            )

    @property
    def role(self) -> Preg1Role:
        return Preg1Role(self.key.role)

    @property
    def condition(self) -> str:
        return self.key.condition

    def require_role(self, expected: Preg1Role, what: str) -> None:
        require_role(self.role, expected, what)

    def require_same_source(self, other: "ScfBoundRepresentations") -> None:
        """Both tensors came from the same checkpoint, fusion and Stage-2 commit."""
        for name in (
            "pathway_id",
            "stage1_checkpoint_sha256",
            "stage1_selected_update",
            "fusion_id",
            "stage1_source_repository_head",
            "stage2_repository_head",
        ):
            if getattr(self.key, name) != getattr(other.key, name):
                raise ScfPathwayViolation(
                    f"cross-source use: {name} is {getattr(self.key, name)!r} on one "
                    f"tensor and {getattr(other.key, name)!r} on the other. A head must be "
                    "trained and selected on one pathway's representations only."
                )


class ScfRepresentationCache:
    """Immutable fail-closed cache for frozen V2-SCF representations.

    Stores `[N, 768]` FP32 first-token vectors beside their key. **The filenames
    differ from the historical cache's on purpose**: pointing this cache at an
    UNMARK-A/B directory finds no metadata and fails, and pointing the historical
    cache at a V2-SCF directory does the same. Two artifact families that cannot
    be confused by a path typo are worth two constants.
    """

    METADATA_NAME = "stage2-v2-scf-representation-key.json"
    TENSOR_NAME = "stage2-v2-scf-representations.pt"
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

    def read_key(self) -> ScfRepresentationKey:
        if not self.metadata_path.is_file():
            raise ScfPathwayViolation(
                f"no V2-SCF cache metadata at {self.metadata_path}. A historical "
                "UNMARK-A/B cache directory is not a V2-SCF cache and is not adopted as one."
            )
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ScfPathwayViolation(
                f"V2-SCF cache metadata is not valid JSON: {error}"
            ) from error
        return ScfRepresentationKey.from_dict(payload)

    def save(self, key: ScfRepresentationKey, representations: Any) -> None:
        """Write once. An existing cache may only be rewritten byte-identically."""
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

    def load(self, key: ScfRepresentationKey) -> ScfBoundRepresentations:
        """Load only if the stored key matches **exactly**."""
        import torch

        key.require_compatible(self.read_key())
        tensor = torch.load(self.tensor_path, map_location="cpu")
        _require_fp32_matrix(tensor, key)
        return ScfBoundRepresentations(values=tensor, key=key)


def _require_fp32_matrix(tensor: Any, key: ScfRepresentationKey) -> None:
    shape = tuple(getattr(tensor, "shape", ()))
    if shape != (key.count, key.hidden_size):
        raise ScfPathwayViolation(
            f"representation tensor {shape} does not match key "
            f"{(key.count, key.hidden_size)}"
        )
    dtype = str(getattr(tensor, "dtype", ""))
    if dtype != key.dtype:
        raise ScfPathwayViolation(
            f"representation dtype {dtype!r} is not the frozen {key.dtype!r}"
        )


# ---------------------------------------------------------------------------
# Extraction plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScfExtractionRequest:
    """One (role, condition) extraction the V2-SCF campaign needs."""

    pathway_id: str
    role: str
    condition: str
    corruption_seed: int | None
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "role": self.role,
            "condition": self.condition,
            "corruption_seed": self.corruption_seed,
            "purpose": self.purpose,
        }


def scf_training_extraction_plan() -> tuple[ScfExtractionRequest, ...]:
    """Clean `FULL` protocol-train and protocol-dev. **Two items, no more.**

    Official validation is absent by construction: this plan is what the clean
    campaign executes, and it contains no measurement role at all.
    """
    require_frozen_scf_protocol_spec()
    return (
        ScfExtractionRequest(
            pathway_id=V2_SCF_PATHWAY_ID,
            role=SCF_TRAINING_ROLE.value,
            condition=STAGE2_CLEAN_CONDITION,
            corruption_seed=None,
            purpose="head_training",
        ),
        ScfExtractionRequest(
            pathway_id=V2_SCF_PATHWAY_ID,
            role=SCF_SELECTION_ROLE.value,
            condition=STAGE2_CLEAN_CONDITION,
            corruption_seed=None,
            purpose="head_checkpoint_selection",
        ),
    )


def scf_representation_key_for(
    request: ScfExtractionRequest,
    *,
    stage2_repository_head: str,
    ordered_ids: Sequence[str],
    labels: Sequence[int],
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT,
) -> ScfRepresentationKey:
    """Build the cache key a request's tensor must be stored under."""
    require_scf_pathway_id(request.pathway_id)
    return ScfRepresentationKey(
        stage2_repository_head=stage2_repository_head,
        pathway_id=request.pathway_id,
        stage1_stage=identity.stage,
        stage1_source_repository_head=identity.source_repository_head,
        stage1_checkpoint_sha256=identity.checkpoint_sha256,
        stage1_selected_update=identity.update,
        fusion_id=identity.fusion_id,
        objective_id=identity.objective_id,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=V2_SCF_POSTHOC_PROTOCOL_VERSION,
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


def extract_and_cache_scf_representations(
    pathway: FrozenScfPathway,
    batches: Sequence[Mapping[str, Any]],
    key: ScfRepresentationKey,
    cache: ScfRepresentationCache,
) -> ScfBoundRepresentations:
    """Run the frozen V2-SCF pathway over batches and cache the result.

    A driver, not a forward pass: the encoder/adapter call is
    :func:`extract_scf_representations`, which itself delegates to the accepted
    Audit-050 implementation. Its job is to concatenate per-batch `[b, 768]`
    outputs in order, check the row count, and store the tensor under a key that
    binds the checkpoint, fusion, both commits, role, condition and seed.
    """
    # Identity guards run BEFORE torch is imported, so a mislabelled extraction is
    # refused on any machine and leaves no artifact behind.
    if not isinstance(pathway, FrozenScfPathway):
        raise ScfPathwayViolation(
            f"expected a FrozenScfPathway, got {type(pathway).__name__}; a historical arm "
            "pathway may never write a V2-SCF cache"
        )
    binding = pathway.binding
    if binding.pathway_id != key.pathway_id:
        raise ScfPathwayViolation(
            f"pathway carries {binding.pathway_id!r} but the cache key is "
            f"{key.pathway_id!r}"
        )
    if binding.checkpoint_sha256 != key.stage1_checkpoint_sha256:
        raise ScfPathwayViolation(
            "pathway checkpoint sha256 does not match the cache key"
        )
    if binding.fusion_id != key.fusion_id:
        raise ScfPathwayViolation("pathway fusion does not match the cache key")
    if not batches:
        raise ScfPathwayViolation("no batches supplied for extraction")

    # Reused verbatim from the historical runner: it reads only `.condition` and
    # `.corruption_seed` off the key, so one implementation serves both families
    # and a V2-SCF tensor cannot claim a realisation it was not produced under.
    require_batch_provenance(batches, key)

    import torch

    pooled = [extract_scf_representations(pathway, batch) for batch in batches]
    values = torch.cat(pooled, dim=0)
    if int(values.shape[0]) != key.count:
        raise ScfPathwayViolation(
            f"extracted {int(values.shape[0])} rows but the key declares {key.count}"
        )
    cache.save(key, values)
    return cache.load(key)


# ---------------------------------------------------------------------------
# Head training over cached representations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScfHeadRun:
    """One completed V2-SCF head run: identity, full history, selected epoch."""

    pathway_id: str
    seed: int
    learning_rate: float
    epochs: int
    scores: tuple[EpochScore, ...]
    selected: EpochScore
    train_key: ScfRepresentationKey
    dev_key: ScfRepresentationKey
    initial_head_fingerprint: str
    selected_head_state_sha256: str
    history_digest: str
    selected_head_state: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V2_SCF_CAMPAIGN_SCHEMA_VERSION,
            "pathway_id": self.pathway_id,
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


def scf_history_digest(scores: Sequence[EpochScore]) -> str:
    payload = json.dumps([s.to_dict() for s in scores], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_scf_head(
    train: ScfBoundRepresentations,
    train_labels: Sequence[int],
    dev: ScfBoundRepresentations,
    dev_labels: Sequence[int],
    *,
    seed: int,
    learning_rate: float = STAGE2_HEAD_LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> ScfHeadRun:
    """Train one V2-SCF head for **all** `epochs` over cached representations.

    Byte-for-byte the corrected historical Stage-2 head protocol, and nothing
    about it is conditional on this being a post-hoc candidate: the same locked
    primitives, the same frozen LR, the same 30-epoch schedule with no early
    stopping, the same `seed * 1000 + epoch` batch order, the same
    macro-F1 -> accuracy -> earliest-epoch selection on clean protocol-dev.

    No encoder and no adapter are touched: this takes tensors, never a pathway,
    so an epoch loop cannot accidentally re-run PhoBERT.

    **Official validation cannot enter.** `require_role` reads the role off the
    tensor's own key, and there is no argument by which a caller can declare one:
    a measurement tensor handed to either position is refused.
    """
    import torch
    from torch import nn

    require_protocol_settings()
    require_frozen_scf_protocol_spec()

    train.require_role(SCF_TRAINING_ROLE, "V2-SCF head training")
    dev.require_role(SCF_SELECTION_ROLE, "V2-SCF head checkpoint selection")
    require_clean_condition(train.condition, "V2-SCF head training")
    require_clean_condition(dev.condition, "V2-SCF head checkpoint selection")
    train.require_same_source(dev)

    if learning_rate != STAGE2_HEAD_LEARNING_RATE:
        raise ScfPathwayViolation(
            f"the Stage-2 head LR is frozen at {STAGE2_HEAD_LEARNING_RATE} (D-S2-001) and "
            f"is inherited unchanged by this post-hoc campaign; got {learning_rate!r}. "
            "There is no V2-SCF LR pilot, and no Stage-2 hyperparameter may be retuned "
            "for a candidate."
        )
    if epochs != EPOCHS:
        raise ScfPathwayViolation(
            f"the Stage-2 epoch budget is frozen at {EPOCHS}; got {epochs!r}"
        )
    if seed not in SCF_CAMPAIGN_SEEDS:
        raise ScfPathwayViolation(
            f"seed {seed!r} is not one of the frozen campaign seeds "
            f"{list(SCF_CAMPAIGN_SEEDS)}"
        )

    features, dev_features = train.values, dev.values
    if features.shape[0] != len(train_labels):
        raise ScfPathwayViolation("train features and labels differ in length")
    if dev_features.shape[0] != len(dev_labels):
        raise ScfPathwayViolation("dev features and labels differ in length")
    if train.key.label_digest != label_digest(train_labels):
        raise ScfPathwayViolation("train labels do not match the cached label digest")
    if dev.key.label_digest != label_digest(dev_labels):
        raise ScfPathwayViolation("dev labels do not match the cached label digest")

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
    return ScfHeadRun(
        pathway_id=train.key.pathway_id,
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs,
        scores=tuple(scores),
        selected=selected,
        train_key=train.key,
        dev_key=dev.key,
        initial_head_fingerprint=initial_fingerprint,
        selected_head_state_sha256=_state_digest(selected_state),
        history_digest=scf_history_digest(scores),
        selected_head_state=selected_state,
    )


# ---------------------------------------------------------------------------
# Head artifact
# ---------------------------------------------------------------------------
SCF_HEAD_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "pathway_id",
    "posthoc_exploratory",
    "stage1_stage",
    "stage1_source_repository_head",
    "stage1_checkpoint_sha256",
    "stage1_selected_update",
    "fusion_id",
    "objective_id",
    "stage2_repository_head",
    "protocol_version",
    "seed",
    "initial_head_fingerprint",
    "head_architecture",
    "optimizer",
    "learning_rate",
    "epochs",
    "early_stopping",
    "selected_epoch",
    "selected_macro_f1",
    "selected_accuracy",
    "selected_head_state_sha256",
    "train_cache_key",
    "protocol_dev_cache_key",
    "history_digest",
    "precision",
    "selection_role",
    "selection_condition",
    "measurement_used_for_selection",
    "best_seed_selection_performed",
    "ranked_against_historical_arms",
    "official_test_used",
)
"""Closed schema. It carries `pathway_id`, never `arm`, so a V2-SCF head artifact
and a historical one cannot validate as each other in either direction."""


def build_scf_head_artifact(
    run: ScfHeadRun,
    *,
    stage2_repository_head: str,
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT,
) -> dict[str, Any]:
    """The immutable record of one completed V2-SCF head run."""
    if run.train_key.stage2_repository_head != stage2_repository_head:
        raise ScfPathwayViolation(
            "run caches were produced under a different Stage-2 repository head"
        )
    artifact = {
        "schema_version": V2_SCF_CAMPAIGN_SCHEMA_VERSION,
        "pathway_id": run.pathway_id,
        "posthoc_exploratory": True,
        "stage1_stage": identity.stage,
        "stage1_source_repository_head": identity.source_repository_head,
        "stage1_checkpoint_sha256": identity.checkpoint_sha256,
        "stage1_selected_update": identity.update,
        "fusion_id": identity.fusion_id,
        "objective_id": identity.objective_id,
        "stage2_repository_head": stage2_repository_head,
        "protocol_version": V2_SCF_POSTHOC_PROTOCOL_VERSION,
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
        "selection_role": SCF_SELECTION_ROLE.value,
        "selection_condition": STAGE2_CLEAN_CONDITION,
        "measurement_used_for_selection": False,
        "best_seed_selection_performed": False,
        "ranked_against_historical_arms": False,
        "official_test_used": False,
    }
    validate_scf_head_artifact(artifact, expected_seed=run.seed)
    return artifact


def validate_scf_head_artifact(
    artifact: Mapping[str, Any], *, expected_seed: int
) -> None:
    """Fail closed unless the artifact fully and consistently binds one run."""
    if not isinstance(artifact, Mapping):
        raise ScfPathwayViolation("V2-SCF head artifact is not a JSON object")
    missing = [f for f in SCF_HEAD_ARTIFACT_FIELDS if f not in artifact]
    if missing:
        raise ScfPathwayViolation(
            f"V2-SCF head artifact is missing {missing}; partial binding is refused"
        )
    unknown = sorted(set(artifact) - set(SCF_HEAD_ARTIFACT_FIELDS))
    if unknown:
        raise ScfPathwayViolation(
            f"V2-SCF head artifact carries unknown field(s) {unknown}; the schema is closed"
        )
    if artifact["schema_version"] != V2_SCF_CAMPAIGN_SCHEMA_VERSION:
        raise ScfPathwayViolation("V2-SCF head artifact schema version is wrong")
    require_scf_pathway_id(artifact["pathway_id"])
    if artifact["seed"] != expected_seed:
        raise ScfPathwayViolation(
            f"V2-SCF head artifact seed {artifact['seed']!r} != expected {expected_seed!r}"
        )
    if artifact["seed"] not in SCF_CAMPAIGN_SEEDS:
        raise ScfPathwayViolation(
            f"V2-SCF head artifact seed {artifact['seed']!r} is not a frozen seed"
        )
    for name, want in (
        ("stage1_stage", V2_SCF_CHECKPOINT.stage),
        ("stage1_source_repository_head", V2_SCF_CHECKPOINT.source_repository_head),
        ("stage1_checkpoint_sha256", V2_SCF_CHECKPOINT.checkpoint_sha256),
        ("stage1_selected_update", V2_SCF_CHECKPOINT.update),
        ("fusion_id", V2_SCF_CHECKPOINT.fusion_id),
        ("objective_id", V2_SCF_CHECKPOINT.objective_id),
        ("protocol_version", V2_SCF_POSTHOC_PROTOCOL_VERSION),
    ):
        if artifact[name] != want:
            raise ScfPathwayViolation(
                f"V2-SCF head artifact {name} is {artifact[name]!r}, expected {want!r}"
            )
    if artifact["learning_rate"] != STAGE2_HEAD_LEARNING_RATE:
        raise ScfPathwayViolation(
            f"V2-SCF head artifact LR is not the frozen {STAGE2_HEAD_LEARNING_RATE}"
        )
    if artifact["epochs"] != EPOCHS or artifact["early_stopping"] is not False:
        raise ScfPathwayViolation(
            "V2-SCF head artifact budget or stopping rule drifted"
        )
    if artifact["selection_role"] != SCF_SELECTION_ROLE.value:
        raise ScfPathwayViolation(
            "V2-SCF head artifact was not selected on protocol-dev"
        )
    if artifact["selection_condition"] != STAGE2_CLEAN_CONDITION:
        raise ScfPathwayViolation("V2-SCF head artifact was not selected on clean FULL")
    if artifact["posthoc_exploratory"] is not True:
        raise ScfPathwayViolation(
            "V2-SCF head artifact must declare posthoc_exploratory=true; this campaign is "
            "not confirmatory evaluation and its artifacts may not imply that it is"
        )
    for flag in (
        "measurement_used_for_selection",
        "best_seed_selection_performed",
        "ranked_against_historical_arms",
        "official_test_used",
    ):
        if artifact[flag] is not False:
            raise ScfPathwayViolation(f"V2-SCF head artifact {flag} must be false")
    for name, expected_role in (
        ("train_cache_key", SCF_TRAINING_ROLE),
        ("protocol_dev_cache_key", SCF_SELECTION_ROLE),
    ):
        key = ScfRepresentationKey.from_dict(artifact[name])
        if key.role != expected_role.value:
            raise ScfPathwayViolation(f"{name} has role {key.role!r}")
        if key.condition != STAGE2_CLEAN_CONDITION:
            raise ScfPathwayViolation(f"{name} is not the clean condition")
        if key.pathway_id != artifact["pathway_id"]:
            raise ScfPathwayViolation(f"{name} belongs to a different pathway")
        if key.stage1_checkpoint_sha256 != artifact["stage1_checkpoint_sha256"]:
            raise ScfPathwayViolation(f"{name} binds a different Stage-1 checkpoint")
        if key.stage2_repository_head != artifact["stage2_repository_head"]:
            raise ScfPathwayViolation(f"{name} was produced under another Stage-2 commit")


# ---------------------------------------------------------------------------
# Run store: completed runs are immutable; partial runs never resume
# ---------------------------------------------------------------------------
SCF_RESUME_CONTRACT = "atomic_rerun_no_midrun_resume"
"""A head run is atomic: complete, or absent. Same reasoning as the historical
store -- one run is 30 deterministic epochs of a linear head over cached
vectors, so a restart reproduces it bit-for-bit and a resume surface would buy
nothing while adding a way to reattach a partial run to a drifted identity."""


class ScfHeadRunStore:
    """Immutable per-seed output directory with a fail-closed rerun rule."""

    ARTIFACT_NAME = "stage2-v2-scf-head-artifact.json"
    STATE_NAME = "stage2-v2-scf-selected-head.pt"
    IN_PROGRESS_NAME = "IN_PROGRESS"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def artifact_path(self) -> Path:
        return self.directory / self.ARTIFACT_NAME

    def is_complete(self) -> bool:
        return self.artifact_path.is_file()

    def read_artifact(self, *, expected_seed: int) -> dict[str, Any]:
        if not self.is_complete():
            raise ScfPathwayViolation(
                f"no completed V2-SCF head run at {self.directory}"
            )
        payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        validate_scf_head_artifact(payload, expected_seed=expected_seed)
        return payload

    def require_writable(self, *, expected_seed: int) -> None:
        """Refuse to overwrite completed evidence; permit a clean rerun otherwise."""
        if self.is_complete():
            raise ScfPathwayViolation(
                f"{self.directory} already holds a completed V2-SCF head run for seed "
                f"{expected_seed}. Completed runs are immutable: delete it deliberately, "
                "or write elsewhere. Nothing here overwrites scientific evidence."
            )
        partial = self.directory / self.IN_PROGRESS_NAME
        if partial.is_file():
            recorded = json.loads(partial.read_text(encoding="utf-8"))
            if recorded.get("seed") != expected_seed or recorded.get(
                "pathway_id"
            ) != V2_SCF_PATHWAY_ID:
                raise ScfPathwayViolation(
                    f"{self.directory} holds a partial run for "
                    f"{recorded.get('pathway_id')!r}/{recorded.get('seed')!r}, not "
                    f"{V2_SCF_PATHWAY_ID!r}/{expected_seed!r}. Resume contract is "
                    f"{SCF_RESUME_CONTRACT!r}: a partial run is discarded and rerun, never "
                    "adopted under a different identity."
                )

    def mark_in_progress(self, *, seed: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            self.directory / self.IN_PROGRESS_NAME,
            (
                json.dumps(
                    {
                        "pathway_id": V2_SCF_PATHWAY_ID,
                        "seed": seed,
                        "resume_contract": SCF_RESUME_CONTRACT,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def commit(
        self, artifact: Mapping[str, Any], selected_head_state: Mapping[str, Any]
    ) -> None:
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
# Campaign plan -- exactly five seeds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScfCampaignRun:
    pathway_id: str
    seed: int


def scf_campaign_plan() -> tuple[ScfCampaignRun, ...]:
    """Exactly `1 x 5 = 5` runs. No sixth seed, no dropped seed, no second arm."""
    require_frozen_scf_protocol_spec()
    plan = tuple(
        ScfCampaignRun(pathway_id=V2_SCF_PATHWAY_ID, seed=seed)
        for seed in SCF_CAMPAIGN_SEEDS
    )
    require_scf_campaign_plan(plan)
    return plan


def require_scf_campaign_plan(plan: Sequence[ScfCampaignRun]) -> None:
    """Every frozen seed exactly once, for the one pathway. Nothing else."""
    if len(plan) != SCF_CAMPAIGN_RUN_COUNT:
        raise ScfPathwayViolation(
            f"the V2-SCF campaign is exactly {SCF_CAMPAIGN_RUN_COUNT} runs "
            f"(1 pathway x {len(SCF_CAMPAIGN_SEEDS)} seeds), got {len(plan)}"
        )
    seen: set[int] = set()
    for run in plan:
        require_scf_pathway_id(run.pathway_id)
        if run.seed not in SCF_CAMPAIGN_SEEDS:
            raise ScfPathwayViolation(f"seed {run.seed} is not a frozen seed")
        if run.seed in seen:
            raise ScfPathwayViolation(f"duplicate run for seed {run.seed}")
        seen.add(run.seed)
    if sorted(seen) != sorted(SCF_CAMPAIGN_SEEDS):
        raise ScfPathwayViolation(
            f"campaign seeds {sorted(seen)} drifted from the frozen "
            f"{sorted(SCF_CAMPAIGN_SEEDS)}. All five heads are retained and reported; a "
            "seed may never be dropped because a result looks bad."
        )


# ---------------------------------------------------------------------------
# Campaign manifest and registry
# ---------------------------------------------------------------------------
SCF_CAMPAIGN_CACHE_ROLES = (SCF_TRAINING_ROLE, SCF_SELECTION_ROLE)
SCF_CAMPAIGN_CACHE_SLOTS: tuple[str, ...] = tuple(
    f"{V2_SCF_PATHWAY_ID}/{role.value}" for role in SCF_CAMPAIGN_CACHE_ROLES
)


def scf_cache_slot(role: Any) -> str:
    """Stable manifest key for the V2-SCF cache in one role."""
    if role not in SCF_CAMPAIGN_CACHE_ROLES:
        raise ScfPathwayViolation(
            f"{role!r} is not one of the clean V2-SCF campaign roles "
            f"{[r.value for r in SCF_CAMPAIGN_CACHE_ROLES]}. Official validation has no "
            "slot in the clean campaign."
        )
    return f"{V2_SCF_PATHWAY_ID}/{role.value}"


@dataclass(frozen=True)
class ScfCampaignManifest:
    """The one identity every run in the V2-SCF campaign must share."""

    stage2_repository_head: str
    protocol_version: str
    pathway_id: str
    seeds: tuple[int, ...]
    cache_keys: Mapping[str, ScfRepresentationKey]
    expected_runs: tuple[ScfCampaignRun, ...]
    schema_version: str = V2_SCF_CAMPAIGN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage2_repository_head": self.stage2_repository_head,
            "protocol_version": self.protocol_version,
            "pathway_id": self.pathway_id,
            "posthoc_exploratory": True,
            "stage1_checkpoint_sha256": V2_SCF_CHECKPOINT.checkpoint_sha256,
            "stage1_source_repository_head": V2_SCF_CHECKPOINT.source_repository_head,
            "stage1_selected_update": V2_SCF_CHECKPOINT.update,
            "fusion_id": V2_SCF_CHECKPOINT.fusion_id,
            "seeds": list(self.seeds),
            "cache_keys": {
                slot: key.to_dict() for slot, key in sorted(self.cache_keys.items())
            },
            "expected_runs": [
                {"pathway_id": r.pathway_id, "seed": r.seed} for r in self.expected_runs
            ],
            "best_seed_rule": SCF_BEST_SEED_RULE,
            "winner_rule": SCF_WINNER_RULE,
            "ranks_against_historical_arms": SCF_RANKS_AGAINST_HISTORICAL_ARMS,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


def build_scf_campaign_manifest(
    *, stage2_repository_head: str, cache_keys: Mapping[str, ScfRepresentationKey]
) -> ScfCampaignManifest:
    """Assemble and fully validate one V2-SCF campaign identity."""
    manifest = ScfCampaignManifest(
        stage2_repository_head=stage2_repository_head,
        protocol_version=V2_SCF_POSTHOC_PROTOCOL_VERSION,
        pathway_id=V2_SCF_PATHWAY_ID,
        seeds=tuple(SCF_CAMPAIGN_SEEDS),
        cache_keys=dict(cache_keys),
        expected_runs=scf_campaign_plan(),
    )
    validate_scf_campaign_manifest(manifest)
    return manifest


def validate_scf_campaign_manifest(manifest: ScfCampaignManifest) -> None:
    """Fail closed unless this is exactly the frozen V2-SCF campaign.

    Every rule here is one a per-run check cannot see: a run knows its own
    commit, but only the manifest knows whether the other four agree.
    """
    require_frozen_scf_protocol_spec()

    if manifest.schema_version != V2_SCF_CAMPAIGN_SCHEMA_VERSION:
        raise ScfPathwayViolation(
            f"campaign manifest schema {manifest.schema_version!r} is not "
            f"{V2_SCF_CAMPAIGN_SCHEMA_VERSION!r}"
        )
    if manifest.protocol_version != V2_SCF_POSTHOC_PROTOCOL_VERSION:
        raise ScfPathwayViolation(
            f"campaign protocol {manifest.protocol_version!r} is not the frozen "
            f"{V2_SCF_POSTHOC_PROTOCOL_VERSION!r}"
        )
    require_scf_pathway_id(manifest.pathway_id)
    if not isinstance(manifest.stage2_repository_head, str) or not (
        manifest.stage2_repository_head.strip()
    ):
        raise ScfPathwayViolation(
            "campaign stage2_repository_head must be a non-empty string"
        )
    if tuple(manifest.seeds) != tuple(SCF_CAMPAIGN_SEEDS):
        raise ScfPathwayViolation(
            f"campaign seeds {list(manifest.seeds)} are not exactly the frozen "
            f"{list(SCF_CAMPAIGN_SEEDS)}: no seed added, none dropped, order fixed"
        )

    missing = [s for s in SCF_CAMPAIGN_CACHE_SLOTS if s not in manifest.cache_keys]
    if missing:
        raise ScfPathwayViolation(
            f"campaign manifest is missing cache slot(s) {missing}; the pathway needs both "
            "a protocol-train and a protocol-dev clean cache"
        )
    unknown = sorted(set(manifest.cache_keys) - set(SCF_CAMPAIGN_CACHE_SLOTS))
    if unknown:
        raise ScfPathwayViolation(
            f"campaign manifest carries unknown cache slot(s) {unknown}; the slot set is "
            "closed, and official validation has no slot in the clean campaign"
        )
    for role in SCF_CAMPAIGN_CACHE_ROLES:
        slot = scf_cache_slot(role)
        key = manifest.cache_keys[slot]
        if not isinstance(key, ScfRepresentationKey):
            raise ScfPathwayViolation(f"{slot} is not a ScfRepresentationKey")
        if key.pathway_id != manifest.pathway_id:
            raise ScfPathwayViolation(
                f"cache slot {slot} holds pathway {key.pathway_id!r}"
            )
        if key.role != role.value:
            raise ScfPathwayViolation(
                f"cache slot {slot} holds role {key.role!r}, expected {role.value!r}"
            )
        if key.condition != STAGE2_CLEAN_CONDITION:
            raise ScfPathwayViolation(
                f"cache slot {slot} is condition {key.condition!r}; head training and "
                f"checkpoint selection use clean {STAGE2_CLEAN_CONDITION} only"
            )
        if key.stage2_repository_head != manifest.stage2_repository_head:
            raise ScfPathwayViolation(
                f"cache slot {slot} was produced at {key.stage2_repository_head!r} but the "
                f"campaign is {manifest.stage2_repository_head!r}: a campaign may not mix "
                "commits"
            )
        if key.protocol_version != manifest.protocol_version:
            raise ScfPathwayViolation(
                f"cache slot {slot} is protocol {key.protocol_version!r} but the campaign "
                f"is {manifest.protocol_version!r}"
            )

    train_key = manifest.cache_keys[scf_cache_slot(SCF_TRAINING_ROLE)]
    dev_key = manifest.cache_keys[scf_cache_slot(SCF_SELECTION_ROLE)]
    if train_key.ordered_id_digest == dev_key.ordered_id_digest:
        raise ScfPathwayViolation(
            "protocol-train and protocol-dev bind the same ordered-id digest; the two "
            "splits would be the same rows and checkpoint selection would be scored on "
            "the training set"
        )

    require_scf_campaign_plan(manifest.expected_runs)


class ScfCampaignRegistry:
    """Persisted V2-SCF campaign identity plus per-run completion state.

    Re-entering a campaign is permitted only under a **byte-identical manifest**,
    so a partial campaign cannot adopt run state produced under a different
    commit, protocol, checkpoint or cache set.
    """

    MANIFEST_NAME = "stage2-v2-scf-campaign-manifest.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST_NAME

    def run_directory(self, seed: int) -> Path:
        if seed not in SCF_CAMPAIGN_SEEDS:
            raise ScfPathwayViolation(f"seed {seed!r} is not a frozen seed")
        return self.directory / V2_SCF_PATHWAY_ID / f"seed-{seed}"

    def store_for(self, seed: int) -> ScfHeadRunStore:
        return ScfHeadRunStore(self.run_directory(seed))

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    def open(self, manifest: ScfCampaignManifest) -> ScfCampaignManifest:
        """Create or re-open. A drifted manifest is refused, never adopted."""
        validate_scf_campaign_manifest(manifest)
        if self.exists():
            recorded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if recorded.get("manifest_digest") != manifest.digest:
                raise ScfPathwayViolation(
                    f"{self.directory} holds a campaign whose manifest digest is "
                    f"{recorded.get('manifest_digest')!r}, not {manifest.digest!r}. A "
                    "partial campaign may not adopt an incompatible identity. Start a new "
                    "campaign directory."
                )
            return manifest
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"manifest_digest": manifest.digest, "manifest": manifest.to_dict()}
        atomic_write_bytes(
            self.manifest_path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return manifest

    def completed_runs(self, manifest: ScfCampaignManifest) -> tuple[ScfCampaignRun, ...]:
        done = []
        for run in manifest.expected_runs:
            store = self.store_for(run.seed)
            if store.is_complete():
                store.read_artifact(expected_seed=run.seed)
                done.append(run)
        return tuple(done)

    def pending_runs(self, manifest: ScfCampaignManifest) -> tuple[ScfCampaignRun, ...]:
        done = {r.seed for r in self.completed_runs(manifest)}
        return tuple(r for r in manifest.expected_runs if r.seed not in done)

    def campaign_status(self, manifest: ScfCampaignManifest) -> dict[str, Any]:
        """Completion report. **Not** a comparison: no scores, no ranking."""
        completed = self.completed_runs(manifest)
        pending = self.pending_runs(manifest)
        return {
            "schema_version": V2_SCF_CAMPAIGN_SCHEMA_VERSION,
            "pathway_id": manifest.pathway_id,
            "posthoc_exploratory": True,
            "manifest_digest": manifest.digest,
            "expected": len(manifest.expected_runs),
            "completed": len(completed),
            "pending": [{"seed": r.seed} for r in pending],
            "completed_seeds": sorted(r.seed for r in completed),
            "complete": len(completed) == len(manifest.expected_runs),
            "best_seed_rule": SCF_BEST_SEED_RULE,
            "ranks_against_historical_arms": SCF_RANKS_AGAINST_HISTORICAL_ARMS,
        }

    def require_campaign_complete(self, manifest: ScfCampaignManifest) -> None:
        """All five heads. A partial campaign may not be reported or measured."""
        status = self.campaign_status(manifest)
        if not status["complete"]:
            raise ScfPathwayViolation(
                f"the V2-SCF clean campaign is {status['completed']}/{status['expected']} "
                f"complete; missing {status['pending']}. All five heads must exist before "
                "anything is reported, and before the measurement boundary may be crossed."
            )


def run_scf_campaign(
    manifest: ScfCampaignManifest,
    registry: ScfCampaignRegistry,
    *,
    cache_directories: Mapping[str, str | Path],
    labels: Mapping[str, Sequence[int]],
    stage2_repository_head: str,
    only: Sequence[ScfCampaignRun] | None = None,
) -> dict[str, Any]:
    """Execute the campaign's pending head runs over cached representations.

    **Reads no dataset, and cannot read official validation.** `cache_directories`
    maps the two clean manifest slots to directories already produced by
    extraction; `labels` maps the same slots to the caller's label vectors, which
    the caches' `label_digest` then verifies. The slot set is closed to the two
    clean roles, so there is no argument by which a measurement cache could be
    routed into head training.

    A completed run is skipped, never re-executed and never overwritten.
    """
    registry.open(manifest)
    if stage2_repository_head != manifest.stage2_repository_head:
        raise ScfPathwayViolation(
            f"caller head {stage2_repository_head!r} does not match the campaign's "
            f"{manifest.stage2_repository_head!r}"
        )

    missing = [s for s in SCF_CAMPAIGN_CACHE_SLOTS if s not in cache_directories]
    if missing:
        raise ScfPathwayViolation(f"no cache directory supplied for {missing}")
    missing_labels = [s for s in SCF_CAMPAIGN_CACHE_SLOTS if s not in labels]
    if missing_labels:
        raise ScfPathwayViolation(f"no labels supplied for {missing_labels}")

    scheduled = list(only) if only is not None else list(registry.pending_runs(manifest))
    for run in scheduled:
        if run not in manifest.expected_runs:
            raise ScfPathwayViolation(
                f"{run} is not one of the campaign's {SCF_CAMPAIGN_RUN_COUNT} expected runs"
            )

    train_slot = scf_cache_slot(SCF_TRAINING_ROLE)
    dev_slot = scf_cache_slot(SCF_SELECTION_ROLE)

    executed: list[dict[str, Any]] = []
    for run in scheduled:
        store = registry.store_for(run.seed)
        store.require_writable(expected_seed=run.seed)

        train = ScfRepresentationCache(cache_directories[train_slot]).load(
            manifest.cache_keys[train_slot]
        )
        dev = ScfRepresentationCache(cache_directories[dev_slot]).load(
            manifest.cache_keys[dev_slot]
        )

        store.mark_in_progress(seed=run.seed)
        head_run = train_scf_head(
            train, labels[train_slot], dev, labels[dev_slot], seed=run.seed
        )
        artifact = build_scf_head_artifact(
            head_run, stage2_repository_head=stage2_repository_head
        )
        validate_scf_head_artifact(artifact, expected_seed=run.seed)
        store.commit(artifact, head_run.selected_head_state)
        executed.append({"seed": run.seed, "selected_epoch": head_run.selected.epoch})

    return {
        "schema_version": V2_SCF_CAMPAIGN_SCHEMA_VERSION,
        "pathway_id": manifest.pathway_id,
        "manifest_digest": manifest.digest,
        "executed": executed,
        "status": registry.campaign_status(manifest),
    }


__all__ = [
    "SCF_BEST_SEED_RULE",
    "SCF_BEST_SEED_SELECTION_IMPLEMENTED",
    "SCF_CAMPAIGN_CACHE_ROLES",
    "SCF_CAMPAIGN_CACHE_SLOTS",
    "SCF_CAMPAIGN_RUN_COUNT",
    "SCF_CAMPAIGN_SEEDS",
    "SCF_HEAD_ARTIFACT_FIELDS",
    "SCF_MEASUREMENT_MAY_SELECT",
    "SCF_MEASUREMENT_ROLE",
    "SCF_OFFICIAL_TEST_ROLE_EXISTS",
    "SCF_RANKS_AGAINST_HISTORICAL_ARMS",
    "SCF_RESUME_CONTRACT",
    "SCF_SELECTION_ROLE",
    "SCF_TIE_BREAK",
    "SCF_TRAINING_ROLE",
    "SCF_WINNER_RULE",
    "STAGE2_DEGRADED_CONDITIONS",
    "STAGE2_UNMARK_CONDITIONS",
    "V2_SCF_CAMPAIGN_SCHEMA_VERSION",
    "ScfBoundRepresentations",
    "ScfCampaignManifest",
    "ScfCampaignRegistry",
    "ScfCampaignRun",
    "ScfExtractionRequest",
    "ScfHeadRun",
    "ScfHeadRunStore",
    "ScfRepresentationCache",
    "ScfRepresentationKey",
    "build_scf_campaign_manifest",
    "build_scf_head_artifact",
    "extract_and_cache_scf_representations",
    "require_scf_campaign_plan",
    "run_scf_campaign",
    "scf_cache_slot",
    "scf_campaign_plan",
    "scf_history_digest",
    "scf_representation_key_for",
    "scf_training_extraction_plan",
    "train_scf_head",
    "validate_scf_campaign_manifest",
    "validate_scf_head_artifact",
]
