"""Frozen-encoder linear-head trainer and evaluator for the pre-G1 diagnostic.

This implements the machinery for one question and no other: **how much does
stripping Vietnamese orthographic marks cost, on its own, in front of a frozen
encoder?** Vanilla feeds `canon(x)`; Base-only feeds `b(canon(x))`. Same
tokenizer, same frozen encoder, same pooling, same head protocol, same seeds.
The only difference between the two arms is the stripping step, which is the
entire point -- anything else that differs becomes a rival explanation for the
gap.

**This is not UNMARK.** No tone channel, no letter channel, no adapter, no
RESTORE, no ALIGN, no Stage-1 adaptation appears here or may be added here. It
measures the burden that UNMARK would later have to recover; it says nothing
about whether UNMARK recovers it.

**Nothing in this module runs an experiment.** It is the mechanism. The LR grid
is not swept here, the paired measurement is not executed here, and no
downstream score exists.

Scientific constants are **imported** from `preg1_protocol`, never restated:
seeds, LR grid, epochs, batch size, optimiser settings, decay groups, pooling,
`max_length` and the checkpoint / aggregation rules all live there. Nothing in
this file may reinterpret them.

**Torch is imported lazily**, inside the functions that need it, so the
selection logic, the membership guards and the cache contract stay importable
and testable in an ML-free environment.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from unmark.evaluation.contracts import (
    EvaluationContractViolation,
    SplitLeakage,
    SystemPathway,
)
from unmark.evaluation.metrics import accuracy, macro_f1
from unmark.evaluation.preg1_protocol import (
    ADAMW_BETAS,
    ADAMW_EPS,
    AMSGRAD,
    BATCH_SIZE,
    CHECKPOINT_ELIGIBLE_EPOCHS,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIPPING,
    HEAD_DROPOUT,
    LABEL_MAPPING,
    LR_GRID,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PREG1_POOLING,
    PREG1_POOLING_SCOPE,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_NUM_LABELS,
    PRIMARY_TASK,
    TRUNCATION,
    TUNING_SEEDS,
    WEIGHT_DECAY_BIAS,
    WEIGHT_DECAY_WEIGHT,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    is_checkpoint_eligible,
)

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor

PREG1_HEAD_SCHEMA_VERSION = "preg1-head-v1"
"""The one authoritative schema constant for this diagnostic's artifacts."""

PREG1_TOKENIZATION = (
    f"max_length={MAX_LENGTH}, truncation={TRUNCATION}, padding={PADDING!r}, "
    "no word segmentation in either pathway"
)

POOLING_SCOPE_WARNING = PREG1_POOLING_SCOPE

NO_SIGNIFICANCE_TEST = (
    "This diagnostic reports raw scores, per-seed deltas, means and SAMPLE "
    "standard deviations. It defines no significance threshold, computes no "
    "p-value, and declares no pass/fail criterion. Five seeds cannot support a "
    "hypothesis test, and inventing an acceptance threshold after seeing a "
    "burden would make the number decorative."
)

DETERMINISM_SCOPE = (
    "GUARANTEED, and tested: for a given seed the classifier initialisation is "
    "bit-identical, Vanilla and Base-only start from identical parameters, the "
    "batch order is identical, and checkpoint / LR tie-breaks are total orders "
    "with no dependence on mapping insertion order or iteration accidents.\n"
    "NOT GUARANTEED, and not claimed: bitwise identity of trained parameters or "
    "scores across different hardware, CUDA/cuBLAS versions, or PyTorch builds. "
    "Floating-point reduction order on a GPU is not fixed by a seed. A run "
    "reproduced on different hardware should be expected to agree closely, not "
    "exactly, and this module does not pretend otherwise."
)


def require_protocol_settings() -> None:
    """Assert this module implements the settings the protocol actually locks.

    Not decorative. The trainer hard-codes several behaviours -- no clipping, no
    accumulation, no dropout, constant LR -- because implementing a knob for a
    value that is locked invites someone to turn it. This function is the link
    back: if `preg1_protocol` ever changes one of those values, the trainer
    stops matching the spec and this raises, instead of the mismatch surviving
    as a comment that used to be true.
    """
    mismatches: list[str] = []
    if GRADIENT_CLIPPING is not None:
        mismatches.append(f"GRADIENT_CLIPPING={GRADIENT_CLIPPING!r} but the trainer clips nothing")
    if GRADIENT_ACCUMULATION_STEPS != 1:
        mismatches.append(
            f"GRADIENT_ACCUMULATION_STEPS={GRADIENT_ACCUMULATION_STEPS} but the trainer "
            "steps every batch"
        )
    if HEAD_DROPOUT != 0.0:
        mismatches.append(f"HEAD_DROPOUT={HEAD_DROPOUT} but the head is a bare Linear")
    if PADDING != "max_length":
        mismatches.append(f"PADDING={PADDING!r} but the cache is keyed on a fixed shape")
    if mismatches:
        raise EvaluationContractViolation(
            "preg1_head no longer implements the locked protocol: " + "; ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Roles. Official TEST is deliberately not representable here.
# ---------------------------------------------------------------------------
class Preg1Role(Enum):
    """The three data roles this diagnostic may touch.

    **Official TEST has no member.** It is not merely forbidden by a check; it
    cannot be named, so no code path in this module can reach it and no CLI flag
    can carry it. `OFFICIAL_TEST` is absent by construction (D-PREG1-003).
    """

    PROTOCOL_TRAIN = "protocol-train"
    """Head training. Derived from the approved TRAIN pool only."""

    PROTOCOL_DEV = "protocol-dev"
    """Checkpoint selection and LR selection. **Never** official validation."""

    OFFICIAL_VALIDATION = "official-validation"
    """Measurement-dev. Read only AFTER the LR is frozen; never for selection."""

    @property
    def may_train_head(self) -> bool:
        return self is Preg1Role.PROTOCOL_TRAIN

    @property
    def may_select(self) -> bool:
        """Whether this role may drive checkpoint or LR selection."""
        return self is Preg1Role.PROTOCOL_DEV


LABEL_ORDER: tuple[str, ...] = tuple(
    name for name, _ in sorted(LABEL_MAPPING.items(), key=lambda item: item[1])
)
"""Explicit label ordering, by encoded index: negative, neutral, positive.

Metrics are averaged over `range(num_labels)` rather than over the labels that
happen to appear, so a split missing a class scores 0 for it instead of
silently averaging over fewer classes -- which matters here, where `neutral` is
about 4% of the pool.
"""


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitMembership:
    """Verified protocol-train / protocol-dev membership, ids only.

    Holds no text. The join back to text happens against the approved derived
    pool, and this object exists to make that join checkable.
    """

    protocol_train: tuple[str, ...]
    protocol_dev: tuple[str, ...]
    assignment_digest: str
    source_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        for name, ids in (
            ("protocol-train", self.protocol_train),
            ("protocol-dev", self.protocol_dev),
        ):
            if not ids:
                raise EvaluationContractViolation(f"{name} membership is empty")
            duplicates = _duplicates(ids)
            if duplicates:
                raise EvaluationContractViolation(
                    f"{name} has duplicate ids: {duplicates[:10]}"
                )
        overlap = sorted(set(self.protocol_train) & set(self.protocol_dev))
        if overlap:
            raise SplitLeakage(
                f"protocol-train and protocol-dev overlap on {len(overlap)} ids: "
                f"{overlap[:10]}"
            )

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.protocol_train) | set(self.protocol_dev)))

    def ids_for(self, role: Preg1Role) -> tuple[str, ...]:
        if role is Preg1Role.PROTOCOL_TRAIN:
            return self.protocol_train
        if role is Preg1Role.PROTOCOL_DEV:
            return self.protocol_dev
        raise EvaluationContractViolation(
            f"{role.value} has no internal membership; official validation is read "
            "from its own file and is never a partition of the train pool"
        )

    def require_partitions(self, pool_ids: Sequence[str]) -> None:
        """The two parts must be a complete, exact partition of the pool.

        Checked in both directions. A missing id is a silently shrunken training
        set; an unknown id is a join that will fail or, worse, succeed against
        the wrong pool.
        """
        pool = set(pool_ids)
        if len(pool) != len(pool_ids):
            raise EvaluationContractViolation(
                f"pool has duplicate ids: {_duplicates(pool_ids)[:10]}"
            )
        mine = set(self.all_ids)
        missing = sorted(pool - mine)
        unknown = sorted(mine - pool)
        if missing:
            raise EvaluationContractViolation(
                f"{len(missing)} pool ids are in neither part: {missing[:10]}"
            )
        if unknown:
            raise EvaluationContractViolation(
                f"{len(unknown)} membership ids are not in the pool: {unknown[:10]}"
            )
        if len(self.protocol_train) + len(self.protocol_dev) != len(pool):
            raise EvaluationContractViolation(
                "membership is not a partition: parts sum to "
                f"{len(self.protocol_train) + len(self.protocol_dev)} for {len(pool)} ids"
            )


def _duplicates(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for sample_id in ids:
        (repeated if sample_id in seen else seen).add(sample_id)
    return sorted(repeated)


def load_membership(
    directory: str | Path,
    *,
    expected_digests: Mapping[str, str] | None = None,
) -> SplitMembership:
    """Load a `preg1-split-v1` membership directory produced by Audit 023.

    Args:
        expected_digests: filename -> SHA-256. When supplied, every named file
            must match. This is how a run pins itself to the *approved* split
            rather than to any split that happens to be on disk.
    """
    from unmark.evaluation.preg1_split import (
        ID_FILE_NAMES,
        PROTOCOL_DEV,
        PROTOCOL_TRAIN,
        SPLIT_SCHEMA_VERSION,
    )

    root = Path(directory)
    if not root.is_dir():
        raise EvaluationContractViolation(f"membership directory not found: {root}")

    manifest_path = root / "split-manifest.json"
    if not manifest_path.is_file():
        raise EvaluationContractViolation(f"missing split-manifest.json in {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationContractViolation(
            f"split-manifest.json is malformed: {error}"
        ) from error
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise EvaluationContractViolation(
            f"membership schema must be {SPLIT_SCHEMA_VERSION}, got "
            f"{manifest.get('schema_version')!r}"
        )

    if expected_digests:
        for filename, digest in expected_digests.items():
            target = root / filename
            if not target.is_file():
                raise EvaluationContractViolation(f"missing pinned artifact {filename}")
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != digest:
                raise EvaluationContractViolation(
                    f"{filename} digest mismatch: expected {digest}, got {actual}"
                )

    def read_ids(part: str) -> tuple[str, ...]:
        path = root / ID_FILE_NAMES[part]
        if not path.is_file():
            raise EvaluationContractViolation(f"missing {ID_FILE_NAMES[part]}")
        lines = path.read_text(encoding="utf-8").split()
        if not lines:
            raise EvaluationContractViolation(f"{ID_FILE_NAMES[part]} is empty")
        return tuple(lines)

    result = manifest.get("result", {})
    return SplitMembership(
        protocol_train=read_ids(PROTOCOL_TRAIN),
        protocol_dev=read_ids(PROTOCOL_DEV),
        assignment_digest=result.get("assignment_digest", ""),
        source_manifest_digest=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Representation cache provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RepresentationKey:
    """Everything a cached representation tensor is bound to.

    A cache is a correctness hazard: reusing Vanilla vectors as Base-only would
    produce a *zero* burden and look like a finding. So the key carries every
    input that could change the tensor, comparison is exact, and a mismatch is
    an error rather than a recomputation.

    **No raw text.** Sample identity travels as an ordered-id digest, so the key
    pins *which examples in which order* without storing a corpus.
    """

    dataset: str
    dataset_version: str
    task: str
    role: Preg1Role
    pathway: SystemPathway
    source_identity: str
    """Digest of the underlying data file -- the approved pool's SHA-256."""
    ordered_id_digest: str
    """Digest of the ordered sample ids. Order matters: row i must stay row i."""
    tokenizer_id: str
    model_revision: str
    max_length: int
    truncation: bool
    padding: str
    pooling: str
    dtype: str
    hidden_size: int
    count: int
    schema_version: str = PREG1_HEAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "role": self.role.value,
            "pathway": self.pathway.value,
            "source_identity": self.source_identity,
            "ordered_id_digest": self.ordered_id_digest,
            "tokenizer_id": self.tokenizer_id,
            "model_revision": self.model_revision,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "count": self.count,
            "representation_shape": [self.count, self.hidden_size],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepresentationKey:
        try:
            return cls(
                dataset=payload["dataset"],
                dataset_version=payload["dataset_version"],
                task=payload["task"],
                role=Preg1Role(payload["role"]),
                pathway=SystemPathway(payload["pathway"]),
                source_identity=payload["source_identity"],
                ordered_id_digest=payload["ordered_id_digest"],
                tokenizer_id=payload["tokenizer_id"],
                model_revision=payload["model_revision"],
                max_length=payload["max_length"],
                truncation=payload["truncation"],
                padding=payload["padding"],
                pooling=payload["pooling"],
                dtype=payload["dtype"],
                hidden_size=payload["hidden_size"],
                count=payload["count"],
                schema_version=payload["schema_version"],
            )
        except (KeyError, ValueError) as error:
            raise EvaluationContractViolation(
                f"cache metadata is malformed or from an unknown schema: {error}"
            ) from error

    def require_compatible(self, other: RepresentationKey) -> None:
        """Exact match on every field, or fail. No tolerance, no coercion."""
        differences = [
            name
            for name in self.to_dict()
            if self.to_dict()[name] != other.to_dict()[name]
        ]
        if differences:
            detail = ", ".join(
                f"{name}: cached={other.to_dict()[name]!r} wanted={self.to_dict()[name]!r}"
                for name in differences
            )
            raise EvaluationContractViolation(
                f"representation cache is incompatible on {len(differences)} field(s): "
                f"{detail}. Refusing to reuse it -- a cache reused across pathways or "
                "configurations produces a silent, plausible-looking result."
            )


def ordered_id_digest(sample_ids: Sequence[str]) -> str:
    """Digest of ids **in order**. Reordering changes it; that is the point."""
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


class RepresentationCache:
    """Fail-closed cache for frozen-encoder representations.

    Stores `[N, d]` FP32 pooled vectors beside their `RepresentationKey`. The
    tensor file is an experiment resource and is **never committed to git**.
    """

    METADATA_NAME = "representation-key.json"
    TENSOR_NAME = "representations.pt"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def exists(self) -> bool:
        return (self.directory / self.METADATA_NAME).is_file() and (
            self.directory / self.TENSOR_NAME
        ).is_file()

    def read_key(self) -> RepresentationKey:
        path = self.directory / self.METADATA_NAME
        if not path.is_file():
            raise EvaluationContractViolation(f"no cache metadata at {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EvaluationContractViolation(
                f"cache metadata is not valid JSON: {error}"
            ) from error
        return RepresentationKey.from_dict(payload)

    def save(self, key: RepresentationKey, representations: "Tensor") -> None:
        import torch

        _require_fp32_matrix(representations, key)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / self.METADATA_NAME).write_text(
            json.dumps(key.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        torch.save(representations, self.directory / self.TENSOR_NAME)

    def load(self, key: RepresentationKey) -> BoundRepresentations:
        """Load only if the stored key matches **exactly**.

        Returns a `BoundRepresentations`, not a bare tensor: the whole point of
        validating a key is lost if the caller then carries the values away from
        it. The key returned is the one that was checked.
        """
        import torch

        key.require_compatible(self.read_key())
        tensor = torch.load(self.directory / self.TENSOR_NAME, map_location="cpu")
        _require_fp32_matrix(tensor, key)
        return BoundRepresentations(values=tensor, key=key)


@dataclass(frozen=True)
class BoundRepresentations:
    """A representation tensor **and** the provenance saying what it is.

    The reason this type exists: an earlier version passed a bare tensor
    alongside a free `dev_role` argument, so the role was a *claim about* the
    tensor rather than a *property of* it. A caller could hand official
    validation features to checkpoint selection while declaring
    `PROTOCOL_DEV`, and every guard would pass. The role now comes from the
    `RepresentationKey` that was validated when the tensor was produced or
    loaded, and there is no separate role argument anywhere to contradict it.

    Validation is duck-typed on `.shape` and `.dtype` so the binding contract is
    testable without torch; the values are real tensors in practice.
    """

    values: "Tensor"
    key: RepresentationKey

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
                f"representation dtype {dtype!r} contradicts its key's "
                f"{self.key.dtype!r} (FP32 is required; no AMP)"
            )
        if getattr(self.values, "requires_grad", False):
            raise EvaluationContractViolation(
                "representations must be detached: a gradient path back to the "
                "encoder would make this fine-tuning, not a frozen-encoder probe"
            )

    @property
    def role(self) -> Preg1Role:
        return self.key.role

    @property
    def pathway(self) -> SystemPathway:
        return self.key.pathway

    def __len__(self) -> int:
        return self.key.count

    def require_role(self, expected: Preg1Role, purpose: str) -> None:
        """Fail closed unless the **bound** role is the one this use permits."""
        if self.role is not expected:
            raise SplitLeakage(
                f"{purpose} requires representations bound to {expected.value}, but "
                f"these are bound to {self.role.value}. The role travels with the "
                "tensor and cannot be overridden by an argument."
            )

    def require_same_pathway(self, other: "BoundRepresentations") -> None:
        if self.pathway is not other.pathway:
            raise SplitLeakage(
                f"pathway mismatch: {self.pathway.value} vs {other.pathway.value}. "
                "A head fitted to one pathway's representation geometry has no "
                "defined meaning applied to another's."
            )

    def require_same_geometry(self, other: "BoundRepresentations") -> None:
        if self.key.hidden_size != other.key.hidden_size:
            raise EvaluationContractViolation(
                f"hidden size mismatch: {self.key.hidden_size} vs {other.key.hidden_size}"
            )
        for field_name in ("tokenizer_id", "model_revision", "max_length", "pooling",
                           "truncation", "padding", "dtype", "schema_version"):
            mine = getattr(self.key, field_name)
            theirs = getattr(other.key, field_name)
            if mine != theirs:
                raise EvaluationContractViolation(
                    f"representation sets disagree on {field_name}: {mine!r} vs {theirs!r}"
                )


def _require_fp32_matrix(tensor: "Tensor", key: RepresentationKey) -> None:
    import torch

    if tensor.dtype is not torch.float32:
        raise EvaluationContractViolation(
            f"representations must be FP32 (no AMP), got {tensor.dtype}"
        )
    if tuple(tensor.shape) != (key.count, key.hidden_size):
        raise EvaluationContractViolation(
            f"representation shape {tuple(tensor.shape)} does not match the key's "
            f"{(key.count, key.hidden_size)}"
        )


# ---------------------------------------------------------------------------
# Frozen encoder and pooling
# ---------------------------------------------------------------------------
def require_frozen_encoder(encoder: Any) -> None:
    """Every parameter frozen and the module in eval mode, or fail.

    Checked before extraction rather than trusted: `requires_grad` is easy to
    leave on, and a single trainable encoder parameter would turn this from a
    frozen-representation probe into a fine-tuning experiment.
    """
    trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    if trainable:
        raise EvaluationContractViolation(
            f"{len(trainable)} encoder parameter(s) require grad, e.g. {trainable[:5]}. "
            "The pre-G1 diagnostic uses a FROZEN encoder."
        )
    if encoder.training:
        raise EvaluationContractViolation(
            "encoder must be in eval mode: dropout active during extraction would "
            "make the cached representations nondeterministic"
        )


def PREG1_ONLY_first_token_pool(hidden_states: "Tensor") -> "Tensor":
    """`<s>` pooling: `[N, L, d] -> [N, d]`, taking position 0.

    **Scoped to this pre-G1 diagnostic only.** The name says so on purpose.
    Final Stage-2 pooling for the full grid remains **OPEN** (D-G1-005); this
    resolves pooling for the burden diagnostic and for nothing else, and no
    other module may import it as "the" pooling rule.
    """
    if hidden_states.dim() != 3:
        raise EvaluationContractViolation(
            f"expected [N, L, d] hidden states, got {tuple(hidden_states.shape)}"
        )
    return hidden_states[:, 0, :].contiguous()


def extract_bound_representations(
    encoder: Any,
    input_ids: "Tensor",
    attention_mask: "Tensor",
    key: RepresentationKey,
) -> BoundRepresentations:
    """Fresh extraction, returned in the same bound shape a cache load gives.

    Fresh and cached representations are therefore interchangeable at the type
    level, and neither can reach the trainer without provenance.
    """
    values = extract_representations(encoder, input_ids, attention_mask)
    return BoundRepresentations(values=values, key=key)


def extract_representations(
    encoder: Any,
    input_ids: "Tensor",
    attention_mask: "Tensor",
) -> "Tensor":
    """Frozen-encoder `<s>` representations, FP32, under `no_grad`.

    Returns a detached FP32 `[N, d]` tensor with no graph attached, so a later
    `backward()` in the head trainer cannot reach the encoder even if a caller
    forgets to guard it.
    """
    import torch

    require_frozen_encoder(encoder)
    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", outputs)
        pooled = PREG1_ONLY_first_token_pool(hidden)
    return pooled.detach().to(torch.float32)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
def build_head(hidden_size: int, seed: int, num_labels: int = PRIMARY_NUM_LABELS):
    """`Linear(d, num_labels, bias=True)`, Xavier-uniform weight, zero bias.

    The RNG is reset from `seed` **immediately before** initialisation, and the
    initialisation is applied explicitly rather than relying on `nn.Linear`'s
    default -- which is a Kaiming variant whose exact form has changed across
    PyTorch versions, and would make the comparison silently version-sensitive.

    Resetting here is what makes the paired guarantee hold: Vanilla and
    Base-only call this with the same seed and get bit-identical parameters,
    regardless of which arm runs first or what consumed RNG in between.
    """
    import torch
    from torch import nn

    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise EvaluationContractViolation(
            f"hidden_size must be a positive int, got {hidden_size!r}"
        )
    generator_state = torch.Generator()
    generator_state.manual_seed(seed)
    torch.manual_seed(seed)

    head = nn.Linear(hidden_size, num_labels, bias=True)
    with torch.no_grad():
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
    return head


def build_optimizer(head: Any, learning_rate: float):
    """AdamW with the locked settings and **two** parameter groups.

    Weight decay applies to the weight matrix and **not** to the bias. Decaying
    a 3-element bias is not a meaningful regulariser, and letting it happen by
    default would be an unrecorded hyperparameter difference from the spec.
    """
    import torch

    if not isinstance(learning_rate, float) or learning_rate <= 0:
        raise EvaluationContractViolation(
            f"learning_rate must be a positive float, got {learning_rate!r}"
        )
    groups = [
        {"params": [head.weight], "weight_decay": WEIGHT_DECAY_WEIGHT},
        {"params": [head.bias], "weight_decay": WEIGHT_DECAY_BIAS},
    ]
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        amsgrad=AMSGRAD,
    )


def deterministic_batches(count: int, seed: int, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Shuffled index batches from a dedicated generator.

    Uses its own `random.Random`, not the ambient global RNG, so batch order
    depends on the seed and on nothing else -- not on what initialised the head,
    not on which arm ran first. `drop_last` is False, so every example is seen
    in every epoch.
    """
    import random

    if count <= 0:
        raise EvaluationContractViolation("cannot batch an empty split")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, count, batch_size)]


# ---------------------------------------------------------------------------
# Scores and checkpoint selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpochScore:
    """One epoch's protocol-dev score. Epoch numbering starts at 1."""

    epoch: int
    macro_f1: float
    accuracy: float

    def __post_init__(self) -> None:
        if not is_checkpoint_eligible(self.epoch):
            raise EvaluationContractViolation(
                f"epoch {self.epoch} is not checkpoint-eligible; eligible epochs are "
                f"{CHECKPOINT_ELIGIBLE_EPOCHS[0]}..{CHECKPOINT_ELIGIBLE_EPOCHS[-1]} "
                "(epoch 0, the untrained head, never is)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "macro_f1": self.macro_f1, "accuracy": self.accuracy}


def select_checkpoint(scores: Sequence[EpochScore]) -> EpochScore:
    """Highest macro-F1, then highest accuracy, then **earliest** epoch.

    A total order, evaluated by sorting on an explicit key rather than by
    scanning and comparing -- a scan with `>` keeps the first maximum it meets
    and a scan with `>=` keeps the last, so the tie-break would be decided by
    iteration order rather than by the rule.

    This is **checkpoint selection, not early stopping.** All epochs run; this
    picks among their scores afterwards.
    """
    if not scores:
        raise EvaluationContractViolation("cannot select a checkpoint from no scores")
    epochs = [score.epoch for score in scores]
    if len(set(epochs)) != len(epochs):
        raise EvaluationContractViolation(f"duplicate epochs in scores: {sorted(epochs)}")
    return min(scores, key=lambda s: (-s.macro_f1, -s.accuracy, s.epoch))


def require_full_schedule(scores: Sequence[EpochScore], epochs: int = EPOCHS) -> None:
    """Every epoch 1..N must be present. Guards against early stopping.

    `EARLY_STOPPING` is False in the protocol, but a truncated score list is
    what early stopping would actually look like downstream, so the runner
    checks the evidence rather than the flag.
    """
    present = sorted(score.epoch for score in scores)
    if present != list(range(1, epochs + 1)):
        missing = sorted(set(range(1, epochs + 1)) - set(present))
        raise EvaluationContractViolation(
            f"all {epochs} epochs must run -- there is no early stopping. "
            f"Missing: {missing[:10]}"
        )


def score_predictions(predictions: Sequence[int], labels: Sequence[int]) -> tuple[float, float]:
    """`(macro_f1, accuracy)` over the explicit label ordering."""
    return (
        macro_f1(predictions, labels, num_labels=PRIMARY_NUM_LABELS),
        accuracy(predictions, labels),
    )


# ---------------------------------------------------------------------------
# Head training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadRun:
    """One completed head-training run: its identity and its per-epoch scores."""

    pathway: SystemPathway
    learning_rate: float
    seed: int
    scores: tuple[EpochScore, ...]
    hidden_size: int
    scored_on: Preg1Role = Preg1Role.PROTOCOL_DEV
    """Which role produced these scores. Recorded rather than assumed, so an
    artifact says on its face what the checkpoint was selected against."""

    def __post_init__(self) -> None:
        require_full_schedule(self.scores)
        if not self.scored_on.may_select:
            raise SplitLeakage(
                f"a head run scored on {self.scored_on.value} cannot drive selection; "
                f"only {Preg1Role.PROTOCOL_DEV.value} may"
            )

    @property
    def selected(self) -> EpochScore:
        return select_checkpoint(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway.value,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "scored_on": self.scored_on.value,
            "hidden_size": self.hidden_size,
            "epochs_run": len(self.scores),
            "selected_epoch": self.selected.epoch,
            "selected_macro_f1": self.selected.macro_f1,
            "selected_accuracy": self.selected.accuracy,
            "per_epoch": [score.to_dict() for score in self.scores],
        }


def require_training_roles(
    train: BoundRepresentations, dev: BoundRepresentations
) -> None:
    """The two role checks that make the diagnostic mean what it claims.

    Both roles come from provenance. **There is no argument by which a caller
    can declare a role**, so a mislabelled tensor cannot pass by assertion --
    only by having been produced or loaded under a key that already said so.
    """
    train.require_role(Preg1Role.PROTOCOL_TRAIN, "head training")
    dev.require_role(Preg1Role.PROTOCOL_DEV, "checkpoint selection")
    train.require_same_pathway(dev)
    train.require_same_geometry(dev)


def train_head(
    train: BoundRepresentations,
    train_labels: Sequence[int],
    dev: BoundRepresentations,
    dev_labels: Sequence[int],
    *,
    learning_rate: float,
    seed: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    on_epoch: Callable[[int, EpochScore], None] | None = None,
) -> HeadRun:
    """Train the linear head for **all** `epochs`, scoring after each one.

    Every epoch executes; nothing stops early. Selection happens afterwards,
    from the returned scores.

    **Roles and pathway are read from the representations' provenance**, not
    passed in. Training representations must be bound to `PROTOCOL_TRAIN` and
    the scoring set to `PROTOCOL_DEV`; official validation reaching checkpoint
    selection is the single substitution that would invalidate the diagnostic
    while leaving every number looking reasonable, and it is now impossible on
    the supported path rather than merely checked.
    """
    import torch
    from torch import nn

    require_training_roles(train, dev)
    train_features, dev_features = train.values, dev.values
    pathway = train.pathway

    if train_features.dim() != 2 or dev_features.dim() != 2:
        raise EvaluationContractViolation("features must be [N, d]")
    if train_features.shape[0] != len(train_labels):
        raise EvaluationContractViolation("train features and labels differ in length")
    if dev_features.shape[0] != len(dev_labels):
        raise EvaluationContractViolation("dev features and labels differ in length")
    if train_features.dtype is not torch.float32 or dev_features.dtype is not torch.float32:
        raise EvaluationContractViolation("features must be FP32 (no AMP)")

    require_protocol_settings()
    hidden_size = int(train_features.shape[1])
    head = build_head(hidden_size, seed)
    optimizer = build_optimizer(head, learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # unweighted, no label smoothing
    train_y = torch.as_tensor(list(train_labels), dtype=torch.long)
    dev_y = torch.as_tensor(list(dev_labels), dtype=torch.long)

    scores: list[EpochScore] = []
    for epoch in range(1, epochs + 1):
        head.train()
        # Batch order is derived from (seed, epoch) so the schedule differs across
        # epochs but is identical for both pathways under the same seed.
        for batch in deterministic_batches(
            len(train_labels), seed * 1000 + epoch, batch_size
        ):
            index = torch.as_tensor(batch, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_features[index])
            loss = loss_fn(logits, train_y[index])
            loss.backward()
            optimizer.step()  # no clipping, no accumulation, constant LR
        head.eval()
        with torch.no_grad():
            predictions = head(dev_features).argmax(dim=1).tolist()
        f1, acc = score_predictions(predictions, dev_y.tolist())
        score = EpochScore(epoch=epoch, macro_f1=f1, accuracy=acc)
        scores.append(score)
        if on_epoch is not None:
            on_epoch(epoch, score)

    return HeadRun(
        pathway=pathway,
        learning_rate=learning_rate,
        seed=seed,
        scores=tuple(scores),
        hidden_size=hidden_size,
        scored_on=dev.role,
    )


# ---------------------------------------------------------------------------
# LR selection -- Vanilla only
# ---------------------------------------------------------------------------
def sample_stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (n-1). Zero for a single observation.

    `statistics.stdev`, not `pstdev`: three tuning seeds are a *sample* of the
    seed distribution, not the population, and the population form would
    understate the spread by a factor of sqrt(2/3) here -- which is exactly the
    quantity the third tie-break turns on.
    """
    if not values:
        raise EvaluationContractViolation("cannot take the SD of no values")
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


@dataclass(frozen=True)
class LrCandidate:
    """One learning rate's aggregate over the tuning seeds. Vanilla only."""

    learning_rate: float
    runs: tuple[HeadRun, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise EvaluationContractViolation("an LR candidate needs at least one run")
        for run in self.runs:
            if run.pathway is not SystemPathway.VANILLA:
                raise SplitLeakage(
                    f"primary LR selection accepts VANILLA runs only, got "
                    f"{run.pathway.value}. Letting Base-only influence the shared LR "
                    "would tune the protocol on the arm being measured."
                )
            if run.learning_rate != self.learning_rate:
                raise EvaluationContractViolation(
                    f"run LR {run.learning_rate} does not match candidate "
                    f"{self.learning_rate}"
                )
            if run.scored_on is not Preg1Role.PROTOCOL_DEV:
                raise SplitLeakage(
                    f"LR selection accepts runs scored on "
                    f"{Preg1Role.PROTOCOL_DEV.value} only, got {run.scored_on.value}. "
                    "Official validation is measurement-dev; selecting on it would "
                    "tune the protocol on the set the result is reported from."
                )
        seeds = [run.seed for run in self.runs]
        if len(set(seeds)) != len(seeds):
            raise EvaluationContractViolation(f"duplicate tuning seeds: {sorted(seeds)}")

    @property
    def mean_macro_f1(self) -> float:
        return statistics.fmean(run.selected.macro_f1 for run in self.runs)

    @property
    def mean_accuracy(self) -> float:
        return statistics.fmean(run.selected.accuracy for run in self.runs)

    @property
    def stdev_macro_f1(self) -> float:
        return sample_stdev([run.selected.macro_f1 for run in self.runs])

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "seeds": sorted(run.seed for run in self.runs),
            "mean_macro_f1": self.mean_macro_f1,
            "mean_accuracy": self.mean_accuracy,
            "sample_stdev_macro_f1": self.stdev_macro_f1,
            "per_seed": [run.to_dict() for run in self.runs],
        }


def select_learning_rate(
    candidates: Iterable[LrCandidate],
    *,
    require_full_grid: bool = True,
    expected_seeds: Sequence[int] = TUNING_SEEDS,
) -> LrCandidate:
    """Pick the primary LR: mean F1, then mean accuracy, then lower sample SD,
    then smaller LR.

    Ordering is computed from a sort key over candidate **content**, so the
    result cannot depend on the order the candidates were supplied or on any
    mapping's insertion order.
    """
    ordered = list(candidates)
    if not ordered:
        raise EvaluationContractViolation("cannot select an LR from no candidates")
    rates = [candidate.learning_rate for candidate in ordered]
    if len(set(rates)) != len(rates):
        raise EvaluationContractViolation(f"duplicate learning rates: {sorted(rates)}")
    if require_full_grid and sorted(rates) != sorted(LR_GRID):
        raise EvaluationContractViolation(
            f"the primary grid is {sorted(LR_GRID)}; got {sorted(rates)}. The grid is "
            "precommitted and is not altered after seeing results."
        )
    if expected_seeds is not None:
        for candidate in ordered:
            seeds = sorted(run.seed for run in candidate.runs)
            if seeds != sorted(expected_seeds):
                raise EvaluationContractViolation(
                    f"LR {candidate.learning_rate} used seeds {seeds}, expected "
                    f"{sorted(expected_seeds)}"
                )
    return min(
        ordered,
        key=lambda c: (-c.mean_macro_f1, -c.mean_accuracy, c.stdev_macro_f1, c.learning_rate),
    )


@dataclass(frozen=True)
class FrozenLearningRate:
    """The LR after selection. Exists so "frozen" is a value, not a promise.

    The paired measurement takes this object rather than a bare float, so a
    measurement cannot be run with an LR that was never selected on Vanilla.
    """

    value: float
    selected_on: SystemPathway = SystemPathway.VANILLA
    grid: tuple[float, ...] = LR_GRID
    tuning_seeds: tuple[int, ...] = TUNING_SEEDS

    def __post_init__(self) -> None:
        if self.selected_on is not SystemPathway.VANILLA:
            raise SplitLeakage(
                "the primary shared LR is selected on VANILLA only "
                f"(got {self.selected_on.value})"
            )
        if self.value not in self.grid:
            raise EvaluationContractViolation(
                f"{self.value} is not in the precommitted grid {sorted(self.grid)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.value,
            "selected_on": self.selected_on.value,
            "grid": list(self.grid),
            "tuning_seeds": list(self.tuning_seeds),
        }


def freeze_learning_rate(winner: LrCandidate) -> FrozenLearningRate:
    return FrozenLearningRate(value=winner.learning_rate)


def score_measurement(
    head: Any, measurement: BoundRepresentations, labels: Sequence[int]
) -> tuple[float, float]:
    """Score a trained head on **official-validation-bound** representations.

    The final measurement is reported on measurement-dev and nothing else.
    Protocol-dev is refused here for the mirror-image reason official validation
    is refused during selection: reporting the headline number on the set that
    chose the checkpoint would report the selection, not the pathway.
    """
    import torch

    measurement.require_role(
        Preg1Role.OFFICIAL_VALIDATION, "the primary paired measurement"
    )
    if measurement.values.shape[0] != len(labels):
        raise EvaluationContractViolation(
            "measurement features and labels differ in length"
        )
    head.eval()
    with torch.no_grad():
        predictions = head(measurement.values).argmax(dim=1).tolist()
    return score_predictions(predictions, list(labels))


# ---------------------------------------------------------------------------
# Paired measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedSeedResult:
    """One measurement seed: both arms, from identical initial parameters."""

    seed: int
    vanilla_macro_f1: float
    vanilla_accuracy: float
    base_only_macro_f1: float
    base_only_accuracy: float

    @property
    def macro_f1_delta(self) -> float:
        """Vanilla minus Base-only. Positive means Base-only scored lower."""
        return self.vanilla_macro_f1 - self.base_only_macro_f1

    @property
    def accuracy_delta(self) -> float:
        return self.vanilla_accuracy - self.base_only_accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "vanilla_macro_f1": self.vanilla_macro_f1,
            "vanilla_accuracy": self.vanilla_accuracy,
            "base_only_macro_f1": self.base_only_macro_f1,
            "base_only_accuracy": self.base_only_accuracy,
            "macro_f1_delta": self.macro_f1_delta,
            "accuracy_delta": self.accuracy_delta,
        }


@dataclass(frozen=True)
class PairedDiagnostic:
    """The descriptive pre-G1 burden report. **No hypothesis test.**"""

    learning_rate: FrozenLearningRate
    results: tuple[PairedSeedResult, ...]
    measured_on: Preg1Role = Preg1Role.OFFICIAL_VALIDATION
    schema_version: str = PREG1_HEAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.measured_on is not Preg1Role.OFFICIAL_VALIDATION:
            raise EvaluationContractViolation(
                "the primary paired measurement is reported on official validation "
                f"(measurement-dev), not {self.measured_on.value}"
            )
        seeds = sorted(result.seed for result in self.results)
        if seeds != sorted(MEASUREMENT_SEEDS):
            raise EvaluationContractViolation(
                f"the paired measurement uses the precommitted seeds "
                f"{sorted(MEASUREMENT_SEEDS)}, got {seeds}"
            )

    def to_dict(self) -> dict[str, Any]:
        f1_deltas = [r.macro_f1_delta for r in self.results]
        acc_deltas = [r.accuracy_delta for r in self.results]
        return {
            "schema_version": self.schema_version,
            "dataset": PRIMARY_DATASET,
            "dataset_version": PRIMARY_DATASET_VERSION,
            "task": PRIMARY_TASK,
            "measured_on": self.measured_on.value,
            "learning_rate": self.learning_rate.to_dict(),
            "pooling": PREG1_POOLING.value,
            "pooling_scope": PREG1_POOLING_SCOPE,
            "encoder": {"checkpoint": ENCODER_CHECKPOINT, "revision": ENCODER_REVISION},
            "per_seed": [result.to_dict() for result in self.results],
            "vanilla": _aggregate([r.vanilla_macro_f1 for r in self.results],
                                  [r.vanilla_accuracy for r in self.results]),
            "base_only": _aggregate([r.base_only_macro_f1 for r in self.results],
                                    [r.base_only_accuracy for r in self.results]),
            "delta_vanilla_minus_base_only": {
                "mean_macro_f1": statistics.fmean(f1_deltas),
                "sample_stdev_macro_f1": sample_stdev(f1_deltas),
                "mean_accuracy": statistics.fmean(acc_deltas),
                "sample_stdev_accuracy": sample_stdev(acc_deltas),
            },
            "interpretation": NO_SIGNIFICANCE_TEST,
            "determinism": DETERMINISM_SCOPE,
        }


def _aggregate(f1_values: Sequence[float], accuracy_values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_macro_f1": statistics.fmean(f1_values),
        "sample_stdev_macro_f1": sample_stdev(f1_values),
        "mean_accuracy": statistics.fmean(accuracy_values),
        "sample_stdev_accuracy": sample_stdev(accuracy_values),
    }
