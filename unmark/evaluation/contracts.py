"""Pure-data contracts for downstream (Stage-2) evaluation.

**No torch.** Task/split identity, system pathways, head configuration, and the
register of values the proposal leaves OPEN.

Scope: the minimum needed for the **Vanilla vs Base-only** fail-fast diagnostic
that proposal §4.5 identifies as "exactly hypothesis H1, and exactly what G1
measures". It is deliberately not a benchmark platform.

The rule carried over from Stage-1: **an API default is a scientific decision if
it can reach an experiment.** §5's open-items table names "Classification head
concrete values" as blocking **G1** specifically, so those values are required
arguments here, never defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class Split(Enum):
    """Proposal §5.4 split discipline.

    | Split | Permitted use |
    |---|---|
    | train | module training, head training |
    | dev | every architecture and hyperparameter decision, all ablations |
    | test | one final evaluation, for the tables that appear in the paper |
    """

    TRAIN = "TRAIN"
    DEV = "DEV"
    TEST = "TEST"

    @property
    def may_train_head(self) -> bool:
        return self is Split.TRAIN

    @property
    def is_final_evaluation(self) -> bool:
        return self is Split.TEST


class SystemPathway(Enum):
    """Input pathways compared downstream.

    Only the two the H1 diagnostic needs are defined. `RESTORE` and `ALIGN` are
    §6.4 systems but are **not implemented here** and are deliberately absent
    rather than stubbed.
    """

    VANILLA = "VANILLA"
    """§6.4 `UPPER`/`FLOOR`: the unmodified model on its own tokenization of the
    text as given. Clean marked text at `FULL`; corrupted text at other
    conditions."""

    BASE_ONLY = "BASE_ONLY"
    """`b(x)` through the frozen tokenizer and frozen encoder, with **no
    adapter and no orthography channels**.

    This is the `g -> 0` pathway of §4.5, **not UNMARK**. §4.5: "the gate
    recovers the **base-only pathway**, not the original model." Calling it
    UNMARK would claim a result about a module that is not present.
    """

    @property
    def uses_base_text(self) -> bool:
        return self is SystemPathway.BASE_ONLY

    @property
    def uses_orthography_channels(self) -> bool:
        """Neither pathway does. `BASE_ONLY` has no channels by definition, and
        restoring or guessing marks is forbidden on both."""
        return False


class EvaluationContractViolation(ValueError):
    """Raised when downstream inputs contradict a locked contract."""


class SplitLeakage(EvaluationContractViolation):
    """Raised on a train/dev/test discipline violation (§5.4)."""


class UnresolvedEvaluationValue(RuntimeError):
    """Raised when a scientifically OPEN value is needed but was never supplied."""


# ---------------------------------------------------------------------------
# What the proposal locks, and what it does not
# ---------------------------------------------------------------------------
STAGE1_POOLING_DOES_NOT_TRANSFER = (
    "§4.6 locks attention-masked mean over non-special content tokens for the "
    "**Stage-1 alignment objective**. §5.2 lists the **classification head's** pooling "
    "among the values pinned during spec lock, and it is still OPEN. These are two "
    "different decisions about two different things; the first does not settle the "
    "second merely by existing first."
)
"""Why Stage-2 extraction does not pool. See D-G1-005."""


LOCKED_EVALUATION_VALUES: dict[str, str] = {
    "task_level_metrics": "macro-F1 and accuracy per task and condition (§6.5)",
    "headline_metric": "GRR = (S_system - S_FLOOR) / (S_UPPER - S_FLOOR) (§6.5)",
    "grr_anchors": (
        "§6.4: UPPER = clean input, unmodified model; FLOOR = corrupted input, "
        "unmodified model -- so both anchors are the VANILLA pathway"
    ),
    "head_trained_on_clean_only": (
        "§5.2 and §8.3: the head is trained on clean data only, then frozen and "
        "evaluated under every condition. Locked as a substantive experimental "
        "decision, not a detail"
    ),
    "head_protocol_identical_across_systems": (
        "§5.2: one architecture, identical across all five systems, with the same "
        "schedule, epoch budget, early-stopping criterion and seed list; §8.3: "
        "'Run identically for all five systems'"
    ),
    "split_discipline": "§5.4: train / dev / test permitted uses",
    "encoder_frozen": "§5.1: fully frozen; no layer unfrozen without a logged decision",
    "corruption_conditions": "§6.3: FULL, P25, P50, P75, P100, STRIP-ALL, VARIANT",
    "seed_minimum": "§6.6: at least three seeds per configuration; report mean and sd",
}

OPEN_EVALUATION_VALUES: dict[str, str] = {
    "task_dataset": (
        "§6.2 names four task *categories* (emotion recognition, hate-speech "
        "detection, sentiment analysis, spam-review detection) but no dataset. "
        "§5's open-items table: 'Dataset versions and splits' blocks G2 and the "
        "full grid; §13 item 2 repeats it. Not chosen here."
    ),
    "g1_task_choice": (
        "§7 G1 says 'evaluate on one classification task' without naming which."
    ),
    "head_architecture": (
        "§5.2: concrete values (hidden size, pooling) 'are pinned during spec "
        "lock'. §5's table names 'Classification head concrete values' as "
        "blocking **G1**."
    ),
    "head_pooling": (
        "§5.2 lists the classification head's pooling among the concrete values "
        "'pinned during spec lock'; §13 item 4 repeats it. The §4.6 masked-mean rule "
        "is the **Stage-1 alignment** pooling and does NOT transfer to Stage-2. "
        "Extraction therefore returns unpooled hidden states."
    ),
    "head_optimizer": "not specified.",
    "head_learning_rate": "§5.2 lists it among values pinned during spec lock.",
    "head_batch_size": "not specified.",
    "head_epochs": "§5.2 'epoch budget' -- named but not valued.",
    "head_early_stopping": "§5.2 'early-stopping criterion' / 'patience' -- named, not valued.",
    "seed_list": "§6.6 fixes a minimum of three; §5.2 requires a pinned list. No list given.",
    "max_length": "§5.3 pins a maximum sequence length per task; no value is given.",
    "checkpoint_selection": "not specified for the head.",
    "g1_pass_threshold_precision": (
        "§7 G1: 'within ~1 point of the unmodified model'. The '~' is not a "
        "decision rule, and the metric it applies to (accuracy or macro-F1) is "
        "not stated."
    ),
    "grr_degenerate_denominator_policy": (
        "§6.5 gives no policy for S_UPPER == S_FLOOR. No epsilon or fallback is "
        "invented here; GRR is reported undefined."
    ),
    "backbone_finalisation": "D-B3B0-002 is OPEN.",
}


def require_resolved(name: str) -> None:
    """Refuse to proceed on a value the project has not decided."""
    if name in OPEN_EVALUATION_VALUES:
        raise UnresolvedEvaluationValue(
            f"{name} is scientifically OPEN: {OPEN_EVALUATION_VALUES[name]} "
            "Supply it explicitly; it must not acquire a default."
        )


# ---------------------------------------------------------------------------
# Task data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskExample:
    """One labelled downstream example.

    Dataset-agnostic on purpose: no benchmark name appears in this library, so
    selecting a task later does not require rewriting the evaluator.
    """

    sample_id: str
    text: str
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise EvaluationContractViolation(
                "sample_id must be a non-empty stable identity; row order is not an identity"
            )
        if not isinstance(self.text, str):
            raise EvaluationContractViolation("text must be a string")
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise EvaluationContractViolation(f"label must be an int, got {self.label!r}")


@dataclass(frozen=True)
class TaskSplit:
    """A named split of one task. Carries its `Split` identity explicitly.

    The identity travels with the data so that a head cannot be trained on a
    split without the API being able to see which split it was.
    """

    task_id: str
    split: Split
    examples: tuple[TaskExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise EvaluationContractViolation("task_id must be a non-empty string")
        if not isinstance(self.split, Split):
            raise EvaluationContractViolation("split must be a Split; it has no default")
        seen: set[str] = set()
        for example in self.examples:
            if example.sample_id in seen:
                raise EvaluationContractViolation(
                    f"duplicate sample_id {example.sample_id!r} within {self.task_id}/{self.split.value}"
                )
            seen.add(example.sample_id)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(e.sample_id for e in self.examples)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(e.label for e in self.examples)

    def require_trainable(self) -> None:
        """§5.4: only `train` may train a head."""
        if not self.split.may_train_head:
            raise SplitLeakage(
                f"cannot train a head on {self.split.value}: proposal §5.4 permits head "
                "training on train only. dev is for decisions and ablations; test is for "
                "one final evaluation."
            )


def assert_disjoint_splits(splits: Iterable[TaskSplit]) -> None:
    """No `sample_id` may appear in two splits of the same task.

    A shared id between train and test is the cheapest possible leak and the
    hardest to notice once representations are cached under those ids.
    """
    seen: dict[str, tuple[str, str]] = {}
    for split in splits:
        for sample_id in split.sample_ids:
            key = f"{split.task_id}::{sample_id}"
            if key in seen:
                first_task, first_split = seen[key]
                raise SplitLeakage(
                    f"sample_id {sample_id!r} appears in both {first_task}/{first_split} "
                    f"and {split.task_id}/{split.split.value}"
                )
            seen[key] = (split.task_id, split.split.value)


# ---------------------------------------------------------------------------
# Head and evaluation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadConfig:
    """Classification-head configuration. **Every field is required.**

    §5.2: "The concrete values (hidden size, pooling, learning rate, epochs,
    patience) are pinned during spec lock; 'identical' is not a specification
    until the numbers are written down." §5's open-items table names these as
    blocking **G1**.

    So this record exists to *carry* a decision, not to make one. Constructing it
    requires stating every value; nothing here supplies one.
    """

    pooling: str
    """The Stage-2 head's pooling rule. **OPEN** (§5.2 pins it during spec lock).

    Required and load-bearing: `encoder_hidden_states` returns unpooled
    `[N, L, d]`, so a head must apply *this* rule. Nothing in the harness pools
    on a scientific path, which is what stops the Stage-1 §4.6 masked-mean rule
    from becoming a Stage-2 decision by inheritance.

    A name prefixed `TEST_ONLY_` marks a diagnostic placeholder and is rejected
    by a `SCIENTIFIC` run configuration.
    """
    hidden_size: int | None
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int | None
    max_length: int
    seed: int
    num_labels: int

    TEST_ONLY_PREFIX = "TEST_ONLY_"

    @property
    def pooling_is_test_only(self) -> bool:
        return self.pooling.startswith(HeadConfig.TEST_ONLY_PREFIX)

    def __post_init__(self) -> None:
        if not isinstance(self.pooling, str) or not self.pooling:
            raise EvaluationContractViolation(
                "pooling must be a non-empty string; Stage-2 pooling is OPEN (§5.2) and "
                "has no default"
            )
        if self.num_labels < 2:
            raise EvaluationContractViolation(
                f"num_labels must be at least 2, got {self.num_labels}"
            )
        for name in ("batch_size", "epochs", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EvaluationContractViolation(f"{name} must be a positive int, got {value!r}")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise EvaluationContractViolation(
                f"learning_rate must be a positive float, got {self.learning_rate!r}"
            )
        if self.hidden_size is not None and (
            isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int)
            or self.hidden_size <= 0
        ):
            raise EvaluationContractViolation(f"hidden_size must be a positive int or None")

    def identical_protocol_to(self, other: HeadConfig) -> bool:
        """§5.2 requires the head protocol to be identical across systems.

        `num_labels` is a property of the task, not of the protocol, so it is
        compared too only when the same task is meant -- callers comparing across
        systems on one task will have it equal anyway.
        """
        return self == other

    def to_dict(self) -> dict[str, Any]:
        return {
            "pooling": self.pooling,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "max_length": self.max_length,
            "seed": self.seed,
            "num_labels": self.num_labels,
        }


class EvaluationPurpose(Enum):
    """Why an evaluation configuration exists.

    Mirrors `Stage1Purpose` and B2's `CorruptionPurpose`: a `SCIENTIFIC`
    configuration cannot be built while the values it needs are OPEN, so a
    diagnostic number cannot drift into a real experiment.
    """

    DIAGNOSTIC = "DIAGNOSTIC"
    SCIENTIFIC = "SCIENTIFIC"


SCIENTIFIC_REQUIRED_VALUES: tuple[str, ...] = (
    "task_dataset",
    "head_architecture",
    "head_pooling",
    "head_optimizer",
    "head_learning_rate",
    "head_batch_size",
    "head_epochs",
    "head_early_stopping",
    "seed_list",
    "max_length",
)


@dataclass(frozen=True)
class EvaluationRunConfig:
    """A downstream evaluation configuration, stamped with why it exists."""

    purpose: EvaluationPurpose
    task_id: str
    head: HeadConfig
    pathways: tuple[SystemPathway, ...]
    resolved_values: frozenset[str] = frozenset()
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, EvaluationPurpose):
            raise EvaluationContractViolation("purpose must be an EvaluationPurpose")
        if not self.pathways:
            raise EvaluationContractViolation("at least one pathway is required")
        unknown = set(self.resolved_values) - set(OPEN_EVALUATION_VALUES)
        if unknown:
            raise EvaluationContractViolation(
                f"resolved_values names items not in the OPEN register: {sorted(unknown)}"
            )
        if self.purpose is EvaluationPurpose.SCIENTIFIC:
            if self.head.pooling_is_test_only:
                raise UnresolvedEvaluationValue(
                    f"head pooling {self.head.pooling!r} is a TEST_ONLY placeholder and "
                    "cannot define a SCIENTIFIC run. §5.2 pins the head's pooling during "
                    "spec lock; the Stage-1 §4.6 masked-mean rule is a different decision "
                    "and is not inherited."
                )
            missing = [v for v in SCIENTIFIC_REQUIRED_VALUES if v not in self.resolved_values]
            if missing:
                raise UnresolvedEvaluationValue(
                    "a SCIENTIFIC downstream configuration requires these OPEN values to be "
                    f"resolved first: {missing}. §5's open-items table names the "
                    "classification-head concrete values as blocking G1."
                )

    @property
    def is_diagnostic_only(self) -> bool:
        return self.purpose is EvaluationPurpose.DIAGNOSTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "diagnostic_only": self.is_diagnostic_only,
            "values_are_scientific": not self.is_diagnostic_only,
            "task_id": self.task_id,
            "pathways": [p.value for p in self.pathways],
            "resolved_values": sorted(self.resolved_values),
            "note": self.note,
            "head": self.head.to_dict(),
        }
