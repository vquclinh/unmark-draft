"""Vanilla and Base-only encoder pathways. **This module imports torch.**

Not re-exported from `unmark.evaluation.__init__` -- the local environment is
ML-free. Import explicitly::

    from unmark.evaluation.pathways import encode_split, encoder_hidden_states

Two pathways, and the distinction is the whole experiment:

* **VANILLA** -- `canon(x)` through the frozen tokenizer and frozen encoder.
  §6.4's `UPPER` at `FULL` and `FLOOR` at a corrupted condition.
* **BASE_ONLY** -- `b(x)` through the *same* frozen tokenizer and encoder, with
  **no adapter and no orthography channels**.

**Neither pathway restores or guesses missing marks**, and `BASE_ONLY` is **not
UNMARK** -- §4.5 calls it the base-only pathway precisely to keep them apart.

**Nothing here trains.** There is no optimizer, no head training loop and no
parameter update; representations are extracted under `no_grad` from a frozen
encoder.

**Extraction does not pool.** Stage-2 head pooling is OPEN (§5.2), so
`encoder_hidden_states` returns `[N, L, d]` plus masks and leaves pooling to the
future head. The Stage-1 §4.6 masked-mean rule is a different decision about a
different objective and is not inherited here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from unmark.evaluation.contracts import (
    EvaluationContractViolation,
    EvaluationPurpose,
    HeadConfig,
    Split,
    SplitLeakage,
    SystemPathway,
    TaskSplit,
    UnresolvedEvaluationValue,
)
from unmark.orthography import canon, decompose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

BASE_ONLY_EQUIVALENCE_EVIDENCE = (
    "B4B run 20260820T081554Z: model(input_ids=...) vs "
    "model(inputs_embeds=Emb(input_ids), position_ids=authoritative, attention_mask=...) "
    "gave max_abs_diff = 0.0 exactly, including padding; and the forced g:=0 wiring "
    "identity gives z = g*f + (1-g)*e = e. Therefore running the frozen encoder directly "
    "on T(b(x)) is numerically identical to the adapter pathway at g = 0."
)
"""Why `BASE_ONLY` is implemented without the adapter.

**Caveat that must travel with this claim:** `g = 0` is *not attainable* by the
locked sigmoid gate (D-B4A-004) -- it is a limit. `BASE_ONLY` therefore
implements the **architectural limit** §4.5 describes, not the behaviour of any
initialised or trained adapter.
"""


def pathway_text(text: str, pathway: SystemPathway) -> str:
    """The string a pathway feeds to the frozen tokenizer.

    Both pathways canonicalise first, for the reason D-B2-004 and D-S1A-001 give:
    corruption is defined on `canon(x)`, so two inputs differing only in NFC/NFD
    form or tone placement are *the same example*. Letting the raw form through
    here would make the score depend on incoming spelling, which is the separate
    `VARIANT` condition (§6.3).

    `BASE_ONLY` then strips to `b(x)`. **No mark is restored or guessed.**
    """
    canonical = canon(text)
    if pathway is SystemPathway.VANILLA:
        return canonical
    if pathway is SystemPathway.BASE_ONLY:
        return decompose(canonical).base_text
    raise EvaluationContractViolation(f"unsupported pathway {pathway!r}")


@dataclass(frozen=True)
class EncodedSplit:
    """Tokenized inputs for one split under one pathway.

    Carries the task, split and pathway identity so a downstream consumer cannot
    silently mix representations from different pathways or splits.
    """

    task_id: str
    split: Split
    pathway: SystemPathway
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    input_ids: "Tensor"
    attention_mask: "Tensor"
    special_tokens_mask: "Tensor"

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.task_id, self.split.value, self.pathway.value)


def encode_split(
    task_split: TaskSplit,
    pathway: SystemPathway,
    tokenizer: Any,
    head_config: HeadConfig,
    padding: bool | str = True,
) -> EncodedSplit:
    """Tokenize one split under one pathway. **Imports torch lazily.**

    `head_config` is required rather than optional because `max_length` is one of
    the §5.2 values pinned during spec lock -- there is no default to fall back
    on, and truncation length is a scientific choice.

    Args:
        padding: passed straight through to the tokenizer. The default `True`
            (pad to the longest sequence in the batch) is the general Stage-2
            behaviour. The pre-G1 diagnostic pins `"max_length"` instead, because
            its representation cache is keyed on a fixed shape -- see
            `preg1_head.PREG1_TOKENIZATION`. It is a parameter rather than a
            constant so neither caller inherits the other's choice by accident.
    """
    import torch

    texts = [pathway_text(example.text, pathway) for example in task_split.examples]
    encoded = tokenizer(
        texts,
        padding=padding,
        truncation=True,
        max_length=head_config.max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    return EncodedSplit(
        task_id=task_split.task_id,
        split=task_split.split,
        pathway=pathway,
        sample_ids=task_split.sample_ids,
        labels=task_split.labels,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        special_tokens_mask=encoded["special_tokens_mask"],
    )


@dataclass(frozen=True)
class HiddenStateSet:
    """Final encoder hidden states `[N, L, d]`, **unpooled**, bound to their origin.

    **Deliberately not pooled.** §5.2 lists the classification head's `pooling`
    among the concrete values "pinned during spec lock", and §13 item 4 repeats
    it -- so Stage-2 pooling is **OPEN**. The masked-mean rule locked in §4.6 is
    the **Stage-1 alignment** pooling and does not transfer here just because it
    already exists.

    The masks travel with the states so that whichever pooling the researcher
    eventually pins can be applied correctly, and the `(task, split, pathway)`
    identity travels with them so a head cannot be fed another pathway's
    representations.
    """

    task_id: str
    split: Split
    pathway: SystemPathway
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    hidden_states: "Tensor"
    attention_mask: "Tensor"
    special_tokens_mask: "Tensor"

    def __post_init__(self) -> None:
        if self.hidden_states.dim() != 3:
            raise EvaluationContractViolation(
                f"hidden_states must be [N, L, d], got {tuple(self.hidden_states.shape)}. "
                "Stage-2 pooling is OPEN, so extraction does not pool."
            )
        if self.hidden_states.shape[0] != len(self.sample_ids):
            raise EvaluationContractViolation(
                f"{self.hidden_states.shape[0]} rows for {len(self.sample_ids)} sample ids"
            )
        if len(self.labels) != len(self.sample_ids):
            raise EvaluationContractViolation("labels and sample ids differ in length")

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.task_id, self.split.value, self.pathway.value)

    def require_trainable(self) -> None:
        """§5.4: a head may be trained on `train` only."""
        if not self.split.may_train_head:
            raise SplitLeakage(
                f"cannot train a head on {self.split.value} representations: §5.4 permits "
                "head training on train only"
            )

    def require_same_pathway(self, other: "HiddenStateSet | HeadBinding") -> None:
        """Refuse to mix pathways or tasks.

        §4.5 and §6.6: `UPPER` and the adapted pathways are *different input
        pathways*, so a head fitted to one pathway's representation geometry has
        no defined meaning applied to another's.
        """
        if self.pathway is not other.pathway:
            raise SplitLeakage(
                f"pathway mismatch: representations are {self.pathway.value} but the head "
                f"is {other.pathway.value}. §8.3 runs the head protocol identically for "
                "each system, on that system's own clean pathway -- it does not share one "
                "head across pathways."
            )
        if self.task_id != other.task_id:
            raise SplitLeakage(f"task mismatch: {self.task_id} vs {other.task_id}")


@dataclass(frozen=True)
class HeadBinding:
    """What a trained head is bound to.

    A record, not a model: this task implements the plumbing and the guards, not
    the head trainer, because the architecture, pooling and optimizer are OPEN
    (§5.2, and §5's open-items table names them as blocking G1).
    """

    task_id: str
    pathway: SystemPathway
    trained_on_split: Split
    config: HeadConfig

    def __post_init__(self) -> None:
        if not self.trained_on_split.may_train_head:
            raise SplitLeakage(
                f"a head cannot be bound to {self.trained_on_split.value}: §5.4 permits "
                "head training on train only"
            )

    def require_clean_training(self, condition: str) -> None:
        """§5.2 / §8.3: the head is trained on **clean data only**."""
        if condition != "FULL":
            raise SplitLeakage(
                f"a head may not be trained on condition {condition!r}: §5.2 locks "
                "clean-only head training, then freezes the head and evaluates it under "
                "every condition. Training on corrupted data would let robustness come "
                "from supervised noise augmentation in the head."
            )


def encoder_hidden_states(encoded: EncodedSplit, encoder: Any) -> HiddenStateSet:
    """Final hidden states from the **frozen** encoder. `[N, L, d]`, **unpooled**.

    This is the scientific extraction path. It returns hidden states and masks,
    **not** a pooled vector, because Stage-2 pooling is OPEN (§5.2). Pooling here
    would silently promote the Stage-1 §4.6 rule into a Stage-2 decision nobody
    made.

    Runs under `no_grad` with the encoder in `eval`: this is feature extraction
    for a downstream head, and §5.1 keeps the encoder fully frozen.
    """
    import torch

    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            outputs = encoder(
                input_ids=encoded.input_ids, attention_mask=encoded.attention_mask
            )
            hidden = getattr(outputs, "last_hidden_state", outputs)
    finally:
        if was_training:  # pragma: no cover - the encoder should already be frozen
            encoder.train()
    return HiddenStateSet(
        task_id=encoded.task_id,
        split=encoded.split,
        pathway=encoded.pathway,
        sample_ids=encoded.sample_ids,
        labels=encoded.labels,
        hidden_states=hidden,
        attention_mask=encoded.attention_mask,
        special_tokens_mask=encoded.special_tokens_mask,
    )


def TEST_ONLY_masked_mean_pool(
    hidden_set: HiddenStateSet, purpose: EvaluationPurpose
) -> "Tensor":
    """Masked-mean pooling **for diagnostics and tests only**. Returns `[N, d]`.

    **This is not a Stage-2 scientific pooling rule and must never become one.**
    §4.6 locks masked mean for the **Stage-1 alignment objective**; §5.2 lists the
    classification head's pooling among values pinned during spec lock, and it is
    still OPEN. The two are different decisions about different things, and the
    only reason this function exists is that a smoke test needs *some* way to turn
    `[N, L, d]` into `[N, d]`.

    Raises:
        UnresolvedEvaluationValue: if called with `SCIENTIFIC` purpose. A
            scientific run must resolve `head_pooling` and apply its own rule.
    """
    if purpose is not EvaluationPurpose.DIAGNOSTIC:
        raise UnresolvedEvaluationValue(
            "TEST_ONLY_masked_mean_pool is diagnostic-only. Stage-2 head pooling is "
            "OPEN (§5.2 pins it during spec lock); a SCIENTIFIC run must resolve "
            "head_pooling and apply the resolved rule, not inherit the Stage-1 §4.6 "
            "alignment pooling."
        )
    from unmark.modeling.pooling import masked_mean_non_special

    return masked_mean_non_special(
        hidden_set.hidden_states, hidden_set.attention_mask, hidden_set.special_tokens_mask
    )
