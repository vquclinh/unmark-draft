"""The Stage-1 held-out UNLABELED evaluator. **Imports torch lazily.**

Every pilot candidate and every checkpoint is scored here and nowhere else. It
reads **no labels and no downstream task** (D-S1B-001): the only signal is the
cosine distance between adapted and reference pooled representations on the
Stage-1 dev split.

Three properties make candidates comparable, and all three are enforced:

* the **same held-out examples** for every candidate;
* the **same fixed condition grid** ``FULL, P50, P100, STRIP_ALL`` -- never a
  random `p`, which would compare candidates on different corruptions;
* the **same deterministic corruption realization**, keyed on the dedicated
  ``UNMARK-STAGE1-v1|validation-corruption`` seed so that **changing the
  training seed cannot change a validation corruption**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from unmark.corruption.conditions import get_condition
from unmark.stage1.contracts import Stage1ContractViolation, TruncationPolicy
from unmark.stage1.data import Stage1Example, prepare_with_condition
from unmark.stage1.protocol import (
    METRIC_UNIT,
    VALIDATION_CONDITIONS,
    VALIDATION_CORRUPTION_SEED,
    VALIDATION_CORRUPTION_SEED_TAG,
)
from unmark.stage1.selection import ValidationPoint

VALIDATION_VISIT = 0
"""Validation corruption is drawn at a fixed visit. It must not drift with the
training pass, or the metric would move for reasons unrelated to the model."""


class ValidationContractViolation(Stage1ContractViolation):
    """Raised when the evaluator is asked to depart from the locked grid."""


def condition_for(name: str):
    """The locked evaluation condition, from the audited B2 condition set.

    `FULL`, `P50`, `P100` and `STRIP_ALL` already exist there with exactly the
    proposal §6.3 semantics; re-deriving them here would be a second definition
    of the evaluation grid.
    """
    if name not in VALIDATION_CONDITIONS:
        raise ValidationContractViolation(
            f"{name!r} is not in the locked validation grid {list(VALIDATION_CONDITIONS)}"
        )
    return get_condition(name)


@dataclass(frozen=True)
class HeldOutExample:
    """One dev chunk. `chunk_id` is the corruption key, so it must be stable."""

    chunk_id: str
    text: str


def prepare_condition_batch(
    examples: Sequence[HeldOutExample],
    tokenizer: Any,
    condition_name: str,
    *,
    truncation: TruncationPolicy,
    classifier: Callable[[str], Any] | None = None,
    unk_token_id: int | None = None,
):
    """Prepare the held-out set under one fixed condition.

    Uses the **same** `prepare_with_condition` the training path uses, with the
    dedicated validation corruption seed.
    """
    condition = condition_for(condition_name)
    prepared = []
    for example in examples:
        item = prepare_with_condition(
            Stage1Example(text=example.text, sample_id=example.chunk_id),
            tokenizer,
            condition=condition,
            corruption_seed=VALIDATION_CORRUPTION_SEED,
            truncation=truncation,
            visit=VALIDATION_VISIT,
            classifier=classifier,
            unk_token_id=unk_token_id,
            extra_metadata={
                "validation_condition": condition_name,
                "validation_seed_tag": VALIDATION_CORRUPTION_SEED_TAG,
            },
        )
        if item is None:
            raise ValidationContractViolation(
                f"held-out chunk {example.chunk_id!r} overflowed under condition "
                f"{condition_name}; after correct pre-chunking this cannot happen"
            )
        prepared.append(item)
    return prepared


def evaluate(
    objective: Any,
    prepared_by_condition: dict[str, Sequence[Any]],
    pad_token_id: int,
    *,
    batch_size: int,
) -> ValidationPoint:
    """Compute `d_c` for every locked condition plus `d_clean`. **Lazy torch.**

    The metric unit is the prepared **chunk**, unweighted (`METRIC_UNIT`), which
    is recorded in the protocol rather than left to an implementation accident:
    document-weighted aggregation would silently re-weight long articles.
    """
    import torch

    from unmark.stage1.data import batch_to_device, collate_stage1_batch, module_device
    from unmark.stage1.objective import representation_distance

    missing = [c for c in VALIDATION_CONDITIONS if c not in prepared_by_condition]
    if missing:
        raise ValidationContractViolation(f"missing condition(s) {missing}")

    distances: dict[str, float] = {}
    clean_total: list[float] = []
    # The batch follows the model. `collate_stage1_batch` builds CPU tensors and
    # the objective never moves its inputs, so an objective on an accelerator
    # would otherwise be handed CPU ids (Audit 030 §Y). Derived from the
    # objective's own parameters; a no-op on CPU.
    device = module_device(objective)
    objective.eval()
    with torch.no_grad():
        for condition in VALIDATION_CONDITIONS:
            prepared = list(prepared_by_condition[condition])
            per_example: list[float] = []
            for start in range(0, len(prepared), batch_size):
                batch = batch_to_device(
                    collate_stage1_batch(prepared[start : start + batch_size], pad_token_id),
                    device,
                )
                reference = objective.reference_representation(
                    batch["reference_input_ids"],
                    batch["reference_attention_mask"],
                    batch["reference_special_tokens_mask"],
                )
                corrupted = objective.adapted_representation(
                    batch["base_input_ids"], batch["base_attention_mask"],
                    batch["base_special_tokens_mask"],
                    batch["corrupt_tone_ids"], batch["corrupt_tone_mask"],
                    batch["corrupt_letter_ids"], batch["corrupt_letter_mask"],
                )
                per_example.extend(representation_distance(corrupted, reference).tolist())
                if condition == VALIDATION_CONDITIONS[0]:
                    # d_clean uses the ADAPTED CLEAN channels, computed once
                    adapted_clean = objective.adapted_representation(
                        batch["base_input_ids"], batch["base_attention_mask"],
                        batch["base_special_tokens_mask"],
                        batch["clean_tone_ids"], batch["clean_tone_mask"],
                        batch["clean_letter_ids"], batch["clean_letter_mask"],
                    )
                    clean_total.extend(
                        representation_distance(adapted_clean, reference).tolist()
                    )
            if not per_example:
                raise ValidationContractViolation(f"condition {condition} produced no examples")
            distances[condition] = sum(per_example) / len(per_example)

    return ValidationPoint(
        update=0,  # the caller stamps the real update
        distances=distances,
        d_clean=sum(clean_total) / len(clean_total),
    )


def at_update(point: ValidationPoint, update: int) -> ValidationPoint:
    """Restamp a computed point with the update it belongs to."""
    return ValidationPoint(update=update, distances=point.distances, d_clean=point.d_clean)


VALIDATION_CONTRACT = {
    "conditions": list(VALIDATION_CONDITIONS),
    "corruption_seed": VALIDATION_CORRUPTION_SEED,
    "corruption_seed_tag": VALIDATION_CORRUPTION_SEED_TAG,
    "visit": VALIDATION_VISIT,
    "metric_unit": METRIC_UNIT,
    "labels_used": False,
    "downstream_task_used": False,
    "training_seed_affects_validation_corruption": False,
}
