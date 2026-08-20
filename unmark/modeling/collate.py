"""Channel metadata -> adapter tensors (B4B).

**Torch is imported lazily, inside `collate_examples` only.** Everything up to
the final tensor packing -- label-to-id mapping, the `NA` sentinel decision, and
laying content projections out against the model's own special tokens -- is pure
Python, so it is testable in the ML-free local environment. Only the last step,
padding to `[B, L]` and `[B, L, K]` tensors, needs torch.

Not exported from `unmark.modeling.__init__`. Import explicitly::

    from unmark.modeling.collate import build_example, collate_examples

This is the seam where the deterministic pipeline meets the neural one:

    text  --B1A/B2/B3-->  metadata  --this module-->  tensors  --A_phi-->  z

It performs no tokenization, no orthographic decomposition and no corruption; it
consumes `TokenOrthographyProjection` records that the B3B channel projection
already produced, and lays them out on the **authoritative** encoder sequence
including the model's own special tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from unmark.alignment.channels import TokenOrthographyProjection, TokenToneLabel
from unmark.modeling.contracts import (
    LETTER_LABEL_IDS,
    LETTER_NA_SENTINEL,
    OBSERVABLE_TONE_IDS,
    TONE_NA_SENTINEL,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from torch import Tensor


def tone_id_and_mask(label: TokenToneLabel) -> tuple[int, bool]:
    """`NA` becomes the out-of-table sentinel with a false mask (D-B4A-002)."""
    if label is TokenToneLabel.NA:
        return TONE_NA_SENTINEL, False
    return OBSERVABLE_TONE_IDS[label.value], True


def letter_contributor_ids(projection: TokenOrthographyProjection) -> list[int]:
    """Applicable letter contributors, in source order.

    `NONE` participates; `NA` contributors are already excluded by
    `LetterProjection.applicable`. A token with none yields an empty list, which
    the adapter turns into the exact zero vector (D-B4A-005).
    """
    return [LETTER_LABEL_IDS[label.value] for label in projection.letter.applicable_labels]


@dataclass
class EncodedExample:
    """One example laid out on the authoritative encoder sequence.

    `input_ids` includes the model's special tokens; the channel rows for those
    positions carry `NA` in both channels and no source range.
    """

    input_ids: list[int]
    special_tokens_mask: list[int]
    tone_ids: list[int]
    tone_mask: list[bool]
    letter_ids: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.special_tokens_mask),
            len(self.tone_ids),
            len(self.tone_mask),
            len(self.letter_ids),
        }
        if len(lengths) != 1:
            raise ValueError(f"example fields have inconsistent lengths: {sorted(lengths)}")

    @property
    def length(self) -> int:
        return len(self.input_ids)


def build_example(
    input_ids: Sequence[int],
    special_tokens_mask: Sequence[int],
    projections: Sequence[TokenOrthographyProjection],
) -> EncodedExample:
    """Interleave content projections with the model's special tokens.

    `input_ids` and `special_tokens_mask` come from the tokenizer, so the special
    token identity and order are the model's, never guessed here. Content
    projections fill the non-special slots in order; special positions get `NA`
    in both channels.

    Raises:
        ValueError: if the number of non-special positions does not equal the
            number of projections. That mismatch means the alignment and the
            encoder sequence disagree, which must never be papered over.
    """
    if len(input_ids) != len(special_tokens_mask):
        raise ValueError(
            f"input_ids ({len(input_ids)}) and special_tokens_mask "
            f"({len(special_tokens_mask)}) differ in length"
        )
    content_slots = [i for i, flag in enumerate(special_tokens_mask) if not flag]
    if len(content_slots) != len(projections):
        raise ValueError(
            f"{len(content_slots)} non-special positions but {len(projections)} "
            "projections: the alignment and the encoder sequence disagree"
        )

    tone_ids = [TONE_NA_SENTINEL] * len(input_ids)
    tone_mask = [False] * len(input_ids)
    letter_ids: list[list[int]] = [[] for _ in input_ids]

    for slot, projection in zip(content_slots, projections):
        identifier, live = tone_id_and_mask(projection.tone.label)
        tone_ids[slot] = identifier
        tone_mask[slot] = live
        letter_ids[slot] = letter_contributor_ids(projection)

    return EncodedExample(
        input_ids=list(input_ids),
        special_tokens_mask=list(special_tokens_mask),
        tone_ids=tone_ids,
        tone_mask=tone_mask,
        letter_ids=letter_ids,
    )


def padded_batch(
    examples: Sequence[EncodedExample], pad_token_id: int
) -> dict[str, list[Any]]:
    """Right-pad a batch to `L_max` and `K_max`, as nested Python lists.

    Torch-free, so the padding semantics are testable in the ML-free
    environment; `collate_examples` only wraps the result in tensors.

    `K` is the **batch's** maximum contributor count, not a global constant: no
    scientific maximum is invented. Padding positions get `attention_mask = 0`,
    `special_tokens_mask = 1`, tone `NA` and no letter contributors -- so their
    channels are exactly zero and Stage-1 pooling excludes them twice over.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(example.length for example in examples)
    depth = max(
        (len(slot) for example in examples for slot in example.letter_ids),
        default=0,
    )
    depth = max(depth, 1)  # keep a K axis even when nothing is applicable

    input_ids, attention, special, tone_ids, tone_mask = [], [], [], [], []
    letter_ids, letter_mask = [], []

    for example in examples:
        pad = width - example.length
        input_ids.append(example.input_ids + [pad_token_id] * pad)
        attention.append([1] * example.length + [0] * pad)
        special.append(example.special_tokens_mask + [1] * pad)
        tone_ids.append(example.tone_ids + [TONE_NA_SENTINEL] * pad)
        tone_mask.append(example.tone_mask + [False] * pad)

        rows, row_masks = [], []
        for slot in example.letter_ids + [[] for _ in range(pad)]:
            filled = list(slot) + [LETTER_NA_SENTINEL] * (depth - len(slot))
            rows.append(filled)
            row_masks.append([True] * len(slot) + [False] * (depth - len(slot)))
        letter_ids.append(rows)
        letter_mask.append(row_masks)

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "special_tokens_mask": special,
        "tone_ids": tone_ids,
        "tone_mask": tone_mask,
        "letter_ids": letter_ids,
        "letter_mask": letter_mask,
    }


_BOOL_FIELDS = frozenset({"tone_mask", "letter_mask"})


def collate_examples(
    examples: Sequence[EncodedExample], pad_token_id: int
) -> dict[str, "Tensor"]:
    """`padded_batch`, wrapped in tensors. **Imports torch lazily.**"""
    import torch

    rows = padded_batch(examples, pad_token_id)
    return {
        key: torch.tensor(value, dtype=torch.bool if key in _BOOL_FIELDS else torch.long)
        for key, value in rows.items()
    }
