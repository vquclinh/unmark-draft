"""Readouts: sequence hidden states to one feature vector. **Torch is imported lazily.**

    FIRST_TOKEN  hidden[:, 0, :]
    MASKED_MEAN  mean over positions that are neither padding nor special tokens
    CONCAT       concat([FIRST_TOKEN, MASKED_MEAN], dim=-1)        # exactly [FT ; MM]

`special_tokens_mask` must come from the tokenizer that produced the ids. That
mask, not a guess about which positions are special, decides what MASKED_MEAN
excludes.

CONCAT is built from ONE hidden-state tensor: `first_token_and_masked_mean`
reads both halves from the same `hidden`, so a CONCAT feature cannot mix two
different encoder forwards.
"""

from __future__ import annotations

from typing import Any

from unmark.viunmark.config import ReadoutKind, ViUnMarkContractError


def _require_sequence_hidden(hidden: Any) -> None:
    if getattr(hidden, "dim", None) is None or hidden.dim() != 3:
        raise ViUnMarkContractError(
            f"hidden states must be [batch, length, hidden], got "
            f"{tuple(getattr(hidden, 'shape', ()))}"
        )


def first_token(hidden: Any) -> Any:
    """`hidden[:, 0, :]`."""
    _require_sequence_hidden(hidden)
    return hidden[:, 0, :]


def masked_mean(hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> Any:
    """Mean over content positions: attended AND not a tokenizer special token.

    Delegates to the repository's accepted pooling, which fails loudly for a row
    with no content position instead of returning a vector that represents nothing.
    """
    from unmark.modeling.pooling import masked_mean_non_special

    _require_sequence_hidden(hidden)
    return masked_mean_non_special(hidden, attention_mask, special_tokens_mask)


def first_token_and_masked_mean(
    hidden: Any, attention_mask: Any, special_tokens_mask: Any
) -> tuple[Any, Any]:
    """Both readouts from the same `hidden` tensor."""
    return first_token(hidden), masked_mean(hidden, attention_mask, special_tokens_mask)


def concat_first_token_masked_mean(
    hidden: Any, attention_mask: Any, special_tokens_mask: Any
) -> Any:
    """`concat([FIRST_TOKEN, MASKED_MEAN], dim=-1)` from one hidden-state tensor."""
    import torch

    first, mean = first_token_and_masked_mean(hidden, attention_mask, special_tokens_mask)
    return torch.cat([first, mean], dim=-1)


def readout(
    kind: ReadoutKind, hidden: Any, attention_mask: Any, special_tokens_mask: Any
) -> Any:
    """Apply one readout. Masks are required for every kind, so call sites stay uniform."""
    if not isinstance(kind, ReadoutKind):
        raise ViUnMarkContractError(f"kind must be a ReadoutKind, got {kind!r}")
    if kind is ReadoutKind.FIRST_TOKEN:
        return first_token(hidden)
    if kind is ReadoutKind.MASKED_MEAN:
        return masked_mean(hidden, attention_mask, special_tokens_mask)
    return concat_first_token_masked_mean(hidden, attention_mask, special_tokens_mask)


__all__ = [
    "concat_first_token_masked_mean",
    "first_token",
    "first_token_and_masked_mean",
    "masked_mean",
    "readout",
]
