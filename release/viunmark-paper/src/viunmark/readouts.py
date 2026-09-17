"""Readouts from sequence hidden states to feature vectors."""

from __future__ import annotations

from typing import Any

from viunmark.config import ReadoutKind, ViUnMarkContractError


def _require_sequence_hidden(hidden: Any) -> None:
    if getattr(hidden, "dim", None) is None or hidden.dim() != 3:
        raise ViUnMarkContractError("hidden states must be [batch, length, hidden]")


def content_mask(attention_mask: Any, special_tokens_mask: Any) -> Any:
    if attention_mask.shape != special_tokens_mask.shape:
        raise ViUnMarkContractError("attention_mask and special_tokens_mask shapes differ")
    keep = attention_mask.bool() if attention_mask.dtype != getattr(attention_mask, "bool", None) else attention_mask
    special = special_tokens_mask.bool() if special_tokens_mask.dtype != getattr(special_tokens_mask, "bool", None) else special_tokens_mask
    return keep & ~special


def first_token(hidden: Any) -> Any:
    _require_sequence_hidden(hidden)
    return hidden[:, 0, :]


def masked_mean(hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> Any:
    _require_sequence_hidden(hidden)
    if attention_mask.shape != hidden.shape[:2]:
        raise ViUnMarkContractError("attention_mask does not match hidden states")
    mask = content_mask(attention_mask, special_tokens_mask)
    counts = mask.sum(dim=1)
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty:
        raise ViUnMarkContractError(f"examples {empty} have no content positions")
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)


def first_token_and_masked_mean(hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> tuple[Any, Any]:
    return first_token(hidden), masked_mean(hidden, attention_mask, special_tokens_mask)


def concat_first_token_masked_mean(hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> Any:
    import torch

    first, mean = first_token_and_masked_mean(hidden, attention_mask, special_tokens_mask)
    return torch.cat([first, mean], dim=-1)


def readout(kind: ReadoutKind, hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> Any:
    if not isinstance(kind, ReadoutKind):
        raise ViUnMarkContractError("kind must be a ReadoutKind")
    if kind is ReadoutKind.FIRST_TOKEN:
        return first_token(hidden)
    if kind is ReadoutKind.MASKED_MEAN:
        return masked_mean(hidden, attention_mask, special_tokens_mask)
    return concat_first_token_masked_mean(hidden, attention_mask, special_tokens_mask)
