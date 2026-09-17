"""Checkpoint-free pooling bridge."""

from __future__ import annotations

from typing import Any

from viunmark.readouts import first_token_and_masked_mean

HEAD_TRAINED = False
DECODERS_IMPLEMENTED = False


def same_forward_pooling_pair(hidden: Any, attention_mask: Any, special_tokens_mask: Any) -> tuple[Any, Any]:
    return first_token_and_masked_mean(hidden, attention_mask, special_tokens_mask)
