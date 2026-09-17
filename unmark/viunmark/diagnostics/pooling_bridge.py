"""Checkpoint-Free Pooling Bridge. **Torch is imported lazily.**

Compares FIRST_TOKEN with MASKED_MEAN without training a classification head.
Both features must come from the SAME encoder forward, so a difference between
them is a difference of pooling, not of two forwards.

Implemented here: extracting the paired features from one hidden-state tensor.

Not implemented, deliberately: the checkpoint-free decoders that score the pair
(class-centroid and 1-nearest-neighbour decoding). Their distance metric,
normalisation and tie-breaking are not recorded, and inventing them would
produce numbers that look comparable to the recorded analysis without being so.
"""

from __future__ import annotations

from typing import Any

from unmark.viunmark.readout import first_token_and_masked_mean

HEAD_TRAINED = False
DECODERS_IMPLEMENTED = False


def same_forward_pooling_pair(
    hidden: Any, attention_mask: Any, special_tokens_mask: Any
) -> tuple[Any, Any]:
    """`(FIRST_TOKEN, MASKED_MEAN)` from one hidden-state tensor."""
    return first_token_and_masked_mean(hidden, attention_mask, special_tokens_mask)


__all__ = ["DECODERS_IMPLEMENTED", "HEAD_TRAINED", "same_forward_pooling_pair"]
