"""Stage-1 pooling (D-B4A-006). **This module imports torch.**

Not exported from `unmark.modeling.__init__` -- the local development
environment is ML-free. Import explicitly::

    from unmark.modeling.pooling import masked_mean_non_special

**This is the pooling utility only. There is no Stage-1 loss and no training
loop here**; §4.6's `L_align` / `L_clean` belong to a later phase.
"""

from __future__ import annotations

import torch
from torch import Tensor

from unmark.modeling.contracts import Stage1PoolingError


def content_mask(attention_mask: Tensor, special_tokens_mask: Tensor) -> Tensor:
    """`m_i = attention_mask_i AND NOT special_tokens_mask_i`.

    The two masks do different jobs and neither substitutes for the other:
    `attention_mask` excludes **padding**, `special_tokens_mask` excludes
    **model special tokens**. Padding is never counted as content.
    """
    if attention_mask.shape != special_tokens_mask.shape:
        raise ValueError(
            f"mask shapes differ: {tuple(attention_mask.shape)} vs "
            f"{tuple(special_tokens_mask.shape)}"
        )
    keep = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
    special = (
        special_tokens_mask.bool()
        if special_tokens_mask.dtype != torch.bool
        else special_tokens_mask
    )
    return keep & ~special


def masked_mean_non_special(
    hidden_states: Tensor,
    attention_mask: Tensor,
    special_tokens_mask: Tensor,
) -> Tensor:
    """Attention-masked mean over non-special content tokens. Returns `[B, d]`.

    ::

        m_i = attention_mask_i * (1 - special_tokens_mask_i)
        h   = sum_i m_i H_i / sum_i m_i

    Works for arbitrary `L`, and is computed independently per example -- the
    clean reference branch and the adapted branch do **not** share a sequence
    length, and no per-token correspondence is assumed (§4.6).

    Raises:
        Stage1PoolingError: if **any** example has zero content positions. Fails
            loud rather than falling back to `<s>`, to an unmasked mean, or to a
            zero vector, each of which would hand the cosine objective a value
            that represents nothing.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, L, d], got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states [B, L] = {tuple(hidden_states.shape[:2])}"
        )

    mask = content_mask(attention_mask, special_tokens_mask)
    counts = mask.sum(dim=1)  # [B]
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty:
        raise Stage1PoolingError(
            f"examples {empty} have no content positions after masking: every position "
            "is padding or a special token. Stage-1 pooling fails loud rather than "
            "falling back to <s>, an unmasked mean, or a zero vector (D-B4A-006)."
        )

    weights = mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden_states.dtype)
