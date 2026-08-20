"""The Stage-1 representation-alignment objective. **This module imports torch.**

Not re-exported from `unmark.stage1.__init__` -- the local environment is
ML-free. Import explicitly::

    from unmark.stage1.objective import Stage1Objective

Proposal §4.6, in the proposal's own notation::

    L_align = D( h'(x_p), h(x) )
    L_clean = D( h'(x),   h(x) )
    L       = lambda_a * L_align + lambda_c * L_clean

`D` is cosine distance, at the **pooled** representation level. `h(x)` comes
from the clean text through the bare frozen encoder; `h'(.)` from the base grid
through `A_phi` and the same frozen encoder.

**There is no optimizer, no scheduler, no training loop and no checkpointing
here.** This module computes a loss; nothing calls `.backward()` except tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from unmark.modeling.pooling import masked_mean_non_special
from unmark.stage1.contracts import ObjectiveWeights, Stage1ContractViolation

COSINE_EPS = 1e-8
"""Denominator floor for the cosine similarity.

A numerical guard against a zero-norm pooled vector, **not a tuned
hyperparameter**. `torch.nn.functional.cosine_similarity` clamps the norm
product at its own `eps` for the same reason; the value is fixed and is not an
experiment knob.
"""


@dataclass
class Stage1LossResult:
    """Structured loss output. Not one opaque scalar.

    Per-example distances are retained because they are small (`[B]`) and are the
    diagnostic that shows whether the two terms are behaving differently. Pooled
    representations are **not** retained by default -- they are `[B, d]` per
    branch and would pin three of them alive for every batch.

    No raw text is stored here: the loss object travels into logs.
    """

    loss: Tensor
    loss_align: Tensor
    loss_clean: Tensor
    distance_align_per_example: Tensor
    distance_clean_per_example: Tensor
    weights: ObjectiveWeights

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach()),
            "loss_align": float(self.loss_align.detach()),
            "loss_clean": float(self.loss_clean.detach()),
            "mean_distance_align": float(self.distance_align_per_example.detach().mean()),
            "mean_distance_clean": float(self.distance_clean_per_example.detach().mean()),
            "batch_size": int(self.distance_align_per_example.shape[0]),
            **self.weights.to_dict(),
        }


def representation_distance(a: Tensor, b: Tensor) -> Tensor:
    """Cosine distance per example: `1 - cos(a, b)`. `[B, d] x [B, d] -> [B]`.

    Computed over the **feature** dimension, never over the batch, and never at
    token level -- the branches do not share a token grid, which is why §4.6
    aligns pooled representations.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise Stage1ContractViolation(
            f"representation_distance expects [B, d] tensors, got {tuple(a.shape)} "
            f"and {tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise Stage1ContractViolation(
            f"pooled representations must match: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=COSINE_EPS)
    return 1.0 - similarity


def _require_finite(value: Tensor, what: str) -> Tensor:
    if not torch.isfinite(value).all():
        raise Stage1ContractViolation(
            f"{what} is not finite. Stage-1 fails loud rather than propagating a "
            "NaN/Inf that would silently poison an optimizer step."
        )
    return value


class Stage1Objective(nn.Module):
    """The three-branch objective.

    Args:
        unmark_encoder: the B4B `UnmarkEncoder` -- frozen encoder plus adapter.
            **The same frozen theta serves all three branches**; no second
            pretrained model is loaded for the reference.
        weights: `lambda_a` and `lambda_c`. Required; the proposal tunes them on
            a development split and no default may reach an experiment.
    """

    def __init__(self, unmark_encoder: nn.Module, weights: ObjectiveWeights) -> None:
        super().__init__()
        if not isinstance(weights, ObjectiveWeights):
            raise Stage1ContractViolation(
                "weights must be an ObjectiveWeights; lambda_align and lambda_clean "
                "are required and have no defaults"
            )
        self.unmark_encoder = unmark_encoder
        self.weights = weights

    # -- branches ---------------------------------------------------------
    def reference_representation(
        self, input_ids: Tensor, attention_mask: Tensor, special_tokens_mask: Tensor
    ) -> Tensor:
        """`h(x)`: bare frozen encoder on the clean text. `[B, d]`.

        Under `no_grad` because it is a **target**: theta is frozen, and no
        gradient may flow into it. This is the one branch where `no_grad` is
        correct -- the adapted branches must never be wrapped in it.
        """
        encoder = self.unmark_encoder.encoder
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = _hidden_states(outputs)
            return masked_mean_non_special(hidden, attention_mask, special_tokens_mask)

    def adapted_representation(
        self,
        base_input_ids: Tensor,
        base_attention_mask: Tensor,
        base_special_tokens_mask: Tensor,
        tone_ids: Tensor,
        tone_mask: Tensor,
        letter_ids: Tensor,
        letter_mask: Tensor,
    ) -> Tensor:
        """`h'(.)`: adapter + frozen encoder on the base grid. `[B, d]`.

        **Not** under `no_grad`, and nothing is detached: the graph must run from
        the loss through the frozen encoder into `A_phi`.

        `position_ids` is deliberately omitted so `UnmarkEncoder` derives and
        enforces the authoritative values from these same base ids (D-B4B-002).
        Stage-1 does not reimplement that.
        """
        outputs = self.unmark_encoder(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            tone_ids=tone_ids,
            tone_mask=tone_mask,
            letter_ids=letter_ids,
            letter_mask=letter_mask,
        )
        hidden = _hidden_states(outputs)
        return masked_mean_non_special(
            hidden, base_attention_mask, base_special_tokens_mask
        )

    # -- objective --------------------------------------------------------
    def forward(self, batch: dict[str, Any]) -> Stage1LossResult:
        """Run all three branches and combine them.

        Each branch is pooled **independently**; the reference and base sequence
        lengths may differ, and no token-level correspondence is assumed.
        """
        missing = _REQUIRED_FIELDS - set(batch)
        if missing:
            raise Stage1ContractViolation(f"batch is missing fields: {sorted(missing)}")

        h_ref = self.reference_representation(
            batch["reference_input_ids"],
            batch["reference_attention_mask"],
            batch["reference_special_tokens_mask"],
        )
        h_adapt_clean = self.adapted_representation(
            batch["base_input_ids"],
            batch["base_attention_mask"],
            batch["base_special_tokens_mask"],
            batch["clean_tone_ids"],
            batch["clean_tone_mask"],
            batch["clean_letter_ids"],
            batch["clean_letter_mask"],
        )
        h_adapt_corrupt = self.adapted_representation(
            batch["base_input_ids"],
            batch["base_attention_mask"],
            batch["base_special_tokens_mask"],
            batch["corrupt_tone_ids"],
            batch["corrupt_tone_mask"],
            batch["corrupt_letter_ids"],
            batch["corrupt_letter_mask"],
        )

        distance_align = _require_finite(
            representation_distance(h_adapt_corrupt, h_ref), "L_align distance"
        )
        distance_clean = _require_finite(
            representation_distance(h_adapt_clean, h_ref), "L_clean distance"
        )

        # Mean over examples, never sum: a summed loss would scale with batch size
        # and silently change the effective learning rate.
        loss_align = distance_align.mean()
        loss_clean = distance_clean.mean()
        loss = self.weights.lambda_align * loss_align + self.weights.lambda_clean * loss_clean

        return Stage1LossResult(
            loss=_require_finite(loss, "total loss"),
            loss_align=loss_align,
            loss_clean=loss_clean,
            distance_align_per_example=distance_align,
            distance_clean_per_example=distance_clean,
            weights=self.weights,
        )

    def train(self, mode: bool = True) -> Stage1Objective:
        """Delegate to `UnmarkEncoder`, which keeps the frozen encoder in eval.

        Overridden for the same reason as D-B4B-004: `nn.Module.train()` recurses
        into children, and the frozen encoder must not pick up train mode from a
        wrapper two levels above it.
        """
        super().train(mode)
        self.unmark_encoder.train(mode)
        return self


_REQUIRED_FIELDS = frozenset(
    {
        "reference_input_ids",
        "reference_attention_mask",
        "reference_special_tokens_mask",
        "base_input_ids",
        "base_attention_mask",
        "base_special_tokens_mask",
        "clean_tone_ids",
        "clean_tone_mask",
        "clean_letter_ids",
        "clean_letter_mask",
        "corrupt_tone_ids",
        "corrupt_tone_mask",
        "corrupt_letter_ids",
        "corrupt_letter_mask",
    }
)


def _hidden_states(outputs: Any) -> Tensor:
    """Final hidden states from a HuggingFace output or a bare tensor."""
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
        hidden = outputs[0]
    if hidden is None:
        hidden = outputs
    if not isinstance(hidden, Tensor):
        raise Stage1ContractViolation(
            f"could not obtain final hidden states from {type(outputs).__name__}"
        )
    return hidden
