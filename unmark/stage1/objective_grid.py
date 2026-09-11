"""**V2-GC** -- the Stage-1 grid-consistency objective. **This module imports torch.**

Not re-exported from `unmark.stage1.__init__` -- the local environment is
ML-free. Import explicitly::

    from unmark.stage1.objective_grid import GridConsistencyObjective

The first post-hoc UNMARK-v2 research candidate. It changes **one** thing: the
training objective. The adapter architecture, the gate, the tone and letter
embedding tables, the fusion, the tokenizer, the position-id semantics and the
frozen PhoBERT backbone are all untouched, and the candidate adds **zero** model
parameters.

The objective, locked::

    L_total = lambda_align * L_align
            + lambda_clean * L_clean
            + lambda_grid  * L_grid

    lambda_align = 1.0    (r = 1.0, the closed campaign's selected ratio)
    lambda_clean = 1.0
    lambda_grid  = 1.0    (protocol.LAMBDA_GRID -- pinned, never swept)

`L_align` and `L_clean` are the historical pooled terms, computed by the
historical code in `unmark.stage1.objective` and not reimplemented here.

`L_grid` is new. For each example and each **attended non-special base-grid
token** `i`::

    d_i = 1 - cos( H_corrupt[i], stop_gradient(H_clean[i]) )

    L_grid = mean over examples of ( mean over valid tokens i of d_i )

where `H_clean` and `H_corrupt` are the **final contextual hidden states** of the
frozen encoder -- before masked-mean pooling -- for the adapted CLEAN and adapted
CORRUPT branches.

**Why a token-level term is legitimate here and nowhere else.** §4.6 defers
per-token alignment because the branches do not share a token grid, and that is
still true of the Vanilla clean reference `h(x)`: it is tokenized from the
diacritized text and has its own, different, sequence. `L_grid` never touches it.
The two adapted branches, by contrast, are run on the *same* `base_input_ids`
tensor -- `b(C(x)) = b(x)` is the deterministic invariance D-B3B2-001 established
-- so position `i` means the identical base token in both, and the
correspondence is exact rather than inferred. This module fails closed rather
than repairing a grid that does not line up.

**Why the clean branch is the detached target.** `L_clean` already pulls the
adapted clean branch toward the pretrained reference, and that pull must stay
the only thing shaping it. If `L_grid` back-propagated into both branches it
could be minimised by moving the clean representation toward the corrupted one --
consistency bought by degrading the clean path. Detaching `H_clean` makes that
direction structurally unavailable: the corrupted branch is pulled onto the clean
branch's states, never the reverse.

**There is no optimizer, no scheduler, no training loop and no checkpointing
here.** This module computes a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from unmark.modeling.pooling import content_mask
from unmark.stage1.contracts import (
    GRID_CONSISTENCY_OBJECTIVE,
    GridConsistencyWeights,
    ObjectiveIdentity,
    Stage1ContractViolation,
)
from unmark.stage1.objective import (
    COSINE_EPS,
    _REQUIRED_FIELDS,
    Stage1Objective,
    _require_finite,
    representation_distance,
)

OBJECTIVE = GRID_CONSISTENCY_OBJECTIVE
"""The durable scientific identity a V2-GC run stamps into its artifacts."""


@dataclass
class GridConsistencyLossResult:
    """Structured V2-GC loss output. Deliberately NOT a `Stage1LossResult`.

    It carries the historical three fields under their historical names, so the
    trainer's existing telemetry and `fused`'s bookkeeping read it unchanged, and
    adds the grid term beside them. A separate type rather than a subclass,
    because `Stage1LossResult` describes an objective with exactly two terms and
    a V2-GC result is not one -- code that means to handle only the historical
    objective should not receive this by accident.

    Per-example distances are retained (they are `[B]`, and they are the
    diagnostic that shows the three terms behaving differently); hidden states
    are not, because they are `[B, L, d]` per branch and would pin two of them
    alive for every batch.

    No raw text is stored here: the loss object travels into logs.
    """

    loss: Tensor
    loss_align: Tensor
    loss_clean: Tensor
    loss_grid: Tensor
    distance_align_per_example: Tensor
    distance_clean_per_example: Tensor
    distance_grid_per_example: Tensor
    weights: GridConsistencyWeights
    objective: ObjectiveIdentity = OBJECTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach()),
            "loss_align": float(self.loss_align.detach()),
            "loss_clean": float(self.loss_clean.detach()),
            "loss_grid": float(self.loss_grid.detach()),
            "mean_distance_align": float(self.distance_align_per_example.detach().mean()),
            "mean_distance_clean": float(self.distance_clean_per_example.detach().mean()),
            "mean_distance_grid": float(self.distance_grid_per_example.detach().mean()),
            "batch_size": int(self.distance_align_per_example.shape[0]),
            **self.weights.to_dict(),
            **self.objective.to_dict(),
        }


def _require_shared_grid(
    hidden_corrupt: Tensor,
    hidden_clean: Tensor,
    attention_mask: Tensor,
    special_tokens_mask: Tensor,
) -> None:
    """Fail closed unless both branches really are on one base grid.

    Every check here is an equality that the caller's own construction should
    already guarantee -- both adapted branches are handed the same
    `base_input_ids`, `base_attention_mask` and `base_special_tokens_mask`. That
    is exactly why they are asserted: a token-level term is only meaningful while
    position `i` denotes the same base token in both branches, and the failure
    mode of a silently mismatched grid is a loss that still converges to
    something and means nothing. Nothing is padded, trimmed or reshaped to make
    the shapes agree.
    """
    if hidden_corrupt.dim() != 3 or hidden_clean.dim() != 3:
        raise Stage1ContractViolation(
            f"token-grid consistency expects [B, L, d] hidden states, got "
            f"{tuple(hidden_corrupt.shape)} and {tuple(hidden_clean.shape)}"
        )
    if hidden_corrupt.shape != hidden_clean.shape:
        raise Stage1ContractViolation(
            f"the adapted branches are not on one grid: corrupt hidden states "
            f"{tuple(hidden_corrupt.shape)} vs clean {tuple(hidden_clean.shape)}. "
            "L_grid is defined only where b(C(x)) = b(x) gives an exact token "
            "correspondence; it is not repaired here."
        )
    if attention_mask.shape != hidden_corrupt.shape[:2]:
        raise Stage1ContractViolation(
            f"attention_mask {tuple(attention_mask.shape)} does not match the base "
            f"grid [B, L] = {tuple(hidden_corrupt.shape[:2])}"
        )
    if special_tokens_mask.shape != attention_mask.shape:
        raise Stage1ContractViolation(
            f"special_tokens_mask {tuple(special_tokens_mask.shape)} does not match "
            f"attention_mask {tuple(attention_mask.shape)}"
        )


def token_grid_distance(
    hidden_corrupt: Tensor,
    hidden_clean_target: Tensor,
    attention_mask: Tensor,
    special_tokens_mask: Tensor,
) -> Tensor:
    """Per-example mean token cosine distance on the shared base grid. `-> [B]`.

    ::

        m_i  = attention_mask_i AND NOT special_tokens_mask_i
        d_i  = 1 - cos( H_corrupt[i], stop_gradient(H_clean[i]) )
        out  = sum_i m_i d_i / sum_i m_i          (per example)

    The mask is the **locked** one: `unmark.modeling.pooling.content_mask`, the
    same `attention_mask == 1 AND special_tokens_mask == 0` rule D-B4A-006 fixed
    for Stage-1 pooling. Padding and model special tokens are excluded, and this
    module does not define a second rule for what counts as a content token.

    The cosine is `torch.nn.functional.cosine_similarity` at the locked
    `objective.COSINE_EPS`, over the **feature** dimension -- the same call and
    the same epsilon semantics `representation_distance` uses, so the two Stage-1
    distances cannot acquire different numerics.

    `hidden_clean_target` is **detached here**, unconditionally. It is the
    consistency target: the term may pull the corrupted branch onto the clean
    branch's states and must never pull the clean branch toward the corrupted
    one. Callers cannot opt out, because the asymmetry is the scientific content
    of the term rather than a default.

    Raises:
        Stage1ContractViolation: on a grid mismatch, or if **any** example has
            zero valid tokens. Fails loud rather than emitting a zero distance
            for an example that was never measured, exactly as
            `masked_mean_non_special` does.
    """
    _require_shared_grid(
        hidden_corrupt, hidden_clean_target, attention_mask, special_tokens_mask
    )

    # THE stop-gradient. `L_clean` still back-propagates through the clean
    # adapted branch by its own path; this term contributes nothing to it.
    target = hidden_clean_target.detach()

    mask = content_mask(attention_mask, special_tokens_mask)  # [B, L] bool
    counts = mask.sum(dim=1)
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty:
        raise Stage1ContractViolation(
            f"examples {empty} have no valid base-grid tokens after masking: every "
            "position is padding or a special token. L_grid fails loud rather than "
            "scoring an example it never measured."
        )

    similarity = torch.nn.functional.cosine_similarity(
        hidden_corrupt, target, dim=-1, eps=COSINE_EPS
    )  # [B, L]
    distance = 1.0 - similarity

    # Mean over VALID tokens per example. Masked positions are zeroed before the
    # sum, so a padded position cannot contribute, and the denominator counts
    # only what was measured.
    weights = mask.to(distance.dtype)
    return (distance * weights).sum(dim=1) / counts.to(distance.dtype)


class GridConsistencyObjective(Stage1Objective):
    """V2-GC: the historical two terms, plus `L_grid`. **Zero new parameters.**

    Subclasses `Stage1Objective` so the branches, the frozen-encoder guards, the
    train-mode delegation and the pooled distance are the *same code* the
    historical runs used, not a second implementation that could drift. The
    historical class is untouched: constructing `Stage1Objective` still yields
    exactly the historical objective.

    **Three encoder forwards per batch, the same as the historical objective.**
    The reference is one; the two adapted branches are one each, taken through
    `adapted_branch`, which hands back the final hidden states and the pooled
    representation from a single call. `L_grid` reuses those hidden states and
    runs no forward of its own.

    Args:
        unmark_encoder: the B4B `UnmarkEncoder` -- frozen encoder plus adapter.
        weights: a `GridConsistencyWeights`. The plain `ObjectiveWeights` the
            historical objective takes is refused: it cannot state `lambda_grid`,
            and an objective with a third term whose weight came from a default
            is not an experiment anyone configured.
    """

    def __init__(
        self, unmark_encoder: nn.Module, weights: GridConsistencyWeights
    ) -> None:
        if not isinstance(weights, GridConsistencyWeights):
            raise Stage1ContractViolation(
                "V2-GC requires GridConsistencyWeights: lambda_grid is part of the "
                f"objective and is LOCKED at {GRID_CONSISTENCY_OBJECTIVE.lambda_grid}. "
                f"Got {type(weights).__name__}."
            )
        super().__init__(unmark_encoder, weights)
        self.objective = GRID_CONSISTENCY_OBJECTIVE

    def forward(self, batch: dict[str, Any]) -> GridConsistencyLossResult:
        """Run the three branches once each and combine the three terms."""
        missing = _REQUIRED_FIELDS - set(batch)
        if missing:
            raise Stage1ContractViolation(f"batch is missing fields: {sorted(missing)}")

        # The base grid. Read ONCE and passed to both adapted branches and to
        # L_grid, so "the two branches share a grid" is true by construction and
        # not by three separate lookups that a future edit could desynchronise.
        base_input_ids = batch["base_input_ids"]
        base_attention_mask = batch["base_attention_mask"]
        base_special_tokens_mask = batch["base_special_tokens_mask"]

        # Branch 1/3. The Vanilla clean reference, on its OWN tokenization. It
        # has no token correspondence with the base grid and takes no part in
        # L_grid -- only in the two pooled terms.
        h_ref = self.reference_representation(
            batch["reference_input_ids"],
            batch["reference_attention_mask"],
            batch["reference_special_tokens_mask"],
        )
        # Branch 2/3 and 3/3. Same base grid, clean vs corrupted channels.
        hidden_clean, h_adapt_clean = self.adapted_branch(
            base_input_ids,
            base_attention_mask,
            base_special_tokens_mask,
            batch["clean_tone_ids"],
            batch["clean_tone_mask"],
            batch["clean_letter_ids"],
            batch["clean_letter_mask"],
        )
        hidden_corrupt, h_adapt_corrupt = self.adapted_branch(
            base_input_ids,
            base_attention_mask,
            base_special_tokens_mask,
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
        distance_grid = _require_finite(
            token_grid_distance(
                hidden_corrupt,
                hidden_clean,
                base_attention_mask,
                base_special_tokens_mask,
            ),
            "L_grid distance",
        )

        # Mean over examples, never sum -- as in the historical objective, a
        # summed loss would scale with batch size and silently change the
        # effective learning rate.
        loss_align = distance_align.mean()
        loss_clean = distance_clean.mean()
        loss_grid = distance_grid.mean()
        loss = (
            self.weights.lambda_align * loss_align
            + self.weights.lambda_clean * loss_clean
            + self.weights.lambda_grid * loss_grid
        )

        return GridConsistencyLossResult(
            loss=_require_finite(loss, "total loss"),
            loss_align=loss_align,
            loss_clean=loss_clean,
            loss_grid=loss_grid,
            distance_align_per_example=distance_align,
            distance_clean_per_example=distance_clean,
            distance_grid_per_example=distance_grid,
            weights=self.weights,
            objective=self.objective,
        )
