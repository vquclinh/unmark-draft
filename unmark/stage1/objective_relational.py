"""**V2-GRD (C2)** -- geometry / relational distillation. **Imports torch.**

Not re-exported from `unmark.stage1.__init__` -- the local environment is
ML-free. Import explicitly::

    from unmark.stage1.objective_relational import RelationalDistillationObjective

The second post-hoc UNMARK-v2 research candidate. It changes **one** thing: the
training objective. The adapter, the gate, both embedding tables, the fusion, the
tokenizer, the position-id semantics and the frozen PhoBERT backbone are all
untouched, the fusion stays `historical-fusion-v1`, and the candidate adds
**zero** model parameters.

The objective, locked::

    L_total = lambda_align * L_align
            + lambda_clean * L_clean
            + lambda_grd   * L_grd

    lambda_align = 1.0    (r = 1.0, the closed campaign's selected ratio)
    lambda_clean = 1.0
    lambda_grd   = 1.0    (protocol.LAMBDA_GRD -- pinned, never swept)

`L_align` and `L_clean` are the historical pooled terms, computed by the
historical code in `unmark.stage1.objective` and not reimplemented here.

`L_grd` is new. From the FINAL contextual hidden states of the three branches
Stage-1 already computes, take the FIRST_TOKEN row::

    T  = native_clean_first    [B, d]      the TEACHER
    Sc = adapted_clean_first   [B, d]
    Sr = adapted_corrupt_first [B, d]

and for any `X` define the row-normalised pairwise cosine Gram matrix::

    X_hat_i = X_i / clamp(||X_i||_2, min=1e-8)
    G(X)    = X_hat @ X_hat.T                        [B, B]

    R(S, T) = mean over i != j of ( G(S)[i,j] - G(T)[i,j] )^2

    L_rel_clean   = R( Sc, stop_gradient(T) )
    L_rel_corrupt = R( Sr, stop_gradient(T) )
    L_grd         = 0.5 * L_rel_clean + 0.5 * L_rel_corrupt

**The teacher is ALWAYS native PhoBERT on the CLEAN ORIGINAL input.** Never
PhoBERT on a corrupted condition. Distilling relations from a corrupted teacher
would pull the severe conditions toward Vanilla's *degraded* geometry, which is
the opposite of what UNMARK is for, and the clean teacher is what makes
"preserve the native decision geometry" a coherent target at every corruption
rate.

**Why FIRST_TOKEN and not the pooled vector.** `L_clean` already constrains the
masked-mean pooled space. Post-hoc diagnostic D4 found that even FULL UNMARK-A
had a native-vs-UNMARK FIRST_TOKEN cosine distance of ~0.303 and same-head
prediction agreement of ~0.485: the space Stage-2 actually reads was not
preserved. Relating pooled vectors here would re-test what the historical
objective already does.

**Why relations and not a direct alignment.** There is deliberately NO term of
the form `D(student_FIRST_TOKEN, teacher_FIRST_TOKEN)`. That is a different
hypothesis (direct pointwise distillation), and it would force the adapted
representation onto the native one rather than asking it to preserve the
*structure* among examples. FIRST_TOKEN is used here only to construct the
inter-example geometry.

**Why the diagonal is excluded.** `G(X)[i,i] = 1` by construction for both
student and teacher, so every diagonal term contributes exactly zero error. It is
dropped rather than tolerated: including `B` constant zeros in a mean over `B^2`
entries would silently scale the loss and its gradient by `(B-1)/B` for reasons
that have nothing to do with geometry.

**There is no optimizer, no scheduler, no training loop and no checkpointing
here.** This module computes a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from unmark.stage1.contracts import (
    GEOMETRY_RELATIONAL_OBJECTIVE,
    GEOMETRY_RELATIONAL_SPEC,
    ObjectiveIdentity,
    ObjectiveWeights,
    RelationalSpec,
    Stage1ContractViolation,
)
from unmark.stage1.objective import (
    _REQUIRED_FIELDS,
    Stage1Objective,
    _require_finite,
    representation_distance,
)
from unmark.stage1.protocol import RELATION_EPSILON

OBJECTIVE = GEOMETRY_RELATIONAL_OBJECTIVE
"""The durable scientific identity a V2-GRD run stamps into its artifacts."""

SPEC: RelationalSpec = GEOMETRY_RELATIONAL_SPEC
"""How the geometry is measured. Recorded in provenance, not re-typed here."""

MINIMUM_BATCH = 2
"""Relational geometry needs at least one off-diagonal pair.

At `B = 1` the Gram matrix is the single entry `G[0,0] = 1` for both student and
teacher, every off-diagonal set is empty, and the honest answer is not `0.0` --
`0.0` would read as "the geometry matched perfectly". The term fails closed
instead."""


@dataclass
class RelationalLossResult:
    """Structured V2-GRD loss output. Deliberately NOT a `Stage1LossResult`.

    It carries the historical three fields under their historical names, so the
    trainer's existing telemetry reads it unchanged, and adds the relational
    terms beside them. A separate type rather than a subclass, because
    `Stage1LossResult` describes an objective with exactly two terms.

    The relational losses are already scalars -- a Gram comparison reduces over
    the whole batch and has no per-example decomposition -- so unlike the pooled
    terms there is no `[B]` diagnostic to retain. The off-diagonal statistics are
    kept instead, because they are what a reader needs to see whether the
    geometry is actually converging.

    No raw text is stored here: the loss object travels into logs.
    """

    loss: Tensor
    loss_align: Tensor
    loss_clean: Tensor
    loss_grd: Tensor
    loss_rel_clean: Tensor
    loss_rel_corrupt: Tensor
    distance_align_per_example: Tensor
    distance_clean_per_example: Tensor
    teacher_offdiagonal_mean: Tensor
    weights: ObjectiveWeights
    objective: ObjectiveIdentity = OBJECTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach()),
            "loss_align": float(self.loss_align.detach()),
            "loss_clean": float(self.loss_clean.detach()),
            "loss_grd": float(self.loss_grd.detach()),
            "loss_rel_clean": float(self.loss_rel_clean.detach()),
            "loss_rel_corrupt": float(self.loss_rel_corrupt.detach()),
            "mean_distance_align": float(self.distance_align_per_example.detach().mean()),
            "mean_distance_clean": float(self.distance_clean_per_example.detach().mean()),
            "teacher_offdiag_cos_mean": float(self.teacher_offdiagonal_mean.detach()),
            "batch_size": int(self.distance_align_per_example.shape[0]),
            **self.weights.to_dict(),
            **self.objective.to_dict(),
        }


def first_token(hidden: Tensor) -> Tensor:
    """`hidden[:, 0, :]` -- the FIRST_TOKEN row. `[B, L, d] -> [B, d]`.

    The space Stage-2 reads. Deliberately **not** the masked mean: that space is
    already constrained by `L_clean`, and D4 measured the drift in this one.

    No mask is applied and none is needed -- position 0 is `<s>`, which is
    present and attended in every prepared example by construction.
    """
    if hidden.dim() != 3:
        raise Stage1ContractViolation(
            f"first_token expects [B, L, d] hidden states, got {tuple(hidden.shape)}"
        )
    if hidden.shape[1] < 1:
        raise Stage1ContractViolation(
            f"hidden states have sequence length {hidden.shape[1]}; there is no "
            "FIRST_TOKEN to take"
        )
    return hidden[:, 0, :]


def cosine_gram(x: Tensor, eps: float = RELATION_EPSILON) -> Tensor:
    """Row-normalised pairwise cosine Gram matrix. `[B, d] -> [B, B]`.

    ::

        x_hat_i = x_i / clamp(||x_i||_2, min=eps)
        G       = x_hat @ x_hat.T

    The norm reduces the **feature** dimension only (`dim=-1`), so `G[i, j]` is
    the cosine between example `i` and example `j`. No centering, no temperature,
    no top-k, no subsampling: every ordered pair in the batch participates.

    `clamp(min=eps)` at exactly `1e-8` is a finiteness guard for a degenerate
    zero-norm row, not a tuned value.
    """
    if x.dim() != 2:
        raise Stage1ContractViolation(
            f"cosine_gram expects a [B, d] matrix, got {tuple(x.shape)}"
        )
    normalised = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return normalised @ normalised.transpose(0, 1)


def offdiagonal_mask(batch_size: int, device: Any = None) -> Tensor:
    """The `[B, B]` boolean mask that selects `i != j`. Fails closed below 2."""
    if batch_size < MINIMUM_BATCH:
        raise Stage1ContractViolation(
            f"relational geometry needs at least {MINIMUM_BATCH} examples, got "
            f"{batch_size}. With a single example there is no off-diagonal pair, and "
            "returning 0.0 would read as a perfectly matched geometry rather than as "
            "an unmeasured one."
        )
    return ~torch.eye(batch_size, dtype=torch.bool, device=device)


def relational_distance(
    student: Tensor, teacher_target: Tensor, eps: float = RELATION_EPSILON
) -> Tensor:
    """`R(S, T)`: off-diagonal MSE between the two cosine Gram matrices. Scalar.

    ::

        R(S, T) = mean over i != j of ( G(S)[i,j] - G(T)[i,j] )^2

    The reduction is **one mean over all ordered off-diagonal pairs**, `B(B-1)`
    of them. Both Grams are symmetric, so an unordered-pair mean would give the
    identical number; the ordered definition is the one implemented and the one
    documented, and there is exactly one of them.

    Squared error, never L1, never KL, never a covariance loss. No labels, no
    positive/negative mining, no temperature, no top-k neighbours, no random pair
    subsampling, no feature centering, and no token-level relations -- the
    relations are strictly **between examples in the batch**.

    `teacher_target` is **detached here**, unconditionally. The native encoder is
    frozen and its branch already runs under `no_grad`, so this is belt and
    braces -- but the stop-gradient is part of the scientific statement, not an
    artefact of how the teacher happened to be computed, so it is written
    explicitly and tested explicitly.

    Raises:
        Stage1ContractViolation: on a shape mismatch, on `B < 2`, or on a
            non-finite input. Nothing is sanitised.
    """
    if student.dim() != 2 or teacher_target.dim() != 2:
        raise Stage1ContractViolation(
            f"relational_distance expects [B, d] matrices, got "
            f"{tuple(student.shape)} and {tuple(teacher_target.shape)}"
        )
    if student.shape != teacher_target.shape:
        raise Stage1ContractViolation(
            f"student {tuple(student.shape)} and teacher {tuple(teacher_target.shape)} "
            "must describe the same batch: relations are compared example to example."
        )
    _require_finite(student, "relational student representation")
    _require_finite(teacher_target, "relational teacher representation")

    # THE stop-gradient. The teacher is a fixed target; the students are pulled
    # onto its geometry, never the reverse.
    target = teacher_target.detach()

    mask = offdiagonal_mask(student.shape[0], device=student.device)
    difference = cosine_gram(student, eps) - cosine_gram(target, eps)
    return (difference[mask] ** 2).mean()


def offdiagonal_mean(gram: Tensor) -> Tensor:
    """Mean off-diagonal entry of a Gram matrix. **Diagnostic only.**"""
    return gram[offdiagonal_mask(gram.shape[0], device=gram.device)].mean()


class RelationalDistillationObjective(Stage1Objective):
    """V2-GRD: the historical two terms, plus `L_grd`. **Zero new parameters.**

    Subclasses `Stage1Objective` so the branches, the frozen-encoder guards, the
    train-mode delegation and the pooled distance are the *same code* the
    historical runs used, not a second implementation that could drift. The
    historical class is untouched.

    **Three encoder forwards per batch, the same as every other candidate.** The
    reference is one; the two adapted branches are one each. All three are taken
    through the `*_branch` primitives, which hand back the final hidden states
    and the pooled representation from a single call, so `L_grd` reads the
    FIRST_TOKEN rows of forwards that had to happen anyway and runs none of its
    own.

    Args:
        unmark_encoder: the B4B `UnmarkEncoder` -- frozen encoder plus adapter,
            built with the HISTORICAL fusion. C2 changes the loss, not the model.
        weights: `lambda_align` and `lambda_clean`. `lambda_grd` is not among
            them: it is locked in `protocol.LAMBDA_GRD` and carried by the
            objective identity, so there is nothing here to mis-set.
    """

    def __init__(self, unmark_encoder: nn.Module, weights: ObjectiveWeights) -> None:
        super().__init__(unmark_encoder, weights)
        self.objective = GEOMETRY_RELATIONAL_OBJECTIVE
        self.spec = SPEC

    def forward(self, batch: dict[str, Any]) -> RelationalLossResult:
        """Run the three branches once each and combine the three terms."""
        missing = _REQUIRED_FIELDS - set(batch)
        if missing:
            raise Stage1ContractViolation(f"batch is missing fields: {sorted(missing)}")

        base_input_ids = batch["base_input_ids"]
        base_attention_mask = batch["base_attention_mask"]
        base_special_tokens_mask = batch["base_special_tokens_mask"]

        # Branch 1/3. The native CLEAN reference -- the TEACHER. Its own
        # tokenization, under `no_grad`, exactly as the historical objective runs
        # it. It is never the corrupted encoder: there is no batch field here
        # that could supply one.
        hidden_reference, h_ref = self.reference_branch(
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

        # The FIRST_TOKEN rows of the three forwards above. No forward here.
        teacher = first_token(hidden_reference)
        student_clean = first_token(hidden_clean)
        student_corrupt = first_token(hidden_corrupt)

        loss_rel_clean = _require_finite(
            relational_distance(student_clean, teacher, self.spec.epsilon),
            "L_rel_clean",
        )
        loss_rel_corrupt = _require_finite(
            relational_distance(student_corrupt, teacher, self.spec.epsilon),
            "L_rel_corrupt",
        )
        # Exactly one half each. Not a tradeoff, and not a knob.
        loss_grd = (
            self.spec.relational_clean_weight * loss_rel_clean
            + self.spec.relational_corrupt_weight * loss_rel_corrupt
        )

        loss_align = distance_align.mean()
        loss_clean = distance_clean.mean()
        loss = (
            self.weights.lambda_align * loss_align
            + self.weights.lambda_clean * loss_clean
            + self.spec.lambda_grd * loss_grd
        )

        return RelationalLossResult(
            loss=_require_finite(loss, "total loss"),
            loss_align=loss_align,
            loss_clean=loss_clean,
            loss_grd=loss_grd,
            loss_rel_clean=loss_rel_clean,
            loss_rel_corrupt=loss_rel_corrupt,
            distance_align_per_example=distance_align,
            distance_clean_per_example=distance_clean,
            # DIAGNOSTIC. Computed from the teacher Gram this forward already
            # needed; detached, and read by telemetry only.
            teacher_offdiagonal_mean=offdiagonal_mean(
                cosine_gram(teacher.detach(), self.spec.epsilon)
            ),
            weights=self.weights,
            objective=self.objective,
        )
