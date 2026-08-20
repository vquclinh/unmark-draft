"""The UNMARK orthography input adapter `A_phi` (B4B).

Implements proposal §4.5 exactly as locked by B4A. **This module imports torch**
and is therefore *not* exported from `unmark.modeling.__init__`: the local
development environment is deliberately ML-free, and importing the package must
keep working there. Import this module explicitly, from Colab or any environment
that has torch::

    from unmark.modeling.adapter import OrthographyInputAdapter

**Nothing here trains.** There is no optimizer, no training loop, no checkpoint
saving, no dataset. A single backward pass is exercised only by the gradient
routing probe.

The locked equation (§4.5, §5.1, D-B4A-001)::

    q_i = [ e_i ; t_i ; l_i ]
    f_i = LN( W_f q_i + c_f )
    g_i = sigmoid( W_g q_i + c_g )
    z_i = g_i * f_i + (1 - g_i) * e_i

`z` replaces the **word** embedding only. Position and token-type embeddings are
supplied by the frozen encoder downstream of `inputs_embeds`, exactly once --
§4.5 calls double position encoding "the single most likely implementation bug in
the project", and the failure is silent.

Scope boundary. This module operates on **tensors**. It performs no
tokenization, no orthographic decomposition, no corruption, and no eligibility
classification: `text -> metadata` is the deterministic B1A/B2/B3 pipeline,
`metadata -> tensors` is a collator, and `tensors -> z` is this module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from unmark.modeling.config import AdapterConfig
from unmark.modeling.contracts import (
    GATE_INIT_BIAS,
    GATE_INIT_WEIGHT,
    LETTER_TABLE_ROWS,
    TONE_TABLE_ROWS,
)


class ChannelContractViolation(ValueError):
    """Raised when input tensors contradict the locked channel contract.

    Loud by design. A silently accepted out-of-range id would index the wrong
    learned row and the model would train happily on it.
    """


def _as_bool_mask(mask: Tensor, name: str) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if not mask.dtype.is_floating_point:
        return mask != 0
    raise ChannelContractViolation(
        f"{name} must be a boolean or integer mask, got dtype {mask.dtype}"
    )


def _validate_ids(ids: Tensor, mask: Tensor, rows: int, name: str) -> None:
    """Every *unmasked* id must index a real row.

    Masked positions are unconstrained -- that is where the out-of-table `NA`
    sentinel lives, and the point of the mask is that the sentinel never reaches
    the embedding table.
    """
    if ids.shape != mask.shape:
        raise ChannelContractViolation(
            f"{name}_ids {tuple(ids.shape)} and {name}_mask {tuple(mask.shape)} must match"
        )
    if ids.dtype.is_floating_point:
        raise ChannelContractViolation(f"{name}_ids must be integer, got {ids.dtype}")
    if not mask.any():
        return
    live = ids[mask]
    if live.numel() == 0:
        return
    lowest, highest = int(live.min()), int(live.max())
    if lowest < 0 or highest >= rows:
        raise ChannelContractViolation(
            f"{name}_ids has unmasked value(s) outside [0, {rows}): "
            f"min={lowest}, max={highest}. The NA sentinel must be masked out, not "
            f"passed through as a table index."
        )


def convex_combination(gate: Tensor, fused: Tensor, base: Tensor) -> Tensor:
    """`z = g * f + (1 - g) * e` -- the locked combination (§4.5).

    Exposed as a free function so the forced `g := 0` wiring identity
    (D-B4A-004) can be tested **on the primitive** without the adapter carrying
    a public gate-zero flag. Such a flag could silently enter an experiment;
    this cannot, because it takes the gate as an argument.
    """
    return gate * fused + (1.0 - gate) * base


class OrthographyInputAdapter(nn.Module):
    """Trainable input adapter over a frozen encoder's word embeddings.

    Args:
        config: the locked `AdapterConfig`. `hidden_size` must come from the real
            model's `config.hidden_size` -- D-B3B0-002 is OPEN and no backbone
            dimension is assumed.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        if config.fusion_kind != "linear":
            raise NotImplementedError(
                "only the locked single linear fusion is implemented; an MLP is an "
                "ablation and must state its own hidden size, activation and dropout (§4.5)"
            )
        self.config = config
        d = config.hidden_size
        wide = config.fusion_input_size  # 3d

        # Exactly 7 tone rows and 5 letter rows. NA is outside both tables.
        self.tone_embedding = nn.Embedding(TONE_TABLE_ROWS, d)
        self.letter_embedding = nn.Embedding(LETTER_TABLE_ROWS, d)

        self.fusion = nn.Linear(wide, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)
        self.gate = nn.Linear(wide, d, bias=True) if config.use_gate else None

        self.reset_gate_parameters()

    # -- initialisation ---------------------------------------------------
    def reset_gate_parameters(self) -> None:
        """D-B4A-003: `W_g = 0`, `c_g = logit(0.01)`, so `g = 0.01` everywhere.

        `W_g = 0` makes the gate input-independent at step zero, so every token
        and every hidden dimension starts at `sigmoid(c_g)`. This starts training
        close to the base-only pathway without injecting a randomly initialised
        fusion branch at half weight, and keeps a usable derivative
        (`g(1-g) ~= 0.0099`).

        `W_f`, the adapter LayerNorm and both embedding tables keep PyTorch's
        conventional defaults -- the proposal locks no separate initialisation
        for them, and inventing one would be a new hyperparameter.
        """
        if self.gate is None:
            return
        with torch.no_grad():
            self.gate.weight.fill_(GATE_INIT_WEIGHT)
            self.gate.bias.fill_(GATE_INIT_BIAS)

    # -- channels ---------------------------------------------------------
    def tone_channel(self, tone_ids: Tensor, tone_mask: Tensor) -> Tensor:
        """`t_i`, exactly zero at `NA` positions (D-B4A-002).

        The `-1` sentinel never reaches `nn.Embedding`: it is replaced by row 0
        before the lookup and the result is then zeroed by the mask, so the
        substituted row cannot influence the output.
        """
        mask = _as_bool_mask(tone_mask, "tone_mask")
        _validate_ids(tone_ids, mask, TONE_TABLE_ROWS, "tone")
        safe_ids = torch.where(mask, tone_ids, torch.zeros_like(tone_ids))
        embedded = self.tone_embedding(safe_ids)
        return embedded * mask.unsqueeze(-1).to(embedded.dtype)

    def letter_channel(self, letter_ids: Tensor, letter_mask: Tensor) -> Tensor:
        """`l_i`: arithmetic mean over applicable contributors, in embedding space.

        `NONE` is an applicable contributor and participates in the mean; `NA` is
        excluded by the mask. A token with **zero** applicable contributors gets
        the **exact zero vector** (D-B4A-005).

        The denominator is clamped only for vectorisation; the trailing
        `has_any` multiply is what makes the zero-contributor output exactly
        zero, rather than leaving it true by the accident that `sum(none) == 0`.
        """
        mask = _as_bool_mask(letter_mask, "letter_mask")
        _validate_ids(letter_ids, mask, LETTER_TABLE_ROWS, "letter")
        safe_ids = torch.where(mask, letter_ids, torch.zeros_like(letter_ids))
        embedded = self.letter_embedding(safe_ids)  # [B, L, K, d]

        weights = mask.unsqueeze(-1).to(embedded.dtype)
        numerator = (embedded * weights).sum(dim=-2)  # [B, L, d]
        count = weights.sum(dim=-2)  # [B, L, 1]
        pooled = numerator / count.clamp(min=1.0)
        has_any = (count > 0).to(pooled.dtype)
        return pooled * has_any

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        base_embeddings: Tensor,
        tone_ids: Tensor,
        tone_mask: Tensor,
        letter_ids: Tensor,
        letter_mask: Tensor,
    ) -> Tensor:
        """Return `z`, the adapted **word** embeddings, shaped `[B, L, d]`.

        Args:
            base_embeddings: `e = Emb_theta(b_i)`, `[B, L, d]`, from the frozen
                encoder's input word embedding table. Must **not** already carry
                position or token-type embeddings.
            tone_ids: `[B, L]`, rows `0..6`; `NA` positions may hold the sentinel.
            tone_mask: `[B, L]`, false at `NA`.
            letter_ids: `[B, L, K]`, rows `0..4`; `K` is the batch's `K_max`.
            letter_mask: `[B, L, K]`, false where a contributor is not applicable.
        """
        e = base_embeddings
        if e.dim() != 3:
            raise ChannelContractViolation(
                f"base_embeddings must be [B, L, d], got {tuple(e.shape)}"
            )
        if e.shape[-1] != self.config.hidden_size:
            raise ChannelContractViolation(
                f"base_embeddings last dim {e.shape[-1]} != hidden_size "
                f"{self.config.hidden_size}"
            )
        if letter_ids.dim() != 3:  # [B, L, K]
            raise ChannelContractViolation(
                f"letter_ids must be [B, L, K], got {tuple(letter_ids.shape)}"
            )

        t = self.tone_channel(tone_ids, tone_mask)
        l = self.letter_channel(letter_ids, letter_mask)
        q = torch.cat([e, t, l], dim=-1)

        f = self.layer_norm(self.fusion(q))
        if self.gate is None:
            return f
        g = torch.sigmoid(self.gate(q))
        return convex_combination(g, f, e)

    # -- introspection ----------------------------------------------------
    def gate_values(
        self,
        base_embeddings: Tensor,
        tone_ids: Tensor,
        tone_mask: Tensor,
        letter_ids: Tensor,
        letter_mask: Tensor,
    ) -> Tensor:
        """`g` for the given input, for probes that report the initial gate."""
        if self.gate is None:
            raise ValueError("this adapter was built without a gate")
        t = self.tone_channel(tone_ids, tone_mask)
        l = self.letter_channel(letter_ids, letter_mask)
        return torch.sigmoid(self.gate(torch.cat([base_embeddings, t, l], dim=-1)))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def expected_parameter_count(self) -> int:
        """`6d^2 + 16d`, from the locked contract."""
        return self.config.parameter_count().total


# ---------------------------------------------------------------------------
# Frozen-encoder integration
# ---------------------------------------------------------------------------
def freeze_encoder(encoder: nn.Module) -> int:
    """Freeze every pretrained parameter. Returns how many were frozen.

    §5.1: "Encoder: fully frozen; no layer unfrozen without a logged decision."
    Word embeddings, positional embeddings, token-type embeddings, every
    transformer block, every pretrained LayerNorm, and the pooler if present.
    """
    frozen = 0
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        frozen += 1
    encoder.eval()
    return frozen


def trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def base_word_embeddings(encoder: nn.Module, input_ids: Tensor) -> Tensor:
    """`Emb_theta(input_ids)` -- the **word** embedding table only.

    Deliberately *not* `encoder.embeddings(...)`, which would also add position
    and token-type embeddings and run the encoder's embedding LayerNorm and
    dropout. Those belong downstream of `inputs_embeds` and must happen exactly
    once.

    No copy is made: this reads the frozen table in place, so the adapter never
    owns a trainable alias of a pretrained parameter.
    """
    table = encoder.get_input_embeddings()
    if table is None:
        raise ValueError("encoder.get_input_embeddings() returned None")
    return table(input_ids)


class UnmarkEncoder(nn.Module):
    """Frozen encoder + trainable adapter, wired through `inputs_embeds`.

    The encoder is held as a submodule but every one of its parameters is
    frozen, so `trainable_parameters(self)` counts only `A_phi`.

    **Two different contracts, easy to conflate.** `requires_grad = False`
    freezes *weights*; `eval()` disables *stochastic training behaviour* such as
    dropout. Freezing does not imply eval, and `nn.Module.train()` recursively
    puts registered children into train mode -- so a plain
    `wrapper.train()` would silently reactivate the pretrained encoder's dropout
    while every encoder parameter stayed frozen. For UNMARK that would inject
    avoidable stochasticity into the alignment objective: the reference branch
    `h(x)` and the adapted branch would see different dropout draws of the same
    frozen encoder. `train()` below is overridden to prevent it.
    """

    def __init__(self, encoder: nn.Module, adapter: OrthographyInputAdapter) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        freeze_encoder(self.encoder)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> UnmarkEncoder:
        """Put the adapter in `mode`; keep the frozen encoder in eval, always.

        `nn.Module.train()` recurses into children first, which would flip the
        encoder to train mode. This calls `super().train(mode)` for the normal
        semantics -- the adapter follows `mode`, and the module's own
        `self.training` flag is set -- and then explicitly restores the encoder
        to eval.

        Requires-grad state is untouched: this is about module mode only. There
        is no flag to disable it; a frozen representation encoder running
        dropout is not a configuration anyone should be able to select by
        accident. Changing it needs a logged scientific decision.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def adapted_embeddings(
        self,
        input_ids: Tensor,
        tone_ids: Tensor,
        tone_mask: Tensor,
        letter_ids: Tensor,
        letter_mask: Tensor,
    ) -> Tensor:
        e = base_word_embeddings(self.encoder, input_ids)
        return self.adapter(e, tone_ids, tone_mask, letter_ids, letter_mask)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tone_ids: Tensor,
        tone_mask: Tensor,
        letter_ids: Tensor,
        letter_mask: Tensor,
        position_ids: Tensor | None = None,
        **encoder_kwargs: Any,
    ) -> Any:
        """Run the frozen encoder on the adapted word embeddings.

        `position_ids` is passed straight through when supplied. Whether it
        *must* be supplied is an empirical question about the real checkpoint,
        answered by `scripts/b4b_phobert_adapter_probe.py`; nothing is assumed
        here.
        """
        z = self.adapted_embeddings(input_ids, tone_ids, tone_mask, letter_ids, letter_mask)
        if position_ids is not None:
            encoder_kwargs["position_ids"] = position_ids
        return self.encoder(inputs_embeds=z, attention_mask=attention_mask, **encoder_kwargs)

    def trainable_parameter_count(self) -> int:
        return trainable_parameters(self)

    def frozen_encoder_trainable_count(self) -> int:
        return trainable_parameters(self.encoder)
