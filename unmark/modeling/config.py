"""Immutable configuration and symbolic parameter accounting for `A_phi` (B4A).

**No torch.** Nothing here builds a tensor; it validates dimensions, records the
locked architecture, and derives the trainable parameter count symbolically so
the figure can be checked against proposal §4.7 without a backbone being chosen.

`d` is never defaulted to 768. D-B3B0-002 (backbone checkpoint) is OPEN, so the
contract stays backbone-parameterized; `d` must be supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from unmark.modeling.contracts import (
    FUSION_IS_CONVEX,
    GATE_ZERO_IS_WIRING_TEST_ONLY,
    GateContract,
    LetterChannelContract,
    Stage1PoolingContract,
    ToneChannelContract,
    TonePolicy,
    UnresolvedAdapterContract,
)

FUSION_INPUT_MULTIPLIER = 3
"""`[e_i ; t_i ; l_i]` -- three `d`-dimensional channels (§4.5)."""

FUSION_KIND = "linear"
"""§4.5: "a **single linear projection** `W_f in R^(d x 3d)`". MLP is an ablation
(§5.1, §6.6) and, if run, must state `h`, activation, dropout and its own count."""

LAYERNORM_POSITION = "after_fusion_before_gate_combination"
"""§5.1, verbatim: "LayerNorm after fusion, before the gate combination"."""

PARAMETER_FORMULA = "6*d**2 + 16*d"
"""`|phi| = 6d^2 + (4 + n_tau + n_lambda) d` with `n_tau = 7` and `n_lambda = 5`
(D-B4A-002, D-B4A-007) -- so `6d^2 + 16d`. `d` stays symbolic: D-B3B0-002 is OPEN."""

POSITION_EMBEDDINGS_SOURCE = "encoder_via_inputs_embeds"
"""§4.5: `z_i` replaces the WORD embedding only. Position and token-type
embeddings are supplied by the encoder, exactly once, downstream of
`inputs_embeds`. Adding them inside the adapter double-counts them and the
failure is silent -- the proposal calls this "the single most likely
implementation bug in the project"."""


@dataclass(frozen=True)
class ParameterCount:
    """Symbolic breakdown of `|phi|`, one term per component."""

    tone_embedding: int
    letter_embedding: int
    fusion_weight: int
    fusion_bias: int
    gate_weight: int
    gate_bias: int
    layernorm: int

    @property
    def total(self) -> int:
        return (
            self.tone_embedding
            + self.letter_embedding
            + self.fusion_weight
            + self.fusion_bias
            + self.gate_weight
            + self.gate_bias
            + self.layernorm
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "tone_embedding": self.tone_embedding,
            "letter_embedding": self.letter_embedding,
            "fusion_weight": self.fusion_weight,
            "fusion_bias": self.fusion_bias,
            "gate_weight": self.gate_weight,
            "gate_bias": self.gate_bias,
            "layernorm": self.layernorm,
            "total": self.total,
        }


@dataclass(frozen=True)
class AdapterConfig:
    """The adapter architecture, as locked by §4.5 / §5.1.

    Args:
        hidden_size: `d`, the frozen encoder's input-embedding dimension. Must be
            supplied -- there is no default while D-B3B0-002 is OPEN.
    """

    hidden_size: int
    tone: ToneChannelContract = field(default_factory=ToneChannelContract)
    letter: LetterChannelContract = field(default_factory=LetterChannelContract)
    gate: GateContract = field(default_factory=GateContract)
    stage1_pooling: Stage1PoolingContract = field(default_factory=Stage1PoolingContract)
    fusion_kind: str = FUSION_KIND
    use_gate: bool = True
    encoder_frozen: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_size, int) or isinstance(self.hidden_size, bool):
            raise TypeError(f"hidden_size must be an int, got {type(self.hidden_size).__name__}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not self.encoder_frozen:
            raise ValueError(
                "the encoder is fully frozen (proposal §5.1: 'no layer unfrozen without a "
                "logged decision'). Unfreezing requires a decision-log entry, not a flag."
            )
        if self.fusion_kind not in {"linear", "mlp"}:
            raise ValueError(f"unknown fusion_kind {self.fusion_kind!r}")

    @property
    def fusion_input_size(self) -> int:
        """`3d` -- the concatenated channel width."""
        return FUSION_INPUT_MULTIPLIER * self.hidden_size

    @property
    def resolved_decisions(self) -> tuple[str, ...]:
        """The B4A decisions this configuration embodies.

        All were OPEN when Audit 014 was first written; the researcher resolved
        them before B4B. Kept as a list rather than a boolean so the evidence
        chain stays visible in the object itself.
        """
        return ("D-B4A-002", "D-B4A-003", "D-B4A-004", "D-B4A-005", "D-B4A-006", "D-B4A-007")

    def parameter_count(self) -> ParameterCount:
        """Trainable parameters of `A_phi`, from the exact architecture.

        `|phi| = 6d^2 + (4 + n_tau + n_lambda) * d`

        With `n_tau = 7` and `n_lambda = 5` this is `6d^2 + 16d`.

        Excludes the Stage-2 task head: §8.3 trains it in a separate stage with
        the module frozen, and §4.7's budget lists no head.
        """
        d = self.hidden_size
        if self.fusion_kind != "linear":
            raise UnresolvedAdapterContract(
                "only the locked linear fusion is costed here; an MLP ablation must state "
                "its own hidden size, activation, dropout and parameter count (§4.5) "
                "before it can be costed"
            )
        return ParameterCount(
            tone_embedding=self.tone.embedding_rows * d,
            letter_embedding=self.letter.embedding_rows * d,
            fusion_weight=self.fusion_input_size * d,
            fusion_bias=d,
            gate_weight=self.fusion_input_size * d if self.use_gate else 0,
            gate_bias=d if self.use_gate else 0,
            layernorm=2 * d,
        )

    def initialisation_plan(self) -> dict[str, object]:
        """Exactly what B4B must set at construction time (D-B4A-003).

        `W_g = 0` makes the gate input-independent on step zero, so every
        position and dimension starts at `sigma(c_g) = 0.01` -- close to the
        pretrained base-only pathway, without the randomly initialised fusion
        branch arriving at half weight.
        """
        return {
            "gate_weight": self.gate.init_weight,
            "gate_bias": self.gate.init_bias,
            "initial_gate_value": self.gate.initial_gate_value,
            "initial_gate_derivative": self.gate.initial_gate_derivative,
            "gate_zero_is_attainable": self.gate.zero_is_attainable,
            "gate_zero_is_wiring_test_only": GATE_ZERO_IS_WIRING_TEST_ONLY,
            "note": (
                "g = 0.01 is close to the base-only pathway, not equal to it. No exact "
                "base-only equality is claimed at initialisation."
            ),
        }

    def tensor_shapes(self, batch_size: str | int = "B", seq_len: str | int = "L") -> dict[str, tuple]:
        """Symbolic shapes for every tensor in the contract.

        `K` is the per-token letter-contributor count. It is **ragged**: a batch
        property, not a locked hyperparameter, and no maximum is invented here.

        `tone_mask` is not decoration: D-B4A-002 puts `NA` outside the embedding
        table, so `tone_ids` carries an out-of-table sentinel there and the mask
        is what selects between a safe placeholder lookup and exact zero.
        """
        b, l, d = batch_size, seq_len, self.hidden_size
        return {
            "input_ids": (b, l),
            "attention_mask": (b, l),
            "special_tokens_mask": (b, l),
            "tone_ids": (b, l),
            "tone_mask": (b, l),
            "base_embeddings": (b, l, d),
            "tone_embeddings": (b, l, d),
            "letter_contributor_ids": (b, l, "K"),
            "letter_contributor_mask": (b, l, "K"),
            "pooled_letter": (b, l, d),
            "concatenated_channels": (b, l, self.fusion_input_size),
            "fusion_weight": (d, self.fusion_input_size),
            "fusion_output": (b, l, d),
            "layernorm_output": (b, l, d),
            "gate_weight": (d, self.fusion_input_size),
            "gate": (b, l, d),
            "inputs_embeds": (b, l, d),
            "encoder_hidden_states": (b, l, d),
            "stage1_pooled": (b, d),
        }

    def with_policy(self, policy: TonePolicy) -> AdapterConfig:
        """The same architecture under a different H4 tone policy.

        All three policies share one `7 x d` table, so `|phi|` is identical --
        that identity is the point of the equalization (§4.3). `NA` is outside
        the table (D-B4A-002), so it cannot perturb the count either.
        """
        return replace(self, tone=replace(self.tone, policy=policy))


def h4_equalized(config: AdapterConfig) -> bool:
    """Whether all three H4 policies would get identical trainable capacity.

    §4.3 introduces the 7-slot table specifically to "remove any objection that
    the oracle was granted extra capacity". D-B4A-002 keeps `NA` outside the
    table precisely so this stays true.
    """
    counts = {
        config.with_policy(policy).parameter_count().total for policy in TonePolicy
    }
    return len(counts) == 1


def fusion_equation() -> str:
    """The locked fusion equation, in the proposal's own notation (§4.5)."""
    return (
        "f_i = LN( W_f [ e_i ; t_i ; l_i ] + c_f )\n"
        "g_i = sigma( W_g [ e_i ; t_i ; l_i ] + c_g )  in (0,1)^d\n"
        "z_i = g_i * f_i + (1 - g_i) * e_i"
    )


assert FUSION_IS_CONVEX, "z_i is a convex combination, not a residual addition"
