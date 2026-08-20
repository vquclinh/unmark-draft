"""Pure-data contracts for the UNMARK neural adapter `A_phi` (B4A).

**No torch. No neural implementation.** This module holds the enums, label
mappings, initialisation plan and invariants extracted from
`unmark-proposal.md` §4.3-§4.7 and §5.1, so that B4B's `nn.Module` implements a
written contract instead of re-deriving one. The full extraction lives in
`docs/spec/neural-adapter.md`.

Audit 014 originally found three blocking ambiguities here -- how `NA` enters the
tone table (D-B4A-002), how the gate is initialised (D-B4A-003), and what the
letter channel contributes when no contributor is applicable (D-B4A-005). **All
three are now resolved by researcher decision**, together with the gate-zero test
form (D-B4A-004), Stage-1 pooling (D-B4A-006) and the letter cardinality
(D-B4A-007).

The rejected alternatives are kept in the enums rather than deleted: they carry
the evidence chain, and `require_locked()` rejects them explicitly instead of
letting a later reader rediscover them as plausible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def logit(p: float) -> float:
    """`ln(p / (1 - p))`, the inverse of the logistic sigmoid.

    Python stdlib only -- this module never imports numpy or torch.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit is defined on the open interval (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """The logistic sigmoid, for verifying the initialisation plan arithmetic."""
    return 1.0 / (1.0 + math.exp(-x))


class LockedContractViolation(ValueError):
    """Raised when a configuration contradicts a locked decision.

    Distinct from `UnresolvedAdapterContract`: that meant "nobody has decided
    yet", this means "somebody decided, and this contradicts it".
    """


class UnresolvedAdapterContract(RuntimeError):
    """Raised when something needs a decision that has not been made.

    Retained after the B4A resolutions: `AdapterConfig` still refuses to cost a
    fusion variant it has no specification for, and future channels may reuse it.
    Mirrors `EligibilityUnresolved` in B2 -- the unsafe path is the one you have
    to ask for.
    """


# ---------------------------------------------------------------------------
# Tone table
# ---------------------------------------------------------------------------
TONE_TABLE_ROWS = 7
"""Proposal §4.3 / §5.1: 7 slots = 5 marked tones + 2 policy slots.

**Exactly 7 trainable rows for every H4 policy.** That identity is the whole
point of the equalization (§4.3: it "removes any objection that the oracle was
granted extra capacity"). `NA` lives *outside* this table (D-B4A-002), so it
costs no rows and cannot perturb the equalization.
"""

MARKED_TONE_LABELS: tuple[str, ...] = ("SAC", "HUYEN", "HOI", "NGA", "NANG")
"""The five marked tones, in the proposal's own order (§4.3, §1.1)."""


class TonePolicy(Enum):
    """The three H4 tone-state policies (§6.7).

    They share one `7 x d` table and differ only in what the two policy slots
    mean. `ORACLE` is **not deployable** -- it uses knowledge that exists only
    because corruption is synthetic.

    These are the policies already recorded in the repository; no new policy is
    introduced here.
    """

    OBSERVABLE = "OBSERVABLE"
    FORCED_NGANG = "FORCED_NGANG"
    ORACLE = "ORACLE"

    @property
    def is_deployable(self) -> bool:
        return self is not TonePolicy.ORACLE

    @property
    def slot_a(self) -> str:
        return "UNMARKED" if self is TonePolicy.OBSERVABLE else "NGANG"

    @property
    def slot_b(self) -> str | None:
        """`MISSING` for the oracle; unused otherwise (§4.3).

        An unused slot still **exists** as a trainable row. Under `OBSERVABLE`
        the 7-row table is allocated in full even though row 6 is never indexed,
        because dropping it would give the policies different capacity.
        """
        return "MISSING" if self is TonePolicy.ORACLE else None

    @property
    def uses_slot_b(self) -> bool:
        return self.slot_b is not None


# The deployable (`OBSERVABLE`) tone id mapping. Deterministic and stable:
# 5 marked tones in proposal order, then policy slot A.
#
# `NA` is deliberately ABSENT: it is not a table row at all (D-B4A-002).
OBSERVABLE_TONE_IDS: dict[str, int] = {
    **{label: index for index, label in enumerate(MARKED_TONE_LABELS)},
    "UNMARKED": len(MARKED_TONE_LABELS),
}

TONE_SLOT_A_ID = len(MARKED_TONE_LABELS)
"""Index of policy slot A: `UNMARKED` under `OBSERVABLE`, *ngang* otherwise."""

TONE_SLOT_B_ID = TONE_TABLE_ROWS - 1
"""Index of policy slot B: `MISSING` under `ORACLE`, allocated but unused otherwise."""

TONE_NA_SENTINEL = -1
"""Out-of-table sentinel marking a position with no applicable tone.

**Never index an embedding table with this.** A torch implementation must use a
safe placeholder lookup plus masking (or an equivalent safe mechanism) and force
the resulting vector to exact zero. Feeding `-1` to `nn.Embedding` silently wraps
to the last row under some backends and raises under others -- neither is the
contract.
"""

TONE_NA_IS_ZERO_VECTOR = True
"""**D-B4A-002, RESOLVED.** `t_i = 0 in R^d` at non-applicable positions.

Applies to non-Vietnamese positions, special tokens, padding, and
multi-candidate ambiguous pieces. `NA` is structural non-applicability, not an
observable tone state, so it gets no learned row.
"""


class ToneNaTreatment(Enum):
    """How a non-applicable position enters the tone channel.

    **RESOLVED to `ZERO_VECTOR`** (D-B4A-002). The other two members are the
    rejected alternatives, retained so the evidence chain survives and so
    `require_locked()` can reject them by name:

    * `SLOT_B_ROW` -- reuse the unused policy slot. Breaks `ORACLE`, which needs
      slot B for `MISSING`.
    * `EXTRA_ROW` -- an eighth learned row. Contradicts the §5.1 7-slot lock and
      defeats the H4 equalization.
    """

    ZERO_VECTOR = "ZERO_VECTOR"
    SLOT_B_ROW = "SLOT_B_ROW"
    EXTRA_ROW = "EXTRA_ROW"


LOCKED_TONE_NA_TREATMENT = ToneNaTreatment.ZERO_VECTOR


# ---------------------------------------------------------------------------
# Letter table
# ---------------------------------------------------------------------------
APPLICABLE_LETTER_LABELS: tuple[str, ...] = (
    "NONE",
    "BREVE",
    "CIRCUMFLEX",
    "HORN",
    "STROKE",
)
"""The applicable closed set determined by B1A. **D-B4A-007, RESOLVED: n_lambda = 5.**

`NONE` is a learned member: "a letter that could carry a Vietnamese letter
diacritic and does not" is information, and D-B3B1C-001 includes it in the pooled
mean. `NA` is **not** a member -- it marks non-applicability and is excluded from
the mean.

The proposal's earlier `~10` / `n_letter=10` was a budget estimate and a stale
sketch, not the applicable closed set: Vietnamese places at most one
letter-forming mark per character, so the anticipated combination states do not
arise.
"""

LETTER_TABLE_ROWS = len(APPLICABLE_LETTER_LABELS)

LETTER_LABEL_IDS: dict[str, int] = {
    label: index for index, label in enumerate(APPLICABLE_LETTER_LABELS)
}

LETTER_NA_SENTINEL = -1
"""Out-of-table sentinel for a non-applicable contributor. Same warning as
`TONE_NA_SENTINEL`: never index a table with it."""

LETTER_EMPTY_IS_ZERO_VECTOR = True
"""**D-B4A-005, RESOLVED.** `l_i = 0 in R^d` when `|A_i| = 0`.

For a token with applicable contributor set `A_i`::

    |A_i| > 0   ->   l_i = (1 / |A_i|) * sum_{j in A_i} W_lambda[label_ij]
    |A_i| = 0   ->   l_i = 0

A torch implementation **may** clamp the denominator for vectorisation, but only
if the zero-contributor output is then explicitly forced to exact zero. Clamping
alone leaves `sum(0)/1 = 0` by accident rather than by contract, and the accident
stops holding the moment the numerator stops being empty-safe.
"""


class LetterEmptyTreatment(Enum):
    """What `l_i` is when a token has zero applicable contributors.

    **RESOLVED to `ZERO_VECTOR`** (D-B4A-005). Rejected alternatives retained:

    * `LEARNED_NA_ROW` -- a sixth learned row, contradicting `n_lambda = 5`.
    * `MASKED_OUT` -- changes the concatenation *width* at those positions and is
      therefore incompatible with a fixed `W_f in R^(d x 3d)`.
    """

    ZERO_VECTOR = "ZERO_VECTOR"
    LEARNED_NA_ROW = "LEARNED_NA_ROW"
    MASKED_OUT = "MASKED_OUT"


LOCKED_LETTER_EMPTY_TREATMENT = LetterEmptyTreatment.ZERO_VECTOR


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
GATE_TRANSFORM = "sigmoid"
"""Locked by §4.5 (`sigma`) and §5.1 ("per-dimension, sigma(W_g[.])").

"Per-dimension" describes the gate's OUTPUT shape -- one scalar per hidden
dimension per position -- not a position-independent parameter vector. The gate
is a second `3d -> d` projection; §4.7 bills it at ~1.8M. See D-B4A-001.
"""

GATE_IS_PROJECTION = True
"""`g_i = sigma(W_g [e_i ; t_i ; l_i] + c_g)`, not a raw trainable vector."""

FUSION_IS_CONVEX = True
"""`z_i = g_i * f_i + (1 - g_i) * e_i` -- convex, not residual `e + g * f`."""

GATE_INIT_TARGET = 0.01
"""**D-B4A-003, RESOLVED.** The gate's value at every position and dimension
before any learning."""

GATE_INIT_WEIGHT = 0.0
"""`W_g = 0` at initialisation, so `g` is input-independent on step zero."""

GATE_INIT_BIAS = logit(GATE_INIT_TARGET)
"""`c_g = logit(0.01) = ln(0.01/0.99) ~= -4.59511985013459`, every dimension.

Chosen over `c_g = 0` (which gives `g = 0.5`) so that a randomly initialised
fusion branch is not injected at 50% weight on step zero, and over a far more
negative bias so that the sigmoid derivative `g(1-g) ~= 0.0099` stays large
enough for the gate projection to learn.

This starts training **close to** the pretrained base-only pathway. It does not
start *at* it: `g = 0.01` is not `g = 0`, and no exact base-only equality is
claimed. See `GATE_ZERO_IS_ATTAINABLE`.
"""


class GateInit(Enum):
    """Gate initialisation. **RESOLVED to `NEAR_ZERO_LOGIT`** (D-B4A-003).

    Rejected alternatives retained: `ZERO_BIAS` gives `g = 0.5`;
    `POSITIVE_BIAS` starts at near-full fusion with the base term suppressed.
    """

    NEAR_ZERO_LOGIT = "NEAR_ZERO_LOGIT"
    ZERO_BIAS = "ZERO_BIAS"
    POSITIVE_BIAS = "POSITIVE_BIAS"


LOCKED_GATE_INIT = GateInit.NEAR_ZERO_LOGIT

GATE_ZERO_RECOVERS = "BASE_ONLY_PATHWAY"
"""What `g -> 0` recovers (§4.5): the base-only frozen-encoder pathway.

**Not** the clean-text pathway `E_theta(T(x))`. `e_i = Emb_theta(b_i)` is computed
from the stripped stream, so the original model is not recoverable at any gate
value. §4.5 corrects an earlier draft that claimed otherwise.
"""

GATE_ZERO_IS_ATTAINABLE = False
"""`sigma` maps onto the OPEN interval (0,1), so `g = 0` is a limit, not a value.

Exact recovery `z_i = e_i` needs `g_i[k] = 0` or `f_i[k] = e_i[k]` for every
`(i, k)`. The first is unreachable under `sigma`; the second would require the
adapter LayerNorm's affine parameters to invert normalisation for every token at
once. The proposal claims only `g_i -> 0`, so this is not an inconsistency in the
proposal -- it means the exact-equivalence check must be a wiring test. See
`GATE_ZERO_IS_WIRING_TEST_ONLY`.
"""

GATE_ZERO_IS_WIRING_TEST_ONLY = True
"""**D-B4A-004, RESOLVED.** A forced `g := 0` exists only as a test override.

Under that override `z == e` must hold exactly, up to ordinary floating-point
arithmetic. It is **not** a trainable parameterization, **not** an experiment
condition, **not** a claim that sigmoid attains zero, and **not** evidence that
the initialised module is identity.

No casual production "gate zero mode" may be exposed, because such a mode could
silently enter an experiment. B4B should test the fusion-combination primitive
directly rather than adding a public flag to the module.

Separately, B4B must measure the **real initialised** gate at `g = 0.01` against
the base-only pathway and report the difference, which is expected to be nonzero.
"""


# ---------------------------------------------------------------------------
# Stage-1 pooling
# ---------------------------------------------------------------------------
STAGE1_POOLING = "masked_mean_over_non_special_content_tokens"
"""**D-B4A-006, RESOLVED.** For final hidden states `H in R^[B, L, d]`::

    m_i = attention_mask_i * (1 - special_tokens_mask_i)
    h   = sum_i m_i H_i / sum_i m_i

computed independently per branch. Excludes `<s>`, `</s>`, `<pad>` and every
other tokenizer/model special token.
"""

STAGE1_POOLING_EXCLUDES_SPECIAL_TOKENS = True
STAGE1_POOLING_EXCLUDES_PADDING = True
STAGE1_POOLING_REQUIRES_EQUAL_BRANCH_LENGTHS = False
"""The clean reference branch and the adapted branch have different sequence
lengths -- `h(x)` runs the encoder's own tokenization of clean text, `h'(.)` runs
the base grid. Mean pooling maps both to `R^d` independently, and **no per-token
correspondence is assumed** (§4.6 defers per-token alignment for this reason).
"""

STAGE1_ZERO_CONTENT_POLICY = "FAIL_LOUD"
"""An example with no content positions after masking is an error.

Silently falling back to `<s>`, to an unmasked mean, or to a zero vector would
feed the cosine objective a value that is not a representation of anything.
"""


class Stage1PoolingError(ValueError):
    """Raised when an example has zero content positions after masking."""


@dataclass(frozen=True)
class Stage1PoolingContract:
    """The Stage-1 pooled representation (D-B4A-006)."""

    kind: str = STAGE1_POOLING
    exclude_special_tokens: bool = STAGE1_POOLING_EXCLUDES_SPECIAL_TOKENS
    exclude_padding: bool = STAGE1_POOLING_EXCLUDES_PADDING
    requires_equal_branch_lengths: bool = STAGE1_POOLING_REQUIRES_EQUAL_BRANCH_LENGTHS
    zero_content_policy: str = STAGE1_ZERO_CONTENT_POLICY

    def content_mask(
        self, attention_mask: list[int], special_tokens_mask: list[int]
    ) -> list[int]:
        """`m_i = attention_mask_i * (1 - special_tokens_mask_i)`.

        Pure Python on plain lists: this is the *contract*, evaluated on small
        examples so the masking semantics are testable without torch. B4B
        implements the tensor version.
        """
        if len(attention_mask) != len(special_tokens_mask):
            raise ValueError(
                f"mask lengths differ: {len(attention_mask)} vs {len(special_tokens_mask)}"
            )
        return [a * (1 - s) for a, s in zip(attention_mask, special_tokens_mask)]

    def require_content(self, mask: list[int]) -> int:
        """Number of content positions; raises when there are none."""
        count = sum(mask)
        if count == 0:
            raise Stage1PoolingError(
                "no content positions after masking: every position is padding or a "
                "special token. Stage-1 pooling fails loud rather than falling back to "
                "<s>, an unmasked mean, or a zero vector (D-B4A-006)."
            )
        return count


# ---------------------------------------------------------------------------
# Channel contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToneChannelContract:
    """The tone channel (D-B4A-002, RESOLVED)."""

    policy: TonePolicy = TonePolicy.OBSERVABLE
    na_treatment: ToneNaTreatment = LOCKED_TONE_NA_TREATMENT
    rows: int = TONE_TABLE_ROWS
    trainable: bool = True

    def __post_init__(self) -> None:
        self.require_locked()

    @property
    def embedding_rows(self) -> int:
        """Trainable rows in `W_tau`: always 7, for every policy."""
        return self.rows

    @property
    def na_is_zero_vector(self) -> bool:
        return self.na_treatment is ToneNaTreatment.ZERO_VECTOR

    def require_locked(self) -> None:
        if self.na_treatment is not LOCKED_TONE_NA_TREATMENT:
            raise LockedContractViolation(
                f"tone NA treatment {self.na_treatment.value} contradicts D-B4A-002, "
                "which locks the fixed zero vector. A learned NA row (EXTRA_ROW) breaks "
                "the 7-slot lock and the H4 equalization; SLOT_B_ROW collides with the "
                "oracle's MISSING slot."
            )
        if self.rows != TONE_TABLE_ROWS:
            raise LockedContractViolation(
                f"tone table must have exactly {TONE_TABLE_ROWS} trainable rows for every "
                f"H4 policy (§5.1), got {self.rows}"
            )


@dataclass(frozen=True)
class LetterChannelContract:
    """The letter channel (D-B4A-005, D-B4A-007, RESOLVED)."""

    empty_treatment: LetterEmptyTreatment = LOCKED_LETTER_EMPTY_TREATMENT
    rows: int = LETTER_TABLE_ROWS
    include_none_in_pool: bool = True
    exclude_na_from_pool: bool = True
    pooling: str = "mean"
    trainable: bool = True

    def __post_init__(self) -> None:
        self.require_locked()

    @property
    def embedding_rows(self) -> int:
        """Trainable rows in `W_lambda`: exactly 5."""
        return self.rows

    @property
    def empty_is_zero_vector(self) -> bool:
        return self.empty_treatment is LetterEmptyTreatment.ZERO_VECTOR

    def require_locked(self) -> None:
        if self.empty_treatment is not LOCKED_LETTER_EMPTY_TREATMENT:
            raise LockedContractViolation(
                f"letter empty treatment {self.empty_treatment.value} contradicts "
                "D-B4A-005, which locks the exact zero vector. LEARNED_NA_ROW would make "
                "n_lambda 6; MASKED_OUT changes the concatenation width and is "
                "incompatible with a fixed W_f."
            )
        if self.rows != LETTER_TABLE_ROWS:
            raise LockedContractViolation(
                f"letter table must have exactly {LETTER_TABLE_ROWS} rows (D-B4A-007), "
                f"got {self.rows}"
            )
        if not self.include_none_in_pool or not self.exclude_na_from_pool:
            raise LockedContractViolation(
                "D-B3B1C-001 locks NONE included in the pooled mean and NA excluded"
            )


@dataclass(frozen=True)
class GateContract:
    """The gate (D-B4A-003, D-B4A-004, RESOLVED)."""

    transform: str = GATE_TRANSFORM
    is_projection: bool = GATE_IS_PROJECTION
    initialisation: GateInit = LOCKED_GATE_INIT
    init_weight: float = GATE_INIT_WEIGHT
    init_bias: float = GATE_INIT_BIAS

    def __post_init__(self) -> None:
        self.require_locked()

    @property
    def initial_gate_value(self) -> float:
        """`sigma(W_g q + c_g)` at initialisation. `W_g = 0`, so the input drops
        out and every position and dimension starts at `sigma(c_g)`."""
        return sigmoid(self.init_bias)

    @property
    def initial_gate_derivative(self) -> float:
        """`g(1 - g)` -- the sigmoid slope at initialisation.

        Nonzero by design: an initialisation that drove `g` to machine zero would
        also drive this to zero and the gate projection could not learn.
        """
        g = self.initial_gate_value
        return g * (1.0 - g)

    @property
    def zero_is_attainable(self) -> bool:
        return GATE_ZERO_IS_ATTAINABLE

    @property
    def recovers_at_zero(self) -> str:
        return GATE_ZERO_RECOVERS

    def require_locked(self) -> None:
        if self.initialisation is not LOCKED_GATE_INIT:
            raise LockedContractViolation(
                f"gate initialisation {self.initialisation.value} contradicts D-B4A-003, "
                "which locks W_g = 0 and c_g = logit(0.01)"
            )
        if self.init_weight != 0.0:
            raise LockedContractViolation(
                f"D-B4A-003 locks W_g = 0 at initialisation, got {self.init_weight}"
            )
        if not math.isclose(self.init_bias, GATE_INIT_BIAS, rel_tol=0.0, abs_tol=1e-12):
            raise LockedContractViolation(
                f"D-B4A-003 locks c_g = logit(0.01) = {GATE_INIT_BIAS!r}, got {self.init_bias!r}"
            )
        if self.transform != GATE_TRANSFORM:
            raise LockedContractViolation(
                f"the gate transform is locked to {GATE_TRANSFORM} (§4.5, §5.1)"
            )
