"""Pure-data contracts for Stage-1 self-supervised alignment.

**No torch.** Enums, configuration records and the register of values that are
still scientifically OPEN.

Stage-1 aligns two *adapted* representations to a clean pretrained reference
(proposal §4.6)::

    L_align = D( h'(x_p), h(x) )
    L_clean = D( h'(x),   h(x) )
    L       = lambda_a * L_align + lambda_c * L_clean

with `D` the cosine distance at the **pooled** representation level.

The central rule in this module: **an API default is a scientific decision if it
can reach an experiment.** Every value the proposal leaves open is therefore a
*required* argument here, not a convenient default. `lambda_a = 1.0` would be a
choice nobody made, arriving silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

STAGE1_SCHEMA_VERSION = "stage1-v1"
"""Versions the Stage-1 preparation contract, so a later change to how examples
are prepared is visible rather than silent."""


class Stage1Branch(Enum):
    """The three forward pathways, kept distinct and auditable.

    They are named rather than positional because the whole class of bug this
    guards against is wiring one branch's tensors into another.
    """

    REFERENCE_CLEAN = "REFERENCE_CLEAN"
    """`h(x)`: frozen tokenizer on the clean text, frozen encoder, **no
    adapter, no channels, no b(x)**. The target."""

    ADAPTED_CLEAN = "ADAPTED_CLEAN"
    """`h'(x)`: `b(x)` -> `T(b(x))` -> clean channels -> `A_phi` -> frozen encoder."""

    ADAPTED_CORRUPT = "ADAPTED_CORRUPT"
    """`h'(x_p)`: the same base grid, corrupted channels."""

    @property
    def uses_adapter(self) -> bool:
        return self is not Stage1Branch.REFERENCE_CLEAN

    @property
    def requires_gradient(self) -> bool:
        """The reference is a target; only the adapted branches carry a graph."""
        return self.uses_adapter


class Stage1ContractViolation(ValueError):
    """Raised when Stage-1 inputs contradict a locked contract."""


class BaseInvarianceViolation(Stage1ContractViolation):
    """Raised when `b(C(x)) != b(x)` in prepared data.

    Loud by design. The deterministic phase established this equality
    (D-B3B2-001); if it fails here, the corruption or the decomposition has
    changed underneath Stage-1, and repairing it heuristically would hide a real
    regression behind a plausible-looking batch.
    """


class UnresolvedStage1Value(RuntimeError):
    """Raised when a scientifically OPEN value is needed but was never supplied.

    Mirrors B2's `EligibilityUnresolved`: the unsafe path is the one you have to
    ask for.
    """


# ---------------------------------------------------------------------------
# What is locked, and what is not
# ---------------------------------------------------------------------------
LOCKED_STAGE1_VALUES: dict[str, str] = {
    "corruption_redraw_schedule": (
        "redrawn per visit; `visit` = pass index (D-S1B-004)."
    ),
    "corruption_scope_policy": (
        "per-example Bernoulli mixture, P(TONE_AND_LETTER) = pi_strip = 0.25, "
        "drawn from a stream domain-separated from the rate draw (D-S1B-003). "
        "This is the 'optional second rate' proposal §4.6 anticipates; no "
        "independent per-syllable letter-dropout q is introduced."
    ),
    "corruption_rate_distribution": (
        "p ~ U(0,1) per example, continuous -- proposal §4.6 and the §5.1 lock. "
        "Not a fixed rate and not only the endpoints."
    ),
    "distance": "cosine distance (proposal §4.6)",
    "representation_level": (
        "pooled only; per-token alignment deferred because the branches do not "
        "share a token grid (proposal §4.6)"
    ),
    "pooling": (
        "attention-masked mean over non-special content tokens (D-B4A-006), "
        "computed independently per branch"
    ),
    "encoder": "fully frozen (proposal §5.1); one shared theta for all branches",
}

OPEN_STAGE1_VALUES: dict[str, str] = {
    "lambda_align": (
        "proposal §4.6: 'L = lambda_a*L_align + lambda_c*L_clean, tuned on a "
        "development split'. No value is locked. REQUIRED argument."
    ),
    "lambda_clean": "as lambda_align; REQUIRED argument.",
    "corpus": (
        "proposal §5 open-items table and §13 item 3: size, domain mix, and "
        "whether it should match the downstream task domains. Not chosen here."
    ),
    "max_length": (
        "no Stage-1 maximum sequence length is specified; §5.3's is about "
        "Stage-2 task datasets. REQUIRED when truncation is used."
    ),
    "truncation_behaviour": (
        "what to do when a sequence exceeds max_length, given that truncating "
        "input ids without the channel metadata would desynchronise B3 projection."
    ),
    "corpus_revision_pin": (
        "the UVW-2026 dataset is locked (D-S1B-002); the exact revision and the "
        "three parquet sha256 values are pinned in configs/data/uvw_2026.json and "
        "verified at load."
    ),
    "stage1_seed": "the concrete experiment seed; the API requires one explicitly.",
    "batch_size": "not specified for Stage-1.",
    "optimizer": "not implemented in this phase.",
    "learning_rate": "not specified.",
    "epochs_or_steps": "not specified.",
    "warmup_or_scheduler": "not specified.",
    "gradient_accumulation": "not specified.",
    "checkpoint_selection": "not specified.",
    "backbone_finalisation": "D-B3B0-002 is OPEN; the pinned revision is a probe revision.",
}


def require_resolved(name: str) -> None:
    """Refuse to proceed on a value the project has not decided."""
    if name in OPEN_STAGE1_VALUES:
        raise UnresolvedStage1Value(
            f"{name} is scientifically OPEN: {OPEN_STAGE1_VALUES[name]} "
            "Supply it explicitly; it must not acquire a default."
        )


@dataclass(frozen=True)
class ObjectiveWeights:
    """`lambda_a` and `lambda_c`. **Both required -- no defaults.**

    The proposal tunes these on a development split, so any default here would
    be a scientific value nobody chose, silently reaching an experiment.
    """

    lambda_align: float
    lambda_clean: float

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_align", self.lambda_align),
            ("lambda_clean", self.lambda_clean),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Stage1ContractViolation(f"{name} must be a real number, got {value!r}")
            if value != value or value in (float("inf"), float("-inf")):
                raise Stage1ContractViolation(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise Stage1ContractViolation(f"{name} must be non-negative, got {value!r}")
        if self.lambda_align == 0 and self.lambda_clean == 0:
            raise Stage1ContractViolation(
                "both weights are zero: the objective would be identically zero"
            )

    def to_dict(self) -> dict[str, float]:
        return {"lambda_align": self.lambda_align, "lambda_clean": self.lambda_clean}


class OverflowBehaviour(Enum):
    """What happens to a sequence longer than `max_length`.

    **Both FAIL and SKIP are scientific choices.** SKIP silently changes the
    Stage-1 corpus distribution by dropping long examples; FAIL changes which
    corpora are usable at all. Neither may be selected by omitting an argument.

    `TRUNCATE` does not exist: trimming ids without the channel metadata would
    desynchronise the B3 projection, and trimming both raises a question the
    proposal does not answer (what happens to a syllable cut in half).
    """

    FAIL = "FAIL"
    SKIP = "SKIP"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    """Only valid when `max_length is None`: nothing can overflow."""


@dataclass(frozen=True)
class TruncationPolicy:
    """Stage-1 length handling. **Every field is required -- no defaults.**

    `max_length` is scientifically OPEN, so `TruncationPolicy()` must not be
    constructible: an omitted argument would select "unbounded, fail" for an
    experiment without anyone choosing it.

    An **explicit** `max_length=None` is a legitimate caller statement --
    "intentionally unbounded for this call" -- and is different from an implicit
    default of `None`. Use `TruncationPolicy.unbounded()` to say so at the call
    site.
    """

    max_length: int | None
    on_overflow: OverflowBehaviour

    def __post_init__(self) -> None:
        if not isinstance(self.on_overflow, OverflowBehaviour):
            raise Stage1ContractViolation(
                f"on_overflow must be an OverflowBehaviour, got {self.on_overflow!r}. "
                "Truncation is not offered: it would desynchronise channels from ids."
            )
        if self.max_length is None:
            if self.on_overflow is not OverflowBehaviour.NOT_APPLICABLE:
                raise Stage1ContractViolation(
                    f"max_length is None but on_overflow is {self.on_overflow.value}: "
                    "nothing can overflow an unbounded policy. Use NOT_APPLICABLE."
                )
            return
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise Stage1ContractViolation(f"max_length must be an int or None, got {self.max_length!r}")
        if self.max_length <= 0:
            raise Stage1ContractViolation(f"max_length must be positive, got {self.max_length}")
        if self.on_overflow is OverflowBehaviour.NOT_APPLICABLE:
            raise Stage1ContractViolation(
                f"max_length is {self.max_length} but on_overflow is NOT_APPLICABLE: "
                "a bounded policy must state FAIL or SKIP, and both are scientific choices"
            )

    @classmethod
    def unbounded(cls) -> TruncationPolicy:
        """An explicit "no length bound for this call".

        Named so the intent is visible at the call site. It is a *statement*, not
        a default -- `prepare_example` still requires the argument.
        """
        return cls(max_length=None, on_overflow=OverflowBehaviour.NOT_APPLICABLE)

    @property
    def is_enabled(self) -> bool:
        return self.max_length is not None

    def check(self, length: int, what: str) -> bool:
        """True to keep the example. Raises or returns False when it overflows."""
        if not self.is_enabled or length <= self.max_length:
            return True
        if self.on_overflow is OverflowBehaviour.SKIP:
            return False
        raise Stage1ContractViolation(
            f"{what} length {length} exceeds max_length {self.max_length}. "
            "Stage-1 does not truncate, because trimming ids without the channel "
            "metadata would desynchronise the B3 projection (max_length policy is OPEN)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {"max_length": self.max_length, "on_overflow": self.on_overflow.value}


class Stage1Purpose(Enum):
    """Why a Stage-1 configuration is being built.

    Mirrors B2's `CorruptionPurpose`: the unsafe path is the one you have to ask
    for. A `DIAGNOSTIC` config may carry explicit wiring values; a `SCIENTIFIC`
    one **cannot be constructed at all** until the researcher has resolved the
    OPEN values, so a diagnostic number cannot drift into a training run.
    """

    DIAGNOSTIC = "DIAGNOSTIC"
    """Explicit values used only to exercise a forward/backward path. No
    optimizer, no parameter update. **Resolves nothing.**"""

    SCIENTIFIC = "SCIENTIFIC"
    """Defines a real Stage-1 experiment. Requires every OPEN value to be named
    as resolved."""


SCIENTIFIC_REQUIRED_VALUES: tuple[str, ...] = (
    "lambda_align",
    "lambda_clean",
    "corpus",
    "max_length",
    "truncation_behaviour",
    "stage1_seed",
    "batch_size",
)
"""OPEN values a scientific Stage-1 configuration must have resolved.

Training hyperparameters beyond these belong to the runner, which does not exist
yet, and to the PRE-TRAIN audit that must inspect it.
"""


@dataclass(frozen=True)
class Stage1RunConfig:
    """A Stage-1 configuration, stamped with why it exists.

    Args:
        purpose: `DIAGNOSTIC` or `SCIENTIFIC`. Required.
        resolved_values: names from `OPEN_STAGE1_VALUES` the researcher has
            decided. `SCIENTIFIC` raises unless it covers
            `SCIENTIFIC_REQUIRED_VALUES`.
    """

    purpose: Stage1Purpose
    weights: ObjectiveWeights
    truncation: TruncationPolicy
    corruption: CorruptionRatePolicy
    resolved_values: frozenset[str] = frozenset()
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, Stage1Purpose):
            raise Stage1ContractViolation("purpose must be a Stage1Purpose; it has no default")
        unknown = set(self.resolved_values) - set(OPEN_STAGE1_VALUES)
        if unknown:
            raise Stage1ContractViolation(
                f"resolved_values names items that are not in the OPEN register: {sorted(unknown)}"
            )
        if self.purpose is Stage1Purpose.SCIENTIFIC:
            if not self.corruption.is_locked_mixture:
                raise Stage1ContractViolation(
                    "a SCIENTIFIC Stage-1 configuration requires the locked corruption "
                    "mixture (D-S1B-003): forced_scope must be None and pi_strip must "
                    f"be {PI_STRIP}. A pinned scope is diagnostic only -- a run-global "
                    "TONE scope is exactly the defect that left STRIP-ALL with zero "
                    "training support."
                )
            missing = [v for v in SCIENTIFIC_REQUIRED_VALUES if v not in self.resolved_values]
            if missing:
                raise UnresolvedStage1Value(
                    "a SCIENTIFIC Stage-1 configuration requires these OPEN values to be "
                    f"resolved first: {missing}. They are not decided, so no scientific "
                    "Stage-1 run can be configured yet. A DIAGNOSTIC configuration may "
                    "carry explicit wiring values, which resolve nothing."
                )

    @property
    def is_diagnostic_only(self) -> bool:
        return self.purpose is Stage1Purpose.DIAGNOSTIC

    def to_dict(self) -> dict[str, Any]:
        """Run-artifact record. Diagnostic configurations are labelled as such."""
        return {
            "purpose": self.purpose.value,
            "diagnostic_only": self.is_diagnostic_only,
            "values_are_scientific": not self.is_diagnostic_only,
            "resolved_values": sorted(self.resolved_values),
            "note": self.note,
            **self.weights.to_dict(),
            **self.truncation.to_dict(),
            "corruption": self.corruption.to_dict(),
        }


SUPPORTED_SCOPES = ("TONE", "TONE_AND_LETTER")

RATE_NAMESPACE = "stage1-rate"
SCOPE_NAMESPACE = "stage1-scope"
"""Domain separation tags for the two independent draws (D-S1B-003)."""

from unmark.stage1.protocol import PI_STRIP  # noqa: E402 - single source of truth

"""`P(scope = TONE_AND_LETTER)` per example/visit -- **imported, never retyped.**

Until Audit 030 F2 this module typed its own `0.25` literal while
`protocol.PI_STRIP` typed another. The two agreed, so the science was correct
and the manifest was honest -- but they were independent: the corruption engine
read *this* one and every recorded artifact read *that* one, and a one-sided
edit would have gone unnoticed by the entire repository.

`protocol.py` is authoritative because it declares itself "the locked Stage-1
protocol. One source of truth", and because it already *derives* rather than
types (its seven role seeds come from `derive_seeds`). This module is a
mechanism module -- policies and config objects -- so it consumes the locked
value rather than declaring it. The direction is acyclic: `protocol` imports
only `unmark.evaluation.profiling`, which reaches `evaluation.contracts` and
`orthography` and never returns to `stage1.contracts`.

Changing the value now necessarily changes **both** corruption behaviour and
recorded protocol/manifest identity, which is the property F2 required.

Locked a-priori by the researcher (D-S1B-003) **before any Stage-1 result
existed**, and never tuned -- not on UIT-VSFC, not on any downstream score, and
not on the Stage-1 held-out signal.

It exists because the previous single-scope policy gave the headline evaluation
condition `STRIP-ALL` **zero training support**: with a run-global `"TONE"`
scope the corrupted branch's letter channel was bit-identical to the clean
branch's in every prepared example.
"""


@dataclass(frozen=True)
class CorruptionRatePolicy:
    """How the per-example corruption rate `p` **and scope** are drawn.

    **Locked:** `p ~ U(0,1)` per example, continuous (§4.6, §5.1); redraw per
    visit (D-S1B-004); scope drawn per example from a Bernoulli mixture with
    `P(TONE_AND_LETTER) = pi_strip = 0.25` (D-S1B-003).

    Two **domain-separated** keyed digests, never one shared scalar::

        rate_for (id, visit) -> blake2b(RATE_NAMESPACE  | schema | seed | id | visit)
        scope_for(id, visit) -> blake2b(SCOPE_NAMESPACE | schema | seed | id | visit)

    so that

        P(p | scope = TONE)            = U(0, 1)
        P(p | scope = TONE_AND_LETTER) = U(0, 1)

    up to the deterministic finite sample. Deriving `scope` from `p` -- for
    example "strip letters only when p > 0.9" -- would confine the
    letter-degraded regime to part of the rate range, confounding "letters
    missing" with corruption severity, so any measured STRIP-ALL behaviour could
    not be attributed to letter information alone.

    Both digests are reproducible from their key alone and **use no
    module-global RNG**: Python's `random` would make the same batch differ
    between processes.

    Args:
        forced_scope: **DIAGNOSTIC ONLY.** Pins one scope for every example
            instead of drawing the locked mixture. A scientific configuration
            must leave it `None`; `is_locked_mixture` reports which state this
            policy is in, and `Stage1RunConfig` refuses a SCIENTIFIC purpose
            without the mixture.
    """

    seed: int
    forced_scope: str | None = None
    pi_strip: float = PI_STRIP
    schema_version: str = STAGE1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise Stage1ContractViolation(f"seed must be an int, got {self.seed!r}")
        if self.forced_scope is not None and self.forced_scope not in SUPPORTED_SCOPES:
            raise Stage1ContractViolation(
                f"unsupported corruption scope {self.forced_scope!r}; supported: "
                f"{list(SUPPORTED_SCOPES)}"
            )
        if isinstance(self.pi_strip, bool) or not isinstance(self.pi_strip, (int, float)):
            raise Stage1ContractViolation(f"pi_strip must be a real number, got {self.pi_strip!r}")
        if not 0.0 <= self.pi_strip <= 1.0:
            raise Stage1ContractViolation(f"pi_strip must be in [0, 1], got {self.pi_strip}")

    @property
    def is_locked_mixture(self) -> bool:
        """True when this policy is the locked scientific one (D-S1B-003)."""
        return self.forced_scope is None and self.pi_strip == PI_STRIP

    @property
    def scope(self) -> str:
        """Back-compatible read of a **pinned** scope. Raises for the mixture.

        The mixture has no single scope, and returning one would be a lie that a
        caller could act on. Ask `scope_for(sample_id, visit)` instead.
        """
        if self.forced_scope is None:
            raise Stage1ContractViolation(
                "this policy draws the scope per example (the locked mixture); there "
                "is no run-global scope. Use scope_for(sample_id, visit)."
            )
        return self.forced_scope

    def _unit_draw(self, namespace: str, sample_id: str, visit: int) -> float:
        """One `[0, 1)` draw from a namespaced key. The only randomness here."""
        payload = "|".join(
            (namespace, self.schema_version, str(self.seed), str(sample_id), str(visit))
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(1 << 64)

    def rate_for(self, sample_id: str, visit: int = 0) -> float:
        """`p` for one example, in [0, 1). Deterministic and reproducible."""
        return self._unit_draw(RATE_NAMESPACE, sample_id, visit)

    def scope_for(self, sample_id: str, visit: int = 0) -> str:
        """The corruption scope for one example. Independent of `rate_for`.

        `"TONE_AND_LETTER"` with probability `pi_strip`, else `"TONE"`. Uses
        `< pi_strip`, matching B2's own `score < probability` selection rule.
        """
        if self.forced_scope is not None:
            return self.forced_scope
        draw = self._unit_draw(SCOPE_NAMESPACE, sample_id, visit)
        return "TONE_AND_LETTER" if draw < self.pi_strip else "TONE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "pi_strip": self.pi_strip,
            "forced_scope": self.forced_scope,
            "is_locked_mixture": self.is_locked_mixture,
            "rate_namespace": RATE_NAMESPACE,
            "scope_namespace": SCOPE_NAMESPACE,
            "schema_version": self.schema_version,
        }
