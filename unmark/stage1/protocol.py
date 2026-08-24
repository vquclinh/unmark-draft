"""The locked Stage-1 scientific protocol. **One source of truth, no torch.**

Every constant here was locked by [Audit 028 Revision 2] and recorded in
`docs/spec/decisions.md` (D-S1B-001 … D-S1B-004). Nothing in this module is a
tunable: a value that can reach an experiment does not get a convenient default
somewhere else, it gets pinned here and imported.

Mirrors `unmark/evaluation/preg1_protocol.py`, deliberately -- the pre-G1
protocol module is the audited house style for "the constants live in one place
and the runner imports them".

**Seeds are derived, never typed.** `derive_seeds(tag, count)` reads
`sha256(tag)` as consecutive 2-byte big-endian integers, so every seed below is
recomputable from its tag string alone and none can have been chosen to flatter
a result.
"""

from __future__ import annotations

from typing import Any

from unmark.evaluation.profiling import derive_seeds

STAGE1_PROTOCOL_VERSION = "stage1-protocol-v1"
"""Bump this and old Stage-1 artifacts stop being comparable."""

# ---------------------------------------------------------------------------
# Backbone -- D-B3B0-007
# ---------------------------------------------------------------------------
ENCODER_CHECKPOINT = "vinai/phobert-base"
ENCODER_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
HIDDEN_SIZE = 768
ENCODER_FROZEN = True
ADAPTER_TRAINABLE_PARAMETERS = 3_551_232
"""`6d^2 + 16d` at `d = 768`. Confirmed on the real model in Audit 019."""

# ---------------------------------------------------------------------------
# Corpus -- D-S1B-002
# ---------------------------------------------------------------------------
CORPUS_DATASET = "undertheseanlp/UVW-2026"
CORPUS_REVISION = "a0a79294e4568137e25828bb3f2a4cde8546e1fb"
CORPUS_SHARD_ORDER: tuple[str, ...] = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
)
"""Concatenation order. **Load-bearing:** it fixes document enumeration, and
`sample_id` keys the corruption draw, so a different order is a different
corruption stream at an identical revision."""

CORPUS_SHARD_LABELS_ARE_A_SPLIT = False
"""The upstream `train`/`validation`/`test` names are SOURCE SHARDS of one
unlabeled Wikipedia corpus. `test.parquet` is unrelated to UIT-VSFC's sealed
official TEST."""

REQUIRED_CORPUS_COLUMNS: tuple[str, ...] = ("id", "content")
"""Scientific correctness depends on these two and nothing else. Optional
metadata may exist and is never required."""

# ---------------------------------------------------------------------------
# Split -- D-S1B-002 / D-S1B-004
# ---------------------------------------------------------------------------
DEV_DOCUMENTS = 5_000
"""Exactly this many documents enter Stage-1 dev. A count, not a fraction, so
the selection signal's variance does not move with corpus size."""

# ---------------------------------------------------------------------------
# Sequence and chunking
# ---------------------------------------------------------------------------
MAX_LENGTH = 256
TRUNCATION_OFFERED = False
"""Trimming ids without the channel metadata would desynchronise the B3
projection. `OverflowBehaviour` has no TRUNCATE member for this reason."""
ON_OVERFLOW = "FAIL"
"""After correct pre-chunking nothing can overflow, so this is a **guard**, not
a data policy. Silent SKIP would bias the corpus toward short documents."""
CHUNK_ID_TEMPLATE = "{document_id}#{chunk_index}"
CHUNK_SCHEMA_VERSION = "stage1-chunk-v1"

RAW_BASE_POLICY = "RAW_BASE"
"""The base-pathway identity (D-B3B1A-001): no word segmentation, `b(canon(x))`.

Named so an operational checkpoint can bind it and refuse to resume a stream
prepared under a different base policy."""

# ---------------------------------------------------------------------------
# Corruption -- D-S1B-003
# ---------------------------------------------------------------------------
PI_STRIP = 0.25
"""P(scope = TONE_AND_LETTER) per example/visit. An a-priori researcher
decision, fixed before any Stage-1 result existed. **Never tuned** -- not on
UIT-VSFC, not on any downstream score, not on the Stage-1 held-out signal."""

RATE_NAMESPACE = "stage1-rate"
SCOPE_NAMESPACE = "stage1-scope"
"""Domain separation. The two draws must not share a scalar, and `scope` must
not be derived from `p` -- otherwise the letter-degraded regime would be
confined to part of the rate range and "letters missing" would be confounded
with corruption severity."""

CORRUPTION_RATE_DISTRIBUTION = "uniform_0_1_per_example"
CORRUPTION_REDRAW = "per_visit"

# ---------------------------------------------------------------------------
# Objective -- proposal 4.6, scale locked by Audit 028 G.2
# ---------------------------------------------------------------------------
DISTANCE = "cosine"
REPRESENTATION_LEVEL = "pooled"
STAGE1_POOLING = "attention_masked_mean_non_special"
"""**Stage-1 pooling. Not FIRST_TOKEN.** `FIRST_TOKEN` was scoped to the pre-G1
burden diagnostic only (D-G1-005 keeps final Stage-2 pooling OPEN); adopting it
here would silently change the Stage-1 objective."""

LAMBDA_SCALE_SUM = 2.0
"""`lambda_align + lambda_clean = 2` for every `r`, so varying the tradeoff does
not also vary the absolute loss scale. Under AdamW the scale is not neutral:
`eps` breaks scale invariance and decoupled weight decay does not scale with the
loss."""


def lambdas_for_r(r: float) -> tuple[float, float]:
    """`(lambda_align, lambda_clean)` for a ratio `r = lambda_clean/lambda_align`."""
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise TypeError(f"r must be a real number, got {r!r}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    return LAMBDA_SCALE_SUM / (1.0 + r), LAMBDA_SCALE_SUM * r / (1.0 + r)


# ---------------------------------------------------------------------------
# Optimizer -- D-S1B-004
# ---------------------------------------------------------------------------
OPTIMIZER = "adamw"
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
AMSGRAD = False
LR_SCHEDULE = "CONSTANT"
WARMUP = None
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CLIPPING = None
WEIGHT_DECAY_WEIGHTS = 0.01
WEIGHT_DECAY_EXEMPT = 0.0
"""Applies to biases, LayerNorm parameters, and **both embedding tables**.
Decaying the tone/letter tables would shrink channel information toward zero,
the opposite of Stage-1's purpose."""

PRECISION = "fp32"
"""**A-priori implementation choice, recorded before the first run.** No AMP,
no bf16, no fp16. The training GPU has 90+ GB; mixed precision would be a
memory optimisation that changes numerics for no scientific reason."""

BATCH_SIZE = 128
EVAL_EVERY_UPDATES = 500
INITIAL_MAX_UPDATES = 20_000
EXTENDED_MAX_UPDATES = 40_000
"""One continuation only. See `budget` in `unmark.stage1.selection`."""

# ---------------------------------------------------------------------------
# Validation -- D-S1B-004
# ---------------------------------------------------------------------------
VALIDATION_CONDITIONS: tuple[str, ...] = ("FULL", "P50", "P100", "STRIP_ALL")
"""Fixed grid. Candidates must be compared on identical corruptions, never on
random `p`."""

SELECTION_SCORE = "max over VALIDATION_CONDITIONS of mean cosine distance to h(x)"
CHECKPOINT_TIE_BREAK: tuple[str, ...] = ("lower d_clean", "earliest update")
R_CANDIDATE_TIE_BREAK: tuple[str, ...] = ("lower d_clean", "smaller r")
METRIC_UNIT = "prepared_chunk"
"""Aggregation unit for `d_c` and `d_clean`: the Stage-1 example, i.e. one
prepared chunk, unweighted. Recorded explicitly (D-S1B-005) rather than left to
an implementation accident -- document-weighted aggregation would silently
re-weight long articles."""

# ---------------------------------------------------------------------------
# Run plan -- D-S1B-004, exactly 11 runs
# ---------------------------------------------------------------------------
LR_PILOT_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
LR_PILOT_R = 1.0
R_PHASE1_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
TOTAL_NOMINAL_RUNS = 11

# ---------------------------------------------------------------------------
# Seeds -- D-S1B-004, derived and domain-separated
# ---------------------------------------------------------------------------
SEED_ROOT_TAG = "UNMARK-STAGE1-v1"

SELECTION_SEED_TAG = f"{SEED_ROOT_TAG}|selection"
TRAIN_SEED_TAGS: tuple[str, ...] = tuple(f"{SEED_ROOT_TAG}|train|{i}" for i in range(3))
CORRUPTION_SEED_TAG = f"{SEED_ROOT_TAG}|corruption"
SPLIT_SEED_TAG = f"{SEED_ROOT_TAG}|split"
VALIDATION_CORRUPTION_SEED_TAG = f"{SEED_ROOT_TAG}|validation-corruption"

SELECTION_SEED: int = derive_seeds(SELECTION_SEED_TAG, 1)[0]
TRAIN_SEEDS: tuple[int, ...] = tuple(derive_seeds(t, 1)[0] for t in TRAIN_SEED_TAGS)
CORRUPTION_SEED: int = derive_seeds(CORRUPTION_SEED_TAG, 1)[0]
SPLIT_SEED: int = derive_seeds(SPLIT_SEED_TAG, 1)[0]
VALIDATION_CORRUPTION_SEED: int = derive_seeds(VALIDATION_CORRUPTION_SEED_TAG, 1)[0]

ADAPTER_INIT_SEED_TAG = f"{SEED_ROOT_TAG}|adapter-init"
"""Domain tag for adapter initialisation (**D-S1B-016**).

`run_seed` keeps its existing meaning -- it seeds `DeterministicSampler`, and
that data-order semantics is unchanged. Initialisation gets its **own**
domain-separated stream derived from the same `run_seed`, in the established
style of `CORRUPTION_SEED_TAG` / `SPLIT_SEED_TAG`.

**Nothing else may enter the derivation.** Learning rate, `r`, candidate label,
execution order, device and GPU identity are all excluded, because all eight
hyperparameter-selection candidates deliberately share `run_seed`
(`SELECTION_SEED`): they must therefore share one initialisation, so an LR or
`r` sweep is a **paired** comparison that varies only its target. If LR entered
this derivation, "LR A beats LR B" would be confounded with "initialisation A
was luckier than initialisation B".
"""


def adapter_init_seed(run_seed: int) -> int:
    """The deterministic adapter-initialisation seed for a nominal run.

    A pure function of `run_seed` alone (D-S1B-016). Recomputable by anyone from
    the tag string, exactly as every other Stage-1 seed is.
    """
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError(f"run_seed must be an int, got {run_seed!r}")
    return derive_seeds(f"{ADAPTER_INIT_SEED_TAG}|{run_seed}", 1)[0]


ADAPTER_INIT_SEEDS: dict[int, int] = {
    seed: adapter_init_seed(seed) for seed in (SELECTION_SEED, *TRAIN_SEEDS)
}
"""The four init seeds the locked schedule actually uses. Recorded so the
FINAL CONFIGURATION FREEZE can compare code against a written table."""

ALL_SEEDS: dict[str, int] = {
    SELECTION_SEED_TAG: SELECTION_SEED,
    **{tag: seed for tag, seed in zip(TRAIN_SEED_TAGS, TRAIN_SEEDS)},
    CORRUPTION_SEED_TAG: CORRUPTION_SEED,
    SPLIT_SEED_TAG: SPLIT_SEED,
    VALIDATION_CORRUPTION_SEED_TAG: VALIDATION_CORRUPTION_SEED,
}

if set(ADAPTER_INIT_SEEDS.values()) & set(ALL_SEEDS.values()):  # pragma: no cover - import guard
    raise AssertionError(
        f"adapter-init seeds collide with role seeds: {sorted(ADAPTER_INIT_SEEDS.items())}. "
        "Domain separation exists so initialisation and data order cannot share an integer."
    )

if len(set(ALL_SEEDS.values())) != len(ALL_SEEDS):  # pragma: no cover - import guard
    raise AssertionError(
        f"Stage-1 role seeds collide: {sorted(ALL_SEEDS.items())}. Domain separation "
        "exists so training, selection and corruption cannot share an integer."
    )

# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
OFFICIAL_TEST_ACCESSIBLE = False
"""There is no argument, path or code route to UIT-VSFC official TEST anywhere
in the Stage-1 stack."""

CONTAMINATION_SCREEN_INPUTS: tuple[str, ...] = (
    "uitvsfc_derived_train",
    "uitvsfc_official_validation",
)
"""The only UIT-VSFC material the screen may read -- both already legitimately
opened by the pre-G1 protocol."""

CONTAMINATION_METHOD = "exact_canonical_duplicate"
"""`canon(x)` equality / sha256. **No fuzzy or semantic screening.**"""

NO_DOWNSTREAM_SELECTION = (
    "No UIT-VSFC or other downstream score may influence any Stage-1 value. "
    "Stage-1 selection uses held-out UNLABELED signals only (D-S1B-001)."
)

NO_RAW_TEXT_IN_REPORTS = (
    "Scientific run reports carry ids, digests, counts and provenance only. The "
    "prepared-corpus data artifact contains text because it IS the training "
    "dataset; reports and audits never copy it."
)


def protocol_dict() -> dict[str, Any]:
    """The whole locked protocol, for stamping into an artifact."""
    return {
        "version": STAGE1_PROTOCOL_VERSION,
        "encoder": {
            "checkpoint": ENCODER_CHECKPOINT,
            "revision": ENCODER_REVISION,
            "hidden_size": HIDDEN_SIZE,
            "frozen": ENCODER_FROZEN,
            "adapter_trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        },
        "corpus": {
            "dataset": CORPUS_DATASET,
            "revision": CORPUS_REVISION,
            "shard_order": list(CORPUS_SHARD_ORDER),
            "shard_labels_are_a_split": CORPUS_SHARD_LABELS_ARE_A_SPLIT,
            "required_columns": list(REQUIRED_CORPUS_COLUMNS),
            "dev_documents": DEV_DOCUMENTS,
        },
        "sequence": {
            "max_length": MAX_LENGTH,
            "on_overflow": ON_OVERFLOW,
            "truncation_offered": TRUNCATION_OFFERED,
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        },
        "corruption": {
            "rate_distribution": CORRUPTION_RATE_DISTRIBUTION,
            "redraw": CORRUPTION_REDRAW,
            "pi_strip": PI_STRIP,
            "rate_namespace": RATE_NAMESPACE,
            "scope_namespace": SCOPE_NAMESPACE,
        },
        "objective": {
            "distance": DISTANCE,
            "level": REPRESENTATION_LEVEL,
            "pooling": STAGE1_POOLING,
            "lambda_scale_sum": LAMBDA_SCALE_SUM,
        },
        "optimizer": {
            "name": OPTIMIZER,
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "amsgrad": AMSGRAD,
            "schedule": LR_SCHEDULE,
            "warmup": WARMUP,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "gradient_clipping": GRADIENT_CLIPPING,
            "weight_decay_weights": WEIGHT_DECAY_WEIGHTS,
            "weight_decay_exempt": WEIGHT_DECAY_EXEMPT,
            "precision": PRECISION,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "eval_every_updates": EVAL_EVERY_UPDATES,
            "initial_max_updates": INITIAL_MAX_UPDATES,
            "extended_max_updates": EXTENDED_MAX_UPDATES,
        },
        "validation": {
            "conditions": list(VALIDATION_CONDITIONS),
            "score": SELECTION_SCORE,
            "checkpoint_tie_break": list(CHECKPOINT_TIE_BREAK),
            "r_candidate_tie_break": list(R_CANDIDATE_TIE_BREAK),
            "metric_unit": METRIC_UNIT,
        },
        "run_plan": {
            "lr_pilot_grid": list(LR_PILOT_GRID),
            "lr_pilot_r": LR_PILOT_R,
            "r_phase1_grid": list(R_PHASE1_GRID),
            "final_seeds": list(TRAIN_SEEDS),
            "total_nominal_runs": TOTAL_NOMINAL_RUNS,
        },
        "seeds": dict(ALL_SEEDS),
        "boundaries": {
            "official_test_accessible": OFFICIAL_TEST_ACCESSIBLE,
            "contamination_screen_inputs": list(CONTAMINATION_SCREEN_INPUTS),
            "contamination_method": CONTAMINATION_METHOD,
            "no_downstream_selection": NO_DOWNSTREAM_SELECTION,
            "no_raw_text_in_reports": NO_RAW_TEXT_IN_REPORTS,
        },
    }
