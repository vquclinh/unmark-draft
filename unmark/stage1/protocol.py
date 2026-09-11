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
from unmark.modeling.contracts import (
    FUSION_IDS as _FUSION_IDS,
    HISTORICAL_FUSION_ID as _HISTORICAL_FUSION_ID,
    SCALE_CALIBRATED_FUSION_ID as _SCALE_CALIBRATED_FUSION_ID,
)

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
# Objective identity -- post-hoc V2 research candidates
# ---------------------------------------------------------------------------
HISTORICAL_OBJECTIVE_ID = "align-clean-pooled-v1"
"""The objective every historical UNMARK-A / UNMARK-B run was trained under.

    L = lambda_align * L_align + lambda_clean * L_clean

on the **pooled** representations, and nothing else. Named here so the
historical objective has a positive identity rather than being "whatever has no
name": a checkpoint that records this id is making a statement, and one that
records a different id can never be mistaken for a historical finalist.

Provenance written before this constant existed carries no objective id at all.
That absence is unambiguous -- only the historical objective could have produced
it -- so `RunProvenance.require_match` READS a missing block as this identity.
That is what keeps the frozen UNMARK-A/B checkpoints verifiable under the
unchanged finalist gate.
"""

GRID_CONSISTENCY_OBJECTIVE_ID = "grid-consistency-v1"
"""**V2-GC.** The first post-hoc UNMARK-v2 research candidate.

    L = lambda_align * L_align + lambda_clean * L_clean + lambda_grid * L_grid

`L_grid` is a token-level cosine consistency term between the adapted CLEAN and
adapted CORRUPT branches on the **invariant base token grid**, with the clean
branch detached. It adds ZERO model parameters: the adapter, the gate, both
embedding tables, the fusion, the tokenizer, the position-id semantics and the
frozen backbone are all untouched. The single isolated change is the training
objective.

Motivated by an UNLABELED diagnostic: UNMARK clean and corrupted inputs share a
bit-identical base-token grid, yet final hidden-token cosine drift is ~0.141
(P50), ~0.254 (P100) and ~0.314 (STRIP_ALL). The base embeddings agree exactly;
the contextual states do not.
"""

HISTORICAL_FUSION_ID = _HISTORICAL_FUSION_ID
SCALE_CALIBRATED_FUSION_ID = _SCALE_CALIBRATED_FUSION_ID
FUSION_IDS: tuple[str, ...] = _FUSION_IDS
"""The fusion rules, re-exported from `unmark.modeling.contracts`.

**Imported, never retyped.** The architecture owns the identity of its own
mixture rule; Stage-1 consumes it so a run artifact and the adapter it built can
never disagree about which equation was trained. Same direction as `PI_STRIP`:
one declaration, many consumers."""


RELATIONAL_OBJECTIVE_ID = "geometry-relational-distillation-v1"
"""**V2-GRD (C2).** The second post-hoc UNMARK-v2 research candidate.

    L = lambda_align * L_align + lambda_clean * L_clean + lambda_grd * L_grd

`L_grd` preserves the CLEAN NATIVE decision GEOMETRY -- the relations *between*
examples in a batch -- in the FIRST_TOKEN space Stage-2 actually reads, rather
than the masked-mean pooled space the historical objective constrains.

Motivated by post-hoc diagnostic D4 on protocol-dev: even FULL UNMARK-A showed a
native-vs-UNMARK FIRST_TOKEN cosine distance of ~0.303, same-Vanilla-head
prediction agreement of ~0.485 and a centered-logit cosine of ~0.23. The
historical objective constrains per-example pooled vectors and says nothing about
how examples sit relative to one another where the downstream head looks.

**The teacher is ALWAYS native PhoBERT on the CLEAN ORIGINAL input** -- never
PhoBERT on a corrupted condition. Distilling from a corrupted teacher would pull
severe conditions toward Vanilla's *degraded* behaviour, which is the opposite of
what UNMARK is for.

Adds ZERO model parameters and keeps the historical fusion: C2 is a
TRAINING-OBJECTIVE-ONLY candidate."""

OBJECTIVE_IDS: tuple[str, ...] = (
    HISTORICAL_OBJECTIVE_ID,
    GRID_CONSISTENCY_OBJECTIVE_ID,
    RELATIONAL_OBJECTIVE_ID,
)
"""Every objective identity this repository can train or verify. Closed set."""

LAMBDA_GRID = 1.0
"""`lambda_grid` for V2-GC. **LOCKED at 1.0 a-priori -- not a tuning grid.**

Deliberately a single pinned scalar and not a sweep: V2-GC exists to isolate the
effect of *adding* the grid-consistency term, so its weight is fixed before any
V2 number exists, exactly as `PI_STRIP` was. `GridConsistencyWeights` refuses any
other value, so there is no CLI flag, no config key and no code path that can
retune it without changing this line under review.

Note that it does NOT enter `LAMBDA_SCALE_SUM`: `lambda_align + lambda_clean = 2`
still holds for the two historical terms at every `r`, so the V2-GC run at
`r = 1.0` carries exactly the historical `(1.0, 1.0)` and adds a third term of
weight 1.0 beside them.
"""


LAMBDA_GRD = 1.0
"""`lambda_grd` for V2-GRD. **LOCKED at 1.0 a-priori -- not a tuning grid.**

Pinned before any C2 number exists, exactly as `LAMBDA_GRID` and `PI_STRIP` were.
`RelationalWeights` refuses any other value, so there is no CLI flag, no config
key and no code path that could retune it without changing this line under
review."""

RELATION_SPACE = "FIRST_TOKEN"
"""`hidden[:, 0, :]` of each branch's FINAL contextual hidden state.

**Deliberately not the masked mean.** The historical objective already
constrains the pooled space; C2 exists because Stage-2 reads FIRST_TOKEN and the
geometry there was not preserved. Using the pooled vector would re-test what
`L_clean` already does."""

RELATION_METRIC = "pairwise-cosine-gram"
"""`G(X) = X_hat @ X_hat.T`, `[B, B]`, with `X_hat` row-wise L2-normalised."""

RELATION_LOSS = "off-diagonal-mse"
"""Mean squared difference over the OFF-DIAGONAL entries of the two Gram
matrices. The diagonal is **excluded**, not merely expected to be small: every
`G[i,i]` is 1 by construction for both student and teacher, so including it would
dilute the loss with a constant-zero term whose only effect is to shrink the
gradient by a known factor."""

RELATIONAL_CLEAN_WEIGHT = 0.5
RELATIONAL_CORRUPT_WEIGHT = 0.5
"""`L_grd = 0.5 * (L_rel_clean + L_rel_corrupt)`. Exactly one half each.

Not a tradeoff to tune: the candidate asks whether relational geometry should be
preserved at all, so the clean and corrupted branches enter symmetrically. An
asymmetric weighting is a different experiment and needs its own objective id."""

RELATION_EPSILON = 1e-8
"""Denominator floor for the row-wise L2 normalisation. **Not tuned.**

The same order and the same role as `objective.COSINE_EPS` and
`modeling.contracts.FUSION_SCALE_EPSILON`: a guard so a degenerate zero-norm row
yields a finite, deterministic value rather than a NaN."""


def lambda_grd_for(objective_id: str) -> float | None:
    """The relational weight an objective identity implies. `None` = no GRD term.

    `None` is honest absence, never zero -- the historical, C1 and C3 objectives
    have no relational term at all, and recording `0.0` would describe a term
    that was computed and weighted away.
    """
    if objective_id not in OBJECTIVE_IDS:
        raise ValueError(
            f"unknown objective id {objective_id!r}; the closed set is {list(OBJECTIVE_IDS)}"
        )
    return LAMBDA_GRD if objective_id == RELATIONAL_OBJECTIVE_ID else None


def lambda_grid_for(objective_id: str) -> float | None:
    """The grid weight an objective identity implies. `None` = no grid term.

    `None` is honest absence, never zero: the historical objective has no third
    term at all, and recording `0.0` would describe a term that was computed and
    weighted away.
    """
    if objective_id not in OBJECTIVE_IDS:
        raise ValueError(
            f"unknown objective id {objective_id!r}; the closed set is {list(OBJECTIVE_IDS)}"
        )
    return LAMBDA_GRID if objective_id == GRID_CONSISTENCY_OBJECTIVE_ID else None

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
# V2-GC run plan -- ONE run, every value taken from the closed Stage-1 campaign
# ---------------------------------------------------------------------------
V2_GC_STAGE = "v2_gc"
"""Stage name. Its own namespace, so a V2-GC artifact can never be read as one
of the eleven historical nominal runs."""

V2_GC_LEARNING_RATE = LR_PILOT_GRID[0]
V2_GC_R = LR_PILOT_R
V2_GC_RUN_SEED = TRAIN_SEEDS[0]
"""**Nothing here is retuned.** The learning rate is the one the closed LR pilot
selected (1e-4, the first point of the locked grid), `r` is the one the closed
`r` phase selected (1.0, giving `lambda_align = lambda_clean = 1.0`), and the run
seed is the first FINAL MAIN train seed -- the same seed UNMARK-A came from. The
adapter init seed follows from it through `adapter_init_seed`, and the corruption
stream is the campaign-wide `CORRUPTION_SEED`.

Sharing the seed with the historical run is the point: V2-GC is a **paired**
comparison that varies the objective and nothing else.
"""

V2_GC_MAX_UPDATES = INITIAL_MAX_UPDATES
"""The same precommitted budget every historical run started under."""


# ---------------------------------------------------------------------------
# V2-SCF run plan -- ONE run, paired with V2-GC and with the historical run
# ---------------------------------------------------------------------------
V2_SCF_STAGE = "v2_scf"
"""Stage name for C1. Its own namespace and its own output directory."""

V2_SCF_LEARNING_RATE = V2_GC_LEARNING_RATE
V2_SCF_R = V2_GC_R
V2_SCF_RUN_SEED = V2_GC_RUN_SEED
V2_GRD_STAGE = "v2_grd"
V2_GRD_LEARNING_RATE = V2_GC_LEARNING_RATE
V2_GRD_R = V2_GC_R
V2_GRD_RUN_SEED = V2_GC_RUN_SEED
V2_GRD_MAX_UPDATES = V2_GC_MAX_UPDATES
"""**V2-GRD (C2) run plan.** The same already-closed values as C1 and C3.

All three candidates and the historical UNMARK-A run start from
`run_seed = 36930`, `init_seed = 51800`, `CORRUPTION_SEED`, LR 1e-4 and `r = 1.0`
under the same 20 000-update hard cap. C1 varies the FUSION, C2 and C3 vary the
OBJECTIVE in different ways, and none of them varies anything else."""

V2_SCF_MAX_UPDATES = V2_GC_MAX_UPDATES
"""**Inherited, never retuned -- and deliberately identical to V2-GC's.**

C1, C3 and the historical UNMARK-A run all start from `run_seed = 36930`,
`init_seed = adapter_init_seed(36930) = 51800`, `CORRUPTION_SEED`, LR 1e-4 and
`r = 1.0`, under the same 20 000-update hard cap. Three candidates that share
every value except the one thing each is testing is what makes the comparison
paired: C1 varies the FUSION, C3 varies the OBJECTIVE, and neither varies
anything else.

C1's adapter has the same parameter count and the same initialisation seed as the
historical adapter, so `expected_fresh_init_hash` is identical for both -- the
run starts from literally the same weights and diverges only through the mixture
rule."""


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
