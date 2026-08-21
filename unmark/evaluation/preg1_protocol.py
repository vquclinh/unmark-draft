"""The precommitted pre-G1 protocol. **A record, not a runner.**

**No torch. Nothing here trains.** This module fixes the protocol in the
repository *before* any Vanilla-vs-Base-only number exists, so it can be checked
afterwards against what was actually run.

Scope, kept deliberately narrow: this is the protocol for **one descriptive
pre-G1 clean-path burden diagnostic** on one dataset. It is **not** full G1
(§7 attaches the fusion layer and trains briefly), **not** the §6 multi-task
protocol, and **not** a final Stage-2 decision for the paper's grid.

**Supersession.** An earlier revision selected SA-VLSP2016. That is superseded
by UIT-VSFC v1.0 (D-PREG1-001b), decided while **zero** real Vanilla-vs-Base-only
scores existed. SA-VLSP2016 remains eligible for the later full benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from unmark.evaluation.profiling import (
    FIXED_MAX_LENGTH,
    LENGTH_REPORT_THRESHOLDS,
    SEED_DERIVATION_RULE,
    derive_seeds,
)

PREG1_PROTOCOL_VERSION = "preg1-protocol-v4"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
PRIMARY_DATASET = "UIT-VSFC"
PRIMARY_DATASET_VERSION = "1.0"
PRIMARY_DATASET_FULL_NAME = "Vietnamese Students' Feedback Corpus"
PRIMARY_TASK = "sentiment"
PRIMARY_NUM_LABELS = 3

LABEL_MAPPING: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
"""UIT-VSFC sentiment labels. Only the **sentiment** task is used; the corpus
also carries a topic annotation, which this diagnostic does not touch."""

PUBLISHED_SPLIT_SIZES: dict[str, int] = {"train": 11426, "validation": 1583, "test": 3166}

PUBLISHED_LABEL_COUNTS: dict[str, dict[str, int]] = {
    "train": {"negative": 5325, "neutral": 458, "positive": 5643},
    "validation": {"negative": 705, "neutral": 73, "positive": 805},
    "test": {"negative": 1409, "neutral": 167, "positive": 1590},
}
"""Published per-split class counts. Each sums exactly to its split size --
verified, and worth stating because the corpus is strongly imbalanced:
`neutral` is about 4% of train, which is why macro-F1 and per-class F1 are the
reported metrics rather than accuracy alone."""

# ---------------------------------------------------------------------------
# Conflicting canonical groups, resolved against real data (D-PREG1-011)
# ---------------------------------------------------------------------------
CONFLICTING_GROUP_POLICY = "EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP"
"""What to do when one canonical text carries more than one gold label.

`DUPLICATE_CONTRACT` requires a STOP for researcher review rather than a silent
fix. The review happened, and the resolution is to drop the **entire** group --
every member -- not to keep one member, majority-vote, or relabel.

Keeping a member would require choosing which annotation is correct, which this
diagnostic has no evidence to do and no need to do.

The contradictory supervision is avoidable annotation noise. Pairing does **not
guarantee** that its effect cancels: Vanilla and Base-only use different
representations and may respond differently during optimization or checkpoint
selection, so a shared noisy label can still reach `Delta_s` asymmetrically.
Excluding the whole group removes the ambiguity symmetrically without asserting
that either annotation is correct.
"""

CONFLICTING_GROUP_EXCLUSION_SCOPE = "protocol-train pool only; the official validation and test splits are untouched"

OBSERVED_CONFLICTING_GROUPS: dict[str, tuple[str, ...]] = {
    "a193a8ff49cc5ab43da189f9126aea19a0a0e9df1e16acc0a710cf7e880d0daa": (
        "train:11293",
        "train:11417",
    ),
}
"""Canonical-digest -> member sample ids, observed on the real TRAIN split.

Digests and ids only. The raw sentence is **not** recorded here: it is corpus
text, and the profiler's whole discipline is that committed evidence carries
hashes and counts rather than data.

Externally observed on Colab. Exactly one such group exists in TRAIN; the
official validation and test splits have none.
"""

DERIVED_TRAIN_SIZE = 11424
"""`PUBLISHED_SPLIT_SIZES['train']` (11426) minus the two excluded members."""

DERIVED_TRAIN_LABEL_COUNTS: dict[str, int] = {
    "negative": 5324,
    "neutral": 458,
    "positive": 5642,
}
"""The excluded pair is one `negative` and one `positive` -- which is what made
the group conflicting. `neutral` is unchanged, so the 4% minority class the
metric choice depends on is not affected."""

DERIVED_TRAIN_CSV_SHA256 = (
    "a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301"
)
"""SHA-256 of the derived exclusion-applied TRAIN csv, as produced on Colab. The
file is **not** in this repository; the digest is the reproducibility handle."""

SUPERSEDED_DATASET = "SA-VLSP2016"
SUPERSESSION_NOTE = (
    "SA-VLSP2016 was the Audit-021 selection and is SUPERSEDED for the pre-G1 "
    "diagnostic. The change was made while ZERO real Vanilla-vs-Base-only scores "
    "existed, so no result influenced it. SA-VLSP2016 remains eligible for the "
    "later full benchmark."
)

DATASET_SELECTION_RATIONALE: tuple[str, ...] = (
    "the diagnostic wants the cleanest identifiable x -> b(x) manipulation, not "
    "the most realistic noisy social-media benchmark",
    "the UIT-VSFC paper describes an explicit normalization phase: sentence "
    "segmentation, abbreviation expansion, misspelling correction and personal-name "
    "anonymisation, producing >16,000 normalized sentences",
    "it has an official train/validation/test structure",
    "its size makes a stable paired probe inexpensive",
    "its official validation split can stay untouched by head-protocol tuning",
)

DATASET_CAVEAT = (
    "This does NOT claim the corpus is perfectly diacritized. The paper's "
    "normalization description is not evidence about orthographic exposure -- the "
    "profiler must measure that directly on the real data."
)

DATASET_LOCK_RULE = (
    "The dataset is LOCKED for this diagnostic. Profiling is an integrity and "
    "characterisation gate, NOT a downstream-score-based selection contest. If "
    "profiling reveals a catastrophic integrity problem, STOP and require a new "
    "explicit researcher decision -- do not automatically switch datasets."
)

COMPARATOR_DATASETS: tuple[str, ...] = ("UIT-VSMEC", "UIT-ViHSD", "ViSpamReviews")


# ---------------------------------------------------------------------------
# Pathways
# ---------------------------------------------------------------------------
PATHWAY_CONTRACT = (
    "VANILLA: canon(x) -> PhoBERT tokenizer -> frozen encoder. "
    "BASE_ONLY: canon(x) -> b(x) -> the SAME tokenizer -> the SAME frozen encoder."
)

NO_WORD_SEGMENTER = (
    "No VnCoreNLP or other word segmenter is introduced into either pathway. The "
    "diagnostic must differ in the pathway transformation only; adding "
    "segmentation would introduce a second variable. This is consistent with the "
    "locked RAW_BASE contract (D-B3B1A-001)."
)

RAW_BASE_CAVEAT = (
    "Standard PhoBERT usage expects word-segmented Vietnamese, while RAW_BASE is "
    "a deliberate project design choice and a possible source of distribution "
    "shift. That caveat is preserved, not silently 'fixed' by segmenting."
)


# ---------------------------------------------------------------------------
# Pooling and head
# ---------------------------------------------------------------------------
class Preg1Pooling(Enum):
    """Stage-2 pooling for **this** pre-G1 protocol only."""

    FIRST_TOKEN = "FIRST_TOKEN"
    """The `<s>` classifier-token representation. No mean pooling."""


PREG1_POOLING = Preg1Pooling.FIRST_TOKEN

PREG1_POOLING_SCOPE = (
    "This resolves Stage-2 pooling for the pre-G1 burden diagnostic ONLY. It does "
    "NOT change the Stage-1 masked-mean pooling locked in §4.6, and it does NOT "
    "lock pooling for the final full experimental grid -- §5.2's head pooling "
    "remains OPEN (D-G1-005)."
)

HEAD_ARCHITECTURE = "linear"
HEAD_SHAPE = "Linear(d, 3, bias=True)"
HEAD_HIDDEN_LAYERS = 0
HEAD_DROPOUT = 0.0
HEAD_BIAS = True
HEAD_LAYERNORM = False
HEAD_ACTIVATION = None
HEAD_HIDDEN_SIZE_SOURCE = "model.config.hidden_size"
"""`d` is read from the model. **Never hardcoded** -- D-B3B0-002 is OPEN."""

HEAD_INIT_WEIGHT = "torch.nn.init.xavier_uniform_"
HEAD_INIT_BIAS = "torch.nn.init.zeros_"
HEAD_INIT_IS_EXPLICIT = True

HEAD_INIT_RULE = (
    "The classifier weight is initialised with xavier_uniform_ and the bias with "
    "zeros_, EXPLICITLY. nn.Linear's implicit default is NOT relied upon: it is a "
    "Kaiming-uniform variant whose exact form has changed across PyTorch versions, "
    "so depending on it would make the paired comparison silently version-sensitive."
)

RUN_INITIALISATION_SEQUENCE: tuple[str, ...] = (
    "reset the run RNGs from the declared run seed BEFORE head construction",
    "construct the head",
    "explicitly apply xavier_uniform_ to the weight and zeros_ to the bias",
    "construct a deterministic shuffle generator from the same run seed",
)

PAIRED_INITIALISATION_RULE = (
    "For measurement seed s, Vanilla(s) and BaseOnly(s) MUST start from "
    "BIT-IDENTICAL classifier parameters. Sharing a seed LABEL is not enough: "
    "running Vanilla and then Base-only from one advancing RNG stream would give "
    "the second pathway a different initialisation, and the difference would be "
    "attributed to the pathway. Each run therefore RE-SEEDS from s before "
    "constructing its head."
)

PAIRED_DATA_ORDER_RULE = (
    "The data order is paired too: for the same seed, both pathways see the same "
    "example ids, the same labels and the same deterministic shuffle schedule. "
    "Only the input pathway differs -- which is what makes Delta_s = Vanilla_s - "
    "BaseOnly_s a genuinely paired measurement rather than two independent runs "
    "subtracted."
)

NO_SHARED_ADVANCING_RNG = (
    "No shared advancing RNG stream across pathways. Both re-seed from the run "
    "seed; neither inherits RNG state from the other."
)


LOSS = "cross_entropy"
LOSS_CLASS_WEIGHTS = None
LOSS_LABEL_SMOOTHING = 0.0
LOSS_REDUCTION = "mean"
LOSS_SPEC = (
    'CrossEntropyLoss(weight=None, label_smoothing=0.0, reduction="mean")'
)
LOSS_RATIONALE = (
    "Ordinary multiclass cross-entropy, with no class weights, no focal loss and "
    "no label smoothing. The downstream instrument stays simple and identical "
    "across pathways; the known class imbalance is EXPOSED through macro-F1 and "
    "per-class F1 rather than compensated by a second modelling intervention that "
    "would itself become a variable."
)


# ---------------------------------------------------------------------------
# Encoder / numerics
# ---------------------------------------------------------------------------
ENCODER_CHECKPOINT = "vinai/phobert-base"
ENCODER_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
ENCODER_FROZEN = True
ENCODER_EVAL_MODE = True
REPRESENTATION_EXTRACTION = "torch.no_grad()"
PRECISION = "FP32"
MIXED_PRECISION = False

ENCODER_PIN_SCOPE = (
    "The pinned revision is a PIN FOR THIS PROBE'S REPRODUCIBILITY. It does NOT "
    "close D-B3B0-002, which remains OPEN for the final paper backbone decision."
)


# ---------------------------------------------------------------------------
# max_length
# ---------------------------------------------------------------------------
MAX_LENGTH = FIXED_MAX_LENGTH
TRUNCATION = True
PADDING = "max_length"

MAX_LENGTH_RATIONALE = (
    "Fixed at 256 for both pathways. Pre-G1 aims to MINIMISE truncation rather "
    "than optimise inference efficiency, and compute is not a constraint here. "
    "PhoBERT's pretrained positional capacity is 256 for a task sequence, so this "
    "is the maximum supported length -- which removes an otherwise data-dependent "
    "protocol decision from the measurement."
)

MAX_LENGTH_SUPERSESSION = (
    "SUPERSEDES the earlier rule 'smallest of {64,128,256} covering >=99% of train "
    "on both pathways'. Length statistics are still reported -- distributions, "
    "coverage at 64/128/256, the overflow rate at 256, and the Vanilla/Base-only "
    "delta -- but they no longer SELECT max_length."
)

OVERFLOW_POLICY = (
    "If records overflow 256, report the exact aggregate rate and apply ordinary "
    "truncation to 256. Do not exceed the verified backbone limit automatically."
)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
OFFICIAL_TEST_SEALED = True
OFFICIAL_TEST_SEAL_RULE = (
    "The official TEST split is SEALED. It may be read for integrity, hash and "
    "duplicate checks ONLY -- never for protocol decisions and never for scores."
)

OFFICIAL_VALIDATION_ROLE = "measurement-dev"
OFFICIAL_VALIDATION_RULE = (
    "The official VALIDATION split is the MEASUREMENT set. It is NOT used to "
    "select the dataset, pooling, learning rate, epoch, or any head "
    "hyperparameter, and the head is never tuned on it."
)

INTERNAL_SPLIT_FRACTIONS: dict[str, float] = {"protocol-train": 0.80, "protocol-dev": 0.20}

INTERNAL_SPLIT_RATIONALE = (
    "SUPERSEDES the earlier 70/15/15 division. Because the official validation "
    "split now serves as measurement-dev, the internal division of official train "
    "needs only two parts. 20% protocol-dev gives a more stable macro-F1 tuning "
    "sample on a corpus where `neutral` is about 4% of train, while still leaving "
    "over 9,000 training examples."
)

SPLIT_SEED_TAG = "UNMARK-PREG1-SPLIT-UITVSFC-v1"
SPLIT_SEED: int = derive_seeds(SPLIT_SEED_TAG, 1)[0]

SPLITTER_REQUIREMENTS: tuple[str, ...] = (
    "deterministic and stable across reruns",
    "label-stratified as closely as grouping allows",
    "group-aware by canonical text, so a canonical duplicate cannot cross splits",
    "independent of any downstream score",
)

SPLITTER_STATUS = (
    "The generic deterministic mechanism is implemented "
    "(`profiling.stratified_group_split`) but is NOT run on real data in this "
    "phase: conflicting-label canonical groups must be inspected first, and how to "
    "handle them is a researcher decision."
)

DUPLICATE_CONTRACT: tuple[str, ...] = (
    "canonical duplicates must stay in one group and cannot cross "
    "protocol-train/protocol-dev",
    "official train <-> validation overlap must be detected and reported before "
    "head training; leakage is never silently permitted",
    "conflicting-label canonical groups are reported with ids and counts, never "
    "silently relabelled or dropped",
    "if real data contains such groups in a way that affects split integrity, STOP "
    "for researcher review before downstream training",
)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
OPTIMIZER = "AdamW"
ADAMW_BETAS: tuple[float, float] = (0.9, 0.999)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.01
WEIGHT_DECAY_WEIGHT = 0.01
WEIGHT_DECAY_BIAS = 0.0
PARAM_GROUP_RULE = (
    "Two AdamW parameter groups: the head WEIGHT matrix decays at 0.01, the head "
    "BIAS at 0.0. The classifier intercept is NOT decayed -- shrinking it pulls "
    "the decision boundary toward the origin, which on a 4%-neutral corpus would "
    "penalise the minority class through a regularisation choice rather than "
    "through the data."
)
AMSGRAD = False
LR_SCHEDULE = "CONSTANT"
WARMUP_STEPS = 0
GRADIENT_CLIPPING = None
BATCH_SIZE = 128
EPOCHS = 30
EARLY_STOPPING = False
SHUFFLE_TRAINING_DATA = True
SHUFFLE_IS_DETERMINISTIC_UNDER_SEED = True
ENCODER_GRADIENT_OR_UPDATE = False
GRADIENT_ACCUMULATION_STEPS = 1
DROP_LAST = False

FIRST_CHECKPOINT_EPOCH = 1
LAST_CHECKPOINT_EPOCH = 30
CHECKPOINT_ELIGIBLE_EPOCHS: tuple[int, ...] = tuple(
    range(FIRST_CHECKPOINT_EPOCH, LAST_CHECKPOINT_EPOCH + 1)
)

CHECKPOINT_ELIGIBILITY_RULE = (
    "Evaluate and select after each COMPLETE training epoch. Epochs are numbered "
    "1..30. Epoch 0 -- the untrained head -- is NOT checkpoint-eligible: an "
    "untrained linear head on a 4%-neutral corpus can post a deceptively "
    "reasonable accuracy by predicting a majority class, and letting it win a "
    "checkpoint would report the initialisation rather than the pathway."
)


def is_checkpoint_eligible(epoch: int) -> bool:
    """Whether an epoch may be selected as a checkpoint. Epoch 0 never can."""
    return epoch in CHECKPOINT_ELIGIBLE_EPOCHS


RUNTIME_OPTIONS_NOTE = (
    "Implementation-level AdamW options (foreach, fused, capturable) vary by "
    "PyTorch version. They are NOT scientific hyperparameters and must not be "
    "tuned; the future run artifact records the actual runtime version and the "
    "options in force."
)

LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)

CHECKPOINT_RULE: tuple[str, ...] = (
    "highest protocol-dev Macro-F1",
    "then higher Accuracy",
    "then earliest epoch",
)

LR_AGGREGATION_RULE: tuple[str, ...] = (
    "highest MEAN selected-checkpoint Macro-F1 across the tuning seeds",
    "then highest MEAN Accuracy",
    "then LOWEST sample SD of Macro-F1",
    "then SMALLER learning rate",
)

PRIMARY_LR_SELECTION = (
    "For each LR and each tuning seed: train the linear head for all 30 epochs, "
    "evaluate every epoch on protocol-dev, and select that run's checkpoint by the "
    "checkpoint rule. Aggregate across the 3 tuning seeds by the aggregation rule. "
    "VANILLA only. The winning LR is then FROZEN and reused unchanged for BOTH "
    "pathways. Official validation is never used in this selection, and the grid "
    "is not altered after viewing Base-only results."
)

PRIMARY_LR_CAVEAT = (
    "Tuning the LR on Vanilla does NOT make Vanilla an upper bound. It makes the "
    "protocol shared and the comparison interpretable; it does not establish that "
    "Base-only could not do better under its own tuning."
)

SECONDARY_SENSITIVITY = (
    "A SECONDARY sensitivity analysis may later let each pathway select its own LR "
    "using exactly the same grid, the same 3 tuning seeds, the same protocol-dev, "
    "the same 30-epoch budget and the same checkpoint rule. It answers 'best "
    "achievable head fit under equal tuning budget' and MUST NOT replace the "
    "headline primary shared-LR result."
)


# ---------------------------------------------------------------------------
# Seeds and reporting
# ---------------------------------------------------------------------------
TUNING_SEED_TAG = "UNMARK-PREG1-TUNE-v1"
MEASUREMENT_SEED_TAG = "UNMARK-PREG1-MEASURE-v1"

TUNING_SEEDS: tuple[int, ...] = derive_seeds(TUNING_SEED_TAG, 3)
MEASUREMENT_SEEDS: tuple[int, ...] = derive_seeds(MEASUREMENT_SEED_TAG, 5)

SEED_SEPARATION_RULE = (
    "Tuning, measurement and split seeds come from THREE different tags, so the "
    "protocol is not selected on the same randomness the final numbers are "
    "reported on."
)

SHARED_ACROSS_PATHWAYS: tuple[str, ...] = (
    "split", "learning rate", "optimizer", "scheduler", "loss", "batch size",
    "epoch budget", "seed", "checkpoint criterion", "architecture", "max_length",
    "numerical precision",
)

PER_PATHWAY_FREEDOM = (
    "Each pathway trains its OWN head through its OWN clean pathway, and may "
    "select its OWN best epoch on protocol-dev under the SAME checkpoint rule. "
    "'Same protocol' does NOT require an identical epoch number -- requiring that "
    "would force one pathway to use a checkpoint its own dev curve did not choose."
)

PRIMARY_METRIC = "macro_f1"
SECONDARY_METRIC = "accuracy"
DIAGNOSTIC_METRICS: tuple[str, ...] = ("per_class_f1_negative", "per_class_f1_neutral", "per_class_f1_positive")

PAIRED_REPORTING_RULE = (
    "For each of the 5 measurement seeds, train a Vanilla head and a Base-only "
    "head, freeze each selected head, and evaluate on the untouched official "
    "VALIDATION split. Report Delta_s = Score_vanilla_s - Score_baseonly_s. "
    "Report all five raw paired scores, all five Delta_s, mean(Delta), sample "
    "std(Delta), and raw Vanilla and Base-only mean/std. Pairing removes "
    "seed-to-seed variance that would otherwise swamp the effect."
)

NO_SIGNIFICANCE_CLAIM = (
    "No p-value is required for n = 5, and none is invented. Do not make a "
    "significance claim from five paired observations."
)

NO_PREG1_THRESHOLD = (
    "There is NO pre-G1 PASS/FAIL threshold. The result is DESCRIPTIVE. Full G1's "
    "'within approximately 1 point' belongs to the fusion-attached measurement and "
    "is NOT borrowed here."
)

NOT_A_CEILING = (
    "Neither the primary nor the secondary result may be called an upper bound or "
    "ceiling on UNMARK. Base-only has no channels and no adapter; UNMARK adds both."
)


@dataclass(frozen=True)
class Preg1Protocol:
    """The whole precommitment, in one inspectable record."""

    version: str = PREG1_PROTOCOL_VERSION
    dataset: str = PRIMARY_DATASET
    dataset_version: str = PRIMARY_DATASET_VERSION
    task: str = PRIMARY_TASK
    num_labels: int = PRIMARY_NUM_LABELS
    pooling: Preg1Pooling = PREG1_POOLING
    max_length: int = MAX_LENGTH
    optimizer: str = OPTIMIZER
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr_grid: tuple[float, ...] = LR_GRID
    tuning_seeds: tuple[int, ...] = TUNING_SEEDS
    measurement_seeds: tuple[int, ...] = MEASUREMENT_SEEDS
    split_seed: int = SPLIT_SEED
    internal_split_fractions: dict[str, float] = field(
        default_factory=lambda: dict(INTERNAL_SPLIT_FRACTIONS)
    )

    @property
    def max_length_resolved(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scope": "pre-G1 clean-path burden diagnostic only; NOT full G1",
            "dataset": {
                "name": self.dataset,
                "version": self.dataset_version,
                "full_name": PRIMARY_DATASET_FULL_NAME,
                "task": self.task,
                "num_labels": self.num_labels,
                "label_mapping": dict(LABEL_MAPPING),
                "published_split_sizes": dict(PUBLISHED_SPLIT_SIZES),
                "published_label_counts": {
                    k: dict(v) for k, v in PUBLISHED_LABEL_COUNTS.items()
                },
                "selection_rationale": list(DATASET_SELECTION_RATIONALE),
                "caveat": DATASET_CAVEAT,
                "lock_rule": DATASET_LOCK_RULE,
                "superseded_dataset": SUPERSEDED_DATASET,
                "supersession_note": SUPERSESSION_NOTE,
            },
            "pathways": {
                "contract": PATHWAY_CONTRACT,
                "no_word_segmenter": NO_WORD_SEGMENTER,
                "raw_base_caveat": RAW_BASE_CAVEAT,
            },
            "pooling": self.pooling.value,
            "pooling_scope": PREG1_POOLING_SCOPE,
            "head": {
                "architecture": HEAD_ARCHITECTURE,
                "shape": HEAD_SHAPE,
                "hidden_layers": HEAD_HIDDEN_LAYERS,
                "dropout": HEAD_DROPOUT,
                "bias": HEAD_BIAS,
                "layer_norm": HEAD_LAYERNORM,
                "activation": HEAD_ACTIVATION,
                "hidden_size_source": HEAD_HIDDEN_SIZE_SOURCE,
                "init_weight": HEAD_INIT_WEIGHT,
                "init_bias": HEAD_INIT_BIAS,
                "init_is_explicit": HEAD_INIT_IS_EXPLICIT,
                "init_rule": HEAD_INIT_RULE,
            },
            "reproducibility": {
                "run_initialisation_sequence": list(RUN_INITIALISATION_SEQUENCE),
                "paired_initialisation": PAIRED_INITIALISATION_RULE,
                "paired_data_order": PAIRED_DATA_ORDER_RULE,
                "no_shared_advancing_rng": NO_SHARED_ADVANCING_RNG,
                "runtime_options_note": RUNTIME_OPTIONS_NOTE,
            },
            "loss": {
                "kind": LOSS,
                "spec": LOSS_SPEC,
                "class_weights": LOSS_CLASS_WEIGHTS,
                "label_smoothing": LOSS_LABEL_SMOOTHING,
                "reduction": LOSS_REDUCTION,
                "rationale": LOSS_RATIONALE,
            },
            "encoder": {
                "checkpoint": ENCODER_CHECKPOINT,
                "revision": ENCODER_REVISION,
                "frozen": ENCODER_FROZEN,
                "eval_mode": ENCODER_EVAL_MODE,
                "extraction": REPRESENTATION_EXTRACTION,
                "precision": PRECISION,
                "mixed_precision": MIXED_PRECISION,
                "pin_scope": ENCODER_PIN_SCOPE,
            },
            "max_length": self.max_length,
            "truncation": TRUNCATION,
            "padding": PADDING,
            "max_length_rationale": MAX_LENGTH_RATIONALE,
            "max_length_supersession": MAX_LENGTH_SUPERSESSION,
            "length_report_thresholds": list(LENGTH_REPORT_THRESHOLDS),
            "overflow_policy": OVERFLOW_POLICY,
            "splits": {
                "official_test_sealed": OFFICIAL_TEST_SEALED,
                "official_test_seal_rule": OFFICIAL_TEST_SEAL_RULE,
                "official_validation_role": OFFICIAL_VALIDATION_ROLE,
                "official_validation_rule": OFFICIAL_VALIDATION_RULE,
                "internal_split_fractions": dict(self.internal_split_fractions),
                "internal_split_rationale": INTERNAL_SPLIT_RATIONALE,
                "split_seed_tag": SPLIT_SEED_TAG,
                "split_seed": self.split_seed,
                "splitter_requirements": list(SPLITTER_REQUIREMENTS),
                "splitter_status": SPLITTER_STATUS,
                "duplicate_contract": list(DUPLICATE_CONTRACT),
                "conflicting_group_policy": CONFLICTING_GROUP_POLICY,
                "conflicting_group_exclusion_scope": CONFLICTING_GROUP_EXCLUSION_SCOPE,
                "observed_conflicting_groups": {
                    k: list(v) for k, v in OBSERVED_CONFLICTING_GROUPS.items()
                },
                "derived_train_size": DERIVED_TRAIN_SIZE,
                "derived_train_label_counts": dict(DERIVED_TRAIN_LABEL_COUNTS),
                "derived_train_csv_sha256": DERIVED_TRAIN_CSV_SHA256,
            },
            "optimisation": {
                "optimizer": self.optimizer,
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "weight_decay_weight": WEIGHT_DECAY_WEIGHT,
                "weight_decay_bias": WEIGHT_DECAY_BIAS,
                "param_group_rule": PARAM_GROUP_RULE,
                "amsgrad": AMSGRAD,
                "lr_schedule": LR_SCHEDULE,
                "warmup_steps": WARMUP_STEPS,
                "gradient_clipping": GRADIENT_CLIPPING,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping": EARLY_STOPPING,
                "shuffle": SHUFFLE_TRAINING_DATA,
                "shuffle_deterministic_under_seed": SHUFFLE_IS_DETERMINISTIC_UNDER_SEED,
                "encoder_gradient_or_update": ENCODER_GRADIENT_OR_UPDATE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "drop_last": DROP_LAST,
                "checkpoint_eligible_epochs": [
                    FIRST_CHECKPOINT_EPOCH, LAST_CHECKPOINT_EPOCH
                ],
                "checkpoint_eligibility_rule": CHECKPOINT_ELIGIBILITY_RULE,
            },
            "lr_search": {
                "grid": list(self.lr_grid),
                "checkpoint_rule": list(CHECKPOINT_RULE),
                "aggregation_rule": list(LR_AGGREGATION_RULE),
                "selection": PRIMARY_LR_SELECTION,
                "caveat": PRIMARY_LR_CAVEAT,
                "secondary_sensitivity": SECONDARY_SENSITIVITY,
            },
            "seeds": {
                "derivation_rule": SEED_DERIVATION_RULE,
                "separation_rule": SEED_SEPARATION_RULE,
                "tuning_tag": TUNING_SEED_TAG,
                "tuning": list(self.tuning_seeds),
                "measurement_tag": MEASUREMENT_SEED_TAG,
                "measurement": list(self.measurement_seeds),
                "split_tag": SPLIT_SEED_TAG,
                "split": self.split_seed,
            },
            "reporting": {
                "primary_metric": PRIMARY_METRIC,
                "secondary_metric": SECONDARY_METRIC,
                "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
                "shared_across_pathways": list(SHARED_ACROSS_PATHWAYS),
                "per_pathway_freedom": PER_PATHWAY_FREEDOM,
                "paired_rule": PAIRED_REPORTING_RULE,
                "no_significance_claim": NO_SIGNIFICANCE_CLAIM,
                "no_preg1_threshold": NO_PREG1_THRESHOLD,
                "not_a_ceiling": NOT_A_CEILING,
            },
        }
