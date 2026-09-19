"""Post-diagnostic PhoNER Stage-II optimization protocol.

This module is intentionally separate from the frozen Audit 076 linear
diagnostic. It defines a staged post-diagnostic development funnel after
corrupted PhoNER DEV results have already been observed. Nothing here may be
represented as a pre-observation protocol freeze.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from unmark.cross_task.phoner_transfer import (
    PHONER_CORRUPTION_SEED,
    PHONER_FINAL_SEEDS,
    PHONER_GATE_SHA256,
    PHONER_IGNORE_INDEX,
    PHONER_SCALE_SHA256,
    PHONER_TASK_ID,
    PhoNERContractViolation,
    PhoNERPathway,
    corruption_summary_metrics,
    mean_and_sample_sd,
)
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE, MAX_LENGTH
from unmark.viunmark.config import SIX_CONDITIONS
from unmark.viunmark.inputs import CorruptionProtocol

PHONER_STAGE2_OPT_NAMESPACE = "phoner_stage2_optimized"
PHONER_STAGE2_OPT_PROTOCOL_VERSION = "phoner-stage2-post-diagnostic-funnel-v2"
PHONER_STAGE2_OPT_EXPERIMENT_NAME = "PhoNER Stage-II Post-Diagnostic Optimization Funnel"
PHONER_STAGE2_OPT_HEAD_SCHEMA = "phoner-stage2-optimized-mlp-head-v2"
PHONER_STAGE2_OPT_BANK_SCHEMA = "phoner-stage2-representation-bank-v1"
PHONER_STAGE2_OPT_RESULT_SCHEMA = "phoner-stage2-funnel-result-v1"
PHONER_STAGE2_OPT_FINAL_SCHEMA = "phoner-stage2-final-freeze-v1"
PHONER_STAGE2_OPT_POST_DIAGNOSTIC = True
PHONER_STAGE2_OPT_TEST_ENABLED = False

MLP_ARCHITECTURE = "LayerNorm(768)->Linear(768,256)->GELU->Dropout(0.1)->Linear(256,num_labels)"
FULL_ONLY = ("FULL",)


class Stage2Stage(Enum):
    PROTOCOL = "protocol"
    BUILD_BANK = "build-bank"
    D1_TRAIN = "d1-train"
    D1_SELECT = "d1-select"
    D2_DECODE = "d2-decode"
    D3_TRAIN = "d3-train"
    D3_SELECT = "d3-select"
    D4_ANALYZE = "d4-analyze"
    SYS1_EVALUATE = "sys1-evaluate"
    SYS2_1_TRAIN = "sys2-1-train"
    SYS2_1_SELECT = "sys2-1-select"
    SYS2_2_EVALUATE = "sys2-2-evaluate"
    COMP_D1 = "comp-d1"
    COMP_D2 = "comp-d2"
    FREEZE_FINAL = "freeze-final"
    TEST_PREDICT = "test-predict"
    TEST_SCORE = "test-score"


OPTIMIZED_STAGE_ORDER: tuple[Stage2Stage, ...] = (
    Stage2Stage.PROTOCOL,
    Stage2Stage.BUILD_BANK,
    Stage2Stage.D1_TRAIN,
    Stage2Stage.D1_SELECT,
    Stage2Stage.D2_DECODE,
    Stage2Stage.D3_TRAIN,
    Stage2Stage.D3_SELECT,
    Stage2Stage.D4_ANALYZE,
    Stage2Stage.SYS1_EVALUATE,
    Stage2Stage.SYS2_1_TRAIN,
    Stage2Stage.SYS2_1_SELECT,
    Stage2Stage.SYS2_2_EVALUATE,
    Stage2Stage.COMP_D1,
    Stage2Stage.COMP_D2,
    Stage2Stage.FREEZE_FINAL,
)


class LossWeighting(Enum):
    UNWEIGHTED = "unweighted"
    SQRT_INVERSE_FREQUENCY = "sqrt_inverse_frequency"


class TrainingDistribution(Enum):
    FULL = "FULL"
    AUG6 = "AUG6"


class DecodePolicy(Enum):
    ARGMAX_IOB2_REPAIR = "argmax_iob2_repair"
    HARD_BIO_VITERBI = "hard_bio_viterbi"


class CandidateKind(Enum):
    READOUT = "readout"
    DECODER = "decoder"
    FUSION = "fusion"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class OptimizedTrainingPolicy:
    """Fixed training mechanics for post-diagnostic Stage-II head development."""

    source: str = (
        "POST-DIAGNOSTIC DEVELOPMENT: corrupted PhoNER DEV metrics were already "
        "observed in Audit 076. This funnel is a development campaign and must "
        "not claim pre-observation finalization."
    )
    architecture: str = MLP_ARCHITECTURE
    corruption_seed: int = PHONER_CORRUPTION_SEED
    optimizer: str = "AdamW"
    learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    complete_distribution_passes: int = 5
    selection_boundaries: int = 30
    scheduler: str = "none"
    warmup_updates: int = 0
    gradient_clipping: str = "none"
    amp: str = "disabled"
    tf32: str = "disabled"
    precision: str = "float32"
    dropout: float = 0.1
    hidden_dim: int = 256
    loss_weight_formula: str = (
        "For weighted CE, count TRAIN first-subtoken labels after word/subword "
        "alignment across the recipe's training conditions; raw_l = "
        "1/sqrt(count_l); weight_l = raw_l / mean(raw over labels). Every label "
        "in the TRAIN-derived inventory must have positive count. The exponent "
        "is fixed at 0.5 and is not tuned."
    )
    per_head_selection: tuple[str, ...] = (
        "All-6 entity micro-F1",
        "worst-condition entity micro-F1",
        "FULL entity micro-F1",
        "earlier checkpoint boundary",
    )
    recipe_selection: tuple[str, ...] = (
        "five-seed mean All-6 entity micro-F1",
        "five-seed mean worst-condition entity micro-F1",
        "five-seed mean FULL entity micro-F1",
        "simpler/fewer-branch system",
    )
    final_seeds: tuple[int, ...] = PHONER_FINAL_SEEDS

    def __post_init__(self) -> None:
        if self.architecture != MLP_ARCHITECTURE:
            raise PhoNERContractViolation("optimized Stage-II MLP architecture drifted")
        if self.corruption_seed != PHONER_CORRUPTION_SEED:
            raise PhoNERContractViolation("PhoNER corruption seed must remain 19225")
        if self.batch_size != 16:
            raise PhoNERContractViolation("optimized Stage-II keeps batch size 16")
        if self.complete_distribution_passes <= 0:
            raise PhoNERContractViolation("complete distribution pass budget must be positive")
        if self.selection_boundaries != 30:
            raise PhoNERContractViolation("optimized Stage-II exposes exactly 30 boundaries")
        if self.amp != "disabled" or self.tf32 != "disabled" or self.precision != "float32":
            raise PhoNERContractViolation("optimized Stage-II precision policy drifted")
        if self.final_seeds != PHONER_FINAL_SEEDS:
            raise PhoNERContractViolation("optimized Stage-II seed set drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "architecture": self.architecture,
            "corruption_seed": self.corruption_seed,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "complete_distribution_passes": self.complete_distribution_passes,
            "selection_boundaries": self.selection_boundaries,
            "scheduler": self.scheduler,
            "warmup_updates": self.warmup_updates,
            "gradient_clipping": self.gradient_clipping,
            "amp": self.amp,
            "tf32": self.tf32,
            "precision": self.precision,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "loss_weight_formula": self.loss_weight_formula,
            "per_head_selection": list(self.per_head_selection),
            "recipe_selection": list(self.recipe_selection),
            "final_seeds": list(self.final_seeds),
        }


@dataclass(frozen=True)
class DerivedTrainingBudget:
    distribution: TrainingDistribution
    train_chunk_count: int
    train_conditions: tuple[str, ...]
    examples_per_pass: int
    batch_size: int
    complete_distribution_passes: int
    updates_per_complete_aug6_pass: int
    max_optimizer_updates: int
    selection_boundaries: int
    boundary_updates: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution.value,
            "train_chunk_count": self.train_chunk_count,
            "train_conditions": list(self.train_conditions),
            "examples_per_pass": self.examples_per_pass,
            "batch_size": self.batch_size,
            "complete_distribution_passes": self.complete_distribution_passes,
            "updates_per_complete_aug6_pass": self.updates_per_complete_aug6_pass,
            "max_optimizer_updates": self.max_optimizer_updates,
            "selection_boundaries": self.selection_boundaries,
            "boundary_updates": list(self.boundary_updates),
        }


def distribution_for_conditions(train_conditions: Sequence[str]) -> TrainingDistribution:
    conditions = tuple(train_conditions)
    if conditions == FULL_ONLY:
        return TrainingDistribution.FULL
    if conditions == SIX_CONDITIONS:
        return TrainingDistribution.AUG6
    raise PhoNERContractViolation(f"unsupported training distribution conditions: {conditions}")


def derive_training_budget(
    train_chunk_count: int,
    train_conditions: Sequence[str],
    policy: OptimizedTrainingPolicy | None = None,
) -> DerivedTrainingBudget:
    """Derive update budget from five complete passes over a candidate's distribution."""

    active_policy = policy or OptimizedTrainingPolicy()
    if train_chunk_count <= 0:
        raise PhoNERContractViolation("TRAIN chunk count must be positive")
    distribution = distribution_for_conditions(train_conditions)
    conditions = tuple(train_conditions)
    examples_per_pass = train_chunk_count * len(conditions)
    updates_per_pass = math.ceil(examples_per_pass / active_policy.batch_size)
    max_updates = updates_per_pass * active_policy.complete_distribution_passes
    if max_updates < active_policy.selection_boundaries:
        raise PhoNERContractViolation("derived update budget cannot support 30 evaluation boundaries")
    boundaries = tuple(
        max(1, math.ceil(max_updates * index / active_policy.selection_boundaries))
        for index in range(1, active_policy.selection_boundaries + 1)
    )
    if len(set(boundaries)) != active_policy.selection_boundaries:
        raise PhoNERContractViolation("derived evaluation boundaries are not unique")
    if boundaries[-1] != max_updates:
        raise PhoNERContractViolation("final boundary must equal max optimizer updates")
    return DerivedTrainingBudget(
        distribution=distribution,
        train_chunk_count=train_chunk_count,
        train_conditions=conditions,
        examples_per_pass=examples_per_pass,
        batch_size=active_policy.batch_size,
        complete_distribution_passes=active_policy.complete_distribution_passes,
        updates_per_complete_aug6_pass=updates_per_pass,
        max_optimizer_updates=max_updates,
        selection_boundaries=active_policy.selection_boundaries,
        boundary_updates=boundaries,
    )


@dataclass(frozen=True)
class ReadoutRecipe:
    recipe_id: str
    stage: Stage2Stage
    pathway: PhoNERPathway
    train_conditions: tuple[str, ...]
    loss_weighting: LossWeighting
    architecture: str = MLP_ARCHITECTURE

    def __post_init__(self) -> None:
        if self.train_conditions not in (FULL_ONLY, SIX_CONDITIONS):
            raise PhoNERContractViolation(f"unsupported train condition set for {self.recipe_id}")
        if self.stage in {Stage2Stage.D1_TRAIN, Stage2Stage.D1_SELECT} and self.pathway is not PhoNERPathway.VIUNMARK_SCALE:
            raise PhoNERContractViolation("D1-NER preflight must use VIUNMARK_SCALE")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage": self.stage.value,
            "pathway": self.pathway.value,
            "train_conditions": list(self.train_conditions),
            "loss_weighting": self.loss_weighting.value,
            "architecture": self.architecture,
        }

    @property
    def training_distribution(self) -> TrainingDistribution:
        return distribution_for_conditions(self.train_conditions)


@dataclass(frozen=True)
class FusionCandidate:
    candidate_id: str
    stage: Stage2Stage
    branches: tuple[str, ...]
    weights: tuple[float, ...]
    branch_head_count: int = 5
    decode_once_after_fusion: bool = True

    def __post_init__(self) -> None:
        if len(self.branches) != len(self.weights):
            raise PhoNERContractViolation("fusion branch/weight count mismatch")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise PhoNERContractViolation(f"{self.candidate_id} weights must sum to 1")
        if self.branch_head_count != 5:
            raise PhoNERContractViolation("each branch must average exactly five heads")
        if not self.decode_once_after_fusion:
            raise PhoNERContractViolation("NER fusion must decode exactly once after raw-emission fusion")

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage.value,
            "branches": list(self.branches),
            "weights": list(self.weights),
            "branch_head_count": self.branch_head_count,
            "decode_once_after_fusion": self.decode_once_after_fusion,
        }


def d1_candidate_family() -> tuple[ReadoutRecipe, ...]:
    scale = PhoNERPathway.VIUNMARK_SCALE
    return (
        ReadoutRecipe("D1-A-NER", Stage2Stage.D1_TRAIN, scale, FULL_ONLY, LossWeighting.UNWEIGHTED),
        ReadoutRecipe("D1-B-NER", Stage2Stage.D1_TRAIN, scale, SIX_CONDITIONS, LossWeighting.UNWEIGHTED),
        ReadoutRecipe("D1-C-NER", Stage2Stage.D1_TRAIN, scale, SIX_CONDITIONS, LossWeighting.SQRT_INVERSE_FREQUENCY),
    )


def d2_candidate_family() -> tuple[DecodePolicy, ...]:
    return (DecodePolicy.ARGMAX_IOB2_REPAIR, DecodePolicy.HARD_BIO_VITERBI)


def matched_d3_recipes(selected_d1: ReadoutRecipe) -> tuple[ReadoutRecipe, ...]:
    return tuple(
        ReadoutRecipe(
            recipe_id=f"D3-NER-{pathway.value}",
            stage=Stage2Stage.D3_TRAIN,
            pathway=pathway,
            train_conditions=selected_d1.train_conditions,
            loss_weighting=selected_d1.loss_weighting,
        )
        for pathway in (
            PhoNERPathway.PHOBERT_NATIVE,
            PhoNERPathway.VIUNMARK_GATE,
            PhoNERPathway.VIUNMARK_SCALE,
        )
    )


def sys1_beta_grid() -> tuple[FusionCandidate, ...]:
    return (
        FusionCandidate("SYS1-NER-GATE_SCALE-075-025", Stage2Stage.SYS1_EVALUATE, ("GATE_RAW5", "SCALE_RAW5"), (0.75, 0.25)),
        FusionCandidate("SYS1-NER-GATE_SCALE-050-050", Stage2Stage.SYS1_EVALUATE, ("GATE_RAW5", "SCALE_RAW5"), (0.50, 0.50)),
        FusionCandidate("SYS1-NER-GATE_SCALE-025-075", Stage2Stage.SYS1_EVALUATE, ("GATE_RAW5", "SCALE_RAW5"), (0.25, 0.75)),
    )


def sys2_2_gamma_grid() -> tuple[FusionCandidate, ...]:
    return (
        FusionCandidate("SYS2-2-NER-NATIVE_ADAPTED-075-025", Stage2Stage.SYS2_2_EVALUATE, ("NATIVE_RAW5", "ADAPTED_SYS1_RAW"), (0.75, 0.25)),
        FusionCandidate("SYS2-2-NER-NATIVE_ADAPTED-050-050", Stage2Stage.SYS2_2_EVALUATE, ("NATIVE_RAW5", "ADAPTED_SYS1_RAW"), (0.50, 0.50)),
        FusionCandidate("SYS2-2-NER-NATIVE_ADAPTED-025-075", Stage2Stage.SYS2_2_EVALUATE, ("NATIVE_RAW5", "ADAPTED_SYS1_RAW"), (0.25, 0.75)),
    )


def sys2_1_native_candidates(*, d1_weighted_survives: bool) -> tuple[ReadoutRecipe, ...]:
    base = [
        ReadoutRecipe("SYS2-1-NER-NATIVE-FULL-UNWEIGHTED", Stage2Stage.SYS2_1_TRAIN, PhoNERPathway.PHOBERT_NATIVE, FULL_ONLY, LossWeighting.UNWEIGHTED),
        ReadoutRecipe("SYS2-1-NER-NATIVE-AUG6-UNWEIGHTED", Stage2Stage.SYS2_1_TRAIN, PhoNERPathway.PHOBERT_NATIVE, SIX_CONDITIONS, LossWeighting.UNWEIGHTED),
    ]
    if d1_weighted_survives:
        base.append(
            ReadoutRecipe(
                "SYS2-1-NER-NATIVE-AUG6-WEIGHTED",
                Stage2Stage.SYS2_1_TRAIN,
                PhoNERPathway.PHOBERT_NATIVE,
                SIX_CONDITIONS,
                LossWeighting.SQRT_INVERSE_FREQUENCY,
            )
        )
    return tuple(base)


def first_subtoken_label_weights(label_ids: Sequence[int], *, num_labels: int) -> tuple[float, ...]:
    counts = [0 for _ in range(num_labels)]
    for label_id in label_ids:
        if label_id == PHONER_IGNORE_INDEX:
            continue
        if label_id < 0 or label_id >= num_labels:
            raise PhoNERContractViolation(f"label id {label_id} outside inventory")
        counts[label_id] += 1
    if any(count <= 0 for count in counts):
        raise PhoNERContractViolation("weighted CE requires every TRAIN label to have positive count")
    raw = [1.0 / math.sqrt(count) for count in counts]
    mean_raw = sum(raw) / len(raw)
    return tuple(value / mean_raw for value in raw)


def _label_parts(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    if "-" not in label:
        raise PhoNERContractViolation(f"invalid BIO label {label!r}")
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"} or not entity_type:
        raise PhoNERContractViolation(f"invalid BIO label {label!r}")
    return prefix, entity_type


def legal_bio_transition(previous: str | None, current: str) -> bool:
    current_prefix, current_type = _label_parts(current)
    if current_prefix in {"O", "B"}:
        return True
    if previous is None:
        return False
    previous_prefix, previous_type = _label_parts(previous)
    return previous_prefix in {"B", "I"} and previous_type == current_type


def hard_bio_constrained_viterbi(logits: Sequence[Sequence[float]], id_to_label: Mapping[int, str]) -> tuple[int, ...]:
    """Decode logits with deterministic hard BIO constraints and no learned transitions."""

    if not logits:
        return ()
    label_ids = tuple(sorted(int(index) for index in id_to_label))
    if not label_ids:
        raise PhoNERContractViolation("decoder requires a non-empty label inventory")
    dp: list[dict[int, tuple[float, tuple[int, ...]]]] = []
    for position, scores in enumerate(logits):
        if len(scores) <= max(label_ids):
            raise PhoNERContractViolation("logit width is smaller than label inventory")
        row: dict[int, tuple[float, tuple[int, ...]]] = {}
        for label_id in label_ids:
            label = id_to_label[label_id]
            best_score = -math.inf
            best_path: tuple[int, ...] | None = None
            if position == 0:
                if legal_bio_transition(None, label):
                    best_score = float(scores[label_id])
                    best_path = (label_id,)
            else:
                for previous_id, (previous_score, previous_path) in dp[-1].items():
                    if legal_bio_transition(id_to_label[previous_id], label):
                        score = previous_score + float(scores[label_id])
                        if score > best_score:
                            best_score = score
                            best_path = previous_path + (label_id,)
            if best_path is not None:
                row[label_id] = (best_score, best_path)
        if not row:
            raise PhoNERContractViolation(f"no legal BIO path at position {position}")
        dp.append(row)
    return max(dp[-1].values(), key=lambda item: item[0])[1]


def is_valid_bio_sequence(labels: Sequence[str]) -> bool:
    previous: str | None = None
    for label in labels:
        if not legal_bio_transition(previous, label):
            return False
        previous = label
    return True


def d4_entity_key(sample_id: str, word_start: int, word_end: int, entity_type: str) -> tuple[str, int, int, str]:
    if not sample_id:
        raise PhoNERContractViolation("D4 entity key requires sample_id")
    if word_start < 0 or word_end <= word_start:
        raise PhoNERContractViolation("D4 entity key requires a valid word span")
    if not entity_type:
        raise PhoNERContractViolation("D4 entity key requires entity type")
    return (sample_id, word_start, word_end, entity_type)


def arithmetic_fuse_logits(weighted_logits: Sequence[tuple[float, Any]]) -> Any:
    if not weighted_logits:
        raise PhoNERContractViolation("cannot fuse no logits")
    total = None
    for weight, logits in weighted_logits:
        piece = logits * float(weight)
        total = piece if total is None else total + piece
    return total


def mean_five_head_logits(logits_by_seed: Mapping[int, Any]) -> Any:
    if tuple(logits_by_seed) != PHONER_FINAL_SEEDS:
        raise PhoNERContractViolation("branch emission must average exactly the five fixed seeds in order")
    return arithmetic_fuse_logits(tuple((1.0 / 5.0, logits_by_seed[seed]) for seed in PHONER_FINAL_SEEDS))


def condition_f1_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition in values:
            raise PhoNERContractViolation(f"duplicate condition row: {condition}")
        values[condition] = float(row["entity_micro_f1"])
    missing = [condition for condition in SIX_CONDITIONS if condition not in values]
    if missing:
        raise PhoNERContractViolation(f"missing condition rows: {missing}")
    return values


def per_head_selection_tuple(rows: Sequence[Mapping[str, Any]], *, boundary: int | None = None) -> tuple[float, float, float, int]:
    values = condition_f1_values(rows)
    all6 = sum(values[condition] for condition in SIX_CONDITIONS) / len(SIX_CONDITIONS)
    worst = min(values.values())
    full = values["FULL"]
    boundary_key = -(boundary if boundary is not None else 0)
    return (all6, worst, full, boundary_key)


def aggregate_selection_tuple(summary: Mapping[str, Any], *, branch_count: int) -> tuple[float, float, float, int]:
    return (
        float(summary["all_6_f1_mean"]),
        float(summary["worst_condition_f1_mean"]),
        float(summary["FULL_f1_mean"]),
        -branch_count,
    )


def ensemble_fusion_selection_tuple(rows: Sequence[Mapping[str, Any]], *, default_rank: int = 0) -> tuple[float, float, float, int]:
    """Rank deployed fusion candidates from one ensemble metric row per condition.

    Fusion weights are selected from the actual deployed five-head branch
    ensemble outputs, not by averaging five independent per-seed F1 values.
    """

    for row in rows:
        if "seed" in row:
            raise PhoNERContractViolation("fusion selection must use deployed ensemble rows, not per-seed rows")
    values = condition_f1_values(rows)
    all6 = sum(values[condition] for condition in SIX_CONDITIONS) / len(SIX_CONDITIONS)
    return (all6, min(values.values()), values["FULL"], default_rank)


def summarize_optimized_candidate_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    by_candidate_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        candidate = str(row["candidate_id"])
        condition = str(row["condition"])
        seed = int(row["seed"])
        grouped.setdefault((candidate, condition), []).append(row)
        by_candidate_seed.setdefault((candidate, seed), []).append(row)

    condition_summary: dict[str, Any] = {}
    for (candidate, condition), group in sorted(grouped.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in group))
        if seeds != tuple(sorted(PHONER_FINAL_SEEDS)):
            raise PhoNERContractViolation(f"{candidate}/{condition} does not cover five seeds")
        f1_mean, f1_sd = mean_and_sample_sd([float(row["entity_micro_f1"]) for row in group])
        condition_summary[f"{candidate}:{condition}"] = {
            "candidate_id": candidate,
            "condition": condition,
            "seeds": list(seeds),
            "entity_micro_f1_mean": f1_mean,
            "entity_micro_f1_sample_sd": f1_sd,
        }

    robustness_rows = []
    for (candidate, seed), group in sorted(by_candidate_seed.items()):
        values = condition_f1_values(group)
        robustness_rows.append({
            "candidate_id": candidate,
            "seed": seed,
            **corruption_summary_metrics(values),
            "worst_condition_f1": min(values.values()),
        })

    robustness_summary: dict[str, Any] = {}
    for candidate in sorted({row["candidate_id"] for row in robustness_rows}):
        group = [row for row in robustness_rows if row["candidate_id"] == candidate]
        seeds = tuple(sorted(int(row["seed"]) for row in group))
        if seeds != tuple(sorted(PHONER_FINAL_SEEDS)):
            raise PhoNERContractViolation(f"{candidate} summary does not cover five seeds")
        metrics: dict[str, float] = {}
        keys = [
            "corrupt_avg_f1",
            "all_6_f1",
            "full_to_strip_absolute_drop",
            "worst_condition_f1",
            *(f"{condition}_robustness_retention" for condition in SIX_CONDITIONS),
        ]
        for key in keys:
            mean, sd = mean_and_sample_sd([float(row[key]) for row in group])
            metrics[f"{key}_mean"] = mean
            metrics[f"{key}_sample_sd"] = sd
        robustness_summary[candidate] = {
            "candidate_id": candidate,
            "seeds": list(seeds),
            **metrics,
            "per_seed": group,
        }
    return {
        "primary_policy": (
            "five-seed aggregate selection; no best-seed selection; corrupt average "
            "and All-6 are computed per seed before summary"
        ),
        "conditions": condition_summary,
        "robustness": robustness_summary,
    }


def stable_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RepresentationBankIdentity:
    split: str
    dataset_sha256: str
    sample_ids_sha256: str
    pathway: PhoNERPathway
    condition: str
    stage1_checkpoint_sha256: str | None
    encoder_checkpoint: str = ENCODER_CHECKPOINT
    encoder_revision: str = ENCODER_REVISION
    tokenizer_max_length: int = MAX_LENGTH
    corruption_seed: int = PHONER_CORRUPTION_SEED
    schema_version: str = PHONER_STAGE2_OPT_BANK_SCHEMA
    dtype: str = "float32"
    includes_test: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev"}:
            raise PhoNERContractViolation("optimized representation banks may contain TRAIN/DEV only")
        if self.includes_test:
            raise PhoNERContractViolation("representation bank must not include TEST")
        if self.condition not in SIX_CONDITIONS:
            raise PhoNERContractViolation(f"unknown corruption condition {self.condition!r}")
        if self.pathway is PhoNERPathway.VIUNMARK_GATE and self.stage1_checkpoint_sha256 != PHONER_GATE_SHA256:
            raise PhoNERContractViolation("Gate representation bank Stage-I SHA mismatch")
        if self.pathway is PhoNERPathway.VIUNMARK_SCALE and self.stage1_checkpoint_sha256 != PHONER_SCALE_SHA256:
            raise PhoNERContractViolation("Scale representation bank Stage-I SHA mismatch")
        if self.pathway is PhoNERPathway.PHOBERT_NATIVE and self.stage1_checkpoint_sha256 is not None:
            raise PhoNERContractViolation("native PhoBERT bank must not bind a Stage-I checkpoint")
        if self.dtype != "float32":
            raise PhoNERContractViolation("representation bank dtype must remain float32 unless separately proved equivalent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "dataset_sha256": self.dataset_sha256,
            "sample_ids_sha256": self.sample_ids_sha256,
            "pathway": self.pathway.value,
            "condition": self.condition,
            "stage1_checkpoint_sha256": self.stage1_checkpoint_sha256,
            "encoder_checkpoint": self.encoder_checkpoint,
            "encoder_revision": self.encoder_revision,
            "tokenizer_max_length": self.tokenizer_max_length,
            "corruption_seed": self.corruption_seed,
            "schema_version": self.schema_version,
            "dtype": self.dtype,
            "includes_test": self.includes_test,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True)
class ScientificHeadIdentity:
    pathway: PhoNERPathway
    representation_identity_digest: str
    representation_bank_digest: str
    head_architecture: str
    training_distribution: TrainingDistribution
    train_conditions: tuple[str, ...]
    loss_definition: str
    loss_weights_digest: str | None
    optimizer_hyperparameters_digest: str
    training_budget_digest: str
    seed: int
    checkpoint_selection_semantics: tuple[str, ...]
    schema_version: str = "phoner-stage2-scientific-head-identity-v1"

    def __post_init__(self) -> None:
        if self.seed not in PHONER_FINAL_SEEDS:
            raise PhoNERContractViolation("head identity seed is outside the fixed five-seed set")
        if self.head_architecture != MLP_ARCHITECTURE:
            raise PhoNERContractViolation("head identity architecture drifted")
        if self.training_distribution != distribution_for_conditions(self.train_conditions):
            raise PhoNERContractViolation("head identity distribution does not match train conditions")
        if not self.representation_identity_digest or not self.representation_bank_digest:
            raise PhoNERContractViolation("head identity requires representation digests")
        if not self.optimizer_hyperparameters_digest or not self.training_budget_digest:
            raise PhoNERContractViolation("head identity requires optimizer and budget digests")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pathway": self.pathway.value,
            "representation_identity_digest": self.representation_identity_digest,
            "representation_bank_digest": self.representation_bank_digest,
            "head_architecture": self.head_architecture,
            "training_distribution": self.training_distribution.value,
            "train_conditions": list(self.train_conditions),
            "loss_definition": self.loss_definition,
            "loss_weights_digest": self.loss_weights_digest,
            "optimizer_hyperparameters_digest": self.optimizer_hyperparameters_digest,
            "training_budget_digest": self.training_budget_digest,
            "seed": self.seed,
            "checkpoint_selection_semantics": list(self.checkpoint_selection_semantics),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


def require_exact_head_reuse(requested: ScientificHeadIdentity, closed_artifact: Mapping[str, Any]) -> str:
    observed_identity = closed_artifact.get("scientific_head_identity")
    checkpoint_sha256 = closed_artifact.get("checkpoint_sha256")
    if not isinstance(observed_identity, Mapping) or not isinstance(checkpoint_sha256, str):
        raise PhoNERContractViolation("closed head artifact lacks identity or checkpoint SHA")
    if requested.to_dict() != dict(observed_identity):
        raise PhoNERContractViolation("requested head identity does not match closed artifact")
    if closed_artifact.get("scientific_head_identity_digest") != requested.digest:
        raise PhoNERContractViolation("closed head identity digest mismatch")
    return checkpoint_sha256


def sample_ids_digest(sample_ids: Sequence[str]) -> str:
    return stable_digest({"sample_ids": list(sample_ids)})


def verify_representation_bank_identity(expected: RepresentationBankIdentity, observed: Mapping[str, Any]) -> None:
    if expected.to_dict() != dict(observed):
        raise PhoNERContractViolation("representation bank provenance mismatch")


@dataclass(frozen=True)
class OptimizedPhoNERStage2Config:
    dataset_id: str = PHONER_TASK_ID
    experiment_name: str = PHONER_STAGE2_OPT_EXPERIMENT_NAME
    namespace: str = PHONER_STAGE2_OPT_NAMESPACE
    protocol_version: str = PHONER_STAGE2_OPT_PROTOCOL_VERSION
    post_diagnostic_development: bool = PHONER_STAGE2_OPT_POST_DIAGNOSTIC
    test_enabled: bool = PHONER_STAGE2_OPT_TEST_ENABLED
    encoder_checkpoint: str = ENCODER_CHECKPOINT
    encoder_revision: str = ENCODER_REVISION
    gate_checkpoint_sha256: str = PHONER_GATE_SHA256
    scale_checkpoint_sha256: str = PHONER_SCALE_SHA256
    hidden_size: int = HIDDEN_SIZE
    max_length: int = MAX_LENGTH
    conditions: tuple[str, ...] = SIX_CONDITIONS
    pathways: tuple[PhoNERPathway, ...] = (
        PhoNERPathway.PHOBERT_NATIVE,
        PhoNERPathway.VIUNMARK_GATE,
        PhoNERPathway.VIUNMARK_SCALE,
    )
    policy: OptimizedTrainingPolicy = field(default_factory=OptimizedTrainingPolicy)

    def __post_init__(self) -> None:
        if self.post_diagnostic_development is not True or self.test_enabled is not False:
            raise PhoNERContractViolation("optimized Stage-II must remain post-diagnostic with TEST disabled")
        if self.encoder_checkpoint != ENCODER_CHECKPOINT or self.encoder_revision != ENCODER_REVISION:
            raise PhoNERContractViolation("PhoBERT identity drifted")
        if self.gate_checkpoint_sha256 != PHONER_GATE_SHA256 or self.scale_checkpoint_sha256 != PHONER_SCALE_SHA256:
            raise PhoNERContractViolation("Stage-I checkpoint SHA identity drifted")
        if self.max_length != MAX_LENGTH or self.hidden_size != HIDDEN_SIZE:
            raise PhoNERContractViolation("PhoNER alignment/model dimensions drifted")
        if self.conditions != SIX_CONDITIONS:
            raise PhoNERContractViolation("condition order drifted")

    @property
    def corruption_protocol(self) -> CorruptionProtocol:
        return CorruptionProtocol(scientific_corruption_seed=self.policy.corruption_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "experiment_name": self.experiment_name,
            "namespace": self.namespace,
            "protocol_version": self.protocol_version,
            "post_diagnostic_development": self.post_diagnostic_development,
            "test_enabled": self.test_enabled,
            "encoder_checkpoint": self.encoder_checkpoint,
            "encoder_revision": self.encoder_revision,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
            "scale_checkpoint_sha256": self.scale_checkpoint_sha256,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "conditions": list(self.conditions),
            "pathways": [pathway.value for pathway in self.pathways],
            "policy": self.policy.to_dict(),
            "stage_order": [stage.value for stage in OPTIMIZED_STAGE_ORDER],
            "d1_candidates": [recipe.to_dict() for recipe in d1_candidate_family()],
            "d2_decoders": [policy.value for policy in d2_candidate_family()],
            "sys1_beta_grid": [candidate.to_dict() for candidate in sys1_beta_grid()],
            "sys2_2_gamma_grid": [candidate.to_dict() for candidate in sys2_2_gamma_grid()],
            "fusion_selection_policy": (
                "SYS1 beta and SYS2-2 gamma are selected on actual deployed branch ensembles: "
                "five-head branch logits are averaged first, candidate weights fuse raw tensors, "
                "one selected decoder is applied, and DEV entity metrics are computed on that ensemble."
            ),
            "negative_controls": {
                "crf_used": False,
                "sentiment_bias_or_calibration_used": False,
                "best_seed_selection_used": False,
                "test_enabled": False,
            },
        }


def build_optimized_mlp_head(num_labels: int, *, seed: int | None = None) -> Any:
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    gelu_class = getattr(torch.nn, "GELU")
    return torch.nn.Sequential(
        torch.nn.LayerNorm(HIDDEN_SIZE),
        torch.nn.Linear(HIDDEN_SIZE, 256),
        gelu_class(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, num_labels),
    )


__all__ = [
    "FULL_ONLY",
    "MLP_ARCHITECTURE",
    "OPTIMIZED_STAGE_ORDER",
    "PHONER_STAGE2_OPT_BANK_SCHEMA",
    "PHONER_STAGE2_OPT_EXPERIMENT_NAME",
    "PHONER_STAGE2_OPT_FINAL_SCHEMA",
    "PHONER_STAGE2_OPT_HEAD_SCHEMA",
    "PHONER_STAGE2_OPT_NAMESPACE",
    "PHONER_STAGE2_OPT_POST_DIAGNOSTIC",
    "PHONER_STAGE2_OPT_PROTOCOL_VERSION",
    "PHONER_STAGE2_OPT_RESULT_SCHEMA",
    "PHONER_STAGE2_OPT_TEST_ENABLED",
    "CandidateKind",
    "DecodePolicy",
    "DerivedTrainingBudget",
    "FusionCandidate",
    "LossWeighting",
    "OptimizedPhoNERStage2Config",
    "OptimizedTrainingPolicy",
    "ReadoutRecipe",
    "RepresentationBankIdentity",
    "ScientificHeadIdentity",
    "Stage2Stage",
    "TrainingDistribution",
    "aggregate_selection_tuple",
    "arithmetic_fuse_logits",
    "build_optimized_mlp_head",
    "condition_f1_values",
    "d1_candidate_family",
    "d2_candidate_family",
    "d4_entity_key",
    "derive_training_budget",
    "distribution_for_conditions",
    "ensemble_fusion_selection_tuple",
    "first_subtoken_label_weights",
    "hard_bio_constrained_viterbi",
    "is_valid_bio_sequence",
    "legal_bio_transition",
    "matched_d3_recipes",
    "mean_five_head_logits",
    "per_head_selection_tuple",
    "require_exact_head_reuse",
    "sample_ids_digest",
    "stable_digest",
    "summarize_optimized_candidate_scores",
    "sys1_beta_grid",
    "sys2_1_native_candidates",
    "sys2_2_gamma_grid",
    "verify_representation_bank_identity",
]
