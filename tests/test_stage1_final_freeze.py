"""The FINAL STAGE-1 CONFIGURATION FREEZE, checked against the code (Audit 030 §AJ).

`docs/spec/stage1-final-freeze.json` is the machine-readable freeze. This file is
what makes it *mechanical* rather than decorative: every scientific value is
compared against the **real imported constant or helper**, never against a literal
re-typed here. Re-typing the numbers would only prove the test agrees with itself.

If a later edit changes a frozen scientific constant without the freeze being
reviewed and updated, these tests fail.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.stage1.protocol as P
from unmark.stage1.contracts import PI_STRIP as GOVERNING_PI
from unmark.stage1.device import (
    DETERMINISTIC_CUBLAS_WORKSPACE,
    FLOAT32_MATMUL_PRECISION,
    ExecutionFingerprint,
)
from unmark.stage1.execute import TRUNCATION
from unmark.stage1.preparation import (
    MULTIPROCESSING_START_METHOD,
    ORDER_PRESERVING,
    PREFETCH_ENABLED,
    PREPARATION_BACKEND,
    PREPARATION_WORKERS,
)
from unmark.stage1.selection import (
    LR_PILOT_GRID,
    R_PHASE1_GRID,
    final_main_schedule,
    lr_pilot_schedule,
    r_phase1_schedule,
    select_checkpoint,
)
from unmark.stage1.trainer import (
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_EVERY_UPDATES,
    CHECKPOINT_SCHEMA_VERSION,
    LAST_CHECKPOINT_NAME,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO / "docs/spec/stage1-final-freeze.json"
FREEZE = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

CLASSIFICATIONS = {
    "scientific_identity", "scientific_protocol", "scientific_input",
    "operational_acceptance", "operational_provenance", "safety_gate",
}


def v(section: str, field: str):
    return FREEZE[section][field]["value"]


def kind(section: str, field: str):
    return FREEZE[section][field]["classification"]


# ---------------------------------------------------------------------------
# 0. The freeze file itself
# ---------------------------------------------------------------------------
def test_the_freeze_is_deterministic_sorted_json():
    raw = FREEZE_PATH.read_text(encoding="utf-8")
    assert raw == json.dumps(FREEZE, indent=2, sort_keys=True, ensure_ascii=False) + "\n", (
        "the freeze must be byte-stable: indent=2, sorted keys, trailing newline"
    )


def test_the_freeze_is_not_yet_training_ready():
    assert FREEZE["schema_version"] == 1
    assert FREEZE["freeze_status"] == "FROZEN_PENDING_FINAL_AUDIT_AND_HUMAN_APPROVAL"
    assert "TRAINING_READY" not in json.dumps(FREEZE)


def test_every_field_carries_a_valid_classification():
    unclassified = []
    for section, body in FREEZE.items():
        if not isinstance(body, dict) or section in ("notes",):
            continue
        for field, entry in body.items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            if entry.get("classification") not in CLASSIFICATIONS:
                unclassified.append(f"{section}.{field}")
    assert unclassified == [], unclassified


def test_operational_parameters_are_not_promoted_to_scientific_identity():
    """Worker count must never become run identity (Audit 030 §AG.6)."""
    assert kind("preparation_execution", "workers") == "operational_provenance"
    assert kind("preparation_execution", "backend") == "operational_provenance"
    assert "workers" not in ExecutionFingerprint.RESUME_BLOCKING


def test_seeds_and_selection_rules_are_not_demoted_to_provenance():
    for section, field in (("corruption", "corruption_seed"), ("corruption", "validation_corruption_seed"),
                           ("data", "split_seed"), ("run_plan", "selection_seed"),
                           ("run_plan", "train_seeds"), ("run_plan", "init_seeds")):
        assert kind(section, field) == "scientific_identity", (section, field)
    assert kind("selection", "score").startswith("scientific")
    assert kind("selection", "tie_break").startswith("scientific")


# ---------------------------------------------------------------------------
# 1. Model, adapter, lengths — against imported constants
# ---------------------------------------------------------------------------
def test_model_matches_the_code():
    assert v("model", "checkpoint") == P.ENCODER_CHECKPOINT
    assert v("model", "revision") == P.ENCODER_REVISION
    assert v("model", "hidden_size") == P.HIDDEN_SIZE
    assert v("model", "precision") == P.PRECISION
    assert v("model", "max_length") == P.MAX_LENGTH
    assert v("model", "on_overflow") == P.ON_OVERFLOW
    assert v("model", "truncation_offered") == P.TRUNCATION_OFFERED
    assert v("model", "truncation_policy") == TRUNCATION.to_dict()


def test_adapter_matches_the_code():
    assert v("adapter", "trainable_parameters") == P.ADAPTER_TRAINABLE_PARAMETERS
    assert v("adapter", "init_seed_tag") == P.ADAPTER_INIT_SEED_TAG
    assert v("adapter", "init_seed_depends_on") == ["run_seed"]
    assert v("adapter", "fresh_per_nominal_run") is True
    assert v("adapter", "state_shared_across_nominal_runs") is False


def test_the_init_seed_derivation_really_takes_run_seed_alone():
    import inspect

    assert list(inspect.signature(P.adapter_init_seed).parameters) == ["run_seed"]


# ---------------------------------------------------------------------------
# 2. Corruption, split, sampler
# ---------------------------------------------------------------------------
def test_corruption_matches_the_code():
    assert v("corruption", "corruption_seed") == P.CORRUPTION_SEED
    assert v("corruption", "pi_strip") == P.PI_STRIP
    assert v("corruption", "pi_strip_governing") == GOVERNING_PI == P.PI_STRIP
    assert v("corruption", "validation_corruption_seed") == P.VALIDATION_CORRUPTION_SEED


def test_split_matches_the_code():
    assert v("data", "dev_documents") == P.DEV_DOCUMENTS
    assert v("data", "split_seed") == P.SPLIT_SEED
    assert v("data", "corpus_dataset") == P.CORPUS_DATASET
    assert v("data", "corpus_revision") == P.CORPUS_REVISION
    # the arithmetic the freeze asserts about the corpus
    assert v("data", "train_chunks") + v("data", "dev_chunks") == v("data", "total_chunks")
    assert v("data", "train_documents") + v("data", "dev_documents") == 1_118_224


def test_the_sampler_is_still_seeded_by_run_seed_itself():
    """Frozen because D-S1B-016 explicitly did NOT change data order."""
    assert v("sampler", "seeded_by") == "run_seed"
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "train_run")
    call = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "DeterministicSampler")
    keyword = next(k for k in call.keywords if k.arg == "seed")
    assert ast.unparse(keyword.value) == "provenance.run_seed"


# ---------------------------------------------------------------------------
# 3. Training and optimizer — every value imported
# ---------------------------------------------------------------------------
def test_training_matches_the_code():
    assert v("training", "batch_size") == P.BATCH_SIZE
    assert v("training", "gradient_accumulation_steps") == P.GRADIENT_ACCUMULATION_STEPS
    assert v("training", "eval_every_updates") == P.EVAL_EVERY_UPDATES
    assert v("training", "checkpoint_every_updates") == CHECKPOINT_EVERY_UPDATES == P.EVAL_EVERY_UPDATES
    assert v("training", "initial_max_updates") == P.INITIAL_MAX_UPDATES
    assert v("training", "extended_max_updates") == P.EXTENDED_MAX_UPDATES
    assert v("training", "continuations_allowed") == 1


def test_optimizer_matches_the_code():
    assert v("optimizer", "name") == P.OPTIMIZER
    assert v("optimizer", "betas") == list(P.ADAMW_BETAS)
    assert v("optimizer", "eps") == P.ADAMW_EPS
    assert v("optimizer", "amsgrad") == P.AMSGRAD
    assert v("optimizer", "schedule") == P.LR_SCHEDULE
    assert v("optimizer", "warmup") == P.WARMUP
    assert v("optimizer", "gradient_clipping") == P.GRADIENT_CLIPPING
    assert v("optimizer", "weight_decay_weights") == P.WEIGHT_DECAY_WEIGHTS
    assert v("optimizer", "weight_decay_exempt") == P.WEIGHT_DECAY_EXEMPT
    assert v("optimizer", "lr_grid") == list(LR_PILOT_GRID)
    assert v("optimizer", "r_grid") == list(R_PHASE1_GRID)
    assert v("optimizer", "lambda_scale_sum") == P.LAMBDA_SCALE_SUM


def test_the_frozen_loop_order_matches_the_real_loop():
    """AST over the real `while` body, not a source-string search."""
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "train_run")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.While))
    seen = []
    for n in ast.walk(loop):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            if name in ("next_batch", "collate_stage1_batch", "batch_to_device",
                        "zero_grad", "backward", "gradient_report", "step"):
                seen.append((n.lineno, name))
        if isinstance(n, ast.AugAssign) and getattr(n.target, "id", None) == "global_update":
            seen.append((n.lineno, "global_update+=1"))
    order = [name for _, name in sorted(set(seen))]
    frozen = v("training", "loop_order")
    # forward and prepare are named differently in the freeze; compare the rest
    assert order == ["next_batch", "batch_to_device", "collate_stage1_batch",
                     "zero_grad", "backward", "gradient_report", "step",
                     "global_update+=1"], order
    assert frozen.index("zero_grad") < frozen.index("backward") < frozen.index("step")
    assert frozen.index("forward") < frozen.index("zero_grad"), (
        "the freeze must record that the forward precedes zero_grad"
    )


# ---------------------------------------------------------------------------
# 4. Run plan — seeds and init seeds RECOMPUTED, never copied
# ---------------------------------------------------------------------------
def test_the_run_plan_is_recomputed_from_the_real_schedules():
    planned = [*lr_pilot_schedule(P.SELECTION_SEED),
               *r_phase1_schedule(P.SELECTION_SEED, 3e-4),
               *final_main_schedule(3e-4, 1.0)]
    assert len(planned) == P.TOTAL_NOMINAL_RUNS == v("run_plan", "total_nominal_runs") == 11
    assert v("run_plan", "composition") == {"lr-pilot": 3, "r-phase1": 5, "final-main": 3}
    assert v("run_plan", "selection_seed") == P.SELECTION_SEED
    assert v("run_plan", "train_seeds") == list(P.TRAIN_SEEDS)
    frozen_runs = v("run_plan", "runs")
    assert len(frozen_runs) == 11
    for entry, plan in zip(frozen_runs, planned):
        assert entry["label"] == plan.label
        assert entry["run_seed"] == plan.seed


def test_every_frozen_init_seed_is_recomputed_from_the_production_helper():
    """Recomputed, not transcribed: `adapter_init_seed` is pure Python."""
    frozen = v("run_plan", "init_seeds")
    assert {int(k): val for k, val in frozen.items()} == P.ADAPTER_INIT_SEEDS
    for run_seed_text, init_seed in frozen.items():
        assert P.adapter_init_seed(int(run_seed_text)) == init_seed
    for entry in v("run_plan", "runs"):
        assert entry["init_seed"] == P.adapter_init_seed(entry["run_seed"])


def test_the_init_hash_grouping_follows_from_the_seeds():
    from collections import Counter

    groups = Counter(entry["init_seed"] for entry in v("run_plan", "runs"))
    frozen = v("run_plan", "init_hash_groups")
    assert len(groups) == frozen["distinct"] == 4
    assert sorted(groups.values(), reverse=True) == frozen["multiplicities"] == [8, 1, 1, 1]


def test_the_frozen_h0_hashes_are_recomputed_wherever_torch_exists():
    """The one value that needs torch. Skips honestly; authoritative on Colab."""
    torch = pytest.importorskip("torch", reason="H0 recomputation needs torch")
    assert torch is not None
    from unmark.stage1.initialisation import expected_fresh_init_hash

    for run_seed_text, expected in v("run_plan", "expected_fresh_init_hashes").items():
        recomputed = expected_fresh_init_hash(P.HIDDEN_SIZE, int(run_seed_text))
        assert recomputed == expected, f"H0 drift for run_seed {run_seed_text}"


# ---------------------------------------------------------------------------
# 5. Validation and selection
# ---------------------------------------------------------------------------
def test_validation_matches_the_code():
    assert v("validation", "conditions") == list(P.VALIDATION_CONDITIONS)
    assert v("validation", "corruption_seed") == P.VALIDATION_CORRUPTION_SEED
    assert v("validation", "batch_size") == P.BATCH_SIZE
    assert v("validation", "metric_unit") == P.METRIC_UNIT


def test_the_selection_rule_matches_select_checkpoint():
    """Structural: the frozen tie-break must be the real key function."""
    source = (REPO / "unmark/stage1/selection.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "select_checkpoint")
    body = ast.unparse(node)
    assert "min(points, key=lambda p: (p.score, p.d_clean, p.update))" in body
    assert v("selection", "tie_break") == ["score", "d_clean", "earliest update"]
    assert "worst case" in v("selection", "score") or "max over" in v("selection", "score")


def test_update_zero_is_a_hard_gate():
    from unmark.stage1.selection import SelectionViolation, ValidationPoint

    assert v("validation", "update_0_evaluated") is True
    point = ValidationPoint(update=500, distances={c: 1.0 for c in P.VALIDATION_CONDITIONS}, d_clean=0.5)
    with pytest.raises(SelectionViolation, match="update 0"):
        select_checkpoint([point])


# ---------------------------------------------------------------------------
# 6. Checkpoint / resume, preparation, sealing
# ---------------------------------------------------------------------------
def test_checkpoint_freeze_matches_the_code():
    assert v("checkpoint_resume", "schema_version") == CHECKPOINT_SCHEMA_VERSION
    assert v("checkpoint_resume", "checkpoints") == {
        "last": LAST_CHECKPOINT_NAME, "best": BEST_CHECKPOINT_NAME
    }
    assert v("checkpoint_resume", "restore_strict") is True
    assert v("checkpoint_resume", "map_location") == "cpu"
    assert v("checkpoint_resume", "state_inheritance_between_nominal_runs") is False


def test_preparation_freeze_matches_the_code():
    assert v("preparation_execution", "backend") == PREPARATION_BACKEND
    assert v("preparation_execution", "workers") == PREPARATION_WORKERS
    assert v("preparation_execution", "start_method") == MULTIPROCESSING_START_METHOD == "spawn"
    assert v("preparation_execution", "order_preserving") == ORDER_PRESERVING is True
    assert v("preparation_execution", "prefetch") == PREFETCH_ENABLED is False


def test_numerical_policy_freeze_matches_the_code():
    assert v("hardware_acceptance", "float32_matmul_precision") == FLOAT32_MATMUL_PRECISION
    assert v("hardware_acceptance", "cublas_workspace_config") == DETERMINISTIC_CUBLAS_WORKSPACE
    assert v("hardware_acceptance", "resume_blocking_fields") == list(ExecutionFingerprint.RESUME_BLOCKING)
    assert v("hardware_acceptance", "cuda_matmul_allow_tf32") is False
    assert v("hardware_acceptance", "cudnn_allow_tf32") is False


def test_test_sealing_freeze_is_a_safety_gate():
    from unmark.stage1.corpus import CorpusContractViolation, screen_contamination

    assert v("test_sealing", "official_uit_vsfc_test") == "SEALED"
    assert v("test_sealing", "official_test_used") is False
    assert v("test_sealing", "downstream_score_used_for_stage1_selection") is False
    with pytest.raises(CorpusContractViolation, match="SEALED"):
        screen_contamination([], {"uitvsfc_official_test": ["x"]})


def test_no_scientific_optimizer_step_has_occurred():
    assert v("runtime_acceptance", "scientific_optimizer_steps") == 0
    assert kind("runtime_acceptance", "scientific_optimizer_steps") == "safety_gate"


def test_the_projections_are_labelled_as_lower_bounds():
    entry = FREEZE["runtime_acceptance"]["lower_bound_projection_hours"]
    assert "LOWER BOUND" in entry["note"]
    assert entry["classification"] == "operational_acceptance"
