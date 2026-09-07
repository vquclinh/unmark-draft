"""Stage-2 head-campaign contract: cache identity, roles, plan, no A/B selection.

**Torch-free.** Everything here exercises the identity, role and selection-safety
contracts, which are the parts that can silently produce a plausible-looking
wrong number. The tensor-level half lives in
`test_stage2_head_campaign_torch.py`: a module-level `importorskip` here would
skip these structural checks too, which is exactly how an earlier repair in this
repository lost its torch-free coverage (see `test_stage1_device_contract.py`).

Nothing here trains, reads a downstream row, or names official TEST.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import Preg1Role, deterministic_batches, select_checkpoint  # noqa: E402
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    EPOCHS,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PRIMARY_DATASET,
    PRIMARY_NUM_LABELS,
    TRUNCATION,
)
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_CONDITIONS,
    Stage2UnmarkArm,
    finalist_for_arm,
)
from unmark.evaluation.stage2_head_campaign import (  # noqa: E402
    STAGE2_AB_SELECTION_IMPLEMENTED,
    STAGE2_AB_TIE_BREAK,
    STAGE2_AB_WINNER_RULE,
    STAGE2_CAMPAIGN_RUN_COUNT,
    STAGE2_CAMPAIGN_SEEDS,
    STAGE2_CLEAN_CONDITION,
    STAGE2_DEGRADED_CONDITIONS,
    STAGE2_HEAD_ARTIFACT_FIELDS,
    STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
    STAGE2_HEAD_LEARNING_RATE,
    STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED,
    STAGE2_MEASUREMENT_ROLE,
    STAGE2_PROTOCOL_VERSION,
    STAGE2_RESUME_CONTRACT,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    Stage2BoundRepresentations,
    Stage2CampaignRun,
    Stage2ConditionScore,
    Stage2HeadRunStore,
    Stage2RepresentationKey,
    aggregate_stage2_campaign,
    label_digest,
    require_clean_condition,
    require_frozen_protocol_spec,
    require_paired_campaign_plan,
    stage2_campaign_plan,
    stage2_measurement_extraction_plan,
    stage2_representation_key_for,
    stage2_training_extraction_plan,
    validate_stage2_head_artifact,
)
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULE = REPO / "unmark/evaluation/stage2_head_campaign.py"
HEAD_SHA = "a" * 40


def key(**overrides) -> Stage2RepresentationKey:
    base = dict(
        repository_head=HEAD_SHA,
        arm=Stage2UnmarkArm.UNMARK_A.value,
        finalist_checkpoint_sha256=finalist_for_arm("UNMARK-A").checkpoint_sha256,
        backbone_checkpoint=ENCODER_CHECKPOINT,
        backbone_revision=ENCODER_REVISION,
        protocol_version=STAGE2_PROTOCOL_VERSION,
        dataset=PRIMARY_DATASET,
        dataset_version="1.0",
        task="sentiment",
        role=STAGE2_TRAINING_ROLE.value,
        condition=STAGE2_CLEAN_CONDITION,
        corruption_seed=None,
        pooling=STAGE2_FIRST_TOKEN_POOLING,
        max_length=MAX_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING,
        ordered_id_digest="d" * 64,
        label_digest="e" * 64,
        dtype=STAGE2_REPRESENTATION_DTYPE,
        hidden_size=HIDDEN_SIZE,
        count=8,
    )
    base.update(overrides)
    return Stage2RepresentationKey(**base)


# ==========================================================================================
# A. Cache identity
# ==========================================================================================

def test_a_well_formed_key_round_trips():
    k = key()
    assert Stage2RepresentationKey.from_dict(k.to_dict()) == k
    k.require_compatible(k)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm", Stage2UnmarkArm.UNMARK_B.value),
        ("finalist_checkpoint_sha256", "f" * 64),
        ("repository_head", "b" * 40),
        ("protocol_version", "stage2-dual-finalist-protocol-v2"),
        ("role", Preg1Role.PROTOCOL_DEV.value),
        ("ordered_id_digest", "0" * 64),
        ("label_digest", "0" * 64),
        ("count", 9),
        ("backbone_revision", "0" * 40),
        ("dataset_version", "2.0"),
        ("max_length", 128),
    ],
)
def test_any_identity_mismatch_refuses_the_cache(field, value):
    wanted = key()
    cached = replace(wanted, **{field: value})
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        wanted.require_compatible(cached)


def test_a_condition_mismatch_refuses_the_cache():
    wanted = key(condition="P50", corruption_seed=11)
    cached = key(condition="P100", corruption_seed=11)
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        wanted.require_compatible(cached)


def test_a_corruption_seed_mismatch_refuses_the_cache():
    wanted = key(condition="P50", corruption_seed=11)
    cached = key(condition="P50", corruption_seed=12)
    with pytest.raises(EvaluationContractViolation, match="incompatible"):
        wanted.require_compatible(cached)


def test_the_clean_condition_may_not_carry_a_corruption_seed():
    with pytest.raises(EvaluationContractViolation, match="no corruption seed"):
        key(condition=STAGE2_CLEAN_CONDITION, corruption_seed=5)


def test_a_corrupted_condition_must_bind_its_seed():
    with pytest.raises(EvaluationContractViolation, match="must bind the corruption seed"):
        key(condition="P50", corruption_seed=None)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pooling", "masked_mean", "pooling is frozen"),
        ("hidden_size", 512, "768-dimensional"),
        ("dtype", "torch.float16", "no AMP"),
        ("count", 0, "count must be positive"),
    ],
)
def test_a_malformed_key_is_refused_at_construction(field, value, match):
    with pytest.raises(EvaluationContractViolation, match=match):
        key(**{field: value})


def test_an_unknown_arm_is_refused():
    with pytest.raises(EvaluationContractViolation, match="unknown Stage-2 UNMARK arm"):
        key(arm="UNMARK-C")


def test_variant_stays_fail_closed():
    """VARIANT is refused by the corruption layer before the grid check."""
    with pytest.raises(EvaluationContractViolation, match="not implemented"):
        key(condition="VARIANT", corruption_seed=3)


def test_a_condition_outside_the_frozen_grid_is_refused():
    with pytest.raises(EvaluationContractViolation, match="unknown|not in the frozen"):
        key(condition="P10", corruption_seed=3)


def test_bound_representations_refuse_a_shape_or_dtype_contradiction():
    class Fake:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

    k = key()
    Stage2BoundRepresentations(values=Fake((8, 768), STAGE2_REPRESENTATION_DTYPE), key=k)
    with pytest.raises(EvaluationContractViolation, match="contradicts its key"):
        Stage2BoundRepresentations(values=Fake((7, 768), STAGE2_REPRESENTATION_DTYPE), key=k)
    with pytest.raises(EvaluationContractViolation, match="dtype"):
        Stage2BoundRepresentations(values=Fake((8, 768), "torch.float16"), key=k)


def test_the_role_and_arm_come_from_provenance_not_an_argument():
    bound = Stage2BoundRepresentations(
        values=type("F", (), {"shape": (8, 768), "dtype": STAGE2_REPRESENTATION_DTYPE})(),
        key=key(role=STAGE2_MEASUREMENT_ROLE.value),
    )
    assert bound.role is STAGE2_MEASUREMENT_ROLE
    with pytest.raises(EvaluationContractViolation, match="requires role"):
        bound.require_role(STAGE2_SELECTION_ROLE, "checkpoint selection")


def test_cross_arm_use_is_refused():
    def bound(arm):
        return Stage2BoundRepresentations(
            values=type("F", (), {"shape": (8, 768), "dtype": STAGE2_REPRESENTATION_DTYPE})(),
            key=key(arm=arm, finalist_checkpoint_sha256=finalist_for_arm(arm).checkpoint_sha256),
        )

    a, b = bound("UNMARK-A"), bound("UNMARK-B")
    a.require_same_arm(a)
    with pytest.raises(EvaluationContractViolation, match="cross-arm"):
        a.require_same_arm(b)


def test_label_digest_is_order_sensitive():
    assert label_digest([0, 1, 2]) != label_digest([2, 1, 0])
    with pytest.raises(EvaluationContractViolation, match="empty label vector"):
        label_digest([])


# ==========================================================================================
# Extraction plan
# ==========================================================================================

def test_the_frozen_protocol_spec_is_loadable_and_matched():
    assert require_frozen_protocol_spec()["schema_version"] == STAGE2_PROTOCOL_VERSION


def test_the_training_plan_is_clean_full_for_both_arms():
    plan = stage2_training_extraction_plan()
    assert len(plan) == 4
    assert {r.arm for r in plan} == {"UNMARK-A", "UNMARK-B"}
    assert {r.role for r in plan} == {STAGE2_TRAINING_ROLE.value, STAGE2_SELECTION_ROLE.value}
    assert {r.condition for r in plan} == {STAGE2_CLEAN_CONDITION}
    assert {r.corruption_seed for r in plan} == {None}


def test_the_measurement_plan_covers_two_arms_by_six_conditions():
    plan = stage2_measurement_extraction_plan(corruption_seed=4242)
    assert len(plan) == 12
    assert {r.role for r in plan} == {STAGE2_MEASUREMENT_ROLE.value}
    assert {r.condition for r in plan} == set(STAGE2_UNMARK_CONDITIONS)
    for r in plan:
        expected = None if r.condition == STAGE2_CLEAN_CONDITION else 4242
        assert r.corruption_seed == expected


def test_the_measurement_corruption_seed_has_no_default():
    """It is not pinned by the frozen protocol, so it must not acquire a default."""
    assert STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED is False
    with pytest.raises(TypeError):
        stage2_measurement_extraction_plan()
    with pytest.raises(EvaluationContractViolation, match="must not acquire a default"):
        stage2_measurement_extraction_plan(corruption_seed="7")


def test_a_key_built_from_a_request_binds_the_arms_own_checkpoint():
    plan = stage2_training_extraction_plan()
    for request in plan:
        k = stage2_representation_key_for(
            request, repository_head=HEAD_SHA, ordered_ids=["s1", "s2"], labels=[0, 1]
        )
        assert k.finalist_checkpoint_sha256 == finalist_for_arm(request.arm).checkpoint_sha256
    a = next(r for r in plan if r.arm == "UNMARK-A")
    b = next(r for r in plan if r.arm == "UNMARK-B")
    ka = stage2_representation_key_for(a, repository_head=HEAD_SHA, ordered_ids=["x"], labels=[0])
    kb = stage2_representation_key_for(b, repository_head=HEAD_SHA, ordered_ids=["x"], labels=[0])
    assert ka.finalist_checkpoint_sha256 != kb.finalist_checkpoint_sha256


# ==========================================================================================
# D/E. Schedule and selection total order
# ==========================================================================================

def test_the_full_thirty_epoch_schedule_is_required():
    from unmark.evaluation.preg1_head import EpochScore, require_full_schedule

    full = [EpochScore(epoch=e, macro_f1=0.5, accuracy=0.5) for e in range(1, EPOCHS + 1)]
    require_full_schedule(full, EPOCHS)
    with pytest.raises(EvaluationContractViolation, match="no early stopping"):
        require_full_schedule(full[:10], EPOCHS)


def test_selection_is_macro_f1_then_accuracy_then_earliest_epoch():
    from unmark.evaluation.preg1_head import EpochScore

    assert select_checkpoint([
        EpochScore(epoch=1, macro_f1=0.50, accuracy=0.90),
        EpochScore(epoch=2, macro_f1=0.60, accuracy=0.10),
    ]).epoch == 2
    assert select_checkpoint([
        EpochScore(epoch=1, macro_f1=0.60, accuracy=0.10),
        EpochScore(epoch=2, macro_f1=0.60, accuracy=0.90),
    ]).epoch == 2
    assert select_checkpoint([
        EpochScore(epoch=3, macro_f1=0.60, accuracy=0.90),
        EpochScore(epoch=7, macro_f1=0.60, accuracy=0.90),
    ]).epoch == 3


def test_a_corrupted_condition_cannot_reach_selection():
    require_clean_condition(STAGE2_CLEAN_CONDITION, "selection")
    for condition in STAGE2_DEGRADED_CONDITIONS:
        with pytest.raises(EvaluationContractViolation, match="only use the clean"):
            require_clean_condition(condition, "Stage-2 head checkpoint selection")


# ==========================================================================================
# F. Deterministic batches
# ==========================================================================================

def test_batch_order_depends_only_on_the_seed():
    import random

    first = deterministic_batches(500, 53148 * 1000 + 1, 128)
    random.seed(999)
    [random.random() for _ in range(50)]
    assert deterministic_batches(500, 53148 * 1000 + 1, 128) == first
    assert deterministic_batches(500, 53148 * 1000 + 2, 128) != first


def test_every_example_is_seen_each_epoch():
    batches = deterministic_batches(500, 7, 128)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(500))


# ==========================================================================================
# G. Artifact identity
# ==========================================================================================

def artifact(**overrides) -> dict:
    arm = overrides.pop("arm", "UNMARK-A")
    train = key(arm=arm, finalist_checkpoint_sha256=finalist_for_arm(arm).checkpoint_sha256)
    dev = replace(train, role=STAGE2_SELECTION_ROLE.value)
    base = {
        "schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
        "arm": arm,
        "finalist_checkpoint_sha256": finalist_for_arm(arm).checkpoint_sha256,
        "repository_head": HEAD_SHA,
        "protocol_version": STAGE2_PROTOCOL_VERSION,
        "seed": STAGE2_CAMPAIGN_SEEDS[0],
        "initial_head_fingerprint": "1" * 64,
        "head_architecture": f"Linear({HIDDEN_SIZE}, {PRIMARY_NUM_LABELS}, bias=True)",
        "optimizer": "AdamW",
        "learning_rate": STAGE2_HEAD_LEARNING_RATE,
        "epochs": EPOCHS,
        "early_stopping": False,
        "selected_epoch": 4,
        "selected_macro_f1": 0.5,
        "selected_accuracy": 0.6,
        "selected_head_state_sha256": "2" * 64,
        "train_cache_key": train.to_dict(),
        "protocol_dev_cache_key": dev.to_dict(),
        "history_digest": "3" * 64,
        "precision": STAGE2_REPRESENTATION_DTYPE,
        "selection_role": STAGE2_SELECTION_ROLE.value,
        "selection_condition": STAGE2_CLEAN_CONDITION,
        "measurement_used_for_selection": False,
        "ab_selection_performed": False,
    }
    base.update(overrides)
    return base


def test_a_well_formed_artifact_validates():
    validate_stage2_head_artifact(
        artifact(), expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
    )


def test_an_a_artifact_cannot_load_as_b():
    with pytest.raises(EvaluationContractViolation, match="may never load as the other"):
        validate_stage2_head_artifact(
            artifact(), expected_arm="UNMARK-B", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


def test_a_seed_mismatch_refuses():
    with pytest.raises(EvaluationContractViolation, match="seed"):
        validate_stage2_head_artifact(
            artifact(), expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[1]
        )


def test_partial_binding_refuses():
    for field in ("selected_epoch", "train_cache_key", "history_digest", "seed"):
        broken = artifact()
        del broken[field]
        with pytest.raises(EvaluationContractViolation, match="missing"):
            validate_stage2_head_artifact(
                broken, expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
            )


def test_an_unknown_artifact_field_refuses():
    broken = artifact()
    broken["downstream_winner"] = "A"
    with pytest.raises(EvaluationContractViolation, match="unknown field"):
        validate_stage2_head_artifact(
            broken, expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


def test_a_mismatched_cache_key_refuses():
    broken = artifact()
    broken["train_cache_key"] = replace(
        key(), role=STAGE2_MEASUREMENT_ROLE.value
    ).to_dict()
    with pytest.raises(EvaluationContractViolation, match="train_cache_key has role"):
        validate_stage2_head_artifact(
            broken, expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


def test_a_cache_from_another_commit_refuses():
    broken = artifact()
    broken["train_cache_key"] = replace(key(), repository_head="9" * 40).to_dict()
    with pytest.raises(EvaluationContractViolation, match="another commit"):
        validate_stage2_head_artifact(
            broken, expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


def test_a_corrupted_selection_condition_refuses():
    broken = artifact(selection_condition="P50")
    with pytest.raises(EvaluationContractViolation, match="not selected on clean FULL"):
        validate_stage2_head_artifact(
            broken, expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


@pytest.mark.parametrize("field", ["measurement_used_for_selection", "ab_selection_performed"])
def test_a_selection_flag_set_true_refuses(field):
    with pytest.raises(EvaluationContractViolation, match=field):
        validate_stage2_head_artifact(
            artifact(**{field: True}),
            expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0],
        )


@pytest.mark.parametrize(("field", "value"), [("learning_rate", 0.001), ("epochs", 10),
                                              ("early_stopping", True)])
def test_a_drifted_budget_refuses(field, value):
    with pytest.raises(EvaluationContractViolation):
        validate_stage2_head_artifact(
            artifact(**{field: value}),
            expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0],
        )


# ==========================================================================================
# H. Resume / recovery
# ==========================================================================================

def test_a_completed_run_is_immutable(tmp_path):
    store = Stage2HeadRunStore(tmp_path / "run")
    store.directory.mkdir(parents=True)
    store.artifact_path.write_text(json.dumps(artifact()), encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="immutable"):
        store.require_writable(expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0])


def test_an_empty_directory_is_writable(tmp_path):
    Stage2HeadRunStore(tmp_path / "fresh").require_writable(
        expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
    )


def test_a_partial_run_may_not_be_adopted_under_another_identity(tmp_path):
    store = Stage2HeadRunStore(tmp_path / "run")
    store.mark_in_progress(arm="UNMARK-A", seed=STAGE2_CAMPAIGN_SEEDS[0])
    store.require_writable(expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0])
    for arm, seed in (("UNMARK-B", STAGE2_CAMPAIGN_SEEDS[0]),
                      ("UNMARK-A", STAGE2_CAMPAIGN_SEEDS[1])):
        with pytest.raises(EvaluationContractViolation, match="never adopted"):
            store.require_writable(expected_arm=arm, expected_seed=seed)


def test_the_resume_contract_is_declared():
    assert STAGE2_RESUME_CONTRACT == "atomic_rerun_no_midrun_resume"


def test_reading_an_absent_run_refuses(tmp_path):
    with pytest.raises(EvaluationContractViolation, match="no completed head run"):
        Stage2HeadRunStore(tmp_path / "nope").read_artifact(
            expected_arm="UNMARK-A", expected_seed=STAGE2_CAMPAIGN_SEEDS[0]
        )


# ==========================================================================================
# Campaign plan
# ==========================================================================================

def test_the_campaign_is_exactly_two_arms_by_five_seeds():
    plan = stage2_campaign_plan()
    assert len(plan) == STAGE2_CAMPAIGN_RUN_COUNT == 10
    assert {r.arm for r in plan} == {"UNMARK-A", "UNMARK-B"}
    assert sorted({r.seed for r in plan}) == sorted(MEASUREMENT_SEEDS)


def test_a_third_arm_or_extra_seed_is_refused():
    plan = list(stage2_campaign_plan())
    with pytest.raises(EvaluationContractViolation, match="exactly 10 runs"):
        require_paired_campaign_plan(plan + [Stage2CampaignRun(arm="UNMARK-A", seed=1)])


def test_dropping_a_seed_breaks_the_pairing():
    plan = [r for r in stage2_campaign_plan() if r.seed != MEASUREMENT_SEEDS[0]]
    with pytest.raises(EvaluationContractViolation, match="exactly 10 runs"):
        require_paired_campaign_plan(plan)


def test_an_unpaired_seed_is_refused():
    plan = list(stage2_campaign_plan())
    plan[0] = Stage2CampaignRun(arm=plan[1].arm, seed=plan[1].seed)
    with pytest.raises(EvaluationContractViolation, match="duplicate run|not paired"):
        require_paired_campaign_plan(plan)


# ==========================================================================================
# J. No A/B selection anywhere
# ==========================================================================================

def score(arm, seed, condition, f1):
    return Stage2ConditionScore(arm=arm, seed=seed, condition=condition, macro_f1=f1,
                                accuracy=f1, per_class_f1=(f1, f1, f1))


def full_scores(offset_b=0.1):
    rows = []
    for seed in MEASUREMENT_SEEDS:
        for condition in STAGE2_UNMARK_CONDITIONS:
            rows.append(score("UNMARK-A", seed, condition, 0.5))
            rows.append(score("UNMARK-B", seed, condition, 0.5 + offset_b))
    return rows


def test_the_aggregate_report_requires_both_arms():
    only_a = [r for r in full_scores() if r.arm == "UNMARK-A"]
    with pytest.raises(EvaluationContractViolation, match="must contain both arms"):
        aggregate_stage2_campaign(only_a)


def test_the_aggregate_report_names_no_winner():
    report = aggregate_stage2_campaign(full_scores())
    assert report["winner"] is None
    assert report["ab_selection_performed"] is False
    assert set(report["arms"]) == {"UNMARK-A", "UNMARK-B"}
    text = json.dumps(report).lower()
    for banned in ("best_arm", "winner_arm", "selected_arm", "ranking"):
        assert banned not in text


def test_the_report_carries_both_frozen_robustness_summaries():
    report = aggregate_stage2_campaign(full_scores())
    for arm in ("UNMARK-A", "UNMARK-B"):
        summaries = report["arms"][arm]["robustness_summaries"]
        assert "strip_all_macro_f1_mean" in summaries
        assert "degraded_equal_weight_macro_f1_mean" in summaries


def test_paired_deltas_are_descriptive_and_signed_b_minus_a():
    report = aggregate_stage2_campaign(full_scores(offset_b=0.2))
    deltas = report["paired_deltas_b_minus_a"]["FULL"]
    assert len(deltas) == len(MEASUREMENT_SEEDS)
    assert all(abs(d["macro_f1_delta_b_minus_a"] - 0.2) < 1e-9 for d in deltas)


def test_a_dropped_seed_in_the_report_is_refused():
    rows = [r for r in full_scores() if r.seed != MEASUREMENT_SEEDS[0]]
    with pytest.raises(EvaluationContractViolation, match="never be dropped"):
        aggregate_stage2_campaign(rows)


def test_the_module_declares_no_selection_machinery():
    assert STAGE2_AB_WINNER_RULE is None
    assert STAGE2_AB_TIE_BREAK is None
    assert STAGE2_AB_SELECTION_IMPLEMENTED is False


def test_no_function_in_the_module_names_a_winner_or_ranking():
    """AST, not grep: prose about *not* selecting must not trip this."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    banned = ("winner", "best_arm", "rank_arms", "select_arm", "choose_arm",
              "pick_finalist", "drop_arm", "promote_arm")
    defined = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for name in defined:
        lowered = name.lower()
        assert not any(b in lowered for b in banned), f"{name} looks like selection machinery"


# ==========================================================================================
# I. Role safety and the structural TEST seal
# ==========================================================================================

def test_official_test_has_no_role_member():
    assert not hasattr(Preg1Role, "OFFICIAL_TEST")
    assert {r.value for r in Preg1Role} == {
        "protocol-train", "protocol-dev", "official-validation"
    }


def test_only_protocol_dev_may_select():
    assert STAGE2_SELECTION_ROLE.may_select is True
    assert STAGE2_TRAINING_ROLE.may_select is False
    assert STAGE2_MEASUREMENT_ROLE.may_select is False


def test_only_protocol_train_may_train_the_head():
    assert STAGE2_TRAINING_ROLE.may_train_head is True
    assert STAGE2_SELECTION_ROLE.may_train_head is False
    assert STAGE2_MEASUREMENT_ROLE.may_train_head is False


def test_the_module_names_official_test_in_no_executable_position():
    """AST over identifiers and non-docstring literals.

    A substring scan would match this module's own explanation of why
    `Preg1Role` has no such member -- the recurring defect in this repository is
    a test that greps its own prose.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    identifiers, literals = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                literals.add(node.value)
    # Exact references, not substrings: the module legitimately declares
    # STAGE2_OFFICIAL_TEST_ROLE_EXISTS = False, whose whole purpose is to record
    # that the role is absent. What must not exist is a way to *reach* it.
    assert "OFFICIAL_TEST" not in identifiers, "module references an OFFICIAL_TEST name"
    for literal in literals:
        assert literal not in {"official-test", "official_test", "OFFICIAL_TEST", "test"}, (
            f"module carries the role literal {literal!r}"
        )
    assert not hasattr(Preg1Role, "OFFICIAL_TEST")


def test_the_artifact_schema_is_closed_and_complete():
    assert len(set(STAGE2_HEAD_ARTIFACT_FIELDS)) == len(STAGE2_HEAD_ARTIFACT_FIELDS)
    assert set(artifact()) == set(STAGE2_HEAD_ARTIFACT_FIELDS)


# ==========================================================================================
# Extraction driver: reuses the Audit-050 pathway, never re-implements it
# ==========================================================================================

def test_the_extraction_driver_delegates_to_the_audit_050_forward():
    """AST: the driver must *call* the accepted forward, not re-derive one."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "extract_stage2_unmark_representations" in called
    source = MODULE.read_text(encoding="utf-8")
    for reimplementation in ("base_word_embeddings", "authoritative_position_ids",
                             "inputs_embeds", "last_hidden_state"):
        assert reimplementation not in source, (
            f"{reimplementation} suggests the frozen forward pass was duplicated"
        )


def test_the_driver_refuses_a_pathway_from_the_other_arm():
    from unmark.evaluation.stage2_head_campaign import (
        Stage2RepresentationCache,
        extract_and_cache_stage2_representations,
    )

    class FakeBinding:
        arm = Stage2UnmarkArm.UNMARK_B
        checkpoint_sha256 = finalist_for_arm("UNMARK-B").checkpoint_sha256

    class FakePathway:
        binding = FakeBinding()

    with pytest.raises(EvaluationContractViolation, match="never write the other"):
        extract_and_cache_stage2_representations(
            FakePathway(), [{}], key(arm="UNMARK-A"), Stage2RepresentationCache("/tmp/nope")
        )


def test_the_driver_refuses_an_empty_batch_list():
    from unmark.evaluation.stage2_head_campaign import (
        Stage2RepresentationCache,
        extract_and_cache_stage2_representations,
    )

    class FakePathway:
        binding = None

    with pytest.raises(EvaluationContractViolation, match="no batches"):
        extract_and_cache_stage2_representations(
            FakePathway(), [], key(), Stage2RepresentationCache("/tmp/nope")
        )
