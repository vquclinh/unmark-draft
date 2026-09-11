"""**V2-GRD (C2)** -- geometry / relational distillation.

C2 tests ONE hypothesis from post-hoc diagnostic D4: preserving the CLEAN NATIVE
relational geometry in FIRST_TOKEN space improves downstream geometry while
retaining robustness. Even FULL UNMARK-A showed a native-vs-UNMARK FIRST_TOKEN
cosine distance of ~0.303, same-Vanilla-head prediction agreement of ~0.485 and a
centered-logit cosine of ~0.23 -- the space Stage-2 reads was not preserved by an
objective that constrains masked-mean pooled vectors.

The candidate adds one term and changes nothing else: historical fusion, historical
pooled terms, zero new parameters, three encoder forwards.

Three tiers, matching the C1 and C3 files: static/AST guards, torch-free contract
and provenance tests, and torch-gated numerics that skip cleanly here.

**Nothing here trains.** Backwards run only in torch-gated tests to verify
gradient routing; no optimizer is constructed and no parameter is updated.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.modeling.contracts import (
    HISTORICAL_FUSION_ID,
    SCALE_CALIBRATED_FUSION_ID,
)
from unmark.stage1.candidates import (
    CANDIDATES,
    FIRST_SCREEN_HARD_CAP,
    HISTORICAL_STAGES,
    PRECOMMITTED_CONTINUATION,
    BudgetPolicyViolation,
    budget_for_identity,
    candidate_for_identity,
    candidate_for_stage,
    wandb_project_for_stage,
)
from unmark.stage1.contracts import (
    GEOMETRY_RELATIONAL_OBJECTIVE,
    GEOMETRY_RELATIONAL_SPEC,
    GRID_CONSISTENCY_OBJECTIVE,
    HISTORICAL_FUSION,
    HISTORICAL_OBJECTIVE,
    SCALE_CALIBRATED_FUSION,
    RelationalSpec,
    Stage1ContractViolation,
    relational_spec_for,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORRUPTION_SEED,
    EXTENDED_MAX_UPDATES,
    HISTORICAL_OBJECTIVE_ID,
    INITIAL_MAX_UPDATES,
    LAMBDA_GRD,
    OBJECTIVE_IDS,
    RELATION_EPSILON,
    RELATION_LOSS,
    RELATION_METRIC,
    RELATION_SPACE,
    RELATIONAL_CLEAN_WEIGHT,
    RELATIONAL_CORRUPT_WEIGHT,
    RELATIONAL_OBJECTIVE_ID,
    V2_GC_STAGE,
    V2_GRD_LEARNING_RATE,
    V2_GRD_MAX_UPDATES,
    V2_GRD_R,
    V2_GRD_RUN_SEED,
    V2_GRD_STAGE,
    V2_SCF_STAGE,
    adapter_init_seed,
    lambda_grd_for,
    lambdas_for_r,
)
from unmark.stage1.reconstruct import (
    candidate_for_payload,
    recorded_identity,
    require_loadable_as,
)
from unmark.stage1.selection import ValidationPoint, v2_grd_schedule
from unmark.stage1.trainer import (
    RunProvenance,
    RunResult,
    TrainerContractViolation,
    checkpoint_payload,
    loss_telemetry,
    resolve_budget,
    resolve_run_cap,
    verify_checkpoint,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
RELATIONAL = "unmark/stage1/objective_relational.py"
OBJECTIVE = "unmark/stage1/objective.py"
GRID = "unmark/stage1/objective_grid.py"
EXECUTE = "unmark/stage1/execute.py"

C1 = candidate_for_stage(V2_SCF_STAGE)
C2 = candidate_for_stage(V2_GRD_STAGE)
C3 = candidate_for_stage(V2_GC_STAGE)
HISTORICAL = candidate_for_stage(HISTORICAL_STAGES[0])


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def function(name: str, module: str = RELATIONAL) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(ast.parse(source(module)))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def method(cls_name: str, name: str, module: str) -> ast.FunctionDef:
    cls = next(n for n in ast.walk(ast.parse(source(module)))
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in ast.walk(cls)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def body_without_docstring(node: ast.FunctionDef) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(s) for s in body)


def code_only(name: str) -> str:
    """Module source with every comment and string literal removed.

    Prose that *describes* a rule must not break a test that checks the rule:
    C2's own docstring says "no labels" and "same-head", which a naive grep for
    "label" or "head" would flag as violations of the very rules they state.
    """
    import io
    import tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source(name)).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def calls_in(node: ast.AST) -> list[str]:
    return [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(node) if isinstance(n, ast.Call)]


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=V2_GRD_RUN_SEED, init_seed=adapter_init_seed(V2_GRD_RUN_SEED),
        corruption_seed=CORRUPTION_SEED, learning_rate=V2_GRD_LEARNING_RATE,
        r=V2_GRD_R, corpus_manifest_digest="d" * 64, repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


HISTORICAL_PROVENANCE = provenance()
C1_PROVENANCE = provenance(fusion=SCALE_CALIBRATED_FUSION)
C2_PROVENANCE = provenance(objective=GEOMETRY_RELATIONAL_OBJECTIVE)
C3_PROVENANCE = provenance(objective=GRID_CONSISTENCY_OBJECTIVE)


def payload_for(p: RunProvenance, *, adapter_state=None) -> dict:
    return checkpoint_payload(
        provenance=p, adapter_state=adapter_state if adapter_state is not None else {},
        optimizer_state={}, global_update=500, sampler_state={"cursor": 3, "visit": 1},
        cap=INITIAL_MAX_UPDATES, budget_limited=False, points=[],
    )


def trajectory(best: int) -> list[ValidationPoint]:
    worse = {"FULL": .9, "P50": .9, "P100": .9, "STRIP_ALL": .9}
    top = {"FULL": .1, "P50": .1, "P100": .1, "STRIP_ALL": .1}
    points = [ValidationPoint(0, worse, .9)]
    if best:
        points.append(ValidationPoint(best, top, .1))
    return points


# ===========================================================================
# A. Relation mathematics (static/contract; numerics are torch-gated below)
# ===========================================================================
def test_the_locked_relational_specification():
    spec = GEOMETRY_RELATIONAL_SPEC
    assert spec.lambda_grd == LAMBDA_GRD == 1.0
    assert spec.relation_space == RELATION_SPACE == "FIRST_TOKEN"
    assert spec.relation_metric == RELATION_METRIC == "pairwise-cosine-gram"
    assert spec.relation_loss == RELATION_LOSS == "off-diagonal-mse"
    assert spec.relational_clean_weight == RELATIONAL_CLEAN_WEIGHT == 0.5
    assert spec.relational_corrupt_weight == RELATIONAL_CORRUPT_WEIGHT == 0.5
    assert spec.epsilon == RELATION_EPSILON == 1e-8


def test_the_relational_weights_must_sum_to_one():
    with pytest.raises(Stage1ContractViolation, match="sum to exactly 1.0"):
        RelationalSpec(
            lambda_grd=1.0, relation_space="FIRST_TOKEN",
            relation_metric="pairwise-cosine-gram", relation_loss="off-diagonal-mse",
            relational_clean_weight=0.7, relational_corrupt_weight=0.7, epsilon=1e-8,
        )


def test_only_the_relational_objective_carries_a_grd_weight():
    for objective_id in OBJECTIVE_IDS:
        weight = lambda_grd_for(objective_id)
        spec = relational_spec_for(objective_id)
        if objective_id == RELATIONAL_OBJECTIVE_ID:
            assert weight == 1.0 and spec is GEOMETRY_RELATIONAL_SPEC
        else:
            assert weight is None and spec is None, objective_id


def test_the_gram_is_row_normalised_over_the_feature_dimension_only():
    body = body_without_docstring(function("cosine_gram"))
    assert "x.norm(dim=-1, keepdim=True)" in body
    assert "clamp(min=eps)" in body
    assert "transpose(0, 1)" in body
    for banned in ("dim=0", "mean(", "- x.mean", "softmax", "exp(", "temperature"):
        assert banned not in body, f"the Gram reaches {banned}"


def test_the_relation_loss_excludes_the_diagonal_structurally():
    body = body_without_docstring(function("relational_distance"))
    assert "offdiagonal_mask(" in body, "the diagonal is not excluded by construction"
    assert "difference[mask] ** 2" in body
    mask = body_without_docstring(function("offdiagonal_mask"))
    assert "~torch.eye(" in mask, "the mask is not the complement of the identity"


def test_the_relation_loss_is_squared_error_and_nothing_else():
    body = body_without_docstring(function("relational_distance"))
    for banned in ("abs(", "l1_loss", "kl_div", "log(", "cov", "softmax",
                   "topk", "randperm", "multinomial", "temperature"):
        assert banned not in body, f"the relation loss reaches {banned}"


def test_the_teacher_is_detached_in_the_loss_path():
    body = body_without_docstring(function("relational_distance"))
    assert "teacher_target.detach()" in body, (
        "the teacher must be an explicit stop-gradient target"
    )
    assert "student.detach()" not in body


def test_the_students_are_never_detached():
    body = body_without_docstring(method(
        "RelationalDistillationObjective", "forward", RELATIONAL))
    for banned in ("student_clean.detach()", "student_corrupt.detach()",
                   "hidden_clean.detach()", "hidden_corrupt.detach()"):
        assert banned not in body, banned


def test_a_batch_below_two_fails_closed():
    """Structural: the guard exists and the constant is 2. Numerics are gated."""
    body = body_without_docstring(function("offdiagonal_mask"))
    assert "MINIMUM_BATCH" in body
    assert "raise Stage1ContractViolation" in body
    # Read the constant from the AST: the module imports torch, so it cannot be
    # imported in the ML-free environment.
    module = ast.parse(source(RELATIONAL))
    assigned = {
        target.id: node.value.value
        for node in module.body if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
    }
    assert assigned["MINIMUM_BATCH"] == 2


def test_the_loss_refuses_non_finite_inputs():
    body = body_without_docstring(function("relational_distance"))
    assert body.count("_require_finite") == 2, "both inputs must be checked"
    for banned in ("nan_to_num", "isnan", "clamp(min=-", "where(torch.isfinite"):
        assert banned not in body, f"the loss sanitises with {banned}"


def test_the_composition_is_exactly_one_half_each_and_lambda_one():
    body = body_without_docstring(method(
        "RelationalDistillationObjective", "forward", RELATIONAL))
    assert "self.spec.relational_clean_weight * loss_rel_clean" in body
    assert "self.spec.relational_corrupt_weight * loss_rel_corrupt" in body
    assert "self.spec.lambda_grd * loss_grd" in body
    assert "self.weights.lambda_align * loss_align" in body
    assert "self.weights.lambda_clean * loss_clean" in body


def test_lambda_grd_is_not_a_command_line_flag():
    declared = set()
    for node in ast.walk(ast.parse(source("scripts/stage1_runner.py"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("-"):
                        declared.add(argument.value)
    for forbidden in ("--lambda-grd", "--lambda_grd", "--relation-weight",
                      "--clean-weight", "--corrupt-weight", "--relation-space",
                      "--epsilon", "--objective", "--lambda-grid"):
        assert forbidden not in declared, f"the runner declares {forbidden}"


# ===========================================================================
# B. FIRST_TOKEN semantics
# ===========================================================================
def test_the_relation_space_is_the_first_token_row():
    body = body_without_docstring(function("first_token"))
    assert "hidden[:, 0, :]" in body


def test_the_objective_relates_first_token_not_the_pooled_vector():
    body = body_without_docstring(method(
        "RelationalDistillationObjective", "forward", RELATIONAL))
    for name in ("teacher = first_token(hidden_reference)",
                 "student_clean = first_token(hidden_clean)",
                 "student_corrupt = first_token(hidden_corrupt)"):
        assert name in body, name
    # The pooled vectors go ONLY to the two historical terms.
    assert "relational_distance(h_adapt_clean" not in body
    assert "relational_distance(h_ref" not in body


def test_the_module_does_not_pool_and_does_not_relate_tokens():
    body = source(RELATIONAL)
    for banned in ("masked_mean_non_special", "content_mask", "def masked_mean"):
        assert banned not in body, f"C2 reaches {banned}"
    # Relations are between EXAMPLES: the Gram takes a [B, d] matrix, never [B, L, d].
    gram = body_without_docstring(function("cosine_gram"))
    assert "x.dim() != 2" in gram, "the Gram accepts a token-grid tensor"


def test_there_is_no_direct_first_token_alignment_term():
    """C2 is NOT DP: FIRST_TOKEN is used only to build inter-example geometry."""
    body = body_without_docstring(method(
        "RelationalDistillationObjective", "forward", RELATIONAL))
    called = calls_in(method("RelationalDistillationObjective", "forward", RELATIONAL))
    # `representation_distance` is the pointwise cosine. It may be applied only to
    # the two POOLED historical terms, never to a FIRST_TOKEN pair.
    assert called.count("representation_distance") == 2
    for banned in ("representation_distance(student_clean",
                   "representation_distance(student_corrupt",
                   "representation_distance(teacher"):
        assert banned not in body, f"C2 added a direct FIRST_TOKEN alignment: {banned}"


# ===========================================================================
# C. Isolation between the three candidates
# ===========================================================================
def test_c2_uses_the_historical_fusion():
    assert C2.fusion is HISTORICAL_FUSION
    assert C2.fusion.fusion_id == HISTORICAL_FUSION_ID


def test_c1_and_c3_are_unchanged_by_c2():
    assert C1.fusion.fusion_id == SCALE_CALIBRATED_FUSION_ID
    assert C1.objective is HISTORICAL_OBJECTIVE
    assert C3.fusion is HISTORICAL_FUSION
    assert C3.objective is GRID_CONSISTENCY_OBJECTIVE
    assert C3.objective.lambda_grid == 1.0


def test_c2_has_no_grid_term_and_no_scale_calibration():
    assert C2.objective.lambda_grid is None
    assert C2.objective.lambda_grd == 1.0
    body = source(RELATIONAL)
    for banned in ("token_grid_distance", "loss_grid", "objective_grid",
                   "scale_calibrated", "fusion_id", "GridConsistency"):
        assert banned not in body, f"C2 reaches {banned}"


def test_c1_has_neither_a_grid_nor_a_relational_term():
    assert C1.objective.lambda_grid is None
    assert C1.objective.lambda_grd is None
    assert C1.objective.relational is None


def test_c3_has_a_grid_term_but_no_relational_one():
    assert C3.objective.lambda_grid == 1.0
    assert C3.objective.lambda_grd is None
    assert C3.objective.relational is None


def test_the_four_identities_are_distinct():
    identities = {HISTORICAL.identity, C1.identity, C2.identity, C3.identity}
    assert len(identities) == 4
    assert C2.identity == (RELATIONAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)


def test_the_historical_objective_module_is_unchanged_by_c2():
    body = source(OBJECTIVE)
    for forbidden in ("relational", "cosine_gram", "first_token", "lambda_grd",
                      "RelationalDistillation"):
        assert forbidden not in body, f"objective.py reaches {forbidden}"
    forward = body_without_docstring(method("Stage1Objective", "forward", OBJECTIVE))
    assert (
        "self.weights.lambda_align * loss_align + self.weights.lambda_clean * loss_clean"
        in forward
    )


def test_the_grid_objective_module_is_unchanged_by_c2():
    body = source(GRID)
    for forbidden in ("relational", "cosine_gram", "first_token", "lambda_grd"):
        assert forbidden not in body, f"objective_grid.py reaches {forbidden}"


# ===========================================================================
# D. Forwards and parameters
# ===========================================================================
def test_c2_runs_exactly_three_branches_statically():
    called = calls_in(method("RelationalDistillationObjective", "forward", RELATIONAL))
    assert called.count("reference_branch") == 1
    assert called.count("adapted_branch") == 2
    assert "reference_representation" not in called
    assert "adapted_representation" not in called


def test_the_relational_term_runs_no_forward_of_its_own():
    for name in ("relational_distance", "cosine_gram", "first_token",
                 "offdiagonal_mask", "offdiagonal_mean"):
        called = calls_in(function(name))
        for forbidden in ("reference_branch", "adapted_branch", "unmark_encoder",
                          "encoder", "reference_representation"):
            assert forbidden not in called, f"{name} reaches {forbidden}"


def test_the_teacher_diagnostic_reuses_the_existing_teacher():
    """`teacher_offdiag_cos_mean` must not trigger a fourth forward."""
    body = body_without_docstring(method(
        "RelationalDistillationObjective", "forward", RELATIONAL))
    assert "offdiagonal_mean(cosine_gram(teacher.detach()" in body.replace("\n", "").replace(" ", "") \
        or "offdiagonal_mean(" in body
    assert body.count("reference_branch") == 1


def test_c2_defines_no_model_parameters():
    import re

    body = code_only(RELATIONAL)
    for forbidden in ("nn.Parameter", "nn.Embedding", "nn.Linear", "nn.LayerNorm",
                      "register_parameter", "register_buffer", "Projection",
                      "head", "temperature"):
        assert not re.search(rf"\b{re.escape(forbidden)}\b", body), f"C2 adds {forbidden}"


def test_the_adapter_parameter_count_is_untouched():
    from unmark.modeling.config import AdapterConfig

    assert ADAPTER_TRAINABLE_PARAMETERS == 3_551_232
    assert AdapterConfig(hidden_size=768).parameter_count().total == 3_551_232


# ===========================================================================
# E. Provenance
# ===========================================================================
def test_the_c2_objective_identity_round_trips_exactly():
    recorded = payload_for(C2_PROVENANCE)["provenance"]["objective"]
    assert recorded == {
        "objective_id": "geometry-relational-distillation-v1",
        "lambda_grid": None,
        "lambda_grd": 1.0,
        "relational": {
            "lambda_grd": 1.0,
            "relation_space": "FIRST_TOKEN",
            "relation_metric": "pairwise-cosine-gram",
            "relation_loss": "off-diagonal-mse",
            "relational_clean_weight": 0.5,
            "relational_corrupt_weight": 0.5,
            "epsilon": 1e-8,
        },
    }


def test_the_relational_configuration_is_durable_through_json():
    import json

    payload = payload_for(C2_PROVENANCE)
    payload["provenance"] = json.loads(json.dumps(payload["provenance"]))
    verify_checkpoint(payload, C2_PROVENANCE)
    spec = payload["provenance"]["objective"]["relational"]
    assert spec["epsilon"] == 1e-8 and spec["relational_clean_weight"] == 0.5


@pytest.mark.parametrize("payload_name,env_name", [
    (p, e)
    for p in ("legacy", "historical", "c1", "c2", "c3")
    for e in ("historical", "c1", "c2", "c3")
])
def test_the_cross_candidate_matrix_is_fail_closed(payload_name, env_name):
    legacy = payload_for(HISTORICAL_PROVENANCE)
    del legacy["provenance"]["objective"]
    del legacy["provenance"]["fusion"]
    payloads = {
        "legacy": legacy,
        "historical": payload_for(HISTORICAL_PROVENANCE),
        "c1": payload_for(C1_PROVENANCE),
        "c2": payload_for(C2_PROVENANCE),
        "c3": payload_for(C3_PROVENANCE),
    }
    environments = {
        "historical": HISTORICAL_PROVENANCE, "c1": C1_PROVENANCE,
        "c2": C2_PROVENANCE, "c3": C3_PROVENANCE,
    }
    expected = "PASS" if (
        payload_name == env_name
        or (payload_name == "legacy" and env_name == "historical")
    ) else "REFUSE"
    try:
        verify_checkpoint(payloads[payload_name], environments[env_name])
        observed = "PASS"
    except TrainerContractViolation:
        observed = "REFUSE"
    assert observed == expected, f"{payload_name} -> {env_name}"


def test_a_missing_objective_can_never_mean_c2():
    legacy = payload_for(HISTORICAL_PROVENANCE)
    del legacy["provenance"]["objective"]
    assert recorded_identity(legacy) == (HISTORICAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    assert candidate_for_payload(legacy).objective is HISTORICAL_OBJECTIVE
    with pytest.raises(TrainerContractViolation, match="objective"):
        verify_checkpoint(legacy, C2_PROVENANCE)


def test_a_c2_checkpoint_is_recoverable_as_c2():
    assert recorded_identity(payload_for(C2_PROVENANCE)) == C2.identity
    assert candidate_for_payload(payload_for(C2_PROVENANCE)).stage == V2_GRD_STAGE
    assert candidate_for_identity(*C2.identity).stage == V2_GRD_STAGE


def test_a_c2_checkpoint_reconstructs_into_the_historical_fusion():
    """C2 changes the loss, not the model, so the architecture IS historical...

    ...but the scientific identity is still C2, and a historical loader must not
    be able to claim the checkpoint as UNMARK-A.
    """
    from unmark.stage1.reconstruct import recorded_fusion_id

    payload = payload_for(C2_PROVENANCE)
    assert recorded_fusion_id(payload["provenance"]) == HISTORICAL_FUSION_ID
    # The fusion check passes -- it really is the historical architecture...
    require_loadable_as(payload, HISTORICAL_FUSION_ID)
    # ...but the identity is C2, and the historical ENVIRONMENT still refuses it.
    assert recorded_identity(payload)[0] == RELATIONAL_OBJECTIVE_ID
    with pytest.raises(TrainerContractViolation):
        verify_checkpoint(payload, HISTORICAL_PROVENANCE)


def test_the_finalist_verifier_still_covers_both_identity_fields():
    from unmark.stage1.finalists import VERIFIED_PROVENANCE_FIELDS

    assert "objective" in VERIFIED_PROVENANCE_FIELDS
    assert "fusion" in VERIFIED_PROVENANCE_FIELDS


# ===========================================================================
# F. Budget
# ===========================================================================
def test_c2_is_hard_capped_at_twenty_thousand():
    assert budget_for_identity(*C2.identity) is FIRST_SCREEN_HARD_CAP
    assert C2.budget.hard_max_updates == V2_GRD_MAX_UPDATES == 20_000
    assert C2.budget.allows_precommitted_continuation is False


def test_c2_at_the_boundary_hard_stops():
    result = resolve_budget(
        RunResult(provenance=C2_PROVENANCE, points=trajectory(20_000), cap=20_000)
    )
    assert result.cap == 20_000 and result.cap != EXTENDED_MAX_UPDATES
    assert result.continued is False
    assert result.hard_capped is True
    assert result.to_dict()["budget_policy"]["hard_max_updates"] == 20_000


def test_c2_cannot_be_handed_a_forty_thousand_cap():
    with pytest.raises(BudgetPolicyViolation, match="HARD-CAPPED"):
        resolve_run_cap(C2_PROVENANCE, EXTENDED_MAX_UPDATES)
    assert resolve_run_cap(C2_PROVENANCE, INITIAL_MAX_UPDATES) == 20_000


@pytest.mark.parametrize("carried", [20_001, 25_000, 40_000])
def test_c2_over_cap_resume_is_refused(carried):
    with pytest.raises(BudgetPolicyViolation, match="refused rather than reinterpreted"):
        resolve_run_cap(C2_PROVENANCE, 20_000,
                        {"global_update": carried, "cap": EXTENDED_MAX_UPDATES})


@pytest.mark.parametrize("carried", [0, 500, 19_500, 20_000])
def test_c2_resume_within_the_cap_is_allowed(carried):
    assert resolve_run_cap(
        C2_PROVENANCE, 20_000, {"global_update": carried, "cap": 20_000}
    ) == 20_000


def test_historical_continuation_is_untouched_by_c2():
    result = resolve_budget(
        RunResult(provenance=HISTORICAL_PROVENANCE, points=trajectory(20_000), cap=20_000)
    )
    assert result.cap == EXTENDED_MAX_UPDATES and result.continued is True
    assert budget_for_identity(*HISTORICAL.identity) is PRECOMMITTED_CONTINUATION


def test_the_c2_plan_is_the_inherited_locked_values():
    (planned,) = v2_grd_schedule()
    assert planned.stage == V2_GRD_STAGE == "v2_grd"
    assert planned.learning_rate == V2_GRD_LEARNING_RATE == 1e-4
    assert planned.r == V2_GRD_R == 1.0
    assert planned.seed == V2_GRD_RUN_SEED == 36930
    assert adapter_init_seed(planned.seed) == 51800
    assert CORRUPTION_SEED == 35422
    assert lambdas_for_r(planned.r) == (1.0, 1.0)


def test_c2_does_not_grow_the_locked_campaign():
    from unmark.stage1.protocol import TOTAL_NOMINAL_RUNS
    from unmark.stage1.selection import total_planned_runs

    assert len(v2_grd_schedule()) == 1
    assert total_planned_runs() == TOTAL_NOMINAL_RUNS == 11


# ===========================================================================
# G. Selection and W&B
# ===========================================================================
def test_stage1_selection_is_unchanged_and_grd_never_reaches_it():
    for module in ("unmark/stage1/validation.py", "unmark/stage1/selection.py"):
        body = source(module)
        for forbidden in ("loss_grd", "relational_distance", "cosine_gram",
                          "objective_relational", "RelationalDistillation",
                          "first_token", "teacher_offdiag"):
            assert forbidden not in body, f"{module} reaches {forbidden}"
    from unmark.stage1.selection import select_checkpoint

    points = [
        ValidationPoint(0, {"FULL": .5, "P50": .5, "P100": .5, "STRIP_ALL": .9}, .5),
        ValidationPoint(500, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .4}, .1),
        ValidationPoint(1000, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .8}, .1),
    ]
    assert select_checkpoint(points).update == 500


def test_no_downstream_signal_enters_c2():
    """D-S1B-001: Stage-1 reads held-out UNLABELED signals only."""
    import re

    body = code_only(RELATIONAL).lower()
    for forbidden in ("uitvsfc", "uit_vsfc", "macro_f1", "labels", "logits",
                      "vanilla_head", "classifier", "severity"):
        assert not re.search(rf"\b{re.escape(forbidden)}\b", body), (
            f"C2 reaches {forbidden}"
        )


def test_each_candidate_resolves_its_own_wandb_project():
    assert wandb_project_for_stage(V2_SCF_STAGE) == "UNMARK-v2-C1-SCF-Stage1"
    assert wandb_project_for_stage(V2_GRD_STAGE) == "UNMARK-v2-C2-GRD-Stage1"
    assert wandb_project_for_stage(V2_GC_STAGE) == "UNMARK-v2-C3-GC-Stage1"
    for stage in HISTORICAL_STAGES:
        assert wandb_project_for_stage(stage) == "unmark-stage1", stage
    assert len({C1.wandb_project, C2.wandb_project, C3.wandb_project}) == 3


def _monitor():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_monitor_under_test_grd", REPO / "scripts/stage1_wandb_monitor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_monitor_resolves_all_three_candidate_projects():
    monitor = _monitor()
    assert monitor.project_for_event({"stage": V2_GRD_STAGE}) == "UNMARK-v2-C2-GRD-Stage1"
    assert monitor.project_for_event({"stage": V2_SCF_STAGE}) == "UNMARK-v2-C1-SCF-Stage1"
    assert monitor.project_for_event({"stage": V2_GC_STAGE}) == "UNMARK-v2-C3-GC-Stage1"


def test_candidate_run_ids_cannot_collide_across_projects():
    """C1/C2/C3 share a label and a seed; only the stage and project differ."""
    monitor = _monitor()
    events = [{"stage": s, "label": "seed=36930", "seed": 36930}
              for s in (V2_SCF_STAGE, V2_GRD_STAGE, V2_GC_STAGE)]
    keys = {
        f"{monitor.project_for_event(e)}/{monitor.candidate_key(e)}" for e in events
    }
    assert len(keys) == 3, f"candidate run keys collide: {keys}"


def test_the_wandb_config_can_carry_the_grd_identity():
    monitor = _monitor()
    for key in ("candidate_id", "objective_id", "fusion_id", "lambda_grd",
                "relational", "hard_max_updates", "repository_head", "seed",
                "init_seed", "corruption_seed", "lr", "batch_size",
                "encoder_checkpoint", "encoder_revision", "corpus_manifest_digest",
                "inventory_sha256"):
        assert key in monitor.SAFE_CONFIG_KEYS, f"{key} cannot reach the W&B config"


def test_the_monitor_maps_the_grd_series_without_touching_the_others():
    body = source("scripts/stage1_wandb_monitor.py")
    assert '"loss_grd", "loss_rel_clean", "loss_rel_corrupt"' in body
    assert '"train/loss": event.get("loss")' in body
    assert '"loss_grid", "mean_distance_grid"' in body, "C3 series was removed"
    assert 'f"train/scf/{key}"' in body, "C1 series was removed"


def test_wandb_is_observational_only():
    """No W&B name may reach a loss, a seed, a budget or a selection.

    The candidate register legitimately HOLDS each candidate's project name --
    that is where the mapping belongs -- but nothing that computes anything may
    mention it, and the scientific package must never import the library.
    """
    for module in ("unmark/stage1/trainer.py", "unmark/stage1/objective.py",
                   "unmark/stage1/objective_grid.py",
                   "unmark/stage1/objective_relational.py",
                   "unmark/stage1/selection.py", "unmark/stage1/validation.py",
                   "unmark/modeling/adapter.py"):
        assert "wandb" not in source(module).lower(), f"{module} reaches wandb"
    for path in (REPO / "unmark").rglob("*.py"):
        assert "import wandb" not in path.read_text(encoding="utf-8"), path


def test_telemetry_keeps_every_candidate_separate():
    class Scalar:
        def __init__(self, v): self.v = v
        def detach(self): return self
        def mean(self): return self
        def __float__(self): return float(self.v)

    class Historical:
        loss = Scalar(.3); loss_align = Scalar(.2); loss_clean = Scalar(.1)

    class Grid(Historical):
        loss_grid = Scalar(.05); distance_grid_per_example = Scalar(.05)

    class Relational(Historical):
        loss_grd = Scalar(.04); loss_rel_clean = Scalar(.03)
        loss_rel_corrupt = Scalar(.05); teacher_offdiagonal_mean = Scalar(.47)

    historical = loss_telemetry(Historical())
    grid = loss_telemetry(Grid())
    relational = loss_telemetry(Relational())
    assert list(historical) == ["loss", "loss_align", "loss_clean"]
    assert "loss_grd" not in grid and "loss_grid" not in relational
    assert set(relational) >= {"loss_grd", "loss_rel_clean", "loss_rel_corrupt",
                               "grd_teacher_offdiag_cos_mean"}
    assert list(relational)[:3] == ["loss", "loss_align", "loss_clean"]


# ===========================================================================
# H. Smoke dispatch
# ===========================================================================
def test_the_smoke_requires_relational_terms_only_for_c2():
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "if candidate.objective.lambda_grd is not None:" in body
    assert "if candidate.objective.lambda_grid is not None:" in body
    for term in ("loss_grd", "loss_rel_clean", "loss_rel_corrupt"):
        assert term in body, f"the smoke does not require {term}"
    for term in ("loss_grid", "mean_distance_grid"):
        assert term in body, f"the smoke stopped requiring {term} for C3"


def test_the_smoke_refuses_a_batch_too_small_for_relational_geometry():
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "candidate.objective.relational is not None" in body
    assert "len(usable) < 2" in body


def test_the_smoke_dispatch_covers_every_candidate_without_cross_wiring():
    from unmark.stage1.candidates import CANDIDATES as registered

    stages = {c.stage: c.identity for c in registered}
    assert stages[V2_GRD_STAGE] == (RELATIONAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    assert stages[V2_SCF_STAGE] == (HISTORICAL_OBJECTIVE_ID, SCALE_CALIBRATED_FUSION_ID)
    assert stages[V2_GC_STAGE] == ("grid-consistency-v1", HISTORICAL_FUSION_ID)
    for stage in HISTORICAL_STAGES:
        assert stages[stage] == (HISTORICAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)


def test_the_shared_constructor_dispatches_c2_on_its_objective_identity():
    builder = function("build_candidate_objective", EXECUTE)
    body = body_without_docstring(builder)
    assert "candidate.objective is GEOMETRY_RELATIONAL_OBJECTIVE" in body
    assert "RelationalDistillationObjective(unmark_encoder, weights)" in body
    conditions = [ast.unparse(n.test) for n in ast.walk(builder) if isinstance(n, ast.If)]
    assert "candidate.objective is GEOMETRY_RELATIONAL_OBJECTIVE" in conditions
    assert not any("V2_GRD_STAGE" in c for c in conditions), "dispatch on a stage name"


# ===========================================================================
# I. Torch-gated numerics
# ===========================================================================
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)

if torch is not None:  # pragma: no cover - exercised only where torch exists
    from test_stage1 import build_stack, synthetic_batch


@requires_torch
def test_runtime_the_gram_matches_hand_computed_cosines():
    """Expected values derived analytically, not read off the implementation."""
    from unmark.stage1.objective_relational import cosine_gram

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    gram = cosine_gram(x)
    assert gram.shape == (3, 3)
    root_half = 2.0 ** -0.5
    expected = torch.tensor([
        [1.0, 0.0, root_half],
        [0.0, 1.0, root_half],
        [root_half, root_half, 1.0],
    ])
    assert torch.allclose(gram, expected, atol=1e-6)
    assert torch.allclose(torch.diagonal(gram), torch.ones(3), atol=1e-6)


@requires_torch
def test_runtime_the_offdiagonal_mse_matches_a_hand_computed_value():
    from unmark.stage1.objective_relational import relational_distance

    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    # Off-diagonal G(T): 0, 0.70710678, 0.70710678 (and symmetric).
    # Off-diagonal G(S): 0, 1.0,        0.0
    # squared errors:    0, 0.08578644, 0.5  -> each twice -> mean 0.19526215
    assert relational_distance(student, teacher).item() == pytest.approx(
        0.19526214587563495, abs=1e-7
    )


@requires_torch
def test_runtime_the_diagonal_is_genuinely_excluded():
    """Including it would scale the loss by exactly (B-1)/B."""
    from unmark.stage1.objective_relational import cosine_gram, relational_distance

    teacher = torch.randn(5, 16)
    student = torch.randn(5, 16)
    off = relational_distance(student, teacher).item()
    full = ((cosine_gram(student) - cosine_gram(teacher)) ** 2).mean().item()
    assert full == pytest.approx(off * 4 / 5, rel=1e-6)
    assert off != pytest.approx(full, rel=1e-4)


@requires_torch
def test_runtime_identical_geometry_gives_exactly_zero():
    from unmark.stage1.objective_relational import relational_distance

    x = torch.randn(6, 16)
    assert relational_distance(x, x.clone()).item() == pytest.approx(0.0, abs=1e-12)
    # And a pure rescaling of every row is the SAME geometry: cosine is scale free.
    assert relational_distance(x * 7.5, x.clone()).item() == pytest.approx(0.0, abs=1e-9)


@requires_torch
def test_runtime_the_row_norm_is_over_the_feature_dimension_only():
    from unmark.stage1.objective_relational import cosine_gram

    x = torch.randn(4, 16)
    manual = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    assert torch.allclose(cosine_gram(x), manual @ manual.T, atol=0, rtol=0)


@requires_torch
def test_runtime_a_zero_row_stays_finite():
    from unmark.stage1.objective_relational import cosine_gram, relational_distance

    x = torch.randn(3, 16)
    x[1] = 0.0
    assert torch.isfinite(cosine_gram(x)).all()
    assert torch.isfinite(relational_distance(x, torch.randn(3, 16)))


@requires_torch
@pytest.mark.parametrize("batch", [0, 1])
def test_runtime_a_batch_below_two_fails_closed(batch):
    from unmark.stage1.objective_relational import relational_distance

    with pytest.raises(Stage1ContractViolation, match="at least 2 examples"):
        relational_distance(torch.randn(batch, 16), torch.randn(batch, 16))


@requires_torch
def test_runtime_a_shape_mismatch_fails_closed():
    from unmark.stage1.objective_relational import relational_distance

    with pytest.raises(Stage1ContractViolation):
        relational_distance(torch.randn(4, 16), torch.randn(5, 16))
    with pytest.raises(Stage1ContractViolation, match=r"\[B, d\]"):
        relational_distance(torch.randn(2, 4, 16), torch.randn(2, 4, 16))


@requires_torch
def test_runtime_non_finite_inputs_fail_closed_and_are_not_sanitised():
    from unmark.stage1.objective_relational import relational_distance

    bad = torch.randn(3, 8)
    bad[0, 0] = float("nan")
    with pytest.raises(Stage1ContractViolation, match="not finite"):
        relational_distance(bad, torch.randn(3, 8))
    bad2 = torch.randn(3, 8)
    bad2[1, 1] = float("inf")
    with pytest.raises(Stage1ContractViolation, match="not finite"):
        relational_distance(torch.randn(3, 8), bad2)


@requires_torch
def test_runtime_the_teacher_receives_no_relational_gradient():
    from unmark.stage1.objective_relational import relational_distance

    student = torch.randn(4, 16, requires_grad=True)
    teacher = torch.randn(4, 16, requires_grad=True)
    relational_distance(student, teacher).backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    assert teacher.grad is None, (
        "the teacher was not detached: the geometry target is being moved"
    )


@requires_torch
def test_runtime_first_token_is_the_row_at_index_zero():
    from unmark.stage1.objective_relational import first_token

    hidden = torch.randn(3, 7, 16)
    assert torch.equal(first_token(hidden), hidden[:, 0, :])
    with pytest.raises(Stage1ContractViolation, match=r"\[B, L, d\]"):
        first_token(torch.randn(3, 16))


def build_c2(d: int = 16):
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.contracts import ObjectiveWeights
    from unmark.stage1.execute import build_candidate_objective

    encoder, adapter, _, historical = build_stack(d)
    objective = build_candidate_objective(
        UnmarkEncoder(encoder, adapter), C2,
        ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
    )
    return encoder, adapter, objective, historical


@requires_torch
def test_runtime_the_builder_returns_the_relational_objective_for_c2():
    from unmark.stage1.objective_grid import GridConsistencyObjective
    from unmark.stage1.objective_relational import RelationalDistillationObjective

    _, _, objective, _ = build_c2()
    assert isinstance(objective, RelationalDistillationObjective)
    assert not isinstance(objective, GridConsistencyObjective)
    assert objective.objective is GEOMETRY_RELATIONAL_OBJECTIVE
    assert objective.spec is GEOMETRY_RELATIONAL_SPEC


@requires_torch
def test_runtime_c2_runs_exactly_three_encoder_forwards():
    encoder, _, objective, historical = build_c2()
    batch = synthetic_batch(batch=4)

    calls = []
    original = encoder.forward
    encoder.forward = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        objective(batch)
        c2_forwards = len(calls)
        calls.clear()
        historical(batch)
        historical_forwards = len(calls)
    finally:
        encoder.forward = original
    assert c2_forwards == 3, f"C2 ran {c2_forwards} encoder forwards, not 3"
    assert c2_forwards == historical_forwards


@requires_torch
def test_runtime_the_composition_is_exact():
    _, _, objective, _ = build_c2()
    result = objective(synthetic_batch(batch=4))
    expected_grd = 0.5 * result.loss_rel_clean + 0.5 * result.loss_rel_corrupt
    assert torch.allclose(result.loss_grd, expected_grd, atol=0, rtol=0)
    expected = result.loss_align + result.loss_clean + 1.0 * result.loss_grd
    assert torch.allclose(result.loss, expected, atol=0, rtol=0)
    assert not hasattr(result, "loss_grid"), "C2 produced a grid term"


@requires_torch
def test_runtime_the_two_pooled_terms_equal_the_historical_objectives():
    _, _, objective, historical = build_c2()
    batch = synthetic_batch(batch=4)
    with torch.no_grad():
        new = objective(batch)
        old = historical(batch)
    assert torch.allclose(new.loss_align, old.loss_align, atol=0, rtol=0)
    assert torch.allclose(new.loss_clean, old.loss_clean, atol=0, rtol=0)


@requires_torch
def test_runtime_both_students_receive_relational_gradient():
    _, adapter, objective, _ = build_c2()
    objective.train()

    result = objective(synthetic_batch(batch=4))
    result.loss_rel_clean.backward(retain_graph=True)
    clean_reached = {n for n, p in adapter.named_parameters()
                     if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert clean_reached, "L_rel_clean reaches no adapter parameter"

    for parameter in adapter.parameters():
        parameter.grad = None
    result.loss_rel_corrupt.backward()
    corrupt_reached = {n for n, p in adapter.named_parameters()
                       if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert corrupt_reached, "L_rel_corrupt reaches no adapter parameter"


@requires_torch
def test_runtime_the_frozen_encoder_receives_no_parameter_gradients():
    encoder, adapter, objective, _ = build_c2()
    objective.train()
    objective(synthetic_batch(batch=4)).loss.backward()
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert any(float(p.grad.abs().sum()) != 0 for p in adapter.parameters())


@requires_torch
def test_runtime_c2_adds_zero_model_parameters():
    _, _, objective, historical = build_c2()
    assert [(n, p.numel()) for n, p in objective.named_parameters()] == (
        [(n, p.numel()) for n, p in historical.named_parameters()]
    )
    assert sorted(objective.state_dict()) == sorted(historical.state_dict())


@requires_torch
def test_runtime_the_result_dict_carries_the_full_identity():
    _, _, objective, _ = build_c2()
    payload = objective(synthetic_batch(batch=4)).to_dict()
    assert payload["objective_id"] == "geometry-relational-distillation-v1"
    assert payload["lambda_grd"] == 1.0
    assert payload["relational"]["relation_space"] == "FIRST_TOKEN"
    for key in ("loss", "loss_align", "loss_clean", "loss_grd",
                "loss_rel_clean", "loss_rel_corrupt", "teacher_offdiag_cos_mean"):
        assert isinstance(payload[key], float), key


@requires_torch
def test_runtime_the_teacher_is_the_clean_reference_not_the_corrupted_branch():
    """The teacher must come from the reference branch's hidden states."""
    from unmark.stage1.objective_relational import first_token

    _, _, objective, _ = build_c2()
    batch = synthetic_batch(batch=4)
    with torch.no_grad():
        hidden_reference, _ = objective.reference_branch(
            batch["reference_input_ids"], batch["reference_attention_mask"],
            batch["reference_special_tokens_mask"],
        )
        hidden_corrupt, _ = objective.adapted_branch(
            batch["base_input_ids"], batch["base_attention_mask"],
            batch["base_special_tokens_mask"],
            batch["corrupt_tone_ids"], batch["corrupt_tone_mask"],
            batch["corrupt_letter_ids"], batch["corrupt_letter_mask"],
        )
    assert not torch.allclose(
        first_token(hidden_reference), first_token(hidden_corrupt)
    ), "the teacher and the corrupt student coincide; the test is not measuring"


@requires_torch
def test_runtime_a_single_example_batch_fails_closed_end_to_end():
    _, _, objective, _ = build_c2()
    with pytest.raises(Stage1ContractViolation, match="at least 2 examples"):
        objective(synthetic_batch(batch=1))
