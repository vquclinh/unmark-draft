"""**V2-SCF (C1)** -- the scale-calibrated fusion candidate.

C1 tests ONE mechanistic hypothesis identified by post-hoc diagnostics D3/D4: the
historical gate mixes two branches whose norms differ by ~20x on protocol-dev
(`mean ||e|| ~= 1.41` against `mean ||f|| ~= 27.9`), so a gate mean of ~0.03 on
FULL clean input still moves the representation by `||z-e||/||e|| ~= 1.42`. A
numerically small gate does not imply a small intervention.

The candidate calibrates `f` to `||e||` per token before the existing gated
mixture, and changes nothing else: same `q`, same `W_f`, same LayerNorm, same
`W_g`, same `g`, same convex combination, same tone and letter channels, same
historical Stage-1 objective, **zero new parameters**.

Three tiers, matching the C3 file: static/AST guards, torch-free contract and
provenance tests, and torch-gated numerics that skip cleanly here and run on
Colab.

**Nothing here trains.** Backwards run only in torch-gated tests to verify
gradient routing; no optimizer is constructed and no parameter is updated.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.modeling.config import AdapterConfig
from unmark.modeling.contracts import (
    FUSION_IDS,
    FUSION_SCALE_EPSILON,
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
    GRID_CONSISTENCY_OBJECTIVE,
    HISTORICAL_FUSION,
    HISTORICAL_OBJECTIVE,
    SCALE_CALIBRATED_FUSION,
    FusionIdentity,
    Stage1ContractViolation,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORRUPTION_SEED,
    EXTENDED_MAX_UPDATES,
    HISTORICAL_OBJECTIVE_ID,
    INITIAL_MAX_UPDATES,
    V2_GC_STAGE,
    V2_SCF_LEARNING_RATE,
    V2_SCF_MAX_UPDATES,
    V2_SCF_R,
    V2_SCF_RUN_SEED,
    V2_SCF_STAGE,
    adapter_init_seed,
    lambdas_for_r,
)
from unmark.stage1.reconstruct import (
    recorded_fusion_id,
    recorded_identity,
    require_loadable_as,
)
from unmark.stage1.selection import v2_scf_schedule
from unmark.stage1.trainer import (
    RunProvenance,
    RunResult,
    TrainerContractViolation,
    checkpoint_payload,
    resolve_budget,
    resolve_run_cap,
    scale_telemetry,
    verify_checkpoint,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = "unmark/modeling/adapter.py"
EXECUTE = "unmark/stage1/execute.py"

C1 = candidate_for_stage(V2_SCF_STAGE)
C3 = candidate_for_stage(V2_GC_STAGE)
HISTORICAL = candidate_for_stage(HISTORICAL_STAGES[0])


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def function(name: str, module: str = ADAPTER) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(ast.parse(source(module)))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def body_without_docstring(node: ast.FunctionDef) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(s) for s in body)


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=V2_SCF_RUN_SEED, init_seed=adapter_init_seed(V2_SCF_RUN_SEED),
        corruption_seed=CORRUPTION_SEED, learning_rate=V2_SCF_LEARNING_RATE,
        r=V2_SCF_R, corpus_manifest_digest="d" * 64, repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


HISTORICAL_PROVENANCE = provenance()
C1_PROVENANCE = provenance(fusion=SCALE_CALIBRATED_FUSION)
C3_PROVENANCE = provenance(objective=GRID_CONSISTENCY_OBJECTIVE)


def payload_for(p: RunProvenance, *, adapter_state=None, global_update: int = 500) -> dict:
    return checkpoint_payload(
        provenance=p, adapter_state=adapter_state if adapter_state is not None else {},
        optimizer_state={}, global_update=global_update,
        sampler_state={"cursor": 3, "visit": 1}, cap=INITIAL_MAX_UPDATES,
        budget_limited=False, points=[],
    )


# ===========================================================================
# A. Fusion mathematics (static + contract; numerics are torch-gated below)
# ===========================================================================
def test_the_epsilon_is_exactly_1e_8():
    assert FUSION_SCALE_EPSILON == 1e-8


def test_the_fusion_id_set_is_closed():
    assert FUSION_IDS == (HISTORICAL_FUSION_ID, SCALE_CALIBRATED_FUSION_ID)
    with pytest.raises(Stage1ContractViolation, match="closed set"):
        FusionIdentity("scale-calibrated-fusion-v2")
    with pytest.raises(ValueError, match="unknown fusion_id"):
        AdapterConfig(hidden_size=8, fusion_id="something-else")


def test_the_calibration_norm_is_over_the_hidden_dimension_only():
    """`dim=-1, keepdim=True`. Not a batch statistic and not a sequence one."""
    body = body_without_docstring(function("scale_calibrated_fusion"))
    assert body.count("dim=-1") == 2, "both norms must reduce the hidden dimension"
    assert "keepdim=True" in body
    for banned in ("dim=0", "dim=1", "dim=(0", "mean(", "flatten", "view(", "reshape"):
        assert banned not in body, f"the calibration reaches {banned}"


def test_the_calibration_clamps_the_denominator_at_the_locked_epsilon():
    body = body_without_docstring(function("scale_calibrated_fusion"))
    assert "clamp(min=FUSION_SCALE_EPSILON)" in body
    assert "1e-8" not in body, "the epsilon must be imported, never retyped"


def test_the_calibration_detaches_nothing():
    """Gradients must flow through `f` and through `scale`."""
    body = body_without_docstring(function("scale_calibrated_fusion"))
    assert ".detach()" not in body
    assert "no_grad" not in body
    assert "requires_grad" not in body


def test_the_calibration_does_not_normalise_e():
    """`e` enters the mixture exactly as it always did."""
    body = body_without_docstring(function("scale_calibrated_fusion"))
    assert "return scale * fused" in body
    assert "base /" not in body and "/ base" not in body


def test_the_adapter_keeps_the_historical_gate_and_combination():
    """`g`, `q`, the LayerNorm and the convex combination are untouched."""
    body = body_without_docstring(function("forward", ADAPTER))
    assert "g = torch.sigmoid(self.gate(q))" in body
    assert "f = self.layer_norm(self.fusion(q))" in body
    assert "q = torch.cat([e, t, l], dim=-1)" in body
    assert "convex_combination(g, f, e)" in body
    # The calibration sits strictly between the LayerNorm and the mixture.
    assert body.index("self.layer_norm") < body.index("is_scale_calibrated")
    assert body.index("is_scale_calibrated") < body.index("torch.sigmoid")


def test_the_calibration_does_not_touch_the_gate_or_the_channels():
    body = body_without_docstring(function("scale_calibrated_fusion"))
    for banned in ("gate", "sigmoid", "tone", "letter", "layer_norm", "clip", "clamp_"):
        assert banned not in body, f"the calibration reaches {banned}"


def test_no_learned_scale_parameter_is_introduced():
    body = source(ADAPTER)
    scf = body_without_docstring(function("scale_calibrated_fusion"))
    for banned in ("nn.Parameter", "register_parameter", "self.scale", "Parameter("):
        assert banned not in scf, f"C1 introduces a learned {banned}"
    assert "nn.Parameter" not in body


# ===========================================================================
# B. Isolation -- historical and C3 are untouched
# ===========================================================================
def test_the_historical_fusion_is_the_default_everywhere():
    assert AdapterConfig(hidden_size=8).fusion_id == HISTORICAL_FUSION_ID
    assert AdapterConfig(hidden_size=8).is_scale_calibrated is False
    assert HISTORICAL.fusion is HISTORICAL_FUSION
    assert C3.fusion is HISTORICAL_FUSION, "C3 must keep the historical fusion"


def test_c1_uses_the_scale_calibrated_fusion():
    assert C1.fusion is SCALE_CALIBRATED_FUSION
    assert AdapterConfig(hidden_size=8, fusion_id=SCALE_CALIBRATED_FUSION_ID).is_scale_calibrated


def test_c1_trains_the_historical_objective_and_has_no_grid_term():
    assert C1.objective is HISTORICAL_OBJECTIVE
    assert C1.objective.objective_id == HISTORICAL_OBJECTIVE_ID
    assert C1.objective.lambda_grid is None, "C1 must have no grid term at all"


def test_c3_still_has_its_grid_term():
    assert C3.objective is GRID_CONSISTENCY_OBJECTIVE
    assert C3.objective.lambda_grid == 1.0


def test_the_three_candidates_are_distinguished_by_the_identity_pair():
    assert HISTORICAL.identity == (HISTORICAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    assert C1.identity == (HISTORICAL_OBJECTIVE_ID, SCALE_CALIBRATED_FUSION_ID)
    assert C3.identity == ("grid-consistency-v1", HISTORICAL_FUSION_ID)
    assert len({HISTORICAL.identity, C1.identity, C3.identity}) == 3


def test_the_historical_objective_module_knows_nothing_about_fusion():
    """C1 is an architecture change; the loss must not learn about it."""
    body = source("unmark/stage1/objective.py")
    for forbidden in ("fusion_id", "scale_calibrated", "SCALE_CALIBRATED"):
        assert forbidden not in body, f"objective.py reaches {forbidden}"


def test_the_grid_objective_module_knows_nothing_about_fusion():
    body = source("unmark/stage1/objective_grid.py")
    for forbidden in ("fusion_id", "scale_calibrated", "SCALE_CALIBRATED"):
        assert forbidden not in body, f"objective_grid.py reaches {forbidden}"


# ===========================================================================
# C. Parameters and forwards
# ===========================================================================
def test_the_fusion_changes_no_parameter_count():
    for d in (8, 64, 768):
        historical = AdapterConfig(hidden_size=d)
        calibrated = AdapterConfig(hidden_size=d, fusion_id=SCALE_CALIBRATED_FUSION_ID)
        assert historical.parameter_count().to_dict() == calibrated.parameter_count().to_dict()
    assert AdapterConfig(hidden_size=768).parameter_count().total == 3_551_232
    assert ADAPTER_TRAINABLE_PARAMETERS == 3_551_232


def test_c1_uses_the_unchanged_historical_objective_so_three_forwards():
    """C1 changes the adapter, not the loss, so the forward count is historical."""
    tree = ast.parse(source("unmark/stage1/objective.py"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "Stage1Objective")
    fwd = next(n for n in ast.walk(cls)
               if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = [getattr(c.func, "id", None) or getattr(c.func, "attr", None)
             for c in ast.walk(fwd) if isinstance(c, ast.Call)]
    assert calls.count("reference_representation") == 1
    assert calls.count("adapted_representation") == 2


def test_the_scale_diagnostics_run_no_encoder_forward():
    recorder = body_without_docstring(function("_record_scale_diagnostics", ADAPTER))
    for banned in ("encoder", "unmark_encoder", "forward(", "self.fusion(", "self.gate("):
        assert banned not in recorder, f"the diagnostics reach {banned}"
    assert "no_grad" in recorder, "diagnostics must not build a graph"
    assert "detach" in recorder or "no_grad" in recorder


def test_the_diagnostics_are_not_a_buffer_and_cannot_reach_a_payload():
    body = source(ADAPTER)
    assert "register_buffer" not in body
    assert "_scale_diagnostics" in body


def test_the_diagnostics_are_inert_for_a_historical_adapter():
    recorder = body_without_docstring(function("_record_scale_diagnostics", ADAPTER))
    assert recorder.startswith("if not self.config.is_scale_calibrated:"), (
        "the historical path must execute no diagnostic op at all"
    )


def test_scale_telemetry_is_empty_without_a_calibrated_adapter():
    class Plain:
        pass

    class Historical:
        def scale_diagnostics(self):
            return {}

    assert scale_telemetry(Plain()) == {}
    assert scale_telemetry(Historical()) == {}


def test_scale_telemetry_prefixes_and_never_raises():
    class Calibrated:
        def scale_diagnostics(self):
            return {"scale_factor": 0.05, "intervention_ratio": 0.2}

    class Broken:
        def scale_diagnostics(self):
            raise RuntimeError("diagnostics must never break a run")

    assert scale_telemetry(Calibrated()) == {
        "scf_scale_factor": 0.05, "scf_intervention_ratio": 0.2
    }
    assert scale_telemetry(Broken()) == {}


# ===========================================================================
# D. Provenance
# ===========================================================================
def test_the_default_fusion_in_provenance_is_historical():
    assert HISTORICAL_PROVENANCE.fusion is HISTORICAL_FUSION
    assert HISTORICAL_PROVENANCE.to_dict()["fusion"] == {"fusion_id": HISTORICAL_FUSION_ID}


def test_a_c1_artifact_records_its_architecture():
    recorded = payload_for(C1_PROVENANCE)["provenance"]
    assert recorded["fusion"] == {"fusion_id": SCALE_CALIBRATED_FUSION_ID}
    assert recorded["objective"]["objective_id"] == HISTORICAL_OBJECTIVE_ID
    assert recorded["repository_head"] == "a" * 40
    assert recorded["run_seed"] == 36930 and recorded["init_seed"] == 51800


@pytest.mark.parametrize("payload_name,env_name,expected", [
    ("legacy", "historical", "PASS"),
    ("legacy", "c1", "REFUSE"),
    ("c1", "historical", "REFUSE"),
    ("c1", "c1", "PASS"),
    ("c1", "c3", "REFUSE"),
    ("c3", "c1", "REFUSE"),
    ("historical", "c1", "REFUSE"),
    ("historical", "historical", "PASS"),
    ("c3", "c3", "PASS"),
])
def test_the_cross_candidate_compatibility_matrix_is_fail_closed(
    payload_name, env_name, expected
):
    legacy = payload_for(HISTORICAL_PROVENANCE)
    del legacy["provenance"]["objective"]
    del legacy["provenance"]["fusion"]
    payloads = {
        "legacy": legacy,
        "historical": payload_for(HISTORICAL_PROVENANCE),
        "c1": payload_for(C1_PROVENANCE),
        "c3": payload_for(C3_PROVENANCE),
    }
    environments = {
        "historical": HISTORICAL_PROVENANCE, "c1": C1_PROVENANCE, "c3": C3_PROVENANCE,
    }
    try:
        verify_checkpoint(payloads[payload_name], environments[env_name])
        observed = "PASS"
    except TrainerContractViolation:
        observed = "REFUSE"
    assert observed == expected


def test_a_missing_fusion_block_means_historical_and_only_historical():
    legacy = payload_for(HISTORICAL_PROVENANCE)
    del legacy["provenance"]["fusion"]
    assert recorded_fusion_id(legacy["provenance"]) == HISTORICAL_FUSION_ID
    verify_checkpoint(legacy, HISTORICAL_PROVENANCE)  # must not raise
    with pytest.raises(TrainerContractViolation, match="fusion"):
        verify_checkpoint(legacy, C1_PROVENANCE)


def test_a_malformed_or_unknown_fusion_block_fails_closed():
    for bad in ({"fusion_id": "invented"}, {"nope": 1}, "historical-fusion-v1"):
        with pytest.raises(Stage1ContractViolation):
            recorded_fusion_id({"fusion": bad})


def test_the_finalist_verifier_covers_the_fusion_field():
    from unmark.stage1.finalists import VERIFIED_PROVENANCE_FIELDS

    assert "fusion" in VERIFIED_PROVENANCE_FIELDS
    assert "objective" in VERIFIED_PROVENANCE_FIELDS


def test_a_checkpoints_candidate_is_recoverable_from_its_identity_alone():
    assert recorded_identity(payload_for(C1_PROVENANCE)) == C1.identity
    assert candidate_for_identity(*C1.identity).stage == V2_SCF_STAGE
    assert candidate_for_identity(*C3.identity).stage == V2_GC_STAGE


def test_a_historical_loader_refuses_a_c1_payload():
    """The shapes match, so without this it would load silently and be wrong."""
    require_loadable_as(payload_for(HISTORICAL_PROVENANCE), HISTORICAL_FUSION_ID)
    with pytest.raises(Stage1ContractViolation, match="wrong and looks fine"):
        require_loadable_as(payload_for(C1_PROVENANCE), HISTORICAL_FUSION_ID)


# ===========================================================================
# E. Budget -- the C1 hard cap, on the real execution path
# ===========================================================================
def test_c1_is_hard_capped_at_twenty_thousand():
    assert budget_for_identity(*C1.identity) is FIRST_SCREEN_HARD_CAP
    assert C1.budget.hard_max_updates == V2_SCF_MAX_UPDATES == 20_000
    assert C1.budget.allows_precommitted_continuation is False


def test_the_historical_budget_is_not_inherited_by_c1():
    """The defect an objective-keyed register would have: C1 shares the objective."""
    assert budget_for_identity(HISTORICAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID) is (
        PRECOMMITTED_CONTINUATION
    )
    assert budget_for_identity(HISTORICAL_OBJECTIVE_ID, SCALE_CALIBRATED_FUSION_ID) is (
        FIRST_SCREEN_HARD_CAP
    )


def test_c1_at_the_boundary_hard_stops_and_says_so():
    worse = {"FULL": .9, "P50": .9, "P100": .9, "STRIP_ALL": .9}
    best = {"FULL": .1, "P50": .1, "P100": .1, "STRIP_ALL": .1}
    from unmark.stage1.selection import ValidationPoint

    result = resolve_budget(RunResult(
        provenance=C1_PROVENANCE,
        points=[ValidationPoint(0, worse, .9), ValidationPoint(20_000, best, .1)],
        cap=20_000,
    ))
    assert result.cap == 20_000 and result.cap != EXTENDED_MAX_UPDATES
    assert result.continued is False
    assert result.hard_capped is True
    assert result.to_dict()["budget_policy"]["hard_max_updates"] == 20_000


def test_c1_cannot_be_handed_a_forty_thousand_cap():
    with pytest.raises(BudgetPolicyViolation, match="HARD-CAPPED"):
        resolve_run_cap(C1_PROVENANCE, EXTENDED_MAX_UPDATES)
    assert resolve_run_cap(C1_PROVENANCE, INITIAL_MAX_UPDATES) == 20_000


@pytest.mark.parametrize("carried", [20_001, 25_000, 40_000])
def test_c1_resume_above_the_cap_is_refused_never_clamped(carried):
    with pytest.raises(BudgetPolicyViolation, match="refused rather than reinterpreted"):
        resolve_run_cap(C1_PROVENANCE, 20_000,
                        {"global_update": carried, "cap": EXTENDED_MAX_UPDATES})


@pytest.mark.parametrize("carried", [0, 500, 19_500, 20_000])
def test_c1_resume_within_the_cap_is_allowed(carried):
    assert resolve_run_cap(
        C1_PROVENANCE, 20_000, {"global_update": carried, "cap": 20_000}
    ) == 20_000


def test_historical_continuation_is_untouched_by_c1():
    from unmark.stage1.selection import ValidationPoint

    worse = {"FULL": .9, "P50": .9, "P100": .9, "STRIP_ALL": .9}
    best = {"FULL": .1, "P50": .1, "P100": .1, "STRIP_ALL": .1}
    result = resolve_budget(RunResult(
        provenance=HISTORICAL_PROVENANCE,
        points=[ValidationPoint(0, worse, .9), ValidationPoint(20_000, best, .1)],
        cap=20_000,
    ))
    assert result.cap == EXTENDED_MAX_UPDATES and result.continued is True


def test_no_cli_flag_can_relax_the_c1_ceiling():
    declared = set()
    for node in ast.walk(ast.parse(source("scripts/stage1_runner.py"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("-"):
                        declared.add(argument.value)
    for forbidden in ("--max-updates", "--cap", "--budget", "--fusion", "--fusion-id",
                      "--scale", "--epsilon", "--lambda-grid", "--objective"):
        assert forbidden not in declared, f"the runner declares {forbidden}"


def test_the_c1_plan_is_exactly_the_inherited_locked_values():
    (planned,) = v2_scf_schedule()
    assert planned.stage == V2_SCF_STAGE == "v2_scf"
    assert planned.learning_rate == V2_SCF_LEARNING_RATE == 1e-4
    assert planned.r == V2_SCF_R == 1.0
    assert planned.seed == V2_SCF_RUN_SEED == 36930
    assert adapter_init_seed(planned.seed) == 51800
    assert CORRUPTION_SEED == 35422
    assert lambdas_for_r(planned.r) == (1.0, 1.0)
    assert V2_SCF_MAX_UPDATES == INITIAL_MAX_UPDATES == 20_000


def test_c1_does_not_grow_the_locked_historical_campaign():
    from unmark.stage1.protocol import TOTAL_NOMINAL_RUNS
    from unmark.stage1.selection import total_planned_runs

    assert len(v2_scf_schedule()) == 1
    assert total_planned_runs() == TOTAL_NOMINAL_RUNS == 11


def test_stage1_selection_is_unchanged_and_scale_never_reaches_it():
    """No scale quantity may reach the held-out criterion or the selection rule.

    `v2_scf_schedule` legitimately lives in `selection.py`, exactly as
    `v2_gc_schedule` does -- a run PLAN is not a selection input. What must not
    appear is any scale telemetry, any fusion dispatch, or the calibration itself.
    """
    for module in ("unmark/stage1/validation.py", "unmark/stage1/selection.py"):
        body = source(module)
        for forbidden in ("scale_calibrated", "fusion_id", "scale_diagnostics",
                          "intervention_ratio", "scale_factor", "postcal",
                          "scf_scale", "scale_telemetry", "AdapterConfig"):
            assert forbidden not in body, f"{module} reaches {forbidden}"

    # And the selection rule itself still reads exactly the held-out distances.
    from unmark.stage1.selection import ValidationPoint, select_checkpoint
    import dataclasses

    assert tuple(f.name for f in dataclasses.fields(ValidationPoint)) == (
        "update", "distances", "d_clean"
    )
    points = [
        ValidationPoint(0, {"FULL": .5, "P50": .5, "P100": .5, "STRIP_ALL": .9}, .5),
        ValidationPoint(500, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .4}, .1),
        ValidationPoint(1000, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .8}, .1),
    ]
    assert select_checkpoint(points).update == 500


# ===========================================================================
# F. W&B -- candidate-separated, observational only
# ===========================================================================
def test_each_candidate_resolves_its_own_wandb_project():
    assert wandb_project_for_stage(V2_SCF_STAGE) == "UNMARK-v2-C1-SCF-Stage1"
    assert wandb_project_for_stage(V2_GC_STAGE) == "UNMARK-v2-C3-GC-Stage1"
    for stage in HISTORICAL_STAGES:
        assert wandb_project_for_stage(stage) == "unmark-stage1", stage


def test_c1_and_c3_cannot_share_a_project_or_a_run_id():
    assert C1.wandb_project != C3.wandb_project
    monitor = _monitor()
    c1_key = monitor.project_for_event({"stage": V2_SCF_STAGE})
    c3_key = monitor.project_for_event({"stage": V2_GC_STAGE})
    assert c1_key != c3_key


def _monitor():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_monitor_under_test", REPO / "scripts/stage1_wandb_monitor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_monitor_resolves_the_project_from_the_register():
    monitor = _monitor()
    assert monitor.project_for_event({"stage": V2_SCF_STAGE}) == "UNMARK-v2-C1-SCF-Stage1"
    assert monitor.project_for_event({"stage": V2_GC_STAGE}) == "UNMARK-v2-C3-GC-Stage1"
    assert monitor.project_for_event({"stage": "lr_pilot"}) == "unmark-stage1"
    # An explicit --project always wins; an unknown stage degrades, never raises.
    assert monitor.project_for_event({"stage": V2_SCF_STAGE}, "explicit") == "explicit"
    assert monitor.project_for_event({"stage": "unregistered"}) == monitor.DEFAULT_PROJECT
    assert monitor.project_for_event({}) == monitor.DEFAULT_PROJECT


def test_the_wandb_config_carries_the_full_candidate_identity():
    monitor = _monitor()
    for key in ("candidate_id", "objective_id", "fusion_id", "repository_head",
                "seed", "init_seed", "corruption_seed", "lr", "batch_size",
                "hard_max_updates", "encoder_checkpoint", "encoder_revision",
                "corpus_manifest_digest", "inventory_sha256"):
        assert key in monitor.SAFE_CONFIG_KEYS, f"{key} cannot reach the W&B config"


def test_the_run_start_event_emits_that_identity():
    """The config is a whitelist FILTER; production must actually emit the values."""
    node = next(
        n for n in ast.walk(ast.parse(source(EXECUTE)))
        if isinstance(n, ast.FunctionDef) and n.name == "execute_stage"
    )
    emits = [
        ast.unparse(n) for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "emit"
    ]
    run_start = next(e for e in emits if "run_start" in e)
    for key in ("candidate_id", "objective_identity.to_dict()",
                "candidate.fusion.to_dict()", "hard_max_updates", "init_seed"):
        assert key in run_start, f"run_start does not emit {key}"


def test_wandb_is_observational_only():
    """No W&B name may reach a loss, a seed, a budget or a selection."""
    for module in ("unmark/stage1/trainer.py", "unmark/stage1/objective.py",
                   "unmark/stage1/objective_grid.py", "unmark/stage1/selection.py",
                   "unmark/stage1/validation.py", "unmark/modeling/adapter.py"):
        body = source(module).lower()
        assert "wandb" not in body, f"{module} reaches wandb"
    # The scientific package never imports it at all.
    for path in (REPO / "unmark").rglob("*.py"):
        assert "import wandb" not in path.read_text(encoding="utf-8"), path


def test_the_project_name_is_not_scientific_identity():
    """It is not in provenance, so it cannot change what a checkpoint means."""
    assert "wandb" not in str(HISTORICAL_PROVENANCE.to_dict()).lower()
    assert "wandb" not in str(payload_for(C1_PROVENANCE)["provenance"]).lower()


# ===========================================================================
# G. Smoke dispatch
# ===========================================================================
def test_the_smoke_builds_the_candidates_own_adapter():
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "candidate_for_stage(stage)" in body
    assert "fusion_id=candidate.fusion.fusion_id" in body, (
        "the smoke builds a hard-wired adapter instead of the candidate's"
    )
    # It resolves the candidate BEFORE building the model.
    assert body.index("candidate_for_stage") < body.index("build_objective")


def test_the_smoke_cli_offers_every_registered_candidate():
    runner = source("scripts/stage1_runner.py")
    assert "choices=[c.stage for c in CANDIDATES]" in runner
    assert {c.stage for c in CANDIDATES} >= {
        V2_SCF_STAGE, V2_GC_STAGE, *HISTORICAL_STAGES
    }


def test_the_smoke_requires_grid_terms_only_where_there_is_a_grid_term():
    """C1 must NOT be asked for an L_grid it does not have."""
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "if candidate.objective.lambda_grid is not None:" in body


def test_no_candidate_cross_wiring():
    """Each registered stage maps to exactly one (objective, fusion) pair."""
    seen = {}
    for candidate in CANDIDATES:
        assert candidate.stage not in seen
        seen[candidate.stage] = candidate.identity
    assert seen[V2_SCF_STAGE] == (HISTORICAL_OBJECTIVE_ID, SCALE_CALIBRATED_FUSION_ID)
    assert seen[V2_GC_STAGE] == ("grid-consistency-v1", HISTORICAL_FUSION_ID)


def test_the_builder_refuses_an_adapter_that_is_not_the_declared_fusion():
    body = body_without_docstring(function("build_candidate_objective", EXECUTE))
    assert "candidate.fusion.fusion_id" in body
    assert "Stage1ContractViolation" in body


# ===========================================================================
# H. Torch-gated numerics
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


def adapters(d: int = 16):
    """A historical and a scale-calibrated adapter from the SAME init seed."""
    from unmark.stage1.initialisation import fresh_adapter

    return (
        fresh_adapter(d, 51800, HISTORICAL_FUSION_ID),
        fresh_adapter(d, 51800, SCALE_CALIBRATED_FUSION_ID),
    )


@requires_torch
def test_runtime_the_scale_formula_matches_a_hand_computed_value():
    """Expected values derived analytically, not read off the implementation."""
    from unmark.modeling.adapter import scale_calibrated_fusion

    # ||e|| = 5, ||f|| = 27.9 -- the D3/D4 ratio, to two decimals.
    e = torch.tensor([[[3.0, 4.0]]])
    f = torch.tensor([[[0.0, 27.9]]])
    out = scale_calibrated_fusion(f, e)
    assert out.shape == f.shape
    assert out.norm(dim=-1).item() == pytest.approx(5.0, abs=1e-5)
    assert out[0, 0, 1].item() == pytest.approx(5.0, abs=1e-5)
    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


@requires_torch
def test_runtime_every_token_is_calibrated_to_its_own_base_norm():
    from unmark.modeling.adapter import scale_calibrated_fusion

    e = torch.randn(3, 7, 16)
    f = torch.randn(3, 7, 16) * 25.0
    out = scale_calibrated_fusion(f, e)
    assert torch.allclose(out.norm(dim=-1), e.norm(dim=-1), atol=1e-5)


@requires_torch
def test_runtime_the_norm_is_not_batch_or_sequence_level():
    """Changing OTHER tokens must not change this token's calibrated output."""
    from unmark.modeling.adapter import scale_calibrated_fusion

    e = torch.randn(2, 5, 16)
    f = torch.randn(2, 5, 16) * 10.0
    baseline = scale_calibrated_fusion(f, e)

    moved_e, moved_f = e.clone(), f.clone()
    moved_e[1, 3] *= 100.0          # a different example AND a different position
    moved_f[1, 3] *= 100.0
    after = scale_calibrated_fusion(moved_f, moved_e)
    assert torch.allclose(baseline[0], after[0], atol=0, rtol=0), "batch-level norm"
    assert torch.allclose(baseline[1, :3], after[1, :3], atol=0, rtol=0), "sequence-level"


@requires_torch
def test_runtime_the_scale_has_a_trailing_singleton_dimension():
    from unmark.modeling.adapter import scale_calibrated_fusion

    e = torch.randn(2, 5, 16)
    f = torch.randn(2, 5, 16)
    # Broadcasting [B, L, 1] over [B, L, d] is what makes it a per-token scalar.
    manual = (e.norm(dim=-1, keepdim=True) / f.norm(dim=-1, keepdim=True).clamp(min=1e-8)) * f
    assert e.norm(dim=-1, keepdim=True).shape == (2, 5, 1)
    assert torch.allclose(scale_calibrated_fusion(f, e), manual, atol=0, rtol=0)


@requires_torch
def test_runtime_a_zero_norm_f_stays_finite_and_deterministic():
    from unmark.modeling.adapter import scale_calibrated_fusion

    e = torch.randn(2, 4, 16)
    f = torch.zeros(2, 4, 16)
    out = scale_calibrated_fusion(f, e)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(out, scale_calibrated_fusion(f, e))


@requires_torch
def test_runtime_gradients_flow_through_f_and_through_the_scale():
    from unmark.modeling.adapter import scale_calibrated_fusion

    e = torch.randn(2, 4, 16, requires_grad=True)
    f = torch.randn(2, 4, 16, requires_grad=True) * 1.0
    f.retain_grad()
    scale_calibrated_fusion(f, e).sum().backward()
    assert f.grad is not None and float(f.grad.abs().sum()) > 0
    assert e.grad is not None and float(e.grad.abs().sum()) > 0, (
        "the scale was detached: no gradient reached e through ||e||"
    )


@requires_torch
def test_runtime_z_is_the_gated_mixture_of_the_calibrated_f():
    from unmark.modeling.adapter import convex_combination, scale_calibrated_fusion

    e = torch.randn(2, 4, 16)
    f = torch.randn(2, 4, 16) * 20.0
    g = torch.full((2, 4, 16), 0.03)
    z = convex_combination(g, scale_calibrated_fusion(f, e), e)
    expected = g * scale_calibrated_fusion(f, e) + (1.0 - g) * e
    assert torch.allclose(z, expected, atol=0, rtol=0)


@requires_torch
def test_runtime_calibration_shrinks_the_intervention_at_an_equal_gate():
    """The C1 hypothesis, measured: same `g`, same `f`, smaller ||z-e||/||e||."""
    from unmark.modeling.adapter import convex_combination, scale_calibrated_fusion

    torch.manual_seed(0)
    e = torch.randn(4, 8, 16)
    f = torch.randn(4, 8, 16) * 20.0          # the ~20x scale gap D3/D4 measured
    g = torch.full_like(e, 0.03)
    historical = convex_combination(g, f, e)
    calibrated = convex_combination(g, scale_calibrated_fusion(f, e), e)
    ratio = lambda z: ((z - e).norm(dim=-1) / e.norm(dim=-1)).mean().item()
    assert ratio(calibrated) < ratio(historical)


@requires_torch
def test_runtime_the_gate_is_identical_under_both_fusions():
    """Same weights, same input, same `q` -- the calibration is downstream of g."""
    historical, calibrated = adapters()
    batch = synthetic_batch(base_len=5)
    args = (torch.randn(2, 5, 16), batch["clean_tone_ids"], batch["clean_tone_mask"],
            batch["clean_letter_ids"], batch["clean_letter_mask"])
    with torch.no_grad():
        assert torch.equal(historical.gate_values(*args), calibrated.gate_values(*args))


@requires_torch
def test_runtime_the_two_adapters_start_from_identical_weights():
    """Same init seed, same parameter set: a paired comparison."""
    from unmark.stage1.initialisation import trainable_state, trainable_state_hash

    historical, calibrated = adapters()
    assert trainable_state_hash(trainable_state(historical)) == (
        trainable_state_hash(trainable_state(calibrated))
    )
    assert sorted(historical.state_dict()) == sorted(calibrated.state_dict())


@requires_torch
def test_runtime_c1_adds_zero_trainable_parameters():
    historical, calibrated = adapters()
    assert sum(p.numel() for p in calibrated.parameters() if p.requires_grad) == (
        sum(p.numel() for p in historical.parameters() if p.requires_grad)
    )
    assert "_scale_diagnostics" not in calibrated.state_dict()


@requires_torch
def test_runtime_the_historical_adapter_output_is_bit_unchanged():
    """The regression that matters most: C1 must not perturb historical UNMARK."""
    historical, calibrated = adapters()
    batch = synthetic_batch(base_len=5)
    e = torch.randn(2, 5, 16)
    args = (e, batch["clean_tone_ids"], batch["clean_tone_mask"],
            batch["clean_letter_ids"], batch["clean_letter_mask"])
    with torch.no_grad():
        z_hist = historical(*args)
        z_cal = calibrated(*args)
    # Reproduce the historical equation independently from the same weights.
    from unmark.modeling.adapter import convex_combination

    with torch.no_grad():
        t = historical.tone_channel(batch["clean_tone_ids"], batch["clean_tone_mask"])
        l = historical.letter_channel(batch["clean_letter_ids"], batch["clean_letter_mask"])
        q = torch.cat([e, t, l], dim=-1)
        f = historical.layer_norm(historical.fusion(q))
        g = torch.sigmoid(historical.gate(q))
        expected = convex_combination(g, f, e)
    assert torch.allclose(z_hist, expected, atol=0, rtol=0)
    assert not torch.allclose(z_hist, z_cal), "the two fusions produced the same z"


@requires_torch
def test_runtime_diagnostics_do_not_change_the_computed_z():
    """Recording is read-only: `z` is bit-identical with and without it."""
    _, calibrated = adapters()
    batch = synthetic_batch(base_len=5)
    e = torch.randn(2, 5, 16)
    args = (e, batch["clean_tone_ids"], batch["clean_tone_mask"],
            batch["clean_letter_ids"], batch["clean_letter_mask"])
    with torch.no_grad():
        first = calibrated(*args)
        recorded = calibrated.scale_diagnostics()
        second = calibrated(*args)
    assert torch.allclose(first, second, atol=0, rtol=0)
    assert set(recorded) == {
        "scale_factor", "postcal_f_over_e", "intervention_ratio", "gate_mean"
    }
    assert all(v == v for v in recorded.values()), "a diagnostic is NaN"
    assert recorded["postcal_f_over_e"] == pytest.approx(1.0, abs=1e-4), (
        "after calibration ||f|| should match ||e|| per token"
    )


@requires_torch
def test_runtime_a_historical_adapter_reports_no_scale_diagnostics():
    historical, _ = adapters()
    batch = synthetic_batch(base_len=5)
    with torch.no_grad():
        historical(torch.randn(2, 5, 16), batch["clean_tone_ids"],
                   batch["clean_tone_mask"], batch["clean_letter_ids"],
                   batch["clean_letter_mask"])
    assert historical.scale_diagnostics() == {}
    from unmark.stage1.trainer import scale_telemetry

    assert scale_telemetry(historical) == {}


@requires_torch
def test_runtime_c1_trains_the_historical_objective_with_three_forwards():
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.execute import build_candidate_objective
    from unmark.stage1.objective import Stage1Objective
    from unmark.stage1.objective_grid import GridConsistencyObjective
    from unmark.stage1.contracts import ObjectiveWeights

    encoder, _, _, _ = build_stack()
    _, calibrated = adapters()
    objective = build_candidate_objective(
        UnmarkEncoder(encoder, calibrated), C1,
        ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
    )
    assert isinstance(objective, Stage1Objective)
    assert not isinstance(objective, GridConsistencyObjective), (
        "C1 must NOT get the grid objective"
    )

    calls = []
    original = encoder.forward
    encoder.forward = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        result = objective(synthetic_batch(base_len=5))
    finally:
        encoder.forward = original
    assert len(calls) == 3, f"C1 ran {len(calls)} encoder forwards, not 3"
    assert not hasattr(result, "loss_grid"), "C1 produced a grid term"
    assert torch.isfinite(result.loss) and torch.isfinite(result.loss_align)
    assert torch.isfinite(result.loss_clean)


@requires_torch
def test_runtime_a_c1_checkpoint_reconstructs_into_the_c1_fusion():
    """Round trip: identity -> correct architecture -> strict load."""
    from unmark.stage1.reconstruct import reconstruct_adapter, require_loadable_as

    _, calibrated = adapters()
    payload = payload_for(C1_PROVENANCE, adapter_state=calibrated.state_dict())

    rebuilt = reconstruct_adapter(payload, hidden_size=16)
    assert rebuilt.config.fusion_id == SCALE_CALIBRATED_FUSION_ID
    assert rebuilt.config.is_scale_calibrated
    for name, tensor in calibrated.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[name], tensor), name

    # The same tensors are shape-compatible with the historical adapter, which is
    # exactly why the guard has to exist.
    historical, _ = adapters()
    historical.load_state_dict(payload["adapter_state"], strict=True)
    with pytest.raises(Stage1ContractViolation):
        require_loadable_as(payload, HISTORICAL_FUSION_ID)


@requires_torch
def test_runtime_a_legacy_checkpoint_reconstructs_into_the_historical_fusion():
    from unmark.stage1.reconstruct import reconstruct_adapter

    historical, _ = adapters()
    payload = payload_for(HISTORICAL_PROVENANCE, adapter_state=historical.state_dict())
    del payload["provenance"]["fusion"]
    rebuilt = reconstruct_adapter(payload, hidden_size=16)
    assert rebuilt.config.fusion_id == HISTORICAL_FUSION_ID
    assert not rebuilt.config.is_scale_calibrated
