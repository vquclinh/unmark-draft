"""The V2-GC 20 000-update HARD CAP, on the real execution path.

Audit 065 BLOCKER 1: V2-GC was not capped. The generic precommitted rule
(`selection.budget_decision`) was applied to every run, so a candidate whose best
held-out checkpoint landed exactly on 20 000 was promoted to the 40 000 leg by
`trainer.resolve_budget` and re-entered `train_run` from `execute_stage`.
`V2_GC_MAX_UPDATES` was consulted by nothing -- an import, a `print()` and one
assertion -- so the suite reported budget coverage it did not have.

**These tests exercise the decision functions the production path actually
calls**, not constants and not mocks:

* `trainer.resolve_budget` -- the promotion decision itself;
* `trainer.resolve_run_cap` -- the structural gate `train_run` runs before its
  loop;
* `execute.continuation_permitted` -- the gate on the second `train_run` call;
* `candidates.ScreeningBudget` -- the policy those three consult.

Every one of them fails against the pre-repair implementation: `resolve_budget`
returned `cap = 40000` for CASE 2, and the other three did not exist.

Two AST tests tie the pure functions to the real control flow, so a future edit
cannot leave the decisions correct while the loop stops calling them.

**Nothing here trains.** No optimizer, no backward, no update.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.candidates import (
    CANDIDATES,
    FIRST_SCREEN_HARD_CAP,
    HISTORICAL_STAGES,
    PRECOMMITTED_CONTINUATION,
    BudgetPolicyViolation,
    ScreeningBudget,
    budget_for_identity,
    candidate_for_stage,
)
from unmark.stage1.contracts import (
    GRID_CONSISTENCY_OBJECTIVE,
    HISTORICAL_FUSION,
    HISTORICAL_OBJECTIVE,
    Stage1ContractViolation,
)
from unmark.stage1.execute import continuation_permitted
from unmark.stage1.protocol import (
    EXTENDED_MAX_UPDATES,
    GRID_CONSISTENCY_OBJECTIVE_ID,
    HISTORICAL_FUSION_ID,
    HISTORICAL_OBJECTIVE_ID,
    INITIAL_MAX_UPDATES,
    SCALE_CALIBRATED_FUSION_ID,
    V2_GC_MAX_UPDATES,
    V2_GC_STAGE,
)
from unmark.stage1.selection import ValidationPoint
from unmark.stage1.trainer import (
    RunProvenance,
    RunResult,
    resolve_budget,
    resolve_run_cap,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

V2_GC_CANDIDATE = candidate_for_stage(V2_GC_STAGE)
HISTORICAL_CANDIDATE = candidate_for_stage(HISTORICAL_STAGES[0])


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=36930, init_seed=51800, corruption_seed=35422,
        learning_rate=1e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


HISTORICAL_PROVENANCE = provenance()
V2_GC_PROVENANCE = provenance(objective=GRID_CONSISTENCY_OBJECTIVE)


def trajectory(best_update: int) -> list[ValidationPoint]:
    """A validation history whose single best point is at `best_update`."""
    worse = {"FULL": .9, "P50": .9, "P100": .9, "STRIP_ALL": .9}
    best = {"FULL": .1, "P50": .1, "P100": .1, "STRIP_ALL": .1}
    points = [ValidationPoint(0, worse, .9)]
    if best_update != 0:
        points.append(ValidationPoint(best_update, best, .1))
    return points


def function(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


# ===========================================================================
# The policy object itself
# ===========================================================================
def test_the_v2_gc_budget_is_a_hard_cap_at_twenty_thousand():
    budget = budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    assert budget is FIRST_SCREEN_HARD_CAP
    assert budget.hard_max_updates == V2_GC_MAX_UPDATES == INITIAL_MAX_UPDATES == 20_000
    assert budget.allows_precommitted_continuation is False
    assert budget.is_hard_capped


def test_the_historical_budget_is_the_untouched_precommitted_rule():
    budget = budget_for_identity(HISTORICAL_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    assert budget is PRECOMMITTED_CONTINUATION
    assert budget.hard_max_updates is None
    assert budget.allows_precommitted_continuation is True


def test_v2_gc_max_updates_is_load_bearing_not_decorative():
    """Audit 065's false-assurance finding: the constant now IS the policy.

    Asserted by identity of the value the enforcement path reads, not by
    comparing two constants to each other.
    """
    assert budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, HISTORICAL_FUSION_ID).hard_max_updates == (
        V2_GC_MAX_UPDATES
    )
    with pytest.raises(BudgetPolicyViolation):
        budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, HISTORICAL_FUSION_ID).require_cap(
            V2_GC_MAX_UPDATES + 1, what="probe"
        )


def test_a_self_contradicting_budget_cannot_be_constructed():
    with pytest.raises(Stage1ContractViolation):
        ScreeningBudget(
            hard_max_updates=20_000,
            allows_precommitted_continuation=True,
            policy="incoherent",
        )


def test_an_unregistered_identity_has_no_budget_and_fails_closed():
    with pytest.raises(Stage1ContractViolation, match="no\n?.*registered screening budget"):
        budget_for_identity("v2-something-nobody-registered", HISTORICAL_FUSION_ID)


def test_an_unregistered_objective_fusion_PAIR_fails_closed():
    """The key is the PAIR: two registered halves that were never combined."""
    with pytest.raises(Stage1ContractViolation):
        budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, SCALE_CALIBRATED_FUSION_ID)


def test_an_unregistered_stage_fails_closed():
    with pytest.raises(Stage1ContractViolation, match="unknown Stage-1 stage"):
        candidate_for_stage("v2_gc_but_longer")


# ===========================================================================
# CASE 1 - V2-GC best checkpoint inside the budget: stop normally, no 40k leg
# ===========================================================================
@pytest.mark.parametrize("best", [0, 500, 3500, 19_500])
def test_case1_v2_gc_inside_the_budget_stops_normally(best):
    result = resolve_budget(
        RunResult(provenance=V2_GC_PROVENANCE, points=trajectory(best), cap=20_000)
    )
    assert result.cap == 20_000, "the cap must not move"
    assert result.continued is False
    assert result.hard_capped is False, "it stopped inside the budget, not against it"
    assert result.budget_limited is False
    assert continuation_permitted(V2_GC_CANDIDATE, 20_000, result.cap) is False


# ===========================================================================
# CASE 2 - V2-GC best checkpoint EXACTLY at 20k: hard stop, no second train_run
# ===========================================================================
def test_case2_v2_gc_at_the_boundary_hard_stops():
    """The exact condition that promoted V2-GC to 40 000 before the repair."""
    result = resolve_budget(
        RunResult(provenance=V2_GC_PROVENANCE, points=trajectory(20_000), cap=20_000)
    )
    assert result.cap == 20_000, (
        "cap was raised to the 40k leg: this is the Audit 065 BLOCKER 1 regression"
    )
    assert result.cap != EXTENDED_MAX_UPDATES
    assert result.continued is False, "V2-GC must not enter the continuation leg"
    assert result.hard_capped is True, "the artifact must say WHY it stopped"
    assert result.budget_limited is True


def test_case2_the_continuation_gate_refuses_the_second_train_run():
    """`execute_stage` asks this before calling `train_run` a second time."""
    assert continuation_permitted(V2_GC_CANDIDATE, INITIAL_MAX_UPDATES,
                                  EXTENDED_MAX_UPDATES) is False
    # Even if some future edit let `result.cap` reach 40 000, the gate still refuses.
    for leg_cap in (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES):
        for result_cap in (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES):
            assert continuation_permitted(V2_GC_CANDIDATE, leg_cap, result_cap) is False


def test_case2_the_result_artifact_is_honest_about_the_hard_cap():
    result = resolve_budget(
        RunResult(provenance=V2_GC_PROVENANCE, points=trajectory(20_000), cap=20_000)
    )
    payload = result.to_dict()
    assert payload["cap"] == 20_000
    assert payload["continued_past_initial_budget"] is False
    assert payload["stopped_at_hard_cap"] is True
    assert payload["budget_policy"] == {
        "policy": "first_screen_hard_cap",
        "hard_max_updates": 20_000,
        "allows_precommitted_continuation": False,
    }


# ===========================================================================
# CASE 3 - historical behaviour is completely unchanged
# ===========================================================================
def test_case3_historical_at_the_boundary_still_continues_to_forty_thousand():
    result = resolve_budget(
        RunResult(provenance=HISTORICAL_PROVENANCE, points=trajectory(20_000), cap=20_000)
    )
    assert result.cap == EXTENDED_MAX_UPDATES, "historical continuation was broken"
    assert result.continued is True
    assert result.hard_capped is False
    assert result.budget_limited is False
    assert continuation_permitted(HISTORICAL_CANDIDATE, INITIAL_MAX_UPDATES,
                                  result.cap) is True


def test_case3_historical_inside_the_budget_is_unchanged():
    result = resolve_budget(
        RunResult(provenance=HISTORICAL_PROVENANCE, points=trajectory(3_500), cap=20_000)
    )
    assert (result.cap, result.continued, result.budget_limited) == (20_000, False, False)
    assert result.hard_capped is False


def test_case3_historical_at_the_extended_cap_is_budget_limited_not_extended():
    """No 60k/80k extension. The locked terminal state, unchanged."""
    result = resolve_budget(
        RunResult(provenance=HISTORICAL_PROVENANCE, points=trajectory(40_000),
                  cap=EXTENDED_MAX_UPDATES)
    )
    assert result.cap == EXTENDED_MAX_UPDATES
    assert result.budget_limited is True
    assert result.continued is False
    assert continuation_permitted(HISTORICAL_CANDIDATE, EXTENDED_MAX_UPDATES,
                                  result.cap) is False


def test_case3_historical_runs_accept_the_forty_thousand_cap():
    assert resolve_run_cap(HISTORICAL_PROVENANCE, EXTENDED_MAX_UPDATES) == (
        EXTENDED_MAX_UPDATES
    )
    assert resolve_run_cap(
        HISTORICAL_PROVENANCE, EXTENDED_MAX_UPDATES,
        {"global_update": 25_000, "cap": EXTENDED_MAX_UPDATES},
    ) == EXTENDED_MAX_UPDATES


# ===========================================================================
# CASE 4 - V2-GC resume from a legal <= 20k checkpoint cannot exceed 20k
# ===========================================================================
@pytest.mark.parametrize("carried_update", [0, 500, 12_000, 19_500, 20_000])
def test_case4_v2_gc_resume_within_the_cap_is_allowed(carried_update):
    assert resolve_run_cap(
        V2_GC_PROVENANCE, 20_000,
        {"global_update": carried_update, "cap": 20_000},
    ) == 20_000


def test_case4_v2_gc_resume_cannot_be_handed_the_forty_thousand_leg():
    with pytest.raises(BudgetPolicyViolation, match="HARD-CAPPED"):
        resolve_run_cap(
            V2_GC_PROVENANCE, EXTENDED_MAX_UPDATES,
            {"global_update": 20_000, "cap": 20_000},
        )


def test_case4_a_fresh_v2_gc_run_cannot_be_handed_the_forty_thousand_leg():
    with pytest.raises(BudgetPolicyViolation, match="HARD-CAPPED"):
        resolve_run_cap(V2_GC_PROVENANCE, EXTENDED_MAX_UPDATES)
    assert resolve_run_cap(V2_GC_PROVENANCE, INITIAL_MAX_UPDATES) == 20_000


# ===========================================================================
# CASE 5 - a V2-GC artifact past the ceiling is REFUSED, never normalised
# ===========================================================================
@pytest.mark.parametrize("carried_update", [20_001, 25_000, 40_000])
def test_case5_v2_gc_resume_above_the_cap_fails_closed(carried_update):
    with pytest.raises(BudgetPolicyViolation, match="refused rather than reinterpreted"):
        resolve_run_cap(
            V2_GC_PROVENANCE, 20_000,
            {"global_update": carried_update, "cap": EXTENDED_MAX_UPDATES},
        )


def test_case5_a_carried_forty_thousand_cap_is_refused_for_v2_gc():
    with pytest.raises(BudgetPolicyViolation):
        resolve_run_cap(
            V2_GC_PROVENANCE, 20_000,
            {"global_update": 20_000, "cap": EXTENDED_MAX_UPDATES},
        )


def test_case5_an_over_cap_payload_is_not_silently_clamped():
    """It must RAISE. Returning 20 000 would reinterpret executed work as absent."""
    try:
        resolve_run_cap(
            V2_GC_PROVENANCE, 20_000,
            {"global_update": 30_000, "cap": EXTENDED_MAX_UPDATES},
        )
    except BudgetPolicyViolation:
        return
    pytest.fail("an over-cap V2-GC payload was accepted or clamped instead of refused")


# ===========================================================================
# CASE 6 - no CLI or environment switch can relax the hard maximum
# ===========================================================================
def test_case6_no_cli_flag_can_raise_the_v2_gc_ceiling():
    runner = (REPO / "scripts/stage1_runner.py").read_text(encoding="utf-8")
    declared = set()
    for node in ast.walk(ast.parse(runner)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("-"):
                        declared.add(argument.value)
    for forbidden in ("--max-updates", "--max_updates", "--cap", "--budget",
                      "--extend", "--extended", "--continue", "--hard-cap",
                      "--updates", "--steps"):
        assert forbidden not in declared, f"the runner declares {forbidden}"


def test_case6_no_environment_variable_can_relax_the_budget():
    for module in ("unmark/stage1/candidates.py", "unmark/stage1/trainer.py",
                   "unmark/stage1/execute.py"):
        body = (REPO / module).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                pytest.fail(f"{module} reads the environment on the budget path")


def test_case6_the_budget_is_frozen_and_cannot_be_mutated_in_place():
    budget = budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, HISTORICAL_FUSION_ID)
    with pytest.raises(Exception):
        budget.hard_max_updates = 40_000  # type: ignore[misc]
    assert budget_for_identity(GRID_CONSISTENCY_OBJECTIVE_ID, HISTORICAL_FUSION_ID).hard_max_updates == 20_000


def test_case6_execute_stage_takes_no_cap_or_budget_argument():
    node = function("unmark/stage1/execute.py", "execute_stage")
    parameters = {a.arg for a in node.args.args + node.args.kwonlyargs}
    for forbidden in ("cap", "max_updates", "budget", "hard_max_updates"):
        assert forbidden not in parameters, f"execute_stage accepts {forbidden}"


# ===========================================================================
# The pure decisions are wired into the real control flow
# ===========================================================================
def test_train_run_resolves_the_cap_before_its_update_loop():
    """The gate must run BEFORE `while global_update < cap`, or it guards nothing."""
    node = function("unmark/stage1/trainer.py", "train_run")
    gates = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_run_cap"
    ]
    assert gates, "train_run does not resolve the candidate's screening budget"
    loops = [n for n in ast.walk(node) if isinstance(n, ast.While)]
    assert loops, "train_run has no update loop"
    assert min(g.lineno for g in gates) < min(w.lineno for w in loops), (
        "the budget gate runs after the loop has already begun"
    )
    # And it is handed this run's own provenance and the resume payload.
    passed = ast.unparse(gates[0])
    assert "provenance" in passed and "cap" in passed and "resume" in passed


def test_the_second_train_run_is_gated_on_the_candidate_budget():
    """`execute_stage` calls `train_run` twice; the second is the 40k leg."""
    node = function("unmark/stage1/execute.py", "execute_stage")
    runs = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "train_run"
    ]
    assert len(runs) == 2, f"expected exactly two train_run calls, found {len(runs)}"

    gated = [
        n for n in ast.walk(node)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and getattr(n.test.func, "id", None) == "continuation_permitted"
    ]
    assert len(gated) == 1, "the continuation leg is not gated on continuation_permitted"
    inner = [
        n for n in ast.walk(gated[0])
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "train_run"
    ]
    assert len(inner) == 1, "the second train_run is not inside the gate"
    assert ast.unparse(gated[0].test).startswith("continuation_permitted(candidate,")


def test_execute_stage_resolves_the_budget_before_starting_a_run():
    node = function("unmark/stage1/execute.py", "execute_stage")
    gates = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_run_cap"
    ]
    assert gates, "execute_stage does not apply the candidate's screening budget"


def test_resolve_budget_consults_the_candidate_before_promoting():
    node = function("unmark/stage1/trainer.py", "resolve_budget")
    body = ast.unparse(node)
    assert "allows_precommitted_continuation" in body, (
        "resolve_budget promotes without asking whether this candidate may continue"
    )


# ===========================================================================
# Candidate isolation: the design must accept C1/C2 without weakening history
# ===========================================================================
def test_every_registered_stage_has_an_objective_and_a_budget():
    for candidate in CANDIDATES:
        assert candidate.objective.objective_id
        assert candidate.fusion.fusion_id
        assert candidate.budget.policy
        assert budget_for_identity(*candidate.identity) is candidate.budget


def test_every_historical_stage_keeps_the_historical_objective_and_budget():
    for stage in HISTORICAL_STAGES:
        candidate = candidate_for_stage(stage)
        assert candidate.objective is HISTORICAL_OBJECTIVE, stage
        assert candidate.fusion is HISTORICAL_FUSION, stage
        assert candidate.budget is PRECOMMITTED_CONTINUATION, stage


def test_nothing_assumes_a_post_hoc_candidate_is_the_grid_one():
    """C1/C2 must be addable by registering a candidate, not by editing dispatch.

    The budget owner is keyed on the objective id and the stage resolver on the
    stage name; neither hard-codes the grid candidate.
    """
    body = (REPO / "unmark/stage1/candidates.py").read_text(encoding="utf-8")
    module = ast.parse(body)
    functions = {
        n.name: ast.unparse(n)
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef)
    }
    for name in ("candidate_for_stage", "objective_for_stage", "budget_for_identity"):
        assert "GRID_CONSISTENCY" not in functions[name], (
            f"{name} special-cases the grid candidate instead of reading the register"
        )
        assert "V2_GC" not in functions[name], name


def test_the_trainer_never_names_a_candidate():
    """The budget reaches the trainer through provenance, not a stage string."""
    body = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    module = ast.parse(body)
    for node in ast.walk(module):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in ("v2_gc", "v2_scf"), (
                "trainer.py hard-codes a candidate stage name"
            )


# ===========================================================================
# Torch-gated: the real train_run refuses an over-cap resume
# ===========================================================================
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


@requires_torch
def test_runtime_train_run_refuses_an_over_cap_v2_gc_resume():
    """End to end through the real `train_run`, before any update is taken."""
    from unmark.stage1.trainer import train_run

    with pytest.raises(BudgetPolicyViolation):
        train_run(
            objective=None,
            provenance=V2_GC_PROVENANCE,
            train_chunks={"c#0": "x"},
            tokenizer=None,
            corruption_policy=None,
            truncation=None,
            evaluate_fn=lambda update: None,
            pad_token_id=1,
            cap=EXTENDED_MAX_UPDATES,
        )
