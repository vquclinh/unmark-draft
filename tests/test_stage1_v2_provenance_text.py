"""Human-readable provenance text must be TRUE for the candidate it describes.

PREFLIGHT 5A found two descriptions that were accurate when written and became
false when a later candidate joined the path they sit on:

1. the first-screen artifact note hard-coded C3's "token-grid consistency term",
   on a branch shared by C1 (`v2_scf`), C2 (`v2_grd`) and C3 (`v2_gc`);
2. the generic runner banner printed "one continuation, then STOP" for every
   command, including the three V2 candidates, whose budgets set
   `allows_precommitted_continuation = False`.

Neither was a behaviour defect -- enforcement was correct in both cases -- but a
persisted artifact and an operator-facing banner that misdescribe the run are
provenance defects, and this file is the regression that would have caught them.

**Nothing here trains, and none of it needs torch.** It reads the source the
production path actually uses and drives the real CLI banner.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.candidates import (
    CANDIDATES,
    HISTORICAL_STAGES,
    candidate_for_stage,
)
from unmark.stage1.execute import FIRST_SCREEN_STAGES
from unmark.stage1.protocol import EXTENDED_MAX_UPDATES, INITIAL_MAX_UPDATES

REPO = pathlib.Path(__file__).resolve().parents[1]
EXECUTE = "unmark/stage1/execute.py"
RUNNER = "scripts/stage1_runner.py"

CANDIDATE_SPECIFIC_VOCABULARY = (
    # C3's loss
    "token-grid", "token grid", "grid consistency", "grid-consistency", "l_grid",
    # C1's architecture
    "scale-calibrated", "scale calibrated", "calibrat",
    # C2's loss
    "relational", "gram", "first_token", "distillation", "l_grd",
)
"""Vocabulary that names ONE candidate. None of it may appear in prose on a
shared path: the machine-readable `objective` field is where a candidate's
identity belongs, and it is authoritative."""


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def first_screen_artifact_note() -> str:
    """The exact note literal the shared first-screen branch persists.

    Read out of the AST rather than the file text, so this tracks the value that
    actually reaches the artifact instead of any comment that happens to resemble
    it.
    """
    tree = ast.parse(source(EXECUTE))
    execute_stage = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_stage"
    )
    # `branch.body` only: `ast.walk` on the `If` would also descend into its
    # `orelse`, which is the `final_main` leg and carries its own, different note.
    branch = first_screen_branch(execute_stage)
    notes = [
        ast.literal_eval(value)
        for statement in branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "note"
    ]
    assert len(notes) == 1, f"expected exactly one first-screen note, found {len(notes)}"
    return notes[0]


def first_screen_branch(execute_stage: ast.FunctionDef) -> ast.If:
    """The `elif stage in FIRST_SCREEN_STAGES:` node inside `execute_stage`."""
    return next(
        n for n in ast.walk(execute_stage)
        if isinstance(n, ast.If) and "FIRST_SCREEN_STAGES" in ast.unparse(n.test)
    )


# ===========================================================================
# 1. The persisted first-screen note must be true for every candidate
# ===========================================================================
def test_the_first_screen_branch_is_shared_by_all_three_candidates():
    """The premise: one branch, three candidates. That is why prose must be neutral."""
    assert set(FIRST_SCREEN_STAGES) == {"v2_scf", "v2_grd", "v2_gc"}
    for stage in FIRST_SCREEN_STAGES:
        assert candidate_for_stage(stage).budget.hard_max_updates == 20_000


def test_the_first_screen_note_names_no_single_candidate():
    """The PREFLIGHT 5A defect: the note claimed a token-grid consistency term.

    That is C3's loss. C1 changes the fusion and C2 relates FIRST_TOKEN geometry,
    so the note was false for two of the three candidates that persist it.
    """
    note = first_screen_artifact_note().lower()
    for term in CANDIDATE_SPECIFIC_VOCABULARY:
        assert term not in note, (
            f"the shared first-screen note claims {term!r}, which is true of only "
            "one candidate. A candidate's identity belongs in the machine-readable "
            "`objective` field, not in prose on a shared branch."
        )


def test_the_first_screen_note_states_what_the_stage_actually_does():
    note = first_screen_artifact_note().lower()
    assert "one" in note and "first-screen" in note
    assert "no between-candidate selection" in note, (
        "the note must say that this stage performs no selection between candidates"
    )
    assert "held-out" in note, (
        "the note must say that the checkpoint WITHIN the run is still chosen by "
        "the locked Stage-1 held-out rule"
    )


def test_the_authoritative_objective_field_is_untouched():
    """The note is prose; `objective` is the record. Both must still be written."""
    tree = ast.parse(source(EXECUTE))
    execute_stage = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_stage"
    )
    branch = first_screen_branch(execute_stage)
    body = "\n".join(ast.unparse(statement) for statement in branch.body)
    assert "'objective': objective_identity.to_dict()" in body
    assert "'budget': candidate.budget.to_dict()" in body
    assert "'selected': screened.selected.to_dict()" in body


# ===========================================================================
# 2. The runner banner must not promise a continuation a candidate cannot take
# ===========================================================================
def runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_runner_under_test", REPO / RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stage", ["v2_scf", "v2_grd", "v2_gc"])
def test_the_banner_never_promises_a_continuation_to_a_hard_capped_candidate(stage):
    """The PREFLIGHT 5A defect: the generic banner said 'one continuation'."""
    line = runner_module().budget_banner(stage)
    assert "one continuation" not in line, (
        f"the banner promises {stage} a continuation its budget forbids"
    )
    assert str(EXTENDED_MAX_UPDATES) not in line.replace(
        f"NO continuation to {EXTENDED_MAX_UPDATES}", ""
    ), "the banner offers the 40k leg to a hard-capped candidate"
    assert "HARD CAP" in line
    assert str(INITIAL_MAX_UPDATES) in line


@pytest.mark.parametrize("stage", list(HISTORICAL_STAGES))
def test_the_historical_banner_line_is_unchanged(stage):
    """Historical commands keep their exact historical description."""
    assert runner_module().budget_banner(stage) == (
        f"  budget         : {INITIAL_MAX_UPDATES} updates, one continuation, then STOP"
    )


def test_a_command_with_no_registered_stage_keeps_the_generic_line():
    """`prepare-corpus` runs no stage; its banner must not change."""
    assert runner_module().budget_banner(None) == (
        f"  budget         : {INITIAL_MAX_UPDATES} updates, one continuation, then STOP"
    )


def test_the_banner_line_is_derived_from_the_register_not_a_restated_constant():
    module = runner_module()
    for candidate in CANDIDATES:
        line = module.budget_banner(candidate.stage)
        if candidate.budget.allows_precommitted_continuation:
            assert "one continuation" in line, candidate.stage
        else:
            assert "one continuation" not in line, candidate.stage
            assert str(candidate.budget.hard_max_updates) in line, candidate.stage
            assert candidate.budget.policy in line, candidate.stage


def test_the_banner_does_not_hard_code_the_continuation_sentence_in_main():
    """`main` must print the derived line, not a literal."""
    tree = ast.parse(source(RUNNER))
    main = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    body = ast.unparse(main)
    assert "budget_banner(banner_stage(args))" in body
    assert "one continuation" not in body, (
        "main still holds a hard-coded continuation sentence"
    )


@pytest.mark.parametrize("command,stage", [
    ("v2-scf", "v2_scf"), ("v2-grd", "v2_grd"), ("v2-gc", "v2_gc"),
    ("lr-pilot", "lr_pilot"), ("r-phase1", "r_phase1"), ("final-main", "final_main"),
])
def test_the_cli_command_maps_to_its_registered_stage(command, stage):
    import argparse

    module = runner_module()
    assert module.banner_stage(argparse.Namespace(command=command)) == stage


def test_smoke_resolves_the_candidate_it_was_asked_for():
    import argparse

    module = runner_module()
    args = argparse.Namespace(command="smoke", candidate="v2_grd")
    assert module.banner_stage(args) == "v2_grd"
    assert "HARD CAP" in module.budget_banner(module.banner_stage(args))


def test_prepare_corpus_resolves_no_stage():
    import argparse

    module = runner_module()
    assert module.banner_stage(argparse.Namespace(command="prepare-corpus")) is None


# -- and the real CLI, end to end -------------------------------------------
@pytest.mark.parametrize("command,continuation_allowed", [
    ("v2-scf", False), ("v2-grd", False), ("v2-gc", False),
    ("lr-pilot", True), ("r-phase1", True), ("final-main", True),
])
def test_the_real_cli_banner_is_truthful(command, continuation_allowed, tmp_path, capsys):
    """Drives `main()` itself. It fails closed on a missing corpus BEFORE any
    model, corpus or optimizer is touched, so the banner is all that runs."""
    module = runner_module()
    argv = [
        command,
        "--prepared-corpus", str(tmp_path / "absent"),
        "--output-dir", str(tmp_path / "out"),
        "--cache-root", str(tmp_path / "cache"),
    ]
    if command in ("r-phase1", "final-main"):
        argv += ["--lr-artifact", str(tmp_path / "lr.json")]
    if command == "final-main":
        argv += ["--r-artifact", str(tmp_path / "r.json")]

    exit_code = module.main(argv)
    printed = capsys.readouterr().out
    assert exit_code == 2, "the command must fail closed on a missing corpus"

    budget_lines = [ln for ln in printed.splitlines() if ln.startswith("  budget")]
    assert budget_lines, "no budget line was printed"
    if continuation_allowed:
        assert any("one continuation" in ln for ln in budget_lines), command
    else:
        assert not any("one continuation" in ln for ln in budget_lines), (
            f"{command} printed a continuation promise: {budget_lines}"
        )
        assert any("HARD CAP" in ln for ln in budget_lines), command


# ===========================================================================
# 3. The facts the text describes must themselves be unchanged
# ===========================================================================
@pytest.mark.parametrize("stage", ["v2_scf", "v2_grd", "v2_gc"])
def test_the_v2_budget_registry_is_unchanged(stage):
    budget = candidate_for_stage(stage).budget
    assert budget.hard_max_updates == 20_000
    assert budget.allows_precommitted_continuation is False
    assert budget.policy == "first_screen_hard_cap"


def test_the_historical_budget_registry_is_unchanged():
    for stage in HISTORICAL_STAGES:
        budget = candidate_for_stage(stage).budget
        assert budget.hard_max_updates is None
        assert budget.allows_precommitted_continuation is True


@pytest.mark.parametrize("stage,project", [
    ("v2_scf", "UNMARK-v2-C1-SCF-Stage1"),
    ("v2_grd", "UNMARK-v2-C2-GRD-Stage1"),
    ("v2_gc", "UNMARK-v2-C3-GC-Stage1"),
])
def test_the_wandb_projects_are_unchanged(stage, project):
    assert candidate_for_stage(stage).wandb_project == project


def test_the_historical_wandb_project_is_unchanged():
    for stage in HISTORICAL_STAGES:
        assert candidate_for_stage(stage).wandb_project == "unmark-stage1"


def test_each_candidate_still_declares_its_own_objective_and_fusion():
    """The identity the artifact's `objective` field records, unchanged."""
    expected = {
        "v2_scf": ("align-clean-pooled-v1", "scale-calibrated-fusion-v1"),
        "v2_grd": ("geometry-relational-distillation-v1", "historical-fusion-v1"),
        "v2_gc": ("grid-consistency-v1", "historical-fusion-v1"),
        "lr_pilot": ("align-clean-pooled-v1", "historical-fusion-v1"),
        "r_phase1": ("align-clean-pooled-v1", "historical-fusion-v1"),
        "final_main": ("align-clean-pooled-v1", "historical-fusion-v1"),
    }
    assert {c.stage for c in CANDIDATES} == set(expected)
    for stage, identity in expected.items():
        assert candidate_for_stage(stage).identity == identity, stage
