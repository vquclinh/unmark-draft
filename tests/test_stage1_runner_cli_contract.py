"""The runner's public CLI must match what its handlers actually read (§AA).

The fifth-smoke runner died like this, 0.09 s in, after argparse had already
accepted the documented command:

    completion_dir=Path(args.completion_dir) if args.completion_dir else None,
    AttributeError: 'Namespace' object has no attribute 'completion_dir'

`smoke` declared only `--prepared-corpus`, while `run_smoke` read
`args.completion_dir` -- the same pair every other corpus consumer takes. The
existing suite could not catch it: a parser that parses and a handler that reads
are two halves nothing was comparing.

These tests compare them, for **every** subcommand, so the defect class is closed
rather than this one instance. No model, no torch, no network.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.stage1_runner import build_parser, run_smoke

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (REPO / "scripts/stage1_runner.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

COMMANDS = ("prepare-corpus", "lr-pilot", "r-phase1", "final-main", "smoke")
HANDLERS = {
    "prepare-corpus": "run_prepare_corpus",
    "lr-pilot": "run_lr_pilot",
    "r-phase1": "run_r_phase1",
    "final-main": "run_final_main",
    "smoke": "run_smoke",
}


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------
def subparser(command: str) -> argparse.ArgumentParser:
    parser = build_parser()
    action = next(a for a in parser._actions
                  if isinstance(a, argparse._SubParsersAction))
    return action.choices[command]


def minimal_argv(command: str) -> list[str]:
    """A command line satisfying exactly the parser's own required options."""
    argv = [command]
    for action in subparser(command)._actions:
        if action.required and action.option_strings:
            argv += [action.option_strings[0], "placeholder"]
    return argv


def functions() -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)}


def args_attributes(name: str, seen: set[str] | None = None) -> set[str]:
    """Every `args.<attr>` the handler reads, following helpers that take `args`.

    Transitive on purpose: `run_lr_pilot` reads little itself but hands `args` to
    `_verified_corpus` and `_execute`, and an attribute missing there fails just
    as hard.
    """
    seen = seen if seen is not None else set()
    if name in seen:
        return set()
    seen.add(name)
    table = functions()
    node = table.get(name)
    if node is None:
        return set()

    found = {
        n.attr for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "args"
    }
    # `getattr(args, "x", ...)` counts too -- and is banned below, but if it ever
    # appeared it would still be a real read.
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        if getattr(call.func, "id", None) == "getattr" and len(call.args) >= 2:
            target, attribute = call.args[0], call.args[1]
            if isinstance(target, ast.Name) and target.id == "args" \
               and isinstance(attribute, ast.Constant):
                found.add(attribute.value)
        # follow helpers that are handed `args`
        callee = getattr(call.func, "id", None)
        if callee in table and any(
            isinstance(a, ast.Name) and a.id == "args" for a in call.args
        ):
            found |= args_attributes(callee, seen)
    return found


# ---------------------------------------------------------------------------
# 1. The defect class: parser and handler must agree, for every subcommand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("command", COMMANDS)
def test_every_attribute_the_handler_reads_exists_in_the_namespace(command):
    """Fails on current HEAD for `smoke`: `completion_dir` is read, never declared."""
    namespace = subparser(command).parse_args(minimal_argv(command)[1:])
    available = set(vars(namespace))
    required = args_attributes(HANDLERS[command])
    missing = sorted(required - available - {"command"})
    assert missing == [], (
        f"`{command}` handler reads {missing} but the parser never creates it; "
        "the command parses and then dies with AttributeError"
    )


@pytest.mark.parametrize("command", COMMANDS)
def test_no_handler_papers_over_a_missing_field_with_getattr(command):
    """`getattr(args, "completion_dir", None)` would hide the mismatch again.

    Worse, for this field it would silently swap an explicit COMPLETE marker for
    an inferred one, which is an integrity gate, not a convenience.
    """
    node = functions()[HANDLERS[command]]
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        if getattr(call.func, "id", None) == "getattr" and call.args:
            target = call.args[0]
            assert not (isinstance(target, ast.Name) and target.id == "args"), (
                f"{HANDLERS[command]} uses getattr on args; declare the option instead"
            )


# ---------------------------------------------------------------------------
# 2. The smoke CLI specifically
# ---------------------------------------------------------------------------
def test_smoke_help_reflects_the_accepted_contract():
    """Help text and accepted options must be the same set."""
    declared = {
        option
        for action in subparser("smoke")._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert declared == {"--prepared-corpus", "--completion-dir", "--revision",
                        "--repository-head"}, declared

    help_text = subparser("smoke").format_help()
    for option in declared:
        assert option in help_text, f"{option} is accepted but undocumented"


def test_the_exact_formerly_failing_command_now_parses():
    """The corrected fifth-smoke invocation, verbatim."""
    namespace = build_parser().parse_args([
        "smoke",
        "--prepared-corpus", "/content/unmark-stage1-prepared-aa49785eadcb",
        "--revision", "01daacda68afe13d83023d16ec647239e344a1e6",
        "--repository-head", "a5da53805498a12ed64ffa28a6a13232dc8e4b1b",
    ])
    assert namespace.completion_dir is None, "the field must exist explicitly"
    assert hasattr(namespace, "prepared_corpus")


def test_the_first_attempt_shape_is_now_accepted_too():
    """The orchestrator's original `--completion-dir` form was not wrong in kind.

    Its two roots are exactly the real deployment: payload on local disk,
    COMPLETE marker on Drive.
    """
    namespace = build_parser().parse_args([
        "smoke",
        "--prepared-corpus", "/content/unmark-stage1-prepared-aa49785eadcb",
        "--completion-dir",
        "/content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb",
        "--revision", "01daacda68afe13d83023d16ec647239e344a1e6",
    ])
    assert namespace.completion_dir.endswith("aa49785eadcb")


# ---------------------------------------------------------------------------
# 3. Parser -> handler, executed. `smoke_check` is stubbed; nothing else is.
# ---------------------------------------------------------------------------
@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_smoke_check(**kwargs):
        seen.update(kwargs)
        return 0

    import unmark.stage1.execute as execute_module

    monkeypatch.setattr(execute_module, "smoke_check", fake_smoke_check)
    return seen


def test_run_smoke_reaches_smoke_check_without_completion_dir(captured):
    """The failing path: it must now get all the way through."""
    namespace = build_parser().parse_args(
        ["smoke", "--prepared-corpus", "/tmp/prepared"]
    )
    assert run_smoke(namespace) == 0
    assert captured["completion_dir"] is None
    assert str(captured["prepared_corpus"]) == "/tmp/prepared"


def test_run_smoke_forwards_an_explicit_completion_dir(captured):
    namespace = build_parser().parse_args([
        "smoke", "--prepared-corpus", "/tmp/prepared",
        "--completion-dir", "/drive/complete",
    ])
    assert run_smoke(namespace) == 0
    assert str(captured["completion_dir"]) == "/drive/complete"


def test_omitting_completion_dir_still_verifies_against_the_co_located_marker():
    """`None` infers `<prepared-corpus>/_checkpoint`; it never skips the check.

    Read off `smoke_check` itself, so the fallback cannot become a bypass without
    this failing.
    """
    from unmark.stage1 import execute as execute_module

    node = next(n for n in ast.walk(ast.parse(inspect.getsource(execute_module)))
                if isinstance(n, ast.FunctionDef) and n.name == "smoke_check")
    body = ast.unparse(node)
    assert "_checkpoint" in body, "the fallback location must be explicit"
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(node) if isinstance(n, ast.Call)}
    assert "verify_prepared_corpus" in called, (
        "smoke must verify the prepared corpus whether or not --completion-dir "
        "was supplied (Audit 030 F1)"
    )


def test_smoke_declares_no_scientific_override_flags():
    """No flag may reach a locked constant, and none may reach official TEST."""
    declared = {
        option
        for action in subparser("smoke")._actions
        for option in action.option_strings
    }
    for forbidden in ("--batch-size", "--max-length", "--seed", "--lr", "--r",
                      "--eval-every", "--precision", "--conditions", "--pi-strip"):
        assert forbidden not in declared, forbidden
    for option in declared:
        assert "test" not in option.lower(), f"{option} could reach official TEST"


# ---------------------------------------------------------------------------
# 4. The smoke's no-update guarantee, from the call graph (not from help prose)
# ---------------------------------------------------------------------------
def test_smoke_cannot_construct_an_optimizer_or_step_one():
    from unmark.stage1 import execute as execute_module

    node = next(n for n in ast.walk(ast.parse(inspect.getsource(execute_module)))
                if isinstance(n, ast.FunctionDef) and n.name == "smoke_check")
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(node) if isinstance(n, ast.Call)}
    for forbidden in ("backward", "step", "zero_grad", "build_optimizer", "AdamW",
                      "train_run", "save_training_checkpoint"):
        assert forbidden not in called, f"smoke_check reaches {forbidden}()"
    assert "no_grad" in called, "the single forward must be under no_grad"
    # and it really does use the real model and the real preparation
    for required in ("build_objective", "verify_model_contract", "prepare_example",
                     "collate_stage1_batch", "verify_scientific_inputs"):
        assert required in called, f"smoke_check must still call {required}()"


def test_run_smoke_itself_reaches_only_smoke_check():
    node = functions()["run_smoke"]
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(node) if isinstance(n, ast.Call)}
    assert "smoke_check" in called
    assert "execute_stage" not in called and "train_run" not in called
