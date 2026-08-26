"""Every global name the Stage-1 production path loads must actually resolve.

Audit 031 B1: `execute_stage()` called `objective_cls(...)`, a name bound
nowhere -- not in the function, not at module scope. The first real `lr-pilot`
would have raised `NameError` after preflight, device resolution, corpus load,
backbone construction and adapter initialisation had all succeeded, and before
`train_run()` was ever reached.

The suite was fully green. Worse, `test_stage1_run_independence.py` asserted
that `"objective_cls"` appeared among the calls inside the nominal-run loop, so
the test suite actively *required* the broken name.

That happened because the existing tests inspect the AST -- they can see that a
call named `objective_cls` occurs, and cannot see that the name is unbound. So
this file does not look at source text at all. It compiles each production
module, walks every code object, and resolves each `LOAD_GLOBAL` against the
module's real namespace plus builtins.

This catches the whole defect class rather than the one instance: any
refactor that leaves a stale global name behind fails here, in the ML-free
environment, without loading torch or running a single update.
"""

from __future__ import annotations

import builtins
import dis
import importlib
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

PRODUCTION_MODULES = sorted(
    "unmark.stage1." + path.stem
    for path in (REPOSITORY / "unmark" / "stage1").glob("*.py")
    if path.stem != "__init__"
)


def _code_objects(code: types.CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _code_objects(constant)


def _unresolved_globals(module) -> list[str]:
    """Global names loaded by `module` that resolve to nothing."""
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    compiled = compile(source, module.__file__, "exec")
    unresolved = []
    for code in _code_objects(compiled):
        for instruction in dis.get_instructions(code):
            if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                continue
            name = instruction.argval
            if not isinstance(name, str):
                continue
            if hasattr(module, name) or hasattr(builtins, name):
                continue
            unresolved.append(f"{code.co_name}: {name}")
    return sorted(set(unresolved))


@pytest.mark.parametrize("module_name", PRODUCTION_MODULES)
def test_every_global_name_in_a_stage1_module_resolves(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        # A module that imports torch at module scope cannot be introspected in
        # the ML-free local venv. `execute`, `trainer`, `selection`, `artifact`
        # and `checkpoint` -- everything the repaired paths live in -- import
        # torch lazily and are therefore covered here.
        pytest.skip(f"{module_name} needs {error.name}, absent in the ML-free venv")
    unresolved = _unresolved_globals(module)
    assert not unresolved, (
        f"{module_name} loads global name(s) that are bound nowhere: {unresolved}. "
        "This is the Audit 031 B1 defect class: the call is visible to an AST "
        "test and raises NameError at runtime."
    )


def test_the_original_defect_is_detected_by_this_test():
    """Mutation check: reintroduce the exact bug and require a failure.

    A regression test that cannot fail is not evidence. This rebuilds the
    pre-repair form of the line and confirms the detector flags it -- so the
    guarantee above is demonstrated rather than asserted.
    """
    module = importlib.import_module("unmark.stage1.execute")
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    broken = source.replace(
        "objective = Stage1Objective(unmark_encoder, provenance.weights)",
        "objective = objective_cls(unmark_encoder, provenance.weights)",
    )
    assert broken != source, "the repaired construction line was not found"

    compiled = compile(broken, module.__file__, "exec")
    loaded = {
        instruction.argval
        for code in _code_objects(compiled)
        for instruction in dis.get_instructions(code)
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }
    assert "objective_cls" in loaded
    assert not hasattr(module, "objective_cls"), (
        "objective_cls resolves to nothing at module scope -- which is exactly "
        "why the pre-repair line raised NameError"
    )


def test_the_objective_is_constructed_from_a_locally_bound_class():
    """`Stage1Objective` is a real local binding in `execute_stage`, not a global.

    It is imported inside the function (torch stays lazy), so it must appear in
    the code object's *locals*. If a future edit moves the import without moving
    the call, the previous test catches it; this one states the intended shape.
    """
    module = importlib.import_module("unmark.stage1.execute")
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    compiled = compile(source, module.__file__, "exec")
    stage = next(
        code for code in _code_objects(compiled) if code.co_name == "execute_stage"
    )
    assert "Stage1Objective" in stage.co_varnames, (
        "execute_stage must bind Stage1Objective locally via its lazy import"
    )
    assert "objective_cls" not in stage.co_varnames
    assert "objective_cls" not in stage.co_names
