"""Stage-1 device ownership -- structural half (Audit 030 §Y).

The fourth real smoke failed with

    RuntimeError: Expected all tensors to be on the same device,
    but got index is on cpu, different from other tensors on cuda:0

The contract, as implemented, is that **the objective never moves its inputs**:
`reference_representation` and `adapted_representation` hand `input_ids` straight
to the encoder. `collate_stage1_batch` builds CPU tensors. So whoever assembles a
batch must place it on the model's device -- and nothing did, while the
measurement tool moved the encoder to CUDA.

The boundary is now explicit and shared: `evaluate` and `train_run` both route the
collated batch through `batch_to_device(..., module_device(objective))`.

**Torch-free**, so it runs in the ML-free venv on every run. Its companion
`test_stage1_device_contract_runtime.py` proves the same contract by executing it,
and the cross-device half of that file needs a CUDA host.

Kept in a separate file deliberately: a module-level `importorskip` would skip
these structural checks too, which is exactly how an earlier repair lost its
torch-free coverage (§V).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Structural -- torch-free, so a regression is caught in this venv
# ---------------------------------------------------------------------------
def batch_assembly_calls(module_path: str, function: str) -> list[str]:
    tree = ast.parse((REPO / module_path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == function)
    return [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(fn) if isinstance(n, ast.Call)]


@pytest.mark.parametrize("module_path,function", [
    ("unmark/stage1/validation.py", "evaluate"),
    ("unmark/stage1/trainer.py", "train_run"),
    ("unmark/stage1/execute.py", "smoke_check"),
])
def test_every_batch_assembler_routes_through_the_shared_boundary(module_path, function):
    """Both places that collate must also place. One boundary, no exceptions."""
    called = batch_assembly_calls(module_path, function)
    assert "collate_stage1_batch" in called, function
    assert "batch_to_device" in called, (
        f"{function} collates a batch but never moves it to the model's device; "
        "an objective on an accelerator would be handed CPU ids"
    )
    assert "module_device" in called, (
        f"{function} must derive the device from the module, not hard-code one"
    )


def test_no_hard_coded_physical_gpu_anywhere_in_stage1():
    """No physical GPU index, no `.cuda()`, no global default device.

    Refined for D-S1B-015: `unmark/stage1/device.py` now legitimately names the
    **logical** device `torch.device("cuda")`, which is precisely what honours
    `CUDA_VISIBLE_DEVICES`. What must never appear is a *physical* index
    (`cuda:0`), a `.cuda()` call, a global default-device mutation, or code that
    rewrites the visibility environment out from under the operator.
    """
    for path in sorted((REPO / "unmark/stage1").glob("*.py")):
        code = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("#")
        )
        assert not re.search(r"cuda:\d", code), f"{path.name} names a physical GPU index"
        assert ".cuda(" not in code, path.name
        assert "set_default_device" not in code, path.name
        assert 'environ["CUDA_VISIBLE_DEVICES"]' not in code, path.name
        assert "get_device_name(0)" not in code, f"{path.name} hardcodes device 0"


def test_only_the_device_resolver_names_cuda():
    """One authoritative resolver. No scattered `.cuda()` or ad-hoc selection."""
    named = sorted(
        path.name for path in (REPO / "unmark/stage1").glob("*.py")
        if '"cuda"' in path.read_text(encoding="utf-8")
    )
    assert named == ["device.py"], named


def test_the_scientific_cli_offers_no_device_or_determinism_override():
    """No `--cpu`, `--device`, `--allow-tf32`, `--init-seed`, ... (D-S1B-015/016)."""
    source = (REPO / "scripts/stage1_runner.py").read_text(encoding="utf-8")
    for forbidden in ("--cpu", "--device", "--allow-cpu", "--no-cuda",
                      "--allow-nondeterministic", "--allow-tf32", "--init-seed",
                      "--reuse-adapter", "--skip-device-check",
                      "--skip-execution-fingerprint"):
        assert forbidden not in source, forbidden


def test_the_objective_does_not_move_its_own_inputs():
    """Pins the half of the contract that makes the caller responsible.

    If a future change makes the objective move inputs internally, the boundary
    above becomes redundant and this contract needs restating deliberately.
    """
    source = (REPO / "unmark/stage1/objective.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("reference_representation", "adapted_representation"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        moved = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) == "to"]
        assert moved == [], f"{name} moves tensors; the device contract changed"


def test_the_measurement_tool_moves_the_model_and_nothing_else():
    """It owns model placement; the shared boundary owns batch placement."""
    called = batch_assembly_calls("scripts/stage1_pretrain_measurements.py",
                                  "validation_timing")
    assert "to" in called, "validation_timing must place the encoder on the device"
    assert "collate_stage1_batch" not in called, (
        "the tool must not assemble batches itself; evaluate owns that"
    )
