"""The measurement tool must really measure (Audit 030 §U).

§T found that `--validation` loaded the real encoder and then timed only
condition *preparation* — zero forwards, no `no_grad`, meaningless GPU numbers.
These tests make that defect unable to return.

Everything here uses a **tiny injected model**. No PhoBERT is downloaded, no
real weights are loaded, and nothing trains.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the injected-model half needs torch")

from scripts.stage1_pretrain_measurements import (  # noqa: E402
    InstrumentedObjective,
    parameter_digest,
    validation_failures,
    validation_timing,
)
from unmark.stage1.protocol import BATCH_SIZE, VALIDATION_CONDITIONS  # noqa: E402


# ---------------------------------------------------------------------------
# A tiny stand-in with the exact surface `validation.evaluate` uses
# ---------------------------------------------------------------------------
class TinyObjective(torch.nn.Module):
    """Real `nn.Module`, real forwards, ~40 parameters. No encoder anywhere."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(dim, dim)      # stands in for the frozen half
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.adapter = torch.nn.Linear(dim, dim)      # the trainable half

    # -- the surface evaluate() calls --
    def reference_representation(self, ids, mask, special):
        return torch.tanh(self.encoder(ids.float()))

    def adapted_representation(self, ids, mask, special, *channels):
        return torch.tanh(self.adapter(self.encoder(ids.float())))

    @property
    def unmark_encoder(self):
        return self


def batch_like(n: int, dim: int = 4):
    ones = torch.ones(n, dim)
    return {k: ones for k in (
        "reference_input_ids", "reference_attention_mask", "reference_special_tokens_mask",
        "base_input_ids", "base_attention_mask", "base_special_tokens_mask",
        "corrupt_tone_ids", "corrupt_tone_mask", "corrupt_letter_ids", "corrupt_letter_mask",
        "clean_tone_ids", "clean_tone_mask", "clean_letter_ids", "clean_letter_mask",
    )}


DEV_EXAMPLES = 10


@pytest.fixture
def evaluated():
    """Run the REAL `validation.evaluate` over a tiny injected objective."""
    from unmark.stage1.validation import evaluate

    objective = TinyObjective()
    proxy = InstrumentedObjective(objective)
    prepared = {c: list(range(DEV_EXAMPLES)) for c in VALIDATION_CONDITIONS}

    import unmark.stage1.validation as validation_module

    original = validation_module.collate_stage1_batch if hasattr(
        validation_module, "collate_stage1_batch") else None
    import unmark.stage1.data as data_module

    saved = data_module.collate_stage1_batch
    data_module.collate_stage1_batch = lambda items, pad: batch_like(len(items))
    try:
        point = evaluate(proxy, prepared, pad_token_id=1, batch_size=4)
    finally:
        data_module.collate_stage1_batch = saved
        if original is not None:
            validation_module.collate_stage1_batch = original
    return proxy, point


# ---------------------------------------------------------------------------
# 1-5: it reaches the authoritative evaluator and really forwards
# ---------------------------------------------------------------------------
def test_the_tool_calls_the_authoritative_evaluate(evaluated):
    """Not a second evaluator: `validation.evaluate` itself must be reached."""
    source = inspect.getsource(validation_timing)
    assert "from unmark.stage1.validation import" in source
    assert "evaluate(" in source
    tree = ast.parse(inspect.getsource(
        sys.modules["scripts.stage1_pretrain_measurements"]))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "validation_timing")
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "evaluate" in called, "evaluate must be CALLED, not merely imported"


def test_real_forwards_happen(evaluated):
    proxy, _ = evaluated
    assert proxy.reference_calls > 0
    assert proxy.adapted_calls > 0


def test_all_four_conditions_execute(evaluated):
    _, point = evaluated
    assert sorted(point.distances) == sorted(VALIDATION_CONDITIONS)


def test_every_dev_example_is_consumed(evaluated):
    """4 conditions x ceil(10/4) batches of reference; adapted adds d_clean on FULL."""
    proxy, _ = evaluated
    batches = -(-DEV_EXAMPLES // 4)
    assert proxy.reference_calls == batches * len(VALIDATION_CONDITIONS)
    assert proxy.adapted_calls == batches * len(VALIDATION_CONDITIONS) + batches


def test_no_grad_is_active_during_every_forward(evaluated):
    proxy, _ = evaluated
    assert proxy.grad_enabled_during_forward is False
    assert proxy.outputs_requiring_grad == 0


# ---------------------------------------------------------------------------
# 6-10: the no-update boundary
# ---------------------------------------------------------------------------
def test_no_optimizer_backward_or_step_is_reachable_from_measurement():
    source = pathlib.Path("scripts/stage1_pretrain_measurements.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for forbidden in ("backward", "step", "zero_grad", "AdamW", "build_optimizer",
                      "train_run", "execute_stage"):
        assert forbidden not in called, f"measurement must not reach {forbidden}()"


def test_parameters_are_identical_across_a_measurement(evaluated):
    objective = TinyObjective()
    before = parameter_digest(objective)
    proxy = InstrumentedObjective(objective)
    proxy.reference_representation(torch.ones(2, 4), None, None)
    after = parameter_digest(objective)
    assert before["trainable_sha256"] == after["trainable_sha256"]
    assert before["frozen_encoder_sha256"] == after["frozen_encoder_sha256"]


def test_the_digest_separates_trainable_from_frozen():
    digest = parameter_digest(TinyObjective())
    assert digest["trainable_tensors"] > 0 and digest["frozen_tensors"] > 0
    assert digest["trainable_sha256"] != digest["frozen_encoder_sha256"]


def test_a_mutated_trainable_parameter_is_detected():
    objective = TinyObjective()
    before = parameter_digest(objective)
    with torch.no_grad():
        objective.adapter.weight.add_(1.0)
    assert parameter_digest(objective)["trainable_sha256"] != before["trainable_sha256"]


def test_a_mutated_frozen_encoder_is_detected():
    objective = TinyObjective()
    before = parameter_digest(objective)
    with torch.no_grad():
        objective.encoder.weight.add_(1.0)
    assert parameter_digest(objective)["frozen_encoder_sha256"] != before["frozen_encoder_sha256"]


# ---------------------------------------------------------------------------
# 11-12: the report cannot claim success without the work
# ---------------------------------------------------------------------------
def healthy_report() -> dict:
    return {
        "forward_passes": 40,
        "conditions_executed": sorted(VALIDATION_CONDITIONS),
        "environment": {"device": "cuda:0", "cuda_synchronized_around_timing": True},
        "no_update_boundary": {
            "optimizer_constructed": False, "backward_calls": 0, "optimizer_steps": 0,
            "grad_enabled_during_forward": False, "outputs_requiring_grad": 0,
            "parameters_identical": True,
        },
    }


def test_a_healthy_report_passes():
    assert validation_failures(healthy_report()) == []


def test_zero_forwards_cannot_masquerade_as_validation():
    report = healthy_report()
    report["forward_passes"] = 0
    failures = validation_failures(report)
    assert any("no forward pass" in f for f in failures)


@pytest.mark.parametrize("dropped", list(VALIDATION_CONDITIONS))
def test_a_missing_condition_fails(dropped):
    report = healthy_report()
    report["conditions_executed"] = [c for c in VALIDATION_CONDITIONS if c != dropped]
    assert any("not executed" in f for f in validation_failures(report))


def test_parameter_mutation_fails_closed():
    report = healthy_report()
    report["no_update_boundary"]["parameters_identical"] = False
    assert any("PARAMETERS CHANGED" in f for f in validation_failures(report))


def test_grad_enabled_fails_closed():
    report = healthy_report()
    report["no_update_boundary"]["grad_enabled_during_forward"] = True
    assert any("no_grad was not active" in f for f in validation_failures(report))


def test_an_optimizer_or_step_fails_closed():
    for field in ("optimizer_constructed", "optimizer_steps", "backward_calls"):
        report = healthy_report()
        report["no_update_boundary"][field] = 1 if field != "optimizer_constructed" else True
        assert validation_failures(report), field


def test_unsynchronized_cuda_timing_fails_closed():
    report = healthy_report()
    report["environment"]["cuda_synchronized_around_timing"] = False
    assert any("not synchronized" in f for f in validation_failures(report))


def test_cpu_measurements_do_not_require_cuda_sync():
    report = healthy_report()
    report["environment"] = {"device": "cpu", "cuda_synchronized_around_timing": False}
    assert validation_failures(report) == []


def test_the_synchronize_hook_is_invoked_around_each_forward():
    """On CUDA the proxy must synchronise, or wall-clock would be fiction."""
    calls = {"n": 0}

    def fake_synchronize():
        calls["n"] += 1

    proxy = InstrumentedObjective(TinyObjective(), synchronize=fake_synchronize)
    proxy.reference_representation(torch.ones(2, 4), None, None)
    assert calls["n"] == 2, "expected a sync immediately before and after"
