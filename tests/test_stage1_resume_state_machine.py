"""The 20k->40k budget state machine, through the REAL production seams.

Two blockers lived here, and both survived a fully green suite because no test
had ever driven `train_run`'s resume path with writer-emitted state:

* **Audit 031 B2 / Audit 032 B1** -- `checkpoint_payload` serialised validation
  points with `to_dict()` (four keys, including the derived `score`) and
  `train_run` restored them with `ValidationPoint(**p)` (three constructor
  fields), so every real `--resume` raised
  `TypeError: ValidationPoint.__init__() got an unexpected keyword argument 'score'`.

* **Audit 031 B3 / Audit 032 B2** -- the checkpoint persisted `cap` and nothing
  read it. `execute_stage` passed `cap=INITIAL_MAX_UPDATES` on every call,
  resume included, so a checkpoint written during the 40k continuation came
  back under a 20k budget. Either `budget_decision` raised
  `SelectionViolation: selected update ... exceeds the cap 20000`, or -- worse,
  because it is silent -- 40k of work was recorded as a complete 20k run.

This file is the **torch-free** half and always runs in the ML-free venv. Its
companion `test_stage1_resume_state_machine_torch.py` drives the real
`train_run` against a real saved/loaded checkpoint.

They are separate FILES on purpose. A module-level `pytest.importorskip` skips
everything below it, so folding these together would silently retire the
torch-free half in exactly this environment -- the defect Audit 030 hit twice
(SS V and Y).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.contracts import CorruptionRatePolicy  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    CORRUPTION_SEED,
    EXTENDED_MAX_UPDATES,
    INITIAL_MAX_UPDATES,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import SelectionViolation, ValidationPoint  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    LEGAL_CAPS,
    RunProvenance,
    TrainerContractViolation,
    checkpoint_payload,
    require_resumable_leg,
    resume_cap,
)


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=36930, init_seed=51800, corruption_seed=CORRUPTION_SEED,
        learning_rate=3e-4, r=1.0, corpus_manifest_digest="d" * 64,
        repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


def point(update: int, worst: float = 0.5) -> ValidationPoint:
    return ValidationPoint(
        update=update,
        distances={c: worst for c in VALIDATION_CONDITIONS},
        d_clean=worst / 2.0,
    )


def payload_for(*, global_update: int, cap: int, points=None, **overrides):
    """A payload built by the REAL writer, so it has the real schema."""
    body = dict(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=global_update, sampler_state={}, cap=cap,
        budget_limited=False,
        points=points if points is not None else [point(0)],
    )
    body.update(overrides)
    return checkpoint_payload(**body)


# ---------------------------------------------------------------------------
# Repair 2 -- the ValidationPoint writer/reader contract
# ---------------------------------------------------------------------------
def test_writer_emitted_points_are_readable_by_the_production_reader():
    """Case H. The exact round-trip that used to raise TypeError."""
    original = [point(0, 0.4), point(500, 0.3)]
    written = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=500, sampler_state={}, cap=INITIAL_MAX_UPDATES,
        budget_limited=False, points=original,
    )["points"]
    assert all("score" in p for p in written), "the writer emits the derived score"
    assert [ValidationPoint.from_dict(p) for p in written] == original


def test_the_pre_repair_reader_would_still_fail_on_writer_output():
    """Mutation check: the defect is real and this file would catch its return."""
    written = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=0, sampler_state={}, cap=INITIAL_MAX_UPDATES,
        budget_limited=False, points=[point(0)],
    )["points"]
    with pytest.raises(TypeError, match="score"):
        ValidationPoint(**written[0])


def test_a_contradictory_persisted_score_is_refused():
    """Case I. The score is derived, so a payload disagreeing with itself is corrupt."""
    raw = point(500, 0.4).to_dict()
    raw["score"] = 0.9
    with pytest.raises(SelectionViolation, match="recompute"):
        ValidationPoint.from_dict(raw)


def test_a_persisted_score_is_not_restored_as_independent_state():
    raw = point(500, 0.4).to_dict()
    restored = ValidationPoint.from_dict(raw)
    assert restored.score == 0.4
    raw_mutated = dict(raw, distances={c: 0.7 for c in VALIDATION_CONDITIONS})
    raw_mutated["score"] = 0.7
    assert ValidationPoint.from_dict(raw_mutated).score == 0.7


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.pop("distances"), "missing required field"),
    (lambda d: d.pop("d_clean"), "missing required field"),
    (lambda d: d.pop("update"), "missing required field"),
    (lambda d: d.update(surprise=1), "unknown field"),
    (lambda d: d.update(distances={"FULL": 0.1}), "missing condition"),
])
def test_a_malformed_validation_point_is_refused(mutate, match):
    """Case G. Fail closed on anything the schema does not describe."""
    raw = point(500).to_dict()
    mutate(raw)
    with pytest.raises(SelectionViolation, match=match):
        ValidationPoint.from_dict(raw)


# ---------------------------------------------------------------------------
# Repair 3 -- the cap is reconstructed from validated persisted state
# ---------------------------------------------------------------------------
def test_the_legal_caps_are_exactly_the_locked_budgets():
    assert LEGAL_CAPS == (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES)
    assert LEGAL_CAPS == (20_000, 40_000)


@pytest.mark.parametrize("global_update,cap", [
    (0, INITIAL_MAX_UPDATES),
    (500, INITIAL_MAX_UPDATES),
    (INITIAL_MAX_UPDATES, INITIAL_MAX_UPDATES),
    (INITIAL_MAX_UPDATES + 500, EXTENDED_MAX_UPDATES),
    (EXTENDED_MAX_UPDATES, EXTENDED_MAX_UPDATES),
])
def test_a_legitimate_leg_reconstructs_its_own_cap(global_update, cap):
    assert resume_cap(payload_for(global_update=global_update, cap=cap)) == cap


@pytest.mark.parametrize("global_update,cap,match", [
    (500, 30_000, "locked budgets"),                       # case F: invalid cap
    (500, 0, "locked budgets"),
    (25_000, INITIAL_MAX_UPDATES, "cannot have progressed"),  # case E: gu > cap
    (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES, "continuation leg begins only"),
    (0, EXTENDED_MAX_UPDATES, "continuation leg begins only"),
])
def test_an_impossible_persisted_state_is_refused(global_update, cap, match):
    payload = payload_for(global_update=0, cap=INITIAL_MAX_UPDATES)
    payload["global_update"] = global_update
    payload["cap"] = cap
    with pytest.raises(TrainerContractViolation, match=match):
        resume_cap(payload)


@pytest.mark.parametrize("bad", [None, "20000", 20_000.0, True])
def test_a_non_integer_cap_is_refused(bad):
    payload = payload_for(global_update=0, cap=INITIAL_MAX_UPDATES)
    payload["cap"] = bad
    with pytest.raises(TrainerContractViolation):
        resume_cap(payload)


def test_the_only_legal_promotion_is_a_completed_initial_leg():
    completed = payload_for(global_update=INITIAL_MAX_UPDATES, cap=INITIAL_MAX_UPDATES)
    require_resumable_leg(completed, EXTENDED_MAX_UPDATES)      # the continuation
    require_resumable_leg(completed, INITIAL_MAX_UPDATES)       # a crash resume

    midway = payload_for(global_update=500, cap=INITIAL_MAX_UPDATES)
    with pytest.raises(TrainerContractViolation, match="not a legal successor"):
        require_resumable_leg(midway, EXTENDED_MAX_UPDATES)


def test_a_continuation_checkpoint_cannot_be_resumed_under_the_initial_cap():
    """THE defect. This is what `execute_stage` did on every `--resume`."""
    continuation = payload_for(
        global_update=INITIAL_MAX_UPDATES + 500, cap=EXTENDED_MAX_UPDATES
    )
    with pytest.raises(TrainerContractViolation, match="smaller cap"):
        require_resumable_leg(continuation, INITIAL_MAX_UPDATES)


def test_no_third_continuation_is_reachable():
    finished = payload_for(global_update=EXTENDED_MAX_UPDATES, cap=EXTENDED_MAX_UPDATES)
    for beyond in (60_000, 80_000):
        with pytest.raises(TrainerContractViolation, match="locked budgets"):
            require_resumable_leg(finished, beyond)
