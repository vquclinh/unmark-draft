"""Training resume equivalence for the **scientific state machine** (F3).

Torch-free, so it runs in the ML-free venv rather than being Colab-gated. Its
companion `test_stage1_training_resume.py` proves tensor and optimizer-state
equality and needs torch; this file proves the part that decides *which data is
seen in which order*, which is what makes a resumed run scientifically the same
run:

* the sampler cursor and `visit`,
* the exact `(chunk_id, visit)` sequence consumed,
* the update count,
* the validation history,
* the checkpoint identity that gates all of it.

A numeric accumulator stands in for the model: it is driven **only** by the
sampled pairs, so if the resumed run saw a different sequence -- or replayed or
skipped one batch -- the number diverges. Exact equality is required.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.sampler import DeterministicSampler
from unmark.stage1.trainer import (
    CHECKPOINT_EVERY_UPDATES,
    REQUIRED_CHECKPOINT_KEYS,
    RunProvenance,
    TrainerContractViolation,
    checkpoint_payload,
    verify_checkpoint,
)

EVERY = 4
CAP = 20
BATCH = 3


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=36930, corruption_seed=35422, learning_rate=3e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


def chunk_ids(n: int = 7) -> tuple[str, ...]:
    """Deliberately not a multiple of BATCH, so passes end mid-batch and the
    visit boundary is crossed inside a batch rather than between two."""
    return tuple(f"doc-{i:04d}#0" for i in range(n))


def accumulate(state: float, pairs) -> float:
    """Order- and content-sensitive. Replaying or skipping a batch changes it."""
    for index, (chunk_id, visit) in enumerate(pairs):
        state = state * 1.000_001 + (len(chunk_id) * (visit + 1) + index) / 997.0
    return state


def run(sampler, *, start_update: int, cap: int, state: float, points=None,
        seen=None, stop_at: int | None = None):
    """A faithful miniature of `train_run`'s loop: step, then validate."""
    points = list(points or [])
    seen = list(seen or [])
    update = start_update
    while update < cap:
        pairs = sampler.next_batch(BATCH)
        seen.extend(pairs)
        state = accumulate(state, pairs)
        update += 1
        if update % EVERY == 0 or update == cap:
            points.append({"update": update, "score": round(state, 12)})
            if stop_at is not None and update >= stop_at:
                break
    return state, update, points, seen


def snapshot(sampler, state, update, points, seen):
    return {
        "state": state, "update": update, "points": points, "seen": seen,
        "visit": sampler.visit, "position": sampler.position,
    }


def assert_equivalent(whole, resumed):
    assert resumed["state"] == whole["state"], "the numeric state diverged"
    assert resumed["seen"] == whole["seen"], "a different (chunk, visit) sequence was consumed"
    assert resumed["update"] == whole["update"], "update count differs"
    assert resumed["visit"] == whole["visit"], "visit/pass state differs"
    assert resumed["position"] == whole["position"], "sampler cursor differs"
    assert resumed["points"] == whole["points"], "validation history differs"


# ---------------------------------------------------------------------------
# The equivalence property at every awkward interruption point
# ---------------------------------------------------------------------------
def _run_whole():
    sampler = DeterministicSampler(chunk_ids(), seed=36930)
    state, update, points, seen = run(sampler, start_update=0, cap=CAP, state=1.0)
    return snapshot(sampler, state, update, points, seen)


@pytest.mark.parametrize("stop_at", [4, 8, 12, 16, 20])
def test_resume_equals_uninterrupted(stop_at):
    ids = chunk_ids()
    whole = _run_whole()

    # First leg, then the process "dies": everything is rebuilt from the payload.
    sampler_a = DeterministicSampler(ids, seed=36930)
    state_a, update_a, points_a, seen_a = run(
        sampler_a, start_update=0, cap=CAP, state=1.0, stop_at=stop_at
    )
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={"state": state_a},
        optimizer_state={}, global_update=update_a,
        sampler_state=sampler_a.state_dict(), cap=CAP, budget_limited=False,
        points=points_a,
    )
    # Survives serialisation -- a real checkpoint is written and read back.
    payload = json.loads(json.dumps(payload))
    del sampler_a, state_a, update_a, points_a

    verify_checkpoint(payload, provenance())
    sampler_b = DeterministicSampler.from_state(ids, payload["sampler_state"])
    state_b, update_b, points_b, seen_b = run(
        sampler_b,
        start_update=int(payload["global_update"]),
        cap=CAP,
        state=payload["adapter_state"]["state"],
        points=payload["points"],
        seen=seen_a,
    )
    assert_equivalent(whole, snapshot(sampler_b, state_b, update_b, points_b, seen_b))


def test_a_resume_that_forgets_the_cursor_is_caught_by_this_test():
    """Mutation check: the test must actually detect a reset cursor."""
    ids = chunk_ids()
    whole = _run_whole()

    sampler_a = DeterministicSampler(ids, seed=36930)
    _, update_a, points_a, seen_a = run(
        sampler_a, start_update=0, cap=CAP, state=1.0, stop_at=8
    )
    # The defect: rebuild from scratch instead of from state.
    sampler_bad = DeterministicSampler(ids, seed=36930)
    state_b, update_b, points_b, seen_b = run(
        sampler_bad, start_update=update_a, cap=CAP, state=1.0,
        points=points_a, seen=seen_a,
    )
    with pytest.raises(AssertionError):
        assert_equivalent(whole, snapshot(sampler_bad, state_b, update_b, points_b, seen_b))


# ---------------------------------------------------------------------------
# The checkpoint contract itself
# ---------------------------------------------------------------------------
def test_points_are_persisted_now():
    """They were read on resume but never written before this hardening."""
    assert "points" in REQUIRED_CHECKPOINT_KEYS
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=500, sampler_state={}, cap=20_000, budget_limited=False,
        points=[{"update": 0}, {"update": 500}],
    )
    assert payload["points"] == [{"update": 0}, {"update": 500}]


@pytest.mark.parametrize("missing", sorted(REQUIRED_CHECKPOINT_KEYS))
def test_an_incomplete_checkpoint_is_refused(missing):
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=1, sampler_state={}, cap=2, budget_limited=False, points=[],
    )
    payload.pop(missing)
    with pytest.raises(TrainerContractViolation):
        verify_checkpoint(payload, provenance())


@pytest.mark.parametrize("field,value", [
    ("run_seed", 7309),
    ("corruption_seed", 1),
    ("learning_rate", 1e-3),
    ("r", 4.0),
    ("corpus_manifest_digest", "e" * 64),
    ("repository_head", "b" * 40),
])
def test_a_foreign_experiment_cannot_resume(field, value):
    payload = checkpoint_payload(
        provenance=provenance(**{field: value}), adapter_state={},
        optimizer_state={}, global_update=1, sampler_state={}, cap=2,
        budget_limited=False, points=[],
    )
    with pytest.raises(Exception):
        verify_checkpoint(payload, provenance())


def test_a_foreign_prepared_corpus_cannot_resume():
    """The digest is now the VERIFIED one (F1), so this also binds the payload."""
    payload = checkpoint_payload(
        provenance=provenance(corpus_manifest_digest="f" * 64), adapter_state={},
        optimizer_state={}, global_update=1, sampler_state={}, cap=2,
        budget_limited=False, points=[],
    )
    with pytest.raises(Exception):
        verify_checkpoint(payload, provenance())


def test_the_cadence_is_an_update_count_not_wall_clock():
    from unmark.stage1.protocol import EVAL_EVERY_UPDATES

    assert CHECKPOINT_EVERY_UPDATES == EVAL_EVERY_UPDATES
    assert isinstance(CHECKPOINT_EVERY_UPDATES, int)


# ---------------------------------------------------------------------------
# D-S1B-004's locked "best + last" persistence
# ---------------------------------------------------------------------------
def test_best_and_last_are_separate_artifacts():
    """D-S1B-004 locks "best + last checkpoint persistence" -- both, not one."""
    from unmark.stage1.trainer import BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME

    assert BEST_CHECKPOINT_NAME != LAST_CHECKPOINT_NAME


def test_best_is_decided_by_the_locked_selection_rule_not_a_new_one():
    """The loop must ask `select_checkpoint`, never re-implement the comparison.

    A second "is this better?" comparison written inside the trainer is exactly
    how a selection rule silently forks. Asserted on the call graph.
    """
    import ast
    import inspect

    import unmark.stage1.trainer as trainer_module

    tree = ast.parse(inspect.getsource(trainer_module))
    train_run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_run"
    )
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(train_run) if isinstance(n, ast.Call)
    }
    assert "select_checkpoint" in called, called


def test_the_cadence_constant_cannot_drift_from_the_eval_cadence():
    """D-S1B-004 locks one cadence; two constants would be two things to keep."""
    from unmark.stage1.protocol import EVAL_EVERY_UPDATES
    from unmark.stage1.trainer import CHECKPOINT_EVERY_UPDATES

    assert CHECKPOINT_EVERY_UPDATES is EVAL_EVERY_UPDATES
