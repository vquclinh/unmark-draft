"""Run schedule, selection rules, budget rule, sampler/resume. All ML-free."""

from __future__ import annotations

import pytest

from unmark.stage1.protocol import (
    EVAL_EVERY_UPDATES,
    EXTENDED_MAX_UPDATES,
    INITIAL_MAX_UPDATES,
    LR_PILOT_GRID,
    LR_PILOT_R,
    R_PHASE1_GRID,
    SELECTION_SEED,
    TOTAL_NOMINAL_RUNS,
    TRAIN_SEEDS,
    VALIDATION_CONDITIONS,
    lambdas_for_r,
)
from unmark.stage1.sampler import DeterministicSampler, SamplerStateViolation, pass_order
from unmark.stage1.selection import (
    Candidate,
    SelectionViolation,
    ValidationPoint,
    budget_decision,
    evaluation_updates,
    final_main_schedule,
    lr_pilot_schedule,
    r_phase1_schedule,
    select_checkpoint,
    select_learning_rate,
    select_r,
    total_planned_runs,
)


def point(update, score, d_clean=0.5):
    return ValidationPoint(update, {c: score for c in VALIDATION_CONDITIONS}, d_clean)


# ---------------------------------------------------------------------------
# Run schedule -- exactly 11
# ---------------------------------------------------------------------------
def test_exactly_three_five_three_equals_eleven():
    assert len(lr_pilot_schedule(SELECTION_SEED)) == 3
    assert len(r_phase1_schedule(SELECTION_SEED, 3e-4)) == 5
    assert len(final_main_schedule(3e-4, 1.0)) == 3
    assert total_planned_runs() == TOTAL_NOMINAL_RUNS == 11


def test_the_pilot_is_the_locked_grid_at_r_one_on_the_selection_seed():
    runs = lr_pilot_schedule(SELECTION_SEED)
    assert sorted(r.learning_rate for r in runs) == sorted(LR_PILOT_GRID)
    assert {r.r for r in runs} == {LR_PILOT_R} == {1.0}
    assert {r.seed for r in runs} == {SELECTION_SEED} == {21230}


def test_phase1_sweeps_r_at_the_frozen_lr_on_the_same_selection_seed():
    runs = r_phase1_schedule(SELECTION_SEED, 1e-3)
    assert sorted(r.r for r in runs) == sorted(R_PHASE1_GRID)
    assert {r.learning_rate for r in runs} == {1e-3}
    assert {r.seed for r in runs} == {SELECTION_SEED}


def test_the_final_three_use_the_three_train_seeds():
    runs = final_main_schedule(3e-4, 2.0)
    assert [r.seed for r in runs] == list(TRAIN_SEEDS) == [36930, 7309, 5993]
    assert {r.stage for r in runs} == {"final_main"}
    assert {r.learning_rate for r in runs} == {3e-4}
    assert {r.r for r in runs} == {2.0}


def test_no_post_hoc_grid_expansion():
    winner = lambda lr, r, s: Candidate(f"c{lr}", lr, r, point(500, s))
    extra = [winner(lr, LR_PILOT_R, 0.4) for lr in LR_PILOT_GRID] + [winner(5e-3, 1.0, 0.1)]
    with pytest.raises(SelectionViolation, match="precommitted and is not extended"):
        select_learning_rate(extra)
    short = [Candidate(f"r{r}", 1e-4, r, point(500, 0.4)) for r in (0.25, 0.5, 1.0)]
    with pytest.raises(SelectionViolation, match="precommitted and is not extended"):
        select_r(short, 1e-4)


def test_lambdas_follow_the_locked_scale():
    for r in R_PHASE1_GRID:
        a, c = lambdas_for_r(r)
        assert a + c == pytest.approx(2.0)
        assert c / a == pytest.approx(r)
    assert lambdas_for_r(1.0) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------
def test_worst_case_score_not_a_mean():
    p = ValidationPoint(0, {"FULL": 0.1, "P50": 0.2, "P100": 0.3, "STRIP_ALL": 0.9}, 0.1)
    assert p.score == 0.9


def test_update_zero_must_be_present():
    with pytest.raises(SelectionViolation, match="update 0 must be evaluated"):
        select_checkpoint([point(500, 0.4), point(1000, 0.3)])


def test_checkpoint_tie_break_is_d_clean_then_earliest():
    points = [point(0, 0.9), point(500, 0.4, 0.30), point(1000, 0.4, 0.20), point(1500, 0.4, 0.20)]
    chosen = select_checkpoint(points)
    assert chosen.update == 1000, "lower d_clean wins, then the EARLIEST update"


def test_r_tie_break_is_d_clean_then_smaller_r():
    cands = [
        Candidate(f"r={r}", 1e-4, r, point(500, 0.4, 0.2 if r in (2.0, 4.0) else 0.3))
        for r in R_PHASE1_GRID
    ]
    assert select_r(cands, 1e-4).r == 2.0, "equal score and d_clean -> smaller r"


def test_r_phase1_requires_the_frozen_lr_everywhere():
    cands = [Candidate(f"r={r}", 1e-4, r, point(500, 0.4)) for r in R_PHASE1_GRID]
    cands[2] = Candidate("r=1", 1e-3, 1.0, point(500, 0.4))
    with pytest.raises(SelectionViolation, match="must all use the frozen LR"):
        select_r(cands, 1e-4)


def test_the_condition_grid_is_locked_in_both_directions():
    with pytest.raises(SelectionViolation, match="missing condition"):
        ValidationPoint(0, {"FULL": 0.1, "P50": 0.2, "P100": 0.3}, 0.1)
    with pytest.raises(SelectionViolation, match="unexpected condition"):
        ValidationPoint(0, {**{c: 0.1 for c in VALIDATION_CONDITIONS}, "P25": 0.1}, 0.1)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def test_evaluation_includes_update_zero_and_the_cap():
    updates = evaluation_updates(INITIAL_MAX_UPDATES)
    assert updates[0] == 0
    assert updates[-1] == INITIAL_MAX_UPDATES
    assert all(b - a == EVAL_EVERY_UPDATES for a, b in zip(updates, updates[1:]))


def test_normal_stop_inside_the_budget():
    d = budget_decision(19500, INITIAL_MAX_UPDATES)
    assert not d.continue_run and not d.budget_limited


def test_selected_at_the_cap_continues_the_same_run():
    d = budget_decision(INITIAL_MAX_UPDATES, INITIAL_MAX_UPDATES)
    assert d.continue_run and d.cap == EXTENDED_MAX_UPDATES and not d.budget_limited
    assert "SAME run" in d.reason
    assert "preserving adapter, optimizer, visit, cursor" in d.reason


def test_selected_at_forty_thousand_stops_and_marks_budget_limited():
    d = budget_decision(EXTENDED_MAX_UPDATES, EXTENDED_MAX_UPDATES)
    assert d.budget_limited and not d.continue_run
    assert "No 60k/80k extension" in d.reason


def test_no_third_budget_exists():
    with pytest.raises(SelectionViolation, match="not one of the locked budgets"):
        budget_decision(100, 60_000)


# ---------------------------------------------------------------------------
# Sampler / resume
# ---------------------------------------------------------------------------
IDS = tuple(f"c{i}" for i in range(20))


def test_order_is_deterministic_and_seed_dependent():
    assert pass_order(IDS, 36930, 0) == pass_order(IDS, 36930, 0)
    assert pass_order(IDS, 36930, 0) != pass_order(IDS, 7309, 0)
    assert pass_order(IDS, 36930, 0) != pass_order(IDS, 36930, 1)
    assert sorted(pass_order(IDS, 36930, 0)) == sorted(IDS)


def test_each_chunk_is_consumed_exactly_once_per_pass():
    s = DeterministicSampler(IDS, seed=36930)
    drawn = [cid for cid, _ in s.next_batch(len(IDS))]
    assert sorted(drawn) == sorted(IDS)
    assert s.visit == 1 and s.position == 0, "visit advances only at the pass boundary"


def test_visit_travels_with_each_example_across_a_pass_boundary():
    s = DeterministicSampler(IDS, seed=36930)
    s.next_batch(len(IDS) - 3)
    straddling = s.next_batch(6)
    visits = {v for _, v in straddling}
    assert visits == {0, 1}, "each half must be corrupted under its own pass"


def test_mid_pass_resume_continues_from_the_exact_next_position():
    a = DeterministicSampler(IDS, seed=36930)
    a.next_batch(7)
    state = a.state_dict()
    assert state["position"] == 7 and state["visit"] == 0

    b = DeterministicSampler.from_state(IDS, state)
    assert b.next_batch(5) == a.next_batch(5), "resume must not restart the pass"


def test_resume_does_not_reserve_earlier_samples_or_bump_visit():
    a = DeterministicSampler(IDS, seed=36930)
    first = [cid for cid, _ in a.next_batch(8)]
    b = DeterministicSampler.from_state(IDS, a.state_dict())
    later = [cid for cid, _ in b.next_batch(8)]
    assert not set(first) & set(later), "resumed batches must not repeat consumed chunks"
    assert b.visit == 0, "resume must not increment the visit early"


def test_uninterrupted_and_resumed_training_see_the_same_stream():
    straight = DeterministicSampler(IDS, seed=5993)
    whole = straight.next_batch(30)

    part = DeterministicSampler(IDS, seed=5993)
    head = part.next_batch(11)
    resumed = DeterministicSampler.from_state(IDS, part.state_dict())
    assert head + resumed.next_batch(19) == whole


def test_resume_fails_closed_on_a_different_corpus():
    a = DeterministicSampler(IDS, seed=36930)
    a.next_batch(4)
    with pytest.raises(SamplerStateViolation, match="different chunk set"):
        DeterministicSampler.from_state(IDS[:-1] + ("other",), a.state_dict())


def test_resume_fails_closed_on_missing_or_stale_state():
    a = DeterministicSampler(IDS, seed=36930)
    state = a.state_dict()
    with pytest.raises(SamplerStateViolation, match="missing"):
        DeterministicSampler.from_state(IDS, {k: v for k, v in state.items() if k != "visit"})
    with pytest.raises(SamplerStateViolation, match="schema"):
        DeterministicSampler.from_state(IDS, {**state, "schema_version": "old"})
