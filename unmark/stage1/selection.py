"""Stage-1 selection, budget and run-schedule rules. **Torch-free.**

Every rule here was locked before any Stage-1 number existed (D-S1B-004), so
each is a pure function of already-computed metrics -- no model, no tensors, and
therefore fully testable in the ML-free environment.

Nothing in this module reads a downstream score. The only inputs are the
Stage-1 held-out UNLABELED distances (D-S1B-001).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    EVAL_EVERY_UPDATES,
    EXTENDED_MAX_UPDATES,
    INITIAL_MAX_UPDATES,
    LR_PILOT_GRID,
    LR_PILOT_R,
    R_PHASE1_GRID,
    TOTAL_NOMINAL_RUNS,
    TRAIN_SEEDS,
    VALIDATION_CONDITIONS,
)


class SelectionViolation(Stage1ContractViolation):
    """Raised when a selection input violates the locked protocol."""


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationPoint:
    """Held-out metrics at one update. `d_c` for every locked condition."""

    update: int
    distances: Mapping[str, float]
    d_clean: float

    def __post_init__(self) -> None:
        if self.update < 0:
            raise SelectionViolation(f"update must be non-negative, got {self.update}")
        missing = [c for c in VALIDATION_CONDITIONS if c not in self.distances]
        if missing:
            raise SelectionViolation(
                f"validation point at update {self.update} is missing condition(s) "
                f"{missing}; the grid {list(VALIDATION_CONDITIONS)} is locked"
            )
        extra = sorted(set(self.distances) - set(VALIDATION_CONDITIONS))
        if extra:
            raise SelectionViolation(
                f"unexpected condition(s) {extra}; the grid is locked and is not "
                "extended after seeing results"
            )

    @property
    def score(self) -> float:
        """The locked selection score: worst case over the fixed condition grid.

        Worst-case rather than a mean: a mean lets a configuration win by being
        excellent at FULL and poor at STRIP-ALL, the reverse of the headline
        claim.
        """
        return max(self.distances[c] for c in VALIDATION_CONDITIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update": self.update,
            "distances": {c: self.distances[c] for c in VALIDATION_CONDITIONS},
            "d_clean": self.d_clean,
            "score": self.score,
        }


def select_checkpoint(points: Sequence[ValidationPoint]) -> ValidationPoint:
    """Lowest worst-case score, then lower `d_clean`, then **earliest** update."""
    if not points:
        raise SelectionViolation("cannot select a checkpoint from no validation points")
    updates = [p.update for p in points]
    if len(set(updates)) != len(updates):
        raise SelectionViolation(f"duplicate validation updates: {sorted(updates)}")
    if 0 not in updates:
        raise SelectionViolation(
            "update 0 must be evaluated BEFORE the first optimizer step, so the "
            "initial clean-path distance is measured rather than assumed"
        )
    return min(points, key=lambda p: (p.score, p.d_clean, p.update))


# ---------------------------------------------------------------------------
# The budget rule -- one continuation, then stop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetDecision:
    cap: int
    continue_run: bool
    budget_limited: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "continue_run": self.continue_run,
            "budget_limited": self.budget_limited,
            "reason": self.reason,
        }


def budget_decision(selected_update: int, cap: int) -> BudgetDecision:
    """What to do after a run reaches `cap`.

    The trigger reads only the Stage-1 held-out selection -- "the best
    checkpoint is the last one we computed", i.e. the budget bound rather than
    the optimum. It reads no downstream score, and the ceiling is fixed before
    any run.
    """
    if cap not in (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES):
        raise SelectionViolation(
            f"cap {cap} is not one of the locked budgets "
            f"({INITIAL_MAX_UPDATES}, {EXTENDED_MAX_UPDATES})"
        )
    if selected_update > cap:
        raise SelectionViolation(f"selected update {selected_update} exceeds the cap {cap}")
    if selected_update < cap:
        return BudgetDecision(cap, False, False, "selected checkpoint is inside the budget")
    if cap == INITIAL_MAX_UPDATES:
        return BudgetDecision(
            EXTENDED_MAX_UPDATES,
            True,
            False,
            f"selected checkpoint is exactly the cap ({cap}); continue the SAME run to "
            f"{EXTENDED_MAX_UPDATES}, preserving adapter, optimizer, visit, cursor and "
            "corruption streams",
        )
    return BudgetDecision(
        EXTENDED_MAX_UPDATES,
        False,
        True,
        f"selected checkpoint is exactly {EXTENDED_MAX_UPDATES}; STOP and mark "
        "BUDGET_LIMITED. No 60k/80k extension may be added after inspecting results",
    )


def evaluation_updates(cap: int) -> tuple[int, ...]:
    """Update 0, then every `EVAL_EVERY_UPDATES`, including the cap itself."""
    if cap % EVAL_EVERY_UPDATES:
        raise SelectionViolation(f"cap {cap} is not a multiple of {EVAL_EVERY_UPDATES}")
    return tuple(range(0, cap + 1, EVAL_EVERY_UPDATES))


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One completed candidate run, reduced to its selected checkpoint."""

    label: str
    learning_rate: float
    r: float
    selected: ValidationPoint
    budget_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "learning_rate": self.learning_rate,
            "r": self.r,
            "budget_limited": self.budget_limited,
            "selected": self.selected.to_dict(),
        }


def select_learning_rate(candidates: Sequence[Candidate]) -> Candidate:
    """The LR pilot winner. Exactly the locked grid, all at `r = 1`."""
    rates = sorted(c.learning_rate for c in candidates)
    if rates != sorted(LR_PILOT_GRID):
        raise SelectionViolation(
            f"the LR pilot grid is {sorted(LR_PILOT_GRID)}; got {rates}. The grid is "
            "precommitted and is not extended after seeing results."
        )
    off = [c.label for c in candidates if c.r != LR_PILOT_R]
    if off:
        raise SelectionViolation(f"LR pilot candidates must all use r={LR_PILOT_R}; {off} did not")
    return min(candidates, key=lambda c: (c.selected.score, c.selected.d_clean, c.learning_rate))


def select_r(candidates: Sequence[Candidate], frozen_learning_rate: float) -> Candidate:
    """The `r` Phase-1 winner: score, then lower `d_clean`, then **smaller r**.

    No seed-variance term: Phase 1 runs one precommitted selection seed, and a
    single run has no sample SD.
    """
    values = sorted(c.r for c in candidates)
    if values != sorted(R_PHASE1_GRID):
        raise SelectionViolation(
            f"the r grid is {sorted(R_PHASE1_GRID)}; got {values}. The grid is "
            "precommitted and is not extended after seeing results."
        )
    off = [c.label for c in candidates if c.learning_rate != frozen_learning_rate]
    if off:
        raise SelectionViolation(
            f"r Phase-1 candidates must all use the frozen LR {frozen_learning_rate}; "
            f"{off} did not"
        )
    return min(candidates, key=lambda c: (c.selected.score, c.selected.d_clean, c.r))


# ---------------------------------------------------------------------------
# The run schedule -- exactly 11
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedRun:
    stage: str
    label: str
    learning_rate: float | None
    r: float | None
    seed: int


def lr_pilot_schedule(selection_seed: int) -> tuple[PlannedRun, ...]:
    return tuple(
        PlannedRun("lr_pilot", f"lr={lr:g}", lr, LR_PILOT_R, selection_seed)
        for lr in LR_PILOT_GRID
    )


def r_phase1_schedule(selection_seed: int, frozen_learning_rate: float) -> tuple[PlannedRun, ...]:
    return tuple(
        PlannedRun("r_phase1", f"r={r:g}", frozen_learning_rate, r, selection_seed)
        for r in R_PHASE1_GRID
    )


def final_main_schedule(learning_rate: float, r: float) -> tuple[PlannedRun, ...]:
    """The three FINAL MAIN Stage-1 runs. **Nothing follows them.**"""
    return tuple(
        PlannedRun("final_main", f"seed={seed}", learning_rate, r, seed) for seed in TRAIN_SEEDS
    )


def total_planned_runs(frozen_learning_rate: float = 1e-4, selected_r: float = 1.0) -> int:
    return (
        len(lr_pilot_schedule(0))
        + len(r_phase1_schedule(0, frozen_learning_rate))
        + len(final_main_schedule(frozen_learning_rate, selected_r))
    )


if total_planned_runs() != TOTAL_NOMINAL_RUNS:  # pragma: no cover - import guard
    raise AssertionError(
        f"the run plan yields {total_planned_runs()} runs, not {TOTAL_NOMINAL_RUNS}"
    )


def descriptive_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Mean and **sample** SD (n-1). Descriptive only -- no test, no threshold."""
    if not values:
        raise SelectionViolation("cannot summarise an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }
