"""The Stage-1 candidate register: which objective a stage trains, and how far.
**No torch.**

Two questions have to be answered the same way by every part of the Stage-1
stack, and until Audit 065 only the first of them had a single answer:

1. **which objective does this stage optimise?** -- resolved here from the stage
   name alone, so no flag, config key or argument can give a stage a loss its
   artifacts do not name;
2. **how far may this stage's run go?** -- resolved here from the candidate's
   `(objective, fusion)` identity, because a screening budget belongs to the
   candidate being screened, not to the trainer that happens to run it.

Audit 065 BLOCKER 1 was that the second question had no owner. The generic
precommitted rule (20 000 updates, then exactly one continuation to 40 000 if
the best held-out checkpoint lands on the cap) was applied to **every** run,
V2-GC included, and `V2_GC_MAX_UPDATES` was a constant nothing consulted -- it
appeared in an import, a `print()` and one test assertion. A candidate that the
protocol hard-caps at 20 000 could therefore execute 40 000 updates, and the
test suite reported budget coverage it did not have.

`ScreeningBudget` is that owner. The historical stages keep
`PRECOMMITTED_CONTINUATION` -- byte-for-byte the policy they have always run
under -- and V2-GC gets `FIRST_SCREEN_HARD_CAP`, which refuses the continuation
outright and fails closed on any cap or resumed update above its ceiling.

**Why the key is a PAIR.** C1 (V2-SCF) trains the *historical* objective with a
*different* adapter fusion, so the objective alone does not identify it. Keying
the budget on `(objective_id, fusion_id)` is what stops C1 inheriting the
historical 40 000-update continuation, and what stops its checkpoints being
mistaken for UNMARK-A's:

    historical  = (align-clean-pooled-v1, historical-fusion-v1)       precommitted
    C1 / V2-SCF = (align-clean-pooled-v1, scale-calibrated-fusion-v1) hard cap
    C3 / V2-GC  = (grid-consistency-v1,   historical-fusion-v1)       hard cap

    C2 / V2-GRD = (geometry-relational-distillation-v1, historical-fusion-v1) hard cap

**Designed for more than one candidate, and now carrying three.** Each was added
by appending one `StageCandidate` here. Nothing in `trainer` or `execute` learns
a candidate's name, and nothing anywhere assumes that a post-hoc candidate is the
grid-consistency one, the scale-calibrated one or the relational one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unmark.stage1.contracts import (
    GEOMETRY_RELATIONAL_OBJECTIVE,
    GRID_CONSISTENCY_OBJECTIVE,
    HISTORICAL_FUSION,
    HISTORICAL_OBJECTIVE,
    SCALE_CALIBRATED_FUSION,
    FusionIdentity,
    ObjectiveIdentity,
    Stage1ContractViolation,
)
from unmark.stage1.protocol import (
    EXTENDED_MAX_UPDATES,
    INITIAL_MAX_UPDATES,
    V2_GC_MAX_UPDATES,
    V2_GC_STAGE,
    V2_GRD_MAX_UPDATES,
    V2_GRD_STAGE,
    V2_SCF_MAX_UPDATES,
    V2_SCF_STAGE,
)


class BudgetPolicyViolation(Stage1ContractViolation):
    """Raised when an execution would exceed a candidate's screening budget."""


@dataclass(frozen=True)
class ScreeningBudget:
    """How far one candidate's run may go, and whether it may be continued.

    Args:
        hard_max_updates: an absolute ceiling in updates, or `None` for "this
            candidate has no ceiling of its own and is governed by the locked
            precommitted rule". A ceiling is not advice: `require_cap` and
            `require_update_within` refuse rather than clamp, so there is no
            silent reinterpretation of a payload that claims to be past it.
        allows_precommitted_continuation: whether the locked 20k -> 40k
            continuation (`selection.budget_decision`) may be *applied* to this
            candidate. The rule itself is untouched and still owns the decision
            for the historical stages; this says who it is allowed to move.
        policy: the name recorded in run artifacts, so a reader can tell which
            regime produced a trajectory without inferring it from the numbers.
    """

    hard_max_updates: int | None
    allows_precommitted_continuation: bool
    policy: str

    def __post_init__(self) -> None:
        if self.hard_max_updates is not None:
            if isinstance(self.hard_max_updates, bool) or not isinstance(
                self.hard_max_updates, int
            ):
                raise Stage1ContractViolation(
                    f"hard_max_updates must be an int or None, got {self.hard_max_updates!r}"
                )
            if self.hard_max_updates <= 0:
                raise Stage1ContractViolation(
                    f"hard_max_updates must be positive, got {self.hard_max_updates}"
                )
        if self.hard_max_updates is not None and self.allows_precommitted_continuation:
            # The continuation's only destination is EXTENDED_MAX_UPDATES, so a
            # budget that permits it AND declares a lower ceiling is a policy
            # that contradicts itself. Caught at construction, not at 20 000.
            if self.hard_max_updates < EXTENDED_MAX_UPDATES:
                raise Stage1ContractViolation(
                    f"a budget with hard_max_updates={self.hard_max_updates} cannot also "
                    f"allow the precommitted continuation, whose destination is "
                    f"{EXTENDED_MAX_UPDATES}"
                )

    @property
    def is_hard_capped(self) -> bool:
        return self.hard_max_updates is not None

    def require_cap(self, cap: int, *, what: str) -> int:
        """The cap is legal for this candidate, or nothing runs. Returns `cap`.

        Refuses; never clamps. A caller that asked for 40 000 under a 20 000
        candidate has a wrong plan, and quietly running 20 000 instead would
        record a trajectory nobody asked for under a budget nobody chose.
        """
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise BudgetPolicyViolation(f"{what}: cap {cap!r} is not an integer")
        if self.hard_max_updates is not None and cap > self.hard_max_updates:
            raise BudgetPolicyViolation(
                f"{what}: cap {cap} exceeds the {self.policy} ceiling of "
                f"{self.hard_max_updates} updates. This candidate is HARD-CAPPED: the "
                "locked 20k->40k continuation does not apply to it, and the cap is not "
                "lowered silently to fit."
            )
        return cap

    def require_update_within(self, update: int, *, what: str) -> int:
        """A recorded update count is inside the ceiling, or the run stops.

        Used on resume. A V2-GC payload recording more updates than the ceiling
        is **real evidence of work that should never have happened**; it is
        refused, not normalised, because normalising it would silently reinterpret
        a checkpoint that did execute past the cap as one that did not.
        """
        if isinstance(update, bool) or not isinstance(update, int):
            raise BudgetPolicyViolation(f"{what}: global_update {update!r} is not an integer")
        if update < 0:
            raise BudgetPolicyViolation(f"{what}: global_update {update} is negative")
        if self.hard_max_updates is not None and update > self.hard_max_updates:
            raise BudgetPolicyViolation(
                f"{what}: the carried state records {update} updates, past the "
                f"{self.policy} ceiling of {self.hard_max_updates}. A checkpoint that is "
                "already beyond this candidate's hard cap is refused rather than "
                "reinterpreted -- the work it describes was never authorised."
            )
        return update

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "hard_max_updates": self.hard_max_updates,
            "allows_precommitted_continuation": self.allows_precommitted_continuation,
        }


PRECOMMITTED_CONTINUATION = ScreeningBudget(
    hard_max_updates=None,
    allows_precommitted_continuation=True,
    policy="precommitted_initial_then_one_continuation",
)
"""**The historical policy, unchanged.** D-S1B-004: run to
`INITIAL_MAX_UPDATES`; if the best held-out checkpoint is exactly the cap,
continue the SAME run once to `EXTENDED_MAX_UPDATES`; then `BUDGET_LIMITED` and
no further extension. Every historical Stage-1 stage keeps this."""

FIRST_SCREEN_HARD_CAP = ScreeningBudget(
    hard_max_updates=V2_GC_MAX_UPDATES,
    allows_precommitted_continuation=False,
    policy="first_screen_hard_cap",
)
"""**Post-hoc candidate screening.** One fixed budget, no continuation.

A first screen exists to compare a candidate objective against the historical
one at a *matched* budget. Letting the candidate continue to 40 000 because its
own held-out curve was still improving would compare 40 000 updates of a new
objective against 20 000 of the old, and the comparison would no longer isolate
the objective -- which is the entire purpose of the candidate.

`V2_GC_MAX_UPDATES` is consumed here and nowhere decorative: this object is what
`trainer.resolve_run_cap` and `trainer.resolve_budget` enforce."""


@dataclass(frozen=True)
class StageCandidate:
    """One stage: what it trains, how it is built, how far it goes, where it logs.

    Four bindings, and all four are scientific except the last:

    * `stage` -- the candidate id, and the only thing a caller ever supplies;
    * `objective` -- which loss (`ObjectiveIdentity`);
    * `fusion` -- which adapter architecture (`FusionIdentity`);
    * `budget` -- how far its first screen may run (`ScreeningBudget`);
    * `wandb_project` -- OPERATIONAL ONLY; observational, never scientific.

    Objective and fusion are BOTH required to identify a candidate. C1 proves
    why: it shares the historical objective and differs only in the fusion, so a
    register keyed on the objective alone would give it the historical budget and
    make its checkpoints indistinguishable from UNMARK-A's.
    """

    stage: str
    objective: ObjectiveIdentity
    fusion: FusionIdentity
    budget: ScreeningBudget
    wandb_project: str

    @property
    def identity(self) -> tuple[str, str]:
        """`(objective_id, fusion_id)` -- what actually distinguishes candidates."""
        return (self.objective.objective_id, self.fusion.fusion_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            **self.objective.to_dict(),
            **self.fusion.to_dict(),
            "budget": self.budget.to_dict(),
        }


HISTORICAL_STAGES: tuple[str, ...] = ("lr_pilot", "r_phase1", "final_main")
"""The three stages of the CLOSED historical campaign. Their behaviour is fixed."""

HISTORICAL_WANDB_PROJECT = "unmark-stage1"
"""The project the historical campaign has always logged to. Unchanged."""

V2_GC_WANDB_PROJECT = "UNMARK-v2-C3-GC-Stage1"
V2_GRD_WANDB_PROJECT = "UNMARK-v2-C2-GRD-Stage1"
V2_SCF_WANDB_PROJECT = "UNMARK-v2-C1-SCF-Stage1"
"""One W&B project per post-hoc candidate. **OPERATIONAL ONLY.**

Separate projects because C1, C2 and C3 will be launched from separate Colab
notebooks off ONE repository HEAD, and a shared project would interleave three
different experiments' curves under one set of axes.

Nothing here is scientific: the project name reaches no loss, no seed, no
selection and no budget, and a W&B failure degrades the run to console-only
without changing a single computed value."""

CANDIDATES: tuple[StageCandidate, ...] = (
    *(
        StageCandidate(
            stage=stage,
            objective=HISTORICAL_OBJECTIVE,
            fusion=HISTORICAL_FUSION,
            budget=PRECOMMITTED_CONTINUATION,
            wandb_project=HISTORICAL_WANDB_PROJECT,
        )
        for stage in HISTORICAL_STAGES
    ),
    StageCandidate(
        stage=V2_GC_STAGE,
        objective=GRID_CONSISTENCY_OBJECTIVE,
        fusion=HISTORICAL_FUSION,
        budget=FIRST_SCREEN_HARD_CAP,
        wandb_project=V2_GC_WANDB_PROJECT,
    ),
    StageCandidate(
        stage=V2_GRD_STAGE,
        objective=GEOMETRY_RELATIONAL_OBJECTIVE,
        fusion=HISTORICAL_FUSION,
        budget=FIRST_SCREEN_HARD_CAP,
        wandb_project=V2_GRD_WANDB_PROJECT,
    ),
    StageCandidate(
        stage=V2_SCF_STAGE,
        objective=HISTORICAL_OBJECTIVE,
        fusion=SCALE_CALIBRATED_FUSION,
        budget=FIRST_SCREEN_HARD_CAP,
        wandb_project=V2_SCF_WANDB_PROJECT,
    ),
)
"""Every stage this repository can execute. A closed register.

C1 (V2-SCF) and C2 (V2-GRD) are added by appending entries here, each with its
own objective identity and its own `ScreeningBudget`. Nothing below reads a
candidate's name."""

_FIRST_SCREEN_CEILINGS = {
    "V2_GC_MAX_UPDATES": V2_GC_MAX_UPDATES,
    "V2_SCF_MAX_UPDATES": V2_SCF_MAX_UPDATES,
    "V2_GRD_MAX_UPDATES": V2_GRD_MAX_UPDATES,
}
if len(set(_FIRST_SCREEN_CEILINGS.values())) != 1:  # pragma: no cover - import guard
    raise AssertionError(
        f"the first-screen candidates declare different ceilings: "
        f"{_FIRST_SCREEN_CEILINGS}. They share one `FIRST_SCREEN_HARD_CAP` object "
        "precisely because a first screen compares candidates at a MATCHED budget; "
        "if they are ever meant to differ, each needs its own budget object and "
        "the comparison needs a recorded justification."
    )

_BY_STAGE = {candidate.stage: candidate for candidate in CANDIDATES}

_BY_IDENTITY = {candidate.identity: candidate.budget for candidate in CANDIDATES}
"""Keyed on `(objective_id, fusion_id)`.

Keyed on the objective alone until C1 existed, which was correct only while every
candidate had its own loss. C1 shares the historical objective, so an
objective-only key would hand it `PRECOMMITTED_CONTINUATION` and let it run to
40 000 -- the exact class of defect Audit 065 raised against C3."""

if len(_BY_STAGE) != len(CANDIDATES):  # pragma: no cover - import guard
    raise AssertionError("two candidates claim the same stage name")


def candidate_for_stage(stage: str) -> StageCandidate:
    """The candidate a stage name denotes. **Fails closed on an unknown stage.**

    This is THE objective-resolution mechanism. `execute_stage` uses it to decide
    which objective to train, and `smoke_check` uses the same call to decide
    which objective to exercise, so a smoke can never validate a different loss
    from the one the run would optimise.
    """
    try:
        return _BY_STAGE[stage]
    except KeyError:
        raise Stage1ContractViolation(
            f"unknown Stage-1 stage {stage!r}; the register holds "
            f"{sorted(_BY_STAGE)}. A stage with no registered candidate has no "
            "objective and no screening budget, so it is refused rather than "
            "defaulted into the historical one."
        ) from None


def objective_for_stage(stage: str) -> ObjectiveIdentity:
    """The objective identity a stage trains."""
    return candidate_for_stage(stage).objective


def budget_for_identity(objective_id: str, fusion_id: str) -> ScreeningBudget:
    """The screening budget a `(objective, fusion)` pair is governed by.

    **Fails closed.** Keyed on the identity rather than the stage because the
    trainer knows which experiment it is running -- `provenance.objective` and
    `provenance.fusion` -- and must not have to be told a stage name to enforce a
    ceiling. A checkpoint therefore carries everything needed to re-derive its own
    budget policy, with no registry lookup by name and no stage string stored.
    """
    try:
        return _BY_IDENTITY[(objective_id, fusion_id)]
    except KeyError:
        raise Stage1ContractViolation(
            f"the identity (objective={objective_id!r}, fusion={fusion_id!r}) has no "
            f"registered screening budget; the register holds {sorted(_BY_IDENTITY)}. "
            "A candidate whose budget nobody declared cannot be trained."
        ) from None


def candidate_for_identity(objective_id: str, fusion_id: str) -> StageCandidate:
    """The registered candidate for an `(objective, fusion)` pair. Fails closed.

    The inverse of `candidate_for_stage`, for a reader that has a checkpoint but
    no stage name -- a future Stage-2 loader, for instance, deciding which adapter
    to reconstruct.
    """
    for candidate in CANDIDATES:
        if candidate.identity == (objective_id, fusion_id):
            return candidate
    raise Stage1ContractViolation(
        f"no registered candidate has identity (objective={objective_id!r}, "
        f"fusion={fusion_id!r}); the register holds "
        f"{sorted(c.identity for c in CANDIDATES)}"
    )


def fusion_for_stage(stage: str) -> FusionIdentity:
    """The adapter architecture a stage trains."""
    return candidate_for_stage(stage).fusion


def wandb_project_for_stage(stage: str) -> str:
    """The W&B project a stage logs to. **OPERATIONAL ONLY, never scientific.**"""
    return candidate_for_stage(stage).wandb_project


__all__ = [
    "BudgetPolicyViolation",
    "CANDIDATES",
    "FIRST_SCREEN_HARD_CAP",
    "HISTORICAL_STAGES",
    "HISTORICAL_WANDB_PROJECT",
    "PRECOMMITTED_CONTINUATION",
    "ScreeningBudget",
    "StageCandidate",
    "V2_GC_WANDB_PROJECT",
    "V2_GRD_WANDB_PROJECT",
    "V2_SCF_WANDB_PROJECT",
    "budget_for_identity",
    "candidate_for_identity",
    "candidate_for_stage",
    "fusion_for_stage",
    "objective_for_stage",
    "wandb_project_for_stage",
]
