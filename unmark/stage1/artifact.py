"""Stage-artifact identity and consumer validation. **Torch-free.**

A Stage-1 campaign is a chain: `lr-pilot` chooses the learning rate, `r-phase1`
runs at that frozen LR and chooses `r`, `final-main` runs the three seeds at
both. Each link hands the next a JSON artifact.

Until this module existed the consumer checked `stage` and `protocol_version`
and then *trusted* `artifact["selected"]["learning_rate"]`. A stale, foreign,
wrong-corpus, wrong-HEAD, wrong-inventory or hand-edited artifact with those two
strings right therefore drove the next stage, and every downstream command still
printed as if it were running the frozen protocol (Audit 031 B4 / Audit 032
MAJ1).

The rules that make that impossible are implemented here:

1. **Identity is compared against the CURRENT execution**, never against values
   read out of the artifact being validated. `CampaignIdentity` is built from
   the verified corpus, the resolved Git HEAD, the pinned backbone and the
   verified inventory of the run that is *about to start*; the artifact must
   match it.
2. **The selection is recomputed**, not read. `validate_selection_artifact`
   rebuilds the candidates and reruns the *production* `select_learning_rate` /
   `select_r`, so an edited winner and edited evidence that would change the
   winner are both refused. The locked-grid checks come along for free, because
   those are the same functions that enforce them.
3. A post-hoc LR-pilot author override is accepted only when it is explicit in
   the artifact, names the superseded locked-rule winner, and selects one of the
   real candidates. This keeps the scientific record honest without requiring
   the three expensive LR pilot runs to be repeated.
4. The post-hoc r-phase1 resource-bounded amendment is accepted only when it
   names its specific kind, preserves the original 20,000-update plan, records
   the 6500-update cutoff, carries the observed-window validation points for all
   five candidates, and recomputes to r=1 under the documented median-score
   rule. This keeps the partial sweep from being represented as a completed
   locked-rule result.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    INITIAL_MAX_UPDATES,
    PRECISION,
    R_PHASE1_GRID,
    SELECTION_SEED,
    STAGE1_PROTOCOL_VERSION,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import (
    Candidate,
    SelectionViolation,
    ValidationPoint,
    select_learning_rate,
    select_r,
)


class ArtifactViolation(Stage1ContractViolation):
    """Raised when a stage artifact does not belong to the current campaign."""


LR_PILOT_AUTHOR_OVERRIDE_KIND = "author_lr_override_after_validation_curve_review"
"""The only accepted post-hoc LR-pilot override kind.

It is deliberately specific. A generic "override" field would become a second
selection protocol by accident; this one records exactly the researcher action
that occurred after inspecting the W&B validation curves.
"""

LOCKED_LR_SELECTION_RULE = (
    "min(candidate.selected.score, candidate.selected.d_clean, "
    "candidate.learning_rate)"
)
"""Human-readable copy of the superseded production LR selector."""

LR_PILOT_OVERRIDE_FIELDS: tuple[str, ...] = (
    "kind",
    "author",
    "created_at",
    "selected_label",
    "selected_learning_rate",
    "superseded_locked_rule",
    "superseded_locked_rule_winner",
    "reason",
    "evidence",
)
"""Closed schema for the explicit LR-pilot author override block."""


R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND = (
    "author_r_override_after_resource_bounded_validation_review"
)
"""The only accepted post-hoc r-phase1 resource-bounded amendment kind."""

LOCKED_R_SELECTION_RULE = (
    "min(candidate.selected.score, candidate.selected.d_clean, candidate.r)"
)
"""Human-readable copy of the original completed-r-phase1 selector."""

RESOURCE_BOUNDED_R_COMPARISON_WINDOW: tuple[int, ...] = (
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
)
"""The observed-window updates used by the resource-bounded r amendment."""

RESOURCE_BOUNDED_R_CUTOFF_UPDATE = RESOURCE_BOUNDED_R_COMPARISON_WINDOW[-1]
"""The durable update at which all five r candidates were stopped."""

RESOURCE_BOUNDED_R_SELECTION_RULE = (
    "min(candidate_summary.median_score, "
    "candidate_summary.median_d_clean, candidate_summary.r)"
)
"""The documented resource-bounded r rule."""

RESOURCE_BOUNDED_R_PRIMARY_CRITERION = (
    "lower median validation/score over comparison_window"
)
RESOURCE_BOUNDED_R_SECONDARY_DIAGNOSTIC = (
    "lower median d_clean over comparison_window"
)
RESOURCE_BOUNDED_R_STABILITY_DIAGNOSTICS: tuple[str, ...] = (
    "score_range",
    "score_std",
)
RESOURCE_BOUNDED_R_SCORE_STD_KIND = "population_stdev_n"

RESOURCE_BOUNDED_R_SELECTED_R = 1.0
"""The one r this amendment kind may adopt.

Audit 047 is a record of a *specific* author decision, not a configurable
override mechanism. Without this pin a coherently edited artifact -- one whose
`selected`, `selected_label`, `selected_r` and recomputed evidence all agree on
some other r -- would validate, because every other check in this path only
proves internal consistency. Pinning the value is what makes the amendment a
transcript of the decision that was actually reviewed.
"""

RESOURCE_BOUNDED_R_FIXED_LEARNING_RATE = 1e-4
"""The frozen LR the resource-bounded r sweep ran under.

The historical locked LR selector preferred 3e-4; the adopted handoff is the
author override at 1e-4. A reissue at any other LR is a different experiment,
so it is refused here rather than silently carried into final-main.
"""

R_PHASE1_OVERRIDE_FIELDS: tuple[str, ...] = (
    "kind",
    "author",
    "created_at",
    "selected_label",
    "selected_r",
    "fixed_learning_rate",
    "resource_bounded_selection_rule",
    "original_locked_r_selection_rule",
    "original_planned_cap",
    "observed_cutoff_update",
    "comparison_window",
    "primary_criterion",
    "secondary_diagnostic",
    "stability_diagnostics",
    "stopped_resource_bounded",
    "global_optimum_claimed",
    "official_test_used",
    "downstream_score_used",
    "historical_tail_after_cutoff_used",
    "reason",
    "evidence",
)
"""Closed schema for the explicit resource-bounded r-phase1 amendment."""

R_PHASE1_OVERRIDE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "source_r_phase1_repository_head",
    "reissued_under_repository_head",
    "previous_artifact_repository_head",
    "previous_artifact_identity_repository_head",
    "telemetry_evidence",
    "candidate_summaries",
    "resource_bounded_order",
    "control_equivalence",
)

R_PHASE1_TELEMETRY_EVIDENCE_FIELDS: tuple[str, ...] = (
    "source_telemetry",
    "source_repository_head",
    "observed_cutoff_update",
    "events_parseable",
    "required_events_by_label",
)

R_PHASE1_TELEMETRY_REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "run_start",
    "train_progress_6500",
    "validation_6500",
    "checkpoint_6500",
    "checkpoint_name",
)

R_PHASE1_LAST_CHECKPOINT_NAME = "training-checkpoint-last.pt"

R_PHASE1_RESOURCE_SUMMARY_FIELDS: tuple[str, ...] = (
    "label",
    "r",
    "learning_rate",
    "source_checkpoint",
    "source_repository_head",
    "checkpoint_schema_version",
    "checkpoint_global_update",
    "checkpoint_cap",
    "observed_cutoff_update",
    "original_planned_cap",
    "comparison_window",
    "validation_points",
    "median_score",
    "mean_score",
    "median_d_clean",
    "score_range",
    "score_std",
    "score_std_kind",
    "score_at_cutoff",
)

R_PHASE1_CONTROL_EQUIVALENCE_FIELDS: tuple[str, ...] = (
    "kind",
    "r_phase1_candidate_label",
    "control_candidate_label",
    "control_source",
    "control_repository_head",
    "control_run_seed",
    "comparison_window",
    "metrics_compared",
    "max_abs_validation_metric_difference_by_update",
)

R_PHASE1_CONTROL_EQUIVALENCE_KIND = (
    "fused_r1_matches_historical_lr_pilot_r1_window"
)

R_PHASE1_CONTROL_METRICS: tuple[str, ...] = (
    "d_clean",
    "score",
    *(f"distances.{condition}" for condition in VALIDATION_CONDITIONS),
)


IDENTITY_FIELDS: tuple[str, ...] = (
    "repository_head",
    "protocol_version",
    "corpus_manifest_digest",
    "encoder_checkpoint",
    "encoder_revision",
    "precision",
    "inventory_source_name",
    "inventory_source_revision",
    "inventory_sha256",
)
"""Everything that must be identical for two stages to belong to one campaign.

Each entry answers "would a difference here make the two stages different
experiments?". The commit that produced the code, the protocol, the exact
prepared-corpus membership, the pinned backbone and its revision, the numeric
precision, and the syllable inventory that decides every corruption denominator
(D-S1A-008) all qualify. Wall-clock, worker count and GPU name do not, and are
deliberately absent -- they live in the operational execution fingerprint.
"""


@dataclass(frozen=True)
class CampaignIdentity:
    """The identity of one Stage-1 campaign, derived from CURRENT trusted inputs."""

    repository_head: str
    corpus_manifest_digest: str
    encoder_checkpoint: str = ENCODER_CHECKPOINT
    encoder_revision: str = ENCODER_REVISION
    precision: str = PRECISION
    protocol_version: str = STAGE1_PROTOCOL_VERSION
    inventory_source_name: str | None = None
    inventory_source_revision: str | None = None
    inventory_sha256: str | None = None

    @classmethod
    def from_inputs(
        cls,
        *,
        repository_head: str,
        corpus_manifest_digest: str,
        encoder_revision: str,
        inventory: Any = None,
    ) -> "CampaignIdentity":
        """Build from the verified inputs of the run that is about to start."""
        return cls(
            repository_head=repository_head,
            corpus_manifest_digest=corpus_manifest_digest,
            encoder_revision=encoder_revision,
            inventory_source_name=getattr(inventory, "source_name", None),
            inventory_source_revision=getattr(inventory, "source_revision", None),
            inventory_sha256=getattr(inventory, "sha256", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in IDENTITY_FIELDS}

    def require_match(self, recorded: Mapping[str, Any], *, what: str) -> None:
        """Refuse an artifact that does not describe THIS campaign.

        `recorded` comes from the artifact; `self` comes from current verified
        inputs. The direction matters: expected values must never be taken from
        the document being validated.
        """
        if not isinstance(recorded, Mapping):
            raise ArtifactViolation(
                f"{what} carries no identity block; it was written by a build that "
                "did not bind its campaign identity and cannot be trusted to belong "
                "to this campaign"
            )
        missing = [f for f in IDENTITY_FIELDS if f not in recorded]
        if missing:
            raise ArtifactViolation(f"{what} identity is missing {missing}")
        mine = self.to_dict()
        differing = {
            field: (mine[field], recorded[field])
            for field in IDENTITY_FIELDS
            if mine[field] != recorded[field]
        }
        if differing:
            detail = "; ".join(
                f"{field}: current {current!r} != artifact {found!r}"
                for field, (current, found) in sorted(differing.items())
            )
            raise ArtifactViolation(
                f"{what} belongs to a different campaign -- {detail}. A stage artifact "
                "may only drive a stage that shares its code, protocol, corpus, "
                "backbone, precision and syllable inventory."
            )


def validate_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_stage: str,
    identity: CampaignIdentity,
    what: str,
) -> Candidate:
    """Validate a stage artifact and RECOMPUTE its winner. Returns the winner.

    `select_learning_rate` / `select_r` are the same functions that produced the
    artifact, so reusing them here means the consumer cannot drift from the
    producer, and the locked-grid rules (exact grid, no missing/duplicate/extra
    candidate, single frozen LR) are enforced by construction rather than
    restated approximately.
    """
    if not isinstance(artifact, Mapping):
        raise ArtifactViolation(f"{what} is not a JSON object")
    if artifact.get("stage") != expected_stage:
        raise ArtifactViolation(
            f"{what} is a {artifact.get('stage')!r} artifact; {expected_stage!r} is required"
        )
    if artifact.get("protocol_version") != STAGE1_PROTOCOL_VERSION:
        raise ArtifactViolation(
            f"{what} was produced under protocol {artifact.get('protocol_version')!r}, "
            f"not {STAGE1_PROTOCOL_VERSION!r}"
        )
    for field in ("official_test_used", "downstream_score_used"):
        if artifact.get(field, False) is not False:
            raise ArtifactViolation(
                f"{what} {field} must be false; Stage-1 handoff selection cannot "
                "consume official TEST or downstream scores"
            )
    identity.require_match(artifact.get("identity"), what=what)

    raw_candidates = artifact.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ArtifactViolation(f"{what} records no candidates to reselect from")
    try:
        candidates = [Candidate.from_dict(c) for c in raw_candidates]
    except SelectionViolation as error:
        raise ArtifactViolation(f"{what} has an unusable candidate: {error}") from error

    recorded = artifact.get("selected")
    if not isinstance(recorded, Mapping):
        raise ArtifactViolation(f"{what} records no selected candidate")

    try:
        if expected_stage == "lr_pilot":
            locked_winner = select_learning_rate(candidates)
            override = artifact.get("selection_override")
            if override is not None:
                return _validate_lr_pilot_author_override(
                    override=override,
                    candidates=candidates,
                    locked_winner=locked_winner,
                    recorded=recorded,
                    what=what,
                )
            winner = locked_winner
        elif expected_stage == "r_phase1":
            frozen = candidates[0].learning_rate
            locked_winner = select_r(candidates, frozen)
            override = artifact.get("selection_override")
            if override is not None:
                return _validate_r_phase1_resource_bounded_author_override(
                    override=override,
                    candidates=candidates,
                    locked_winner=locked_winner,
                    recorded=recorded,
                    identity=identity,
                    what=what,
                )
            winner = locked_winner
        else:
            raise ArtifactViolation(
                f"{expected_stage!r} produces no selection to revalidate"
            )
    except SelectionViolation as error:
        raise ArtifactViolation(
            f"{what} does not satisfy the locked selection contract: {error}"
        ) from error

    expected = winner.to_dict()
    if dict(recorded) != expected:
        raise ArtifactViolation(
            f"{what} records selected {dict(recorded)!r}, but rerunning the locked "
            f"selection rule over its own candidate evidence yields {expected!r}. "
            "Either the recorded winner or the evidence behind it was altered."
        )
    return winner


def resource_bounded_r_summary(
    *,
    label: str,
    r: float,
    learning_rate: float,
    points: Sequence[ValidationPoint],
    source_checkpoint: str,
    source_repository_head: str,
    checkpoint_schema_version: str,
    checkpoint_global_update: int,
    checkpoint_cap: int,
) -> dict[str, Any]:
    """Build the canonical observed-window r summary from validation points."""
    if tuple(point.update for point in points) != RESOURCE_BOUNDED_R_COMPARISON_WINDOW:
        raise ArtifactViolation(
            f"{label} resource-bounded points must be exactly "
            f"{list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW)}, got "
            f"{[point.update for point in points]}"
        )
    scores = [point.score for point in points]
    d_clean = [point.d_clean for point in points]
    return {
        "label": label,
        "r": float(r),
        "learning_rate": float(learning_rate),
        "source_checkpoint": source_checkpoint,
        "source_repository_head": source_repository_head,
        "checkpoint_schema_version": checkpoint_schema_version,
        "checkpoint_global_update": int(checkpoint_global_update),
        "checkpoint_cap": int(checkpoint_cap),
        "observed_cutoff_update": RESOURCE_BOUNDED_R_CUTOFF_UPDATE,
        "original_planned_cap": INITIAL_MAX_UPDATES,
        "comparison_window": list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW),
        "validation_points": [point.to_dict() for point in points],
        "median_score": statistics.median(scores),
        "mean_score": statistics.fmean(scores),
        "median_d_clean": statistics.median(d_clean),
        "score_range": max(scores) - min(scores),
        "score_std": statistics.pstdev(scores),
        "score_std_kind": RESOURCE_BOUNDED_R_SCORE_STD_KIND,
        "score_at_cutoff": scores[-1],
    }


def resource_bounded_r_order(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Labels ordered by the documented resource-bounded r rule."""
    return [
        str(summary["label"])
        for summary in sorted(
            summaries,
            key=lambda summary: (
                float(summary["median_score"]),
                float(summary["median_d_clean"]),
                float(summary["r"]),
            ),
        )
    ]


def _validate_r_phase1_resource_bounded_author_override(
    *,
    override: Any,
    candidates: list[Candidate],
    locked_winner: Candidate,
    recorded: Mapping[str, Any],
    identity: CampaignIdentity,
    what: str,
) -> Candidate:
    """Validate the supported resource-bounded r-phase1 amendment."""
    if not isinstance(override, Mapping):
        raise ArtifactViolation(f"{what} selection_override is not a JSON object")
    missing = [field for field in R_PHASE1_OVERRIDE_FIELDS if field not in override]
    if missing:
        raise ArtifactViolation(f"{what} selection_override is missing {missing}")
    unknown = sorted(set(override) - set(R_PHASE1_OVERRIDE_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} selection_override carries unknown field(s) {unknown}; the "
            "resource-bounded r override schema is closed"
        )
    if override["kind"] != R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND:
        raise ArtifactViolation(
            f"{what} selection_override kind {override['kind']!r} is not "
            f"{R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND!r}"
        )
    if override["resource_bounded_selection_rule"] != RESOURCE_BOUNDED_R_SELECTION_RULE:
        raise ArtifactViolation(
            f"{what} selection_override names resource-bounded rule "
            f"{override['resource_bounded_selection_rule']!r}, not "
            f"{RESOURCE_BOUNDED_R_SELECTION_RULE!r}"
        )
    if override["original_locked_r_selection_rule"] != LOCKED_R_SELECTION_RULE:
        raise ArtifactViolation(
            f"{what} selection_override names original r rule "
            f"{override['original_locked_r_selection_rule']!r}, not "
            f"{LOCKED_R_SELECTION_RULE!r}"
        )
    for field in ("author", "created_at", "selected_label", "reason"):
        value = override[field]
        if not isinstance(value, str) or not value.strip():
            raise ArtifactViolation(
                f"{what} selection_override field {field!r} must be a non-empty string"
            )
    if _sequence(override["comparison_window"], f"{what} comparison_window") != RESOURCE_BOUNDED_R_COMPARISON_WINDOW:
        raise ArtifactViolation(
            f"{what} selection_override comparison_window must be "
            f"{list(RESOURCE_BOUNDED_R_COMPARISON_WINDOW)}"
        )
    if _integer(override["observed_cutoff_update"], f"{what} observed_cutoff_update") != RESOURCE_BOUNDED_R_CUTOFF_UPDATE:
        raise ArtifactViolation(
            f"{what} observed_cutoff_update must be "
            f"{RESOURCE_BOUNDED_R_CUTOFF_UPDATE}"
        )
    if _integer(override["original_planned_cap"], f"{what} original_planned_cap") != INITIAL_MAX_UPDATES:
        raise ArtifactViolation(
            f"{what} original_planned_cap must preserve {INITIAL_MAX_UPDATES}"
        )
    if override["primary_criterion"] != RESOURCE_BOUNDED_R_PRIMARY_CRITERION:
        raise ArtifactViolation(f"{what} primary_criterion is not the documented one")
    if override["secondary_diagnostic"] != RESOURCE_BOUNDED_R_SECONDARY_DIAGNOSTIC:
        raise ArtifactViolation(f"{what} secondary_diagnostic is not the documented one")
    if _sequence(override["stability_diagnostics"], f"{what} stability_diagnostics") != RESOURCE_BOUNDED_R_STABILITY_DIAGNOSTICS:
        raise ArtifactViolation(
            f"{what} stability_diagnostics must be "
            f"{list(RESOURCE_BOUNDED_R_STABILITY_DIAGNOSTICS)}"
        )
    expected_bools = {
        "stopped_resource_bounded": True,
        "global_optimum_claimed": False,
        "official_test_used": False,
        "downstream_score_used": False,
        "historical_tail_after_cutoff_used": False,
    }
    for field, expected in expected_bools.items():
        if override[field] is not expected:
            raise ArtifactViolation(f"{what} selection_override {field} must be {expected}")

    selected_r = _number(override["selected_r"], f"{what} selected_r")
    fixed_lr = _number(
        override["fixed_learning_rate"], f"{what} fixed_learning_rate"
    )
    matches = [
        candidate for candidate in candidates
        if candidate.r == selected_r
        and candidate.learning_rate == fixed_lr
        and candidate.label == override["selected_label"]
    ]
    if len(matches) != 1:
        raise ArtifactViolation(
            f"{what} selection_override selects {override['selected_label']!r} at "
            f"r={selected_r!r}, LR={fixed_lr!r}, but that is not exactly one "
            "r-phase1 candidate"
        )
    winner = matches[0]
    if dict(recorded) != winner.to_dict():
        raise ArtifactViolation(
            f"{what} records selected {dict(recorded)!r}, but the explicit "
            f"resource-bounded selection_override selects {winner.to_dict()!r}"
        )

    summaries = _validate_r_phase1_resource_evidence(
        override["evidence"],
        candidates=candidates,
        fixed_learning_rate=fixed_lr,
        selected_r=selected_r,
        selected_label=override["selected_label"],
        identity=identity,
        what=what,
    )
    order = resource_bounded_r_order(summaries)
    if order[0] != winner.label:
        raise ArtifactViolation(
            f"{what} resource-bounded evidence selects {order[0]!r}, not "
            f"{winner.label!r}"
        )
    # Last gate, deliberately after the evidence was recomputed: everything above
    # proves the artifact is INTERNALLY consistent, which a coherent forgery also
    # is. These two pins are what tie this amendment kind to the one decision
    # Audit 047 records.
    if selected_r != RESOURCE_BOUNDED_R_SELECTED_R:
        raise ArtifactViolation(
            f"{what} selection_override selected_r must be "
            f"{RESOURCE_BOUNDED_R_SELECTED_R!r} for kind "
            f"{R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND!r}, got {selected_r!r}; "
            "this amendment kind records one specific author decision and is not a "
            "general-purpose r override"
        )
    if fixed_lr != RESOURCE_BOUNDED_R_FIXED_LEARNING_RATE:
        raise ArtifactViolation(
            f"{what} selection_override fixed_learning_rate must be "
            f"{RESOURCE_BOUNDED_R_FIXED_LEARNING_RATE!r} for kind "
            f"{R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND!r}, got {fixed_lr!r}"
        )
    # Kept as a named argument to prove the original locked selector was still
    # recomputed before the override was considered.
    if not isinstance(locked_winner, Candidate):  # pragma: no cover - defensive
        raise ArtifactViolation(f"{what} could not recompute the original r selector")
    return winner


def _validate_r_phase1_resource_evidence(
    evidence: Any,
    *,
    candidates: list[Candidate],
    fixed_learning_rate: float,
    selected_r: float,
    selected_label: str,
    identity: CampaignIdentity,
    what: str,
) -> list[dict[str, Any]]:
    if not isinstance(evidence, Mapping):
        raise ArtifactViolation(f"{what} selection_override evidence must be a JSON object")
    missing = [field for field in R_PHASE1_OVERRIDE_EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise ArtifactViolation(f"{what} selection_override evidence is missing {missing}")
    unknown = sorted(set(evidence) - set(R_PHASE1_OVERRIDE_EVIDENCE_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} selection_override evidence carries unknown field(s) {unknown}; "
            "the resource-bounded r evidence schema is closed"
        )
    source_head = _full_sha(
        evidence["source_r_phase1_repository_head"],
        f"{what} source_r_phase1_repository_head",
    )
    reissued_head = _full_sha(
        evidence["reissued_under_repository_head"],
        f"{what} reissued_under_repository_head",
    )
    if reissued_head != identity.repository_head:
        raise ArtifactViolation(
            f"{what} evidence reissued_under_repository_head {reissued_head!r} "
            f"does not match artifact identity {identity.repository_head!r}"
        )
    for nullable in ("previous_artifact_repository_head", "previous_artifact_identity_repository_head"):
        value = evidence[nullable]
        if value is not None:
            _full_sha(value, f"{what} {nullable}")
    _validate_r_phase1_telemetry_evidence(
        evidence["telemetry_evidence"],
        source_head=source_head,
        what=what,
    )

    raw_summaries = evidence["candidate_summaries"]
    if not isinstance(raw_summaries, list):
        raise ArtifactViolation(f"{what} candidate_summaries must be a list")
    if len(raw_summaries) != len(R_PHASE1_GRID):
        raise ArtifactViolation(
            f"{what} candidate_summaries must cover all {len(R_PHASE1_GRID)} "
            "r candidates"
        )
    by_label = {candidate.label: candidate for candidate in candidates}
    summaries = [
        _validate_r_phase1_candidate_summary(
            raw,
            by_label=by_label,
            fixed_learning_rate=fixed_learning_rate,
            source_head=source_head,
            what=what,
        )
        for raw in raw_summaries
    ]
    values = sorted(float(summary["r"]) for summary in summaries)
    if values != sorted(R_PHASE1_GRID):
        raise ArtifactViolation(
            f"{what} candidate_summaries cover r grid {values}, not "
            f"{sorted(R_PHASE1_GRID)}"
        )
    labels = [summary["label"] for summary in summaries]
    if len(set(labels)) != len(labels):
        raise ArtifactViolation(f"{what} candidate_summaries contain duplicate labels")
    order = resource_bounded_r_order(summaries)
    if evidence["resource_bounded_order"] != order:
        raise ArtifactViolation(
            f"{what} records resource_bounded_order {evidence['resource_bounded_order']!r}, "
            f"but recomputing the documented rule yields {order!r}"
        )
    if order[0] != selected_label:
        raise ArtifactViolation(
            f"{what} resource-bounded summaries select {order[0]!r}, not "
            f"{selected_label!r}"
        )
    selected = next(summary for summary in summaries if summary["label"] == selected_label)
    if float(selected["r"]) != selected_r:
        raise ArtifactViolation(
            f"{what} selected summary r={selected['r']!r} does not match "
            f"override selected_r={selected_r!r}"
        )
    _validate_r1_control_equivalence(evidence["control_equivalence"], what=what)
    return summaries


def _validate_r_phase1_telemetry_evidence(
    telemetry: Any,
    *,
    source_head: str,
    what: str,
) -> None:
    if not isinstance(telemetry, Mapping):
        raise ArtifactViolation(f"{what} telemetry_evidence must be a JSON object")
    missing = [
        field for field in R_PHASE1_TELEMETRY_EVIDENCE_FIELDS
        if field not in telemetry
    ]
    if missing:
        raise ArtifactViolation(f"{what} telemetry_evidence is missing {missing}")
    unknown = sorted(set(telemetry) - set(R_PHASE1_TELEMETRY_EVIDENCE_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} telemetry_evidence carries unknown field(s) {unknown}; the "
            "telemetry evidence schema is closed"
        )
    source = telemetry["source_telemetry"]
    if not isinstance(source, str) or not source.strip():
        raise ArtifactViolation(f"{what} telemetry_evidence source_telemetry is empty")
    telemetry_head = _full_sha(
        telemetry["source_repository_head"],
        f"{what} telemetry source_repository_head",
    )
    if telemetry_head != source_head:
        raise ArtifactViolation(
            f"{what} telemetry source head {telemetry_head!r} does not match "
            f"source r-phase1 head {source_head!r}"
        )
    if _integer(telemetry["observed_cutoff_update"], f"{what} telemetry cutoff") != RESOURCE_BOUNDED_R_CUTOFF_UPDATE:
        raise ArtifactViolation(f"{what} telemetry observed_cutoff_update is not 6500")
    if _integer(telemetry["events_parseable"], f"{what} telemetry events_parseable") <= 0:
        raise ArtifactViolation(f"{what} telemetry has no parseable events")
    by_label = telemetry["required_events_by_label"]
    if not isinstance(by_label, Mapping):
        raise ArtifactViolation(
            f"{what} telemetry required_events_by_label must be a JSON object"
        )
    expected_labels = {f"r={r:g}" for r in R_PHASE1_GRID}
    if set(by_label) != expected_labels:
        raise ArtifactViolation(
            f"{what} telemetry labels {sorted(by_label)} do not cover "
            f"{sorted(expected_labels)}"
        )
    for label, raw in by_label.items():
        if not isinstance(raw, Mapping):
            raise ArtifactViolation(f"{what} telemetry block for {label} is not an object")
        missing = [
            field for field in R_PHASE1_TELEMETRY_REQUIRED_EVENT_FIELDS
            if field not in raw
        ]
        if missing:
            raise ArtifactViolation(f"{what} telemetry block for {label} is missing {missing}")
        unknown = sorted(set(raw) - set(R_PHASE1_TELEMETRY_REQUIRED_EVENT_FIELDS))
        if unknown:
            raise ArtifactViolation(
                f"{what} telemetry block for {label} has unknown field(s) {unknown}"
            )
        for field in R_PHASE1_TELEMETRY_REQUIRED_EVENT_FIELDS[:-1]:
            if raw[field] is not True:
                raise ArtifactViolation(f"{what} telemetry {label} {field} must be true")
        if raw["checkpoint_name"] != R_PHASE1_LAST_CHECKPOINT_NAME:
            raise ArtifactViolation(
                f"{what} telemetry {label} checkpoint_name must be "
                f"{R_PHASE1_LAST_CHECKPOINT_NAME!r}"
            )


def _validate_r_phase1_candidate_summary(
    summary: Any,
    *,
    by_label: Mapping[str, Candidate],
    fixed_learning_rate: float,
    source_head: str,
    what: str,
) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ArtifactViolation(f"{what} candidate summary is not a JSON object")
    missing = [field for field in R_PHASE1_RESOURCE_SUMMARY_FIELDS if field not in summary]
    if missing:
        raise ArtifactViolation(f"{what} candidate summary is missing {missing}")
    unknown = sorted(set(summary) - set(R_PHASE1_RESOURCE_SUMMARY_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} candidate summary carries unknown field(s) {unknown}; the "
            "resource-bounded r summary schema is closed"
        )
    label = str(summary["label"])
    if label not in by_label:
        raise ArtifactViolation(f"{what} candidate summary {label!r} is not a candidate")
    candidate = by_label[label]
    r = _number(summary["r"], f"{what} {label} r")
    learning_rate = _number(summary["learning_rate"], f"{what} {label} learning_rate")
    if r != candidate.r:
        raise ArtifactViolation(f"{what} {label} summary r does not match candidate")
    if learning_rate != candidate.learning_rate or learning_rate != fixed_learning_rate:
        raise ArtifactViolation(
            f"{what} {label} summary learning_rate does not match the frozen LR"
        )
    if summary["source_repository_head"] != source_head:
        raise ArtifactViolation(f"{what} {label} source_repository_head mismatch")
    for field in ("source_checkpoint", "source_repository_head", "checkpoint_schema_version"):
        value = summary[field]
        if not isinstance(value, str) or not value.strip():
            raise ArtifactViolation(f"{what} {label} {field} must be a non-empty string")
    checkpoint_global_update = _integer(
        summary["checkpoint_global_update"], f"{what} {label} checkpoint_global_update"
    )
    observed_cutoff = _integer(
        summary["observed_cutoff_update"], f"{what} {label} observed_cutoff_update"
    )
    checkpoint_cap = _integer(summary["checkpoint_cap"], f"{what} {label} checkpoint_cap")
    original_planned_cap = _integer(
        summary["original_planned_cap"], f"{what} {label} original_planned_cap"
    )
    if checkpoint_global_update != RESOURCE_BOUNDED_R_CUTOFF_UPDATE:
        raise ArtifactViolation(f"{what} {label} checkpoint_global_update is not 6500")
    if observed_cutoff != RESOURCE_BOUNDED_R_CUTOFF_UPDATE:
        raise ArtifactViolation(f"{what} {label} observed_cutoff_update is not 6500")
    if checkpoint_cap != INITIAL_MAX_UPDATES:
        raise ArtifactViolation(f"{what} {label} checkpoint_cap is not the original 20000")
    if original_planned_cap != INITIAL_MAX_UPDATES:
        raise ArtifactViolation(f"{what} {label} original_planned_cap is not 20000")
    if _sequence(summary["comparison_window"], f"{what} {label} comparison_window") != RESOURCE_BOUNDED_R_COMPARISON_WINDOW:
        raise ArtifactViolation(f"{what} {label} comparison_window is wrong")
    if summary["score_std_kind"] != RESOURCE_BOUNDED_R_SCORE_STD_KIND:
        raise ArtifactViolation(f"{what} {label} score_std_kind is wrong")

    raw_points = summary["validation_points"]
    if not isinstance(raw_points, list):
        raise ArtifactViolation(f"{what} {label} validation_points must be a list")
    try:
        points = [ValidationPoint.from_dict(point) for point in raw_points]
    except SelectionViolation as error:
        raise ArtifactViolation(f"{what} {label} has an unusable validation point: {error}") from error
    recomputed = resource_bounded_r_summary(
        label=label,
        r=r,
        learning_rate=learning_rate,
        points=points,
        source_checkpoint=summary["source_checkpoint"],
        source_repository_head=summary["source_repository_head"],
        checkpoint_schema_version=summary["checkpoint_schema_version"],
        checkpoint_global_update=checkpoint_global_update,
        checkpoint_cap=checkpoint_cap,
    )
    for field in R_PHASE1_RESOURCE_SUMMARY_FIELDS:
        if field == "source_checkpoint":
            # The path is provenance only; the checkpoint-to-artifact helper
            # verifies it exists before writing. A relocated artifact should
            # still validate internally for final-main.
            continue
        if field in {
            "median_score",
            "mean_score",
            "median_d_clean",
            "score_range",
            "score_std",
            "score_at_cutoff",
        }:
            if not math.isclose(
                float(summary[field]),
                float(recomputed[field]),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ArtifactViolation(
                    f"{what} {label} {field}={summary[field]!r} but recomputes "
                    f"to {recomputed[field]!r}"
                )
        elif summary[field] != recomputed[field]:
            raise ArtifactViolation(
                f"{what} {label} {field}={summary[field]!r} but recomputes to "
                f"{recomputed[field]!r}"
            )
    return dict(summary)


def _validate_r1_control_equivalence(control: Any, *, what: str) -> None:
    if not isinstance(control, Mapping):
        raise ArtifactViolation(f"{what} control_equivalence must be a JSON object")
    missing = [field for field in R_PHASE1_CONTROL_EQUIVALENCE_FIELDS if field not in control]
    if missing:
        raise ArtifactViolation(f"{what} control_equivalence is missing {missing}")
    unknown = sorted(set(control) - set(R_PHASE1_CONTROL_EQUIVALENCE_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} control_equivalence carries unknown field(s) {unknown}; the "
            "control schema is closed"
        )
    if control["kind"] != R_PHASE1_CONTROL_EQUIVALENCE_KIND:
        raise ArtifactViolation(f"{what} control_equivalence kind is wrong")
    if control["r_phase1_candidate_label"] != "r=1":
        raise ArtifactViolation(f"{what} control_equivalence must name r=1")
    for field in ("control_candidate_label", "control_source"):
        if not isinstance(control[field], str) or not control[field]:
            raise ArtifactViolation(f"{what} {field} must be non-empty")
    _full_sha(control["control_repository_head"], f"{what} control_repository_head")
    # The control is the historical LR-pilot leg, which ran at the selection seed.
    # Re-checked here so the pin survives into whatever final-main validates,
    # rather than living only in the builder that produced this block.
    if _integer(control["control_run_seed"], f"{what} control_run_seed") != SELECTION_SEED:
        raise ArtifactViolation(
            f"{what} control_equivalence control_run_seed "
            f"{control['control_run_seed']!r} is not the historical LR-pilot "
            f"selection seed {SELECTION_SEED}"
        )
    if _sequence(control["comparison_window"], f"{what} control comparison_window") != RESOURCE_BOUNDED_R_COMPARISON_WINDOW:
        raise ArtifactViolation(f"{what} control_equivalence comparison_window is wrong")
    if _sequence(control["metrics_compared"], f"{what} metrics_compared") != R_PHASE1_CONTROL_METRICS:
        raise ArtifactViolation(f"{what} control_equivalence metrics_compared is wrong")
    diffs = control["max_abs_validation_metric_difference_by_update"]
    if not isinstance(diffs, Mapping):
        raise ArtifactViolation(
            f"{what} max_abs_validation_metric_difference_by_update must be a JSON object"
        )
    expected_keys = {str(update) for update in RESOURCE_BOUNDED_R_COMPARISON_WINDOW}
    if set(diffs) != expected_keys:
        raise ArtifactViolation(f"{what} control_equivalence update keys are wrong")
    for update, value in diffs.items():
        if float(value) != 0.0:
            raise ArtifactViolation(
                f"{what} control equivalence at update {update} is {value!r}, not 0"
            )


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactViolation(f"{what} must be numeric")
    return float(value)


def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactViolation(f"{what} must be an integer")
    return value


def _sequence(value: Any, what: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ArtifactViolation(f"{what} must be a JSON array")
    return tuple(value)


def _full_sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ArtifactViolation(f"{what} must be a full 40-character commit sha")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactViolation(f"{what} must be a hexadecimal commit sha") from error
    return value.lower()


def _validate_lr_pilot_author_override(
    *,
    override: Any,
    candidates: list[Candidate],
    locked_winner: Candidate,
    recorded: Mapping[str, Any],
    what: str,
) -> Candidate:
    """Validate the one supported post-hoc LR-pilot override.

    The override does not pretend the locked selector chose a different LR. It
    records that the author rejected the locked-rule winner after inspecting
    validation curves and picked another completed LR-pilot candidate.
    """
    if not isinstance(override, Mapping):
        raise ArtifactViolation(f"{what} selection_override is not a JSON object")
    missing = [field for field in LR_PILOT_OVERRIDE_FIELDS if field not in override]
    if missing:
        raise ArtifactViolation(f"{what} selection_override is missing {missing}")
    unknown = sorted(set(override) - set(LR_PILOT_OVERRIDE_FIELDS))
    if unknown:
        raise ArtifactViolation(
            f"{what} selection_override carries unknown field(s) {unknown}; the "
            "override schema is closed"
        )
    if override["kind"] != LR_PILOT_AUTHOR_OVERRIDE_KIND:
        raise ArtifactViolation(
            f"{what} selection_override kind {override['kind']!r} is not "
            f"{LR_PILOT_AUTHOR_OVERRIDE_KIND!r}"
        )
    if override["superseded_locked_rule"] != LOCKED_LR_SELECTION_RULE:
        raise ArtifactViolation(
            f"{what} selection_override names superseded rule "
            f"{override['superseded_locked_rule']!r}, not {LOCKED_LR_SELECTION_RULE!r}"
        )
    recorded_locked_winner = override["superseded_locked_rule_winner"]
    if not isinstance(recorded_locked_winner, Mapping):
        raise ArtifactViolation(
            f"{what} selection_override superseded_locked_rule_winner must be a JSON object"
        )
    if dict(recorded_locked_winner) != locked_winner.to_dict():
        raise ArtifactViolation(
            f"{what} selection_override does not preserve the locked-rule winner; "
            f"rerunning the locked selection yields {locked_winner.to_dict()!r}"
        )
    for field in ("author", "created_at", "selected_label", "reason"):
        value = override[field]
        if not isinstance(value, str) or not value.strip():
            raise ArtifactViolation(
                f"{what} selection_override field {field!r} must be a non-empty string"
            )
    selected_lr = override["selected_learning_rate"]
    if isinstance(selected_lr, bool) or not isinstance(selected_lr, (int, float)):
        raise ArtifactViolation(
            f"{what} selection_override selected_learning_rate must be numeric"
        )
    selected_lr = float(selected_lr)
    if not isinstance(override["evidence"], Mapping):
        raise ArtifactViolation(f"{what} selection_override evidence must be a JSON object")

    matches = [
        candidate for candidate in candidates
        if candidate.learning_rate == selected_lr and candidate.label == override["selected_label"]
    ]
    if len(matches) != 1:
        raise ArtifactViolation(
            f"{what} selection_override selects {override['selected_label']!r} at "
            f"LR {selected_lr!r}, but that is not exactly one completed LR-pilot candidate"
        )
    winner = matches[0]
    if winner.to_dict() == locked_winner.to_dict():
        raise ArtifactViolation(
            f"{what} selection_override selects the locked-rule winner; no override "
            "is needed and the artifact would be ambiguous"
        )
    if dict(recorded) != winner.to_dict():
        raise ArtifactViolation(
            f"{what} records selected {dict(recorded)!r}, but the explicit "
            f"selection_override selects {winner.to_dict()!r}"
        )
    return winner
