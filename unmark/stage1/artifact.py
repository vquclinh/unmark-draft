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

Two rules make that impossible, and both are implemented here:

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    PRECISION,
    STAGE1_PROTOCOL_VERSION,
)
from unmark.stage1.selection import (
    Candidate,
    SelectionViolation,
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
            winner = select_r(candidates, frozen)
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
