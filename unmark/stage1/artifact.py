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
            winner = select_learning_rate(candidates)
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
