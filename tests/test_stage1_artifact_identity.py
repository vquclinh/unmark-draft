"""Stage-artifact handoff integrity. **Torch-free.**

Audit 031 B4 / Audit 032 MAJ1: `_load_selection` checked `stage` and
`protocol_version` and then downstream stages read
`artifact["selected"]["learning_rate"]` on trust. Any stale, foreign,
wrong-corpus, wrong-HEAD, wrong-inventory or hand-edited artifact carrying those
two strings could therefore drive `r-phase1` or `final-main`, and every
downstream command would print as though it were running the frozen protocol.

Two properties are tested here, and they are what make the handoff safe:

1. identity is compared against the **current** campaign, never against values
   taken from the artifact being validated;
2. the winner is **recomputed** with the production selection functions, so an
   edited scalar and edited evidence are both caught.

The grid rules (exact locked grid, no missing/duplicate/extra candidate, one
frozen LR) come along for free, because the recomputation calls the same
`select_learning_rate` / `select_r` that produced the artifact.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.artifact import (  # noqa: E402
    IDENTITY_FIELDS,
    LOCKED_LR_SELECTION_RULE,
    LR_PILOT_AUTHOR_OVERRIDE_KIND,
    ArtifactViolation,
    CampaignIdentity,
    validate_selection_artifact,
)
from unmark.stage1.protocol import (  # noqa: E402
    ENCODER_REVISION,
    LR_PILOT_GRID,
    LR_PILOT_R,
    R_PHASE1_GRID,
    STAGE1_PROTOCOL_VERSION,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import Candidate, ValidationPoint  # noqa: E402

HEAD = "a" * 40
DIGEST = "d" * 64

INVENTORY = types.SimpleNamespace(
    source_name="vn-syllables-v1",
    source_revision="r" * 12,
    sha256="c" * 64,
)


def identity(**overrides) -> CampaignIdentity:
    base = CampaignIdentity.from_inputs(
        repository_head=HEAD,
        corpus_manifest_digest=DIGEST,
        encoder_revision=ENCODER_REVISION,
        inventory=INVENTORY,
    )
    return CampaignIdentity(**{**base.__dict__, **overrides})


def point(worst: float) -> ValidationPoint:
    return ValidationPoint(
        update=500,
        distances={c: worst for c in VALIDATION_CONDITIONS},
        d_clean=worst / 2.0,
    )


def lr_candidates(scores=None) -> list[Candidate]:
    """One candidate per locked LR, all at the pilot's frozen r."""
    scores = scores or {1e-4: 0.5, 3e-4: 0.2, 1e-3: 0.7}
    return [
        Candidate(label=f"lr={lr:g}", learning_rate=lr, r=LR_PILOT_R,
                  selected=point(scores[lr]))
        for lr in LR_PILOT_GRID
    ]


def r_candidates(frozen_lr=3e-4) -> list[Candidate]:
    scores = {0.25: 0.6, 0.5: 0.4, 1.0: 0.2, 2.0: 0.5, 4.0: 0.8}
    return [
        Candidate(label=f"r={r:g}", learning_rate=frozen_lr, r=r,
                  selected=point(scores[r]))
        for r in R_PHASE1_GRID
    ]


def artifact_for(stage, candidates, *, ident=None, selected=None) -> dict:
    from unmark.stage1.selection import select_learning_rate, select_r

    ident = ident or identity()
    if selected is None:
        winner = (select_learning_rate(candidates) if stage == "lr_pilot"
                  else select_r(candidates, candidates[0].learning_rate))
        selected = winner.to_dict()
    return {
        "stage": stage,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": ident.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "selected": selected,
    }


def author_override_for(selected: Candidate, locked: Candidate) -> dict:
    return {
        "kind": LR_PILOT_AUTHOR_OVERRIDE_KIND,
        "author": "test author",
        "created_at": "2026-09-04",
        "selected_label": selected.label,
        "selected_learning_rate": selected.learning_rate,
        "superseded_locked_rule": LOCKED_LR_SELECTION_RULE,
        "superseded_locked_rule_winner": locked.to_dict(),
        "reason": (
            "Author selected a completed LR-pilot candidate after validation-curve "
            "review instead of the single-point locked-rule minimum."
        ),
        "evidence": {
            "source": "unit-test fixture",
            "review_basis": "validation curve stability",
        },
    }


def validate(artifact, stage="lr_pilot", ident=None):
    return validate_selection_artifact(
        artifact, expected_stage=stage, identity=ident or identity(), what="artifact.json"
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_a_genuine_lr_artifact_validates_and_reselects():
    winner = validate(artifact_for("lr_pilot", lr_candidates()))
    assert winner.learning_rate == 3e-4, "the lowest worst-case score must win"


def test_a_genuine_r_artifact_validates_and_reselects():
    winner = validate(artifact_for("r_phase1", r_candidates()), stage="r_phase1")
    assert winner.r == 1.0


def test_the_identity_block_covers_every_campaign_field():
    recorded = identity().to_dict()
    assert set(recorded) == set(IDENTITY_FIELDS)
    assert recorded["repository_head"] == HEAD
    assert recorded["corpus_manifest_digest"] == DIGEST
    assert recorded["precision"] == "fp32"
    assert recorded["inventory_sha256"] == INVENTORY.sha256


# ---------------------------------------------------------------------------
# Adversarial: identity
# ---------------------------------------------------------------------------
def test_a_wrong_stage_is_refused():
    with pytest.raises(ArtifactViolation, match="required"):
        validate(artifact_for("r_phase1", r_candidates()), stage="lr_pilot")


def test_a_wrong_protocol_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["protocol_version"] = "stage1-protocol-v0"
    with pytest.raises(ArtifactViolation, match="protocol"):
        validate(artifact)


@pytest.mark.parametrize("field,value", [
    ("repository_head", "b" * 40),
    ("corpus_manifest_digest", "e" * 64),
    ("encoder_checkpoint", "vinai/phobert-large"),
    ("encoder_revision", "0" * 40),
    ("precision", "fp16"),
    ("inventory_source_name", "some-other-inventory"),
    ("inventory_source_revision", "z" * 12),
    ("inventory_sha256", "f" * 64),
    ("protocol_version", "stage1-protocol-v0"),
])
def test_an_artifact_from_a_different_campaign_is_refused(field, value):
    """Wrong HEAD, corpus, backbone, revision, precision, inventory, protocol."""
    artifact = artifact_for("lr_pilot", lr_candidates(), ident=identity(**{field: value}))
    with pytest.raises(ArtifactViolation, match="different campaign") as caught:
        validate(artifact)
    assert field in str(caught.value)


def test_an_artifact_with_no_identity_block_is_refused():
    """A stale artifact from before this repair cannot drive a stage."""
    artifact = artifact_for("lr_pilot", lr_candidates())
    del artifact["identity"]
    with pytest.raises(ArtifactViolation, match="no identity block"):
        validate(artifact)


def test_a_truncated_identity_block_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    del artifact["identity"]["inventory_sha256"]
    with pytest.raises(ArtifactViolation, match="missing"):
        validate(artifact)


def test_expected_values_never_come_from_the_artifact_itself():
    """The whole point: a self-consistent forgery must still be refused.

    A foreign campaign that rewrites *both* its identity block and its evidence
    is internally consistent. It is caught only because the expected values come
    from the current run's verified inputs, never from the document.
    """
    foreign = identity(repository_head="b" * 40, corpus_manifest_digest="e" * 64)
    artifact = artifact_for("lr_pilot", lr_candidates(), ident=foreign)
    # Internally consistent...
    assert artifact["identity"]["repository_head"] == "b" * 40
    # ...and still refused against the CURRENT campaign.
    with pytest.raises(ArtifactViolation, match="different campaign"):
        validate(artifact, ident=identity())


# ---------------------------------------------------------------------------
# Adversarial: the locked grid
# ---------------------------------------------------------------------------
def test_a_missing_candidate_is_refused():
    candidates = lr_candidates()[:-1]
    with pytest.raises(ArtifactViolation, match="grid"):
        validate(artifact_for("lr_pilot", lr_candidates(), selected=None) | {
            "candidates": [c.to_dict() for c in candidates]
        })


def test_a_duplicate_candidate_is_refused():
    candidates = lr_candidates()
    duplicated = candidates + [candidates[0]]
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["candidates"] = [c.to_dict() for c in duplicated]
    with pytest.raises(ArtifactViolation, match="grid"):
        validate(artifact)


def test_an_extra_candidate_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    extra = Candidate(label="lr=5e-4", learning_rate=5e-4, r=LR_PILOT_R, selected=point(0.01))
    artifact["candidates"].append(extra.to_dict())
    with pytest.raises(ArtifactViolation, match="grid"):
        validate(artifact)


def test_an_off_grid_r_in_the_lr_pilot_is_refused():
    candidates = lr_candidates()
    artifact = artifact_for("lr_pilot", candidates)
    artifact["candidates"][0]["r"] = 4.0
    with pytest.raises(ArtifactViolation, match="r="):
        validate(artifact)


def test_an_r_artifact_with_a_split_learning_rate_is_refused():
    artifact = artifact_for("r_phase1", r_candidates())
    artifact["candidates"][2]["learning_rate"] = 1e-3
    with pytest.raises(ArtifactViolation, match="frozen LR"):
        validate(artifact, stage="r_phase1")


def test_an_artifact_with_no_candidates_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["candidates"] = []
    with pytest.raises(ArtifactViolation, match="no candidates"):
        validate(artifact)


# ---------------------------------------------------------------------------
# Adversarial: the selection itself
# ---------------------------------------------------------------------------
def test_an_edited_selected_scalar_is_refused():
    """The classic attack: keep the evidence, rewrite the winner."""
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["selected"]["learning_rate"] = 1e-3
    with pytest.raises(ArtifactViolation, match="rerunning the locked selection"):
        validate(artifact)


def test_edited_candidate_evidence_that_changes_the_winner_is_refused():
    """Rewrite the evidence, keep the winner: the recomputation disagrees."""
    artifact = artifact_for("lr_pilot", lr_candidates())
    # Make the 1e-3 candidate the best while `selected` still names 3e-4.
    for candidate in artifact["candidates"]:
        if candidate["learning_rate"] == 1e-3:
            candidate["selected"] = point(0.01).to_dict()
    with pytest.raises(ArtifactViolation, match="rerunning the locked selection"):
        validate(artifact)


def test_an_edited_selected_score_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["selected"]["d_clean"] = 0.0
    with pytest.raises(ArtifactViolation, match="rerunning the locked selection"):
        validate(artifact)


def test_an_explicit_lr_pilot_author_override_validates():
    candidates = lr_candidates()
    locked = candidates[1]
    selected = candidates[0]
    artifact = artifact_for("lr_pilot", candidates, selected=selected.to_dict())
    artifact["selection_override"] = author_override_for(selected, locked)

    winner = validate(artifact)

    assert winner.learning_rate == 1e-4


def test_an_lr_pilot_override_must_preserve_the_locked_rule_winner():
    candidates = lr_candidates()
    selected = candidates[0]
    artifact = artifact_for("lr_pilot", candidates, selected=selected.to_dict())
    artifact["selection_override"] = author_override_for(selected, selected)

    with pytest.raises(ArtifactViolation, match="preserve the locked-rule winner"):
        validate(artifact)


def test_an_lr_pilot_override_must_match_the_recorded_selected_candidate():
    candidates = lr_candidates()
    locked = candidates[1]
    selected = candidates[0]
    artifact = artifact_for("lr_pilot", candidates, selected=selected.to_dict())
    artifact["selection_override"] = author_override_for(selected, locked)
    artifact["selected"] = candidates[2].to_dict()

    with pytest.raises(ArtifactViolation, match="explicit selection_override selects"):
        validate(artifact)


def test_a_corrupt_candidate_point_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["candidates"][0]["selected"]["score"] = 99.0
    with pytest.raises(ArtifactViolation, match="unusable candidate"):
        validate(artifact)


def test_a_candidate_with_unknown_fields_is_refused():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["candidates"][0]["note"] = "hand edited"
    with pytest.raises(ArtifactViolation, match="unusable candidate"):
        validate(artifact)


def test_consistent_edits_to_both_are_still_refused_when_off_grid():
    """Editing evidence *and* winner together does not help if the grid breaks."""
    candidates = lr_candidates()
    artifact = artifact_for("lr_pilot", candidates)
    artifact["candidates"][0]["learning_rate"] = 5e-4
    artifact["selected"] = point(0.2).to_dict()
    with pytest.raises(ArtifactViolation):
        validate(artifact)


def test_final_main_has_no_selection_to_revalidate():
    artifact = artifact_for("lr_pilot", lr_candidates())
    artifact["stage"] = "final_main"
    with pytest.raises(ArtifactViolation, match="no selection"):
        validate(artifact, stage="final_main")


# ---------------------------------------------------------------------------
# Producer <-> consumer: the two sides must agree on the same identity object
# ---------------------------------------------------------------------------
def test_the_identity_is_built_from_the_real_inventory_type():
    """`execute_stage` passes a real `InventoryIdentity`, not a stand-in.

    `CampaignIdentity.from_inputs` reads the inventory with `getattr`, so this
    pins the three attribute names it depends on against the actual dataclass.
    """
    from unmark.stage1.preflight import InventoryIdentity

    real = InventoryIdentity(
        inventory_schema_version=1, source_name="vn-syllables-v1", source_author="a",
        source_revision="r" * 12, sha256="c" * 64, size_bytes=123, license_status="L",
    )
    built = CampaignIdentity.from_inputs(
        repository_head=HEAD, corpus_manifest_digest=DIGEST,
        encoder_revision=ENCODER_REVISION, inventory=real,
    )
    assert built.inventory_source_name == real.source_name
    assert built.inventory_source_revision == real.source_revision
    assert built.inventory_sha256 == real.sha256


def test_the_identity_survives_json_exactly():
    """Artifacts are JSON on disk; the consumer compares field by field."""
    import json

    recorded = identity().to_dict()
    assert json.loads(json.dumps(recorded, sort_keys=True)) == recorded


def test_a_producer_written_artifact_validates_unchanged(tmp_path):
    """Write the artifact the way `execute_stage` does, read it the way the
    runner does, through JSON on disk -- the real handoff, end to end."""
    import json

    from unmark.stage1.selection import select_learning_rate

    candidates = lr_candidates()
    ident = identity()
    produced = {
        "stage": "lr_pilot",
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": ident.to_dict(),
        "repository_head": ident.repository_head,
        "corpus_manifest_digest": ident.corpus_manifest_digest,
        "candidates": [c.to_dict() for c in candidates],
        "preparation": {"preparation_backend": "spawn"},
        "raw_text_persisted": False,
        "official_test_used": False,
        "downstream_score_used": False,
        "selected": select_learning_rate(candidates).to_dict(),
    }
    path = tmp_path / "lr_pilot.json"
    path.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    winner = validate_selection_artifact(
        json.loads(path.read_text(encoding="utf-8")),
        expected_stage="lr_pilot", identity=ident, what=str(path),
    )
    assert winner.learning_rate == 3e-4
    assert produced["official_test_used"] is False
