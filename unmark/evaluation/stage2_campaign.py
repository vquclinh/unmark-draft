"""Stage-2 dual-finalist campaign orchestration: one manifest, ten runs.

Audit 051. `stage2_head_campaign` supplies the primitives — cache, head runner,
checkpoint selection, artifact, run store. This module is the thin layer that
binds them into **one campaign identity** and refuses anything that is not
exactly the frozen `2 arms x 5 seeds`.

**Why a registry exists at all.** Each head run already binds its own commit,
protocol and cache keys, but nothing previously asserted that the *ten runs
together* share one commit, one protocol and one pair of caches per arm. A
campaign assembled from runs produced at two different commits would look
complete and be meaningless, and the per-run checks cannot see it. The manifest
is what makes that visible; the registry is what makes re-entering a partial
campaign safe.

**Nothing here duplicates the implementation.** Extraction, training, selection,
artifact construction/validation and measurement are all *called*, never
reimplemented — a test asserts by AST that this module calls
`train_stage2_head`, `build_stage2_head_artifact` and `Stage2RepresentationCache`
rather than defining its own.

**No selection.** Audit 049 option (c) carries both arms. The registry records
which runs are complete; it holds no winner, no ranking and no "best arm", and
`campaign_status` is deliberately a completion report rather than a comparison.

**It never reads UIT-VSFC.** The entry point operates over already-materialised
representation caches and caller-supplied labels; it opens no dataset file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.stage2_head_campaign import (
    STAGE2_CAMPAIGN_ARM_COUNT,
    STAGE2_CAMPAIGN_RUN_COUNT,
    STAGE2_CAMPAIGN_SEEDS,
    STAGE2_CLEAN_CONDITION,
    STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
    STAGE2_PROTOCOL_VERSION,
    STAGE2_SELECTION_ROLE,
    STAGE2_TRAINING_ROLE,
    Stage2CampaignRun,
    Stage2HeadRunStore,
    Stage2RepresentationCache,
    Stage2RepresentationKey,
    Stage2UnmarkArm,
    build_stage2_head_artifact,
    require_frozen_protocol_spec,
    require_paired_campaign_plan,
    require_stage2_unmark_arm,
    stage2_campaign_plan,
    train_stage2_head,
    validate_stage2_head_artifact,
)

STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION = "stage2-campaign-manifest-v1"

STAGE2_CAMPAIGN_CACHE_ROLES = (STAGE2_TRAINING_ROLE, STAGE2_SELECTION_ROLE)
"""The two clean caches each arm needs: protocol-train and protocol-dev."""

STAGE2_CAMPAIGN_WINNER = None
STAGE2_CAMPAIGN_RANKS_ARMS = False
"""Declared so a reviewer can see the absence was decided, not overlooked."""


def cache_slot(arm: str | Stage2UnmarkArm, role: Any) -> str:
    """Stable manifest key for one arm's cache in one role."""
    return f"{require_stage2_unmark_arm(arm).value}/{role.value}"


STAGE2_CAMPAIGN_CACHE_SLOTS: tuple[str, ...] = tuple(
    cache_slot(arm, role)
    for arm in Stage2UnmarkArm
    for role in STAGE2_CAMPAIGN_CACHE_ROLES
)


@dataclass(frozen=True)
class Stage2CampaignManifest:
    """The one identity every run in a campaign must share.

    Four clean caches — protocol-train and protocol-dev for each arm — plus the
    commit and protocol they were produced under, plus the exact ten
    `(arm, seed)` runs the campaign consists of.
    """

    repository_head: str
    protocol_version: str
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    cache_keys: Mapping[str, Stage2RepresentationKey]
    expected_runs: tuple[Stage2CampaignRun, ...]
    schema_version: str = STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "head_campaign_schema_version": STAGE2_HEAD_CAMPAIGN_SCHEMA_VERSION,
            "repository_head": self.repository_head,
            "protocol_version": self.protocol_version,
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "cache_keys": {slot: k.to_dict() for slot, k in sorted(self.cache_keys.items())},
            "expected_runs": [
                {"arm": r.arm, "seed": r.seed} for r in self.expected_runs
            ],
            "winner": STAGE2_CAMPAIGN_WINNER,
            "ranks_arms": STAGE2_CAMPAIGN_RANKS_ARMS,
        }

    @property
    def digest(self) -> str:
        """Identity digest. Re-entering a campaign with a different one refuses."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def run_key(self, arm: str, seed: int) -> str:
        return f"{require_stage2_unmark_arm(arm).value}/seed-{seed}"


def build_stage2_campaign_manifest(
    *,
    repository_head: str,
    cache_keys: Mapping[str, Stage2RepresentationKey],
) -> Stage2CampaignManifest:
    """Assemble and fully validate one campaign identity."""
    manifest = Stage2CampaignManifest(
        repository_head=repository_head,
        protocol_version=STAGE2_PROTOCOL_VERSION,
        arms=tuple(a.value for a in Stage2UnmarkArm),
        seeds=tuple(STAGE2_CAMPAIGN_SEEDS),
        cache_keys=dict(cache_keys),
        expected_runs=stage2_campaign_plan(),
    )
    validate_stage2_campaign_manifest(manifest)
    return manifest


def validate_stage2_campaign_manifest(manifest: Stage2CampaignManifest) -> None:
    """Fail closed unless this is exactly the frozen campaign.

    Every rule here is one a per-run check cannot see: a run knows its own
    commit, but only the manifest knows whether the *other nine* agree.
    """
    require_frozen_protocol_spec()

    if manifest.schema_version != STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION:
        raise EvaluationContractViolation(
            f"campaign manifest schema {manifest.schema_version!r} is not "
            f"{STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.protocol_version != STAGE2_PROTOCOL_VERSION:
        raise EvaluationContractViolation(
            f"campaign protocol {manifest.protocol_version!r} is not the frozen "
            f"{STAGE2_PROTOCOL_VERSION!r}"
        )
    if not isinstance(manifest.repository_head, str) or not manifest.repository_head.strip():
        raise EvaluationContractViolation("campaign repository_head must be a non-empty string")

    expected_arms = tuple(a.value for a in Stage2UnmarkArm)
    if tuple(manifest.arms) != expected_arms:
        raise EvaluationContractViolation(
            f"campaign arms {list(manifest.arms)} are not exactly {list(expected_arms)}. "
            f"Audit 049 freezes {STAGE2_CAMPAIGN_ARM_COUNT} arms: no third arm, none missing."
        )
    if tuple(manifest.seeds) != tuple(STAGE2_CAMPAIGN_SEEDS):
        raise EvaluationContractViolation(
            f"campaign seeds {list(manifest.seeds)} are not exactly the frozen "
            f"{list(STAGE2_CAMPAIGN_SEEDS)}: no seed added, none dropped, order fixed"
        )

    # --- caches: four slots, each bound to its own arm, role, commit, protocol
    missing = [s for s in STAGE2_CAMPAIGN_CACHE_SLOTS if s not in manifest.cache_keys]
    if missing:
        raise EvaluationContractViolation(
            f"campaign manifest is missing cache slot(s) {missing}; each arm needs both a "
            "protocol-train and a protocol-dev cache"
        )
    unknown = sorted(set(manifest.cache_keys) - set(STAGE2_CAMPAIGN_CACHE_SLOTS))
    if unknown:
        raise EvaluationContractViolation(
            f"campaign manifest carries unknown cache slot(s) {unknown}; the slot set is closed"
        )
    for arm in Stage2UnmarkArm:
        for role in STAGE2_CAMPAIGN_CACHE_ROLES:
            slot = cache_slot(arm, role)
            key = manifest.cache_keys[slot]
            if not isinstance(key, Stage2RepresentationKey):
                raise EvaluationContractViolation(f"{slot} is not a Stage2RepresentationKey")
            if key.arm != arm.value:
                raise EvaluationContractViolation(
                    f"cache slot {slot} holds arm {key.arm!r}: one arm's cache may never "
                    "stand in for the other's"
                )
            if key.role != role.value:
                raise EvaluationContractViolation(
                    f"cache slot {slot} holds role {key.role!r}, expected {role.value!r}"
                )
            if key.condition != STAGE2_CLEAN_CONDITION:
                raise EvaluationContractViolation(
                    f"cache slot {slot} is condition {key.condition!r}; head training and "
                    f"checkpoint selection use clean {STAGE2_CLEAN_CONDITION} only"
                )
            if key.repository_head != manifest.repository_head:
                raise EvaluationContractViolation(
                    f"cache slot {slot} was produced at {key.repository_head!r} but the "
                    f"campaign is {manifest.repository_head!r}: a campaign may not mix commits"
                )
            if key.protocol_version != manifest.protocol_version:
                raise EvaluationContractViolation(
                    f"cache slot {slot} is protocol {key.protocol_version!r} but the campaign "
                    f"is {manifest.protocol_version!r}: a campaign may not mix protocols"
                )

    # A and B must not be the same cache wearing two labels.
    digests = {
        slot: manifest.cache_keys[slot].finalist_checkpoint_sha256
        for slot in STAGE2_CAMPAIGN_CACHE_SLOTS
    }
    a_train = digests[cache_slot(Stage2UnmarkArm.UNMARK_A, STAGE2_TRAINING_ROLE)]
    b_train = digests[cache_slot(Stage2UnmarkArm.UNMARK_B, STAGE2_TRAINING_ROLE)]
    if a_train == b_train:
        raise EvaluationContractViolation(
            "both arms' caches bind the same finalist checkpoint sha256; the arms would be "
            "identical and any comparison meaningless"
        )

    require_paired_campaign_plan(manifest.expected_runs)
    if len(manifest.expected_runs) != STAGE2_CAMPAIGN_RUN_COUNT:
        raise EvaluationContractViolation(
            f"campaign has {len(manifest.expected_runs)} runs, not {STAGE2_CAMPAIGN_RUN_COUNT}"
        )


def stage2_campaign_schedule(
    manifest: Stage2CampaignManifest,
) -> tuple[Stage2CampaignRun, ...]:
    """The exact ten runs, validated. Matched seeds stay paired."""
    validate_stage2_campaign_manifest(manifest)
    return tuple(manifest.expected_runs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class Stage2CampaignRegistry:
    """Persisted campaign identity plus per-run completion state.

    Re-entering a campaign is permitted only under a **byte-identical manifest**:
    the stored manifest digest must match, so a partial campaign cannot adopt run
    state produced under a different commit, protocol or cache set.
    """

    MANIFEST_NAME = "stage2-campaign-manifest.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST_NAME

    def run_directory(self, arm: str, seed: int) -> Path:
        return self.directory / require_stage2_unmark_arm(arm).value / f"seed-{seed}"

    def store_for(self, arm: str, seed: int) -> Stage2HeadRunStore:
        return Stage2HeadRunStore(self.run_directory(arm, seed))

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    def open(self, manifest: Stage2CampaignManifest) -> Stage2CampaignManifest:
        """Create or re-open. A drifted manifest is refused, never adopted."""
        from unmark.stage1.checkpoint import atomic_write_bytes

        validate_stage2_campaign_manifest(manifest)
        if self.exists():
            recorded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if recorded.get("manifest_digest") != manifest.digest:
                raise EvaluationContractViolation(
                    f"{self.directory} holds a campaign whose manifest digest is "
                    f"{recorded.get('manifest_digest')!r}, not {manifest.digest!r}. A partial "
                    "campaign may not adopt an incompatible identity: commit, protocol, "
                    "caches, arms and seeds must all match. Start a new campaign directory."
                )
            return manifest
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"manifest_digest": manifest.digest, "manifest": manifest.to_dict()}
        atomic_write_bytes(
            self.manifest_path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return manifest

    def completed_runs(self, manifest: Stage2CampaignManifest) -> tuple[Stage2CampaignRun, ...]:
        done = []
        for run in manifest.expected_runs:
            store = self.store_for(run.arm, run.seed)
            if store.is_complete():
                store.read_artifact(expected_arm=run.arm, expected_seed=run.seed)
                done.append(run)
        return tuple(done)

    def pending_runs(self, manifest: Stage2CampaignManifest) -> tuple[Stage2CampaignRun, ...]:
        done = set((r.arm, r.seed) for r in self.completed_runs(manifest))
        return tuple(r for r in manifest.expected_runs if (r.arm, r.seed) not in done)

    def campaign_status(self, manifest: Stage2CampaignManifest) -> dict[str, Any]:
        """Completion report. **Not** a comparison: no scores, no ranking."""
        completed = self.completed_runs(manifest)
        pending = self.pending_runs(manifest)
        per_arm = {
            arm.value: sorted(r.seed for r in completed if r.arm == arm.value)
            for arm in Stage2UnmarkArm
        }
        return {
            "schema_version": STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            "manifest_digest": manifest.digest,
            "expected": len(manifest.expected_runs),
            "completed": len(completed),
            "pending": [{"arm": r.arm, "seed": r.seed} for r in pending],
            "completed_seeds_by_arm": per_arm,
            "complete": len(completed) == len(manifest.expected_runs),
            "winner": STAGE2_CAMPAIGN_WINNER,
            "ranks_arms": STAGE2_CAMPAIGN_RANKS_ARMS,
        }

    def require_campaign_complete(self, manifest: Stage2CampaignManifest) -> None:
        """Both arms, all five seeds. A partial campaign may not be reported."""
        status = self.campaign_status(manifest)
        if not status["complete"]:
            raise EvaluationContractViolation(
                f"campaign is {status['completed']}/{status['expected']} complete; missing "
                f"{status['pending']}. Both arms must be reported: Audit 049 forbids dropping "
                "one, and a partial campaign report would be exactly that."
            )


# ---------------------------------------------------------------------------
# Entry point -- operates over already-materialised caches only
# ---------------------------------------------------------------------------
def run_stage2_campaign(
    manifest: Stage2CampaignManifest,
    registry: Stage2CampaignRegistry,
    *,
    cache_directories: Mapping[str, str | Path],
    labels: Mapping[str, Sequence[int]],
    repository_head: str,
    only: Sequence[Stage2CampaignRun] | None = None,
) -> dict[str, Any]:
    """Execute the campaign's pending head runs over cached representations.

    **Reads no dataset.** `cache_directories` maps the four manifest slots to
    directories already produced by extraction; `labels` maps the same slots to
    the caller's label vectors, which the caches' `label_digest` then verifies.
    Every step delegates: `Stage2RepresentationCache.load`, `train_stage2_head`,
    `build_stage2_head_artifact`, `Stage2HeadRunStore.commit`.

    A completed run is skipped, never re-executed and never overwritten.
    """
    registry.open(manifest)
    if repository_head != manifest.repository_head:
        raise EvaluationContractViolation(
            f"caller head {repository_head!r} does not match the campaign's "
            f"{manifest.repository_head!r}"
        )

    missing = [s for s in STAGE2_CAMPAIGN_CACHE_SLOTS if s not in cache_directories]
    if missing:
        raise EvaluationContractViolation(f"no cache directory supplied for {missing}")
    missing_labels = [s for s in STAGE2_CAMPAIGN_CACHE_SLOTS if s not in labels]
    if missing_labels:
        raise EvaluationContractViolation(f"no labels supplied for {missing_labels}")

    scheduled = list(only) if only is not None else list(registry.pending_runs(manifest))
    for run in scheduled:
        if run not in manifest.expected_runs:
            raise EvaluationContractViolation(
                f"{run} is not one of the campaign's {STAGE2_CAMPAIGN_RUN_COUNT} expected runs"
            )

    executed: list[dict[str, Any]] = []
    for run in scheduled:
        store = registry.store_for(run.arm, run.seed)
        store.require_writable(expected_arm=run.arm, expected_seed=run.seed)

        train_slot = cache_slot(run.arm, STAGE2_TRAINING_ROLE)
        dev_slot = cache_slot(run.arm, STAGE2_SELECTION_ROLE)
        train = Stage2RepresentationCache(cache_directories[train_slot]).load(
            manifest.cache_keys[train_slot]
        )
        dev = Stage2RepresentationCache(cache_directories[dev_slot]).load(
            manifest.cache_keys[dev_slot]
        )

        store.mark_in_progress(arm=run.arm, seed=run.seed)
        head_run = train_stage2_head(
            train, labels[train_slot], dev, labels[dev_slot], seed=run.seed
        )
        artifact = build_stage2_head_artifact(head_run, repository_head=repository_head)
        validate_stage2_head_artifact(
            artifact, expected_arm=run.arm, expected_seed=run.seed
        )
        store.commit(artifact, head_run.selected_head_state)
        executed.append({"arm": run.arm, "seed": run.seed,
                         "selected_epoch": head_run.selected.epoch})

    return {
        "schema_version": STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "manifest_digest": manifest.digest,
        "executed": executed,
        "status": registry.campaign_status(manifest),
    }


__all__ = [
    "STAGE2_CAMPAIGN_CACHE_ROLES",
    "STAGE2_CAMPAIGN_CACHE_SLOTS",
    "STAGE2_CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "STAGE2_CAMPAIGN_RANKS_ARMS",
    "STAGE2_CAMPAIGN_WINNER",
    "Stage2CampaignManifest",
    "Stage2CampaignRegistry",
    "build_stage2_campaign_manifest",
    "cache_slot",
    "run_stage2_campaign",
    "stage2_campaign_schedule",
    "validate_stage2_campaign_manifest",
]
