"""Mandatory external scientific inputs, verified BEFORE any model work.

The second real no-update smoke (Audit 030 §W) downloaded and loaded the real
PhoBERT encoder, then failed closed in condition preparation because the pinned
Vietnamese syllable inventory was not present in the fresh runtime. The failure
itself was correct -- D-B3A-001 designed it that way, and `active_eligibility_policy()`
is computed from whether the inventory loads and verifies, precisely so that a
missing cache re-arms the guard. What was wrong is *when* it happened: after the
expensive, network-bound part of the run.

This module answers "does every mandatory scientific input exist and match its
locked identity?" before `build_objective` is called.

It adds **no** verification of its own. `load_inventory` is the authoritative
loader and already hashes the cached bytes against the pinned SHA-256, checks
the declared shape, and raises distinct exceptions; this module calls it, turns
its exceptions into one actionable operator message, and returns the identity
that D-S1A-008 requires a scientific run to persist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unmark.corruption.eligibility import (
    CorruptionPurpose,
    EligibilityPolicy,
    active_eligibility_policy,
)
from unmark.linguistics import (
    DEFAULT_MANIFEST,
    InventoryChecksumMismatch,
    InventoryUnavailable,
    load_inventory,
)


class ScientificInputsUnavailable(RuntimeError):
    """A mandatory external scientific input is missing or does not match its pin."""


@dataclass(frozen=True)
class InventoryIdentity:
    """Exactly the fields D-S1A-008 requires a scientific Stage-1 run to persist.

    Deliberately **not** larger. `sha256` pins the raw bytes and `source_revision`
    pins the upstream object, so any other inventory is a different `sha256`, and
    `repository_head` is already part of `RunProvenance`.

    Three roles are kept distinct (Audit 030 §W.6):

    * **identity** -- these seven fields, compared by `RunProvenance.require_match`;
    * **locked shape** -- `size_bytes` (also an identity field) and the three entry
      counts D-B3A-001 locks (17 974 / 17 954 / 2 489). `load_inventory` verifies
      all four fail-closed on every load. They are *not* duplicated here, because
      D-S1A-008 does not list them and `sha256` already distinguishes any other
      byte sequence;
    * **report-only derived evidence** -- `parsed_membership_digest` and the
      observed counts. No decision locks a parsed digest, so it is recorded for
      run-to-run comparison and compared against no constant; promoting it would
      invent a scientific constant the decision log never locked.
    """

    inventory_schema_version: str
    source_name: str
    source_author: str
    source_revision: str
    sha256: str
    size_bytes: int | None
    license_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_schema_version": self.inventory_schema_version,
            "source_name": self.source_name,
            "source_author": self.source_author,
            "source_revision": self.source_revision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "license_status": self.license_status,
        }


@dataclass(frozen=True)
class ScientificInputs:
    """The verified inputs, plus descriptive evidence for the run artifact."""

    inventory: InventoryIdentity
    report: dict[str, Any]


_MISSING = """\
PREFLIGHT FAILED: the pinned Vietnamese syllable inventory is not available.

    {detail}

Scientific corruption cannot run without it. Proposal §4.3 defines a Vietnamese
candidate by membership of this inventory, so it decides every corruption
denominator and every channel projection; D-B3A-001 pins the exact revision and
SHA-256, and `active_eligibility_policy()` reports UNRESOLVED whenever the cache
is absent or does not match.

The raw list is NOT committed. The upstream gist carries no license statement,
so redistributing it here would be an unlicensed redistribution (manifest
`license_status: NO_EXPLICIT_LICENSE`); only its provenance is in git.

To provision it into the git-ignored repo-local cache:

    python scripts/fetch_vietnamese_syllable_inventory.py

That downloads exactly the pinned revision, verifies its SHA-256 before writing,
and never advances the pin. This check ran BEFORE the encoder was loaded, so
nothing has been downloaded or trained.
"""


def verify_scientific_inputs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repo_root: str | Path | None = None,
    *,
    purpose: CorruptionPurpose = CorruptionPurpose.SCIENTIFIC,
) -> ScientificInputs:
    """Verify every mandatory external scientific input. Fail closed.

    Raises `ScientificInputsUnavailable` unless the pinned inventory is present,
    matches its pinned SHA-256 and declared shape, and leaves the eligibility
    policy resolved. Returns the identity D-S1A-008 requires the run to persist.
    """
    if purpose is not CorruptionPurpose.SCIENTIFIC:
        raise ScientificInputsUnavailable(
            f"preflight is for the scientific path; purpose={purpose.name} would run "
            "under the provisional candidate-span fallback, whose counts are named "
            "`candidate_*` and whose artifacts are stamped provisional_eligibility=True. "
            "SELF_CHECK must never reach a scientific route."
        )

    try:
        inventory = load_inventory(manifest_path, repo_root)
    except InventoryChecksumMismatch as exc:
        # A distinct diagnosis. `try_load_inventory` degrades this to None
        # because `InventoryChecksumMismatch` subclasses `InventoryUnavailable`,
        # so without this the operator is told the file is missing when it is
        # actually present and wrong.
        raise ScientificInputsUnavailable(
            _MISSING.format(detail=f"The cached file is present but is NOT the pinned "
                                   f"revision.\n\n    {exc}")
        ) from exc
    except InventoryUnavailable as exc:
        raise ScientificInputsUnavailable(_MISSING.format(detail=str(exc).strip())) from exc

    provenance = inventory.provenance
    if provenance is None:  # pragma: no cover - load_inventory always attaches it
        raise ScientificInputsUnavailable("inventory loaded without provenance")

    policy = active_eligibility_policy()
    if policy is not EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY:
        raise ScientificInputsUnavailable(
            f"the inventory verified but the eligibility policy is {policy.name}; "
            "scientific corruption requires VIETNAMESE_SYLLABLE_INVENTORY"
        )

    identity = InventoryIdentity(
        inventory_schema_version=provenance.schema_version,
        source_name=provenance.source_name,
        source_author=provenance.source_author,
        source_revision=provenance.source_revision,
        sha256=provenance.sha256,
        size_bytes=provenance.size_bytes,
        license_status=provenance.license_status,
    )
    return ScientificInputs(
        inventory=identity,
        report={
            "eligibility_policy": policy.name,
            "purpose": purpose.name,
            "inventory": identity.to_dict(),
            "inventory_shape": {
                "raw_entry_count": inventory.raw_entry_count,
                "unique_canonical_entry_count": inventory.unique_canonical_entry_count,
                "unique_stripped_form_count": inventory.unique_stripped_form_count,
                "collisions_after_stripping": inventory.collisions_after_stripping,
            },
            # Derived, not pinned: a digest of the parsed membership set.
            # Recorded so two runs can be compared; NOT compared against a
            # constant, because -- unlike the three entry counts, which
            # D-B3A-001 locks and `load_inventory` verifies -- no decision locks
            # a parsed digest.
            "parsed_membership_digest": hashlib.sha256(
                "\n".join(sorted(inventory.forms)).encode("utf-8")
            ).hexdigest(),
            "source_url": provenance.source_url,
            "raw_url": provenance.raw_url,
            "cache_relative_path": provenance.cache_relative_path,
        },
    )
