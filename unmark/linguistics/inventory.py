"""The pinned Vietnamese syllable inventory.

Proposal v1.3 §4.3 decides Vietnamese candidacy by membership in "the Vietnamese
syllable inventory after stripping". This module supplies that inventory, from a
pinned external resource, verified by checksum.

Why an external resource, and why not committed
-----------------------------------------------
The inventory is ~18,000 syllables. Enumerating it by hand would be inventing
linguistic data; the project instead pins a published structural enumeration
(every onset × every rime). The upstream gist carries **no license statement**,
so the raw file is deliberately *not* redistributed in this repository. What is
committed is provenance: source, revision, SHA-256, counts. The file itself is
fetched into a repo-local, git-ignored cache by
`scripts/fetch_vietnamese_syllable_inventory.py`, and everything scientific
fails loudly when it is absent.

Membership form
---------------
Membership is tested on the **stripped** form, never on the observed spelling:

    canon(entry) -> strip_to_base(...) -> casefold() -> set

That is what makes eligibility a pure function of the stripped form, so a clean
syllable and its corrupted counterpart receive identical labels and the base grid
stays invariant (§4.3). Many diacritized syllables collapse to one stripped form
-- `má`, `mà`, `mã`, `mạ`, `mả`, `ma` all become `ma` -- which is expected, not an
error.

No network access happens here. This module only reads a file that the fetch
script has already verified.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unmark.orthography import canon, strip_to_base

ELIGIBILITY_SCHEMA_VERSION = "vn-syllables-v1"
"""Version of the eligibility *policy*, distinct from the corruption schema.

Bumping it -- by changing the inventory revision, the membership form or the case
rule -- changes which spans are eligible and therefore every corruption
denominator. It is recorded in every scientific artifact so such a change can
never become invisible.
"""

DEFAULT_MANIFEST = "configs/linguistics/vietnamese_syllables.yaml"


class InventoryUnavailable(RuntimeError):
    """Raised when the pinned inventory is missing, unreadable or unverified."""


class InventoryChecksumMismatch(InventoryUnavailable):
    """Raised when the cached file does not match the pinned SHA-256."""


def membership_form(text: str) -> str:
    """The form membership is tested on: canonical, stripped, casefolded.

    Deliberately independent of whether `text` carries diacritics, so
    `membership_form("má") == membership_form("ma") == "ma"`.
    """
    return strip_to_base(canon(text)).casefold()


@dataclass(frozen=True)
class InventoryProvenance:
    """Everything needed to identify exactly which inventory was used."""

    schema_version: str
    source_name: str
    source_author: str
    source_url: str
    source_revision: str
    raw_url: str
    sha256: str
    retrieved_at: str
    canonicalization_mode: str
    membership_form: str
    license_status: str
    expected_entry_count: int
    cache_relative_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligibility_schema_version": self.schema_version,
            "source_name": self.source_name,
            "source_author": self.source_author,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "raw_url": self.raw_url,
            "sha256": self.sha256,
            "retrieved_at": self.retrieved_at,
            "canonicalization_mode": self.canonicalization_mode,
            "membership_form": self.membership_form,
            "license_status": self.license_status,
            "expected_entry_count": self.expected_entry_count,
            "cache_relative_path": self.cache_relative_path,
        }


@dataclass(frozen=True)
class SyllableInventory:
    """A loaded, stripped-form Vietnamese syllable inventory."""

    forms: frozenset[str]
    provenance: InventoryProvenance | None
    raw_entry_count: int
    unique_canonical_entry_count: int
    stats: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, text: str) -> bool:
        return membership_form(text) in self.forms

    def contains_membership_form(self, form: str) -> bool:
        """Membership for a value already reduced by :func:`membership_form`."""
        return form in self.forms

    @property
    def unique_stripped_form_count(self) -> int:
        return len(self.forms)

    @property
    def collisions_after_stripping(self) -> int:
        """Canonical entries that collapsed onto an already-present stripped form.

        Large by construction: an inventory enumerating every tone and letter
        diacritic collapses heavily. Not an error.
        """
        return self.unique_canonical_entry_count - self.unique_stripped_form_count

    def summary(self) -> dict[str, Any]:
        return {
            "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
            "raw_entry_count": self.raw_entry_count,
            "unique_canonical_entry_count": self.unique_canonical_entry_count,
            "unique_stripped_form_count": self.unique_stripped_form_count,
            "collisions_after_stripping": self.collisions_after_stripping,
            **({"provenance": self.provenance.to_dict()} if self.provenance else {}),
        }


def build_inventory(
    entries: list[str],
    provenance: InventoryProvenance | None = None,
) -> SyllableInventory:
    """Build an inventory from raw source lines. Deterministic and order-free."""
    cleaned = [line.strip() for line in entries if line.strip()]
    canonical = {canon(entry) for entry in cleaned}
    forms = frozenset(membership_form(entry) for entry in canonical)
    return SyllableInventory(
        forms=forms,
        provenance=provenance,
        raw_entry_count=len(cleaned),
        unique_canonical_entry_count=len(canonical),
    )


def load_manifest(manifest_path: str | Path = DEFAULT_MANIFEST) -> InventoryProvenance:
    """Read the committed provenance manifest. No network, no file download."""
    import yaml

    path = Path(manifest_path)
    if not path.is_file():
        raise InventoryUnavailable(f"inventory manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return InventoryProvenance(
            schema_version=data["inventory_schema_version"],
            source_name=data["source_name"],
            source_author=data["source_author"],
            source_url=data["source_url"],
            source_revision=data["source_revision"],
            raw_url=data["raw_url"],
            sha256=data["sha256"],
            retrieved_at=data["retrieved_at"],
            canonicalization_mode=data["canonicalization_mode"],
            membership_form=data["membership_form"],
            license_status=data["license_status"],
            expected_entry_count=int(data["expected_entry_count"]),
            cache_relative_path=data["cache_relative_path"],
        )
    except KeyError as exc:  # noqa: PERF203 - one clear message beats a KeyError
        raise InventoryUnavailable(f"{path}: manifest is missing required field {exc}") from exc


_MISSING_MESSAGE = """\
The pinned Vietnamese syllable inventory is not available at:
    {path}

Eligibility, and therefore scientific corruption, cannot be resolved without it.

The raw list is NOT committed to this repository. The upstream gist has
no license statement, so redistributing it here would be an unlicensed
redistribution; only its provenance (source, revision, SHA-256, counts) is in
git.

To fetch and verify it into the repo-local, git-ignored cache:

    .venv/bin/python scripts/fetch_vietnamese_syllable_inventory.py

That script downloads exactly the pinned revision, checks its SHA-256 against
the manifest, and refuses on any mismatch. It never advances the pin.
"""


def load_inventory(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repo_root: str | Path | None = None,
    *,
    verify_checksum: bool = True,
) -> SyllableInventory:
    """Load the pinned inventory from the repo-local cache.

    Raises `InventoryUnavailable` when the cache is absent and
    `InventoryChecksumMismatch` when it does not match the pinned digest, so a
    silently substituted inventory cannot reach an experiment.
    """
    provenance = load_manifest(manifest_path)
    root = Path(repo_root) if repo_root is not None else Path(manifest_path).resolve().parents[2]
    path = root / provenance.cache_relative_path
    if not path.is_file():
        raise InventoryUnavailable(_MISSING_MESSAGE.format(path=path))

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_checksum and digest != provenance.sha256:
        raise InventoryChecksumMismatch(
            f"inventory checksum mismatch at {path}\n"
            f"  expected {provenance.sha256}\n"
            f"  found    {digest}\n"
            "The cached file is not the pinned revision. Delete it and re-run "
            "scripts/fetch_vietnamese_syllable_inventory.py. Changing the pinned "
            "revision is a scientific spec change and must be recorded in "
            "docs/spec/decisions.md."
        )

    inventory = build_inventory(raw.decode("utf-8").splitlines(), provenance)
    if provenance.expected_entry_count and inventory.raw_entry_count != provenance.expected_entry_count:
        raise InventoryUnavailable(
            f"inventory entry count {inventory.raw_entry_count} != pinned "
            f"{provenance.expected_entry_count}; the resource changed shape"
        )
    return inventory


# Loading parses ~18,000 entries and hashes a 116 KB file. Callers ask for the
# inventory once per corruption call, so the result is memoised. The cache is
# keyed by manifest path, and `clear_inventory_cache()` drops it -- tests that
# simulate a missing or corrupted resource must call that, otherwise a stale
# object would mask the very failure they are checking.
_CACHE: dict[tuple[str, str | None], SyllableInventory] = {}
_CACHE_LOCK = threading.Lock()


def clear_inventory_cache() -> None:
    """Forget any memoised inventory. Call after changing the cached file."""
    with _CACHE_LOCK:
        _CACHE.clear()


def load_inventory_cached(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repo_root: str | Path | None = None,
) -> SyllableInventory:
    """`load_inventory` with memoisation. Raises exactly as `load_inventory` does."""
    key = (str(manifest_path), str(repo_root) if repo_root is not None else None)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    inventory = load_inventory(manifest_path, repo_root)
    with _CACHE_LOCK:
        _CACHE[key] = inventory
    return inventory


def try_load_inventory(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    repo_root: str | Path | None = None,
) -> SyllableInventory | None:
    """Load the inventory, or return `None` when it is unavailable.

    For callers that must degrade gracefully -- B1A decomposition stays
    deterministic without it. Scientific paths use `load_inventory` and let the
    exception through.
    """
    try:
        return load_inventory_cached(manifest_path, repo_root)
    except InventoryUnavailable:
        return None
