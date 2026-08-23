#!/usr/bin/env python3
"""Fetch and verify the pinned Vietnamese syllable inventory.

The ONLY network operation in this project outside the Colab G-1 path. It
downloads exactly the revision pinned in
`configs/linguistics/vietnamese_syllables.yaml`, checks its SHA-256, and writes
it into a repo-local, git-ignored cache.

It never advances the pin. If upstream changes, the checksum fails and this
script refuses -- changing the inventory revision is a scientific spec change
that must be made deliberately in the manifest and recorded in
`docs/spec/decisions.md`.

The raw list is not committed because the upstream gist carries no license
statement; see the manifest's `license_status`.

    .venv/bin/python scripts/fetch_vietnamese_syllable_inventory.py
    .venv/bin/python scripts/fetch_vietnamese_syllable_inventory.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.linguistics.inventory import (  # noqa: E402
    DEFAULT_MANIFEST,
    InventoryProvenance,
    build_inventory,
    load_manifest,
)

TIMEOUT_SECONDS = 60
MAX_BYTES = 8 * 1024 * 1024  # the pinned file is ~116 KB; refuse anything absurd


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "unmark-b3a-fetch"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise SystemExit(f"refusing: {url} returned more than {MAX_BYTES} bytes")
    return data


def publish(path: Path, raw: bytes) -> None:
    """Atomic publication: temp -> flush -> fsync -> replace -> dir fsync.

    The same discipline Stage 6 uses. A crash or a lost Colab runtime part-way
    through the write must leave either the previous verified file or nothing --
    never a truncated one that would later be reported as a checksum mismatch.
    Only reached AFTER the digest has been checked, so what is published is
    always the pinned bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    except OSError:  # pragma: no cover - not every filesystem allows it
        pass
    finally:
        os.close(directory_fd)


def report(provenance: InventoryProvenance, path: Path, digest: str, raw: bytes) -> None:
    inventory = build_inventory(raw.decode("utf-8").splitlines(), provenance)
    print("Vietnamese syllable inventory")
    print(f"  source        : {provenance.source_name} by {provenance.source_author}")
    print(f"  gist          : {provenance.source_url}")
    print(f"  revision      : {provenance.source_revision}")
    print(f"  raw url       : {provenance.raw_url}")
    print(f"  sha256        : {digest}")
    print(f"  bytes         : {len(raw)}")
    print(f"  license       : {provenance.license_status}  (raw file NOT committed)")
    print(f"  retrieved_at  : {provenance.retrieved_at}")
    print()
    print(f"  raw entries              : {inventory.raw_entry_count}")
    print(f"  unique canonical entries : {inventory.unique_canonical_entry_count}")
    print(f"  unique stripped forms    : {inventory.unique_stripped_form_count}")
    print(f"  collisions after strip   : {inventory.collisions_after_stripping}")
    print()
    print(f"  cached at     : {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the pinned syllable inventory.")
    parser.add_argument("--manifest", default=str(REPO_ROOT / DEFAULT_MANIFEST))
    parser.add_argument("--verify-only", action="store_true", help="check the cache without downloading")
    parser.add_argument("--force", action="store_true", help="re-download even if the cache is already valid")
    args = parser.parse_args(argv)

    provenance = load_manifest(args.manifest)
    path = REPO_ROOT / provenance.cache_relative_path

    if path.is_file() and not args.force:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest == provenance.sha256:
            print("Cache already present and verified.\n")
            report(provenance, path, digest, raw)
            return 0
        if args.verify_only:
            print(f"CHECKSUM MISMATCH at {path}", file=sys.stderr)
            print(f"  expected {provenance.sha256}", file=sys.stderr)
            print(f"  found    {digest}", file=sys.stderr)
            return 1
        print(f"Cached file does not match the pin ({digest}); re-downloading.", file=sys.stderr)

    if args.verify_only:
        print(f"No cached inventory at {path}", file=sys.stderr)
        return 1

    print(f"Downloading pinned revision {provenance.source_revision} ...")
    raw = download(provenance.raw_url)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != provenance.sha256:
        print("REFUSING TO WRITE: checksum mismatch.", file=sys.stderr)
        print(f"  expected {provenance.sha256}", file=sys.stderr)
        print(f"  found    {digest}", file=sys.stderr)
        print(
            "\nUpstream may have changed. This script never advances the pin: update\n"
            "configs/linguistics/vietnamese_syllables.yaml deliberately and record the\n"
            "change in docs/spec/decisions.md, because it alters every corruption\n"
            "denominator.",
            file=sys.stderr,
        )
        return 1

    publish(path, raw)
    print("Verified and cached.\n")
    report(provenance, path, digest, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
