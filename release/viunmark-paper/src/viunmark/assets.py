"""Public asset-contract verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from viunmark.config import ViUnMarkContractError
from viunmark.provenance import load_reproduction_manifest, load_uit_vsfc_reproduction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_public_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ViUnMarkContractError(f"missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_assets(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else load_reproduction_manifest()
    assets = payload.get("model_assets")
    if not isinstance(assets, list):
        raise ViUnMarkContractError("reproduction manifest is missing model_assets")
    return assets


def asset_root_from_config(config: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    root = Path(config["paths"]["asset_root"])
    if root.is_absolute() or base_dir is None:
        return root
    return base_dir / root


def verify_model_assets(
    assets: Iterable[dict[str, Any]],
    asset_root: Path,
    *,
    check_files: bool = True,
) -> dict[str, Any]:
    expected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []

    for item in assets:
        expected_path = asset_root / item["relative_path"]
        record = {**item, "expected_path": str(expected_path)}
        expected.append(record)
        if not check_files:
            continue
        if not expected_path.is_file():
            missing.append(record)
            continue
        actual_sha256 = sha256_file(expected_path)
        if actual_sha256 != item["sha256"]:
            mismatched.append({**record, "actual_sha256": actual_sha256})
            continue
        verified.append(record)

    status = "ok" if check_files and not missing and not mismatched else "missing_or_mismatched_assets"
    if not check_files:
        status = "manifest_only"
    return {
        "status": status,
        "asset_root": str(asset_root),
        "required_model_asset_count": len(expected),
        "expected_assets": expected,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def verify_from_config(config_path: Path, *, manifest_only: bool = False) -> dict[str, Any]:
    config = load_public_config(config_path)
    manifest = load_reproduction_manifest()
    reproduction = load_uit_vsfc_reproduction()
    assets = model_assets(manifest)
    if manifest.get("required_model_asset_count") != len(assets):
        raise ViUnMarkContractError("required_model_asset_count does not match model_assets")
    if reproduction.head_count != manifest.get("stage2_head_count"):
        raise ViUnMarkContractError("manifest head count does not match final-system provenance")
    base_dir = config_path.resolve().parents[2]
    report = verify_model_assets(
        assets,
        asset_root_from_config(config, base_dir=base_dir),
        check_files=not manifest_only,
    )
    report["dataset"] = config["dataset"]["name"]
    report["heads"] = reproduction.head_count
    report["heads_per_branch"] = manifest["scientific_invariants"]["heads_per_branch"]
    return report


def raise_for_asset_report(report: dict[str, Any]) -> None:
    if report["status"] in {"ok", "manifest_only"}:
        return
    missing = len(report["missing"])
    mismatched = len(report["mismatched"])
    raise ViUnMarkContractError(
        f"asset verification failed: {missing} missing, {mismatched} sha256 mismatched; "
        "run with --manifest-only to inspect the expected public asset layout"
    )
