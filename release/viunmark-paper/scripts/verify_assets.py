#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from viunmark.provenance import load_reproduction_manifest, load_uit_vsfc_reproduction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/uit_vsfc/paper.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = load_reproduction_manifest()
    reproduction = load_uit_vsfc_reproduction()
    asset_root = Path(config["paths"]["asset_root"])
    required = []
    for item in manifest["external_assets"]:
        required.append({**item, "expected_path": str(asset_root / item["relative_path"])})
    print(json.dumps({"status": "ok", "heads": reproduction.head_count, "required_assets": required}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
