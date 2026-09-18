#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from viunmark.assets import verify_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the public ViUnMark frozen-asset contract.")
    parser.add_argument("--config", type=Path, default=Path("configs/uit_vsfc/paper.json"))
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print the expected asset layout without requiring files to be present.",
    )
    args = parser.parse_args()
    report = verify_from_config(args.config, manifest_only=args.manifest_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] not in {"ok", "manifest_only"}:
        print(
            f"asset verification failed: {len(report['missing'])} missing, "
            f"{len(report['mismatched'])} sha256 mismatched; "
            "run with --manifest-only to inspect expected paths",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
