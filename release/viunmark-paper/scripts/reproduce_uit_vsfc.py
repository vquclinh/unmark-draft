#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Official high-level UIT-VSFC reproduction driver.")
    parser.add_argument("--config", default="configs/uit_vsfc/paper.json")
    parser.add_argument("--mode", choices=["frozen", "retrain-readouts"], default="frozen")
    parser.add_argument(
        "--stage",
        choices=["verify", "prepare", "representations", "train-readouts", "predict", "score", "diagnostics", "all"],
        default="verify",
    )
    args = parser.parse_args()
    if args.stage in ("verify", "all"):
        return subprocess.call([sys.executable, "scripts/verify_assets.py", "--config", args.config])
    if args.mode == "retrain-readouts":
        return subprocess.call([sys.executable, "scripts/train_readouts.py", "--config", args.config])
    raise SystemExit(f"Stage {args.stage!r} needs external frozen assets and is intentionally not stubbed as a result.")


if __name__ == "__main__":
    raise SystemExit(main())
