#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Official high-level UIT-VSFC reproduction driver.")
    parser.add_argument("--config", default="configs/uit_vsfc/paper.json")
    parser.add_argument("--mode", choices=["frozen"], default="frozen")
    parser.add_argument(
        "--stage",
        choices=["verify", "diagnostics"],
        default="verify",
    )
    args = parser.parse_args()
    if args.stage == "verify":
        return subprocess.call([sys.executable, "scripts/verify_assets.py", "--config", args.config])
    return subprocess.call([sys.executable, "scripts/run_diagnostic.py", "--analysis", "scale-preflight"])


if __name__ == "__main__":
    raise SystemExit(main())
