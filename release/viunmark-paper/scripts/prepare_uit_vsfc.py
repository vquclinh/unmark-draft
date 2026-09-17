#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate user-supplied UIT-VSFC paths without reading labels.")
    parser.add_argument("--config", type=Path, default=Path("configs/uit_vsfc/paper.json"))
    parser.parse_args()
    print("Dataset preparation is path/schema validation only. Raw UIT-VSFC data is not shipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
