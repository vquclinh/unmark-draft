#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Build representation banks from supplied frozen assets.")
    parser.add_argument("--config", default="configs/uit_vsfc/paper.json")
    parser.parse_args()
    raise SystemExit("Representation building requires external checkpoints/assets that are not shipped in this export.")


if __name__ == "__main__":
    main()
