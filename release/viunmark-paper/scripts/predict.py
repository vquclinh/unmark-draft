#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate predictions from supplied frozen logits/assets. Labels are not read.")
    parser.add_argument("--config", default="configs/uit_vsfc/paper.json")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.parse_args()
    raise SystemExit("Frozen prediction generation requires released external logits/checkpoints; none are bundled.")


if __name__ == "__main__":
    main()
