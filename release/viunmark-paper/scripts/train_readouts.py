#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Public readout retraining entry point.")
    parser.add_argument("--config", default="configs/uit_vsfc/training.json")
    parser.parse_args()
    raise SystemExit(
        "Readout retraining is protocol-level recovered, not bit-exact. "
        "The public executable policy must be reviewed before training."
    )


if __name__ == "__main__":
    main()
