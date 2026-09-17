#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from viunmark.evaluation import score_prediction_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a sealed prediction artifact against user-supplied labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/scores.json"))
    args = parser.parse_args()
    result = score_prediction_file(args.predictions, args.labels)
    write_json(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
