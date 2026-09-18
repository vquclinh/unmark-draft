#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from viunmark.assets import sha256_file
from viunmark.config import ViUnMarkContractError
from viunmark.evaluation import score_prediction_file, write_json


def verify_prediction_seal(predictions: Path, seal_path: Path) -> None:
    if not seal_path.is_file():
        raise ViUnMarkContractError(f"missing prediction seal manifest: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("sealed") is not True:
        raise ViUnMarkContractError("prediction manifest must contain sealed=true")
    expected_sha = seal.get("prediction_sha256")
    if not expected_sha:
        raise ViUnMarkContractError("prediction manifest must contain prediction_sha256")
    actual_sha = sha256_file(predictions)
    if actual_sha != expected_sha:
        raise ViUnMarkContractError(
            f"prediction SHA-256 mismatch: expected {expected_sha}, observed {actual_sha}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a sealed prediction artifact against user-supplied labels.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/scores.json"))
    args = parser.parse_args()
    verify_prediction_seal(args.predictions, args.prediction_manifest)
    result = score_prediction_file(args.predictions, args.labels)
    write_json(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
