#!/usr/bin/env python3
"""Materialise the pre-G1 protocol-train / protocol-dev split. ML-free.

Consumes **only** the derived pre-G1 train csv — the exclusion-applied pool
approved in Audit 022. Official validation and official test are not inputs to
this program and cannot be passed to it.

Nothing here restates a locked value: fractions, seed, seed tag, expected row
count, expected class counts and the input digest all come from
`unmark.evaluation.preg1_protocol`. There is no `--seed` and no `--fractions`
flag, because a command-line override of a precommitted scientific constant is
exactly the hole this protocol exists to close.

**Nothing trains.** No head, no optimizer, no model weights, no downstream score.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    DERIVED_TRAIN_CSV_SHA256,
    DERIVED_TRAIN_SIZE,
    SPLIT_SEED,
    SPLIT_SEED_TAG,
)
from unmark.evaluation.preg1_split import (  # noqa: E402
    SPLIT_SCHEMA_VERSION,
    expected_split_counts,
    expected_split_totals,
    load_derived_pool,
    materialize_split,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-train", required=True,
        help="the exclusion-applied derived train csv; its SHA-256 must match the locked value",
    )
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument(
        "--output-dir", required=True,
        help="must NOT already exist; membership artifacts are immutable",
    )
    parser.add_argument(
        "--repository-head", default=None,
        help="commit sha recorded as provenance in the deterministic manifest",
    )
    parser.add_argument(
        "--record-runtime", action="store_true",
        help="also write runtime-environment.json (NOT part of the deterministic artifact)",
    )
    args = parser.parse_args(argv)

    try:
        pool = load_derived_pool(
            args.derived_train,
            text_column=args.text_column,
            label_column=args.label_column,
            id_column=args.id_column,
        )
        runtime = None
        if args.record_runtime:
            runtime = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "note": "runtime evidence only; excluded from the deterministic membership manifest",
            }
        manifest = materialize_split(
            pool,
            args.output_dir,
            repository_head=args.repository_head,
            runtime_evidence=runtime,
        )
    except EvaluationContractViolation as error:
        print(f"SPLIT REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {args.output_dir}")
    print(f"  schema        : {SPLIT_SCHEMA_VERSION}")
    print(f"  input sha256  : {DERIVED_TRAIN_CSV_SHA256}")
    print(f"  input rows    : {DERIVED_TRAIN_SIZE}")
    print(f"  seed tag/seed : {SPLIT_SEED_TAG} / {SPLIT_SEED}")
    for name, total in sorted(manifest["result"]["totals"].items()):
        counts = manifest["result"]["label_counts"][name]
        print(f"  {name:15s} {total:6d}  {counts}")
    print(f"  assignment    : {manifest['result']['assignment_digest']}")
    print(f"  expected      : {expected_split_totals()} {expected_split_counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
