#!/usr/bin/env python3
"""Pre-G1 frozen-encoder burden diagnostic — Colab runner.

Two subcommands, in the order the protocol requires:

    tune       sweep the precommitted LR grid on **VANILLA ONLY**, on
               protocol-dev, and freeze the winner
    measure    run the paired Vanilla-vs-Base-only measurement on official
               validation, using an already-frozen LR

**There is no `--test` flag and no official-test argument.** `Preg1Role` has no
`OFFICIAL_TEST` member, so the sealed split is unreachable from this program
rather than merely discouraged.

**There is no `--learning-rate` flag on `tune` and no `--seeds` flag anywhere.**
The grid, the tuning seeds and the measurement seeds are precommitted in
`preg1_protocol`; a command-line override of a precommitted constant is the hole
the protocol exists to close.

`measure` refuses to run without `--frozen-lr`, which must name an LR that a
completed tuning artifact selected. Base-only can therefore never influence the
learning rate: by the time it is encoded at all, the LR is already a value in a
file.

Nothing here downloads a model on import. Torch and transformers are imported
lazily inside the run path, which is Colab-only; the local environment is ML-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import (  # noqa: E402
    PREG1_HEAD_SCHEMA_VERSION,
    DETERMINISM_SCOPE,
    FrozenLearningRate,
    NO_SIGNIFICANCE_TEST,
    Preg1Role,
    load_membership,
)
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    BATCH_SIZE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EPOCHS,
    LR_GRID,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    TUNING_SEEDS,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-dir", required=True, help="preg1-split-v1 membership directory")
    parser.add_argument("--derived-train", required=True, help="approved derived TRAIN csv")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--cache-root", required=True, help="representation cache root")
    parser.add_argument("--output-dir", required=True, help="must not already exist")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tune = sub.add_parser("tune", help="VANILLA-only LR sweep on protocol-dev")
    _common(tune)

    measure = sub.add_parser("measure", help="paired measurement on official validation")
    _common(measure)
    measure.add_argument(
        "--official-validation", required=True,
        help="official validation csv; read ONLY after the LR is frozen",
    )
    measure.add_argument(
        "--frozen-lr", required=True, type=float,
        help="the LR a completed VANILLA tuning run selected; must be in the grid",
    )

    args = parser.parse_args(argv)

    try:
        membership = load_membership(args.split_dir)
        if args.command == "measure":
            frozen = FrozenLearningRate(value=args.frozen_lr)
        else:
            frozen = None
    except EvaluationContractViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"pre-G1 burden diagnostic — {PREG1_HEAD_SCHEMA_VERSION}")
    print(f"  command        : {args.command}")
    print(f"  encoder        : {ENCODER_CHECKPOINT} @ {ENCODER_REVISION}")
    print(f"  protocol-train : {len(membership.protocol_train)}")
    print(f"  protocol-dev   : {len(membership.protocol_dev)}")
    print(f"  max_length     : {MAX_LENGTH} | batch {BATCH_SIZE} | epochs {EPOCHS}")
    if args.command == "tune":
        print(f"  grid           : {list(LR_GRID)}")
        print(f"  tuning seeds   : {list(TUNING_SEEDS)} (VANILLA only)")
        print(f"  selection set  : {Preg1Role.PROTOCOL_DEV.value}")
    else:
        print(f"  frozen LR      : {frozen.value} (selected on {frozen.selected_on.value})")
        print(f"  seeds          : {list(MEASUREMENT_SEEDS)}")
        print(f"  measured on    : {Preg1Role.OFFICIAL_VALIDATION.value}")

    print("\nNOT RUN IN THIS BUILD.")
    print(
        "  The encoder pass, the LR sweep and the paired measurement execute on\n"
        "  Colab only. This entry point validates the membership, the frozen-LR\n"
        "  contract and the role boundaries; it produces no downstream score."
    )
    print(f"\n{NO_SIGNIFICANCE_TEST}\n\n{DETERMINISM_SCOPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
