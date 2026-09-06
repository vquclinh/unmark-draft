#!/usr/bin/env python3
"""READ-ONLY verifier for the two frozen Stage-1 finalist adapters.

Audit 048. Stage-1 training is CLOSED and the candidate set is frozen at exactly
two already-trained checkpoints (`A` = final_main seed 36930 @ 3500, `B` =
lr_pilot seed 21230 @ 14500). This script proves that a file the operator points
it at **is** one of them, or refuses.

**It trains nothing and writes nothing to the checkpoint.** No optimizer is
constructed, no backward is called, the file is opened read-only and is never
copied or modified. Official UIT-VSFC TEST is never touched, and no downstream
data is read: this verifies identity, not quality, and cannot participate in
choosing between A and B.

The checkpoint path is supplied by the operator. Production scientific code in
this repository does not hard-code `/content/drive/...`; the heavy artifacts live
outside git (`*.pt` is git-ignored) and their location is an operator fact.

Usage::

    python -B scripts/stage1_verify_finalist_checkpoint.py \\
        --finalist A --checkpoint /path/to/training-checkpoint-3500.pt

    # Finalist B's digest is not yet bound. Establish it from the real file:
    python -B scripts/stage1_verify_finalist_checkpoint.py \\
        --finalist B --checkpoint /path/to/training-checkpoint-14500.pt \\
        --emit-binding docs/audits/evidence/048-finalist-b-binding.json

`--emit-binding` deliberately does NOT edit the freeze artifact. It verifies
every field except the digest, then writes a small evidence record (identifiers,
scalars and hashes only -- no tensors) for a human to review and paste into
`docs/spec/stage1-adapter-finalists.json`. A tool that both discovers and
installs its own evidence would be self-certifying.

Exit codes:

* ``0`` -- the checkpoint is the named finalist.
* ``1`` -- it is not, or the freeze artifact is inconsistent (fail closed).
* ``2`` -- the operator's invocation was wrong (bad path, unknown finalist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unmark.stage1.finalists import (  # noqa: E402
    BINDING_FIELDS,
    FINALIST_FREEZE_SCHEMA_VERSION,
    FinalistFreezeViolation,
    finalist_for,
    finalists_from_freeze,
    freeze_is_complete,
    load_freeze,
    pending_finalists,
    verify_finalist_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only identity check for a frozen Stage-1 finalist adapter.",
    )
    parser.add_argument(
        "--finalist", required=True, choices=("A", "B"),
        help="which frozen finalist the checkpoint is claimed to be",
    )
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="operator-supplied checkpoint path; never inferred, never modified",
    )
    parser.add_argument(
        "--freeze", default=None, metavar="PATH",
        help="override the committed freeze artifact (testing only)",
    )
    parser.add_argument(
        "--emit-binding", default=None, metavar="PATH",
        help=(
            "for a finalist whose digest is still PENDING: verify every other field, "
            "then write a reviewable evidence record carrying the computed sha256. "
            "Does NOT edit the freeze artifact."
        ),
    )
    parser.add_argument(
        "--report", default=None, metavar="PATH",
        help="also write the evidence record here",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"invocation error: no such checkpoint: {checkpoint}", file=sys.stderr)
        return 2

    try:
        freeze = load_freeze(args.freeze) if args.freeze else load_freeze()
    except FinalistFreezeViolation as error:
        print(f"REFUSED: the finalist freeze artifact is not valid:\n{error}", file=sys.stderr)
        return 1

    finalists = finalists_from_freeze(freeze)
    try:
        finalist = finalist_for(args.finalist, finalists)
    except FinalistFreezeViolation as error:
        print(f"invocation error: {error}", file=sys.stderr)
        return 2

    binding = args.emit_binding is not None
    if binding and finalist.sha256_bound:
        print(
            f"invocation error: finalist {finalist.key} already has a bound digest "
            f"({finalist.checkpoint_sha256}); --emit-binding is only for a PENDING one. "
            "Run without it to verify.",
            file=sys.stderr,
        )
        return 2

    try:
        evidence = verify_finalist_checkpoint(
            checkpoint, finalist, require_bound_digest=not binding
        )
    except FinalistFreezeViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "REFUSED: torch is required to inspect adapter tensors and is not installed "
            "in this environment. Run this on the machine that holds the checkpoints.",
            file=sys.stderr,
        )
        return 1

    evidence["freeze_schema_version"] = FINALIST_FREEZE_SCHEMA_VERSION
    evidence["checkpoint_path"] = str(checkpoint)

    document = json.dumps(evidence, indent=2, sort_keys=True)
    print(document)

    for destination in (args.report, args.emit_binding):
        if destination:
            out = Path(destination)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(document + "\n", encoding="utf-8")

    if binding:
        checklist = "\n".join(
            f"  {n}. {field}" for n, field in enumerate(BINDING_FIELDS, start=1)
        )
        print(
            f"\nBINDING EVIDENCE WRITTEN: {args.emit_binding}\n"
            f"Finalist {finalist.key} sha256 = {evidence['checkpoint_sha256']}\n\n"
            f"This script does not edit the freeze. Binding is a "
            f"{len(BINDING_FIELDS)}-field edit; all of them must change together, and "
            f"every partial edit is refused by validate_freeze_payload:\n"
            f"{checklist}\n\n"
            f"  1 and 2 take the digest above; 3 becomes [] once no digest is pending; "
            f"4 becomes true.\n"
            f"Then re-run: pytest -q tests/test_stage1_finalist_freeze.py",
            file=sys.stderr,
        )
        return 0

    still_pending = pending_finalists(finalists)
    print(
        f"\nVERIFIED: {checkpoint} is Stage-1 finalist {finalist.key} "
        f"({finalist.source_stage} seed {finalist.run_seed} @ {finalist.update}).",
        file=sys.stderr,
    )
    if not freeze_is_complete(finalists):
        print(
            f"NOTE: the finalist freeze is INCOMPLETE; digests still pending: "
            f"{list(still_pending)}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
