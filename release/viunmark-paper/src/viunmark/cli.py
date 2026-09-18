"""Command-line entry point for publication utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from viunmark.assets import verify_from_config


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_from_config(args.config, manifest_only=args.manifest_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] not in {"ok", "manifest_only"}:
        print(
            f"asset verification failed: {len(report['missing'])} missing, "
            f"{len(report['mismatched'])} sha256 mismatched; "
            "run with --manifest-only to inspect expected paths",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_diagnostic(args: argparse.Namespace) -> int:
    from viunmark.diagnostics.scale_pathway_preflight import validate_scale_pathway_config

    analyses = {
        "scale-preflight": validate_scale_pathway_config,
        "pooling-bridge": lambda: {"implemented": "same-forward feature extraction"},
        "pooling-comparison": lambda: {"implemented": "matched-plan validation"},
        "decision-geometry": lambda: {"implemented": "cosine distances and prediction agreement"},
        "complementarity": lambda: {"implemented": "matched-recipe stream discovery and averaging"},
        "gain-factorization": lambda: {"implemented": "final-system computable factors"},
        "adapted-only": lambda: {"implemented": "diagnostic fusion graph"},
        "phobert-readout": lambda: {"implemented": "provenance validation"},
        "viunmark": lambda: {"implemented": "final fusion graph"},
    }
    print(json.dumps(analyses[args.analysis](), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="viunmark")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-assets")
    verify.add_argument("--config", type=Path, default=Path("configs/uit_vsfc/paper.json"))
    verify.add_argument("--manifest-only", action="store_true")
    verify.set_defaults(func=_cmd_verify)
    diag = sub.add_parser("diagnostic")
    diag.add_argument(
        "--analysis",
        required=True,
        choices=[
            "scale-preflight",
            "pooling-bridge",
            "pooling-comparison",
            "decision-geometry",
            "complementarity",
            "gain-factorization",
            "adapted-only",
            "phobert-readout",
            "viunmark",
        ],
    )
    diag.set_defaults(func=_cmd_diagnostic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
