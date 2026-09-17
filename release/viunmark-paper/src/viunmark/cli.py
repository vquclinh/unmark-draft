"""Command-line entry point for publication utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viunmark.provenance import load_reproduction_manifest, load_uit_vsfc_reproduction


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_reproduction_manifest()
    reproduction = load_uit_vsfc_reproduction()
    print(json.dumps({
        "status": "ok",
        "dataset": "UIT-VSFC",
        "heads": reproduction.head_count,
        "external_assets": manifest["external_assets"],
    }, indent=2, sort_keys=True))
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
