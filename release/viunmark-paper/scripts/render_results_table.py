#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from viunmark.config import ViUnMarkContractError


def render_markdown_table(result_path: Path) -> str:
    if not result_path.is_file():
        raise ViUnMarkContractError(
            f"missing canonical result artifact: {result_path}; "
            "supply the author-approved machine-readable result file before rendering tables"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ViUnMarkContractError("canonical result artifact must contain a rows array")
    lines = ["| Condition | Macro-F1 | Accuracy |", "| --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['macro_f1']} | {row['accuracy']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render tables from canonical frozen result artifacts.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = render_markdown_table(args.results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
