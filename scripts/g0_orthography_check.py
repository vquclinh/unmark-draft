#!/usr/bin/env python3
"""G0 round-trip checker for the Vietnamese orthography core.

Verifies the proposal's G0 invariant on a corpus::

    recompose(decompose(x)) == canon(x)

for every input unit, and enumerates every difference between `x` and
`canon(x)` rather than absorbing it (proposal v1.3 sections 4.2 and 7).

It evaluates the locked canonicalisation policy: `canon` applies UNMARK's fixed
nucleus-based tone placement (`TonePlacement.MODERN`) and returns NFC, so
placement variants such as `hòa`/`hoà` collapse to one canonical form. The
summary separates the two kinds of canonical-only difference -- Unicode
normalisation and tone-placement collapsing -- so neither is hidden inside the
other.

WHAT THIS SCRIPT CANNOT TELL YOU
--------------------------------
**No corpus ships with this repository.** G0 requires >=100K sentences of real
Vietnamese, and nothing here downloads one. Running this script on the built-in
self-check units, or on a small file, does not constitute a G0 result.

The strongest status it will report is therefore
`ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK`. It never prints `G0 PASS`.

Usage
-----
    python scripts/g0_orthography_check.py --self-check
    python scripts/g0_orthography_check.py --input corpus.txt --max-samples 100000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unmark.orthography import (  # noqa: E402
    DEFAULT_TONE_PLACEMENT,
    TonePlacement,
    canon,
    canonical_differences,
    decompose,
    recompose,
)

SCRIPT_VERSION = "g0_orthography_check/1.0"
DEFAULT_OUTPUT_ROOT = "results/g0"

STATUS_READY = "ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK"
STATUS_FAILURES = "ORTHOGRAPHY_CORE_ROUND_TRIP_FAILURES"

# Curated units used by --self-check. Implementation verification only: this is
# not a corpus and a clean result here is not a G0 pass.
SELF_CHECK_UNITS: tuple[str, ...] = (
    "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.",
    "Đường Nguyễn Huệ",
    "ĐẠI HỌC KHOA HỌC TỰ NHIÊN",
    "được đường ưu tiên",
    "hoà",
    "hòa",
    "toi dang hoc khong dau",
    "ắằẳẵặ ấầẩẫậ ếềểễệ ốồổỗộ ớờởỡợ ứừửữự",
    "đĐ ăĂ âÂ êÊ ôÔ ơƠ ưƯ",
    "Nam 2026, GDP tang 6,5% (VAT 10%)!",
    "lien.he@example.com",
    "https://example.edu.vn/tuyen-sinh?id=42&lang=vi",
    "hom nay toi rat vui 😄🎉",
    "Müller façade naïve",
    "toi dang hoc machine learning tai VNU-HCM",
    unicodedata.normalize("NFD", "Tiếng Việt"),
    unicodedata.normalize("NFD", "Đường"),
    # Tone-placement variants: these must collapse onto the nucleus form.
    "hòa", "hoà", "thúy", "thuý", "khỏe", "khoẻ", "qùa", "gía",
    "người được rượu", "nguyễn khuyến", "mùa mưa tuổi",
    "",
    "   ",
)


def iter_units(path: Path, *, max_samples: int | None) -> Iterator[str]:
    """Yield one unit per line of a UTF-8 corpus, without loading it all."""
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if max_samples is not None and count >= max_samples:
                return
            yield line.rstrip("\n")
            count += 1


def check_unit(index: int, text: str, placement: TonePlacement) -> dict[str, Any]:
    """Run the round-trip on one unit and describe the result."""
    record: dict[str, Any] = {
        "index": index,
        "input": text,
        "passed": False,
        "error": None,
        "canonical_equals_input": None,
        "tone_placement_changed": None,
        "input_form": None,
        "reconstructed": None,
        "canonical": None,
        "first_divergence_index": None,
        "anomalies": [],
    }
    try:
        canonical = canon(text, placement)
        parts = decompose(text, placement=placement)
        reconstructed = recompose(parts)
        differences = canonical_differences(text, placement)

        record["canonical"] = canonical
        record["reconstructed"] = reconstructed
        record["passed"] = reconstructed == canonical
        record["canonical_equals_input"] = differences["already_canonical"]
        record["tone_placement_changed"] = differences["tone_placement_changed"]
        record["input_form"] = differences["input_form"]
        record["anomalies"] = [a.value for a in parts.anomalies]
        if not record["passed"]:
            record["first_divergence_index"] = _first_divergence(reconstructed, canonical)
    except Exception as exc:  # noqa: BLE001 - one bad unit must not abort the run
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _first_divergence(a: str, b: str) -> int | None:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(records: Sequence[dict[str, Any]], placement: TonePlacement) -> dict[str, Any]:
    checked = len(records)
    failures = [r for r in records if not r["passed"]]
    errors = [r for r in records if r["error"] is not None]
    canonical_only = [r for r in records if r["passed"] and r["canonical_equals_input"] is False]
    placement_collapsed = [r for r in records if r["tone_placement_changed"]]
    forms = Counter(r["input_form"] for r in records if r["input_form"] is not None)
    anomalies: Counter[str] = Counter()
    for record in records:
        anomalies.update(record["anomalies"])
    return {
        "num_checked": checked,
        "num_passed": checked - len(failures),
        "num_failed": len(failures),
        "num_errors": len(errors),
        "failure_rate": (len(failures) / checked) if checked else None,
        "num_canonical_only_differences": len(canonical_only),
        "canonical_only_difference_rate": (len(canonical_only) / checked) if checked else None,
        "num_tone_placement_collapsed": len(placement_collapsed),
        "tone_placement_collapse_rate": (len(placement_collapsed) / checked) if checked else None,
        "input_forms": dict(forms),
        "anomaly_counts": dict(anomalies),
        "tone_placement": placement.name,
        "variant_collapsing_implemented": placement is not TonePlacement.PRESERVE,
    }


def render_report(run_config: dict[str, Any], summary: dict[str, Any], failures: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# G0 orthography round-trip check")
    a("")
    a(f"Run id: `{run_config['run_id']}`  ")
    a(f"Timestamp (UTC): `{run_config['timestamp_utc']}`  ")
    a(f"Source: `{run_config['source']}`")
    a("")
    a("> **This is not a G0 pass.** G0 requires >=100K sentences of real Vietnamese,")
    a("> and no corpus ships with this repository. This run checks only the units it")
    a("> was given. Read the status line accordingly.")
    a("")
    a("Canonicalisation uses UNMARK's fixed nucleus-based tone-placement convention")
    a("(`TonePlacement.MODERN`), so `hòa` and `hoà` collapse to `hoà`. That is a project")
    a("convention adopted for reproducibility, not a claim about correct Vietnamese")
    a("orthography; see `docs/spec/orthography.md`.")
    a("")
    a("## Configuration")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key in ("source", "max_samples", "tone_placement", "python_version", "platform", "script_version"):
        if run_config.get(key) is not None:
            a(f"| `{key}` | {run_config[key]} |")
    a("")
    a("## Summary")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| units checked | {summary['num_checked']} |")
    a(f"| passed | {summary['num_passed']} |")
    a(f"| failed | {summary['num_failed']} |")
    a(f"| errors | {summary['num_errors']} |")
    rate = summary["failure_rate"]
    a(f"| failure rate | {'n/a' if rate is None else f'{rate:.6f}'} |")
    a(f"| canonical-only differences | {summary['num_canonical_only_differences']} |")
    a(f"| of which tone-placement collapsed | {summary['num_tone_placement_collapsed']} |")
    a(f"| variant collapsing implemented | {summary['variant_collapsing_implemented']} |")
    a("")
    a("`canonical-only differences` counts units that round-tripped correctly but whose")
    a("input was not already canonical. The proposal requires these to be enumerated")
    a("rather than absorbed, so the two causes are separated: Unicode normalisation, and")
    a("tone-placement collapsing (an input written in a different placement convention).")
    a("")
    a("## Input normalisation forms")
    a("")
    a("| Form | Units |")
    a("|---|---:|")
    for form, count in sorted(summary["input_forms"].items()):
        a(f"| {form} | {count} |")
    a("")
    if summary["anomaly_counts"]:
        a("## Orthographic anomalies observed")
        a("")
        a("| Anomaly | Units |")
        a("|---|---:|")
        for name, count in sorted(summary["anomaly_counts"].items()):
            a(f"| `{name}` | {count} |")
        a("")
    a("## Representative failures")
    a("")
    if not failures:
        a("None. Every checked unit satisfied `recompose(decompose(x)) == canon(x)`.")
        a("")
    else:
        for record in failures[:20]:
            a(f"### unit {record['index']}")
            a("")
            a("```text")
            a(f"input        : {record['input']}")
            a(f"canonical    : {record['canonical']}")
            a(f"reconstructed: {record['reconstructed']}")
            if record["error"]:
                a(f"error        : {record['error']}")
            a(f"diverges at  : {record['first_divergence_index']}")
            a("```")
            a("")
        if len(failures) > 20:
            a(f"_{len(failures) - 20} further failures are in `failures.jsonl`._")
            a("")
    a("## Status")
    a("")
    a(f"**`{run_config['status']}`**")
    a("")
    a("Meaning: the orthography core round-tripped every unit it was given. Whether G0")
    a("passes is a separate question that requires a real corpus and a decided")
    a("tone-placement convention. This script never answers it.")
    a("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G0 round-trip check for the Vietnamese orthography core.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None, help="UTF-8 corpus, one unit per line")
    parser.add_argument("--max-samples", type=int, default=None, help="cap the number of units checked")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="root directory for run artifacts")
    parser.add_argument("--run-id", default=None, help="override the timestamped run id")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the built-in curated units instead of a corpus (implementation check, not G0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    placement = DEFAULT_TONE_PLACEMENT

    if args.self_check:
        source = "built-in self-check units"
        units: Iterable[str] = SELF_CHECK_UNITS
    elif args.input:
        path = Path(args.input)
        if not path.is_file():
            raise SystemExit(f"corpus not found: {path}")
        source = str(path)
        units = iter_units(path, max_samples=args.max_samples)
    else:
        raise SystemExit("pass --input <corpus.txt> or --self-check (nothing is downloaded)")

    records = [check_unit(i, text, placement) for i, text in enumerate(units)]
    failures = [r for r in records if not r["passed"]]
    summary = summarize(records, placement)
    status = STATUS_READY if not failures else STATUS_FAILURES

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "source": source,
        "max_samples": args.max_samples,
        "tone_placement": placement.name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "status": status,
        "is_g0_corpus_run": not args.self_check,
        "note": (
            "Not a G0 result. G0 requires >=100K sentences of real Vietnamese and no "
            "corpus ships with this repository. Canonicalisation uses UNMARK's fixed "
            "nucleus-based tone-placement convention (MODERN); see "
            "docs/spec/orthography.md."
        ),
    }

    _write_json(run_dir / "config.json", run_config)
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "failures.jsonl").open("w", encoding="utf-8") as fh:
        for record in failures:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(run_config, summary, failures), encoding="utf-8")

    print(f"checked : {summary['num_checked']}")
    print(f"passed  : {summary['num_passed']}")
    print(f"failed  : {summary['num_failed']}")
    print(f"canonical-only differences: {summary['num_canonical_only_differences']}")
    print(f"  of which tone-placement collapsed: {summary['num_tone_placement_collapsed']}")
    print(f"tone placement: {summary['tone_placement']} (variant collapsing: {summary['variant_collapsing_implemented']})")
    print()
    print(f"Status: {status}")
    print("This is NOT a G0 pass: no corpus ships with this repository, so only the")
    print("units supplied above were checked.")
    print()
    print(f"Results: {run_dir}")
    return 0 if not failures else 1


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
