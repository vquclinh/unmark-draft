#!/usr/bin/env python3
"""B3A eligibility check against the real pinned Vietnamese syllable inventory.

A linguistic implementation check, not a downstream benchmark: it reports what
the pinned inventory accepts and rejects, and confirms that the B2 scientific
guard is now satisfied. It measures nothing about task performance and produces
no dataset.

Requires the inventory to have been fetched and verified:

    .venv/bin/python scripts/fetch_vietnamese_syllable_inventory.py
    .venv/bin/python scripts/b3a_eligibility_check.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.corruption import (  # noqa: E402
    CORRUPTION_SCHEMA_VERSION,
    EligibilityPolicy,
    active_eligibility_policy,
    corrupt,
)
from unmark.linguistics import (  # noqa: E402
    ELIGIBILITY_SCHEMA_VERSION,
    InventoryUnavailable,
    classify_candidate,
    load_inventory,
)
from unmark.orthography import Eligibility  # noqa: E402

STATUS_OK = "B3A_ELIGIBILITY_RESOLVED"
STATUS_FAIL = "B3A_ELIGIBILITY_CHECK_FAIL"

KNOWN_VALID = [
    "tôi", "đang", "nghiên", "cứu", "xử", "lý", "ngôn", "ngữ", "tự", "nhiên",
    "học", "đường", "phở", "người", "được", "nguyễn", "hoà", "thuý", "khoẻ", "quả",
]
KNOWN_INVALID = [
    "machine", "learning", "python", "pytorch", "café", "google", "server",
    "email", "javascript", "strength", "qwerty", "transformer",
]
AMBIGUOUS_ASCII = ["ban", "the", "com", "on", "in", "an", "la", "co", "nam", "ma", "cam", "hoa"]
MIXED_TEXTS = [
    "toi dung Python va PyTorch de train model",
    "Tôi đang học machine learning tại VNU-HCM",
    "café ngon lắm",
    "Xem tại https://example.edu.vn nhé",
    "Liên hệ lien.he@example.com",
    "Năm 2026, GDP tăng 6,5% 😄",
]


def run(inventory: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    examples: list[dict[str, Any]] = []

    def record(kind: str, text: str, **extra: Any) -> Eligibility:
        verdict = classify_candidate(text, inventory)
        examples.append({"kind": kind, "text": text, "eligibility": verdict.value, **extra})
        return verdict

    accepted = 0
    for text in KNOWN_VALID:
        if record("known_valid", text) is Eligibility.VIETNAMESE_CANDIDATE:
            accepted += 1
        else:
            failures.append(f"known-valid Vietnamese syllable rejected: {text}")

    rejected = 0
    for text in KNOWN_INVALID:
        if record("known_invalid", text) is Eligibility.NOT_APPLICABLE:
            rejected += 1
        else:
            failures.append(f"known-foreign token accepted: {text}")

    ambiguous_accepted = [
        text for text in AMBIGUOUS_ASCII if record("ambiguous_ascii", text) is Eligibility.VIETNAMESE_CANDIDATE
    ]

    mixed: list[dict[str, Any]] = []
    for text in MIXED_TEXTS:
        result = corrupt(text, "STRIP_ALL", seed=20260819, sample_id=f"mix-{len(mixed)}")
        row = {
            "kind": "mixed_text",
            "text": text,
            "corrupted": result.corrupted_text,
            "candidate_units": result.candidate_units,
            "eligible_units": result.eligible_units,
            "eligible_spans": [d.base_text for d in result.decisions],
        }
        mixed.append(row)
        examples.append(row)
        if result.eligible_units > result.candidate_units:
            failures.append(f"eligible exceeded candidates on: {text}")

    # The B2 guard must now be satisfied, and eligibility must be invariant
    # across every corruption condition.
    policy = active_eligibility_policy()
    if policy is not EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY:
        failures.append(f"eligibility policy is {policy.name}, expected VIETNAMESE_SYLLABLE_INVENTORY")

    baseline_text = "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."
    baseline = {
        d.base_text: d.eligibility for d in corrupt(baseline_text, "FULL", seed=1, sample_id="inv").decisions
    }
    for condition in ("P25", "P50", "P75", "P100", "STRIP_ALL"):
        current = {
            d.base_text: d.eligibility
            for d in corrupt(baseline_text, condition, seed=1, sample_id="inv").decisions
        }
        if current != baseline:
            failures.append(f"eligibility changed under {condition}")

    summary = {
        "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "corruption_schema_version": CORRUPTION_SCHEMA_VERSION,
        "eligibility_policy": policy.value,
        "b2_scientific_guard": "SATISFIED" if policy is EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY else "BLOCKING",
        "inventory": inventory.summary(),
        "known_valid_total": len(KNOWN_VALID),
        "known_valid_accepted": accepted,
        "known_invalid_total": len(KNOWN_INVALID),
        "known_invalid_rejected": rejected,
        "ambiguous_ascii_total": len(AMBIGUOUS_ASCII),
        "ambiguous_ascii_accepted": len(ambiguous_accepted),
        "ambiguous_ascii_accepted_examples": ambiguous_accepted,
        "mixed_text_examples": len(mixed),
        "num_failures": len(failures),
        "failures": failures,
    }
    return summary, examples, failures


def render_report(config: dict[str, Any], summary: dict[str, Any], examples: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    a = lines.append
    inv = summary["inventory"]
    prov = inv["provenance"]
    a("# B3A Vietnamese syllable eligibility check")
    a("")
    a(f"Run id: `{config['run_id']}`  ")
    a(f"Timestamp (UTC): `{config['timestamp_utc']}`")
    a("")
    a("> A linguistic implementation check, not a downstream benchmark. It reports what")
    a("> the pinned inventory accepts and rejects. No task performance is measured and no")
    a("> dataset is produced.")
    a("")
    a("## Inventory provenance")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    a(f"| source | `{prov['source_name']}` by `{prov['source_author']}` |")
    a(f"| revision | `{prov['source_revision']}` |")
    a(f"| sha256 | `{prov['sha256']}` |")
    a(f"| license | `{prov['license_status']}` (raw file not committed) |")
    a(f"| eligibility schema | `{summary['eligibility_schema_version']}` |")
    a(f"| corruption schema | `{summary['corruption_schema_version']}` (unchanged by B3A) |")
    a("")
    a("| Count | Value |")
    a("|---|---:|")
    a(f"| raw entries | {inv['raw_entry_count']} |")
    a(f"| unique canonical entries | {inv['unique_canonical_entry_count']} |")
    a(f"| unique stripped forms | {inv['unique_stripped_form_count']} |")
    a(f"| collisions after stripping | {inv['collisions_after_stripping']} |")
    a("")
    a("Collisions are expected: `ma`, `má`, `mà`, `mả`, `mã`, `mạ` all reduce to one")
    a("stripped form. Membership is tested on that form, which is what makes eligibility")
    a("identical for clean and corrupted text.")
    a("")
    a("## Acceptance and rejection")
    a("")
    a("| Group | Result |")
    a("|---|---|")
    a(f"| known-valid Vietnamese | {summary['known_valid_accepted']}/{summary['known_valid_total']} accepted |")
    a(f"| known-foreign tokens | {summary['known_invalid_rejected']}/{summary['known_invalid_total']} rejected |")
    a(f"| ambiguous ASCII | {summary['ambiguous_ascii_accepted']}/{summary['ambiguous_ascii_total']} accepted |")
    a("")
    a("### Ambiguous ASCII accepted (the deliberate error mode)")
    a("")
    a(f"`{'`, `'.join(summary['ambiguous_ascii_accepted_examples'])}`")
    a("")
    a("These are real stripped Vietnamese syllables that are also English words. Proposal")
    a("§4.3 resolves such spans towards Vietnamese and calls it \"a known and deliberate")
    a("error mode\". The classifier is orthographic, never semantic: no language")
    a("identification, frequency list, dictionary or context is consulted.")
    a("")
    a("## Mixed-language classification")
    a("")
    a("| Input | STRIP_ALL output | Candidates | Eligible |")
    a("|---|---|---:|---:|")
    for row in examples:
        if row["kind"] != "mixed_text":
            continue
        a(f"| {row['text']} | {row['corrupted']} | {row['candidate_units']} | {row['eligible_units']} |")
    a("")
    a("## B2 scientific guard")
    a("")
    a(f"Eligibility policy: `{summary['eligibility_policy']}`  ")
    a(f"Guard status: **{summary['b2_scientific_guard']}**")
    a("")
    a("Eligibility was verified identical across `FULL`, `P25`, `P50`, `P75`, `P100` and")
    a("`STRIP_ALL`, which is the invariance the base grid depends on.")
    a("")
    a("## Result")
    a("")
    if summary["num_failures"]:
        a(f"**`{STATUS_FAIL}`** — {summary['num_failures']} failure(s):")
        a("")
        for failure in summary["failures"]:
            a(f"* {failure}")
    else:
        a(f"**`{STATUS_OK}`**")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B3A eligibility check (offline, needs the fetched inventory).")
    parser.add_argument("--output-root", default="results/b3a")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    try:
        inventory = load_inventory(REPO_ROOT / "configs/linguistics/vietnamese_syllables.yaml", REPO_ROOT)
    except InventoryUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary, examples, failures = run(inventory)
    status = STATUS_FAIL if failures else STATUS_OK

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_root)
    run_dir = root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "corruption_schema_version": CORRUPTION_SCHEMA_VERSION,
        "inventory_provenance": inventory.provenance.to_dict() if inventory.provenance else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "status": status,
        "note": (
            "Linguistic implementation check, not a downstream benchmark or dataset. "
            "The upstream syllable list is not redistributed by this repository."
        ),
    }

    for name, payload in (("config.json", config), ("summary.json", summary)):
        with (run_dir / name).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    with (run_dir / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for row in examples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(config, summary, examples), encoding="utf-8")

    print(f"revision            : {summary['inventory']['provenance']['source_revision']}")
    print(f"sha256              : {summary['inventory']['provenance']['sha256']}")
    print(f"raw entries         : {summary['inventory']['raw_entry_count']}")
    print(f"unique stripped     : {summary['inventory']['unique_stripped_form_count']}")
    print(f"collisions          : {summary['inventory']['collisions_after_stripping']}")
    print(f"known-valid accepted: {summary['known_valid_accepted']}/{summary['known_valid_total']}")
    print(f"known-foreign reject: {summary['known_invalid_rejected']}/{summary['known_invalid_total']}")
    print(f"ambiguous accepted  : {summary['ambiguous_ascii_accepted']}/{summary['ambiguous_ascii_total']}")
    print(f"B2 guard            : {summary['b2_scientific_guard']}")
    print()
    print(f"Status: {status}")
    print("Linguistic implementation check only - not a benchmark or dataset.")
    print()
    print(f"Results: {run_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
