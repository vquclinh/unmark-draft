#!/usr/bin/env python3
"""B2 corruption self-check: implementation verification, offline.

Exercises every implemented condition on curated examples and asserts the
contracts that matter: repeatability, seed and sample_id sensitivity, order
independence, canonical-variant equivalence, base invariance, and the
`ngang`-versus-stripped-tone ambiguity metadata.

The examples here are **implementation verification only**. They are not a
dataset, not a benchmark and not a training corpus, and no downstream dataset
exists at B2. Nothing is downloaded.

    .venv/bin/python scripts/b2_corruption_self_check.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unmark.corruption import (  # noqa: E402
    ACTIVE_ELIGIBILITY_POLICY,
    CONDITIONS,
    CORRUPTION_SCHEMA_VERSION,
    UNIMPLEMENTED_CONDITIONS,
    CorruptionPurpose,
    EligibilityUnresolved,
    corrupt,
    corrupt_batch,
    is_resolved,
)
from unmark.orthography import Eligibility, ObservedTone, Tone, canon, strip_to_base  # noqa: E402

DEFAULT_OUTPUT_ROOT = "results/b2"
STATUS_OK = "B2_CORRUPTION_SELF_CHECK_PASS"
STATUS_FAIL = "B2_CORRUPTION_SELF_CHECK_FAIL"

# Curated verification examples. Not a corpus.
CASES: tuple[tuple[str, str], ...] = (
    ("vi_plain", "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."),
    ("vi_tones", "má mà mả mã mạ ma"),
    ("vi_letters", "đường ăn cân ơn ưu êm ôm"),
    ("vi_upper", "ĐẠI HỌC KHOA HỌC TỰ NHIÊN"),
    ("vi_ngang_only", "toi di hoc"),
    ("variant_a", "hòa"),
    ("variant_b", "hoà"),
    ("mixed", "toi dung Python va PyTorch de train model"),
    ("punct_digits", "Năm 2026, GDP tăng 6,5% (VAT 10%)!"),
    ("url", "Xem tại https://example.edu.vn/tuyen-sinh?id=42&lang=vi"),
    ("email", "Liên hệ qua lien.he@example.com nhé"),
    ("emoji", "hôm nay tôi rất vui 😄🎉"),
    ("foreign_marks", "Müller façade naïve"),
    ("nfd", unicodedata.normalize("NFD", "Tiếng Việt")),
    ("no_syllables", "2026 6,5% !!! 😄"),
    ("one_syllable", "phở"),
    ("empty", ""),
)

SEED = 20260819

# The self-check runs in explicit provisional mode: it is implementation
# verification, not scientific corruption generation. Under the default
# SCIENTIFIC purpose every call here would (correctly) raise.
PURPOSE = CorruptionPurpose.SELF_CHECK


def run_checks() -> tuple[list[dict[str, Any]], list[str]]:
    """Returns (per-case records, failure descriptions)."""
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    for sample_id, text in CASES:
        for name in CONDITIONS:
            result = corrupt(text, name, seed=SEED, sample_id=sample_id, purpose=PURPOSE)
            record = result.to_dict()
            records.append(record)

            check(
                result.original_text == text,
                f"{sample_id}/{name}: original_text was mutated",
            )
            check(
                result.canonical_clean_text == canon(text),
                f"{sample_id}/{name}: canonical_clean_text != canon(text)",
            )
            check(
                result.schema_version == CORRUPTION_SCHEMA_VERSION,
                f"{sample_id}/{name}: schema version not recorded",
            )
            # Base invariance: corruption removes only information already
            # represented outside the base channel.
            check(
                strip_to_base(result.canonical_clean_text) == strip_to_base(result.corrupted_text),
                f"{sample_id}/{name}: base changed under corruption",
            )
            # Repeatability.
            again = corrupt(text, name, seed=SEED, sample_id=sample_id, purpose=PURPOSE)
            check(
                again.corrupted_text == result.corrupted_text
                and [d.selected for d in again.decisions] == [d.selected for d in result.decisions],
                f"{sample_id}/{name}: not repeatable",
            )
            # The corrupted text must itself be canonical.
            check(
                canon(result.corrupted_text) == result.corrupted_text,
                f"{sample_id}/{name}: corrupted text is not canonical",
            )
            # Selected-unit metadata must match the actual output.
            for decision in result.decisions:
                if decision.modified:
                    check(
                        decision.selected,
                        f"{sample_id}/{name}: unit {decision.unit_index} modified without being selected",
                    )
                if decision.tone_mark_removed:
                    check(
                        decision.corrupted_observed_tone is ObservedTone.UNMARKED,
                        f"{sample_id}/{name}: stripped tone did not become UNMARKED",
                    )

    # --- canonical placement variants corrupt identically -------------------
    for name in CONDITIONS:
        a = corrupt("hòa", name, seed=SEED, sample_id="v", purpose=PURPOSE)
        b = corrupt("hoà", name, seed=SEED, sample_id="v", purpose=PURPOSE)
        check(
            a.corrupted_text == b.corrupted_text and a.text_identity == b.text_identity,
            f"variants/{name}: placement variants corrupted differently",
        )

    # --- seed and sample_id sensitivity; order independence -----------------
    long_text = " ".join(["má", "mà", "mả", "mã", "mạ", "phở", "cứu", "học", "tự", "ngữ"] * 6)
    base = corrupt(long_text, "P50", seed=1, sample_id="a", purpose=PURPOSE)
    other_seed = corrupt(long_text, "P50", seed=2, sample_id="a", purpose=PURPOSE)
    other_id = corrupt(long_text, "P50", seed=1, sample_id="b", purpose=PURPOSE)
    check(base.corrupted_text != other_seed.corrupted_text, "seed did not change selection")
    check(base.corrupted_text != other_id.corrupted_text, "sample_id did not change selection")

    samples = [(text, sid) for sid, text in CASES]
    forward = {r.sample_id: r.corrupted_text for r in corrupt_batch(samples, "P50", seed=SEED, purpose=PURPOSE)}
    reversed_ = {r.sample_id: r.corrupted_text for r in corrupt_batch(list(reversed(samples)), "P50", seed=SEED, purpose=PURPOSE)}
    check(forward == reversed_, "dataset reordering changed per-sample corruption")

    # --- ambiguity metadata --------------------------------------------------
    ambiguity = corrupt("ma má", "P100", seed=SEED, sample_id="amb", purpose=PURPOSE)
    genuine, stripped = ambiguity.decisions
    check(
        genuine.corrupted_observed_tone is ObservedTone.UNMARKED
        and genuine.clean_lexical_tone is Tone.NGANG
        and not genuine.tone_mark_removed,
        "genuine ngang metadata wrong",
    )
    check(
        stripped.corrupted_observed_tone is ObservedTone.UNMARKED
        and stripped.clean_lexical_tone is Tone.SAC
        and stripped.tone_mark_removed,
        "stripped-tone metadata wrong",
    )
    check(
        genuine.oracle_tone_is_genuine_ngang and stripped.oracle_tone_is_missing,
        "H4 oracle views wrong",
    )
    # --- the pre-training guard must be active -----------------------------
    check(not is_resolved(), "eligibility policy is resolved but GAP-2 is still open")
    try:
        corrupt("Tôi", "P50", seed=SEED, sample_id="guard")
        failures.append("SCIENTIFIC-purpose corruption did not raise while eligibility is unresolved")
    except EligibilityUnresolved as exc:
        message = str(exc)
        check("GAP-2" in message, "guard message does not name GAP-2")
        check("B3" in message, "guard message does not name the B3 resolution owner")

    # --- provisional counts must not be presented as eligible syllables ------
    provisional = corrupt("toi dung Python", "P50", seed=SEED, sample_id="prov", purpose=PURPOSE)
    check(provisional.provisional_eligibility, "provisional result not flagged")
    check(
        all(d.eligibility is Eligibility.UNDECIDED for d in provisional.decisions),
        "a candidate span was reported as resolved-eligible",
    )
    payload = provisional.to_dict()
    check("eligible_units" not in payload, "artifact exposes eligible_units while GAP-2 is open")
    check("realized_probability" not in payload, "artifact exposes realized_probability")
    try:
        provisional.eligible_units
        failures.append("eligible_units returned a provisional number")
    except EligibilityUnresolved:
        pass

    return records, failures


def summarize(records: Sequence[dict[str, Any]], failures: Sequence[str]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for name in CONDITIONS:
        rows = [r for r in records if r["condition"] == name]
        changed = [r for r in rows if r["corrupted_text"] != r["canonical_clean_text"]]
        by_condition[name] = {
            "cases": len(rows),
            "cases_changed": len(changed),
            "candidate_units": sum(r["candidate_units"] for r in rows),
            "selected_candidates": sum(r["selected_candidates"] for r in rows),
            "modified_candidates": sum(r["modified_candidates"] for r in rows),
            "base_invariant_all": all(r["base_invariant"] for r in rows),
        }
    return {
        "schema_version": CORRUPTION_SCHEMA_VERSION,
        "eligibility_policy": ACTIVE_ELIGIBILITY_POLICY.value,
        "provisional_eligibility": not is_resolved(),
        "purpose": PURPOSE.name,
        "num_cases": len(CASES),
        "num_conditions": len(CONDITIONS),
        "num_records": len(records),
        "num_failures": len(failures),
        "failures": list(failures),
        "by_condition": by_condition,
        "unimplemented_conditions": sorted(UNIMPLEMENTED_CONDITIONS),
        "seed": SEED,
    }


def render_report(run_config: dict[str, Any], summary: dict[str, Any], records: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# B2 corruption self-check")
    a("")
    a(f"Run id: `{run_config['run_id']}`  ")
    a(f"Timestamp (UTC): `{run_config['timestamp_utc']}`  ")
    a(f"Corruption schema: `{summary['schema_version']}`")
    a("")
    a(f"Eligibility policy: `{summary['eligibility_policy']}`  ")
    a(f"Purpose: `{summary['purpose']}`")
    a("")
    a("> **Provisional eligibility.** GAP-2 is open, so the counts below are over")
    a("> *candidate spans* -- every maximal alphabetic run, English words included --")
    a("> not over confirmed Vietnamese syllables. This is an implementation fallback,")
    a("> not the scientific protocol: `corrupt()` refuses by default and had to be")
    a("> asked for `SELF_CHECK` mode explicitly. GAP-2 must be closed before stage-1")
    a("> training or the main experiments. Owner: B3 / pre-training.")
    a("")
    a("> These are curated implementation-verification examples.")
    a("> They are **not a dataset, not a benchmark and not a training corpus**, and")
    a("> nothing was downloaded. No claim about natural-corpus behaviour follows from")
    a("> this run.")
    a("")
    a("## Conditions")
    a("")
    a("| Condition | Cases | Changed | Candidate spans | Selected | Modified | Base invariant |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for name, block in summary["by_condition"].items():
        a(
            f"| `{name}` | {block['cases']} | {block['cases_changed']} | {block['candidate_units']} | "
            f"{block['selected_candidates']} | {block['modified_candidates']} | {block['base_invariant_all']} |"
        )
    a("")
    a("`Selected` counts units the deterministic score chose; `Modified` counts units whose")
    a("text actually changed. They differ because a selected `ngang` syllable has no mark to")
    a("remove -- proposal section 4.3: \"a *ngang* syllable is *invariant*\".")
    a("")
    a(f"Not implemented: {', '.join(summary['unimplemented_conditions']) or 'none'}")
    a("")
    a("## Representative output")
    a("")
    a("| Condition | Clean | Corrupted |")
    a("|---|---|---|")
    for name in summary["by_condition"]:
        row = next(r for r in records if r["condition"] == name and r["sample_id"] == "vi_plain")
        a(f"| `{name}` | {row['canonical_clean_text']} | {row['corrupted_text']} |")
    a("")
    a("## Result")
    a("")
    if summary["num_failures"]:
        a(f"**`{STATUS_FAIL}`** -- {summary['num_failures']} check(s) failed:")
        a("")
        for failure in summary["failures"]:
            a(f"* {failure}")
    else:
        a(f"**`{STATUS_OK}`**")
        a("")
        a("Every curated case was repeatable, canonical, base-invariant and correctly")
        a("annotated. This says nothing about natural-corpus behaviour.")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 corruption self-check (offline).")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    records, failures = run_checks()
    summary = summarize(records, failures)
    status = STATUS_FAIL if failures else STATUS_OK

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_root)
    run_dir = root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "corruption_schema_version": CORRUPTION_SCHEMA_VERSION,
        "seed": SEED,
        "conditions": sorted(CONDITIONS),
        "unimplemented_conditions": sorted(UNIMPLEMENTED_CONDITIONS),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "status": status,
        "eligibility_policy": ACTIVE_ELIGIBILITY_POLICY.value,
        "provisional_eligibility": not is_resolved(),
        "purpose": PURPOSE.name,
        "note": (
            "Curated implementation-verification examples only. Not a dataset, benchmark "
            "or training corpus; nothing was downloaded. Counts are over CANDIDATE spans, "
            "not confirmed Vietnamese syllables: GAP-2 is open and must be closed before "
            "stage-1 training or the main experiments (owner: B3/pre-training)."
        ),
    }

    _write_json(run_dir / "config.json", run_config)
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(run_config, summary, records), encoding="utf-8")

    print(f"schema   : {CORRUPTION_SCHEMA_VERSION}")
    print(f"policy   : {ACTIVE_ELIGIBILITY_POLICY.value} (provisional: {not is_resolved()})")
    print(f"cases    : {summary['num_cases']} x {summary['num_conditions']} conditions = {summary['num_records']} records")
    print(f"failures : {summary['num_failures']}")
    for failure in failures[:10]:
        print(f"  - {failure}")
    print()
    print(f"Status: {status}")
    print("Curated verification examples only - not a dataset, benchmark or corpus.")
    print("Counts are over CANDIDATE spans, not confirmed Vietnamese syllables (GAP-2 open).")
    print()
    print(f"Results: {run_dir}")
    return 0 if not failures else 1


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
