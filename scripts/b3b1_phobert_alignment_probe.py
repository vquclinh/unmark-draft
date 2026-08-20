#!/usr/bin/env python3
"""B3B-1 manual alignment probe for PhoBERT's slow tokenizer.

**FOR GOOGLE COLAB (CPU), NOT THE LOCAL `.venv`.** Tokenizer only — no model
weights, no torch, no training.

Why
---
The scientifically usable B3B-0 run reported `offset_availability = ABSENT` for
every path: the authoritative tokenizer is `PhobertTokenizer` (`is_fast=False`)
and returns no `offset_mapping`. Proposal §4.4 propagates channel labels "by
tracking character offsets through tokenization", so that step needs an
alternative — and switching to a fast tokenizer for offsets is not one, because
the slow tokenizer's ids are the frozen encoder's own vocabulary.

The hypothesis under test: tokenizing each base span independently and stripping
the fastBPE `@@` continuation marker reconstructs the span's surface exactly,
which yields deterministic half-open character ranges per piece.

**This probe tests that hypothesis. Nothing here declares it validated** — that
is what the run's output is for.

Usage
-----
    pip install "transformers==4.57.6"
    export HF_HOME="$PWD/.hf-cache"
    python scripts/fetch_vietnamese_syllable_inventory.py
    python scripts/b3b1_phobert_alignment_probe.py \
        --checkpoint vinai/phobert-base \
        --revision 01daacda68afe13d83023d16ec647239e344a1e6
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.alignment import (  # noqa: E402
    AlignmentFailureReason,
    SpanAlignmentStatus,
    align_span,
    compare_sequences,
    summarize_alignments,
)
from unmark.alignment.contracts import REPO_LOCAL_HF_CACHE, TokenizerContract  # noqa: E402
from unmark.linguistics import load_inventory, make_classifier  # noqa: E402
from unmark.orthography import Eligibility, canon, decompose  # noqa: E402

STATUS_OK = "B3B1_ALIGNMENT_PROBE_COMPLETE"
STATUS_FAIL = "B3B1_ALIGNMENT_PROBE_INCOMPLETE"

DEFAULT_CHECKPOINT = "vinai/phobert-base"
# The revision verified by the scientifically usable B3B-0 run. Reused here for
# comparability. It is a PROBE revision, not the final backbone lock: D-B3B0-002
# is still open.
B3B0_PROBE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"

FULL_SHA_LENGTH = 40
_SNAPSHOT_PATTERN = re.compile(r"[/\\]snapshots[/\\]([0-9a-f]{40})(?:[/\\]|$)")

# Curated syllables covering all six tones, every Vietnamese letter diacritic,
# case and both normalisation forms.
TONE_CASES = ["ma", "má", "mà", "mả", "mã", "mạ"]
LETTER_CASES = ["ăn", "cân", "êm", "ôm", "ơn", "ưu", "đi", "Đại", "được", "người"]
CASE_CASES = ["Tôi", "TÔI", "tôi", "ĐẠI", "Học"]

SENTENCES: tuple[tuple[str, str], ...] = (
    ("vi_research", "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."),
    ("vi_multisyllable", "Trường đại học công nghệ thông tin"),
    ("vi_city", "Thành phố Hồ Chí Minh rất đẹp"),
    ("vi_proper_names", "Nguyễn Việt Anh đang làm việc tại Hà Nội"),
    ("vi_uppercase", "ĐẠI HỌC KHOA HỌC TỰ NHIÊN"),
    ("mixed_en", "tôi dùng Python và PyTorch để train model"),
    ("mixed_ml", "Tôi đang học machine learning tại VNU-HCM"),
    ("punctuation", "Năm 2026, GDP tăng 6,5% (VAT 10%)!"),
    ("url", "Xem tại https://example.edu.vn/tuyen-sinh?id=42&lang=vi"),
    ("email", "Liên hệ qua lien.he@example.com nhé"),
    ("emoji", "hôm nay tôi rất vui 😄🎉"),
    ("hyphenated", "Việt-Nam và VNU-HCM là tên riêng"),
    ("long_sentence", (
        "Trường Đại học Khoa học Tự nhiên trực thuộc Đại học Quốc gia Thành phố Hồ Chí Minh "
        "là một trong những cơ sở đào tạo và nghiên cứu khoa học công nghệ hàng đầu của cả nước."
    )),
)


# ---------------------------------------------------------------------------
# Provenance (same mechanism as audit 010)
# ---------------------------------------------------------------------------
def is_full_commit_sha(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == FULL_SHA_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def extract_snapshot_revision(path: str) -> str | None:
    if not isinstance(path, str):
        return None
    match = _SNAPSHOT_PATTERN.search(path)
    return match.group(1) if match else None


def observe_tokenizer_revision(tokenizer) -> tuple[str | None, tuple[str, ...], str]:
    """Read the resolved commit back off the loaded tokenizer (audit 010)."""
    import os as _os

    candidates: list[str] = []

    def consider(value: Any) -> None:
        if isinstance(value, str) and value and _os.sep in value and value not in candidates:
            candidates.append(value)

    for attribute in ("vocab_file", "merges_file", "tokenizer_file", "name_or_path"):
        consider(getattr(tokenizer, attribute, None))
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for value in init_kwargs.values():
            consider(value)

    found: dict[str, list[str]] = {}
    for path in candidates:
        revision = extract_snapshot_revision(path)
        if revision:
            found.setdefault(revision, []).append(path)
    if not found:
        return None, tuple(candidates[:5]), "no Hugging Face snapshot path among the resolved files"
    if len(found) > 1:
        return None, tuple(candidates[:5]), f"resolved files disagree: {sorted(found)}"
    revision, evidence = next(iter(found.items()))
    return revision, tuple(evidence[:5]), "hugging face cache snapshot path of the loaded tokenizer files"


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def tokenize_span(tokenizer, text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Tokenize one span alone, with no special tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tuple(tokenizer.convert_ids_to_tokens(ids)), tuple(ids)


def probe_inventory(tokenizer, inventory, unk: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align every unique stripped form in the B3A inventory."""
    alignments = []
    failures: list[dict[str, Any]] = []
    for form in sorted(inventory.forms):
        tokens, ids = tokenize_span(tokenizer, form)
        alignment = align_span(
            form, tokens, ids, eligibility=Eligibility.VIETNAMESE_CANDIDATE, unk_token=unk
        )
        alignments.append(alignment)
        if not alignment.aligned:
            failures.append(alignment.to_dict())
    summary = summarize_alignments(alignments)
    summary["total_unique_stripped_forms"] = len(inventory.forms)
    summary["tokenizable_forms"] = summary["aligned"]
    return summary, failures


def probe_sentence(tokenizer, case_id: str, text: str, classifier, unk: str | None) -> dict[str, Any]:
    """Align every span of one sentence and reconcile with the full sequence."""
    base_text = decompose(canon(text), eligibility_classifier=classifier).base_text
    parts = decompose(base_text, eligibility_classifier=classifier)

    # The authoritative tokenization of the whole tokenizer input.
    full_ids = tokenizer.encode(base_text, add_special_tokens=False)
    full_tokens = tuple(tokenizer.convert_ids_to_tokens(full_ids))

    # Compose from EVERY region of the input, not just the Vietnamese spans:
    # punctuation and non-candidate text are part of the input and must be
    # accounted for, even though they carry no orthography channels.
    regions: list[tuple[str, Eligibility]] = []
    cursor = 0
    for span in parts.syllables:
        if span.base_start > cursor:
            regions.append((base_text[cursor : span.base_start], Eligibility.NOT_APPLICABLE))
        regions.append((span.base_text, span.eligibility))
        cursor = span.base_end
    if cursor < len(base_text):
        regions.append((base_text[cursor:], Eligibility.NOT_APPLICABLE))

    alignments = []
    composed: list[str] = []
    for region_text, eligibility in regions:
        if not region_text:
            continue
        tokens, ids = tokenize_span(tokenizer, region_text)
        composed.extend(tokens)
        alignments.append(
            align_span(region_text, tokens, ids, eligibility=eligibility, unk_token=unk)
        )

    comparison = compare_sequences(full_tokens, tuple(composed), getattr(tokenizer, "all_special_tokens", ()) or ())
    return {
        "case_id": case_id,
        "text": text,
        "base_text": base_text,
        "full_sequence_tokens": list(full_tokens),
        "composed_tokens": composed,
        "sequence_consistent": comparison["consistent"],
        "unexplained_tokens": comparison["unexplained_tokens"],
        "sequence_detail": comparison["detail"],
        "spans": [a.to_dict() for a in alignments],
        "eligibility_counts": _count_eligibility(alignments),
        **{f"summary_{k}": v for k, v in summarize_alignments(alignments).items()},
    }


def _count_eligibility(alignments: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for alignment in alignments:
        key = alignment.eligibility.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def probe_fast_tokenizer_diagnostic(checkpoint: str, revision: str, slow_tokenizer) -> dict[str, Any]:
    """OPTIONAL: compare the backend tokenizer's ids against the slow one's.

    The slow tokenizer stays authoritative regardless of the outcome. This exists
    only to record whether the shipped `tokenizer.json` agrees, and never to
    borrow its offsets as authority.
    """
    try:
        from transformers import AutoTokenizer

        fast = AutoTokenizer.from_pretrained(checkpoint, revision=revision, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if not getattr(fast, "is_fast", False):
        return {"available": False, "reason": "no fast tokenizer backend for this checkpoint"}

    mismatches = []
    for _case_id, text in SENTENCES:
        base = decompose(canon(text)).base_text
        slow_ids = slow_tokenizer.encode(base, add_special_tokens=False)
        fast_ids = fast.encode(base, add_special_tokens=False)
        if slow_ids != fast_ids:
            mismatches.append({"text": base, "slow": slow_ids, "fast": fast_ids})
    return {
        "available": True,
        "authoritative": "slow",
        "ids_identical": not mismatches,
        "mismatches": mismatches[:5],
        "note": (
            "Diagnostic only. The slow PhobertTokenizer remains the token-id authority; "
            "its offsets are not adopted even when present."
        ),
    }


def render_report(config: dict[str, Any], summary: dict[str, Any], sentences: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# B3B-1 manual alignment probe")
    a("")
    a(f"Run id: `{config['run_id']}`  ")
    a(f"Timestamp (UTC): `{config['timestamp_utc']}`")
    a("")
    a("> Tests whether fastBPE `@@` reconstruction yields deterministic character ranges")
    a("> for the slow PhoBERT tokenizer. **The hypothesis is not declared validated by")
    a("> this script** -- read the numbers below. No model weights were loaded.")
    a("")
    a("## Provenance")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key in ("checkpoint", "revision_requested", "revision_observed", "revision_verified", "tokenizer_class", "is_fast"):
        a(f"| `{key}` | {config['tokenizer'].get(key)} |")
    a(f"| `model_weights_loaded` | {config['model_weights_loaded']} |")
    a(f"| `eligibility_resolved` | {config['eligibility_resolved']} |")
    a("")
    a("## B3A inventory coverage")
    a("")
    inv = summary["inventory"]
    a("| Metric | Value |")
    a("|---|---:|")
    for key in (
        "total_unique_stripped_forms", "tokenizable_forms", "aligned", "failed",
        "spans_with_unknown_token", "surface_reconstruction_failures",
        "mean_subwords_per_span", "max_subwords_per_span",
    ):
        a(f"| {key} | {inv.get(key)} |")
    a("")
    if inv.get("failure_reasons"):
        a("Failure reasons: " + ", ".join(f"`{k}` × {v}" for k, v in inv["failure_reasons"].items()))
        a("")
    a("## Full-sequence consistency")
    a("")
    a("| Case | spans | aligned | channels | sequence consistent | unexplained |")
    a("|---|---:|---:|---:|---|---:|")
    for row in sentences:
        a(
            f"| `{row['case_id']}` | {row['summary_total']} | {row['summary_aligned']} | "
            f"{sum(1 for s in row['spans'] if s['carries_channels'])} | "
            f"{row['sequence_consistent']} | {len(row['unexplained_tokens'])} |"
        )
    a("")
    a("Composition covers **every region** of the tokenizer input, including punctuation and")
    a("non-candidate spans; orthography channels are attached only to eligible Vietnamese")
    a("spans. An inconsistent row means per-span tokenization does not reproduce the")
    a("authoritative sequence, which would sink the manual-alignment strategy.")
    a("")
    a("## Eligibility resolution")
    a("")
    a("| Case | " + " | ".join(e.value for e in Eligibility) + " |")
    a("|---|" + "---|" * len(Eligibility))
    for row in sentences:
        counts = row["eligibility_counts"]
        a(f"| `{row['case_id']}` | " + " | ".join(str(counts.get(e.value, 0)) for e in Eligibility) + " |")
    a("")
    a("`UNDECIDED` must be zero on a resolved run: it means the B3A inventory was not")
    a("consulted, which is the defect audit 011 repaired.")
    a("")
    if config.get("fast_tokenizer_diagnostic"):
        a("## Optional fast-tokenizer diagnostic")
        a("")
        a(f"```json\n{json.dumps(config['fast_tokenizer_diagnostic'], ensure_ascii=False, indent=2)}\n```")
        a("")
    a("## Status")
    a("")
    a(f"**`{config['status']}`**")
    a("")
    a("Manual alignment is validated only if the inventory shows zero surface-reconstruction")
    a("failures and every sentence is sequence-consistent. Read both before relying on it.")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B3B-1 manual alignment probe (Colab, tokenizer only).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=B3B0_PROBE_REVISION, help="full 40-char commit SHA")
    parser.add_argument("--output-root", default="results/b3b1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fast-diagnostic", action="store_true", help="optional tokenizer.json id comparison")
    parser.add_argument("--max-inventory-forms", type=int, default=None, help="cap for a quick run")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    if not is_full_commit_sha(args.revision):
        print(
            f"--revision {args.revision!r} is not a full 40-character lowercase commit SHA.",
            file=sys.stderr,
        )
        return 2

    try:
        import transformers  # noqa: F401
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "transformers is not installed. This probe is for Colab, not the ML-free local\n"
            ".venv. In Colab:\n\n"
            '    pip install "transformers==4.57.6"\n'
            f'    export HF_HOME="$PWD/{REPO_LOCAL_HF_CACHE}"\n'
            "    python scripts/fetch_vietnamese_syllable_inventory.py\n"
            f"    python scripts/b3b1_phobert_alignment_probe.py --revision {args.revision}\n",
            file=sys.stderr,
        )
        return 2

    # The B3A inventory is mandatory: without it every span would be UNDECIDED,
    # which is exactly the defect this probe must not reproduce.
    inventory = load_inventory()
    classifier = make_classifier(inventory)

    print(f"Loading slow tokenizer only (authoritative): {args.checkpoint}@{args.revision[:12]}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision, use_fast=False)
    observed, evidence, source = observe_tokenizer_revision(tokenizer)
    verified = observed is not None and observed == args.revision
    if observed is not None and observed != args.revision:
        print(
            f"REFUSING: tokenizer resolved to {observed}, not {args.revision}", file=sys.stderr
        )
        return 3
    unk = getattr(tokenizer, "unk_token", None)
    print(f"  class={type(tokenizer).__name__} fast={getattr(tokenizer, 'is_fast', False)} verified={verified}")

    inventory_summary, inventory_failures = probe_inventory(tokenizer, inventory, unk)
    curated = []
    for group, items in (("tone", TONE_CASES), ("letter", LETTER_CASES), ("case", CASE_CASES)):
        for text in items:
            for form, label in ((text, "NFC"), (unicodedata.normalize("NFD", text), "NFD")):
                base = decompose(canon(form), eligibility_classifier=classifier).base_text
                tokens, ids = tokenize_span(tokenizer, base)
                alignment = align_span(
                    base, tokens, ids,
                    eligibility=classifier(base), unk_token=unk,
                )
                curated.append({"group": group, "source": text, "form": label, **alignment.to_dict()})

    sentences = [probe_sentence(tokenizer, cid, text, classifier, unk) for cid, text in SENTENCES]

    fast_diagnostic = (
        probe_fast_tokenizer_diagnostic(args.checkpoint, args.revision, tokenizer)
        if args.fast_diagnostic
        else None
    )

    all_consistent = all(row["sequence_consistent"] for row in sentences)
    inventory_clean = inventory_summary["failed"] == 0
    status = STATUS_OK if (all_consistent and inventory_clean) else STATUS_FAIL

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_contract = TokenizerContract(
        checkpoint=args.checkpoint,
        revision_requested=args.revision,
        revision_observed=observed,
        revision_verified=verified,
        revision_evidence=evidence,
        revision_evidence_source=source,
        tokenizer_class=type(tokenizer).__name__,
        is_fast=bool(getattr(tokenizer, "is_fast", False)),
        vocab_size=getattr(tokenizer, "vocab_size", None),
        unk_token=unk,
        transformers_version=transformers.__version__,
    )
    config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_contract.to_dict(),
        "model_weights_loaded": False,
        "eligibility_resolved": True,
        "inventory": inventory.summary(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "status": status,
        "fast_tokenizer_diagnostic": fast_diagnostic,
        "note": (
            "Manual-alignment hypothesis test. The slow PhobertTokenizer is the token-id "
            "authority; no offsets are borrowed from any other implementation. No model "
            "weights were loaded and nothing was trained."
        ),
    }
    summary = {
        "inventory": inventory_summary,
        "sentences_total": len(sentences),
        "sentences_sequence_consistent": sum(1 for r in sentences if r["sequence_consistent"]),
        "curated_total": len(curated),
        "curated_aligned": sum(1 for c in curated if c["status"] == SpanAlignmentStatus.ALIGNED.value),
        "undecided_spans": sum(
            sum(1 for s in r["spans"] if s["eligibility"] == Eligibility.UNDECIDED.value)
            for r in sentences
        ),
        "status": status,
    }

    for name, payload in (("config.json", config), ("summary.json", summary)):
        with (run_dir / name).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
    with (run_dir / "inventory_failures.jsonl").open("w", encoding="utf-8") as fh:
        for row in inventory_failures:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for row in curated:
            fh.write(json.dumps({"kind": "curated", **row}, ensure_ascii=False) + "\n")
        for row in sentences:
            fh.write(json.dumps({"kind": "sentence", **row}, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(config, summary, sentences), encoding="utf-8")

    print()
    print(f"  inventory forms      : {inventory_summary['total_unique_stripped_forms']}")
    print(f"  aligned              : {inventory_summary['aligned']}")
    print(f"  failed               : {inventory_summary['failed']} {inventory_summary['failure_reasons'] or ''}")
    print(f"  mean subwords/form   : {inventory_summary['mean_subwords_per_span']}")
    print(f"  max subwords/form    : {inventory_summary['max_subwords_per_span']}")
    print(f"  sentences consistent : {summary['sentences_sequence_consistent']}/{summary['sentences_total']}")
    print(f"  UNDECIDED spans      : {summary['undecided_spans']} (must be 0)")
    print()
    print(f"Status: {status}")
    print("Manual alignment is validated only by these numbers, not by this script running.")
    print()
    print(f"Results: {run_dir}")
    return 0 if status == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
