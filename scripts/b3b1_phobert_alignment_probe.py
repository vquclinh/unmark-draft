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
    AlignmentStatusB,
    OrthographicRegion,
    align_chunk,
    compose,
    overlay_orthography,
    summarize_chunk_alignments,
    verify_token_grid,
    whitespace_chunks,
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
def raw_tokenize(tokenizer, text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """RAW BPE pieces and their ids, with no special tokens.

    Uses `tokenizer.tokenize()` rather than `convert_ids_to_tokens(encode(...))`:
    the id round trip replaces an out-of-vocabulary surface with `<unk>` and the
    characters become unrecoverable. That conflation is what made B3B-1A report
    `khut` as an alignment failure when its raw surface was exactly recoverable.
    """
    tokens = tuple(tokenizer.tokenize(text))
    ids = tuple(tokenizer.convert_tokens_to_ids(list(tokens)))
    return tokens, ids


def probe_inventory(tokenizer, inventory, unk_id: int | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Align every unique stripped form in the B3A inventory.

    Each form is a single whitespace chunk, so it is aligned as one. An unknown
    vocabulary id is REPORTED, never counted as a surface failure.
    """
    alignments = []
    failures: list[dict[str, Any]] = []
    unknown_id_forms: list[str] = []
    for form in sorted(inventory.forms):
        chunks = whitespace_chunks(form)
        if not chunks:
            continue
        tokens, ids = raw_tokenize(tokenizer, form)
        alignment = align_chunk(chunks[0], tokens, ids, unk_token_id=unk_id)
        alignments.append(alignment)
        if not alignment.aligned:
            failures.append({"form": form, **alignment.to_dict()})
        elif alignment.unknown_id_count:
            unknown_id_forms.append(form)

    summary = summarize_chunk_alignments(alignments)
    summary["total_unique_stripped_forms"] = len(inventory.forms)
    summary["raw_surface_reconstructable_forms"] = summary["aligned"]
    summary["forms_with_unknown_token_id"] = len(unknown_id_forms)
    summary["raw_surface_reconstruction_failures"] = summary["surface_reconstruction_failures"]
    summary["mean_subwords_per_form"] = summary["mean_subwords_per_chunk"]
    summary["max_subwords_per_form"] = summary["max_subwords_per_chunk"]
    return summary, failures, unknown_id_forms


def probe_sentence(tokenizer, case_id: str, text: str, classifier, unk_id: int | None) -> dict[str, Any]:
    """Align one sentence over whitespace chunks and verify the token grid.

    The authoritative grid is `T(b(x))` on the whole base text. Chunk alignment
    only reconstructs a character map beside it, and must reproduce it exactly.
    """
    base_text = decompose(canon(text), eligibility_classifier=classifier).base_text
    parts = decompose(base_text, eligibility_classifier=classifier)

    authoritative_tokens, authoritative_ids = raw_tokenize(tokenizer, base_text)

    chunks = whitespace_chunks(base_text)
    alignments = [
        align_chunk(chunk, *raw_tokenize(tokenizer, chunk.text), unk_token_id=unk_id)
        for chunk in chunks
    ]
    composed_tokens, composed_ids = compose(alignments)
    grid = verify_token_grid(composed_tokens, composed_ids, authoritative_tokens, authoritative_ids)

    # Orthographic regions: syllable spans plus the gaps between them, so every
    # character of the base text belongs to exactly one region.
    regions: list[OrthographicRegion] = []
    cursor = 0
    for span in parts.syllables:
        if span.base_start > cursor:
            regions.append(
                OrthographicRegion(
                    len(regions), base_text[cursor : span.base_start], cursor, span.base_start,
                    Eligibility.NOT_APPLICABLE, is_syllable=False,
                )
            )
        regions.append(
            OrthographicRegion(
                len(regions), span.base_text, span.base_start, span.base_end, span.eligibility
            )
        )
        cursor = span.base_end
    if cursor < len(base_text):
        regions.append(
            OrthographicRegion(
                len(regions), base_text[cursor:], cursor, len(base_text),
                Eligibility.NOT_APPLICABLE, is_syllable=False,
            )
        )

    overlays = []
    for alignment in alignments:
        overlays.extend(o.to_dict() for o in overlay_orthography(alignment.pieces, regions))

    mixed = [o for o in overlays if o["is_mixed"]]
    return {
        "case_id": case_id,
        "text": text,
        "base_text": base_text,
        "chunks": [c.to_dict() for c in chunks],
        "alignments": [a.to_dict() for a in alignments],
        "authoritative_tokens": list(authoritative_tokens),
        "authoritative_ids": list(authoritative_ids),
        "composed_tokens": list(composed_tokens),
        "composed_ids": list(composed_ids),
        "tokens_match": grid["tokens_match"],
        "ids_match": grid["ids_match"],
        "sequence_consistent": grid["consistent"],
        "unexplained_tokens": grid["unexplained_tokens"],
        "grid_detail": grid["detail"],
        "regions": [r.to_dict() for r in regions],
        "overlays": overlays,
        "mixed_pieces": len(mixed),
        "eligibility_counts": _count_eligibility_regions(regions),
        **{f"summary_{k}": v for k, v in summarize_chunk_alignments(alignments).items()},
    }


def _count_eligibility_regions(regions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        key = region.eligibility.value
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
    inv = summary["inventory"]
    a("# B3B-1 manual alignment probe (whitespace-chunk contract)")
    a("")
    a(f"Run id: `{config['run_id']}`  ")
    a(f"Timestamp (UTC): `{config['timestamp_utc']}`")
    a("")
    a("> The authoritative token grid is `T(b(x))` from the pinned slow tokenizer. This")
    a("> probe only reconstructs a character map beside it, over the same maximal")
    a("> non-whitespace chunks the tokenizer uses, and verifies the two agree exactly.")
    a("> No model weights were loaded.")
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
    a("| Metric | Value |")
    a("|---|---:|")
    for key in (
        "total_unique_stripped_forms", "raw_surface_reconstructable_forms",
        "forms_with_unknown_token_id", "raw_surface_reconstruction_failures",
        "range_failures", "mean_subwords_per_form", "max_subwords_per_form",
    ):
        a(f"| {key} | {inv.get(key)} |")
    a("")
    a("`forms_with_unknown_token_id` counts forms whose raw BPE surface reconstructs exactly")
    a("but whose vocabulary id is the unknown id. That is **reported, not a failure**: the")
    a("characters are recoverable, only the vocabulary lookup is not.")
    if summary.get("forms_with_unknown_token_id_list"):
        a("")
        a("Forms: " + ", ".join(f"`{f}`" for f in summary["forms_with_unknown_token_id_list"][:50]))
    a("")
    a("## Full-sequence token-grid agreement")
    a("")
    a("| Case | chunks | aligned | tokens match | ids match | unexplained | mixed pieces |")
    a("|---|---:|---:|---|---|---:|---:|")
    for row in sentences:
        a(
            f"| `{row['case_id']}` | {row['summary_total_chunks']} | {row['summary_aligned']} | "
            f"{row['tokens_match']} | {row['ids_match']} | {len(row['unexplained_tokens'])} | "
            f"{row['mixed_pieces']} |"
        )
    a("")
    a(f"Tokens: {summary['sentences_tokens_match']}/{summary['sentences_total']} · ")
    a(f"IDs: {summary['sentences_ids_match']}/{summary['sentences_total']} · ")
    a(f"chunks aligned {summary['chunks_aligned']}/{summary['total_chunks']}")
    a("")
    a("A row that does not match means chunk composition failed to reproduce the")
    a("authoritative grid, which invalidates the alignment strategy.")
    a("")
    a("## Orthographic overlay")
    a("")
    a("| Case | " + " | ".join(e.value for e in Eligibility) + " | mixed pieces |")
    a("|---|" + "---|" * (len(Eligibility) + 1))
    for row in sentences:
        counts = row["eligibility_counts"]
        a(
            f"| `{row['case_id']}` | "
            + " | ".join(str(counts.get(e.value, 0)) for e in Eligibility)
            + f" | {row['mixed_pieces']} |"
        )
    a("")
    a("`UNDECIDED` must be zero on a resolved run. `mixed pieces` are BPE pieces drawing")
    a("from a Vietnamese candidate **and** something else; they are recorded with their")
    a("contributors and never claimed as Vietnamese -- see `docs/spec/decisions.md`")
    a("D-B3B1B-002, which is OPEN.")
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
    a("Validated only when: provenance verified, eligibility resolved, every inventory form")
    a("raw-surface reconstructable with exact ranges, every chunk reconstructed, chunk")
    a("composition equal to the authoritative tokens **and** ids, and no unexplained token.")
    a("Unknown vocabulary ids alone never invalidate the result.")
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

    unk_id = getattr(tokenizer, "unk_token_id", None)
    inventory_summary, inventory_failures, unknown_id_forms = probe_inventory(
        tokenizer, inventory, unk_id
    )

    curated = []
    for group, items in (("tone", TONE_CASES), ("letter", LETTER_CASES), ("case", CASE_CASES)):
        for text in items:
            for source_form, label in ((text, "NFC"), (unicodedata.normalize("NFD", text), "NFD")):
                base = decompose(canon(source_form), eligibility_classifier=classifier).base_text
                chunks = whitespace_chunks(base)
                if not chunks:
                    continue
                tokens, ids = raw_tokenize(tokenizer, base)
                alignment = align_chunk(chunks[0], tokens, ids, unk_token_id=unk_id)
                curated.append({
                    "group": group, "source": text, "normalisation": label,
                    "base": base, "eligibility": classifier(base).value,
                    **alignment.to_dict(),
                })

    sentences = [probe_sentence(tokenizer, cid, text, classifier, unk_id) for cid, text in SENTENCES]

    fast_diagnostic = (
        probe_fast_tokenizer_diagnostic(args.checkpoint, args.revision, tokenizer)
        if args.fast_diagnostic
        else None
    )

    # Validation criteria (B3B-1B). An unknown vocabulary id is reported, never
    # a reason to invalidate: only surface/range/token-grid problems are.
    all_tokens_match = all(row["tokens_match"] for row in sentences)
    all_ids_match = all(row["ids_match"] for row in sentences)
    inventory_clean = inventory_summary["failed"] == 0
    no_unexplained = all(not row["unexplained_tokens"] for row in sentences)
    status = (
        STATUS_OK
        if (all_tokens_match and all_ids_match and inventory_clean and no_unexplained)
        else STATUS_FAIL
    )

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
        "forms_with_unknown_token_id_list": unknown_id_forms,
        "sentences_total": len(sentences),
        "sentences_tokens_match": sum(1 for r in sentences if r["tokens_match"]),
        "sentences_ids_match": sum(1 for r in sentences if r["ids_match"]),
        "sentences_sequence_consistent": sum(1 for r in sentences if r["sequence_consistent"]),
        "total_chunks": sum(r["summary_total_chunks"] for r in sentences),
        "chunks_aligned": sum(r["summary_aligned"] for r in sentences),
        "chunk_surface_failures": sum(r["summary_surface_reconstruction_failures"] for r in sentences),
        "mixed_pieces": sum(r["mixed_pieces"] for r in sentences),
        "curated_total": len(curated),
        "curated_aligned": sum(1 for c in curated if c["status"] == AlignmentStatusB.ALIGNED.value),
        "undecided_regions": sum(
            r["eligibility_counts"].get(Eligibility.UNDECIDED.value, 0) for r in sentences
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
    with (run_dir / "unknown_token_id_forms.jsonl").open("w", encoding="utf-8") as fh:
        for form in unknown_id_forms:
            fh.write(json.dumps({"form": form}, ensure_ascii=False) + "\n")
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for row in curated:
            fh.write(json.dumps({"kind": "curated", **row}, ensure_ascii=False) + "\n")
        for row in sentences:
            fh.write(json.dumps({"kind": "sentence", **row}, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(config, summary, sentences), encoding="utf-8")

    print()
    print(f"  inventory forms          : {inventory_summary['total_unique_stripped_forms']}")
    print(f"  raw-surface reconstructable: {inventory_summary['raw_surface_reconstructable_forms']}")
    print(f"  surface failures         : {inventory_summary['raw_surface_reconstruction_failures']}")
    print(f"  range failures           : {inventory_summary['range_failures']}")
    print(f"  forms with <unk> id      : {inventory_summary['forms_with_unknown_token_id']} (reported, not a failure)")
    print(f"  mean subwords/form       : {inventory_summary['mean_subwords_per_form']}")
    print(f"  max subwords/form        : {inventory_summary['max_subwords_per_form']}")
    print(f"  chunks aligned           : {summary['chunks_aligned']}/{summary['total_chunks']}")
    print(f"  sentences tokens match   : {summary['sentences_tokens_match']}/{summary['sentences_total']}")
    print(f"  sentences ids match      : {summary['sentences_ids_match']}/{summary['sentences_total']}")
    print(f"  mixed-contributor pieces : {summary['mixed_pieces']} (recorded, see D-B3B1B-002)")
    print(f"  UNDECIDED regions        : {summary['undecided_regions']} (must be 0)")
    print()
    print(f"Status: {status}")
    print("Manual alignment is validated only by these numbers, not by this script running.")
    print()
    print(f"Results: {run_dir}")
    return 0 if status == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
