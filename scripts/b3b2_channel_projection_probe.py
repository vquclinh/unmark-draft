#!/usr/bin/env python3
"""B3B-2 — orthographic channel projection probe (Colab, tokenizer only).

Projects the tone and letter-diacritic channels onto the real PhoBERT token grid
under every B2 corruption condition, and reports what the adapter would receive.

**Do not run this locally.** The local `.venv` is deliberately ML-free; this
script needs `transformers` and the pinned tokenizer, which live on Colab. It
loads the **tokenizer only** — no model weights, no training, no adapter.

What it checks, per sentence and per condition:

1. the token grid over `b(x)` is identical across all six conditions -- one grid
   serves every condition, because corruption never changes base letters;
2. every piece's character range is identical across conditions;
3. tone-channel coverage degrades monotonically FULL -> STRIP_ALL, and
   STRIP_ALL leaves `UNMARKED` on Vietnamese syllables rather than `NA`;
4. how often a piece draws on two or more distinct Vietnamese candidates
   (the corrected B3B-1 run saw **zero**).

Colab::

    pip install "transformers==4.57.6"
    export HF_HOME="$PWD/.hf-cache"
    python scripts/fetch_vietnamese_syllable_inventory.py
    python scripts/b3b2_channel_projection_probe.py --revision <40-char-sha>
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

from unmark.alignment import (  # noqa: E402
    LETTER_POOLING_RULE,
    OrthographicRegion,
    TokenToneLabel,
    ToneOwnership,
    align_chunk,
    character_letter_labels,
    compose,
    overlay_orthography,
    project_piece,
    summarize_projections,
    verify_token_grid,
    whitespace_chunks,
)
from unmark.alignment.contracts import REPO_LOCAL_HF_CACHE  # noqa: E402
from unmark.corruption import CorruptionPurpose, corrupt  # noqa: E402
from unmark.linguistics import load_inventory, make_classifier  # noqa: E402
from unmark.orthography import Eligibility, canon, decompose  # noqa: E402

DEFAULT_CHECKPOINT = "vinai/phobert-base"
CONDITIONS = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")

CASES: tuple[tuple[str, str], ...] = (
    ("plain", "Tôi đang học nghiên cứu tại Đại học Quốc gia"),
    ("tones", "má mà mả mã mạ ma"),
    ("letters", "đường phố cũ ăn ơn ưu"),
    ("mixed_en", "tôi dùng Python và PyTorch để train model"),
    ("punct", "Nghiên cứu, thí nghiệm; kết quả: tốt!"),
    ("url", "xem tại https://tuyensinh.vnu.edu.vn/ nhé"),
    ("digits", "năm 2026 có 12 tháng"),
)


def is_full_commit_sha(value: str | None) -> bool:
    return bool(value) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def build_regions(base_text: str, parts) -> list[OrthographicRegion]:
    """Syllable spans plus the gaps between them: every character belongs to
    exactly one region."""
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
    return regions


def raw_tokenize(tokenizer, text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Raw pieces and their ids. Never `convert_ids_to_tokens`: that round trip
    destroys the surface of any OOV piece (the `khut` finding)."""
    tokens = tuple(tokenizer.tokenize(text))
    return tokens, tuple(tokenizer.convert_tokens_to_ids(list(tokens)))


def project_condition(tokenizer, text: str, classifier, unk_id: int | None) -> dict[str, Any]:
    """Align and project one already-corrupted string."""
    # ONE canonical decomposition. Its unit and syllable offsets are already
    # indexed into `base_text`, so re-decomposing the stripped base would only
    # throw away every mark -- the channels would come back empty.
    parts = decompose(canon(text), eligibility_classifier=classifier)
    base_text = parts.base_text
    labels = character_letter_labels(parts)
    regions = build_regions(base_text, parts)
    tones = {
        region.index: span.observed_tone
        for region in regions
        if region.is_syllable
        for span in parts.syllables
        if span.base_start == region.start
    }

    authoritative_tokens, authoritative_ids = raw_tokenize(tokenizer, base_text)
    alignments = [
        align_chunk(chunk, *raw_tokenize(tokenizer, chunk.text), unk_token_id=unk_id)
        for chunk in whitespace_chunks(base_text)
    ]
    composed_tokens, composed_ids = compose(alignments)
    grid = verify_token_grid(composed_tokens, composed_ids, authoritative_tokens, authoritative_ids)

    projections = []
    for alignment in alignments:
        overlays = overlay_orthography(alignment.pieces, regions)
        for piece, overlay in zip(alignment.pieces, overlays):
            projections.append(
                project_piece(
                    len(projections), piece, overlay, base_text, labels, regions, tones
                )
            )

    summary = summarize_projections(projections)
    return {
        "text": text,
        "base_text": base_text,
        "tokens": list(composed_tokens),
        "ids": list(composed_ids),
        "sequence_consistent": grid["consistent"],
        "piece_ranges": [[p.global_start, p.global_end] for p in projections],
        "tone_labels": [p.tone.label.value for p in projections],
        "letter_labels": [
            [c.letter_diacritic.value for c in p.letter.applicable] for p in projections
        ],
        "marked_tone_tokens": sum(
            1
            for p in projections
            if p.tone.label not in {TokenToneLabel.NA, TokenToneLabel.UNMARKED}
        ),
        "unmarked_tone_tokens": sum(
            1 for p in projections if p.tone.label is TokenToneLabel.UNMARKED
        ),
        "multi_candidate_tokens": sum(
            1
            for p in projections
            if p.tone.ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS
        ),
        "projections": [p.to_dict() for p in projections],
        **{f"summary_{k}": v for k, v in summary.items()},
    }


def probe_case(tokenizer, case_id: str, text: str, classifier, unk_id: int | None, seed: int) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        corrupted = corrupt(
            text, condition, seed=seed, sample_id=case_id, purpose=CorruptionPurpose.SELF_CHECK
        )
        by_condition[condition] = project_condition(
            tokenizer, corrupted.corrupted_text, classifier, unk_id
        )

    reference = by_condition["FULL"]
    grid_invariant = all(
        by_condition[c]["tokens"] == reference["tokens"] for c in CONDITIONS
    )
    ranges_invariant = all(
        by_condition[c]["piece_ranges"] == reference["piece_ranges"] for c in CONDITIONS
    )
    marked = [by_condition[c]["marked_tone_tokens"] for c in CONDITIONS]
    return {
        "case_id": case_id,
        "text": text,
        "grid_invariant_across_conditions": grid_invariant,
        "ranges_invariant_across_conditions": ranges_invariant,
        "monotonic_tone_degradation": marked == sorted(marked, reverse=True),
        "strip_all_has_no_marked_tone": by_condition["STRIP_ALL"]["marked_tone_tokens"] == 0,
        "strip_all_keeps_unmarked": by_condition["STRIP_ALL"]["unmarked_tone_tokens"] > 0,
        "marked_tone_tokens_by_condition": dict(zip(CONDITIONS, marked)),
        "multi_candidate_tokens": sum(
            by_condition[c]["multi_candidate_tokens"] for c in CONDITIONS
        ),
        "sequence_consistent": all(by_condition[c]["sequence_consistent"] for c in CONDITIONS),
        "conditions": by_condition,
    }


def render_report(config: dict[str, Any], summary: dict[str, Any], cases: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# B3B-2 — channel projection probe")
    a("")
    a(f"Run `{config['run_id']}` — {config['checkpoint']}@{config['revision'][:12]}")
    a("")
    a("Tokenizer only. No model weights were loaded; nothing was trained.")
    a("")
    a("## Summary")
    a("")
    a("| Measure | Result |")
    a("|---|---|")
    a(f"| Cases | {summary['cases']} |")
    a(f"| Token grid invariant across all six conditions | {summary['grid_invariant']}/{summary['cases']} |")
    a(f"| Piece ranges invariant across all six conditions | {summary['ranges_invariant']}/{summary['cases']} |")
    a(f"| Monotonic tone degradation FULL→STRIP_ALL | {summary['monotonic']}/{summary['cases']} |")
    a(f"| STRIP_ALL leaves no marked tone | {summary['strip_all_clean']}/{summary['cases']} |")
    a(f"| Sequence consistent under every condition | {summary['sequence_consistent']}/{summary['cases']} |")
    a(f"| Pieces spanning ≥2 distinct candidates | **{summary['multi_candidate_tokens']}** |")
    a("")
    a("## Per case")
    a("")
    a("| Case | grid inv. | ranges inv. | monotonic | marked tones by condition | multi-cand |")
    a("|---|---|---|---|---|---|")
    for row in cases:
        marked = " / ".join(str(row["marked_tone_tokens_by_condition"][c]) for c in CONDITIONS)
        a(
            f"| `{row['case_id']}` | {row['grid_invariant_across_conditions']} |"
            f" {row['ranges_invariant_across_conditions']} |"
            f" {row['monotonic_tone_degradation']} | {marked} | {row['multi_candidate_tokens']} |"
        )
    a("")
    a(f"Conditions, in column order: {' / '.join(CONDITIONS)}.")
    a("")
    a("## Reading this")
    a("")
    a("`b(x)` is invariant under corruption, so a single token grid and a single set")
    a("of character ranges serve every condition; only the channel **values** change.")
    a("A piece drawing on exactly one Vietnamese candidate keeps that candidate's tone")
    a("even alongside punctuation; a piece drawing on two or more gets `NA` with every")
    a("contributor recorded, and is never resolved by length, position or averaging.")
    a("See `docs/spec/decisions.md` D-B3B1C-001.")
    a("")
    a("Letter pooling rule, recorded for the adapter and **not** performed here:")
    a("")
    a(f"> {LETTER_POOLING_RULE}")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B3B-2 channel projection probe (Colab, tokenizer only).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", required=True, help="full 40-char commit SHA")
    parser.add_argument("--output-root", default="results/b3b2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    # Resolve before anything can change the working directory.
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    if not is_full_commit_sha(args.revision):
        print(f"--revision {args.revision!r} is not a full 40-character lowercase commit SHA.", file=sys.stderr)
        return 2

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "transformers is not installed. This probe is for Colab, not the ML-free local\n"
            ".venv. In Colab:\n\n"
            '    pip install "transformers==4.57.6"\n'
            f'    export HF_HOME="$PWD/{REPO_LOCAL_HF_CACHE}"\n'
            "    python scripts/fetch_vietnamese_syllable_inventory.py\n"
            f"    python scripts/b3b2_channel_projection_probe.py --revision {args.revision}\n",
            file=sys.stderr,
        )
        return 2

    inventory = load_inventory()
    classifier = make_classifier(inventory)

    print(f"Loading slow tokenizer only: {args.checkpoint}@{args.revision[:12]}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision, use_fast=False)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    cases = [probe_case(tokenizer, cid, text, classifier, unk_id, args.seed) for cid, text in CASES]

    summary = {
        "cases": len(cases),
        "grid_invariant": sum(1 for c in cases if c["grid_invariant_across_conditions"]),
        "ranges_invariant": sum(1 for c in cases if c["ranges_invariant_across_conditions"]),
        "monotonic": sum(1 for c in cases if c["monotonic_tone_degradation"]),
        "strip_all_clean": sum(1 for c in cases if c["strip_all_has_no_marked_tone"]),
        "sequence_consistent": sum(1 for c in cases if c["sequence_consistent"]),
        "multi_candidate_tokens": sum(c["multi_candidate_tokens"] for c in cases),
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "model_weights_loaded": False,
        "letter_pooling_rule": LETTER_POOLING_RULE,
        "python": platform.python_version(),
        "output_root": str(output_root),
    }

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "projections.json").write_text(
        json.dumps({"config": config, "summary": summary, "cases": cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(render_report(config, summary, cases), encoding="utf-8")

    ok = (
        summary["grid_invariant"] == summary["cases"]
        and summary["ranges_invariant"] == summary["cases"]
        and summary["monotonic"] == summary["cases"]
        and summary["strip_all_clean"] == summary["cases"]
        and summary["sequence_consistent"] == summary["cases"]
    )
    print(f"\nWrote {run_dir}")
    print(f"  grid invariant           : {summary['grid_invariant']}/{summary['cases']}")
    print(f"  ranges invariant         : {summary['ranges_invariant']}/{summary['cases']}")
    print(f"  monotonic degradation    : {summary['monotonic']}/{summary['cases']}")
    print(f"  multi-candidate pieces   : {summary['multi_candidate_tokens']} (tone NA, D-B3B1C-001)")
    print("\n" + ("B3B2_CHANNEL_PROJECTION_COMPLETE" if ok else "B3B2_CHANNEL_PROJECTION_INCOMPLETE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
