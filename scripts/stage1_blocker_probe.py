#!/usr/bin/env python3
"""Re-probe one real Stage-6 blocker region. **Metadata out, never text.**

Stage 6 has now failed closed twice on a single document region, and each time
the question was the same: does the repaired chunker subdivide *that region*, or
only fixtures shaped like it? This answers it against the real bytes, in Colab,
without ever printing or persisting corpus content.

What it emits: the region's sha256, its length, Unicode category counts, both
pathway lengths, the safe-cut count, how many of those cuts leave **both** halves
within `max_length` on **both** pathways, and whether the emitted chunks
reconstruct the region byte-exactly. **No raw text, in any field, ever** --
asserted by test.

    python scripts/stage1_blocker_probe.py \\
        --corpus-root /path/to/uvw --shard train.parquet --row 894182 \\
        --start 4887 --end 4985 \\
        --expect-sha256 e50cf079...52a48

It loads **no encoder**, runs **no forward pass**, constructs **no optimizer**
and takes **no training step** -- AST-asserted, exactly as the tokenizer probe
is. Success requires at least one safe interior cut, successful chunking with
byte-exact reconstruction, and every emitted chunk within `max_length` on both
pathways. The viable-single-cut count is reported always and *required* only
when the region splits in two -- the real blocker's case -- because a region
that legitimately needs three chunks has no single cut that fits both halves.
Exact counts are reported, never asserted: they follow from the tokenizer, not
from a number this script is entitled to predict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.orthography.decompose import source_letter_runs  # noqa: E402
from unmark.stage1.chunking import (  # noqa: E402
    ChunkingViolation,
    chunk_document,
    safe_cut_offsets,
    verify_tiles_source,
)
from unmark.stage1.corpus import CorpusDocument  # noqa: E402
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, MAX_LENGTH  # noqa: E402


def category_census(text: str) -> dict[str, int]:
    """Unicode general-category counts. Structure, never content."""
    census: dict[str, int] = {}
    for char in text:
        category = unicodedata.category(char)
        census[category] = census.get(category, 0) + 1
    return dict(sorted(census.items()))


def script_census(text: str) -> dict[str, int]:
    """Unicode name-family counts (HANGUL, LATIN, CJK, ...). Structure only."""
    census: dict[str, int] = {}
    for char in text:
        family = unicodedata.name(char, "UNNAMED").split()[0]
        census[family] = census.get(family, 0) + 1
    return dict(sorted(census.items()))


def build_report(region: str, reference_length, base_length, max_length: int) -> dict:
    """Everything the reprobe needs to decide, with no corpus text in it."""
    cuts = sorted(safe_cut_offsets(region))
    interior = [c for c in cuts if 0 < c < len(region)]

    viable = []
    for cut in interior:
        left, right = region[:cut], region[cut:]
        if (reference_length(left) <= max_length and base_length(left) <= max_length
                and reference_length(right) <= max_length
                and base_length(right) <= max_length):
            viable.append(cut)

    report = {
        "region_sha256": hashlib.sha256(region.encode("utf-8")).hexdigest(),
        "region_chars": len(region),
        "unicode_categories": category_census(region),
        "unicode_name_families": script_census(region),
        "reference_length": reference_length(region),
        "base_length": base_length(region),
        "max_length": max_length,
        "protected_runs": len(source_letter_runs(region)),
        "safe_interior_cuts": len(interior),
        "viable_cuts_both_pathways": len(viable),
        "encoder_loaded": False,
        "forward_passes": 0,
        "optimizer_steps": 0,
    }

    failures: list[str] = []
    if report["safe_interior_cuts"] < 1:
        failures.append("no safe interior cut exists in the region")

    # Chunk it for real, and require byte-exact reconstruction.
    document = CorpusDocument(
        document_id="probe-region", content=region,
        source_shard="probe", source_row=0,
    )
    try:
        chunks = chunk_document(document, "train", reference_length=reference_length,
                                base_length=base_length, max_length=max_length)
        verify_tiles_source(chunks, region, "probe-region")
        report["chunks"] = len(chunks)
        report["reconstructs_byte_exact"] = "".join(c.text for c in chunks) == region
        report["max_emitted_reference"] = max(c.reference_length for c in chunks)
        report["max_emitted_base"] = max(c.base_length for c in chunks)
        if not report["reconstructs_byte_exact"]:
            failures.append("chunks do not reconstruct the region byte-exactly")
        if report["max_emitted_reference"] > max_length or report["max_emitted_base"] > max_length:
            failures.append("an emitted chunk exceeds max_length on a pathway")
        # A single cut can only ever make *both* halves fit when the region needs
        # exactly two chunks. Requiring it unconditionally would fail a region
        # that is perfectly chunkable into three, so the requirement is scoped to
        # the case it is meaningful in -- which is the real blocker's case.
        if report["chunks"] == 2 and report["viable_cuts_both_pathways"] < 1:
            failures.append(
                "region splits in two, but no single cut leaves both halves "
                "within max_length on both pathways"
            )
    except ChunkingViolation as error:
        report["chunks"] = 0
        report["reconstructs_byte_exact"] = False
        # The message carries ids, ranges and lengths -- never corpus text.
        failures.append(f"ChunkingViolation: {error}")

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--shard", required=True, help="e.g. train.parquet")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--expect-sha256", default=None,
                        help="refuse to proceed unless the region hashes to this")
    parser.add_argument("--revision", default=ENCODER_REVISION)
    args = parser.parse_args(argv)

    # Imported here, after parse_args, so --help never touches transformers.
    from transformers import AutoTokenizer  # noqa: PLC0415

    from unmark.stage1.corpus import read_shard  # noqa: PLC0415
    from unmark.stage1.lengths import build_length_functions  # noqa: PLC0415

    documents = read_shard(pathlib.Path(args.corpus_root) / args.shard, args.shard)
    region = documents[args.row].content[args.start:args.end]

    digest = hashlib.sha256(region.encode("utf-8")).hexdigest()
    if args.expect_sha256 and digest != args.expect_sha256:
        print(json.dumps({
            "status": "FAIL",
            "failures": [f"region sha256 {digest} != expected {args.expect_sha256}"],
        }, indent=2))
        return 1

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CHECKPOINT, revision=args.revision)
    reference_length, base_length, _ = build_length_functions(tokenizer)

    report = build_report(region, reference_length, base_length, MAX_LENGTH)
    report["document_id_sha256"] = hashlib.sha256(
        documents[args.row].document_id.encode("utf-8")
    ).hexdigest()
    report["source_row"] = args.row
    report["source_range"] = [args.start, args.end]
    report["tokenizer_revision"] = args.revision

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
