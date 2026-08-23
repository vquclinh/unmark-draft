#!/usr/bin/env python3
"""Colab-only micro-probe: real PhoBERT tokenizer vs the Stage-6 optimisation.

Validates, on the **pinned** tokenizer, that the optimised length functions
return the **same numbers** as the authoritative pre-optimisation pathway, and
that old and optimised chunking agree exactly.

    vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6

**No encoder is loaded. No forward pass. No optimizer. No training.** Only the
tokenizer, on a few dozen short strings.

Run in Colab::

    python scripts/stage1_tokenizer_probe.py

`--help` prints usage and exits **without loading the tokenizer**.

Exit status 0 means every comparison passed.

**History.** The first version failed on real PhoBERT with `composed 5, exact 7`.
Revision 3a read that as "per-run composition is false" and removed it; the
forensics in Audit 029 section T show the real cause was composing over the
plain non-whitespace run rather than the tokenizer's own unit, which also
absorbs a trailing newline. Revision 3b
composes over the exact unit, so this probe now also checks the tokenizer's own
decomposition and the newline regression cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TITLE = "Đội_tuyển_bóng_đá_quốc_gia_Afghanistan"

PROBE_STRINGS = [
    "Tôi đã đọc",
    "Tôi đã đọc quyển sách này rồi",
    "hoà bình", "hòa bình", "thúy", "thuý", "khỏe", "khoẻ",
    "đường ĐƯỜNG Đ đ", "Müller café naïve",
    "Xem https://vi.wikipedia.org/wiki/Việt_Nam và nguoi.dung@example.vn",
    "Tôi, đã; đọc: quyển! sách? này. rồi...",
    "VNU-HCM (VAT) Viet-Nam nhien.",
    "alpha  beta\tgamma\n\ndelta", "   padded   ", "a\nb", "một\n",
    "Tôi\nđã\nđọc", "Tôi đã đọc\n", "Tôi đã\nđọc", "\nmột", "x\n\n\ny",
    "a\r\nb", "\n", "\n\n", " ",
    TITLE, "_".join([TITLE] * 3),
    "Giảng viên dạy dễ hiểu nhưng đề thi hơi khó",
    "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương " * 12,
]

PROBE_DOCUMENTS = {
    "short": "Tôi đã đọc",
    "vietnamese": "Giảng viên dạy dễ hiểu nhưng đề thi hơi khó so với nội dung " * 12,
    "whitespace": "alpha  beta\tgamma\n\ndelta   ",
    "punctuation": "Tôi, đã; đọc: quyển! sách? này. rồi...",
    "urls": "Xem https://vi.wikipedia.org/wiki/Việt_Nam và a@b.vn " * 8,
    "oversized-unit": "_".join([TITLE] * 12),
    "oversized-in-text": "mở đầu " + "_".join([TITLE] * 12) + " kết thúc",
    "mixed": "Tôi dùng Python và PyTorch cho Deep Learning " * 12,
}

_NON_WHITESPACE = re.compile(r"\S+")
_SEGMENT = re.compile(r"\s+|\S+")


def build_parser() -> argparse.ArgumentParser:
    """Minimal CLI. No scientific override flags."""
    parser = argparse.ArgumentParser(
        description="Real-PhoBERT micro-probe for the Stage-6 length optimisation. "
                    "Loads the pinned tokenizer only: no encoder, no forward pass, "
                    "no optimizer, no training.",
    )
    parser.add_argument(
        "--max-diagnostics", type=int, default=3,
        help="how many mismatches to report in detail (default: 3)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Imports happen AFTER argument parsing so --help never touches the tokenizer.
    from transformers import AutoTokenizer

    from unmark.orthography import canon, decompose
    from unmark.stage1.chunking import ChunkingViolation, chunk_document
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.corpus import CorpusDocument
    from unmark.stage1.lengths import build_length_functions
    from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, MAX_LENGTH

    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=ENCODER_REVISION, use_fast=False
    )
    failures: list[dict] = []
    diagnostics: list[dict] = []

    def whole_length(text: str) -> int:
        """The AUTHORITATIVE definition: full API chain, whole string."""
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        return len(tokenizer.build_inputs_with_special_tokens(list(ids)))

    def base_transform(text: str) -> str:
        return decompose(canon(text)).base_text

    PATHWAYS = (("reference", canon), ("base", base_transform))
    specials = whole_length("")

    def diagnose(index, text, label, transform, exact, optimized) -> dict:
        """Safe diagnostic for one mismatch. Fixtures only -- no corpus text."""
        transformed = transform(text)
        tokens = tokenizer.tokenize(transformed)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        runs = _NON_WHITESPACE.findall(transformed)
        per_run = [
            {"run": run, "tokens": tokenizer.tokenize(run),
             "count": len(tokenizer.tokenize(run))}
            for run in runs
        ]
        return {
            "fixture_index": index,
            "fixture_repr": repr(text),
            "pathway": label,
            "transformed_repr": repr(transformed),
            "whole_tokens": tokens,
            "whole_id_count": len(list(ids)),
            "whole_tokens_before_specials": len(tokens),
            "whole_final_length": exact,
            "special_tokens_assumed": specials,
            "runs": runs,
            "per_run": per_run,
            "sum_of_per_run_counts": sum(r["count"] for r in per_run),
            "removed_shortcut_would_have_given": specials + sum(r["count"] for r in per_run),
            "optimized_length": optimized,
        }

    # --- CHECK 1: optimised length == authoritative length -----------------
    for index, text in enumerate(PROBE_STRINGS):
        ref_opt, base_opt, _ = build_length_functions(tokenizer)
        for label, transform in PATHWAYS:
            exact = whole_length(transform(text))
            try:
                optimized = (ref_opt if label == "reference" else base_opt)(text)
            except Stage1ContractViolation as error:
                failures.append({"check": "length_equality", "pathway": label,
                                 "fixture_index": index, "violation": str(error)})
                if len(diagnostics) < args.max_diagnostics:
                    diagnostics.append(diagnose(index, text, label, transform, exact, None))
                continue
            if optimized != exact:
                failures.append({"check": "length_equality", "pathway": label,
                                 "fixture_index": index,
                                 "optimized": optimized, "authoritative": exact})
                if len(diagnostics) < args.max_diagnostics:
                    diagnostics.append(
                        diagnose(index, text, label, transform, exact, optimized)
                    )

    # --- CHECK 2: the tokenizer's OWN decomposition unit --------------------
    # `tokenize` must equal `_tokenize`, and composing over \S+\n? must be exact
    # while naive \S+ is expected to fail on newline cases.
    phobert_run = re.compile(r"\S+\n?")
    naive_run = re.compile(r"\S+")
    naive_failures = 0
    for index, text in enumerate(PROBE_STRINGS):
        for label, transform in PATHWAYS:
            transformed = transform(text)
            whole = tokenizer.tokenize(transformed)
            private = getattr(tokenizer, "_tokenize", None)
            if private is not None and list(private(transformed)) != list(whole):
                failures.append({"check": "wrapper_vs_private_tokenize",
                                 "fixture_index": index, "pathway": label})
            exact_runs = [t for run in phobert_run.findall(transformed)
                          for t in tokenizer.tokenize(run)]
            if exact_runs != list(whole):
                failures.append({"check": "phobert_run_composition",
                                 "fixture_index": index, "pathway": label,
                                 "whole": len(whole), "composed": len(exact_runs)})
            naive = [t for run in naive_run.findall(transformed)
                     for t in tokenizer.tokenize(run)]
            if naive != list(whole):
                naive_failures += 1

    # --- CHECK 3: transform composability across whitespace segments -------
    for index, text in enumerate(PROBE_STRINGS):
        parts = [m.group(0) for m in _SEGMENT.finditer(text)]
        if "".join(canon(p) for p in parts) != canon(text):
            failures.append({"check": "canon_composability", "fixture_index": index})
        if "".join(base_transform(p) for p in parts) != base_transform(text):
            failures.append({"check": "base_composability", "fixture_index": index})

    # --- CHECK 4: old vs optimised chunk output, field by field ------------
    boundary_checks = 0
    for name, content in PROBE_DOCUMENTS.items():
        document = CorpusDocument(name, content, "train.parquet", 0)

        def run(reference_length, base_length):
            try:
                chunks = chunk_document(document, "train",
                                        reference_length=reference_length,
                                        base_length=base_length, max_length=MAX_LENGTH)
                return ("ok", [(c.chunk_id, c.document_id, c.partition, c.chunk_index,
                                c.text, c.source_start, c.source_end,
                                c.reference_length, c.base_length) for c in chunks])
            except ChunkingViolation as error:
                return ("raised", str(error))
            except Stage1ContractViolation as error:
                return ("verifier", str(error))

        old = run(lambda t: whole_length(canon(t)),
                  lambda t: whole_length(base_transform(t)))
        ref_opt, base_opt, _ = build_length_functions(tokenizer)
        new = run(ref_opt, base_opt)
        boundary_checks += 1
        if old != new:
            failures.append({"check": "chunk_output", "document": name,
                             "old_status": old[0], "new_status": new[0]})

    report = {
        "probe": "STAGE1_TOKENIZER_RUN_COMPOSITION",
        "run_unit": "\\S+\\n?",
        "checkpoint": ENCODER_CHECKPOINT,
        "revision": ENCODER_REVISION,
        "transformers": __import__("transformers").__version__,
        "special_tokens": specials,
        "naive_backslash_S_failures": naive_failures,
        "naive_composition_expected_to_fail": True,
        "probe_strings": len(PROBE_STRINGS),
        "documents_compared": boundary_checks,
        "encoder_loaded": False,
        "forward_passes": 0,
        "optimizer_steps": 0,
        "failures": failures,
        "first_mismatch_diagnostics": diagnostics,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
