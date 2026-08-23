#!/usr/bin/env python3
"""Colab-only micro-probe: real PhoBERT tokenizer vs the Stage-6 optimisation.

Validates, on the **pinned** tokenizer, the two properties Audit 029 §R relies
on, and compares old-vs-optimised chunk boundaries on a small deterministic set.

    vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6

**No encoder is loaded. No forward pass. No optimizer. No training.** Only the
tokenizer, on a few dozen short strings.

Run in Colab::

    python scripts/stage1_tokenizer_probe.py

Exit status 0 means both properties held and every boundary matched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.orthography import canon, decompose  # noqa: E402
from unmark.stage1.chunking import ChunkingViolation, chunk_document  # noqa: E402
from unmark.stage1.corpus import CorpusDocument  # noqa: E402
from unmark.stage1.lengths import build_length_functions  # noqa: E402
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, MAX_LENGTH  # noqa: E402

TITLE = "Đội_tuyển_bóng_đá_quốc_gia_Afghanistan"

PROBE_STRINGS = [
    "Tôi đã đọc quyển sách này rồi",
    "hoà bình", "hòa bình", "thúy", "thuý", "khỏe", "khoẻ",
    "đường ĐƯỜNG Đ đ", "Müller café naïve",
    "Xem https://vi.wikipedia.org/wiki/Việt_Nam và nguoi.dung@example.vn",
    "Tôi, đã; đọc: quyển! sách? này. rồi...",
    "VNU-HCM (VAT) Viet-Nam nhien.",
    "alpha  beta\tgamma\n\ndelta", "   padded   ",
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


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=ENCODER_REVISION, use_fast=False
    )
    failures: list[dict] = []

    def whole_length(text: str) -> int:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        return len(tokenizer.build_inputs_with_special_tokens(list(ids)))

    # --- PROPERTY 1: per-non-whitespace-chunk tokenization (D-B3B1B-001) ----
    specials = whole_length("")
    for text in PROBE_STRINGS:
        for label, transform in (("clean", canon),
                                 ("base", lambda t: decompose(canon(t)).base_text)):
            transformed = transform(text)
            exact = whole_length(transformed)
            composed = specials + sum(
                len(tokenizer.tokenize(transform(run))) for run in text.split()
            )
            if exact != composed:
                failures.append({"property": "per_chunk_composition", "path": label,
                                 "text": text[:60], "exact": exact, "composed": composed})

    # --- PROPERTY 2: transform composability across whitespace segments -----
    import re

    segment = re.compile(r"\s+|\S+")
    for text in PROBE_STRINGS:
        parts = [m.group(0) for m in segment.finditer(text)]
        if "".join(canon(p) for p in parts) != canon(text):
            failures.append({"property": "canon_composability", "text": text[:60]})
        direct = decompose(canon(text)).base_text
        if "".join(decompose(canon(p)).base_text for p in parts) != direct:
            failures.append({"property": "base_composability", "text": text[:60]})

    # --- old vs optimised chunk boundaries, byte for byte ------------------
    def old_lengths():
        return (lambda t: whole_length(canon(t)),
                lambda t: whole_length(decompose(canon(t)).base_text))

    boundary_checks = 0
    for name, content in PROBE_DOCUMENTS.items():
        document = CorpusDocument(name, content, "train.parquet", 0)
        ref_old, base_old = old_lengths()
        ref_new, base_new, _ = build_length_functions(tokenizer)

        def run(r, b):
            try:
                chunks = chunk_document(document, "train", reference_length=r,
                                        base_length=b, max_length=MAX_LENGTH)
                return ("ok", [(c.chunk_id, c.text, c.source_start, c.source_end,
                                c.reference_length, c.base_length) for c in chunks])
            except ChunkingViolation as error:
                return ("raised", str(error))

        old, new = run(ref_old, base_old), run(ref_new, base_new)
        boundary_checks += 1
        if old != new:
            failures.append({"property": "chunk_boundaries", "document": name})

    report = {
        "probe": "STAGE1_TOKENIZER_COMPOSITION",
        "checkpoint": ENCODER_CHECKPOINT,
        "revision": ENCODER_REVISION,
        "transformers": __import__("transformers").__version__,
        "probe_strings": len(PROBE_STRINGS),
        "documents_compared": boundary_checks,
        "encoder_loaded": False,
        "forward_passes": 0,
        "optimizer_steps": 0,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
