#!/usr/bin/env python3
"""Deterministic Stage-6 streaming benchmark. **Operational only.**

Exercises the *whole* streaming writer -- chunk compute, serialisation, shard
buffering and checkpoint commits -- not `chunk_document` in isolation, because
the question this exists to answer is where Stage-6 wall-clock goes.

Runs entirely on a **synthetic, real-shaped workload and a tokenizer double**:
the pinned corpus and the real tokenizer are not available in an ML-free
environment and are not downloaded. **No number printed here is evidence about
the real tokenizer**, and the audit does not present it as such. What it does
establish is byte-equality across worker counts and the relative cost of the
writer against the chunker.

    python scripts/stage1_prepare_benchmark.py --documents 500 --workers 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import resource
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.checkpoint import (  # noqa: E402
    CheckpointIdentity,
    PrepareCheckpoint,
)
from unmark.stage1.corpus import CorpusDocument  # noqa: E402
from unmark.stage1.parallel import ordered_document_chunks  # noqa: E402

PHOBERT_RUN = re.compile(r"\S+\n?")
WORDS = ("Tôi đã đọc một quyển sách rất hay về lịch sử Việt Nam thời kỳ phong "
         "kiến và những biến động chính trị lớn trong khu vực Đông Nam Á vào "
         "thế kỷ trước khi các quốc gia giành được độc lập").split()


class ShapedTokenizer:
    """Runs are ``\\S+\\n?`` and cost is per run, like the pinned tokenizer."""

    all_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

    def get_added_vocab(self):
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    def tokenize(self, text):
        out = []
        for match in PHOBERT_RUN.finditer(text):
            run = match.group(0)
            out.extend(f"{run[:2]}@@{i}" for i in range(max(1, len(run) // 3)))
        return out

    def convert_tokens_to_ids(self, tokens):
        return [len(t) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


def tokenizer_factory():
    return ShapedTokenizer()


def make_documents(count: int, chars: int, seed: int = 20260823):
    """~`chars` per document, which at max_length 256 gives the real corpus's
    order of chunks per document (24.89 measured, Audit 029 §U.2)."""
    rng = random.Random(seed)
    documents = []
    for index in range(count):
        parts, size = [], 0
        while size < chars:
            word = rng.choice(WORDS)
            parts.append(word)
            size += len(word) + 1
        documents.append(
            CorpusDocument(
                document_id=f"doc-{index:07d}",
                content=" ".join(parts),
                source_shard="train.parquet",
                source_row=index,
            )
        )
    return documents


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        repository_head="0" * 40, protocol_version="p", chunk_schema_version="c",
        corpus_dataset="d", corpus_revision="1" * 40, corpus_files=(("a", 1, "b"),),
        tokenizer_checkpoint="t", tokenizer_revision="2" * 40,
        transformers_version="benchmark-double", max_length=256,
        raw_base_policy="RAW_BASE", split_seed=1, dev_documents=5000,
        contamination_method="m", contamination_excluded_count=0,
        document_sequence_digest="x", partition_assignment_digest="y",
    )


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_once(documents, workers: int, interval: int):
    partition_of = {d.document_id: ("dev" if i % 211 == 0 else "train")
                    for i, d in enumerate(documents)}
    root = pathlib.Path(tempfile.mkdtemp(prefix="stage6-bench-"))
    checkpoint = PrepareCheckpoint(root / "ckpt", identity(), len(documents),
                                   interval=interval)
    checkpoint.begin()
    timings = checkpoint.timings

    started = time.monotonic()
    stream = ordered_document_chunks(
        documents, partition_of, start_index=0,
        tokenizer_factory=tokenizer_factory, workers=workers, max_length=256,
        on_wait=lambda s: setattr(timings, "collector_wait_seconds",
                                  timings.collector_wait_seconds + s),
    )
    mark = time.monotonic()
    chunks_total = 0
    for index, document, chunks in stream:
        now = time.monotonic()
        timings.chunk_compute_seconds += now - mark
        chunks_total += checkpoint.add_document(index, document.document_id, chunks)
        mark = time.monotonic()
    checkpoint.commit(force=True)
    timings.stage6_total_seconds = time.monotonic() - started

    payload = b"".join(
        (checkpoint.shard_dir / shard.name).read_bytes()
        for shard in checkpoint.state.shards
    )
    checkpoint.cleanup_staging()
    return timings, chunks_total, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--chars", type=int, default=8_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    documents = make_documents(args.documents, args.chars)
    report, baseline = [], None
    for workers in args.workers:
        timings, chunks, payload = run_once(documents, workers, args.interval)
        total = max(1e-9, timings.stage6_total_seconds)
        if baseline is None:
            baseline = payload
        row = {
            "workers": workers,
            "documents": timings.documents_processed,
            "chunks": chunks,
            "seconds": round(total, 2),
            "docs_per_second": round(timings.documents_processed / total, 2),
            "chunks_per_second": round(chunks / total, 1),
            "chunk_compute_seconds": round(timings.chunk_compute_seconds, 2),
            "serialization_seconds": round(timings.serialization_seconds, 3),
            "checkpoint_commit_seconds": round(timings.checkpoint_commit_seconds, 3),
            "collector_wait_seconds": round(timings.collector_wait_seconds, 2),
            "payload_bytes": len(payload),
            "peak_rss_mb": round(rss_mb()),
            "payload_identical_to_workers_1": payload == baseline,
        }
        report.append(row)
        if not args.json:
            print(f"workers={workers:<2d} {row['docs_per_second']:>7.2f} docs/s  "
                  f"{row['chunks_per_second']:>8.1f} chunks/s  "
                  f"compute {row['chunk_compute_seconds']:>6.2f}s  "
                  f"serialise {row['serialization_seconds']:>6.3f}s  "
                  f"commit {row['checkpoint_commit_seconds']:>6.3f}s  "
                  f"RSS {row['peak_rss_mb']:>4d}MB  "
                  f"identical={row['payload_identical_to_workers_1']}", flush=True)

    if not all(r["payload_identical_to_workers_1"] for r in report):
        print("FAIL: payload differed across worker counts", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
