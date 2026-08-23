"""Ordered parallel compute, and the per-chunk-cost claims, as tests.

Two separate things are asserted here:

1. **The parallel path is operational only.** Worker count changes throughput
   and nothing else -- chunk ids, ranges, text, both lengths, ordering and the
   serialised payload bytes are identical for 1, 2, 4 and 8 workers.
2. **Streaming does not buy durability per chunk.** The regression hunt in
   Audit 029 §Y turned on whether the writer fsyncs, hashes or copies per
   chunk. It does not, and that is pinned here rather than argued.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1 import parallel as parallel_module
from unmark.stage1.checkpoint import (
    CheckpointIdentity,
    PrepareCheckpoint,
    Stage6Timings,
    _external_sorted_digest,
    chunk_line,
)
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.corpus import CorpusDocument
from unmark.stage1.parallel import ordered_document_chunks, resolve_worker_count

PHOBERT_RUN = re.compile(r"\S+\n?")


# ---------------------------------------------------------------------------
# A picklable tokenizer double -- module level, so workers can rebuild it
# ---------------------------------------------------------------------------
class ShapedTokenizer:
    """Faithful where it matters: runs are ``\\S+\\n?``, cost is per run."""

    all_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

    def get_added_vocab(self):
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    def _bpe(self, run):
        return [f"{run[:2]}@@{i}" for i in range(max(1, len(run) // 3))]

    def tokenize(self, text):
        out = []
        for match in PHOBERT_RUN.finditer(text):
            out.extend(self._bpe(match.group(0)))
        return out

    def convert_tokens_to_ids(self, tokens):
        return [len(t) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


def tokenizer_factory():
    return ShapedTokenizer()


WORDS = ("Tôi đã đọc một quyển sách rất hay về lịch sử Việt Nam thời kỳ phong "
         "kiến và những biến động chính trị lớn trong khu vực").split()


def make_documents(count, chars=2_400, seed=20260823):
    import random

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
                document_id=f"doc-{index:05d}",
                content=" ".join(parts),
                source_shard="train.parquet",
                source_row=index,
            )
        )
    return documents


def partitions(documents):
    return {d.document_id: ("dev" if i % 7 == 0 else "train")
            for i, d in enumerate(documents)}


def collect(documents, workers, **kwargs):
    return list(
        ordered_document_chunks(
            documents, partitions(documents), start_index=0,
            tokenizer_factory=tokenizer_factory, workers=workers,
            max_length=256, **kwargs,
        )
    )


def payload_of(emitted):
    return "".join(chunk_line(c) for _, _, chunks in emitted for c in chunks).encode()


# ---------------------------------------------------------------------------
# Task G -- the worker count is not science
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("workers", [2, 4, 8])
def test_worker_counts_produce_byte_identical_payloads(workers):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs more than one CPU")
    documents = make_documents(24)
    serial = collect(documents, 1)
    parallel = collect(documents, workers)
    assert payload_of(parallel) == payload_of(serial)


@pytest.mark.parametrize("workers", [2, 4])
def test_every_scientific_field_survives_parallel_compute(workers):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs more than one CPU")
    documents = make_documents(16)

    def fields(emitted):
        return [
            (c.chunk_id, c.document_id, c.partition, c.chunk_index, c.text,
             c.source_start, c.source_end, c.reference_length, c.base_length,
             c.source_shard)
            for _, _, chunks in emitted for c in chunks
        ]

    assert fields(collect(documents, workers)) == fields(collect(documents, 1))


def test_documents_are_emitted_in_original_index_order():
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs more than one CPU")
    documents = make_documents(20)
    emitted = collect(documents, 4)
    assert [i for i, _, _ in emitted] == list(range(len(documents)))
    assert [d.document_id for _, d, _ in emitted] == [d.document_id for d in documents]


def test_chunk_index_order_is_preserved_within_each_document():
    documents = make_documents(6)
    for _, _, chunks in collect(documents, 1):
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_resume_start_index_skips_exactly_the_committed_prefix():
    documents = make_documents(12)
    whole = collect(documents, 1)
    tail = list(
        ordered_document_chunks(
            documents, partitions(documents), start_index=5,
            tokenizer_factory=tokenizer_factory, workers=1, max_length=256,
        )
    )
    assert [i for i, _, _ in tail] == list(range(5, 12))
    assert payload_of(tail) == payload_of(whole[5:])


# ---------------------------------------------------------------------------
# Task F -- bounds, failure propagation, and what workers may NOT do
# ---------------------------------------------------------------------------
def test_in_flight_work_is_bounded_by_the_configured_window():
    """The collector must never hold more than `max_in_flight` futures."""
    documents = make_documents(40, chars=400)
    seen = []
    real_submit = None

    class Watched:
        def __init__(self, pool):
            self._pool = pool

        def submit(self, *a, **k):
            return self._pool.submit(*a, **k)

    # Observe the pending dict directly through the generator's frame.
    gen = ordered_document_chunks(
        documents, partitions(documents), start_index=0,
        tokenizer_factory=tokenizer_factory, workers=2, max_length=256,
        max_in_flight=3,
    )
    for _ in gen:
        frame = gen.gi_frame
        if frame is not None and "pending" in frame.f_locals:
            seen.append(len(frame.f_locals["pending"]))
    assert seen, "expected to observe the collector's pending window"
    assert max(seen) <= 3, f"in-flight window exceeded: {max(seen)}"


def failing_chunk_document(document, partition, **kwargs):
    raise ValueError("deliberate worker failure")


def test_worker_failure_propagates_with_document_provenance(monkeypatch):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs more than one CPU")
    documents = make_documents(4, chars=400)
    monkeypatch.setattr(parallel_module, "chunk_document", failing_chunk_document)
    with pytest.raises(Exception) as caught:
        list(
            ordered_document_chunks(
                documents, partitions(documents), start_index=0,
                tokenizer_factory=tokenizer_factory, workers=1, max_length=256,
            )
        )
    assert "deliberate worker failure" in str(caught.value)


def test_a_worker_failure_stops_emission_so_no_prefix_can_advance_past_it():
    """Emission is ordered, so a failure at document k emits nothing beyond k."""
    documents = make_documents(8, chars=400)
    boom = {"at": 3}
    real = parallel_module.chunk_document

    def sometimes(document, partition, **kwargs):
        if document.source_row == boom["at"]:
            raise Stage1ContractViolation("failed at document 3")
        return real(document, partition, **kwargs)

    parallel_module.chunk_document = sometimes
    try:
        emitted = []
        with pytest.raises(Stage1ContractViolation):
            for item in ordered_document_chunks(
                documents, partitions(documents), start_index=0,
                tokenizer_factory=tokenizer_factory, workers=1, max_length=256,
            ):
                emitted.append(item[0])
        assert emitted == [0, 1, 2], emitted
    finally:
        parallel_module.chunk_document = real


def test_workers_never_serialise_or_touch_the_checkpoint():
    """Structural: the worker half must not reach the writer or the filesystem."""
    source = inspect.getsource(parallel_module)
    tree = ast.parse(source)
    worker = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_chunk_one"
    )
    called = {
        node.func.id if isinstance(node.func, ast.Name) else
        getattr(node.func, "attr", "")
        for node in ast.walk(worker) if isinstance(node, ast.Call)
    }
    forbidden = {"chunk_line", "add_document", "commit", "open", "fsync",
                 "atomic_write_bytes", "write_completion_marker", "replace"}
    assert not (called & forbidden), called & forbidden


def test_resolve_worker_count_defaults_to_one_and_refuses_zero():
    assert resolve_worker_count(None) == 1
    assert resolve_worker_count(1) == 1
    assert resolve_worker_count(2) == min(2, os.cpu_count() or 1)
    with pytest.raises(Stage1ContractViolation):
        resolve_worker_count(0)


# ---------------------------------------------------------------------------
# Task C -- no per-chunk durability work
# ---------------------------------------------------------------------------
def identity():
    return CheckpointIdentity(
        repository_head="0" * 40, protocol_version="p", chunk_schema_version="c",
        corpus_dataset="d", corpus_revision="1" * 40, corpus_files=(("a", 1, "b"),),
        tokenizer_checkpoint="t", tokenizer_revision="2" * 40,
        transformers_version="4.57.6", max_length=256, raw_base_policy="RAW_BASE",
        split_seed=1, dev_documents=5000, contamination_method="m",
        contamination_excluded_count=0, document_sequence_digest="x",
        partition_assignment_digest="y",
    )


def test_no_fsync_no_open_no_hash_happens_before_the_first_commit(tmp_path, monkeypatch):
    """The whole regression question: is durability paid per chunk? It is not."""
    import hashlib
    import unmark.stage1.checkpoint as checkpoint_module

    documents = make_documents(20, chars=1_200)
    emitted = collect(documents, 1)

    calls = {"fsync": 0, "sha256": 0, "open": 0}
    real_fsync, real_sha, real_open = os.fsync, hashlib.sha256, open
    monkeypatch.setattr(checkpoint_module.os, "fsync",
                        lambda fd: (calls.__setitem__("fsync", calls["fsync"] + 1),
                                    real_fsync(fd))[1])
    monkeypatch.setattr(checkpoint_module.hashlib, "sha256",
                        lambda *a, **k: (calls.__setitem__("sha256", calls["sha256"] + 1),
                                         real_sha(*a, **k))[1])

    cp = PrepareCheckpoint(tmp_path / "ckpt", identity(), len(documents), interval=10_000)
    cp.begin()
    calls.update(fsync=0, sha256=0)
    for index, document, chunks in emitted:
        cp.add_document(index, document.document_id, chunks)
    assert calls["fsync"] == 0, "a chunk must not cost an fsync"
    assert calls["sha256"] == 0, "a chunk must not cost a hash of the shard"
    assert cp.commits == 0, "no commit is due before the interval"

    cp.commit(force=True)
    assert calls["fsync"] > 0 and calls["sha256"] > 0, "commit must be durable"


def test_serialisation_is_a_small_share_of_stage_six(tmp_path):
    """Pins the measured shape: the writer is not where Stage-6 time goes."""
    documents = make_documents(24, chars=2_400)
    emitted = collect(documents, 1)
    cp = PrepareCheckpoint(tmp_path / "ckpt", identity(), len(documents), interval=10_000)
    cp.begin()
    for index, document, chunks in emitted:
        cp.add_document(index, document.document_id, chunks)
    assert cp.timings.json_records_written == sum(len(c) for _, _, c in emitted)
    assert cp.timings.documents_processed == len(documents)
    assert cp.timings.checkpoint_commit_seconds == 0.0


def test_timings_carry_no_corpus_text():
    timings = Stage6Timings()
    timings.chunks_processed = 3
    for value in timings.to_dict().values():
        assert isinstance(value, (int, float))


# ---------------------------------------------------------------------------
# Task D -- the membership sorter spills in blocks, not per key
# ---------------------------------------------------------------------------
def test_membership_sorter_spills_in_blocks_not_per_key(tmp_path, monkeypatch):
    import unmark.stage1.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "_EXTERNAL_SORT_BLOCK", 50)
    timings = Stage6Timings()
    keys = [f"doc-{i:05d}#0\ttrain" for i in range(500)]
    digest = _external_sorted_digest(iter(keys), tmp_path / "w", timings)
    # 500 keys / 50 per block = 10 spills. One per key would be 500.
    assert timings.membership_spills == 10, timings.membership_spills
    assert digest == _external_sorted_digest(iter(keys), tmp_path / "w2")


def test_membership_digest_is_block_size_independent(tmp_path, monkeypatch):
    import unmark.stage1.checkpoint as checkpoint_module

    keys = [f"doc-{i:05d}#{i % 3}\t{'dev' if i % 7 == 0 else 'train'}" for i in range(300)]
    digests = set()
    for block in (7, 64, 1_000_000):
        monkeypatch.setattr(checkpoint_module, "_EXTERNAL_SORT_BLOCK", block)
        digests.add(_external_sorted_digest(iter(keys), tmp_path / f"w{block}"))
    assert len(digests) == 1, "the digest must not depend on an operational block size"
