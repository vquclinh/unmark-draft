#!/usr/bin/env python3
"""READ-ONLY diagnostic for the Stage-6 / Stage-1 base-length disagreement.

Audit 043. The first real `lr-pilot` aborted fail-closed at update ~250 with

    base sequence length 257 exceeds max_length 256

even though Stage-6 recorded `overflow_count = 0` for the same prepared corpus.
This script establishes which side is right, and how many chunks are affected.

**It trains nothing.** No model is loaded, no optimizer is constructed, no
backward is called, no scientific artifact is written. It reads `chunks.jsonl`
and a tokenizer, and prints counts. Official UIT-VSFC TEST is never touched.

The two quantities it compares, per chunk:

* **Stage-6 authoritative length** -- the definition `lengths.py` documents::

      len(build_inputs_with_special_tokens(
          convert_tokens_to_ids(tokenize(base_text))))

  i.e. the tokenizer applied to the WHOLE transformed string. This is what the
  chunker enforced with `chunk.base_length > max_length -> ChunkingViolation`.

* **Stage-1 realised length** -- what `prepare_example` actually builds::

      len(_with_special_tokens(tokenizer, project_text(text, ...)[1])[0])

  `project_text` tokenizes per `whitespace_chunks`, whose `_CHUNK_PATTERN` is
  plain ``\\S+``. The tokenizer's own unit is ``\\S+\\n?`` (`PHOBERT_RUN`), and
  `lengths.py` states of the plain form: *"This regex is the contract; ``\\S+``
  must never be used for it."* Where a chunk contains a newline the two
  decompositions differ, BPE's end-of-word marker lands on a different final
  character, and the token counts can disagree.

Usage (Colab, against the real prepared corpus):

    python -B scripts/stage1_length_contract_scan.py \\
        --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \\
        --partition train --report /content/length-scan.json

    # cheap first look
    python -B scripts/stage1_length_contract_scan.py --prepared-corpus ... --limit 50000

    # sampler-order reproduction of the REAL first failure
    python -B scripts/stage1_length_contract_scan.py --prepared-corpus ... \\
        --reproduce --seed 21230

Exit codes -- a successful measurement and a broken diagnostic never share one:

    0   SUCCESS_NO_VIOLATION       the scan completed, nothing over max_length
    1   DIAGNOSTIC_FAILURE         the scanner could not complete; nothing measured
    2   SUCCESS_VIOLATION_FOUND    the scan completed and DID find a violation

Every mode also prints a machine-readable `"status"` field carrying the same
three names; branch on that in preference to the exit code.

**stdout is exactly one JSON document.** Progress, resume notices and the
"wrote ..." line all go to stderr, so `python ... | jq` works unchanged.

**GPU is deliberately not used.** Every operation this diagnostic performs is
CPU-side and sequential-by-nature: Unicode canonicalisation (`canon`),
orthographic decomposition, `re` scanning for whitespace runs, PhoBERT's BPE
merge loop over short strings, and dictionary memo lookups. None has an exact
GPU implementation in this repository, and a GPU rewrite of BPE or of the
orthography would be a reimplementation of frozen scientific transformations --
exactly what this diagnostic must not do. **GPU not used because no exact GPU
implementation provides a measured benefit.** The scan scales with CPU cores
(`--workers`) instead.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.stage1.manifest import CHUNKS_NAME  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    SELECTION_SEED,
)


EXIT_NO_VIOLATION = 0
"""SUCCESS_NO_VIOLATION -- the scan completed and found nothing over max_length."""

EXIT_DIAGNOSTIC_FAILURE = 1
"""DIAGNOSTIC_FAILURE -- the scanner itself could not complete (bad path,
unverifiable corpus, failed spot check). Nothing was measured."""

EXIT_VIOLATION_FOUND = 2
"""SUCCESS_VIOLATION_FOUND -- the scan completed and DID find a violation.

Deliberately not 1. The `--reproduce` mode originally returned 1 on a successful
find, and the notebook wrapper read that as a scanner crash (Audit 043 §8b).
A successful measurement and a broken diagnostic must not share an exit code.
Every mode also prints a machine-readable `"status"` field, which is the
preferred thing to branch on.
"""

STATUS_BY_EXIT = {
    EXIT_NO_VIOLATION: "SUCCESS_NO_VIOLATION",
    EXIT_DIAGNOSTIC_FAILURE: "DIAGNOSTIC_FAILURE",
    EXIT_VIOLATION_FOUND: "SUCCESS_VIOLATION_FOUND",
}


def stable_id(text: str) -> str:
    """A chunk fingerprint. Never the text itself -- audits carry no corpus."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_tokenizer(revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=revision, use_fast=False
    )


def stage6_length_functions(tokenizer):
    """The REAL Stage-6 length functions. No reimplementation, no fallback.

    `scripts/stage1_runner.py::_length_functions` builds these exact objects for
    the chunker, so measuring with them measures precisely the quantity Stage-6
    enforced with `chunk.base_length > max_length -> ChunkingViolation`.

    The first version of this script instead did its own
    `decompose(canon(text)).base_text` and guessed the import as
    `unmark.linguistics.canon`. `canon` is exported from `unmark.orthography`,
    so the real Colab run died with `ImportError` after loading the corpus and
    walking the sampler -- see Audit 043 §8a. Delegating removes the guess and
    the duplicated normalisation together.
    """
    from unmark.stage1.lengths import build_length_functions

    return build_length_functions(tokenizer)


def authoritative_base_length(base_length, text: str) -> int:
    """Stage-6's authoritative base length, via Stage-6's own function.

    `base_length` is the callable returned by `build_length_functions`. Its
    documented definition is::

        len(build_inputs_with_special_tokens(
            convert_tokens_to_ids(tokenize(transform(x)))))

    Note the classifier is deliberately absent: `ComposedTransforms.base`
    computes `decompose(canon(t)).base_text` with no eligibility classifier, and
    the classifier provably does not change `base_text` -- it only labels spans
    (`decompose` docstring: "the round-trip is unaffected"). So this measures the
    same string `project_text` bases its ids on.
    """
    return base_length(text)


def realised_base_length(text: str, tokenizer, classifier, unk_token_id) -> int:
    """Stage-1's realised length via the FULL production path.

    Kept as the reference implementation: the fast path below is verified
    against this, and offenders are always re-measured with it.
    """
    from unmark.stage1.data import project_text

    _base_text, content_ids, _projections = project_text(
        text, tokenizer, classifier, unk_token_id
    )
    return len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))


class RealisedLengthCounter:
    r"""EXACT count-only Stage-1 realised base length. Same semantics, less work.

    `project_text` returns `(base_text, content_ids, projections)` and appends
    **one id per aligned piece**. For a *length* the projections are pure
    overhead, and they dominate: measured on a realistic chunk,
    `overlay_orthography` + `project_piece` are **71.3 %** of the cost, and the
    `tones` comprehension is `O(regions x syllables)`.

    So this computes exactly `len(content_ids)` and nothing else:

        base_text = transforms.base(text)          # Stage-6's own memoised transform
        n = sum(len(align_chunk(c, ...).pieces) for c in whitespace_chunks(base_text))
        length = len(build_inputs_with_special_tokens([...n...]))

    Three points where it would be easy to be subtly wrong, and is not:

    * it counts `align_chunk(...).pieces`, **not** `len(tokens)`. `align_chunk`
      returns `pieces=()` on failure, so a token count would over-count exactly
      the rows where alignment fails;
    * it uses `whitespace_chunks` -- Stage-1's own `\S+` unit -- so it reproduces
      the very run semantics under investigation rather than correcting them;
    * `transforms.base(text)` is `decompose(canon(text)).base_text` with no
      eligibility classifier. `project_text` passes one, but the classifier only
      labels spans and provably does not change `base_text` (Audit 043 §8a), and
      nothing in the *count* depends on eligibility.

    Two memos make the scan tractable. `ComposedTransforms` already memoises the
    base transform per whitespace segment; this adds a memo from chunk text to
    piece count, since the count depends on `chunk.text` alone. Both are
    per-process caches of pure functions: they change speed, never results.
    """

    def __init__(self, tokenizer, unk_token_id, max_memo: int = 2_000_000) -> None:
        from unmark.stage1.lengths import ComposedTransforms

        self._tokenizer = tokenizer
        self._unk = unk_token_id
        self._transforms = ComposedTransforms()
        self._pieces: dict[str, int] = {}
        self._max_memo = max_memo
        self.memo_hits = 0
        self.memo_misses = 0

    def _piece_count(self, chunk) -> int:
        cached = self._pieces.get(chunk.text)
        if cached is not None:
            self.memo_hits += 1
            return cached
        self.memo_misses += 1
        from unmark.alignment.manual import align_chunk

        tokens = tuple(self._tokenizer.tokenize(chunk.text))
        ids = tuple(self._tokenizer.convert_tokens_to_ids(list(tokens)))
        count = len(align_chunk(chunk, tokens, ids, unk_token_id=self._unk).pieces)
        if len(self._pieces) < self._max_memo:
            self._pieces[chunk.text] = count
        return count

    def length(self, text: str) -> int:
        from unmark.alignment.manual import whitespace_chunks

        base_text = self._transforms.base(text)
        content = sum(self._piece_count(chunk) for chunk in whitespace_chunks(base_text))
        return len(self._tokenizer.build_inputs_with_special_tokens([0] * content))


def chunks_path_for(args) -> Path:
    """The file the scan reads: the staged local copy when there is one."""
    if getattr(args, "local_chunks", None):
        return Path(args.local_chunks)
    return Path(args.prepared_corpus) / CHUNKS_NAME


def iter_chunks(prepared: Path, partition: str | None, chunks_path: Path | None = None):
    path = Path(chunks_path) if chunks_path is not None else Path(prepared) / CHUNKS_NAME
    if not path.is_file():
        raise SystemExit(f"REFUSED: prepared chunks not found: {path}")
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if partition and row.get("partition") != partition:
                continue
            yield index, row


def require_corpus(prepared: Path) -> Path:
    """Fail closed before loading a tokenizer, so a bad path costs no minutes."""
    path = Path(prepared) / CHUNKS_NAME
    if not path.is_file():
        raise SystemExit(f"REFUSED: prepared chunks not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Worker state -- built once per process, never shipped between processes
# ---------------------------------------------------------------------------
def sha256_of(path: Path, block: int = 8 << 20) -> tuple[int, str]:
    """`(bytes, sha256)` of a file, streamed. Never loads it into memory."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(block)
            if not piece:
                break
            size += len(piece)
            digest.update(piece)
    return size, digest.hexdigest()


def stage_to_local(args) -> Path:
    """Copy the VERIFIED `chunks.jsonl` to local SSD, then verify the copy.

    Google Drive is a network filesystem; streaming 2.2 GB of JSON lines from it
    row by row is dominated by I/O latency, not by the length computation this
    diagnostic is actually measuring. Staging onto Colab's local SSD removes that
    from the measurement.

    The staging is verified at both ends, and the Drive source is opened
    read-only and never written:

    1. `verify_prepared_corpus` must succeed on the Drive prepared corpus, so the
       expected size and sha256 come from the re-hashed COMPLETE marker rather
       than from a filename;
    2. the source file's own size and sha256 are recomputed and must match;
    3. the copy is made;
    4. the LOCAL file's size and sha256 are recomputed and must match the source.

    Any mismatch aborts with `DIAGNOSTIC_FAILURE`. The local copy is a
    **disposable cache**: it holds no scientific authority, it is never written
    back, and deleting it costs only the copy time.
    """
    from unmark.stage1.checkpoint import verify_prepared_corpus

    prepared = Path(args.prepared_corpus)
    completion = (Path(args.completion_dir) if getattr(args, "completion_dir", None)
                  else prepared / "_checkpoint")
    verified = verify_prepared_corpus(prepared, completion)
    expected = verified.artifacts.get(CHUNKS_NAME)
    if expected is None:
        raise SystemExit(
            f"REFUSED: the completion marker does not bind {CHUNKS_NAME}; refusing "
            "to stage an artifact whose identity is not verified"
        )
    expected_size, expected_digest = expected

    source = prepared / CHUNKS_NAME
    print(f"staging: verifying source {source}", file=sys.stderr, flush=True)
    source_size, source_digest = sha256_of(source)
    if (source_size, source_digest) != (expected_size, expected_digest):
        raise SystemExit(
            f"REFUSED: {source} does not match the verified corpus identity: "
            f"{source_size} bytes / {source_digest[:16]}... != "
            f"{expected_size} / {expected_digest[:16]}..."
        )

    target_dir = Path(args.stage_local)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / CHUNKS_NAME
    if target.is_file():
        size, digest = sha256_of(target)
        if (size, digest) == (expected_size, expected_digest):
            print(f"staging: reusing verified local copy {target}",
                  file=sys.stderr, flush=True)
            return target
        print("staging: local copy does not verify; recopying",
              file=sys.stderr, flush=True)

    print(f"staging: copying {expected_size / 1e9:.2f} GB -> {target}",
          file=sys.stderr, flush=True)
    started = time.monotonic()
    shutil.copyfile(source, target)          # read-only on the source
    elapsed = time.monotonic() - started

    local_size, local_digest = sha256_of(target)
    if (local_size, local_digest) != (expected_size, expected_digest):
        raise SystemExit(
            f"REFUSED: the staged copy does not match the source: {local_size} "
            f"bytes / {local_digest[:16]}... != {expected_size} / "
            f"{expected_digest[:16]}...  The local cache is disposable -- delete "
            f"{target} and retry."
        )
    throughput = (expected_size / 1e6 / elapsed) if elapsed else 0.0
    print(f"staging: VERIFIED local copy in {elapsed:.1f}s "
          f"({throughput:.0f} MB/s), sha256 {local_digest[:16]}...",
          file=sys.stderr, flush=True)
    return target


_WORKER: dict = {}


def _init_worker(revision: str) -> None:
    """One tokenizer, one classifier, one counter per process."""
    from unmark.linguistics import make_classifier, try_load_inventory

    tokenizer = load_tokenizer(revision)
    _WORKER["tokenizer"] = tokenizer
    _WORKER["classifier"] = make_classifier(try_load_inventory())
    _WORKER["unk"] = getattr(tokenizer, "unk_token_id", None)
    _WORKER["counter"] = RealisedLengthCounter(tokenizer, _WORKER["unk"])
    _, base_length, _t = stage6_length_functions(tokenizer)
    _WORKER["base_length"] = base_length


def _measure_batch(batch):
    """Measure one batch of `(line_index, row)`. Pure; returns plain data.

    Runs in a worker process. Only the rows of this batch cross the process
    boundary -- never the 2.2 GB corpus.
    """
    counter = _WORKER["counter"]
    base_length = _WORKER["base_length"]
    out = []
    for line_index, row, verify in batch:
        text = row["text"]
        realised = counter.length(text)
        persisted = row.get("base_length")
        recomputed = None
        # Fail-closed verification: recompute the authoritative length on a
        # deterministic sample, and ALWAYS for anything reported as an offender.
        if verify or realised > MAX_LENGTH or not isinstance(persisted, int):
            recomputed = authoritative_base_length(base_length, text)
        out.append({
            "line_index": line_index,
            "chunk_id": row.get("chunk_id"),
            "partition": row.get("partition"),
            "text_sha256_16": stable_id(text),
            "characters": len(text),
            "contains_newline": "\n" in text,
            "persisted": persisted,
            "recomputed": recomputed,
            "realised": realised,
        })
    return out


def _batches(rows, size, verify_every):
    """Stream `(index, row, verify)` triples in bounded batches."""
    batch = []
    for index, row in rows:
        batch.append((index, row, verify_every > 0 and index % verify_every == 0))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Aggregate:
    """Order-independent totals, so the report cannot depend on worker count."""

    def __init__(self) -> None:
        self.scanned = 0
        self.within = 0
        self.over = 0
        self.disagreements = 0
        self.max_authoritative = 0
        self.max_realised = 0
        self.verified = 0
        self.over_histogram: collections.Counter = collections.Counter()
        self.delta_histogram: collections.Counter = collections.Counter()
        self.offenders: list[dict] = []
        self.next_index = 0

    def absorb(self, measured, max_offenders: int) -> None:
        for item in measured:
            authoritative = (item["recomputed"] if item["recomputed"] is not None
                             else item["persisted"])
            realised = item["realised"]
            self.scanned += 1
            self.next_index = max(self.next_index, item["line_index"] + 1)
            self.max_realised = max(self.max_realised, realised)
            if isinstance(authoritative, int):
                self.max_authoritative = max(self.max_authoritative, authoritative)
                delta = realised - authoritative
                if delta:
                    self.disagreements += 1
                    self.delta_histogram[delta] += 1
            if item["recomputed"] is not None:
                self.verified += 1
            if realised > MAX_LENGTH:
                self.over += 1
                self.over_histogram[realised] += 1
                self.offenders.append({
                    "line_index": item["line_index"],
                    "chunk_id": item["chunk_id"],
                    "partition": item["partition"],
                    "text_sha256_16": item["text_sha256_16"],
                    "characters": item["characters"],
                    "contains_newline": item["contains_newline"],
                    "stage6_persisted_base_length": item["persisted"],
                    "stage6_recomputed_base_length": item["recomputed"],
                    "stage1_realised_base_length": realised,
                    "delta": realised - (item["recomputed"] if item["recomputed"]
                                         is not None else item["persisted"] or 0),
                })
                # Deterministic and bounded, independent of arrival order.
                self.offenders.sort(key=lambda o: o["line_index"])
                del self.offenders[max_offenders:]

    def state(self) -> dict:
        return {
            "scanned": self.scanned, "within": self.within, "over": self.over,
            "disagreements": self.disagreements, "verified": self.verified,
            "max_authoritative": self.max_authoritative,
            "max_realised": self.max_realised,
            "over_histogram": {str(k): v for k, v in self.over_histogram.items()},
            "delta_histogram": {str(k): v for k, v in self.delta_histogram.items()},
            "offenders": self.offenders, "next_index": self.next_index,
        }

    @classmethod
    def restore(cls, state: dict) -> "Aggregate":
        aggregate = cls()
        aggregate.scanned = state["scanned"]
        aggregate.within = state["within"]
        aggregate.over = state["over"]
        aggregate.disagreements = state["disagreements"]
        aggregate.verified = state.get("verified", 0)
        aggregate.max_authoritative = state["max_authoritative"]
        aggregate.max_realised = state["max_realised"]
        aggregate.over_histogram = collections.Counter(
            {int(k): v for k, v in state["over_histogram"].items()})
        aggregate.delta_histogram = collections.Counter(
            {int(k): v for k, v in state["delta_histogram"].items()})
        aggregate.offenders = list(state["offenders"])
        aggregate.next_index = state["next_index"]
        return aggregate


def diagnostic_identity(args) -> dict:
    """What a resumed run must match before it may continue."""
    return {
        "prepared_corpus": str(args.prepared_corpus),
        "partition": args.partition,
        "revision": args.revision,
        "max_length": MAX_LENGTH,
        "encoder_checkpoint": ENCODER_CHECKPOINT,
        "verify_every": args.verify_every,
    }


def scan(args) -> int:
    require_corpus(Path(args.prepared_corpus))

    identity = diagnostic_identity(args)
    state_path = Path(args.report + ".state.json") if args.report else None
    aggregate = Aggregate()
    if args.resume and state_path and state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            print("REFUSED: resume state describes a different diagnostic:\n"
                  f"  saved   {saved.get('identity')}\n  current {identity}",
                  file=sys.stderr)
            return EXIT_DIAGNOSTIC_FAILURE
        aggregate = Aggregate.restore(saved["aggregate"])
        print(f"resuming at row {aggregate.next_index} "
              f"({aggregate.scanned} already scanned)", file=sys.stderr, flush=True)

    skip_to = aggregate.next_index
    rows = ((i, r) for i, r
            in iter_chunks(Path(args.prepared_corpus), args.partition,
                           chunks_path_for(args))
            if i >= skip_to)
    if args.limit:
        rows = itertools.islice(rows, args.limit)
    work = _batches(rows, args.batch_size, args.verify_every)

    started = time.monotonic()
    failures: list[str] = []

    def absorb(measured):
        for item in measured:
            # Fail closed: a persisted length that disagrees with the real
            # Stage-6 function invalidates the whole shortcut.
            if item["recomputed"] is not None and isinstance(item["persisted"], int):
                if item["recomputed"] != item["persisted"]:
                    failures.append(
                        f"row {item['line_index']} chunk {item['chunk_id']!r}: persisted "
                        f"base_length {item['persisted']} != recomputed "
                        f"{item['recomputed']}"
                    )
        aggregate.absorb(measured, args.max_offenders)

    if args.workers and args.workers > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker,
                          initargs=(args.revision,)) as pool:
            for measured in pool.imap(_measure_batch, work, chunksize=1):
                absorb(measured)
                if failures:
                    break
                _progress(aggregate, args, started, state_path, identity)
    else:
        _init_worker(args.revision)
        for batch in work:
            absorb(_measure_batch(batch))
            if failures:
                break
            _progress(aggregate, args, started, state_path, identity)

    if failures:
        print("REFUSED: persisted Stage-6 base_length failed verification. The "
              "shortcut in §2 is not valid for this corpus:", file=sys.stderr)
        for line in failures[:10]:
            print(f"  {line}", file=sys.stderr)
        return EXIT_DIAGNOSTIC_FAILURE

    aggregate.within = aggregate.scanned - aggregate.over
    exit_code = EXIT_VIOLATION_FOUND if aggregate.over else EXIT_NO_VIOLATION
    elapsed = time.monotonic() - started
    report = {
        "status": STATUS_BY_EXIT[exit_code],
        "exit_code": exit_code,
        "identity": identity,
        "scanned": aggregate.scanned,
        "within_max_length": aggregate.within,
        "over_max_length": aggregate.over,
        "stage6_vs_stage1_disagreements": aggregate.disagreements,
        "authoritative_spot_checks": aggregate.verified,
        "max_stage6_authoritative_base_length": aggregate.max_authoritative,
        "max_stage1_realised_base_length": aggregate.max_realised,
        "over_length_histogram": dict(sorted(aggregate.over_histogram.items())),
        "delta_histogram": dict(sorted(aggregate.delta_histogram.items())),
        "offenders": aggregate.offenders,
        "workers": args.workers,
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(aggregate.scanned / elapsed, 1) if elapsed else None,
        "official_test_used": False,
        "raw_text_persisted": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr, flush=True)
        if state_path and state_path.is_file():
            state_path.unlink()
    return exit_code


def _progress(aggregate, args, started, state_path, identity) -> None:
    if args.progress and aggregate.scanned % args.progress < args.batch_size:
        elapsed = time.monotonic() - started
        rate = aggregate.scanned / elapsed if elapsed else 0
        # stderr, so stdout stays a single parseable JSON document.
        print(f"  scanned {aggregate.scanned}  over={aggregate.over}  "
              f"disagreements={aggregate.disagreements}  "
              f"{rate:.0f} rows/s", file=sys.stderr, flush=True)
    if (state_path and args.checkpoint_every
            and aggregate.scanned % args.checkpoint_every < args.batch_size):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"identity": identity, "aggregate": aggregate.state()},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")


BOUNDARY_MARGIN = 8
"""A chunk is "near-boundary" when its persisted `base_length` is within this
many tokens of `MAX_LENGTH`. These are the only chunks a +1 disagreement can
actually turn into a violation, so they are over-sampled deliberately."""

VALIDATION_STRATA = (("boundary", 0.30), ("newline", 0.40), ("ordinary", 0.30))
"""Quotas of `--validate-rows`, rarest stratum first.

Assignment is rarest-first -- a row that is both near-boundary and
newline-bearing counts as `boundary` -- because near-boundary rows are the rare
and interesting ones. Without this, the measured 92.6 % newline rate would crowd
them out entirely.
"""


class _DeterministicReservoir:
    """Bounded reservoir sampling with NO random state.

    The "random" index comes from a hash of the row's own `chunk_id`, so the
    selection is spread across the whole file, exactly bounded by `capacity`,
    and identical on every rerun over the same corpus. Reservoir sampling is
    what lets a single streaming pass sample uniformly without knowing the
    stratum's size in advance.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self.items: list = []
        self.seen = 0

    def offer(self, key: str, item) -> None:
        if self.capacity == 0:
            return
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        digest = hashlib.sha256(f"{key}|{self.seen}".encode("utf-8")).hexdigest()[:8]
        index = int(digest, 16) % self.seen
        if index < self.capacity:
            self.items[index] = item


def _sample_rows(args, wanted: int):
    """A BOUNDED, deterministic, STRATIFIED sample of real TRAIN rows.

    **`--validate-rows` is a hard cap on expensive work.** The expensive thing
    is the full production `project_text`, and it runs once per *selected* row
    only. Selection itself streams cheap metadata -- `json.loads`, a substring
    test and an integer compare -- over the partition, which is exactly the
    trade the brief permits.

    The bound is `wanted` selected rows, **plus at most one** extra: the known
    offender, which is always included when present even if sampling would have
    missed it. So expensive comparisons <= `--validate-rows` + 1.

    Why stratified. An earlier version took "every stride-th row plus every
    newline-bearing and every near-boundary row, stop at `wanted`". That was
    bounded but badly skewed: with the measured 92.6 % newline rate it filled
    5000 slots from the first ~5 400 lines of the file -- 2.7 % of the corpus,
    4 996 newline rows, **2** near-boundary rows, and the known offender missed
    entirely. Quotas plus reservoir sampling fix all three.
    """
    rows = iter_chunks(Path(args.prepared_corpus), args.partition, chunks_path_for(args))
    quotas = {name: max(1, int(wanted * share)) for name, share in VALIDATION_STRATA}
    reservoirs = {name: _DeterministicReservoir(size) for name, size in quotas.items()}
    counts = collections.Counter()
    offender = None
    stride = max(1, args.validate_stride)
    streamed = 0

    for index, row in rows:
        streamed += 1
        text = row["text"]
        persisted = row.get("base_length")
        chunk_id = str(row.get("chunk_id", index))

        if args.offender_hash and stable_id(text) == args.offender_hash:
            offender = (index, row)

        if isinstance(persisted, int) and persisted >= MAX_LENGTH - BOUNDARY_MARGIN:
            stratum = "boundary"
        elif "\n" in text:
            stratum = "newline"
        elif index % stride == 0:
            stratum = "ordinary"
        else:
            continue
        counts[stratum] += 1
        reservoirs[stratum].offer(chunk_id, (index, row))

        if args.validate_scan_limit and streamed >= args.validate_scan_limit:
            break

    picked = [item for reservoir in reservoirs.values() for item in reservoir.items]
    # Trim to the cap deterministically, then restore file order.
    picked.sort(key=lambda pair: pair[0])
    del picked[wanted:]
    if offender is not None and all(i != offender[0] for i, _ in picked):
        picked.append(offender)          # the ONE documented extra
        picked.sort(key=lambda pair: pair[0])

    selected_newline = sum(1 for _i, r in picked if "\n" in r["text"])
    selected_boundary = sum(
        1 for _i, r in picked
        if isinstance(r.get("base_length"), int)
        and r["base_length"] >= MAX_LENGTH - BOUNDARY_MARGIN
    )
    stats = {
        "streamed_rows": streamed,
        "selected": len(picked),
        "quotas": quotas,
        "available_per_stratum": dict(counts),
        "selected_newline_bearing": selected_newline,
        "selected_near_boundary": selected_boundary,
        "offender_forced_in": offender is not None,
    }
    return picked, selected_boundary, selected_newline, stats


def validate(args) -> int:
    """REAL tokenizer, REAL chunks: fast path == production path. Read-only.

    This is the correctness gate for the §9a optimisation. It loads the pinned
    PhoBERT tokenizer and compares, per chunk:

        RealisedLengthCounter.length(text)   (the optimised count-only path)
        realised_base_length(text, ...)      (the full production project_text)

    No model, no optimizer, no backward, no training. The offender measured in
    Audit 043 §7 is reconfirmed by hash if it appears in the sample.
    """
    from unmark.linguistics import make_classifier, try_load_inventory

    require_corpus(Path(args.prepared_corpus))
    if args.stage_local:
        args.local_chunks = str(stage_to_local(args))

    tokenizer = load_tokenizer(args.revision)
    classifier = make_classifier(try_load_inventory())
    unk = getattr(tokenizer, "unk_token_id", None)
    counter = RealisedLengthCounter(tokenizer, unk)
    _reference, base_length, _transforms = stage6_length_functions(tokenizer)

    picked, boundary, newline, selection = _sample_rows(args, args.validate_rows)
    print(f"validating {len(picked)} real chunks of {selection['streamed_rows']} "
          f"streamed ({newline} newline-bearing, {boundary} near-boundary)",
          file=sys.stderr, flush=True)

    compared = 0
    mismatches = []
    max_delta = 0
    offender = None
    started = time.monotonic()
    for index, row in picked:
        text = row["text"]
        fast = counter.length(text)
        exact = realised_base_length(text, tokenizer, classifier, unk)
        compared += 1
        if fast != exact:
            mismatches.append({
                "line_index": index, "chunk_id": row.get("chunk_id"),
                "text_sha256_16": stable_id(text), "fast": fast, "production": exact,
            })
        authoritative = authoritative_base_length(base_length, text)
        max_delta = max(max_delta, abs(exact - authoritative))
        if stable_id(text) == args.offender_hash:
            offender = {
                "line_index": index, "chunk_id": row.get("chunk_id"),
                "text_sha256_16": stable_id(text),
                "contains_newline": "\n" in text,
                "stage6_authoritative_base_length": authoritative,
                "stage1_realised_base_length": exact,
                "stage1_fast_base_length": fast,
                "delta": exact - authoritative,
                "matches_audit_043": (authoritative == 256 and exact == 257),
            }
        if args.progress and compared % args.progress == 0:
            print(f"  compared {compared}  mismatches {len(mismatches)}",
                  file=sys.stderr, flush=True)

    ok = not mismatches
    exit_code = EXIT_NO_VIOLATION if ok else EXIT_DIAGNOSTIC_FAILURE
    report = {
        "status": "SUCCESS_NO_VIOLATION" if ok else "DIAGNOSTIC_FAILURE",
        "mode": "validate",
        "exit_code": exit_code,
        "compared": compared,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "max_stage6_vs_stage1_delta": max_delta,
        "newline_bearing_sampled": newline,
        "near_boundary_sampled": boundary,
        "selection": selection,
        "expensive_comparison_cap": args.validate_rows + 1,
        "offender_reconfirmed": offender,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_test_used": False,
        "raw_text_persisted": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return exit_code


def aggregate_digest(report: dict) -> str:
    """Fingerprint of everything that must NOT depend on worker count."""
    material = {k: report[k] for k in (
        "scanned", "within_max_length", "over_max_length",
        "stage6_vs_stage1_disagreements", "max_stage6_authoritative_base_length",
        "max_stage1_realised_base_length", "over_length_histogram",
        "delta_histogram", "offenders",
    )}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def benchmark(args) -> int:
    """Time the SAME deterministic real subset at several worker counts.

    Reports rows/sec, CPU utilisation and peak RSS per configuration, checks
    that every aggregate digest is identical, and recommends the fastest
    configuration that actually completed. **The recommendation is measured, not
    assumed** -- 8 workers is not taken to be optimal a priori.
    """
    import copy
    import resource

    require_corpus(Path(args.prepared_corpus))
    if args.stage_local:
        args.local_chunks = str(stage_to_local(args))

    counts = [int(w) for w in args.benchmark_workers.split(",") if w.strip()]
    results = []
    for workers in counts:
        trial = copy.copy(args)
        trial.workers = workers
        trial.limit = args.benchmark_rows
        trial.report = None
        trial.resume = False
        trial.checkpoint_every = 0
        trial.progress = 0

        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        self_before = resource.getrusage(resource.RUSAGE_SELF)
        buffer = io.StringIO()
        started = time.monotonic()
        with contextlib.redirect_stdout(buffer):
            code = scan(trial)
        elapsed = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        self_after = resource.getrusage(resource.RUSAGE_SELF)

        if code == EXIT_DIAGNOSTIC_FAILURE:
            print(f"  workers={workers}: DIAGNOSTIC_FAILURE", file=sys.stderr)
            results.append({"workers": workers, "status": "DIAGNOSTIC_FAILURE"})
            continue
        report = json.loads(buffer.getvalue())
        cpu = ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
               + (self_after.ru_utime - self_before.ru_utime)
               + (self_after.ru_stime - self_before.ru_stime))
        peak = max(after.ru_maxrss, self_after.ru_maxrss) / 1024
        results.append({
            "workers": workers,
            "status": report["status"],
            "rows": report["scanned"],
            "elapsed_seconds": round(elapsed, 3),
            "rows_per_second": round(report["scanned"] / elapsed, 1) if elapsed else None,
            "cpu_seconds": round(cpu, 3),
            "cpu_utilisation": round(cpu / elapsed, 2) if elapsed else None,
            "peak_rss_mib": round(peak, 1),
            "aggregate_digest": aggregate_digest(report),
        })
        print(f"  workers={workers:<3} {results[-1]['rows_per_second']:>9} rows/s  "
              f"cpu={results[-1]['cpu_utilisation']}x  "
              f"rss={results[-1]['peak_rss_mib']:.0f}MiB  "
              f"digest={results[-1]['aggregate_digest']}", file=sys.stderr, flush=True)

    completed = [r for r in results if r.get("aggregate_digest")]
    digests = {r["aggregate_digest"] for r in completed}
    identical = len(digests) <= 1
    fastest = max(completed, key=lambda r: r["rows_per_second"] or 0) if completed else None
    report = {
        "status": ("SUCCESS_NO_VIOLATION" if identical and completed
                   else "DIAGNOSTIC_FAILURE"),
        "mode": "benchmark",
        "rows_per_configuration": args.benchmark_rows,
        "results": results,
        "all_digests_identical": identical,
        "distinct_digests": sorted(digests),
        "recommended_workers": fastest["workers"] if fastest else None,
        "recommended_rows_per_second": fastest["rows_per_second"] if fastest else None,
        "official_test_used": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return EXIT_NO_VIOLATION if identical and completed else EXIT_DIAGNOSTIC_FAILURE


def reproduce(args) -> int:
    """Walk the REAL sampler order and report the FIRST offending position.

    The offending sample is established from the sampler, not assumed from the
    last telemetry event: preparation may be dispatched in batches, so
    "position 32000 was the last reported cursor" does not by itself identify
    which sample raised.
    """
    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.execute import load_prepared_chunks
    from unmark.stage1.sampler import DeterministicSampler

    require_corpus(Path(args.prepared_corpus))
    tokenizer = load_tokenizer(args.revision)
    classifier = make_classifier(try_load_inventory())
    unk = getattr(tokenizer, "unk_token_id", None)
    _reference_length, base_length, _transforms = stage6_length_functions(tokenizer)
    counter = RealisedLengthCounter(tokenizer, unk)

    train_text, _dev = load_prepared_chunks(Path(args.prepared_corpus))
    sampler = DeterministicSampler(tuple(sorted(train_text)), seed=args.seed)
    print(f"sampler seed={args.seed} over {len(train_text)} train chunks, "
          f"batch={BATCH_SIZE}", flush=True)

    position = 0
    update = 0
    while update < args.max_updates:
        pairs = sampler.next_batch(BATCH_SIZE)
        update += 1
        for offset, (chunk_id, visit) in enumerate(pairs):
            text = train_text[chunk_id]
            # Fast scan; the moment something looks like an offender it is
            # re-measured with the FULL production path, so the reported number
            # is never the shortcut's.
            realised = counter.length(text)
            if realised > MAX_LENGTH:
                exact = realised_base_length(text, tokenizer, classifier, unk)
                if exact != realised:
                    print(json.dumps({
                        "status": "DIAGNOSTIC_FAILURE",
                        "reason": "fast path disagreed with the production path",
                        "fast": realised, "production": exact,
                        "chunk_id": chunk_id,
                    }, indent=2, sort_keys=True), file=sys.stderr)
                    return EXIT_DIAGNOSTIC_FAILURE
                realised = exact
                authoritative = authoritative_base_length(base_length, text)
                print(json.dumps({
                    "status": STATUS_BY_EXIT[EXIT_VIOLATION_FOUND],
                    "exit_code": EXIT_VIOLATION_FOUND,
                    "first_offence": True,
                    "would_fail_at_update": update,
                    "batch_offset": offset,
                    "sampler_position": position + offset,
                    "visit": visit,
                    "chunk_id": chunk_id,
                    "text_sha256_16": stable_id(text),
                    "characters": len(text),
                    "contains_newline": "\n" in text,
                    "stage1_realised_base_length": realised,
                    "stage6_authoritative_base_length": authoritative,
                    "delta": realised - authoritative,
                    "max_length": MAX_LENGTH,
                }, indent=2, sort_keys=True))
                return EXIT_VIOLATION_FOUND
        position += len(pairs)
        if args.progress and update % args.progress == 0:
            print(f"  update {update} position {position} clean", flush=True)

    print(json.dumps({
        "status": STATUS_BY_EXIT[EXIT_NO_VIOLATION],
        "exit_code": EXIT_NO_VIOLATION,
        "first_offence": False,
        "updates_walked": args.max_updates,
        "sample_visits": position,
    }, indent=2, sort_keys=True))
    return EXIT_NO_VIOLATION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-corpus", required=True)
    parser.add_argument("--partition", default="train", choices=["train", "dev", "all"])
    parser.add_argument("--revision", default=ENCODER_REVISION)
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole partition")
    parser.add_argument("--progress", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the scope scan. 1 = in-process. A "
                             "conservative default: raise it explicitly (Colab "
                             "commonly has 2-8 usable cores). Results are "
                             "independent of this value.")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="rows per work unit; bounds worker memory")
    parser.add_argument("--verify-every", type=int, default=1000,
                        help="recompute the authoritative Stage-6 base_length on "
                             "every Nth row and require it to equal the persisted "
                             "value. 0 disables the shortcut check (not advised). "
                             "Offenders are ALWAYS recomputed regardless.")
    parser.add_argument("--checkpoint-every", type=int, default=100_000,
                        help="persist diagnostic aggregates every N rows next to "
                             "--report, so an interrupt does not restart at zero")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the persisted diagnostic state, after "
                             "verifying it describes this same diagnostic")
    parser.add_argument("--max-offenders", type=int, default=50)
    parser.add_argument("--report", default=None)
    parser.add_argument("--stage-local", default=None, metavar="DIR",
                        help="stage the VERIFIED chunks.jsonl onto local SSD "
                             "(e.g. /content/unmark-stage1-diagnostics) and scan "
                             "that copy. Source size+sha256 are verified before "
                             "the copy and the local copy is verified after it. "
                             "The Drive source is opened read-only and never "
                             "written; the local copy is a disposable cache.")
    parser.add_argument("--completion-dir", default=None,
                        help="COMPLETE marker directory; defaults to "
                             "<prepared-corpus>/_checkpoint")
    parser.add_argument("--local-chunks", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true",
                        help="REAL-tokenizer correctness gate: optimised fast path "
                             "vs full production project_text on real chunks")
    parser.add_argument("--validate-rows", type=int, default=5000)
    parser.add_argument("--validate-scan-limit", type=int, default=0,
                        help="stop STREAMING metadata after N rows (0 = the whole "
                             "partition). Does not change the expensive-work cap, "
                             "which is --validate-rows.")
    parser.add_argument("--validate-stride", type=int, default=97,
                        help="deterministic sampling stride; every newline-bearing "
                             "and near-boundary row is also taken")
    parser.add_argument("--offender-hash", default="7d99f2dba18e45c0",
                        help="the Audit 043 §7 offender, reconfirmed if sampled")
    parser.add_argument("--benchmark", action="store_true",
                        help="time one fixed real subset at several worker counts")
    parser.add_argument("--benchmark-rows", type=int, default=50_000)
    parser.add_argument("--benchmark-workers", default="1,2,4,8")
    parser.add_argument("--reproduce", action="store_true",
                        help="walk the real sampler order to the first offence")
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--max-updates", type=int, default=1000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition == "all":
        args.partition = None
    if args.validate:
        return validate(args)
    if args.benchmark:
        return benchmark(args)
    if args.reproduce:
        return reproduce(args)
    if args.stage_local:
        args.local_chunks = str(stage_to_local(args))
    return scan(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
