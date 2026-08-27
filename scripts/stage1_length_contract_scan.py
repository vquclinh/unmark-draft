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

  `project_text` tokenizes per `whitespace_chunks`. That unit was plain
  ``\\S+`` until the Audit 044 repair, while the tokenizer's own unit is
  ``\\S+\\n?`` (`PHOBERT_RUN`) -- so on newline-bearing chunks the two grids
  disagreed. **Both now use** ``\\S+\\n?``, so this scanner is the post-repair
  acceptance gate: it should now report `over_max_length = 0` and
  `realised == authoritative` on every spot check.

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
    * it uses `whitespace_chunks` -- Stage-1's own run unit, whatever that
      currently is -- so it measures the shipped behaviour rather than a
      corrected model of it. Post-Audit-044 that unit is `\S+\n?`;
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
    """Measure one batch of `(line_index, row, spot_check)`. Pure; plain data out.

    **The prepared row schema does not persist `base_length`.**
    `unmark/stage1/checkpoint.py::chunk_record` writes exactly `chunk_id`,
    `document_id`, `partition`, `chunk_index`, `text`, `source_start`,
    `source_end`, `source_shard` -- the lengths are computed by the chunker,
    used by its guard, and discarded. An earlier version of this scanner read
    `row.get("base_length")`; on the real artifact that is always `None`, which
    silently emptied the near-boundary stratum and forced an authoritative
    recomputation on every row (Audit 043 §9e). Nothing here reads it now.

    So the authoritative Stage-6 length is computed **only** where it is
    scientifically needed:

    * for every **offender** (`realised > MAX_LENGTH`), to confirm Stage-6
      admitted it at `<= MAX_LENGTH` and to record the true delta; and
    * on a deterministic **spot-check** sample, which re-tests the Stage-6
      guarantee rather than assuming it.

    An offender additionally gets the FULL production `project_text` length, and
    the fast and full values must agree -- so no violation is ever reported on
    the shortcut's word alone.

    Runs in a worker process; only this batch's rows cross the boundary.
    """
    counter = _WORKER["counter"]
    base_length = _WORKER["base_length"]
    tokenizer = _WORKER["tokenizer"]
    classifier = _WORKER["classifier"]
    unk = _WORKER["unk"]
    out = []
    for line_index, row, spot_check in batch:
        text = row["text"]
        realised = counter.length(text)
        over = realised > MAX_LENGTH
        authoritative = None
        full = None
        if over:
            full = realised_base_length(text, tokenizer, classifier, unk)
            authoritative = authoritative_base_length(base_length, text)
        elif spot_check:
            authoritative = authoritative_base_length(base_length, text)
        out.append({
            "line_index": line_index,
            "chunk_id": row.get("chunk_id"),
            "document_id": row.get("document_id"),
            "partition": row.get("partition"),
            "text_sha256_16": stable_id(text),
            "characters": len(text),
            "contains_newline": "\n" in text,
            "realised": realised,
            "full_project_text": full,
            "authoritative": authoritative,
            "spot_checked": bool(spot_check) and not over,
        })
    return out


def _batches(rows, size, verify_every):
    """Stream `(index, row, spot_check)` triples in bounded batches."""
    batch = []
    for index, row in rows:
        batch.append((index, row, verify_every > 0 and index % verify_every == 0))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Aggregate:
    """Order-independent totals, so the report cannot depend on worker count.

    The MUST-HAVE scientific quantity is the exact **overflow scope**: how many
    TRAIN chunks have Stage-1 realised base length > `MAX_LENGTH`. That is what
    `over`, `realised_histogram`, `max_realised` and `offenders` carry, and it
    is computed for **every** row.

    `spot_check_disagreements` is a *sample* statistic over the deterministic
    spot-check subset only -- it is explicitly NOT an all-row Stage-6-vs-Stage-1
    delta histogram, because obtaining that would require authoritative Stage-6
    tokenization on all 2.6 million rows for a quantity that does not change the
    repair decision (Audit 043 §9f).
    """

    def __init__(self) -> None:
        self.scanned = 0
        self.within = 0
        self.over = 0
        self.spot_checked = 0
        self.spot_check_disagreements = 0
        self.max_realised = 0
        self.max_authoritative_seen = 0
        self.realised_histogram: collections.Counter = collections.Counter()
        self.over_histogram: collections.Counter = collections.Counter()
        self.spot_delta_histogram: collections.Counter = collections.Counter()
        self.offenders: list[dict] = []
        self.contract_failures: list[str] = []
        self.next_index = 0

    def absorb(self, measured, max_offenders: int) -> None:
        for item in measured:
            realised = item["realised"]
            self.scanned += 1
            self.next_index = max(self.next_index, item["line_index"] + 1)
            self.max_realised = max(self.max_realised, realised)
            self.realised_histogram[realised] += 1

            if item["spot_checked"] and isinstance(item["authoritative"], int):
                self.spot_checked += 1
                self.max_authoritative_seen = max(self.max_authoritative_seen,
                                                  item["authoritative"])
                delta = realised - item["authoritative"]
                if delta:
                    self.spot_check_disagreements += 1
                    self.spot_delta_histogram[delta] += 1
                if item["authoritative"] > MAX_LENGTH:
                    # Stage-6's own guard should make this impossible.
                    self.contract_failures.append(
                        f"row {item['line_index']} chunk {item['chunk_id']!r}: "
                        f"authoritative Stage-6 length {item['authoritative']} > "
                        f"{MAX_LENGTH}, contradicting the Stage-6 completion guarantee"
                    )

            if realised > MAX_LENGTH:
                self.over += 1
                self.over_histogram[realised] += 1
                full = item["full_project_text"]
                authoritative = item["authoritative"]
                if full is not None and full != realised:
                    self.contract_failures.append(
                        f"row {item['line_index']} chunk {item['chunk_id']!r}: fast "
                        f"realised {realised} != full project_text {full}"
                    )
                if isinstance(authoritative, int) and authoritative > MAX_LENGTH:
                    self.contract_failures.append(
                        f"row {item['line_index']} chunk {item['chunk_id']!r}: Stage-6 "
                        f"admitted a chunk at {authoritative} > {MAX_LENGTH}"
                    )
                self.offenders.append({
                    "line_index": item["line_index"],
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "partition": item["partition"],
                    "text_sha256_16": item["text_sha256_16"],
                    "characters": item["characters"],
                    "contains_newline": item["contains_newline"],
                    "stage6_authoritative_base_length": authoritative,
                    "stage1_fast_base_length": realised,
                    "stage1_full_project_text_length": full,
                    "delta": (realised - authoritative)
                             if isinstance(authoritative, int) else None,
                })
                self.offenders.sort(key=lambda o: o["line_index"])
                del self.offenders[max_offenders:]
            else:
                self.within += 1

    def state(self) -> dict:
        return {
            "scanned": self.scanned, "within": self.within, "over": self.over,
            "spot_checked": self.spot_checked,
            "spot_check_disagreements": self.spot_check_disagreements,
            "max_realised": self.max_realised,
            "max_authoritative_seen": self.max_authoritative_seen,
            "realised_histogram": {str(k): v for k, v in self.realised_histogram.items()},
            "over_histogram": {str(k): v for k, v in self.over_histogram.items()},
            "spot_delta_histogram": {str(k): v
                                     for k, v in self.spot_delta_histogram.items()},
            "offenders": self.offenders,
            "contract_failures": self.contract_failures,
            "next_index": self.next_index,
        }

    @classmethod
    def restore(cls, state: dict) -> "Aggregate":
        aggregate = cls()
        aggregate.scanned = state["scanned"]
        aggregate.within = state["within"]
        aggregate.over = state["over"]
        aggregate.spot_checked = state.get("spot_checked", 0)
        aggregate.spot_check_disagreements = state.get("spot_check_disagreements", 0)
        aggregate.max_realised = state["max_realised"]
        aggregate.max_authoritative_seen = state.get("max_authoritative_seen", 0)
        aggregate.realised_histogram = collections.Counter(
            {int(k): v for k, v in state.get("realised_histogram", {}).items()})
        aggregate.over_histogram = collections.Counter(
            {int(k): v for k, v in state["over_histogram"].items()})
        aggregate.spot_delta_histogram = collections.Counter(
            {int(k): v for k, v in state.get("spot_delta_histogram", {}).items()})
        aggregate.offenders = list(state["offenders"])
        aggregate.contract_failures = list(state.get("contract_failures", []))
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

    def absorb(measured):
        aggregate.absorb(measured, args.max_offenders)

    if args.workers and args.workers > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker,
                          initargs=(args.revision,)) as pool:
            for measured in pool.imap(_measure_batch, work, chunksize=1):
                absorb(measured)
                if aggregate.contract_failures:
                    break
                _progress(aggregate, args, started, state_path, identity)
    else:
        _init_worker(args.revision)
        for batch in work:
            absorb(_measure_batch(batch))
            if aggregate.contract_failures:
                break
            _progress(aggregate, args, started, state_path, identity)

    if aggregate.contract_failures:
        print("REFUSED: a Stage-6/Stage-1 contract guarantee failed:", file=sys.stderr)
        for line in aggregate.contract_failures[:10]:
            print(f"  {line}", file=sys.stderr)
        return EXIT_DIAGNOSTIC_FAILURE

    exit_code = EXIT_VIOLATION_FOUND if aggregate.over else EXIT_NO_VIOLATION
    elapsed = time.monotonic() - started
    report = {
        "status": STATUS_BY_EXIT[exit_code],
        "exit_code": exit_code,
        "identity": identity,
        "line_index_base": 0,
        # --- MUST-HAVE: the exact overflow scope, over EVERY row ---
        "scanned": aggregate.scanned,
        "within_max_length": aggregate.within,
        "over_max_length": aggregate.over,
        "max_stage1_realised_base_length": aggregate.max_realised,
        "over_length_histogram": dict(sorted(aggregate.over_histogram.items())),
        "realised_length_histogram": dict(sorted(aggregate.realised_histogram.items())),
        "offenders": aggregate.offenders,
        # --- contract guard: a deterministic SAMPLE, not an all-row statistic ---
        "authoritative_spot_checks": aggregate.spot_checked,
        "spot_check_disagreements": aggregate.spot_check_disagreements,
        "spot_check_delta_histogram": dict(sorted(aggregate.spot_delta_histogram.items())),
        "max_stage6_authoritative_seen": aggregate.max_authoritative_seen,
        "all_row_stage6_delta_histogram": None,
        "all_row_delta_histogram_note": (
            "not computed: the prepared row schema does not persist base_length, so "
            "an all-row Stage-6-vs-Stage-1 histogram would require authoritative "
            "tokenization on every row for a quantity that does not change the "
            "repair decision. Overflow scope above is exact and complete."
        ),
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
              f"spot_checks={aggregate.spot_checked}  "
              f"{rate:.0f} rows/s", file=sys.stderr, flush=True)
    if (state_path and args.checkpoint_every
            and aggregate.scanned % args.checkpoint_every < args.batch_size):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"identity": identity, "aggregate": aggregate.state()},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")


BOUNDARY_MARGIN = 8
"""A chunk is "near-boundary" when its Stage-1 **realised** length is within
this many tokens of `MAX_LENGTH`. These are the only chunks a +1 disagreement
can turn into a violation, so they are over-sampled deliberately."""

CHARACTER_PREFILTER = 200
"""Cheap NECESSARY condition for a row to be near-boundary or overflowing.

Computing the fast realised length for all 2.6 M rows during *selection* would
cost as much as the scan itself, so selection first applies a character filter
and only measures rows that pass it. The filter is sound, not heuristic:

* every BPE token covers at least one character, so
  `content_tokens <= len(base_text)`;
* `base_text` is a per-character mapping of `canon(text)` with combining marks
  removed, so `len(base_text) <= len(canon(text)) <= len(text)`;
* therefore `realised = content_tokens + specials <= len(text) + specials`.

A row with `realised >= MAX_LENGTH - BOUNDARY_MARGIN` (248) therefore needs at
least ~246 characters. 200 leaves a wide margin, so **no near-boundary or
overflowing row can be filtered out**.
"""

VALIDATION_STRATA = (("overflow", 0.15), ("boundary", 0.25), ("newline", 0.35),
                     ("ordinary", 0.25))
"""Quotas of `--validate-rows`, rarest stratum first.

Assignment is rarest-first, so a row that is both overflowing and
newline-bearing counts as `overflow`. Without this the measured 92.6 % newline
rate would crowd the interesting strata out entirely.

**`boundary` and `overflow` are keyed on the Stage-1 FAST realised length, not
on a persisted field.** The prepared row schema does not carry `base_length`
(Audit 043 §9e), and an earlier version that read `row.get("base_length")`
silently produced `selected_near_boundary = 0` on the real corpus.
"""


class _DeterministicReservoir:
    """Bounded reservoir sampling with NO random state.

    The "random" index comes from a hash of the row's own `chunk_id`, so the
    selection is spread across the whole file, exactly bounded by `capacity`,
    and identical on every rerun over the same corpus.
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


def _sample_rows(args, wanted: int, counter=None):
    """A BOUNDED, deterministic, STRATIFIED sample of real TRAIN rows.

    **`--validate-rows` is a hard cap on expensive work.** The expensive thing
    is the full production `project_text`, which `validate` runs once per
    *selected* row. Selection itself streams cheap metadata and measures the
    fast realised length only for rows passing `CHARACTER_PREFILTER`.

    The bound is `wanted` selected rows **plus at most one**: the known
    offender, always included when present even if sampling would have missed
    it. So expensive comparisons <= `--validate-rows` + 1.

    `line_index` is **0-based** throughout: it is the index of the row within
    the streamed partition, so a row reported at `line_index = N` is physical
    line `N + 1` under `enumerate(file, 1)` (Audit 043 §9g).
    """
    rows = iter_chunks(Path(args.prepared_corpus), args.partition, chunks_path_for(args))
    quotas = {name: max(1, int(wanted * share)) for name, share in VALIDATION_STRATA}
    reservoirs = {name: _DeterministicReservoir(size) for name, size in quotas.items()}
    counts = collections.Counter()
    offender = None
    stride = max(1, args.validate_stride)
    streamed = 0
    measured = 0

    for index, row in rows:
        streamed += 1
        text = row["text"]
        chunk_id = str(row.get("chunk_id", index))

        if args.offender_hash and stable_id(text) == args.offender_hash:
            offender = (index, row)

        realised = None
        if counter is not None and len(text) >= CHARACTER_PREFILTER:
            realised = counter.length(text)
            measured += 1

        if realised is not None and realised > MAX_LENGTH:
            stratum = "overflow"
        elif realised is not None and realised >= MAX_LENGTH - BOUNDARY_MARGIN:
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
    picked.sort(key=lambda pair: pair[0])
    del picked[wanted:]
    if offender is not None and all(i != offender[0] for i, _ in picked):
        picked.append(offender)
        picked.sort(key=lambda pair: pair[0])

    selected_newline = sum(1 for _i, r in picked if "\n" in r["text"])
    selected_near_boundary = 0
    if counter is not None:
        for _i, r in picked:
            if len(r["text"]) >= CHARACTER_PREFILTER:
                if counter.length(r["text"]) >= MAX_LENGTH - BOUNDARY_MARGIN:
                    selected_near_boundary += 1
    stats = {
        "selected_near_boundary": selected_near_boundary,
        "streamed_rows": streamed,
        "fast_length_measured_rows": measured,
        "character_prefilter": CHARACTER_PREFILTER,
        "selected": len(picked),
        "quotas": quotas,
        "available_per_stratum": dict(counts),
        "selected_newline_bearing": selected_newline,
        "offender_forced_in": offender is not None,
        "line_index_base": 0,
    }
    return picked, selected_near_boundary, selected_newline, stats


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

    picked, boundary, newline, selection = _sample_rows(
        args, args.validate_rows, counter=counter)
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
        # OBJECT or null -- never a boolean. A caller must test `is not None`
        # and read the evidence fields, not truthiness of a flag.
        "offender_reconfirmed": offender,
        "offender_reconfirmed_schema": "object-or-null",
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
        "max_stage1_realised_base_length", "over_length_histogram",
        "realised_length_histogram", "offenders",
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
