#!/usr/bin/env python3
"""Stage-1 execution stack — Colab runner.

Subcommands, in the order the protocol requires:

    prepare-corpus  pinned UVW-2026 bytes -> verified prepared-corpus artifact
    lr-pilot        3 runs, {1e-4, 3e-4, 1e-3} at r = 1, selection seed
    r-phase1        5 runs, r in {0.25, 0.5, 1, 2, 4} at the frozen LR
    final-main      3 runs at the selected LR and r, on the three train seeds
    smoke           no-update real-model integration check (PRE-TRAIN audit)

**Exactly 11 nominal scientific runs: 3 + 5 + 3.** A budget continuation is a
continuation of the SAME run, not a twelfth candidate. The three `final-main`
runs ARE the final main Stage-1 adapters; no training round follows them.

**There is no official UIT-VSFC TEST argument anywhere in this file.** The
contamination screen accepts only the two sources the pre-G1 protocol already
opened, and `Preg1Role` has no TEST member to name even if one tried.

**There are no scientific override flags.** No `--lr`, no `--r`, no
`--batch-size`, no `--epochs`, no `--pi-strip`, no corruption-scope override,
and no `--max-updates` that bypasses the locked budget. Every such value is
pinned in `unmark.stage1.protocol` and imported. A command-line override of a
precommitted constant is the hole the protocol exists to close.

`smoke` is **structurally incapable of an optimizer step**: it never constructs
an optimizer and never calls `.backward()`.

Nothing here downloads a model or a corpus on import. Torch, transformers and
pyarrow are imported lazily inside the run path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.stage1.contracts import (  # noqa: E402
    CorruptionRatePolicy,
    OverflowBehaviour,
    Stage1ContractViolation,
    TruncationPolicy,
)
from unmark.stage1.corpus import (  # noqa: E402
    CorpusContractViolation,
    check_schema,
    concatenate,
    load_pin,
    partition_documents,
    read_shard,
    screen_contamination,
    verify_corpus_root,
)
from unmark.linguistics import make_classifier, try_load_inventory  # noqa: E402
from unmark.stage1.parallel import (  # noqa: E402
    ordered_document_chunks,
    resolve_worker_count,
)
from unmark.stage1.checkpoint import (  # noqa: E402
    COMPLETE_NAME,
    Stage6Timings,
    resolve_repository_head,
    CheckpointIdentity,
    PrepareCheckpoint,
    atomic_write_bytes,
    concatenate_shards,
    document_sequence_digest,
    read_completion,
    stream_counts,
    write_completion_marker,
)
from unmark.stage1.manifest import (  # noqa: E402
    CHUNKS_NAME,
    MANIFEST_NAME,
    build_manifest_from_counts,
    load_manifest,
)
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
    CHUNK_SCHEMA_VERSION,
    CORPUS_DATASET,
    CORPUS_REVISION,
    CORPUS_SHARD_ORDER,
    CORRUPTION_SEED,
    DEV_DOCUMENTS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EVAL_EVERY_UPDATES,
    INITIAL_MAX_UPDATES,
    LR_PILOT_GRID,
    LR_PILOT_R,
    MAX_LENGTH,
    PI_STRIP,
    RAW_BASE_POLICY,
    R_PHASE1_GRID,
    SELECTION_SEED,
    SPLIT_SEED,
    STAGE1_PROTOCOL_VERSION,
    TOTAL_NOMINAL_RUNS,
    TRAIN_SEEDS,
    VALIDATION_CONDITIONS,
    VALIDATION_CORRUPTION_SEED,
    protocol_dict,
)
from unmark.stage1.selection import (  # noqa: E402
    final_main_schedule,
    lr_pilot_schedule,
    r_phase1_schedule,
    select_learning_rate,
    select_r,
)

# ---------------------------------------------------------------------------
# prepare-corpus
# ---------------------------------------------------------------------------
def run_prepare_corpus(args) -> int:
    """Pinned bytes -> verified prepared corpus. Verify BEFORE reading a row."""
    output = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output / "_checkpoint"
    if output.exists() and not (checkpoint_root / "state.json").is_file() \
            and not (checkpoint_root / COMPLETE_NAME).is_file():
        # Immutable unless this is a genuine resume: a checkpoint state or a
        # completion marker is the only thing that licenses reusing the directory.
        print(f"REFUSED: {output} already exists and holds no checkpoint; prepared "
              "corpora are immutable", file=sys.stderr)
        return 2

    print("[1/6] verifying the corpus pin", flush=True)
    pin = load_pin()
    print(f"corpus pin: {pin.dataset} @ {pin.revision}")
    source = verify_corpus_root(Path(args.corpus_root), pin)
    print("  all three shards verified: name, byte size and sha256")

    print("[2/6] reading and concatenating the three shards", flush=True)
    documents_by_shard = {}
    for name in CORPUS_SHARD_ORDER:
        path = Path(args.corpus_root) / name
        docs = read_shard(path, name)
        check_schema(("id", "content"), name)
        documents_by_shard[name] = docs
        print(f"  {name}: {len(docs)} documents")

    print("[3/6] schema + duplicate-id check", flush=True)
    documents = concatenate(documents_by_shard)
    print(f"  concatenated in locked order: {len(documents)} documents, ids unique")

    print("[4/6] contamination screen (exact/canonical, opened material only)", flush=True)
    reference: dict[str, list[str]] = {}
    if args.uitvsfc_derived_train:
        reference["uitvsfc_derived_train"] = _read_text_column(args.uitvsfc_derived_train)
    if args.uitvsfc_official_validation:
        reference["uitvsfc_official_validation"] = _read_text_column(
            args.uitvsfc_official_validation
        )
    total_documents = len(documents)
    step = max(1, total_documents // 100)

    def screen_progress(seen: int, candidates: int, matches: int) -> None:
        if seen % step == 0 or seen == total_documents:
            print(f"    contamination: {seen}/{total_documents} docs "
                  f"({100 * seen / total_documents:.1f}%), candidates={candidates}, "
                  f"matches={matches}", flush=True)

    kept, contamination = screen_contamination(
        documents, reference, on_progress=screen_progress
    )
    counters = contamination.counters
    print(f"  contamination screen ({contamination.method}): "
          f"{contamination.excluded_count} excluded of {len(documents)}")
    print(f"    length-guard skips {counters.length_guard_skips}, "
          f"prefilter checks {counters.cheap_prefilter_checks}, "
          f"candidates {counters.prefilter_candidates}, "
          f"full canon calls {counters.full_canon_calls_for_corpus_candidates}")
    print("  official TEST: SEALED, not opened, not screened")

    print("[5/6] document-level split (BEFORE chunking)", flush=True)
    partition = partition_documents([d.document_id for d in kept])
    print(f"  document split (seed {SPLIT_SEED}): "
          f"train {len(partition.train)}, dev {len(partition.dev)}")

    print("[6/6] deterministic pre-chunking", flush=True)
    tokenizer = _load_tokenizer(args.revision)
    reference_length, base_length, transforms = _length_functions(tokenizer)
    classifier = make_classifier(try_load_inventory())

    identity = CheckpointIdentity(
        # The ACTUAL HEAD of the executing tree, never a caller-supplied value:
        # a checkpoint written by commit A must not resume under commit B.
        repository_head=resolve_repository_head(),
        protocol_version=STAGE1_PROTOCOL_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        corpus_dataset=pin.dataset,
        corpus_revision=pin.revision,
        corpus_files=tuple((f["name"], f["bytes"], f["sha256"]) for f in source["files"]),
        tokenizer_checkpoint=ENCODER_CHECKPOINT,
        tokenizer_revision=args.revision,
        transformers_version=_transformers_version(),
        max_length=MAX_LENGTH,
        raw_base_policy=RAW_BASE_POLICY,
        split_seed=SPLIT_SEED,
        dev_documents=DEV_DOCUMENTS,
        contamination_method=contamination.method,
        contamination_excluded_count=contamination.excluded_count,
        document_sequence_digest=document_sequence_digest(
            [d.document_id for d in kept]
        ),
        partition_assignment_digest=partition.membership_digest,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output / "_checkpoint"
    already = read_completion(checkpoint_dir, output, identity)
    if already is not None:
        print(f"  ALREADY_COMPLETE: every artifact verified against {COMPLETE_NAME}")
        print(f"    {json.dumps(already['counts'], sort_keys=True)}")
        return 0

    checkpoint = PrepareCheckpoint(checkpoint_dir, identity, len(kept))
    state = checkpoint.begin()
    resuming = state.next_document_index > 0
    print(f"  {'RESUME' if resuming else 'START'}: "
          f"{state.next_document_index}/{len(kept)} documents already committed, "
          f"{state.chunks_total} chunks in {len(state.shards)} verified shards")
    _report_space(checkpoint_dir, output)

    total = len(kept)
    started = time.monotonic()
    heartbeats = (1, 10, 50, 100, 500, 1_000, 5_000, 10_000)
    produced = state.chunks_total
    every = max(1, total // 100)
    timings = checkpoint.timings

    workers = resolve_worker_count(args.prepare_workers)
    if workers > 1:
        print(f"  compute workers: {workers} (operational only -- output is "
              f"identical for any worker count)")

    def _tokenizer_factory():
        return _load_tokenizer(args.revision)

    try:
        stream = ordered_document_chunks(
            kept, partition.assignment,
            start_index=state.next_document_index,
            tokenizer_factory=_tokenizer_factory,
            workers=workers,
            max_length=MAX_LENGTH,
            serial_length_functions=(reference_length, base_length),
            classifier=classifier,
            on_wait=lambda seconds: setattr(
                timings, "collector_wait_seconds",
                timings.collector_wait_seconds + seconds,
            ),
        )
        compute_mark = time.monotonic()
        for index, document, chunks in stream:
            now = time.monotonic()
            timings.chunk_compute_seconds += now - compute_mark
            produced += checkpoint.add_document(index, document.document_id, chunks)
            compute_mark = time.monotonic()
            done = index + 1
            if done in heartbeats or done % every == 0 or done == total:
                elapsed = time.monotonic() - started
                rate = (done - state.next_document_index) / elapsed if elapsed > 0 else 0.0
                print(f"    chunking {done}/{total} documents ({100 * done / total:.1f}%), "
                      f"{produced} chunks, {elapsed:.1f}s, {rate:.1f} docs/s", flush=True)
        checkpoint.commit(force=True)
    except Stage1ContractViolation as error:
        checkpoint.commit(force=True)
        print(f"\nREFUSED during CHUNKING (stage 6 of 6, after the document split):\n"
              f"  {error}\n  committed prefix preserved: "
              f"{checkpoint.state.next_document_index}/{total} documents",
              file=sys.stderr, flush=True)
        raise

    print(f"  chunked AFTER splitting: {checkpoint.state.chunks_total} chunks from "
          f"{total} documents in {len(checkpoint.state.shards)} shards")
    counters = transforms.counters
    print(f"    length queries {counters.length_queries}, BPE run evaluations "
          f"{counters.bpe_run_evaluations}, run-cache hits {counters.run_cache_hits}, "
          f"evictions {counters.run_cache_evictions}, incremental appends "
          f"{counters.incremental_appends}, full fallbacks {counters.full_fallbacks}, "
          f"authoritative verifications {counters.authoritative_queries}")
    timings.stage6_total_seconds = time.monotonic() - started
    print(timings.report(), flush=True)
    print(f"    checkpoint: {checkpoint.commits} commits, "
          f"{checkpoint.checkpoint_bytes / 1e6:.1f} MB, "
          f"{checkpoint.checkpoint_seconds:.1f}s "
          f"({100 * checkpoint.checkpoint_seconds / max(1e-9, time.monotonic() - started):.2f}% of stage 6)")

    # --- finalisation: streaming, idempotent, COMPLETE written last -------
    finalise_started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    shard_paths = [checkpoint.shard_dir / sh.name for sh in checkpoint.state.shards]
    payload_bytes, payload_digest = concatenate_shards(shard_paths, output / CHUNKS_NAME)

    with open(output / CHUNKS_NAME, encoding="utf-8") as handle:
        counts = stream_counts(handle, checkpoint.staging / "finalise", timings)

    manifest = build_manifest_from_counts(
        source=source,
        contamination=contamination.to_dict(),
        partition=partition.to_dict(),
        chunks_total=counts.chunks_total,
        chunks_by_partition=counts.chunks_by_partition,
        parent_documents_total=counts.parent_documents_total,
        parent_documents_by_partition=counts.parent_documents_by_partition,
        chunk_membership_digest=counts.membership_digest,
        overflow_count=0,
        base_invariance_violations=0,
    )
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_size, manifest_digest = atomic_write_bytes(output / MANIFEST_NAME, manifest_bytes)

    write_completion_marker(
        checkpoint_dir,
        identity=identity,
        artifacts={
            CHUNKS_NAME: (payload_bytes, payload_digest),
            MANIFEST_NAME: (manifest_size, manifest_digest),
        },
        counts=manifest.to_dict()["counts"],
    )
    checkpoint.cleanup_staging()
    print(f"  finalisation {time.monotonic() - finalise_started:.1f}s")
    print(f"\nWrote {output}/ (manifest + {counts.chunks_total} chunks)")
    print("The prepared corpus contains text because it IS the training dataset; "
          "scientific reports carry ids, digests and counts only.")
    return 0


def _read_text_column(path: str) -> list[str]:
    """Read one already-opened UIT-VSFC text column. Never TEST."""
    import csv

    rows: list[str] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in ("text", "sentence", "content"):
                if key in row and row[key]:
                    rows.append(row[key])
                    break
    if not rows:
        raise CorpusContractViolation(f"no text column found in {path}")
    return rows


def _transformers_version() -> str | None:
    """Part of the operational identity: a different tokenizer library is a
    different run, even at the same model revision."""
    try:
        import transformers

        return transformers.__version__
    except Exception:  # noqa: BLE001 - absent in the ML-free local environment
        return None


def _report_space(checkpoint_dir: Path, output_dir: Path) -> None:
    """Print free space before a ten-hour run rather than after it fails."""
    for label, path in (("local output", output_dir), ("checkpoint", checkpoint_dir)):
        target = path
        while not target.exists() and target != target.parent:
            target = target.parent
        try:
            usage = shutil.disk_usage(target)
        except OSError:
            continue
        print(f"    {label} free: {usage.free / 1e9:.1f} GB of "
              f"{usage.total / 1e9:.1f} GB at {target}")
    rss = _resident_bytes()
    if rss is not None:
        print(f"    process RSS: {rss / 1e6:.0f} MB")


def _resident_bytes() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001 - not Linux, or unavailable
        return None


def _load_tokenizer(revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ENCODER_CHECKPOINT, revision=revision, use_fast=False)


def _length_functions(tokenizer):
    """Token counts INCLUDING special tokens, for both Stage-1 paths.

    Delegates to `unmark.stage1.lengths`, which memoises the orthographic
    transforms and composes token counts per non-whitespace run. The values are
    **identical** to canonicalising and tokenizing each candidate whole -- see
    §R -- and the composition is verified against the whole-string result on the
    first queries of every run.
    """
    from unmark.stage1.lengths import build_length_functions

    reference_length, base_length, transforms = build_length_functions(tokenizer)
    return reference_length, base_length, transforms


# ---------------------------------------------------------------------------
# The three scientific stages
# ---------------------------------------------------------------------------
def _verified_corpus(args):
    """Fail closed unless the prepared corpus is the exact completed artifact.

    Runs BEFORE any model load, so a truncated, swapped, foreign or unfinished
    prepared corpus can never reach training (Audit 030 F1).
    """
    from unmark.stage1.checkpoint import verify_prepared_corpus

    prepared = Path(args.prepared_corpus)
    completion = Path(args.completion_dir) if args.completion_dir else prepared / "_checkpoint"
    verified = verify_prepared_corpus(prepared, completion)
    print(f"  prepared corpus VERIFIED against {verified.completion_path}")
    for name, (size, digest) in sorted(verified.artifacts.items()):
        print(f"    {name}: {size} bytes, sha256 {digest[:12]}...")
    print(f"    chunk_membership_digest {verified.chunk_membership_digest}")
    return verified


def run_lr_pilot(args) -> int:
    verified = _verified_corpus(args)
    manifest = verified.manifest
    schedule = lr_pilot_schedule(SELECTION_SEED)
    print(f"LR pilot: {len(schedule)} runs, grid {list(LR_PILOT_GRID)}, r = {LR_PILOT_R}, "
          f"seed {SELECTION_SEED}")
    print(f"  corpus: {manifest['source']['revision']}")
    return _execute(args, schedule, "lr_pilot", verified)


def run_r_phase1(args) -> int:
    verified = _verified_corpus(args)
    manifest = verified.manifest
    pilot = _load_selection(Path(args.lr_artifact), "lr_pilot")
    frozen = pilot["selected"]["learning_rate"]
    schedule = r_phase1_schedule(SELECTION_SEED, frozen)
    print(f"r Phase 1: {len(schedule)} runs, grid {list(R_PHASE1_GRID)}, frozen LR {frozen:g}, "
          f"seed {SELECTION_SEED}")
    return _execute(args, schedule, "r_phase1", verified)


def run_final_main(args) -> int:
    verified = _verified_corpus(args)
    manifest = verified.manifest
    pilot = _load_selection(Path(args.lr_artifact), "lr_pilot")
    phase1 = _load_selection(Path(args.r_artifact), "r_phase1")
    lr = pilot["selected"]["learning_rate"]
    r = phase1["selected"]["r"]
    if phase1["selected"]["learning_rate"] != lr:
        raise Stage1ContractViolation(
            f"the r artifact was produced at LR {phase1['selected']['learning_rate']}, "
            f"but the LR artifact selected {lr}"
        )
    schedule = final_main_schedule(lr, r)
    print(f"FINAL MAIN Stage-1: {len(schedule)} runs, LR {lr:g}, r {r:g}, seeds {list(TRAIN_SEEDS)}")
    print("  these three adapters ARE the final main Stage-1 models; nothing follows them")
    return _execute(args, schedule, "final_main", verified)


def _load_selection(path: Path, expected_stage: str) -> dict:
    if not path.is_file():
        raise Stage1ContractViolation(f"selection artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("stage") != expected_stage:
        raise Stage1ContractViolation(
            f"{path} is a {artifact.get('stage')!r} artifact; {expected_stage!r} is required"
        )
    if artifact.get("protocol_version") != STAGE1_PROTOCOL_VERSION:
        raise Stage1ContractViolation(
            f"{path} was produced under protocol {artifact.get('protocol_version')!r}"
        )
    return artifact


def _execute(args, schedule, stage: str, verified) -> int:
    """Run a stage's schedule and persist its selection artifact."""
    from unmark.stage1.execute import execute_stage

    output = Path(args.output_dir)
    resuming = bool(getattr(args, "resume", False))
    if output.exists() and not resuming:
        print(f"REFUSED: {output} already exists; run artifacts are immutable. "
              "Pass --resume to continue an interrupted stage from its verified "
              "training checkpoints instead of deleting it.", file=sys.stderr)
        return 2
    if resuming and not output.exists():
        print(f"REFUSED: --resume was given but {output} does not exist; there is "
              "nothing to resume", file=sys.stderr)
        return 2
    return execute_stage(
        stage=stage,
        schedule=schedule,
        prepared_corpus=Path(args.prepared_corpus),
        verified=verified,
        output_dir=output,
        cache_root=Path(args.cache_root),
        revision=args.revision,
        repository_head=args.repository_head,
        resume=resuming,
    )


def run_smoke(args) -> int:
    """No-update real-model integration check. **Cannot step an optimizer.**

    Constructs no optimizer and calls no `.backward()`. It exists for the future
    PRE-TRAIN audit, and is deliberately a separate code path so that "no
    parameter was updated" is structural rather than a promise.
    """
    from unmark.stage1.execute import smoke_check

    return smoke_check(
        prepared_corpus=Path(args.prepared_corpus),
        completion_dir=Path(args.completion_dir) if args.completion_dir else None,
        revision=args.revision,
        repository_head=args.repository_head,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _prepared_corpus_inputs(parser: argparse.ArgumentParser) -> None:
    """The prepared corpus and the COMPLETE marker that binds it.

    **Every** consumer that verifies a prepared corpus takes exactly this pair,
    so it is declared once. The `smoke` subcommand previously declared only
    `--prepared-corpus` while `run_smoke` read `args.completion_dir`, and the
    real fifth-smoke runner died on `AttributeError` after argparse had already
    accepted the documented CLI (Audit 030 §AA). Sharing the declaration is what
    stops the parser and the handler drifting apart again.
    """
    parser.add_argument("--prepared-corpus", required=True,
                        help="directory written by prepare-corpus (manifest-bound)")
    parser.add_argument(
        "--completion-dir", default=None,
        help="directory holding COMPLETE.json for that prepared corpus. Defaults "
             "to <prepared-corpus>/_checkpoint, which is where prepare-corpus "
             "puts it when the two are co-located. A real run may persist the "
             "payload and the checkpoint under different roots, so this is "
             "explicit rather than inferred. The prepared corpus is verified "
             "against this marker before any model is loaded (Audit 030 F1).",
    )


def _corpus_consumer(parser: argparse.ArgumentParser) -> None:
    _prepared_corpus_inputs(parser)
    parser.add_argument("--output-dir", required=True,
                        help="must not already exist, unless --resume is given")
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted stage from its per-run training "
             "checkpoints. WITHOUT this flag an existing --output-dir is "
             "REFUSED, so a fresh run can never overwrite scientific evidence; "
             "WITH it the directory must already exist and every checkpoint "
             "found is verified against this run's identity (seed, LR, r, "
             "corpus digest, repository HEAD) before it is used. Nothing is "
             "auto-resumed: a stage resumes only when asked.",
    )
    parser.add_argument("--cache-root", required=True, help="representation/tokenizer cache root")
    parser.add_argument("--revision", default=ENCODER_REVISION,
                        help="encoder revision; defaults to the pinned revision and is "
                             "validated against it")
    parser.add_argument("--repository-head", default=None,
                        help="commit sha recorded as provenance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-corpus", help="verify the pin and build the corpus")
    prepare.add_argument("--corpus-root", required=True,
                         help="directory holding the three pinned UVW parquet shards")
    prepare.add_argument("--output-dir", required=True, help="must not already exist")
    prepare.add_argument("--uitvsfc-derived-train", default=None,
                         help="already-opened UIT-VSFC derived TRAIN csv, for the exact "
                              "contamination screen")
    prepare.add_argument("--uitvsfc-official-validation", default=None,
                         help="already-opened UIT-VSFC official VALIDATION csv, for the "
                              "exact contamination screen")
    prepare.add_argument("--revision", default=ENCODER_REVISION,
                         help="tokenizer revision; defaults to the pinned revision")
    prepare.add_argument(
        "--prepare-workers", type=int, default=None,
        help="OPERATIONAL ONLY: how many processes compute chunks in parallel "
             "(default 1). Workers compute document-local chunks and nothing "
             "else; the main process alone serialises, accumulates membership, "
             "and owns the checkpoint. Output is byte-identical for any worker "
             "count -- asserted for 1/2/4/8 in tests. Changes no scientific "
             "value, no seed, no ordering, no chunk id.",
    )
    prepare.add_argument(
        "--checkpoint-dir", default=None,
        help="OPERATIONAL ONLY: durable directory for Stage-6 resume state and "
             "immutable shards (a Drive mount is fine). Defaults to "
             "<output-dir>/_checkpoint. Resume is automatic: an existing valid "
             "checkpoint continues from its committed prefix, a verified "
             "completion marker short-circuits. Changes no scientific value.",
    )

    pilot = sub.add_parser("lr-pilot", help="3 runs over the locked LR grid at r = 1")
    _corpus_consumer(pilot)

    phase1 = sub.add_parser("r-phase1", help="5 runs over the locked r grid at the frozen LR")
    _corpus_consumer(phase1)
    phase1.add_argument("--lr-artifact", required=True,
                        help="lr-pilot selection artifact; the LR is verified, not supplied")

    final = sub.add_parser("final-main", help="the 3 FINAL MAIN Stage-1 runs")
    _corpus_consumer(final)
    final.add_argument("--lr-artifact", required=True, help="lr-pilot selection artifact")
    final.add_argument("--r-artifact", required=True, help="r-phase1 selection artifact")

    smoke = sub.add_parser("smoke", help="no-update real-model check; cannot step an optimizer")
    _prepared_corpus_inputs(smoke)
    smoke.add_argument("--revision", default=ENCODER_REVISION)
    smoke.add_argument("--repository-head", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if getattr(args, "revision", ENCODER_REVISION) != ENCODER_REVISION:
            raise Stage1ContractViolation(
                f"--revision must be the locked backbone revision {ENCODER_REVISION}, got "
                f"{args.revision}. D-B3B0-007 locks it; a different backbone needs its own "
                "recorded position-id evidence."
            )
    except Stage1ContractViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"UNMARK Stage-1 — {STAGE1_PROTOCOL_VERSION}")
    print(f"  command        : {args.command}")
    print(f"  backbone       : {ENCODER_CHECKPOINT} @ {ENCODER_REVISION} (frozen)")
    print(f"  corpus         : {CORPUS_DATASET} @ {CORPUS_REVISION}")
    print(f"  max_length     : {MAX_LENGTH} | batch {BATCH_SIZE} | eval every {EVAL_EVERY_UPDATES}")
    print(f"  budget         : {INITIAL_MAX_UPDATES} updates, one continuation, then STOP")
    print(f"  corruption     : p ~ U(0,1), redraw per visit, pi_strip {PI_STRIP}, "
          f"seed {CORRUPTION_SEED}")
    print(f"  validation     : {list(VALIDATION_CONDITIONS)}, seed {VALIDATION_CORRUPTION_SEED}")
    print(f"  dev documents  : {DEV_DOCUMENTS} | split seed {SPLIT_SEED}")
    print(f"  run plan       : {TOTAL_NOMINAL_RUNS} nominal runs (3 + 5 + 3)")
    print("  official UIT-VSFC TEST : SEALED — no argument, no route\n")

    handlers = {
        "prepare-corpus": run_prepare_corpus,
        "lr-pilot": run_lr_pilot,
        "r-phase1": run_r_phase1,
        "final-main": run_final_main,
        "smoke": run_smoke,
    }
    try:
        return handlers[args.command](args)
    except Stage1ContractViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
