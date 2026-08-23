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
from unmark.stage1.chunking import chunk_corpus, verify_no_parent_spans_partitions  # noqa: E402
from unmark.stage1.manifest import (  # noqa: E402
    CHUNKS_NAME,
    build_manifest,
    load_manifest,
)
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
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
    if output.exists():
        print(f"REFUSED: {output} already exists; prepared corpora are immutable",
              file=sys.stderr)
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

    total = len(kept)
    every = max(1, total // 100)
    started = time.monotonic()
    # Early heartbeat first: at 1% the first line would arrive after ~11 000
    # documents, which is exactly how a severe slowdown stayed invisible for
    # 13 minutes. Counts and elapsed time only -- never corpus text.
    heartbeats = (1, 10, 50, 100, 500, 1_000, 5_000, 10_000)

    def report(done: int, chunks_so_far: int) -> None:
        if done in heartbeats or done % every == 0 or done == total:
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"    chunking {done}/{total} documents ({100 * done / total:.1f}%), "
                  f"{chunks_so_far} chunks, {elapsed:.1f}s, {rate:.0f} docs/s", flush=True)

    try:
        chunks = chunk_corpus(
            kept, partition.assignment,
            reference_length=reference_length, base_length=base_length,
            max_length=MAX_LENGTH, classifier=classifier, on_progress=report,
        )
    except Stage1ContractViolation as error:
        # Surface WHICH stage failed and on what, without dumping corpus text.
        # The original contract error is re-raised unchanged, never swallowed.
        print(f"\nREFUSED during CHUNKING (stage 5 of 6, after the document split):\n"
              f"  {error}", file=sys.stderr, flush=True)
        raise
    parents = verify_no_parent_spans_partitions(chunks)
    print(f"  chunked AFTER splitting: {len(chunks)} chunks from {parents} documents")
    counters = transforms.counters
    print(f"    length queries {counters.length_queries}, incremental extensions "
          f"{counters.incremental_extensions}, full rescans {counters.full_rescans}, "
          f"canon calls {counters.canon_calls}, tokenizer calls {counters.tokenizer_calls}, "
          f"composition verifications {counters.verifications}")

    manifest = build_manifest(
        source=source,
        contamination=contamination.to_dict(),
        partition=partition.to_dict(),
        chunks=chunks,
        overflow_count=0,
        base_invariance_violations=0,
    )
    output.mkdir(parents=True)
    manifest.write(output)
    with open(output / CHUNKS_NAME, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "partition": chunk.partition,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "source_start": chunk.source_start,
                "source_end": chunk.source_end,
                "source_shard": chunk.source_shard,
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {output}/ (manifest + {len(chunks)} chunks)")
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
def run_lr_pilot(args) -> int:
    manifest = load_manifest(Path(args.prepared_corpus))
    schedule = lr_pilot_schedule(SELECTION_SEED)
    print(f"LR pilot: {len(schedule)} runs, grid {list(LR_PILOT_GRID)}, r = {LR_PILOT_R}, "
          f"seed {SELECTION_SEED}")
    print(f"  corpus: {manifest['source']['revision']}")
    return _execute(args, schedule, "lr_pilot", manifest)


def run_r_phase1(args) -> int:
    manifest = load_manifest(Path(args.prepared_corpus))
    pilot = _load_selection(Path(args.lr_artifact), "lr_pilot")
    frozen = pilot["selected"]["learning_rate"]
    schedule = r_phase1_schedule(SELECTION_SEED, frozen)
    print(f"r Phase 1: {len(schedule)} runs, grid {list(R_PHASE1_GRID)}, frozen LR {frozen:g}, "
          f"seed {SELECTION_SEED}")
    return _execute(args, schedule, "r_phase1", manifest)


def run_final_main(args) -> int:
    manifest = load_manifest(Path(args.prepared_corpus))
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
    return _execute(args, schedule, "final_main", manifest)


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


def _execute(args, schedule, stage: str, manifest: dict) -> int:
    """Run a stage's schedule and persist its selection artifact."""
    from unmark.stage1.execute import execute_stage

    output = Path(args.output_dir)
    if output.exists():
        print(f"REFUSED: {output} already exists; run artifacts are immutable", file=sys.stderr)
        return 2
    return execute_stage(
        stage=stage,
        schedule=schedule,
        prepared_corpus=Path(args.prepared_corpus),
        manifest=manifest,
        output_dir=output,
        cache_root=Path(args.cache_root),
        revision=args.revision,
        repository_head=args.repository_head,
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
        revision=args.revision,
        repository_head=args.repository_head,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _corpus_consumer(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-corpus", required=True,
                        help="directory written by prepare-corpus (manifest-bound)")
    parser.add_argument("--output-dir", required=True, help="must not already exist")
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
    smoke.add_argument("--prepared-corpus", required=True)
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
