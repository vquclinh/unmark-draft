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
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
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
    """Stage-1's realised length: what `prepare_example` actually builds."""
    from unmark.stage1.data import project_text

    _base_text, content_ids, _projections = project_text(
        text, tokenizer, classifier, unk_token_id
    )
    return len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))


def iter_chunks(prepared: Path, partition: str | None):
    path = Path(prepared) / CHUNKS_NAME
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


def scan(args) -> int:
    from unmark.linguistics import make_classifier, try_load_inventory

    require_corpus(Path(args.prepared_corpus))

    tokenizer = load_tokenizer(args.revision)
    classifier = make_classifier(try_load_inventory())
    unk = getattr(tokenizer, "unk_token_id", None)
    _reference_length, base_length, _transforms = stage6_length_functions(tokenizer)

    scanned = 0
    within = 0
    over = 0
    disagreements = 0
    max_authoritative = 0
    max_realised = 0
    over_histogram: collections.Counter = collections.Counter()
    delta_histogram: collections.Counter = collections.Counter()
    offenders: list[dict] = []

    for index, row in iter_chunks(Path(args.prepared_corpus), args.partition):
        text = row["text"]
        realised = realised_base_length(text, tokenizer, classifier, unk)
        authoritative = authoritative_base_length(base_length, text)

        scanned += 1
        max_realised = max(max_realised, realised)
        max_authoritative = max(max_authoritative, authoritative)
        delta = realised - authoritative
        if delta:
            disagreements += 1
            delta_histogram[delta] += 1

        if realised > MAX_LENGTH:
            over += 1
            over_histogram[realised] += 1
            if len(offenders) < args.max_offenders:
                offenders.append({
                    "line_index": index,
                    "chunk_id": row.get("chunk_id"),
                    "partition": row.get("partition"),
                    "text_sha256_16": stable_id(text),
                    "characters": len(text),
                    "contains_newline": "\n" in text,
                    "stage6_recorded_base_length": row.get("base_length"),
                    "stage6_authoritative_base_length": authoritative,
                    "stage1_realised_base_length": realised,
                    "delta": delta,
                })
        else:
            within += 1

        if args.limit and scanned >= args.limit:
            break
        if args.progress and scanned % args.progress == 0:
            print(f"  scanned {scanned}  over={over}  disagreements={disagreements}",
                  flush=True)

    report = {
        "prepared_corpus": str(args.prepared_corpus),
        "partition": args.partition,
        "max_length": MAX_LENGTH,
        "encoder_checkpoint": ENCODER_CHECKPOINT,
        "encoder_revision": args.revision,
        "scanned": scanned,
        "within_max_length": within,
        "over_max_length": over,
        "stage6_vs_stage1_disagreements": disagreements,
        "max_stage6_authoritative_base_length": max_authoritative,
        "max_stage1_realised_base_length": max_realised,
        "over_length_histogram": dict(sorted(over_histogram.items())),
        "delta_histogram": dict(sorted(delta_histogram.items())),
        "offenders": offenders,
        "official_test_used": False,
        "raw_text_persisted": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        print(f"\nwrote {args.report}", flush=True)
    return 1 if over else 0


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
            realised = realised_base_length(text, tokenizer, classifier, unk)
            if realised > MAX_LENGTH:
                authoritative = authoritative_base_length(base_length, text)
                print(json.dumps({
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
                return 1
        position += len(pairs)
        if args.progress and update % args.progress == 0:
            print(f"  update {update} position {position} clean", flush=True)

    print(f"no offence in the first {args.max_updates} updates "
          f"({position} sample-visits)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-corpus", required=True)
    parser.add_argument("--partition", default="train", choices=["train", "dev", "all"])
    parser.add_argument("--revision", default=ENCODER_REVISION)
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole partition")
    parser.add_argument("--progress", type=int, default=100_000)
    parser.add_argument("--max-offenders", type=int, default=50)
    parser.add_argument("--report", default=None)
    parser.add_argument("--reproduce", action="store_true",
                        help="walk the real sampler order to the first offence")
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--max-updates", type=int, default=1000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition == "all":
        args.partition = None
    return reproduce(args) if args.reproduce else scan(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
