#!/usr/bin/env python3
"""Pre-G1 frozen-encoder burden diagnostic — Colab runner.

Two subcommands, in the order the protocol requires:

    tune       sweep the precommitted LR grid on **VANILLA ONLY**, on
               protocol-dev, and freeze the winner
    measure    run the paired Vanilla-vs-Base-only measurement on official
               validation, using an already-frozen LR

**There is no `--test` flag and no official-test argument.** `Preg1Role` has no
`OFFICIAL_TEST` member, so the sealed split is unreachable from this program
rather than merely discouraged.

**There is no `--learning-rate` flag on `tune` and no `--seeds` flag anywhere.**
The grid, the tuning seeds and the measurement seeds are precommitted in
`preg1_protocol`; a command-line override of a precommitted constant is the hole
the protocol exists to close.

`measure` refuses to run without `--frozen-lr`, which must name an LR that a
completed tuning artifact selected. Base-only can therefore never influence the
learning rate: by the time it is encoded at all, the LR is already a value in a
file.

Nothing here downloads a model on import. Torch and transformers are imported
lazily inside the run path, which is Colab-only; the local environment is ML-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.evaluation.contracts import (  # noqa: E402
    EvaluationContractViolation,
    SystemPathway,
)
from unmark.evaluation.preg1_head import (  # noqa: E402
    PREG1_HEAD_SCHEMA_VERSION,
    DETERMINISM_SCOPE,
    FrozenLearningRate,
    LrCandidate,
    NO_SIGNIFICANCE_TEST,
    Preg1Role,
    RepresentationCache,
    RepresentationKey,
    SplitMembership,
    freeze_learning_rate,
    load_membership,
    ordered_id_digest,
    select_learning_rate,
)
from unmark.evaluation.preg1_split import load_derived_pool  # noqa: E402
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_RULE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EPOCHS,
    LR_AGGREGATION_RULE,
    LR_GRID,
    MAX_LENGTH,
    MEASUREMENT_SEEDS,
    PADDING,
    PREG1_POOLING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_LR_SELECTION,
    PRIMARY_TASK,
    TRUNCATION,
    TUNING_SEEDS,
    Preg1Protocol,
)

# ---------------------------------------------------------------------------
# The tuning pathway is a constant, not an option
# ---------------------------------------------------------------------------
TUNING_PATHWAY = SystemPathway.VANILLA
"""Primary LR selection is VANILLA-only (D-PREG1-009).

A module constant rather than a parameter: there is no argument anywhere in the
tune path that could carry `BASE_ONLY`, so Base-only cannot influence the shared
learning rate even by mistake.
"""

TUNING_ROLES: tuple[Preg1Role, Preg1Role] = (
    Preg1Role.PROTOCOL_TRAIN,
    Preg1Role.PROTOCOL_DEV,
)
"""The only two roles `tune` may touch. Official validation is absent, and
official TEST is not representable at all."""

ENCODER_BATCH = 32
"""Rows per encoder forward. A runtime knob for memory, not a scientific value:
the representations it produces are identical for any batching."""


def tuning_schedule() -> tuple[tuple[float, int], ...]:
    """The precommitted runs: every LR in the grid x every tuning seed.

    Derived from `preg1_protocol`, so the count follows the protocol rather than
    a number typed here -- 5 x 3 = 15 today because the grid holds five rates and
    the tag derives three seeds.
    """
    return tuple((lr, seed) for lr in LR_GRID for seed in TUNING_SEEDS)


def materialise_split(pool, sample_ids: Sequence[str], role: Preg1Role):
    """Rows for `sample_ids`, **in the order the membership file gives them**.

    Order is load-bearing: it is what `ordered_id_digest` pins, and row *i* of a
    cached representation tensor must stay the row the label list calls *i*.
    """
    if role not in TUNING_ROLES:
        raise EvaluationContractViolation(
            f"tune may only materialise {[r.value for r in TUNING_ROLES]}, not {role.value}"
        )
    by_id = {sample_id: (text, label) for sample_id, text, label in pool.records}
    missing = [s for s in sample_ids if s not in by_id]
    if missing:
        raise EvaluationContractViolation(
            f"{len(missing)} membership ids are not in the derived pool: {missing[:5]}"
        )
    texts = [by_id[s][0] for s in sample_ids]
    labels = [_label_index(by_id[s][1]) for s in sample_ids]
    return texts, labels


def _label_index(label) -> int:
    from unmark.evaluation.preg1_protocol import LABEL_MAPPING

    text = str(label).strip()
    if text in LABEL_MAPPING:
        return LABEL_MAPPING[text]
    inverse = {str(v): v for v in LABEL_MAPPING.values()}
    if text in inverse:
        return inverse[text]
    raise EvaluationContractViolation(f"unknown label {label!r}")


def representation_key(
    role: Preg1Role, sample_ids: Sequence[str], source_sha256: str, hidden_size: int
) -> RepresentationKey:
    """Provenance for one cached representation set. Every field is locked."""
    return RepresentationKey(
        dataset=PRIMARY_DATASET,
        dataset_version=PRIMARY_DATASET_VERSION,
        task=PRIMARY_TASK,
        role=role,
        pathway=TUNING_PATHWAY,
        source_identity=source_sha256,
        ordered_id_digest=ordered_id_digest(sample_ids),
        tokenizer_id=ENCODER_CHECKPOINT,
        model_revision=ENCODER_REVISION,
        max_length=MAX_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING,
        pooling=PREG1_POOLING.value,
        dtype="torch.float32",
        hidden_size=hidden_size,
        count=len(sample_ids),
    )


def tuning_artifact(
    *,
    repository_head: str | None,
    membership: SplitMembership,
    source_sha256: str,
    keys: dict[str, RepresentationKey],
    candidates: Sequence[LrCandidate],
    winner: LrCandidate,
    frozen,
) -> dict:
    """The deterministic tuning record. **Ids, counts and digests only.**

    No raw text, and no timestamp, hostname or path -- the same inputs and seeds
    must produce the same artifact bytes, or "precommitted" means nothing.
    """
    return {
        "schema_version": PREG1_HEAD_SCHEMA_VERSION,
        "repository_head": repository_head,
        "protocol": Preg1Protocol().to_dict(),
        "pathway": TUNING_PATHWAY.value,
        "roles_used": [role.value for role in TUNING_ROLES],
        "official_validation_used": False,
        "official_test_used": False,
        "input": {
            "derived_train_sha256": source_sha256,
            "assignment_digest": membership.assignment_digest,
            "protocol_train_count": len(membership.protocol_train),
            "protocol_dev_count": len(membership.protocol_dev),
        },
        "representations": {name: key.to_dict() for name, key in keys.items()},
        "schedule": {
            "learning_rates": list(LR_GRID),
            "tuning_seeds": list(TUNING_SEEDS),
            "planned_runs": len(tuning_schedule()),
        },
        "runs": [run.to_dict() for candidate in candidates for run in candidate.runs],
        "per_learning_rate": [candidate.to_dict() for candidate in candidates],
        "selection": {
            "selected_learning_rate": winner.learning_rate,
            "frozen": frozen.to_dict(),
            "rule": list(LR_AGGREGATION_RULE),
            "checkpoint_rule": list(CHECKPOINT_RULE),
            "policy": PRIMARY_LR_SELECTION,
        },
        "boundaries": {
            "raw_text_persisted": False,
            "downstream_score": None,
            "head_trained_on": Preg1Role.PROTOCOL_TRAIN.value,
            "selected_on": Preg1Role.PROTOCOL_DEV.value,
        },
        "determinism": DETERMINISM_SCOPE,
        "interpretation": NO_SIGNIFICANCE_TEST,
    }


# ---------------------------------------------------------------------------
# Executable path -- torch and transformers are imported HERE, not at module
# scope, so this file stays importable in the ML-free local environment.
# ---------------------------------------------------------------------------
def load_frozen_encoder(revision: str):
    """The exact pinned checkpoint, frozen and in eval. Nothing else is accepted."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=revision, use_fast=False
    )
    encoder = AutoModel.from_pretrained(ENCODER_CHECKPOINT, revision=revision)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return tokenizer, encoder


def encode_texts(tokenizer, texts: Sequence[str]):
    """Tokenize under the locked contract. `pathway_text` has already run."""
    encoded = tokenizer(
        list(texts),
        padding=PADDING,
        truncation=TRUNCATION,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def extract_or_load(
    cache_root: Path,
    name: str,
    key: RepresentationKey,
    tokenizer,
    encoder,
    texts: Sequence[str],
):
    """Frozen `<s>` representations for one split, extracted **once**.

    Cached under the full `RepresentationKey`, so a reload can only succeed for
    the identical role, pathway, source, order and geometry. On a cache hit the
    encoder is never touched.
    """
    import torch

    from unmark.evaluation.preg1_head import (
        BoundRepresentations,
        extract_representations,
    )

    cache = RepresentationCache(cache_root / name)
    if cache.exists():
        bound = cache.load(key)  # fails closed on any provenance mismatch
        print(f"  [{name}] cache HIT  -> {tuple(bound.values.shape)} {bound.key.dtype}")
        return bound

    print(f"  [{name}] cache MISS -> extracting {len(texts)} rows in batches of {ENCODER_BATCH}")
    chunks = []
    for start in range(0, len(texts), ENCODER_BATCH):
        batch = texts[start : start + ENCODER_BATCH]
        input_ids, attention_mask = encode_texts(tokenizer, batch)
        # `extract_representations` is the committed primitive that
        # `extract_bound_representations` itself calls: it checks the encoder is
        # frozen and in eval, runs under no_grad, pools position 0 and returns a
        # detached FP32 tensor. Provenance is bound ONCE below, over the whole
        # set, because the key's `count` describes the full split.
        chunks.append(extract_representations(encoder, input_ids, attention_mask))
        done = min(start + ENCODER_BATCH, len(texts))
        if done % (ENCODER_BATCH * 25) == 0 or done == len(texts):
            print(f"    {done}/{len(texts)}")
    values = torch.cat(chunks, dim=0)
    bound = BoundRepresentations(values=values, key=key)
    cache.save(key, values)
    print(f"  [{name}] cached     -> {tuple(values.shape)} {values.dtype}")
    return bound


def run_tune(args) -> int:
    """The 15 precommitted VANILLA runs, then the committed selector."""
    from unmark.evaluation.pathways import pathway_text

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"REFUSED: {output_dir} already exists; tuning artifacts are immutable",
              file=sys.stderr)
        return 2

    pool = load_derived_pool(
        args.derived_train, args.text_column, args.label_column, args.id_column
    )
    membership = load_membership(args.split_dir)
    membership.require_partitions([sample_id for sample_id, _, _ in pool.records])
    print(f"  derived pool   : {len(pool.records)} rows, sha {pool.source_sha256[:16]}…")
    print(f"  protocol-train : {len(membership.protocol_train)}")
    print(f"  protocol-dev   : {len(membership.protocol_dev)}")

    splits = {}
    for role in TUNING_ROLES:
        ids = membership.ids_for(role)
        texts, labels = materialise_split(pool, ids, role)
        # RAW_BASE: canon(x) for VANILLA, and no segmenter anywhere.
        splits[role] = (ids, [pathway_text(t, TUNING_PATHWAY) for t in texts], labels)

    print(f"\nLoading {ENCODER_CHECKPOINT} @ {args.revision[:12]}… (frozen, eval)")
    tokenizer, encoder = load_frozen_encoder(args.revision)
    hidden_size = int(encoder.config.hidden_size)
    print(f"  hidden size    : {hidden_size}")

    cache_root = Path(args.cache_root)
    bound, keys = {}, {}
    for role in TUNING_ROLES:
        ids, texts, _ = splits[role]
        key = representation_key(role, ids, pool.source_sha256, hidden_size)
        keys[role.value] = key
        bound[role] = extract_or_load(cache_root, role.value, key, tokenizer, encoder, texts)

    train_bound, dev_bound = bound[Preg1Role.PROTOCOL_TRAIN], bound[Preg1Role.PROTOCOL_DEV]
    train_labels = splits[Preg1Role.PROTOCOL_TRAIN][2]
    dev_labels = splits[Preg1Role.PROTOCOL_DEV][2]

    from unmark.evaluation.preg1_head import train_head

    schedule = tuning_schedule()
    print(f"\n{len(schedule)} runs = {len(LR_GRID)} learning rates x "
          f"{len(TUNING_SEEDS)} tuning seeds, VANILLA only, {EPOCHS} epochs each")

    runs_by_lr: dict[float, list] = {lr: [] for lr in LR_GRID}
    for index, (lr, seed) in enumerate(schedule, start=1):
        print(f"\n[{index}/{len(schedule)}] lr={lr:g} seed={seed}")

        def on_epoch(epoch, score, _lr=lr, _seed=seed):
            if epoch % 5 == 0 or epoch == EPOCHS:
                print(f"    epoch {epoch:2d}/{EPOCHS}  "
                      f"dev macro-F1 {score.macro_f1:.4f}  acc {score.accuracy:.4f}")

        run = train_head(
            train_bound, train_labels, dev_bound, dev_labels,
            learning_rate=lr, seed=seed, epochs=EPOCHS, batch_size=BATCH_SIZE,
            on_epoch=on_epoch,
        )
        chosen = run.selected
        print(f"  -> selected epoch {chosen.epoch}: "
              f"macro-F1 {chosen.macro_f1:.4f}  acc {chosen.accuracy:.4f}")
        runs_by_lr[lr].append(run)

    candidates = [LrCandidate(lr, tuple(runs_by_lr[lr])) for lr in LR_GRID]
    winner = select_learning_rate(candidates)
    frozen = freeze_learning_rate(winner)

    print("\nPer learning rate (mean over tuning seeds):")
    for candidate in candidates:
        mark = " <-- selected" if candidate is winner else ""
        print(f"  lr={candidate.learning_rate:<8g} macro-F1 {candidate.mean_macro_f1:.4f} "
              f"acc {candidate.mean_accuracy:.4f} sd {candidate.stdev_macro_f1:.4f}{mark}")
    print(f"\nSELECTED LR: {frozen.value:g}  (selected on {frozen.selected_on.value})")

    artifact = tuning_artifact(
        repository_head=args.repository_head,
        membership=membership,
        source_sha256=pool.source_sha256,
        keys=keys,
        candidates=candidates,
        winner=winner,
        frozen=frozen,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "tuning.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_dir / 'tuning.json'} "
          f"({len(artifact['runs'])} runs recorded)")
    print(f"\nNext: measure --frozen-lr {frozen.value:g}")
    return 0


def run_measure(args) -> int:
    """Gated on an already-frozen LR. **Not executed in this build.**"""
    frozen = FrozenLearningRate(value=args.frozen_lr)
    print(f"  frozen LR      : {frozen.value:g} (selected on {frozen.selected_on.value})")
    print(f"  seeds          : {list(MEASUREMENT_SEEDS)}")
    print(f"  measured on    : {Preg1Role.OFFICIAL_VALIDATION.value}")
    print("\nMEASUREMENT NOT EXECUTED IN THIS BUILD.")
    print(
        "  The paired Vanilla-vs-Base-only measurement is the first downstream\n"
        "  number in this project and is deliberately not wired here. `tune`\n"
        "  must complete and its LR be reviewed first."
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-dir", required=True, help="preg1-split-v1 membership directory")
    parser.add_argument("--derived-train", required=True, help="approved derived TRAIN csv")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--cache-root", required=True, help="representation cache root")
    parser.add_argument("--output-dir", required=True, help="must not already exist")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tune = sub.add_parser("tune", help="VANILLA-only LR sweep on protocol-dev")
    _common(tune)
    tune.add_argument(
        "--revision", default=ENCODER_REVISION,
        help="encoder revision; defaults to the pinned probe revision and is "
             "validated against it",
    )
    tune.add_argument(
        "--repository-head", default=None,
        help="commit sha recorded as provenance in the tuning artifact",
    )

    measure = sub.add_parser("measure", help="paired measurement on official validation")
    _common(measure)
    measure.add_argument(
        "--official-validation", required=True,
        help="official validation csv; read ONLY after the LR is frozen",
    )
    measure.add_argument(
        "--frozen-lr", required=True, type=float,
        help="the LR a completed VANILLA tuning run selected; must be in the grid",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "tune" and args.revision != ENCODER_REVISION:
            raise EvaluationContractViolation(
                f"--revision must be the pinned diagnostic revision {ENCODER_REVISION}, "
                f"got {args.revision}. D-B3B0-002 is OPEN; a different backbone needs "
                "its own recorded position-id evidence."
            )
        membership = load_membership(args.split_dir)
        frozen = FrozenLearningRate(value=args.frozen_lr) if args.command == "measure" else None
    except EvaluationContractViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"pre-G1 burden diagnostic — {PREG1_HEAD_SCHEMA_VERSION}")
    print(f"  command        : {args.command}")
    print(f"  encoder        : {ENCODER_CHECKPOINT} @ {ENCODER_REVISION}")
    print(f"  protocol-train : {len(membership.protocol_train)}")
    print(f"  protocol-dev   : {len(membership.protocol_dev)}")
    print(f"  max_length     : {MAX_LENGTH} | padding {PADDING} | batch {BATCH_SIZE} "
          f"| epochs {EPOCHS}")
    print(f"  pooling        : {PREG1_POOLING.value} (pre-G1 only; Stage-2 pooling OPEN)")

    try:
        if args.command == "tune":
            print(f"  grid           : {list(LR_GRID)}")
            print(f"  tuning seeds   : {list(TUNING_SEEDS)} ({TUNING_PATHWAY.value} only)")
            print(f"  selection set  : {Preg1Role.PROTOCOL_DEV.value}")
            print(f"  official validation / TEST : not read\n")
            status = run_tune(args)
        else:
            status = run_measure(args)
    except EvaluationContractViolation as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    print(f"\n{NO_SIGNIFICANCE_TEST}\n\n{DETERMINISM_SCOPE}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
