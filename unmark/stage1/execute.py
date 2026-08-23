"""Stage orchestration: build the model, run a schedule, persist selection.

**Imports torch, transformers and the prepared corpus lazily.** Nothing here
runs at import time, and Audit 029 executes none of it.

`execute_stage` is the only place a scientific Stage-1 run is launched, and
`smoke_check` is deliberately a *separate* function that constructs no optimizer
and calls no `.backward()` -- so "the smoke path cannot update a parameter" is
structural rather than a promise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from unmark.stage1.contracts import (
    CorruptionRatePolicy,
    OverflowBehaviour,
    Stage1ContractViolation,
    TruncationPolicy,
)
from unmark.stage1.checkpoint import VerifiedCorpus
from unmark.stage1.trainer import load_training_checkpoint
from unmark.stage1.manifest import CHUNKS_NAME
from unmark.stage1.protocol import (
    BATCH_SIZE,
    CORRUPTION_SEED,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    HIDDEN_SIZE,
    INITIAL_MAX_UPDATES,
    MAX_LENGTH,
    STAGE1_PROTOCOL_VERSION,
    lambdas_for_r,
)
from unmark.stage1.selection import (
    Candidate,
    PlannedRun,
    select_learning_rate,
    select_r,
)
from unmark.stage1.trainer import RunProvenance, train_run, verify_model_contract

TRUNCATION = TruncationPolicy(max_length=MAX_LENGTH, on_overflow=OverflowBehaviour.FAIL)
"""`FAIL` is a guard: after correct pre-chunking nothing can overflow."""


def load_prepared_chunks(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    """`(train_text_by_chunk_id, dev_text_by_chunk_id)` from the prepared corpus."""
    path = Path(directory) / CHUNKS_NAME
    if not path.is_file():
        raise Stage1ContractViolation(f"prepared chunks not found: {path}")
    train: dict[str, str] = {}
    dev: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = train if row["partition"] == "train" else dev
            target[row["chunk_id"]] = row["text"]
    if not train or not dev:
        raise Stage1ContractViolation(
            f"prepared corpus has {len(train)} train and {len(dev)} dev chunks; both "
            "partitions must be non-empty"
        )
    return train, dev


def build_objective(revision: str):
    """Frozen pinned encoder + the locked adapter. **Lazy torch/transformers.**"""
    from transformers import AutoModel, AutoTokenizer

    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.config import AdapterConfig
    from unmark.stage1.objective import Stage1Objective

    if revision != ENCODER_REVISION:
        raise Stage1ContractViolation(
            f"backbone revision {revision!r} is not the locked {ENCODER_REVISION!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=revision, use_fast=False
    )
    encoder = AutoModel.from_pretrained(ENCODER_CHECKPOINT, revision=revision)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    hidden = int(encoder.config.hidden_size)
    if hidden != HIDDEN_SIZE:
        raise Stage1ContractViolation(f"hidden size {hidden} != locked {HIDDEN_SIZE}")
    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=hidden))
    return tokenizer, UnmarkEncoder(encoder=encoder, adapter=adapter), Stage1Objective


def execute_stage(
    *,
    stage: str,
    schedule: Sequence[PlannedRun],
    prepared_corpus: Path,
    verified: "VerifiedCorpus",
    output_dir: Path,
    cache_root: Path,
    revision: str,
    repository_head: str | None,
    resume: bool = False,
) -> int:
    """Run every planned run of one stage and persist the selection artifact.

    With `resume`, each planned run continues from its own verified training
    checkpoint if one exists, and starts fresh if it does not -- so a stage
    interrupted after two of five runs redoes neither of the two.
    """
    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.objective import Stage1Objective
    from unmark.stage1.preflight import verify_scientific_inputs
    from unmark.stage1.validation import HeldOutExample, at_update, evaluate, prepare_condition_batch
    from unmark.stage1.protocol import VALIDATION_CONDITIONS

    # Every mandatory external scientific input, BEFORE the encoder is fetched
    # or loaded. The second real smoke (Audit 030 §W) discovered the missing
    # pinned syllable inventory only after the model was already resident.
    inputs = verify_scientific_inputs()
    print(f"scientific inputs VERIFIED: eligibility {inputs.report['eligibility_policy']}")
    print(f"  inventory {inputs.inventory.source_name} @ {inputs.inventory.source_revision[:12]} "
          f"sha256 {inputs.inventory.sha256[:12]} ({inputs.report['inventory_shape']['unique_stripped_form_count']} stripped forms)")

    train_text, dev_text = load_prepared_chunks(prepared_corpus)
    tokenizer, unmark_encoder, objective_cls = build_objective(revision)
    classifier = make_classifier(try_load_inventory())
    pad_token_id = tokenizer.pad_token_id
    held_out = [HeldOutExample(cid, text) for cid, text in sorted(dev_text.items())]

    # The held-out corruption realization is built ONCE and reused by every
    # candidate: the same examples under the same fixed conditions, so candidates
    # differ only by the model. Independent of any training seed.
    prepared_by_condition = {
        condition: prepare_condition_batch(
            held_out, tokenizer, condition, truncation=TRUNCATION, classifier=classifier
        )
        for condition in VALIDATION_CONDITIONS
    }

    # The VERIFIED digest (Audit 030 F1). Until the hardening this read
    # `manifest["counts"][...]` -- a declaration accepted on trust, so a run
    # could record a digest describing data it had not trained on. `verified`
    # can only exist if every artifact COMPLETE.json binds was re-hashed from
    # disk and matched.
    manifest_digest = verified.chunk_membership_digest
    candidates: list[Candidate] = []
    output_dir.mkdir(parents=True, exist_ok=resume)

    for planned in schedule:
        # One checkpoint namespace per run, named by the run's own label, so two
        # runs in a stage can never overwrite each other's state.
        run_checkpoints = output_dir / f"run-{planned.label.replace('=', '')}" / "_checkpoint"
        lambda_align, lambda_clean = lambdas_for_r(planned.r)
        provenance = RunProvenance(
            run_seed=planned.seed,
            corruption_seed=CORRUPTION_SEED,
            learning_rate=planned.learning_rate,
            r=planned.r,
            corpus_manifest_digest=manifest_digest,
            repository_head=repository_head,
            inventory=inputs.inventory,
        )
        objective = objective_cls(unmark_encoder, provenance.weights)
        corruption = CorruptionRatePolicy(seed=CORRUPTION_SEED)

        def evaluate_fn(update: int, _obj=objective) -> Any:
            return at_update(
                evaluate(_obj, prepared_by_condition, pad_token_id, batch_size=BATCH_SIZE),
                update,
            )

        result = train_run(
            objective=objective,
            provenance=provenance,
            train_chunks=train_text,
            tokenizer=tokenizer,
            corruption_policy=corruption,
            truncation=TRUNCATION,
            evaluate_fn=evaluate_fn,
            pad_token_id=pad_token_id,
            classifier=classifier,
            cap=INITIAL_MAX_UPDATES,
            checkpoint_dir=run_checkpoints,
            # Explicit: a checkpoint is used only when the operator asked to
            # resume. `train_run` verifies its identity before touching it.
            resume=load_training_checkpoint(run_checkpoints) if resume else None,
        )
        if result.continued:
            # SAME run, continued -- not a new candidate. The locked budget rule
            # requires preserving adapter, optimizer, visit, cursor and streams
            # across the 20k boundary. This passed `resume=None` until the Audit
            # 030 F3 hardening, which rebuilt the optimizer and restarted the
            # sampler at visit 0 -- a continuation in name only. It now resumes
            # from the checkpoint the first leg wrote at exactly `cap`, so the
            # continuation uses the same mechanism as a crash resume.
            carried = load_training_checkpoint(run_checkpoints)
            if carried is None:
                raise Stage1ContractViolation(
                    "the 20k leg produced no checkpoint to continue from; a "
                    "continuation must preserve optimizer and sampler state"
                )
            result = train_run(
                objective=objective,
                provenance=provenance,
                train_chunks=train_text,
                tokenizer=tokenizer,
                corruption_policy=corruption,
                truncation=TRUNCATION,
                evaluate_fn=evaluate_fn,
                pad_token_id=pad_token_id,
                classifier=classifier,
                cap=result.cap,
                resume=carried,
                checkpoint_dir=run_checkpoints,
            )
        candidates.append(
            Candidate(
                label=planned.label,
                learning_rate=planned.learning_rate,
                r=planned.r,
                selected=result.selected,
                budget_limited=result.budget_limited,
            )
        )
        (output_dir / f"run-{planned.label.replace('=', '')}.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    artifact = {
        "stage": stage,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "repository_head": repository_head,
        "corpus_manifest_digest": manifest_digest,
        "candidates": [c.to_dict() for c in candidates],
        "raw_text_persisted": False,
        "official_test_used": False,
        "downstream_score_used": False,
    }
    if stage == "lr_pilot":
        artifact["selected"] = select_learning_rate(candidates).to_dict()
    elif stage == "r_phase1":
        frozen = candidates[0].learning_rate
        artifact["selected"] = select_r(candidates, frozen).to_dict()
    else:
        from unmark.stage1.selection import descriptive_summary

        artifact["final_main"] = {
            "note": "these three adapters ARE the final main Stage-1 models",
            "score": descriptive_summary([c.selected.score for c in candidates]),
            "d_clean": descriptive_summary([c.selected.d_clean for c in candidates]),
        }
    (output_dir / f"{stage}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {output_dir}/{stage}.json")
    return 0


def _resident_bytes() -> int | None:
    """Process RSS, for the Audit 030 F4 measurement. Linux only; None elsewhere."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001 - measurement only, never load-bearing
        return None


def smoke_check(
    *,
    prepared_corpus: Path,
    revision: str,
    repository_head: str | None,
    completion_dir: Path | None = None,
) -> int:
    """No-update real-model integration check.

    **Constructs no optimizer and calls no `.backward()`.** It reports the model
    contract and one forward pass, and cannot change a parameter.

    Verifies the prepared corpus **before the model is loaded** (Audit 030 F1),
    so the smoke exercises exactly the gate a training run will pass through.
    """
    import time

    import torch

    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.checkpoint import verify_prepared_corpus
    from unmark.stage1.data import (
        Stage1Example,
        batch_to_device,
        collate_stage1_batch,
        module_device,
        prepare_example,
    )
    from unmark.stage1.preflight import verify_scientific_inputs

    inputs = verify_scientific_inputs()
    print(f"scientific inputs VERIFIED: eligibility {inputs.report['eligibility_policy']}")
    print(f"  inventory sha256 {inputs.inventory.sha256}")

    completion = Path(completion_dir) if completion_dir else Path(prepared_corpus) / "_checkpoint"
    verified = verify_prepared_corpus(Path(prepared_corpus), completion)
    print(f"prepared corpus VERIFIED against {verified.completion_path}")
    print(f"  chunk_membership_digest {verified.chunk_membership_digest}")
    print(f"  counts {json.dumps(verified.counts, sort_keys=True)}")

    # Audit 030 F4 is measured here, on the real corpus, and nowhere else.
    started = time.monotonic()
    train_text, dev_text = load_prepared_chunks(prepared_corpus)
    print(f"  loaded {len(train_text)} train and {len(dev_text)} dev chunks in "
          f"{time.monotonic() - started:.1f}s")
    resident = _resident_bytes()
    if resident is not None:
        print(f"  process RSS after load: {resident / 1e9:.2f} GB")
    tokenizer, unmark_encoder, objective_cls = build_objective(revision)
    contract = verify_model_contract(unmark_encoder)
    objective = objective_cls(unmark_encoder, lambdas_to_weights(1.0))
    corruption = CorruptionRatePolicy(seed=CORRUPTION_SEED)
    classifier = make_classifier(try_load_inventory())

    sample = sorted(train_text)[:8]
    prepared = [
        prepare_example(
            Stage1Example(text=train_text[cid], sample_id=cid),
            tokenizer, corruption_policy=corruption, truncation=TRUNCATION,
            visit=0, classifier=classifier,
        )
        for cid in sample
    ]
    # The same one boundary `evaluate` and `train_run` use: the batch follows the
    # model, derived from the objective's own parameters. A no-op on CPU.
    batch = batch_to_device(
        collate_stage1_batch([p for p in prepared if p is not None], tokenizer.pad_token_id),
        module_device(objective),
    )
    with torch.no_grad():
        result = objective(batch)
    print(json.dumps({
        "smoke": "STAGE1_NO_UPDATE_FORWARD_ONLY",
        "repository_head": repository_head,
        "model_contract": contract,
        "losses": result.to_dict(),
        "optimizer_constructed": False,
        "backward_called": False,
        "parameters_updated": 0,
    }, indent=2, sort_keys=True))
    return 0


def lambdas_to_weights(r: float):
    from unmark.stage1.contracts import ObjectiveWeights

    lambda_align, lambda_clean = lambdas_for_r(r)
    return ObjectiveWeights(lambda_align=lambda_align, lambda_clean=lambda_clean)
