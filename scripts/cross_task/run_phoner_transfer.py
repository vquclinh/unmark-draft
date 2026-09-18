#!/usr/bin/env python
"""Colab entry point for the ViUnMark Cross-Task NER Transfer Probe.

The script accepts external roots only. It does not download PhoNER data, commit
data, or embed personal Drive paths.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from unmark.corruption import CorruptionPurpose
from unmark.cross_task.phoner_transfer import (
    PHONER_GATE_SHA256,
    PHONER_FINAL_SEEDS,
    PHONER_IGNORE_INDEX,
    PHONER_SCALE_SHA256,
    EntityF1,
    FrozenTokenProbe,
    PhoNERCrossTaskConfig,
    PhoNERPathway,
    PhoNERSplit,
    build_phoner_dataset_identity,
    canonical_stage,
    collate_tokenized_ner_batch,
    corrupt_phoner_example,
    corruption_summary_metrics,
    encode_adapted_example,
    encode_native_example,
    ids_to_word_predictions,
    label_inventory_from_train,
    materialize_condition_invariant_chunks,
    parse_phoner_split,
    protocol_digest,
    require_checkpoint_sha256,
    require_frozen_module,
    require_phobert_identity,
    require_stage_access,
    summarize_five_seed_scores,
)
from unmark.linguistics import make_classifier, try_load_inventory
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(cache_dir: str | Path | None = None) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT,
        revision=ENCODER_REVISION,
        use_fast=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    require_phobert_identity(tokenizer=tokenizer)
    return tokenizer


def load_native_encoder(cache_dir: str | Path | None = None) -> Any:
    from transformers import AutoModel

    from unmark.cross_task.phoner_transfer import freeze_module

    encoder = AutoModel.from_pretrained(
        ENCODER_CHECKPOINT,
        revision=ENCODER_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    setattr(encoder, "_unmark_requested_revision", ENCODER_REVISION)
    freeze_module(encoder)
    require_phobert_identity(encoder=encoder)
    return encoder


def default_gate_checkpoint(asset_root: Path) -> Path:
    return asset_root / "stage1" / "viunmark_gate.pt"


def default_scale_checkpoint(asset_root: Path) -> Path:
    return asset_root / "stage1" / "viunmark_scale.pt"


def load_pathway(pathway: PhoNERPathway, args: argparse.Namespace, *, inventory: Any) -> Any:
    if pathway is PhoNERPathway.PHOBERT_NATIVE:
        return load_native_encoder(args.asset_root)
    if pathway is PhoNERPathway.VIUNMARK_GATE:
        from unmark.evaluation.stage2_dual_finalist import load_frozen_unmark_pathway

        checkpoint = Path(args.gate_checkpoint or default_gate_checkpoint(Path(args.asset_root)))
        require_checkpoint_sha256(checkpoint, PHONER_GATE_SHA256, label="ViUnMark-Gate")
        return load_frozen_unmark_pathway(
            "UNMARK-A",
            checkpoint,
            inventory=inventory,
            cache_dir=args.asset_root,
        )
    if pathway is PhoNERPathway.VIUNMARK_SCALE:
        from unmark.evaluation.stage2_scf_pathway import load_frozen_scf_pathway

        checkpoint = Path(args.scale_checkpoint or default_scale_checkpoint(Path(args.asset_root)))
        require_checkpoint_sha256(checkpoint, PHONER_SCALE_SHA256, label="ViUnMark-Scale")
        return load_frozen_scf_pathway(
            checkpoint,
            inventory=inventory,
            cache_dir=args.asset_root,
        )
    raise ValueError(pathway)


def pathway_hidden_states(pathway: PhoNERPathway, loaded: Any, batch: dict[str, Any], device: Any) -> Any:
    import torch
    from unmark.modeling.adapter import authoritative_position_ids, base_word_embeddings

    if pathway is PhoNERPathway.PHOBERT_NATIVE:
        require_frozen_module(loaded, "native PhoBERT encoder")
        with torch.no_grad():
            output = loaded(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            hidden = getattr(output, "last_hidden_state", output)
        return hidden.detach().to(torch.float32)

    loaded.require_frozen()
    encoder = loaded.encoder
    adapter = loaded.adapter
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        z = adapter(
            base_word_embeddings(encoder, input_ids),
            batch["tone_ids"].to(device),
            batch["tone_mask"].to(device),
            batch["letter_ids"].to(device),
            batch["letter_mask"].to(device),
        )
        output = encoder(
            inputs_embeds=z,
            attention_mask=batch["attention_mask"].to(device),
            position_ids=authoritative_position_ids(encoder, input_ids),
        )
        hidden = getattr(output, "last_hidden_state", output)
    return hidden.detach().to(torch.float32)


def encode_examples(
    examples: Sequence[Any],
    *,
    pathway: PhoNERPathway,
    condition: str,
    tokenizer: Any,
    label_to_id: dict[str, int],
    config: PhoNERCrossTaskConfig,
    classifier: Any,
) -> tuple[Any, ...]:
    corrupted = [
        corrupt_phoner_example(
            item,
            condition,
            config=config,
            purpose=CorruptionPurpose.SCIENTIFIC,
        )
        for item in examples
    ]
    if pathway is PhoNERPathway.PHOBERT_NATIVE:
        return tuple(encode_native_example(item, tokenizer, label_to_id, max_length=config.max_length) for item in corrupted)
    unk = getattr(tokenizer, "unk_token_id", None)
    return tuple(
        encode_adapted_example(
            item,
            tokenizer,
            label_to_id,
            classifier=classifier,
            unk_token_id=unk,
            max_length=config.max_length,
        )
        for item in corrupted
    )


def evaluate_dev(
    examples: Sequence[Any],
    *,
    pathway: PhoNERPathway,
    loaded: Any,
    probe: Any,
    tokenizer: Any,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    config: PhoNERCrossTaskConfig,
    classifier: Any,
    condition: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    probe.eval()
    gold: list[Sequence[str]] = []
    predicted: list[Sequence[str]] = []
    for chunk in batched(list(examples), config.probe_policy.batch_size):
        encoded = encode_examples(
            chunk,
            pathway=pathway,
            condition=condition,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            config=config,
            classifier=classifier,
        )
        batch = collate_tokenized_ner_batch(encoded, pad_token_id=tokenizer.pad_token_id)
        hidden = pathway_hidden_states(pathway, loaded, batch, device)
        with torch.no_grad():
            logits = probe(hidden.to(device))
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for item, ids in zip(encoded, pred_ids):
            predicted.append(
                ids_to_word_predictions(ids, item.word_ids, id_to_label, num_words=len(item.words))
            )
            if item.labels is not None:
                gold.append(item.labels)
    score = EntityF1.from_sequences(gold, predicted)
    return score.to_dict()


def train_one(
    train_examples: Sequence[Any],
    dev_examples: Sequence[Any],
    *,
    pathway: PhoNERPathway,
    seed: int,
    loaded: Any,
    tokenizer: Any,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    config: PhoNERCrossTaskConfig,
    classifier: Any,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    import torch

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pathway is PhoNERPathway.PHOBERT_NATIVE:
        loaded.to(device)
    else:
        loaded.encoder.to(device)
        loaded.adapter.to(device)
    probe = FrozenTokenProbe(HIDDEN_SIZE, len(label_to_id)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.probe_policy.learning_rate,
        betas=config.probe_policy.betas,
        eps=config.probe_policy.eps,
        weight_decay=config.probe_policy.weight_decay,
    )
    loss_class = getattr(torch.nn, "CrossEntropyLoss")
    loss_fn = loss_class(ignore_index=PHONER_IGNORE_INDEX)
    updates = 0
    best: dict[str, Any] | None = None
    max_updates = 2 if smoke else config.probe_policy.max_optimizer_updates
    train_rows = list(train_examples)
    while updates < max_updates:
        random.Random(seed * 100000 + updates).shuffle(train_rows)
        for chunk in batched(train_rows, config.probe_policy.batch_size):
            encoded = encode_examples(
                chunk,
                pathway=pathway,
                condition="FULL",
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                config=config,
                classifier=classifier,
            )
            batch = collate_tokenized_ner_batch(encoded, pad_token_id=tokenizer.pad_token_id)
            hidden = pathway_hidden_states(pathway, loaded, batch, device)
            probe.train()
            logits = probe(hidden.to(device))
            labels = batch["label_ids"].to(device)
            loss = loss_fn(logits.view(-1, len(label_to_id)), labels.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            if updates % config.probe_policy.eval_every_updates == 0 or updates == max_updates:
                metrics = evaluate_dev(
                    dev_examples,
                    pathway=pathway,
                    loaded=loaded,
                    probe=probe,
                    tokenizer=tokenizer,
                    label_to_id=label_to_id,
                    id_to_label=id_to_label,
                    config=config,
                    classifier=classifier,
                    condition="FULL",
                    device=device,
                )
                record = {"update": updates, "dev_full": metrics}
                if best is None or metrics["entity_micro_f1"] > best["dev_full"]["entity_micro_f1"]:
                    best = record
                    torch.save(
                        {
                            "schema_version": "phoner-frozen-token-probe-checkpoint-v1",
                            "pathway": pathway.value,
                            "seed": seed,
                            "update": updates,
                            "probe_state": probe.state_dict(),
                            "label_to_id": label_to_id,
                            "config": config.to_dict(),
                            "verified_asset_identities": {
                                "phobert_checkpoint": ENCODER_CHECKPOINT,
                                "phobert_revision": ENCODER_REVISION,
                                "gate_checkpoint_sha256": config.gate_checkpoint_sha256,
                                "scale_checkpoint_sha256": config.scale_checkpoint_sha256,
                            },
                            "selection": record,
                            "smoke": smoke,
                        },
                        output_dir / f"{pathway.value}_seed-{seed}_best.pt",
                    )
            if updates >= max_updates:
                break
    return {
        "pathway": pathway.value,
        "seed": seed,
        "smoke": smoke,
        "best": best,
        "max_updates": max_updates,
    }


def stage_audit(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    tokenizer = load_tokenizer(args.asset_root)
    identity = build_phoner_dataset_identity(
        args.data_root,
        stage="dataset-audit",
        tokenizer=tokenizer,
        coverage_purpose=CorruptionPurpose.SCIENTIFIC,
    )
    write_json(
        Path(args.output_root) / "dataset_identity.json",
        {"config": config.to_dict(), "dataset_identity": identity.to_dict(), "protocol_digest": protocol_digest(config, identity)},
    )


def stage_train(args: argparse.Namespace, config: PhoNERCrossTaskConfig, *, smoke: bool) -> None:
    import torch

    output_root = Path(args.output_root)
    train = parse_phoner_split(args.data_root, PhoNERSplit.TRAIN, include_labels=True, stage="smoke-train" if smoke else "train-dev")
    dev = parse_phoner_split(args.data_root, PhoNERSplit.DEV, include_labels=True, stage="smoke-train" if smoke else "train-dev")
    labels = label_inventory_from_train(train)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = load_tokenizer(args.asset_root)
    inventory = try_load_inventory()
    classifier = make_classifier(inventory) if inventory is not None else None
    train = materialize_condition_invariant_chunks(
        train, tokenizer, config, purpose=CorruptionPurpose.SCIENTIFIC
    )
    dev = materialize_condition_invariant_chunks(
        dev, tokenizer, config, purpose=CorruptionPurpose.SCIENTIFIC
    )
    seeds = (args.seed,) if smoke else PHONER_FINAL_SEEDS
    results = []
    for pathway in config.pathways:
        loaded = load_pathway(pathway, args, inventory=inventory)
        for seed in seeds:
            results.append(
                train_one(
                    train,
                    dev,
                    pathway=pathway,
                    seed=seed,
                    loaded=loaded,
                    tokenizer=tokenizer,
                    label_to_id=label_to_id,
                    id_to_label=id_to_label,
                    config=config,
                    classifier=classifier,
                    output_dir=output_root / "checkpoints",
                    smoke=smoke,
                )
            )
        if pathway is PhoNERPathway.PHOBERT_NATIVE:
            loaded.cpu()
        else:
            loaded.encoder.cpu()
            loaded.adapter.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_root / ("smoke_results.json" if smoke else "train_dev_results.json"), results)


def stage_freeze(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    tokenizer = load_tokenizer(args.asset_root)
    identity = build_phoner_dataset_identity(
        args.data_root,
        stage="freeze-protocol",
        tokenizer=tokenizer,
        coverage_purpose=CorruptionPurpose.SCIENTIFIC,
    )
    write_json(
        Path(args.output_root) / "frozen_protocol.json",
        {
            "schema_version": "phoner-cross-task-frozen-protocol-v1",
            "config": config.to_dict(),
            "dataset_identity": identity.to_dict(),
            "protocol_digest": protocol_digest(config, identity),
            "test_read_before_freeze": False,
            "test_scored_before_freeze": False,
            "protocol_frozen": True,
            "pre_freeze_dev_corruption_metrics_exposed": False,
            "post_freeze_rule": (
                "corrupted DEV metrics may be generated diagnostically only after this "
                "artifact exists; no protocol, hyperparameter, model, or selection "
                "change may follow from them"
            ),
        },
    )


def stage_test_predict(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    import torch

    require_stage_access("test-predict", split=PhoNERSplit.TEST, gold=False)
    test_examples = parse_phoner_split(
        args.data_root, PhoNERSplit.TEST, include_labels=False, stage="test-predict"
    )
    train_examples = parse_phoner_split(
        args.data_root, PhoNERSplit.TRAIN, include_labels=True, stage="test-predict"
    )
    labels = label_inventory_from_train(train_examples)
    tokenizer = load_tokenizer(args.asset_root)
    inventory = try_load_inventory()
    classifier = make_classifier(inventory) if inventory is not None else None
    output_path = Path(args.prediction_artifact or Path(args.output_root) / "test_predictions_sealed.json")
    runs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pathway in config.pathways:
        loaded = load_pathway(pathway, args, inventory=inventory)
        if pathway is PhoNERPathway.PHOBERT_NATIVE:
            loaded.to(device)
        else:
            loaded.encoder.to(device)
            loaded.adapter.to(device)
        for seed in PHONER_FINAL_SEEDS:
            checkpoint_path = Path(args.output_root) / "checkpoints" / f"{pathway.value}_seed-{seed}_best.pt"
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
            id_to_label = {index: label for label, index in label_to_id.items()}
            probe = FrozenTokenProbe(HIDDEN_SIZE, len(label_to_id)).to(device)
            probe.load_state_dict(payload["probe_state"])
            probe.eval()
            for condition in config.conditions:
                predictions = []
                for chunk in batched(list(test_examples), config.probe_policy.batch_size):
                    encoded = encode_examples(
                        chunk,
                        pathway=pathway,
                        condition=condition,
                        tokenizer=tokenizer,
                        label_to_id=label_to_id,
                        config=config,
                        classifier=classifier,
                    )
                    batch = collate_tokenized_ner_batch(encoded, pad_token_id=tokenizer.pad_token_id)
                    hidden = pathway_hidden_states(pathway, loaded, batch, device)
                    with torch.no_grad():
                        pred_ids = probe(hidden.to(device)).argmax(dim=-1).detach().cpu().tolist()
                    for item, ids in zip(encoded, pred_ids):
                        predictions.append(
                            {
                                "sample_id": item.sample_id,
                                "word_count": len(item.words),
                                "predicted_labels": list(
                                    ids_to_word_predictions(
                                        ids, item.word_ids, id_to_label, num_words=len(item.words)
                                    )
                                ),
                            }
                        )
                runs.append(
                    {
                        "pathway": pathway.value,
                        "seed": seed,
                        "condition": condition,
                        "checkpoint_path": str(checkpoint_path),
                        "predictions": predictions,
                    }
                )
    write_json(
        output_path,
        {
            "schema_version": "phoner-sealed-test-predictions-v1",
            "sealed": True,
            "contains_gold_labels": False,
            "contains_original_or_corrupted_text": False,
            "config": config.to_dict(),
            "runs": runs,
        },
    )


def stage_test_score(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    require_stage_access("test-score", split=PhoNERSplit.TEST, gold=True)
    if not args.prediction_artifact:
        raise SystemExit("--prediction-artifact is required for test-score")
    prediction_path = Path(args.prediction_artifact)
    if not prediction_path.exists():
        raise SystemExit(f"sealed prediction artifact not found: {prediction_path}")
    artifact = json.loads(prediction_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "phoner-sealed-test-predictions-v1":
        raise SystemExit("prediction artifact has the wrong schema")
    if artifact.get("sealed") is not True or artifact.get("contains_gold_labels") is not False:
        raise SystemExit("prediction artifact is not a sealed text-only prediction artifact")
    gold_examples = parse_phoner_split(
        args.data_root, PhoNERSplit.TEST, include_labels=True, stage="test-score"
    )
    gold_by_id = {item.sample_id: item for item in gold_examples}
    scores = []
    by_run_key: dict[tuple[str, int], dict[str, float]] = {}
    for run in artifact.get("runs", []):
        gold, pred = [], []
        for row in run["predictions"]:
            gold_item = gold_by_id[row["sample_id"]]
            if int(row["word_count"]) != len(gold_item.words):
                raise SystemExit(f"word count mismatch for {row['sample_id']}")
            gold.append(gold_item.labels)
            pred.append(tuple(row["predicted_labels"]))
        metric = EntityF1.from_sequences(gold, pred)
        record = {
            "pathway": run["pathway"],
            "seed": run["seed"],
            "condition": run["condition"],
            **metric.to_dict(),
        }
        scores.append(record)
        by_run_key.setdefault((run["pathway"], int(run["seed"])), {})[run["condition"]] = metric.f1
    write_json(
        Path(args.output_root) / "test_scores_from_sealed_predictions.json",
        {
            "schema_version": "phoner-test-score-v1",
            "prediction_artifact": str(prediction_path),
            "scores": scores,
            "summaries": summarize_five_seed_scores(scores),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhoNER frozen cross-task transfer probe.")
    parser.add_argument("--stage", required=True, choices=sorted({
        "audit", "dataset-audit", "smoke", "smoke-train", "train-dev",
        "dev-evaluate", "freeze", "freeze-protocol", "test-predict", "test-score",
    }))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--scale-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=PhoNERCrossTaskConfig().probe_policy.smoke_seed)
    parser.add_argument("--prediction-artifact", default=None)
    args = parser.parse_args()

    config = PhoNERCrossTaskConfig()
    stage = canonical_stage(args.stage)
    if stage == "dataset-audit":
        stage_audit(args, config)
    elif stage == "smoke-train":
        stage_train(args, config, smoke=True)
    elif stage in {"train-dev", "dev-evaluate"}:
        stage_train(args, config, smoke=False)
    elif stage == "freeze-protocol":
        stage_freeze(args, config)
    elif stage == "test-predict":
        stage_test_predict(args, config)
    elif stage == "test-score":
        stage_test_score(args, config)
    else:  # pragma: no cover
        raise AssertionError(stage)


if __name__ == "__main__":
    main()
