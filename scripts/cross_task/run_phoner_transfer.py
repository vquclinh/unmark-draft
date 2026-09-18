#!/usr/bin/env python
"""Colab entry point for the ViUnMark Cross-Task NER Transfer Probe.

The script accepts external roots only. It does not download PhoNER data, commit
data, or embed personal Drive paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Mapping
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
    sha256_file,
    summarize_five_seed_scores,
)
from unmark.linguistics import make_classifier, try_load_inventory
from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION, HIDDEN_SIZE


FROZEN_PROBE_HEAD_SCHEMA = "phoner-frozen-probe-heads-v1"
DEV_EVALUATE_SCHEMA = "phoner-dev-evaluate-results-v1"
EXPECTED_FROZEN_PROBE_HEAD_SHA256: dict[str, str] = {
    "PHOBERT_NATIVE_seed-42941_best.pt": "bbdb765a885afcb0855dcaa25d4444b91ab3fbf1ee8e14127bce4c696e4208fd",
    "PHOBERT_NATIVE_seed-53148_best.pt": "971606d6f0f82e04c2a9892a1bb1320577739223bec303edfad44f178730a228",
    "PHOBERT_NATIVE_seed-59945_best.pt": "ca485ad85c66058923cf1ffaaed9af676dc09d13cdd4c79cce97d165e8c3c30b",
    "PHOBERT_NATIVE_seed-720_best.pt": "6ad51575fd04ec31e1bdd291d2d363be23748bc20cca91a3d2575f91c825508b",
    "PHOBERT_NATIVE_seed-9428_best.pt": "bd7c688bccca42f618ef9b9955b35b03496f82af13dc92547920ea6ac8f0d640",
    "VIUNMARK_GATE_seed-42941_best.pt": "a8e8fcffa5b2be02dc794296df31aedaf2c3bbd9a6f709edf1ef89d0063f330d",
    "VIUNMARK_GATE_seed-53148_best.pt": "273c5a16e970084ba10056f648135ec36966ebeed1bc62e464a6e6835c3e7efd",
    "VIUNMARK_GATE_seed-59945_best.pt": "395ce4514e741c57f20b21d9e37f1e4500eb0d238b46152bb67668d44fd8d978",
    "VIUNMARK_GATE_seed-720_best.pt": "ddaf6ed4df283434255227024060b9be61407eb93dc594f0b68d7ec41424085b",
    "VIUNMARK_GATE_seed-9428_best.pt": "48c2fd27101434671dc960925287dc29b89fd38459fa3b8c0f2a099d2d2af982",
    "VIUNMARK_SCALE_seed-42941_best.pt": "78370a41cbbe014482e69efecd236a0bb2174732581c442b8427a0ed67fe5377",
    "VIUNMARK_SCALE_seed-53148_best.pt": "982ae3d41d8d96849cce33ec782b78f196f8fb26e497bd9f8f58c26941a967d2",
    "VIUNMARK_SCALE_seed-59945_best.pt": "97e5872c6e1b7760665f4705074e8c32cedf73fdcdffbe60981acc354d4dbd03",
    "VIUNMARK_SCALE_seed-720_best.pt": "caa3c9575fc45feb914effb6af478d3e2c2cc691cf919b50da7fb7d5c40bd65d",
    "VIUNMARK_SCALE_seed-9428_best.pt": "a44e45d1e328200421b3ac79f34481bb29326a7939cfc7997d4c2a8752b826c0",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    write_json(path, payload)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_payload_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def repository_head() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


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


def ensure_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    return {"root": root, "checkpoints": checkpoints}


def frozen_protocol_path(output_root: str | Path) -> Path:
    return Path(output_root) / "frozen_protocol.json"


def frozen_probe_heads_path(output_root: str | Path) -> Path:
    return Path(output_root) / "frozen_probe_heads.json"


def dev_evaluate_results_path(output_root: str | Path) -> Path:
    return Path(output_root) / "dev_evaluate_results.json"


def checkpoint_file_name(pathway: PhoNERPathway, seed: int) -> str:
    return f"{pathway.value}_seed-{seed}_best.pt"


def selected_probe_head_plan(config: PhoNERCrossTaskConfig) -> list[tuple[PhoNERPathway, int, str]]:
    plan = []
    for pathway in config.pathways:
        for seed in PHONER_FINAL_SEEDS:
            plan.append((pathway, seed, checkpoint_file_name(pathway, seed)))
    if len(plan) != 15:
        raise SystemExit(f"selected probe head plan must contain exactly 15 heads, got {len(plan)}")
    names = {name for _, _, name in plan}
    if names != set(EXPECTED_FROZEN_PROBE_HEAD_SHA256):
        raise SystemExit("selected probe head plan drifted from frozen SHA-256 inventory")
    return plan


def dev_evaluation_plan(config: PhoNERCrossTaskConfig) -> list[tuple[PhoNERPathway, int, str]]:
    plan = [
        (pathway, seed, condition)
        for pathway in config.pathways
        for seed in PHONER_FINAL_SEEDS
        for condition in config.conditions
    ]
    if len(plan) != 90:
        raise SystemExit(f"DEV evaluation plan must contain exactly 90 records, got {len(plan)}")
    return plan


def load_probe_checkpoint(path: Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"probe checkpoint is not a mapping: {path}")
    return payload


def load_and_verify_frozen_protocol(
    args: argparse.Namespace,
    config: PhoNERCrossTaskConfig,
    tokenizer: Any,
) -> tuple[Mapping[str, Any], Any, str]:
    identity = build_phoner_dataset_identity(
        args.data_root,
        stage="freeze-protocol",
        tokenizer=tokenizer,
        coverage_purpose=CorruptionPurpose.SCIENTIFIC,
    )
    current_digest = protocol_digest(config, identity)
    artifact = read_json(frozen_protocol_path(args.output_root))
    if artifact.get("schema_version") != "phoner-cross-task-frozen-protocol-v1":
        raise SystemExit("frozen_protocol.json has the wrong schema")
    if artifact.get("protocol_frozen") is not True:
        raise SystemExit("frozen_protocol.json does not mark protocol_frozen=true")
    if artifact.get("config") != config.to_dict():
        raise SystemExit("frozen_protocol.json config does not match the current protocol config")
    if artifact.get("protocol_digest") != current_digest:
        raise SystemExit("frozen_protocol.json protocol_digest does not match current DATA_ROOT identity")
    if artifact.get("test_read_before_freeze") is not False:
        raise SystemExit("frozen_protocol.json reports TEST was read before freeze")
    if artifact.get("test_scored_before_freeze") is not False:
        raise SystemExit("frozen_protocol.json reports TEST was scored before freeze")
    test_identity = artifact.get("dataset_identity", {}).get("split_files", {}).get("test", {})
    if test_identity.get("sha256") != "SEALED_UNREAD" or test_identity.get("row_count") != "SEALED_UNREAD":
        raise SystemExit("frozen_protocol.json does not preserve the sealed TEST identity")
    if test_identity.get("gold_labels_read") is not False:
        raise SystemExit("frozen_protocol.json reports TEST gold labels were read")
    return artifact, identity, current_digest


def build_frozen_probe_heads_payload(
    output_root: str | Path,
    config: PhoNERCrossTaskConfig,
    *,
    protocol_digest_value: str,
) -> dict[str, Any]:
    root = Path(output_root)
    entries = []
    for pathway, seed, name in selected_probe_head_plan(config):
        checkpoint_path = root / "checkpoints" / name
        if not checkpoint_path.is_file():
            raise SystemExit(f"selected probe checkpoint is missing: {checkpoint_path}")
        expected_sha = EXPECTED_FROZEN_PROBE_HEAD_SHA256[name]
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"selected probe checkpoint SHA mismatch for {name}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        payload = load_probe_checkpoint(checkpoint_path, map_location="cpu")
        if payload.get("pathway") != pathway.value:
            raise SystemExit(f"{name} records pathway {payload.get('pathway')!r}, expected {pathway.value!r}")
        if int(payload.get("seed", -1)) != seed:
            raise SystemExit(f"{name} records seed {payload.get('seed')!r}, expected {seed}")
        schema = payload.get("schema_version")
        update = payload.get("update")
        if schema != "phoner-frozen-token-probe-checkpoint-v1":
            raise SystemExit(f"{name} has unexpected checkpoint schema: {schema!r}")
        if not isinstance(update, int):
            raise SystemExit(f"{name} does not record an integer selected update")
        entries.append(
            {
                "pathway": pathway.value,
                "seed": seed,
                "relative_checkpoint_path": f"checkpoints/{name}",
                "sha256": actual_sha,
                "selected_update": update,
                "checkpoint_schema_version": schema,
            }
        )
    if len(entries) != 15:
        raise SystemExit(f"frozen_probe_heads must bind exactly 15 heads, got {len(entries)}")
    return {
        "schema_version": FROZEN_PROBE_HEAD_SCHEMA,
        "repository_head": repository_head(),
        "protocol_digest": protocol_digest_value,
        "expected_head_count": 15,
        "heads": entries,
    }


def write_frozen_probe_heads(
    output_root: str | Path,
    config: PhoNERCrossTaskConfig,
    *,
    protocol_digest_value: str,
) -> Path:
    path = frozen_probe_heads_path(output_root)
    payload = build_frozen_probe_heads_payload(
        output_root,
        config,
        protocol_digest_value=protocol_digest_value,
    )
    write_json_once(path, payload)
    return path


def load_and_verify_frozen_probe_heads(
    output_root: str | Path,
    config: PhoNERCrossTaskConfig,
    *,
    protocol_digest_value: str,
) -> tuple[Mapping[str, Any], str]:
    path = frozen_probe_heads_path(output_root)
    artifact = read_json(path)
    if artifact.get("schema_version") != FROZEN_PROBE_HEAD_SCHEMA:
        raise SystemExit("frozen_probe_heads.json has the wrong schema")
    if artifact.get("protocol_digest") != protocol_digest_value:
        raise SystemExit("frozen_probe_heads.json protocol_digest does not match frozen_protocol.json")
    if artifact.get("repository_head") != repository_head():
        raise SystemExit("frozen_probe_heads.json repository HEAD does not match the current checkout")
    if len(artifact.get("heads", [])) != 15:
        raise SystemExit("frozen_probe_heads.json must bind exactly 15 selected heads")
    expected = build_frozen_probe_heads_payload(
        output_root,
        config,
        protocol_digest_value=protocol_digest_value,
    )
    if artifact.get("heads") != expected["heads"]:
        raise SystemExit("frozen_probe_heads.json does not match the selected checkpoint files")
    return artifact, sha256_file(path)


def load_inventory_inputs() -> tuple[Any, Any, dict[str, Any]]:
    """Return runtime inventory and Stage-I provenance identity as distinct objects."""

    from unmark.stage1.preflight import verify_scientific_inputs

    scientific_inputs = verify_scientific_inputs()
    runtime_inventory = try_load_inventory()
    if runtime_inventory is None:
        raise RuntimeError("scientific inventory verified but runtime inventory did not load")
    return runtime_inventory, scientific_inputs.inventory, scientific_inputs.report


def load_pathway(pathway: PhoNERPathway, args: argparse.Namespace, *, inventory_identity: Any) -> Any:
    if pathway is PhoNERPathway.PHOBERT_NATIVE:
        return load_native_encoder(args.asset_root)
    if pathway is PhoNERPathway.VIUNMARK_GATE:
        from unmark.evaluation.stage2_dual_finalist import load_frozen_unmark_pathway

        checkpoint = Path(args.gate_checkpoint or default_gate_checkpoint(Path(args.asset_root)))
        require_checkpoint_sha256(checkpoint, PHONER_GATE_SHA256, label="ViUnMark-Gate")
        return load_frozen_unmark_pathway(
            "UNMARK-A",
            checkpoint,
            inventory=inventory_identity,
            cache_dir=args.asset_root,
        )
    if pathway is PhoNERPathway.VIUNMARK_SCALE:
        from unmark.evaluation.stage2_scf_pathway import load_frozen_scf_pathway

        checkpoint = Path(args.scale_checkpoint or default_scale_checkpoint(Path(args.asset_root)))
        require_checkpoint_sha256(checkpoint, PHONER_SCALE_SHA256, label="ViUnMark-Scale")
        return load_frozen_scf_pathway(
            checkpoint,
            inventory=inventory_identity,
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


def evaluate_dev_fanout(
    examples: Sequence[Any],
    *,
    pathway: PhoNERPathway,
    loaded: Any,
    heads: Mapping[int, tuple[Any, dict[str, int], dict[int, str]]],
    tokenizer: Any,
    config: PhoNERCrossTaskConfig,
    classifier: Any,
    condition: str,
    device: Any,
) -> dict[int, dict[str, Any]]:
    import torch

    if tuple(heads) != PHONER_FINAL_SEEDS:
        raise SystemExit("fan-out evaluation requires the frozen five seeds in protocol order")
    first_label_to_id: dict[str, int] | None = None
    for seed, (probe, label_to_id, _id_to_label) in heads.items():
        probe.eval()
        if first_label_to_id is None:
            first_label_to_id = label_to_id
        elif label_to_id != first_label_to_id:
            raise SystemExit(f"seed {seed} label inventory differs inside fan-out evaluation")
    if first_label_to_id is None:
        raise SystemExit("no frozen probe heads supplied for fan-out evaluation")

    gold_by_seed: dict[int, list[Sequence[str]]] = {seed: [] for seed in heads}
    predicted_by_seed: dict[int, list[Sequence[str]]] = {seed: [] for seed in heads}
    for chunk in batched(list(examples), config.probe_policy.batch_size):
        encoded = encode_examples(
            chunk,
            pathway=pathway,
            condition=condition,
            tokenizer=tokenizer,
            label_to_id=first_label_to_id,
            config=config,
            classifier=classifier,
        )
        batch = collate_tokenized_ner_batch(encoded, pad_token_id=tokenizer.pad_token_id)
        hidden = pathway_hidden_states(pathway, loaded, batch, device)
        for seed, (probe, _label_to_id, id_to_label) in heads.items():
            with torch.no_grad():
                logits = probe(hidden.to(device))
                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            for item, ids in zip(encoded, pred_ids):
                predicted_by_seed[seed].append(
                    ids_to_word_predictions(ids, item.word_ids, id_to_label, num_words=len(item.words))
                )
                if item.labels is not None:
                    gold_by_seed[seed].append(item.labels)
    return {
        seed: EntityF1.from_sequences(gold_by_seed[seed], predicted_by_seed[seed]).to_dict()
        for seed in heads
    }


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
    inventory_identity: Any,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
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
                                "inventory": inventory_identity.to_dict(),
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

    output_dirs = ensure_output_dirs(args.output_root)
    output_root = output_dirs["root"]
    train = parse_phoner_split(args.data_root, PhoNERSplit.TRAIN, include_labels=True, stage="smoke-train" if smoke else "train-dev")
    dev = parse_phoner_split(args.data_root, PhoNERSplit.DEV, include_labels=True, stage="smoke-train" if smoke else "train-dev")
    labels = label_inventory_from_train(train)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = load_tokenizer(args.asset_root)
    runtime_inventory, inventory_identity, inventory_report = load_inventory_inputs()
    classifier = make_classifier(runtime_inventory)
    train = materialize_condition_invariant_chunks(
        train, tokenizer, config, purpose=CorruptionPurpose.SCIENTIFIC
    )
    dev = materialize_condition_invariant_chunks(
        dev, tokenizer, config, purpose=CorruptionPurpose.SCIENTIFIC
    )
    seeds = (args.seed,) if smoke else PHONER_FINAL_SEEDS
    results = []
    for pathway in config.pathways:
        loaded = load_pathway(pathway, args, inventory_identity=inventory_identity)
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
                    inventory_identity=inventory_identity,
                    output_dir=output_dirs["checkpoints"],
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
    write_json(
        output_root / ("smoke_results.json" if smoke else "train_dev_results.json"),
        {
            "verified_asset_identities": {
                "phobert_checkpoint": ENCODER_CHECKPOINT,
                "phobert_revision": ENCODER_REVISION,
                "gate_checkpoint_sha256": config.gate_checkpoint_sha256,
                "scale_checkpoint_sha256": config.scale_checkpoint_sha256,
                "inventory": inventory_identity.to_dict(),
            },
            "inventory_preflight": inventory_report,
            "runs": results,
        },
    )


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


def stage_freeze_heads(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    tokenizer = load_tokenizer(args.asset_root)
    _, _, digest = load_and_verify_frozen_protocol(args, config, tokenizer)
    path = write_frozen_probe_heads(args.output_root, config, protocol_digest_value=digest)
    print(f"wrote frozen probe head manifest: {path}", flush=True)


def load_probe_for_evaluation(payload: Mapping[str, Any], *, device: Any) -> tuple[Any, dict[str, int], dict[int, str]]:
    import torch

    label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
    id_to_label = {index: label for label, index in label_to_id.items()}
    probe = FrozenTokenProbe(HIDDEN_SIZE, len(label_to_id)).to(device)
    probe.load_state_dict(payload["probe_state"])
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return probe, label_to_id, id_to_label


def stage_dev_evaluate(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    import torch

    output_dirs = ensure_output_dirs(args.output_root)
    result_path = dev_evaluate_results_path(output_dirs["root"])
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {result_path}")

    tokenizer = load_tokenizer(args.asset_root)
    frozen_protocol, _, digest = load_and_verify_frozen_protocol(args, config, tokenizer)
    frozen_heads, frozen_heads_sha256 = load_and_verify_frozen_probe_heads(
        output_dirs["root"],
        config,
        protocol_digest_value=digest,
    )
    head_by_key = {
        (str(row["pathway"]), int(row["seed"])): row
        for row in frozen_heads["heads"]
    }
    train = parse_phoner_split(
        args.data_root, PhoNERSplit.TRAIN, include_labels=True, stage="dev-evaluate"
    )
    dev = parse_phoner_split(
        args.data_root, PhoNERSplit.DEV, include_labels=True, stage="dev-evaluate"
    )
    train_label_to_id = {label: index for index, label in enumerate(label_inventory_from_train(train))}
    dev = materialize_condition_invariant_chunks(
        dev, tokenizer, config, purpose=CorruptionPurpose.SCIENTIFIC
    )
    runtime_inventory, inventory_identity, inventory_report = load_inventory_inputs()
    classifier = make_classifier(runtime_inventory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_encoder_passes = len(config.pathways) * len(config.conditions)
    completed_encoder_passes = 0
    records_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for pathway in config.pathways:
        loaded = load_pathway(pathway, args, inventory_identity=inventory_identity)
        if pathway is PhoNERPathway.PHOBERT_NATIVE:
            loaded.to(device)
        else:
            loaded.encoder.to(device)
            loaded.adapter.to(device)
        heads: dict[int, tuple[Any, dict[str, int], dict[int, str]]] = {}
        for seed in PHONER_FINAL_SEEDS:
            head_entry = head_by_key[(pathway.value, seed)]
            checkpoint_path = output_dirs["root"] / str(head_entry["relative_checkpoint_path"])
            payload = load_probe_checkpoint(checkpoint_path, map_location=device)
            probe, label_to_id, id_to_label = load_probe_for_evaluation(payload, device=device)
            if label_to_id != train_label_to_id:
                raise SystemExit(f"{checkpoint_path} label inventory does not match TRAIN-derived labels")
            heads[seed] = (probe, label_to_id, id_to_label)
        for condition in config.conditions:
            completed_encoder_passes += 1
            print(
                f"[dev-evaluate encoder {completed_encoder_passes}/{total_encoder_passes}] "
                f"pathway={pathway.value} condition={condition}",
                flush=True,
            )
            metrics_by_seed = evaluate_dev_fanout(
                dev,
                pathway=pathway,
                loaded=loaded,
                heads=heads,
                tokenizer=tokenizer,
                config=config,
                classifier=classifier,
                condition=condition,
                device=device,
            )
            f1s = " ".join(
                f"seed-{seed}={metrics_by_seed[seed]['entity_micro_f1']:.6f}"
                for seed in PHONER_FINAL_SEEDS
            )
            print(f"[dev-evaluate encoder {completed_encoder_passes}/{total_encoder_passes}] {f1s}", flush=True)
            for seed in PHONER_FINAL_SEEDS:
                records_by_key[(pathway.value, seed, condition)] = {
                    "pathway": pathway.value,
                    "seed": seed,
                    "condition": condition,
                    **metrics_by_seed[seed],
                }
        for probe, _label_to_id, _id_to_label in heads.values():
            probe.cpu()
        if pathway is PhoNERPathway.PHOBERT_NATIVE:
            loaded.cpu()
        else:
            loaded.encoder.cpu()
            loaded.adapter.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = [
        records_by_key[(pathway.value, seed, condition)]
        for pathway, seed, condition in dev_evaluation_plan(config)
    ]
    if completed_encoder_passes != total_encoder_passes or len(records) != 90:
        raise SystemExit(f"DEV evaluation must produce exactly 90 records, got {len(records)}")
    result = {
        "schema_version": DEV_EVALUATE_SCHEMA,
        "repository_head": repository_head(),
        "protocol_digest": digest,
        "frozen_protocol_schema_version": frozen_protocol["schema_version"],
        "frozen_probe_heads": {
            "path": str(frozen_probe_heads_path(output_dirs["root"])),
            "schema_version": frozen_heads["schema_version"],
            "sha256": frozen_heads_sha256,
            "payload_digest": canonical_payload_digest(dict(frozen_heads)),
        },
        "verified_stage1_identities": {
            "phobert_checkpoint": ENCODER_CHECKPOINT,
            "phobert_revision": ENCODER_REVISION,
            "gate_checkpoint_sha256": config.gate_checkpoint_sha256,
            "scale_checkpoint_sha256": config.scale_checkpoint_sha256,
            "inventory": inventory_identity.to_dict(),
        },
        "inventory_preflight": inventory_report,
        "records": records,
        "summaries": summarize_five_seed_scores(records),
    }
    write_json_once(result_path, result)


def stage_test_predict(args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    import torch

    output_dirs = ensure_output_dirs(args.output_root)
    require_stage_access("test-predict", split=PhoNERSplit.TEST, gold=False)
    test_examples = parse_phoner_split(
        args.data_root, PhoNERSplit.TEST, include_labels=False, stage="test-predict"
    )
    train_examples = parse_phoner_split(
        args.data_root, PhoNERSplit.TRAIN, include_labels=True, stage="test-predict"
    )
    labels = label_inventory_from_train(train_examples)
    tokenizer = load_tokenizer(args.asset_root)
    runtime_inventory, inventory_identity, inventory_report = load_inventory_inputs()
    classifier = make_classifier(runtime_inventory)
    output_path = Path(args.prediction_artifact or output_dirs["root"] / "test_predictions_sealed.json")
    runs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pathway in config.pathways:
        loaded = load_pathway(pathway, args, inventory_identity=inventory_identity)
        if pathway is PhoNERPathway.PHOBERT_NATIVE:
            loaded.to(device)
        else:
            loaded.encoder.to(device)
            loaded.adapter.to(device)
        for seed in PHONER_FINAL_SEEDS:
            checkpoint_path = output_dirs["checkpoints"] / f"{pathway.value}_seed-{seed}_best.pt"
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
            "verified_asset_identities": {
                "phobert_checkpoint": ENCODER_CHECKPOINT,
                "phobert_revision": ENCODER_REVISION,
                "gate_checkpoint_sha256": config.gate_checkpoint_sha256,
                "scale_checkpoint_sha256": config.scale_checkpoint_sha256,
                "inventory": inventory_identity.to_dict(),
            },
            "inventory_preflight": inventory_report,
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


def run_stage(stage: str, args: argparse.Namespace, config: PhoNERCrossTaskConfig) -> None:
    if stage == "dataset-audit":
        stage_audit(args, config)
    elif stage == "smoke-train":
        stage_train(args, config, smoke=True)
    elif stage == "train-dev":
        stage_train(args, config, smoke=False)
    elif stage == "dev-evaluate":
        stage_dev_evaluate(args, config)
    elif stage == "freeze-protocol":
        stage_freeze(args, config)
    elif stage == "freeze-heads":
        stage_freeze_heads(args, config)
    elif stage == "test-predict":
        stage_test_predict(args, config)
    elif stage == "test-score":
        stage_test_score(args, config)
    else:  # pragma: no cover
        raise AssertionError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhoNER frozen cross-task transfer probe.")
    parser.add_argument("--stage", required=True, choices=sorted({
        "audit", "dataset-audit", "smoke", "smoke-train", "train-dev",
        "dev-evaluate", "freeze", "freeze-protocol", "freeze-heads",
        "test-predict", "test-score",
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
    run_stage(stage, args, config)


if __name__ == "__main__":
    main()
