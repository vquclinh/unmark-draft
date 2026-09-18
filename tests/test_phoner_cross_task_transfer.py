"""Contracts for the PhoNER cross-task frozen NER probe.

No real PhoNER files, no model downloads, no official TEST scoring. The tests
use synthetic word-level files and tokenizer doubles to exercise the scientific
guards.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pathlib
import types

import pytest

from unmark.corruption import CorruptionPurpose
from unmark.cross_task import phoner_transfer as phoner
from unmark.stage1.preflight import InventoryIdentity

REPO = pathlib.Path(__file__).resolve().parents[1]

try:  # pragma: no cover - environment dependent
    import torch

    TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(not TORCH, reason="torch not installed")

RUNNER = importlib.import_module("scripts.cross_task.run_phoner_transfer")


class StubPhoBERTTokenizer:
    pad_token_id = 1
    unk_token_id = 3
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

    def tokenize(self, text: str) -> list[str]:
        mapping = {
            "Hà_Nội": ["Hà", "@@_Nội"],
            "Ha_Noi": ["Ha_Noi"],
            "thành_phố": ["thành", "@@_phố"],
            "hóa": ["h", "@@óa"],
            "hoa": ["hoa"],
            "Covid-19": ["Covid", "@@-19"],
        }
        return mapping.get(text, [text])

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 10
            ids.append(self.vocab[token])
        return ids

    def build_inputs_with_special_tokens(self, ids):
        return [0] + list(ids) + [2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


def write_conll(root: pathlib.Path) -> None:
    (root / "train_word.conll").write_text(
        "Tôi O\nở O\nHà_Nội B-LOC\n\nBệnh B-DISEASE\nnhân O\n",
        encoding="utf-8",
    )
    (root / "dev_word.conll").write_text(
        "Covid-19 B-DISEASE\nở O\nĐà_Nẵng B-LOC\n",
        encoding="utf-8",
    )
    (root / "test_word.conll").write_text(
        "Không O\nđọc O\nnhãn O\n",
        encoding="utf-8",
    )


def example() -> phoner.PhoNERExample:
    return phoner.PhoNERExample(
        sample_id="sample-a",
        split=phoner.PhoNERSplit.TRAIN,
        words=("Tôi", "đang", "học", "ở", "Hà_Nội"),
        labels=("O", "O", "O", "O", "B-LOC"),
        source_record_index=0,
        word_start=0,
        word_end=5,
    )


class FakeDatasetIdentity:
    def to_dict(self):
        return {
            "dataset_id": "PhoNER_COVID19",
            "representation": "word",
            "split_files": {
                "test": {
                    "sha256": "SEALED_UNREAD",
                    "row_count": "SEALED_UNREAD",
                    "gold_labels_read": False,
                }
            },
        }


def runner_args(tmp_path, data_root=None):
    return types.SimpleNamespace(
        data_root=data_root or tmp_path / "data",
        asset_root=tmp_path / "assets",
        output_root=tmp_path / "out",
        gate_checkpoint=None,
        scale_checkpoint=None,
        seed=phoner.PHONER_SMOKE_SEED,
        prediction_artifact=None,
    )


def fake_probe_payload(pathway, seed):
    return {
        "schema_version": "phoner-frozen-token-probe-checkpoint-v1",
        "pathway": pathway,
        "seed": seed,
        "update": 1000,
        "probe_state": {},
        "label_to_id": {"O": 0, "B-DISEASE": 1, "B-LOC": 2},
    }


def parse_checkpoint_name(name):
    pathway, rest = name.split("_seed-", 1)
    seed = int(rest.split("_best.pt", 1)[0])
    return pathway, seed


def prepare_expected_checkpoint_files(output_root):
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in RUNNER.EXPECTED_FROZEN_PROBE_HEAD_SHA256:
        (checkpoint_dir / name).write_bytes(f"checkpoint:{name}".encode("utf-8"))
    return checkpoint_dir


def frozen_heads_artifact(output_root):
    heads = []
    for pathway in phoner.PhoNERCrossTaskConfig().pathways:
        for seed in phoner.PHONER_FINAL_SEEDS:
            name = f"{pathway.value}_seed-{seed}_best.pt"
            heads.append(
                {
                    "pathway": pathway.value,
                    "seed": seed,
                    "relative_checkpoint_path": f"checkpoints/{name}",
                    "sha256": RUNNER.EXPECTED_FROZEN_PROBE_HEAD_SHA256[name],
                    "selected_update": 1000,
                    "checkpoint_schema_version": "phoner-frozen-token-probe-checkpoint-v1",
                }
            )
    return {
        "schema_version": RUNNER.FROZEN_PROBE_HEAD_SCHEMA,
        "repository_head": "a" * 40,
        "protocol_digest": "digest",
        "expected_head_count": 15,
        "heads": heads,
    }


def frozen_protocol_artifact():
    artifact = {
        "schema_version": "phoner-cross-task-frozen-protocol-v1",
        "protocol_frozen": True,
        "config": phoner.PhoNERCrossTaskConfig().to_dict(),
        "test_read_before_freeze": False,
        "test_scored_before_freeze": False,
        "dataset_identity": {
            "dataset_id": "PhoNER_COVID19",
            "representation": "word",
            "split_files": {
                "test": {
                    "sha256": "SEALED_UNREAD",
                    "row_count": "SEALED_UNREAD",
                    "gold_labels_read": False,
                }
            }
        },
    }
    artifact["protocol_digest"] = RUNNER.frozen_protocol_static_digest(artifact)
    return artifact


def test_conll_dataset_parser_and_stable_sample_ids(tmp_path):
    write_conll(tmp_path)
    first = phoner.parse_phoner_split(tmp_path, "train", stage="dataset-audit")
    second = phoner.parse_phoner_split(tmp_path, "train", stage="dataset-audit")
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert first[0].words == ("Tôi", "ở", "Hà_Nội")
    assert first[0].labels == ("O", "O", "B-LOC")
    assert first[0].sample_id.startswith("PhoNER_COVID19:word:train:00000000:")
    assert first[1].sample_id.startswith("PhoNER_COVID19:word:train:00000001:")
    assert len({item.sample_id for item in first}) == len(first)


def test_json_dataset_parser(tmp_path):
    payload = [{"words": ["Bệnh", "nhân"], "tags": ["B-DISEASE", "O"]}]
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}_word.json").write_text(json.dumps(payload), encoding="utf-8")
    train = phoner.parse_phoner_split(tmp_path, "train", stage="dataset-audit")
    assert train[0].words == ("Bệnh", "nhân")
    assert train[0].labels == ("B-DISEASE", "O")


def test_label_inventory_is_derived_from_train_only(tmp_path):
    write_conll(tmp_path)
    train = phoner.parse_phoner_split(tmp_path, "train", stage="dataset-audit")
    identity = phoner.build_phoner_dataset_identity(tmp_path, stage="dataset-audit")
    assert phoner.label_inventory_from_train(train) == ("O", "B-DISEASE", "B-LOC")
    assert identity.entity_label_inventory == ("O", "B-DISEASE", "B-LOC")
    assert "Đà_Nẵng" not in identity.entity_label_inventory
    assert identity.split_files["test"].sha256 == "SEALED_UNREAD"
    assert identity.split_files["test"].row_count == "SEALED_UNREAD"
    assert "prohibit redistribution" in identity.dataset_terms


def test_dataset_identity_does_not_hash_test_in_audit(tmp_path, monkeypatch):
    write_conll(tmp_path)
    original = phoner.sha256_file

    def guarded(path):
        assert pathlib.Path(path).name != "test_word.conll"
        return original(path)

    monkeypatch.setattr(phoner, "sha256_file", guarded)
    identity = phoner.build_phoner_dataset_identity(tmp_path, stage="dataset-audit")
    assert identity.split_files["test"].sha256 == "SEALED_UNREAD"


@pytest.mark.parametrize("condition", phoner.SIX_CONDITIONS)
def test_word_count_and_bio_labels_are_preserved_under_corruption(condition):
    clean = example()
    corrupted = phoner.corrupt_phoner_example(
        clean, condition, purpose=CorruptionPurpose.SELF_CHECK
    )
    assert len(corrupted.words) == len(clean.words)
    assert corrupted.labels == clean.labels
    assert corrupted.sample_id == clean.sample_id
    assert corrupted.split == clean.split


def test_cross_pathway_corrupted_input_identity():
    shared = phoner.cross_pathway_corrupted_inputs(
        example(), "P50", purpose=CorruptionPurpose.SELF_CHECK
    )
    assert set(shared) == {
        phoner.PhoNERPathway.PHOBERT_NATIVE,
        phoner.PhoNERPathway.VIUNMARK_GATE,
        phoner.PhoNERPathway.VIUNMARK_SCALE,
    }
    assert len({tuple(words) for words in shared.values()}) == 1


def test_corruption_parity_is_sample_level_not_per_word():
    clean = example()
    cfg = phoner.PhoNERCrossTaskConfig()
    sample_level = phoner.corrupt_phoner_example(
        clean, "P50", config=cfg, purpose=CorruptionPurpose.SELF_CHECK
    )
    whole = phoner.corrupt(
        " ".join(clean.words),
        "P50",
        seed=cfg.corruption_protocol.api_seed_for("P50"),
        sample_id=clean.sample_id,
        purpose=CorruptionPurpose.SELF_CHECK,
    ).corrupted_text.split(" ")
    per_word = [
        phoner.corrupt(
            word,
            "P50",
            seed=cfg.corruption_protocol.api_seed_for("P50"),
            sample_id=f"{clean.sample_id}:word-{index}",
            purpose=CorruptionPurpose.SELF_CHECK,
        ).corrupted_text
        for index, word in enumerate(clean.words)
    ]
    assert list(sample_level.words) == whole
    assert phoner.CORRUPTION_PARITY_CONCLUSION.startswith("PhoNER uses the same sample-level")
    assert list(sample_level.words) != per_word


def test_clean_subword_label_alignment_first_subtoken_policy():
    tokenizer = StubPhoBERTTokenizer()
    labels = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    aligned = phoner.align_word_labels(
        ["Tôi", "ở", "Hà_Nội"], ["O", "O", "B-LOC"], tokenizer, labels
    )
    assert aligned.special_tokens_mask == (1, 0, 0, 0, 0, 1)
    assert aligned.label_ids == (
        phoner.PHONER_IGNORE_INDEX,
        0,
        0,
        1,
        phoner.PHONER_IGNORE_INDEX,
        phoner.PHONER_IGNORE_INDEX,
    )
    assert aligned.word_ids == (None, 0, 1, 2, 2, None)


def test_corrupted_alignment_can_change_subword_count():
    tokenizer = StubPhoBERTTokenizer()
    labels = {"O": 0, "B-CHEM": 1}
    clean = phoner.align_word_labels(["hóa"], ["B-CHEM"], tokenizer, labels)
    corrupted = phoner.align_word_labels(["hoa"], ["B-CHEM"], tokenizer, labels)
    assert sum(1 for item in clean.special_tokens_mask if not item) == 2
    assert sum(1 for item in corrupted.special_tokens_mask if not item) == 1
    assert clean.label_ids[1] == corrupted.label_ids[1] == 1


def test_multi_syllable_phobert_word_special_tokens_and_truncation():
    tokenizer = StubPhoBERTTokenizer()
    labels = {"O": 0, "B-LOC": 1}
    aligned = phoner.align_word_labels(
        ["thành_phố", "Hà_Nội"], ["O", "B-LOC"], tokenizer, labels, max_length=4
    )
    assert len(aligned.input_ids) == 4
    assert aligned.truncated is True
    assert aligned.label_ids[0] == phoner.PHONER_IGNORE_INDEX
    assert aligned.label_ids[-1] == phoner.PHONER_IGNORE_INDEX
    assert aligned.word_ids == (None, 0, 0, None)


def test_condition_invariant_chunks_preserve_word_and_entity_coverage():
    tokenizer = StubPhoBERTTokenizer()
    sample = phoner.PhoNERExample(
        sample_id="long",
        split=phoner.PhoNERSplit.DEV,
        words=("Tôi", "ở", "Hà_Nội", "thành_phố", "hóa", "Covid-19"),
        labels=("O", "O", "B-LOC", "I-LOC", "B-CHEM", "B-DISEASE"),
        source_record_index=3,
        word_start=0,
        word_end=6,
    )
    cfg = phoner.PhoNERCrossTaskConfig(max_length=6)
    chunks = phoner.materialize_condition_invariant_chunks(
        [sample], tokenizer, cfg, purpose=CorruptionPurpose.SELF_CHECK
    )
    assert tuple(word for chunk in chunks for word in chunk.words) == sample.words
    assert sum(len(phoner.bio_entities(chunk.labels)) for chunk in chunks) == len(
        phoner.bio_entities(sample.labels)
    )
    audit = phoner.audit_condition_invariant_coverage(
        [sample], tokenizer, cfg, purpose=CorruptionPurpose.SELF_CHECK
    )
    assert audit["evaluated_word_entity_coverage_identical_across_conditions"] is True
    assert set(audit["max_subword_length_by_condition"]) == set(phoner.SIX_CONDITIONS)


@requires_torch
def test_encoder_freezing_guard_and_probe_architecture_identity():
    encoder = torch.nn.Linear(4, 4)
    phoner.freeze_module(encoder)
    phoner.require_frozen_module(encoder, "encoder")
    probes = {
        pathway: phoner.FrozenTokenProbe(4, 5)
        for pathway in (
            phoner.PhoNERPathway.PHOBERT_NATIVE,
            phoner.PhoNERPathway.VIUNMARK_GATE,
            phoner.PhoNERPathway.VIUNMARK_SCALE,
        )
    }
    phoner.require_identical_probe_architecture(probes)
    assert {tuple(p.classifier.weight.shape) for p in probes.values()} == {(5, 4)}


def test_no_sentiment_calibration_or_twenty_head_fusion_enters_protocol():
    cfg = phoner.PhoNERCrossTaskConfig().to_dict()
    controls = cfg["negative_controls"]
    assert controls["sentiment_mlp_used"] is False
    assert controls["mm_cat_pooling_used"] is False
    assert controls["uit_vsfc_class_weights_used"] is False
    assert controls["sentiment_calibration_used"] is False
    assert controls["final_sentiment_fusion_heads"] == 0
    assert cfg["probe_policy"]["architecture"] == "Linear(768, num_ner_labels)"
    assert cfg["probe_policy"]["scheduler"] == "none"
    assert cfg["probe_policy"]["warmup_updates"] == 0
    assert cfg["probe_policy"]["gradient_accumulation_steps"] == 1


def test_fresh_output_root_creates_checkpoint_directory_automatically(tmp_path):
    output_root = tmp_path / "fresh-output-root"
    assert not output_root.exists()
    paths = RUNNER.ensure_output_dirs(output_root)
    assert paths["root"] == output_root
    assert output_root.is_dir()
    assert paths["checkpoints"] == output_root / "checkpoints"
    assert paths["checkpoints"].is_dir()


def inventory_identity() -> InventoryIdentity:
    return InventoryIdentity(
        inventory_schema_version="v1",
        source_name="Vietnamese syllable inventory",
        source_author="undertheseanlp",
        source_revision="135a4d9716e49a981624474156d6f247b9b46f6a",
        sha256="78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",
        size_bytes=116290,
        license_status="NO_EXPLICIT_LICENSE",
    )


def test_gate_and_scale_loaders_receive_inventory_identity_not_runtime_inventory(monkeypatch, tmp_path):
    import unmark.evaluation.stage2_dual_finalist as dual_finalist
    import unmark.evaluation.stage2_scf_pathway as scf_pathway

    identity = inventory_identity()
    runtime_inventory = object()
    captured = {}

    def fake_gate_loader(arm, checkpoint, *, inventory, cache_dir):
        captured["gate"] = inventory
        return types.SimpleNamespace(kind="gate", arm=arm, checkpoint=checkpoint, cache_dir=cache_dir)

    def fake_scale_loader(checkpoint, *, inventory, cache_dir):
        captured["scale"] = inventory
        return types.SimpleNamespace(kind="scale", checkpoint=checkpoint, cache_dir=cache_dir)

    monkeypatch.setattr(RUNNER, "require_checkpoint_sha256", lambda *args, **kwargs: "verified")
    monkeypatch.setattr(dual_finalist, "load_frozen_unmark_pathway", fake_gate_loader)
    monkeypatch.setattr(scf_pathway, "load_frozen_scf_pathway", fake_scale_loader)
    args = types.SimpleNamespace(
        asset_root=tmp_path,
        gate_checkpoint=tmp_path / "gate.pt",
        scale_checkpoint=tmp_path / "scale.pt",
    )

    RUNNER.load_pathway(phoner.PhoNERPathway.VIUNMARK_GATE, args, inventory_identity=identity)
    RUNNER.load_pathway(phoner.PhoNERPathway.VIUNMARK_SCALE, args, inventory_identity=identity)

    assert captured["gate"] is identity
    assert captured["scale"] is identity
    assert captured["gate"] is not runtime_inventory
    assert captured["scale"] is not runtime_inventory
    assert captured["gate"].to_dict()["source_revision"] == "135a4d9716e49a981624474156d6f247b9b46f6a"
    assert captured["scale"].to_dict()["sha256"] == "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2"


def test_checkpoint_verification_remains_active_before_pathway_load(monkeypatch, tmp_path):
    import unmark.evaluation.stage2_dual_finalist as dual_finalist

    called = {"loader": False}

    def fail_verification(*args, **kwargs):
        raise phoner.PhoNERContractViolation("sha mismatch")

    def forbidden_loader(*args, **kwargs):
        called["loader"] = True
        raise AssertionError("loader must not run after checkpoint verification failure")

    monkeypatch.setattr(RUNNER, "require_checkpoint_sha256", fail_verification)
    monkeypatch.setattr(dual_finalist, "load_frozen_unmark_pathway", forbidden_loader)
    args = types.SimpleNamespace(
        asset_root=tmp_path,
        gate_checkpoint=tmp_path / "wrong-gate.pt",
        scale_checkpoint=tmp_path / "unused-scale.pt",
    )

    with pytest.raises(phoner.PhoNERContractViolation, match="sha mismatch"):
        RUNNER.load_pathway(
            phoner.PhoNERPathway.VIUNMARK_GATE,
            args,
            inventory_identity=inventory_identity(),
        )
    assert called["loader"] is False


def test_phoner_scientific_protocol_constants_unchanged():
    cfg = phoner.PhoNERCrossTaskConfig()
    policy = cfg.probe_policy
    assert phoner.PHONER_FINAL_SEEDS == (53148, 59945, 42941, 720, 9428)
    assert phoner.PHONER_CORRUPTION_SEED == 19225
    assert phoner.PHONER_GATE_SHA256 == "6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91"
    assert phoner.PHONER_SCALE_SHA256 == "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685"
    assert cfg.encoder_checkpoint == "vinai/phobert-base"
    assert cfg.encoder_revision == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert cfg.max_length == 256
    assert policy.architecture == "Linear(768, num_ner_labels)"
    assert policy.optimizer == "AdamW"
    assert policy.learning_rate == pytest.approx(1e-3)
    assert policy.betas == (0.9, 0.999)
    assert policy.eps == pytest.approx(1e-8)
    assert policy.weight_decay == pytest.approx(0.0)
    assert policy.batch_size == 16
    assert policy.gradient_accumulation_steps == 1
    assert policy.max_optimizer_updates == 1000
    assert policy.eval_every_updates == 100
    assert policy.scheduler == "none"
    assert policy.warmup_updates == 0
    assert policy.amp == "disabled"


def test_runner_test_stage_behavior_is_unchanged():
    source = (REPO / "scripts/cross_task/run_phoner_transfer.py").read_text(encoding="utf-8")
    assert 'require_stage_access("test-predict", split=PhoNERSplit.TEST, gold=False)' in source
    assert 'require_stage_access("test-score", split=PhoNERSplit.TEST, gold=True)' in source
    assert 'parse_phoner_split(\n        args.data_root, PhoNERSplit.TEST, include_labels=False, stage="test-predict"' in source
    assert "test_scores_from_sealed_predictions.json" in source


def test_dev_evaluate_dispatch_cannot_call_stage_train(monkeypatch, tmp_path):
    called = {}

    def forbidden_train(*args, **kwargs):
        raise AssertionError("dev-evaluate must not dispatch to stage_train")

    def fake_dev(args, config):
        called["dev"] = True

    monkeypatch.setattr(RUNNER, "stage_train", forbidden_train)
    monkeypatch.setattr(RUNNER, "stage_dev_evaluate", fake_dev)
    RUNNER.run_stage("dev-evaluate", runner_args(tmp_path), phoner.PhoNERCrossTaskConfig())
    assert called == {"dev": True}


def test_train_dev_dispatch_still_calls_stage_train(monkeypatch, tmp_path):
    called = {}

    def fake_train(args, config, *, smoke):
        called["smoke"] = smoke

    def forbidden_dev(*args, **kwargs):
        raise AssertionError("train-dev must not dispatch to stage_dev_evaluate")

    monkeypatch.setattr(RUNNER, "stage_train", fake_train)
    monkeypatch.setattr(RUNNER, "stage_dev_evaluate", forbidden_dev)
    RUNNER.run_stage("train-dev", runner_args(tmp_path), phoner.PhoNERCrossTaskConfig())
    assert called == {"smoke": False}


def test_dev_evaluate_stage_source_is_load_only():
    source = inspect.getsource(RUNNER.stage_dev_evaluate)
    assert "AdamW" not in source
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "torch.save" not in source
    assert "train_one(" not in source
    assert "PhoNERSplit.TEST" not in source


def test_dev_evaluate_refuses_missing_frozen_protocol(monkeypatch, tmp_path):
    args = runner_args(tmp_path)
    monkeypatch.setattr(RUNNER, "build_phoner_dataset_identity", lambda *a, **k: FakeDatasetIdentity())
    with pytest.raises(SystemExit, match="frozen_protocol.json"):
        RUNNER.load_and_verify_frozen_protocol(args, phoner.PhoNERCrossTaskConfig(), StubPhoBERTTokenizer())


def test_frozen_protocol_static_checks_refuse_digest_and_config_mismatch(tmp_path):
    args = runner_args(tmp_path)
    args.output_root.mkdir(parents=True)
    artifact = frozen_protocol_artifact()
    artifact["protocol_digest"] = "not-a-digest"
    (args.output_root / "frozen_protocol.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SystemExit, match="valid protocol_digest"):
        RUNNER.load_and_verify_frozen_protocol_artifact(args.output_root, phoner.PhoNERCrossTaskConfig())

    artifact = frozen_protocol_artifact()
    artifact["config"] = {**artifact["config"], "corruption_seed": 1}
    (args.output_root / "frozen_protocol.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SystemExit, match="config does not match"):
        RUNNER.load_and_verify_frozen_protocol_artifact(args.output_root, phoner.PhoNERCrossTaskConfig())


def test_dirty_execution_tree_fails_closed(monkeypatch):
    monkeypatch.setattr(RUNNER, "repository_head", lambda: "c" * 40)
    monkeypatch.setattr(RUNNER, "repository_status_short", lambda: " M scripts/cross_task/run_phoner_transfer.py\n")
    with pytest.raises(SystemExit, match="clean execution tree"):
        RUNNER.require_clean_execution_tree()


def test_dev_evaluate_refuses_missing_or_mismatching_frozen_probe_heads(monkeypatch, tmp_path):
    args = runner_args(tmp_path)
    with pytest.raises(SystemExit, match="required artifact is missing"):
        RUNNER.load_and_verify_frozen_probe_heads(
            args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
        )
    args.output_root.mkdir(parents=True)
    (args.output_root / "frozen_probe_heads.json").write_text(
        json.dumps({"schema_version": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="wrong schema"):
        RUNNER.load_and_verify_frozen_probe_heads(
            args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
        )


def test_old_frozen_manifest_producer_head_is_accepted_without_relabeling(monkeypatch, tmp_path):
    args = runner_args(tmp_path)
    prepare_expected_checkpoint_files(args.output_root)
    producer_head = "1" * 40
    execution_head = "2" * 40
    artifact = frozen_heads_artifact(args.output_root)
    artifact["repository_head"] = producer_head
    (args.output_root / "frozen_probe_heads.json").write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "repository_head", lambda: execution_head)
    monkeypatch.setattr(
        RUNNER,
        "sha256_file",
        lambda path: (
            "manifest-sha"
            if pathlib.Path(path).name == "frozen_probe_heads.json"
            else RUNNER.EXPECTED_FROZEN_PROBE_HEAD_SHA256[pathlib.Path(path).name]
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "load_probe_checkpoint",
        lambda path, *, map_location="cpu": fake_probe_payload(*parse_checkpoint_name(pathlib.Path(path).name)),
    )
    loaded, digest = RUNNER.load_and_verify_frozen_probe_heads(
        args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
    )
    assert loaded["repository_head"] == producer_head
    assert loaded["repository_head"] != execution_head
    assert digest == "manifest-sha"


def test_frozen_probe_heads_protocol_digest_mismatch_fails_closed(tmp_path):
    args = runner_args(tmp_path)
    args.output_root.mkdir(parents=True)
    (args.output_root / "frozen_probe_heads.json").write_text(
        json.dumps(frozen_heads_artifact(args.output_root)), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="protocol_digest"):
        RUNNER.load_and_verify_frozen_probe_heads(
            args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="other"
        )


def test_frozen_probe_heads_refuses_checkpoint_sha_mismatch(monkeypatch, tmp_path):
    args = runner_args(tmp_path)
    prepare_expected_checkpoint_files(args.output_root)
    monkeypatch.setattr(RUNNER, "repository_head", lambda: "test-head")

    def fake_sha(path):
        name = pathlib.Path(path).name
        if name == "PHOBERT_NATIVE_seed-53148_best.pt":
            return "0" * 64
        return RUNNER.EXPECTED_FROZEN_PROBE_HEAD_SHA256[name]

    def fake_load(path, *, map_location="cpu"):
        pathway, seed = parse_checkpoint_name(pathlib.Path(path).name)
        return fake_probe_payload(pathway, seed)

    monkeypatch.setattr(RUNNER, "sha256_file", fake_sha)
    monkeypatch.setattr(RUNNER, "load_probe_checkpoint", fake_load)
    with pytest.raises(SystemExit, match="SHA mismatch"):
        RUNNER.build_frozen_probe_heads_payload(
            args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
        )


def test_frozen_probe_heads_artifact_is_write_once(monkeypatch, tmp_path):
    args = runner_args(tmp_path)
    prepare_expected_checkpoint_files(args.output_root)
    monkeypatch.setattr(RUNNER, "repository_head", lambda: "test-head")
    monkeypatch.setattr(
        RUNNER,
        "sha256_file",
        lambda path: RUNNER.EXPECTED_FROZEN_PROBE_HEAD_SHA256[pathlib.Path(path).name],
    )
    monkeypatch.setattr(
        RUNNER,
        "load_probe_checkpoint",
        lambda path, *, map_location="cpu": fake_probe_payload(*parse_checkpoint_name(pathlib.Path(path).name)),
    )
    path = RUNNER.write_frozen_probe_heads(
        args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
    )
    assert path.is_file()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        RUNNER.write_frozen_probe_heads(
            args.output_root, phoner.PhoNERCrossTaskConfig(), protocol_digest_value="digest"
        )


def test_dev_evaluation_plan_is_exact_3_by_5_by_6():
    plan = RUNNER.dev_evaluation_plan(phoner.PhoNERCrossTaskConfig())
    assert len(plan) == 90
    assert {p for p, _, _ in plan} == set(phoner.PhoNERCrossTaskConfig().pathways)
    assert {seed for _, seed, _ in plan} == set(phoner.PHONER_FINAL_SEEDS)
    assert {condition for _, _, condition in plan} == set(phoner.SIX_CONDITIONS)


def test_stage_dev_evaluate_writes_dedicated_artifact_without_checkpoint_or_train_result_change(
    monkeypatch, tmp_path
):
    if not TORCH:
        pytest.skip("torch not installed")
    args = runner_args(tmp_path)
    args.data_root.mkdir(parents=True)
    write_conll(args.data_root)
    checkpoint_dir = prepare_expected_checkpoint_files(args.output_root)
    before_checkpoints = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in checkpoint_dir.iterdir()
    }
    train_results = args.output_root / "train_dev_results.json"
    train_results.write_text('{"do_not_touch": true}\n', encoding="utf-8")
    train_before = (train_results.read_bytes(), train_results.stat().st_mtime_ns)
    heads = frozen_heads_artifact(args.output_root)
    frozen_protocol_path = args.output_root / "frozen_protocol.json"
    frozen_protocol_path.write_text(json.dumps(frozen_protocol_artifact()), encoding="utf-8")
    frozen_heads_path = args.output_root / "frozen_probe_heads.json"
    frozen_heads_path.write_text(json.dumps(heads), encoding="utf-8")
    frozen_protocol_before = (frozen_protocol_path.read_bytes(), frozen_protocol_path.stat().st_mtime_ns)
    frozen_heads_before = (frozen_heads_path.read_bytes(), frozen_heads_path.stat().st_mtime_ns)
    parse_calls = []

    class FakeProbe:
        def __init__(self, *args, **kwargs):
            self.loaded = False

        def to(self, device):
            return self

        def load_state_dict(self, state):
            self.loaded = True

        def eval(self):
            return self

        def parameters(self):
            return []

        def cpu(self):
            return self

    class FakeNative:
        def to(self, device):
            return self

        def cpu(self):
            return self

    class FakeModule:
        def to(self, device):
            return self

        def cpu(self):
            return self

    class FakeAdapted:
        encoder = FakeModule()
        adapter = FakeModule()

    def fake_parse(root, split, *, include_labels=True, stage):
        parse_calls.append(split)
        assert split is not phoner.PhoNERSplit.TEST
        if split is phoner.PhoNERSplit.TRAIN:
            return phoner.parse_phoner_split(root, split, include_labels=include_labels, stage=stage)
        return phoner.parse_phoner_split(root, split, include_labels=include_labels, stage=stage)

    def fake_load_probe(path, *, map_location="cpu"):
        pathway, seed = parse_checkpoint_name(pathlib.Path(path).name)
        return fake_probe_payload(pathway, seed)

    def fake_load_pathway(pathway, args, *, inventory_identity):
        if pathway is phoner.PhoNERPathway.PHOBERT_NATIVE:
            return FakeNative()
        return FakeAdapted()

    def fake_evaluate(*args, condition, **kwargs):
        offset = phoner.SIX_CONDITIONS.index(condition)
        f1 = 0.5 + 0.01 * offset
        return {
            seed: {
                "entity_true_positive": 10 + offset,
                "entity_false_positive": 2,
                "entity_false_negative": 3,
                "entity_precision": f1,
                "entity_recall": f1,
                "entity_micro_f1": f1,
            }
            for seed in phoner.PHONER_FINAL_SEEDS
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("dev-evaluate reached a training/write API")

    def fake_verify_local(args, frozen_protocol):
        train = phoner.parse_phoner_split(args.data_root, phoner.PhoNERSplit.TRAIN, include_labels=True, stage="dev-evaluate")
        dev = phoner.parse_phoner_split(args.data_root, phoner.PhoNERSplit.DEV, include_labels=True, stage="dev-evaluate")
        parse_calls.extend([phoner.PhoNERSplit.TRAIN, phoner.PhoNERSplit.DEV])
        labels = {label: index for index, label in enumerate(phoner.label_inventory_from_train(train))}
        return train, dev, labels

    monkeypatch.setattr(RUNNER, "load_tokenizer", lambda *a, **k: StubPhoBERTTokenizer())
    monkeypatch.setattr(RUNNER, "require_clean_execution_tree", lambda: "b" * 40)
    monkeypatch.setattr(RUNNER, "load_and_verify_frozen_protocol_artifact", lambda *a, **k: ({"schema_version": "ok"}, "digest"))
    monkeypatch.setattr(RUNNER, "verify_stage1_checkpoint_bytes", lambda *a, **k: {
        "gate_checkpoint_sha256": phoner.PHONER_GATE_SHA256,
        "scale_checkpoint_sha256": phoner.PHONER_SCALE_SHA256,
    })
    monkeypatch.setattr(RUNNER, "load_and_verify_frozen_probe_heads", lambda *a, **k: (heads, "heads-sha"))
    monkeypatch.setattr(RUNNER, "verify_local_train_dev_identity", fake_verify_local)
    monkeypatch.setattr(RUNNER, "materialize_condition_invariant_chunks", lambda rows, *a, **k: rows)
    monkeypatch.setattr(RUNNER, "load_inventory_inputs", lambda: (object(), inventory_identity(), {"ok": True}))
    monkeypatch.setattr(RUNNER, "make_classifier", lambda inventory: None)
    monkeypatch.setattr(RUNNER, "load_pathway", fake_load_pathway)
    monkeypatch.setattr(RUNNER, "load_probe_checkpoint", fake_load_probe)
    monkeypatch.setattr(RUNNER, "FrozenTokenProbe", FakeProbe)
    monkeypatch.setattr(RUNNER, "evaluate_dev_fanout", fake_evaluate)
    monkeypatch.setattr(RUNNER, "repository_head", lambda: "test-head")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)

    RUNNER.stage_dev_evaluate(args, phoner.PhoNERCrossTaskConfig())

    result_path = args.output_root / "dev_evaluate_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == RUNNER.DEV_EVALUATE_SCHEMA
    assert result["execution_repository_head"] == "b" * 40
    assert len(result["records"]) == 90
    assert result["frozen_probe_heads"]["sha256"] == "heads-sha"
    assert result["frozen_probe_heads"]["producer_repository_head"] == "a" * 40
    assert result["frozen_probe_heads"]["producer_repository_head"] != result["execution_repository_head"]
    assert "FULL_robustness_retention_mean" in result["summaries"]["robustness"]["PHOBERT_NATIVE"]
    assert phoner.PhoNERSplit.TEST not in parse_calls
    for path in checkpoint_dir.iterdir():
        before_bytes, before_mtime = before_checkpoints[path.name]
        assert path.read_bytes() == before_bytes
        assert path.stat().st_mtime_ns == before_mtime
    assert (train_results.read_bytes(), train_results.stat().st_mtime_ns) == train_before
    assert (frozen_protocol_path.read_bytes(), frozen_protocol_path.stat().st_mtime_ns) == frozen_protocol_before
    assert (frozen_heads_path.read_bytes(), frozen_heads_path.stat().st_mtime_ns) == frozen_heads_before
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        RUNNER.stage_dev_evaluate(args, phoner.PhoNERCrossTaskConfig())


@requires_torch
def test_dev_evaluate_fanout_is_exactly_equivalent_to_reference_seed_loop(monkeypatch):
    cfg = phoner.PhoNERCrossTaskConfig()
    label_to_id = {"O": 0, "B-LOC": 1, "B-DISEASE": 2}
    id_to_label = {index: label for label, index in label_to_id.items()}
    examples = [object(), object()]
    device = torch.device("cpu")
    counts = {"hidden": 0}

    class FakeProbe:
        def __init__(self, label_id):
            self.label_id = label_id

        def eval(self):
            return self

        def __call__(self, hidden):
            logits = torch.zeros(hidden.shape[0], hidden.shape[1], len(label_to_id))
            logits[:, :, self.label_id] = 10.0
            return logits

    def fake_encode(chunk, **kwargs):
        return tuple(
            types.SimpleNamespace(
                words=("token_a", "token_b"),
                labels=("B-LOC", "O"),
                word_ids=(None, 0, 1, None),
            )
            for _ in chunk
        )

    def fake_collate(encoded, pad_token_id):
        return {"batch_size": len(encoded)}

    def fake_hidden(pathway, loaded, batch, device):
        counts["hidden"] += 1
        return torch.zeros(batch["batch_size"], 4, 1)

    monkeypatch.setattr(RUNNER, "encode_examples", fake_encode)
    monkeypatch.setattr(RUNNER, "collate_tokenized_ner_batch", fake_collate)
    monkeypatch.setattr(RUNNER, "pathway_hidden_states", fake_hidden)
    heads = {
        seed: (FakeProbe(index % len(label_to_id)), label_to_id, id_to_label)
        for index, seed in enumerate(phoner.PHONER_FINAL_SEEDS)
    }

    reference_records = []
    counts["hidden"] = 0
    for pathway in cfg.pathways:
        for seed in phoner.PHONER_FINAL_SEEDS:
            probe, seed_label_to_id, seed_id_to_label = heads[seed]
            for condition in cfg.conditions:
                reference_records.append(
                    {
                        "pathway": pathway.value,
                        "seed": seed,
                        "condition": condition,
                        **RUNNER.evaluate_dev(
                            examples,
                            pathway=pathway,
                            loaded=object(),
                            probe=probe,
                            tokenizer=StubPhoBERTTokenizer(),
                            label_to_id=seed_label_to_id,
                            id_to_label=seed_id_to_label,
                            config=cfg,
                            classifier=None,
                            condition=condition,
                            device=device,
                        ),
                    }
                )
    reference_hidden_calls = counts["hidden"]

    fanout_by_key = {}
    counts["hidden"] = 0
    for pathway in cfg.pathways:
        for condition in cfg.conditions:
            by_seed = RUNNER.evaluate_dev_fanout(
                examples,
                pathway=pathway,
                loaded=object(),
                heads=heads,
                tokenizer=StubPhoBERTTokenizer(),
                config=cfg,
                classifier=None,
                condition=condition,
                device=device,
            )
            for seed in phoner.PHONER_FINAL_SEEDS:
                fanout_by_key[(pathway.value, seed, condition)] = {
                    "pathway": pathway.value,
                    "seed": seed,
                    "condition": condition,
                    **by_seed[seed],
                }
    fanout_records = [
        fanout_by_key[(pathway.value, seed, condition)]
        for pathway, seed, condition in RUNNER.dev_evaluation_plan(cfg)
    ]

    assert reference_records == fanout_records
    assert RUNNER.summarize_five_seed_scores(reference_records) == RUNNER.summarize_five_seed_scores(fanout_records)
    assert reference_hidden_calls == 90
    assert counts["hidden"] == 18


def test_entity_level_f1_exact_span_and_type():
    gold = [["B-PER", "I-PER", "O", "B-LOC"], ["B-DISEASE"]]
    pred = [["B-PER", "I-PER", "O", "B-ORG"], ["O"]]
    score = phoner.EntityF1.from_sequences(gold, pred)
    assert score.true_positive == 1
    assert score.false_positive == 1
    assert score.false_negative == 2
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1 / 3)
    assert score.f1 == pytest.approx(0.4)


def reference_conll_entities(labels):
    return phoner.bio_entities(labels)


def test_invalid_bio_predictions_are_seqeval_compatible_chunks():
    cases = [
        (["I-PER"], {(0, 1, "PER")}),
        (["O", "I-PER"], {(1, 2, "PER")}),
        (["B-ORG", "I-PER"], {(0, 1, "ORG"), (1, 2, "PER")}),
    ]
    for labels, expected in cases:
        assert phoner.bio_entities(labels) == expected
        assert phoner.bio_entities(labels) == reference_conll_entities(labels)


def test_corruption_summary_metrics():
    values = {
        "FULL": 0.8,
        "P25": 0.7,
        "P50": 0.6,
        "P75": 0.5,
        "P100": 0.4,
        "STRIP_ALL": 0.2,
    }
    summary = phoner.corruption_summary_metrics(values)
    assert summary["corrupt_avg_f1"] == pytest.approx(0.48)
    assert summary["all_6_f1"] == pytest.approx(0.5333333333)
    assert summary["full_to_strip_absolute_drop"] == pytest.approx(0.6)
    assert summary["STRIP_ALL_robustness_retention"] == pytest.approx(0.25)


def test_five_seed_reporting_summarizes_without_best_seed_or_ensembling():
    rows = []
    for seed, offset in zip(phoner.PHONER_FINAL_SEEDS, range(5)):
        for condition_index, condition in enumerate(phoner.SIX_CONDITIONS):
            f1 = 0.5 + 0.01 * offset + 0.001 * condition_index
            rows.append(
                {
                    "pathway": "PHOBERT_NATIVE",
                    "seed": seed,
                    "condition": condition,
                    "entity_precision": f1,
                    "entity_recall": f1,
                    "entity_micro_f1": f1,
                }
            )
    summary = phoner.summarize_five_seed_scores(rows)
    full = summary["conditions"]["PHOBERT_NATIVE:FULL"]
    assert full["seeds"] == sorted(phoner.PHONER_FINAL_SEEDS)
    assert full["entity_micro_f1_mean"] == pytest.approx(0.52)
    assert "no logit ensembling" in summary["primary_policy"]
    assert "corrupt_avg_f1_mean" in summary["robustness"]["PHOBERT_NATIVE"]


def test_test_guard_and_prediction_scoring_separation(tmp_path):
    write_conll(tmp_path)
    with pytest.raises(phoner.PhoNERTestSealViolation):
        phoner.parse_phoner_split(tmp_path, "test", include_labels=False, stage="dataset-audit")
    with pytest.raises(phoner.PhoNERTestSealViolation):
        phoner.parse_phoner_split(tmp_path, "test", include_labels=True, stage="test-predict")
    text_only = phoner.parse_phoner_split(tmp_path, "test", include_labels=False, stage="test-predict")
    assert text_only[0].labels is None
    scored = phoner.parse_phoner_split(tmp_path, "test", include_labels=True, stage="test-score")
    assert scored[0].labels == ("O", "O", "O")
    assert [item.sample_id for item in text_only] == [item.sample_id for item in scored]


def test_checkpoint_sha256_and_phobert_identity_guards(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"asset")
    digest = phoner.sha256_file(path)
    assert phoner.require_checkpoint_sha256(path, digest, label="asset") == digest
    with pytest.raises(phoner.PhoNERContractViolation):
        phoner.require_checkpoint_sha256(path, "0" * 64, label="asset")
    class Obj:
        name_or_path = phoner.ENCODER_CHECKPOINT
        _commit_hash = phoner.ENCODER_REVISION
    phoner.require_phobert_identity(tokenizer=Obj())


def test_runner_exposes_required_stages_and_roots():
    source = (REPO / "scripts/cross_task/run_phoner_transfer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for flag in ("--data-root", "--asset-root", "--output-root"):
        assert flag in strings
    for stage in ("audit", "smoke", "train-dev", "dev-evaluate", "freeze", "freeze-heads", "test-predict", "test-score"):
        assert stage in source
    assert "/content/drive" not in source
    assert "MyDrive" not in source
    assert "\"words\"" not in source
    assert "contains_original_or_corrupted_text" in source
