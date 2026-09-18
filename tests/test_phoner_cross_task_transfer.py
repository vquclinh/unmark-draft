"""Contracts for the PhoNER cross-task frozen NER probe.

No real PhoNER files, no model downloads, no official TEST scoring. The tests
use synthetic word-level files and tokenizer doubles to exercise the scientific
guards.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from unmark.corruption import CorruptionPurpose
from unmark.cross_task import phoner_transfer as phoner

REPO = pathlib.Path(__file__).resolve().parents[1]

try:  # pragma: no cover - environment dependent
    import torch

    TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(not TORCH, reason="torch not installed")


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
    for stage in ("audit", "smoke", "train-dev", "freeze", "test-predict", "test-score"):
        assert stage in source
    assert "/content/drive" not in source
    assert "MyDrive" not in source
    assert "\"words\"" not in source
    assert "contains_original_or_corrupted_text" in source
