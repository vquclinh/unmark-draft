"""RESTORE baseline contracts.

These tests are synthetic and ML-free. They verify protocol identity,
source/artifact isolation, write-once resume behavior, and runner guardrails
without loading RESTORE, PhoBERT, UIT-VSFC, official validation, or official
TEST.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import pathlib
import statistics

import pytest

from unmark.baselines.restore.cache import (
    RestorePhaseRegistry,
    RestoreRepresentationCache,
    RestoreRepresentationKey,
    RestoreTextCache,
    RestoreTextCacheRequest,
    RestoreTextRecord,
    file_sha256,
    label_digest,
    semantic_text_digest,
)
from unmark.baselines.restore.config import (
    RESTORE_BEST_SEED_SELECTION,
    RESTORE_CONDITION_AWARE_ROUTING,
    RESTORE_CONDITIONS,
    RESTORE_CORRUPTION_SEED,
    RESTORE_CORRUPTED_LABEL_TRAINING,
    RESTORE_DEGRADED_CONDITIONS,
    RESTORE_EXPECTED_CHANGED_ROW_COUNTS,
    RESTORE_FULL_BYPASS,
    RESTORE_GENERATION_CONFIG,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_ONE_RESTORE_PATHWAY,
    RESTORE_OFFICIAL_TEST_READ,
    RESTORE_PHOBERT_CHECKPOINT,
    RESTORE_PHOBERT_DTYPE,
    RESTORE_PHOBERT_HIDDEN_SIZE,
    RESTORE_PHOBERT_POOLING,
    RESTORE_PHOBERT_REVISION,
    RESTORE_PROTOCOL_DEV_LABEL_DIGEST,
    RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST,
    RESTORE_PROTOCOL_DEV_ROWS,
    RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST,
    RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST,
    RESTORE_PROTOCOL_TRAIN_ROWS,
    RESTORE_RESTORER_FROZEN,
    RESTORE_RESTORER_BATCH_SIZE,
    RESTORE_RUNTIME_TRANSFORMERS_VERSION,
    RESTORE_STAGE2_HEAD_BATCH_SIZE,
    RESTORE_STAGE2_HEAD_EARLY_STOPPING,
    RESTORE_STAGE2_HEAD_EPOCHS,
    RESTORE_STAGE2_HEAD_LR,
    RESTORE_STAGE2_HEAD_SEEDS,
    RESTORE_STAGE2_MAX_LENGTH,
    RESTORE_STAGE2_PADDING,
    RESTORE_STAGE2_TRUNCATION,
    RESTORE_TOKENIZER_ARTIFACT_SHA256S,
    RESTORE_TOKENIZER_CONTRACT,
    RESTORE_VALIDATION_CLASS_COUNTS,
    RESTORE_VALIDATION_LABEL_DIGEST,
    RESTORE_VALIDATION_ORDERED_ID_DIGEST,
    RESTORE_VALIDATION_ROWS,
    RestoreGenerationConfig,
    RestoreTokenizerContract,
    preg1_identity_expectations,
    require_immutable_model_revision,
    require_restore_namespace,
    restore_artifact_root,
    validation_identity_expectations,
)
from unmark.baselines.restore.diagnostics import (
    RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT,
    RestoreDiagnosticRow,
    restore_diagnostics,
)
from unmark.baselines.restore.model import (
    RESTORE_DIRECT_MODEL_API,
    RESTORE_FALLBACKS_ENABLED,
    RESTORE_LLM_FALLBACK,
    RESTORE_RULE_BASED_FALLBACK,
    RESTORE_SPELL_CORRECTION_FALLBACK,
    FrozenDiacriticRestorer,
)
from unmark.baselines.restore.stage2 import (
    RESTORE_SCORE_UNITS,
    RestoreConditionScore,
    RestoreConditionStream,
    aggregate_restore_scores,
    authenticate_read_only_evidence,
    compute_grr,
    load_restore_validation_split,
    require_changed_row_counts,
    restore_records,
    restore_text_cache,
    validate_restore_head_artifact,
)
from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_head import EpochScore, Preg1Role, ordered_id_digest


REPO = pathlib.Path(__file__).resolve().parents[3]
HEAD = "a" * 40


def _source(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def _module_imports(path: pathlib.Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _rep_key(
    role: Preg1Role,
    *,
    condition: str = "FULL",
    corruption_seed: int | None = None,
    namespace: str = "RESTORE",
    count: int = 2,
) -> RestoreRepresentationKey:
    ids = [f"{role.value}-{condition}-{i}" for i in range(count)]
    labels = [i % 3 for i in range(count)]
    return RestoreRepresentationKey(
        repository_head=HEAD,
        role=role.value,
        condition=condition,
        corruption_seed=corruption_seed,
        text_cache_manifest_sha256="1" * 64,
        text_output_digest="2" * 64,
        ordered_id_digest=ordered_id_digest(ids),
        label_digest=label_digest(labels),
        count=count,
        representation_namespace=namespace,
    )


def _head_artifact() -> dict[str, object]:
    history = [
        {"epoch": epoch, "macro_f1": 0.5 + epoch / 1000, "accuracy": 0.6}
        for epoch in range(1, RESTORE_STAGE2_HEAD_EPOCHS + 1)
    ]
    return {
        "schema_version": "restore-stage2-baseline-v1",
        "protocol_version": "restore-baseline-protocol-v1",
        "repository_head": HEAD,
        "restore_model_id": RESTORE_MODEL_ID,
        "restore_model_revision": RESTORE_MODEL_REVISION,
        "seed": RESTORE_STAGE2_HEAD_SEEDS[0],
        "head_architecture": "Linear(768, 3, bias=True)",
        "optimizer": "AdamW",
        "learning_rate": RESTORE_STAGE2_HEAD_LR,
        "batch_size": RESTORE_STAGE2_HEAD_BATCH_SIZE,
        "epochs": RESTORE_STAGE2_HEAD_EPOCHS,
        "early_stopping": RESTORE_STAGE2_HEAD_EARLY_STOPPING,
        "selected_epoch": 30,
        "selected_clean_dev_macro_f1": 0.53,
        "selected_clean_dev_accuracy": 0.6,
        "history": history,
        "history_digest": "3" * 64,
        "initial_head_semantic_sha256": "4" * 64,
        "selected_state_semantic_sha256": "5" * 64,
        "selected_state_raw_sha256": "6" * 64,
        "train_cache_key": _rep_key(Preg1Role.PROTOCOL_TRAIN).to_dict(),
        "protocol_dev_cache_key": _rep_key(Preg1Role.PROTOCOL_DEV).to_dict(),
        "selection_role": Preg1Role.PROTOCOL_DEV.value,
        "selection_condition": "FULL",
        "official_validation_used_for_selection": False,
        "corrupted_label_training": RESTORE_CORRUPTED_LABEL_TRAINING,
        "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
    }


def _validation_stream(condition: str, changed_rows: int) -> RestoreConditionStream:
    clean = tuple(f"clean-{i}" for i in range(RESTORE_VALIDATION_ROWS))
    observed = tuple(
        f"observed-{i}" if i < changed_rows else clean[i]
        for i in range(RESTORE_VALIDATION_ROWS)
    )
    labels = tuple(i % 3 for i in range(RESTORE_VALIDATION_ROWS))
    return RestoreConditionStream(
        condition=condition,
        corruption_seed=RESTORE_CORRUPTION_SEED,
        sample_ids=tuple(f"validation-{i}" for i in range(RESTORE_VALIDATION_ROWS)),
        observed_texts=observed,
        clean_gold_texts=clean,
        labels=labels,
    )


def _score_grid() -> list[RestoreConditionScore]:
    scores = []
    for condition_index, condition in enumerate(RESTORE_CONDITIONS):
        for seed_index, seed in enumerate(RESTORE_STAGE2_HEAD_SEEDS):
            scores.append(
                RestoreConditionScore(
                    seed=seed,
                    condition=condition,
                    macro_f1=0.40 + condition_index / 100 + seed_index / 1000,
                    accuracy=0.60 + condition_index / 100 + seed_index / 1000,
                    per_class_f1=(0.2, 0.3, 0.4),
                )
            )
    return scores


def _condition_table(full: float, degraded: tuple[float, float, float, float, float]) -> dict[str, dict[str, float]]:
    values = {"FULL": full}
    values.update(dict(zip(RESTORE_DEGRADED_CONDITIONS, degraded)))
    return {
        condition: {"macro_f1": value, "accuracy": value}
        for condition, value in values.items()
    }


def _load_runner_module():
    path = REPO / "scripts" / "baselines" / "restore_stage2.py"
    spec = importlib.util.spec_from_file_location("restore_stage2_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_01_restore_package_does_not_import_unmark_adapter_or_stage1():
    forbidden = {"unmark.modeling.adapter", "unmark.stage1"}
    for path in (REPO / "unmark" / "baselines" / "restore").rglob("*.py"):
        imports = _module_imports(path)
        assert not any(
            name == blocked or name.startswith(blocked + ".")
            for name in imports
            for blocked in forbidden
        ), path


def test_02_exact_restore_model_id_is_frozen_in_code_and_spec():
    spec = json.loads(_source("docs/spec/restore-baseline-protocol.json"))
    assert RESTORE_MODEL_ID == "nrl-ai/vn-diacritic-vit5-base"
    assert spec["restore_model"]["model_id"] == RESTORE_MODEL_ID


def test_03_immutable_model_revision_is_required_and_main_is_refused():
    assert require_immutable_model_revision(RESTORE_MODEL_REVISION) == RESTORE_MODEL_REVISION
    with pytest.raises(EvaluationContractViolation, match="branch name"):
        require_immutable_model_revision("main")


def test_04_restorer_contract_requires_zero_trainable_parameters():
    class FakeParam:
        dtype = "torch.float32"
        device = "cpu"

        def __init__(self, trainable: bool = False) -> None:
            self.requires_grad = trainable

        def numel(self) -> int:
            return 1

    class FakeModel:
        training = False

        def __init__(self, trainable: bool = False) -> None:
            self.param = FakeParam(trainable)

        def named_parameters(self):
            return [("p", self.param)]

        def parameters(self):
            return iter([self.param])

    FrozenDiacriticRestorer(object(), FakeModel(False), expected_parameter_count=1)
    with pytest.raises(EvaluationContractViolation, match="trainable"):
        FrozenDiacriticRestorer(object(), FakeModel(True), expected_parameter_count=1)


def test_05_generation_config_is_deterministic_and_frozen():
    assert RESTORE_GENERATION_CONFIG.to_generate_kwargs() == {
        "max_length": 256,
        "do_sample": False,
        "num_beams": 1,
    }
    assert RestoreGenerationConfig() == RESTORE_GENERATION_CONFIG


def test_05a_restorer_tokenizer_contract_is_explicit_and_frozen():
    assert RESTORE_TOKENIZER_CONTRACT.to_load_kwargs() == {
        "use_fast": True,
        "legacy": True,
    }
    assert RESTORE_TOKENIZER_CONTRACT.to_encode_kwargs() == {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 256,
        "add_special_tokens": True,
        "return_attention_mask": True,
    }
    assert RESTORE_TOKENIZER_CONTRACT.to_decode_kwargs() == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    assert RESTORE_TOKENIZER_CONTRACT.transformers_version == RESTORE_RUNTIME_TRANSFORMERS_VERSION
    assert RestoreTokenizerContract() == RESTORE_TOKENIZER_CONTRACT


def test_05b_restorer_uses_tokenizer_contract_at_load_encode_and_decode():
    source = _source("unmark/baselines/restore/model.py")
    assert "**RESTORE_TOKENIZER_CONTRACT.to_load_kwargs()" in source
    assert "self.tokenizer(list(texts), **self.tokenizer_contract.to_encode_kwargs())" in source
    assert "**self.tokenizer_contract.to_decode_kwargs()" in source


def test_05c_restored_text_cache_identity_binds_tokenizer_generation_and_batch_size():
    ids = ("row-1",)
    texts = ("Toi yeu Viet Nam",)
    request = RestoreTextCacheRequest(
        repository_head=HEAD,
        role=Preg1Role.PROTOCOL_TRAIN.value,
        condition="FULL",
        corruption_seed=None,
        input_row_count=1,
        ordered_id_digest=ordered_id_digest(ids),
        input_text_digest=semantic_text_digest(texts),
    )
    payload = request.to_dict()
    assert payload["generation_config"] == RESTORE_GENERATION_CONFIG.to_dict()
    assert payload["tokenizer_contract"] == RESTORE_TOKENIZER_CONTRACT.to_dict()
    assert payload["tokenizer_artifact_sha256s"] == RESTORE_TOKENIZER_ARTIFACT_SHA256S
    assert payload["restore_batch_size"] == RESTORE_RESTORER_BATCH_SIZE

    drifted = RESTORE_TOKENIZER_CONTRACT.to_dict()
    drifted["decode"] = dict(drifted["decode"])
    drifted["decode"]["clean_up_tokenization_spaces"] = True
    with pytest.raises(EvaluationContractViolation, match="tokenizer contract"):
        RestoreTextCacheRequest(
            repository_head=HEAD,
            role=Preg1Role.PROTOCOL_TRAIN.value,
            condition="FULL",
            corruption_seed=None,
            input_row_count=1,
            ordered_id_digest=ordered_id_digest(ids),
            input_text_digest=semantic_text_digest(texts),
            tokenizer_contract=drifted,
        )


def test_05d_restored_text_cache_refuses_runtime_batch_size_drift(tmp_path):
    ids = ("row-1",)
    texts = ("toi yeu ngon ngu",)
    request = RestoreTextCacheRequest(
        repository_head=HEAD,
        role=Preg1Role.OFFICIAL_VALIDATION.value,
        condition="P50",
        corruption_seed=RESTORE_CORRUPTION_SEED,
        input_row_count=1,
        ordered_id_digest=ordered_id_digest(ids),
        input_text_digest=semantic_text_digest(texts),
        restore_batch_size=1,
    )
    with pytest.raises(EvaluationContractViolation, match="batch size drifted"):
        restore_text_cache(
            cache=RestoreTextCache(tmp_path / "P50"),
            request=request,
            restorer=object(),
            sample_ids=ids,
            texts=texts,
            batch_size=2,
        )


def test_06_sampling_controls_are_refused():
    for kwargs in (
        {"do_sample": True},
        {"num_beams": 2},
        {"temperature": 1.0},
        {"top_k": 50},
        {"top_p": 0.9},
    ):
        with pytest.raises(EvaluationContractViolation):
            RestoreGenerationConfig(**kwargs)


def test_07_condition_parameter_cannot_reach_restore_batch_or_generate():
    signature = inspect.signature(restore_records)
    assert "condition" not in signature.parameters
    assert "condition" not in inspect.signature(FrozenDiacriticRestorer.restore_batch).parameters

    class SpyRestorer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def restore_batch(self, texts):
            self.calls.append(tuple(texts))
            return [text + "|restored" for text in texts]

    restorer = SpyRestorer()
    records = restore_records(
        restorer,
        sample_ids=("a", "b", "c"),
        texts=("x", "y", "z"),
        batch_size=2,
    )
    assert restorer.calls == [("x", "y"), ("z",)]
    assert [record.restored_text for record in records] == [
        "x|restored",
        "y|restored",
        "z|restored",
    ]


def test_08_full_condition_is_not_bypassed(tmp_path):
    class ChangingRestorer:
        def __init__(self) -> None:
            self.calls = 0

        def restore_batch(self, texts):
            self.calls += 1
            return [text.upper() for text in texts]

    texts = ("clean input",)
    ids = ("row-1",)
    request = RestoreTextCacheRequest(
        repository_head=HEAD,
        role=Preg1Role.OFFICIAL_VALIDATION.value,
        condition="FULL",
        corruption_seed=RESTORE_CORRUPTION_SEED,
        input_row_count=1,
        ordered_id_digest=ordered_id_digest(ids),
        input_text_digest=semantic_text_digest(texts),
        restore_batch_size=1,
    )
    restorer = ChangingRestorer()
    records = restore_text_cache(
        cache=RestoreTextCache(tmp_path / "FULL"),
        request=request,
        restorer=restorer,
        sample_ids=ids,
        texts=texts,
        batch_size=1,
    )
    assert restorer.calls == 1
    assert records[0].restored_text == "CLEAN INPUT"


def test_09_restore_uses_direct_model_api_without_rule_llm_or_spell_fallbacks():
    assert RESTORE_DIRECT_MODEL_API == ("AutoTokenizer", "AutoModelForSeq2SeqLM")
    assert RESTORE_FALLBACKS_ENABLED is False
    assert RESTORE_RULE_BASED_FALLBACK is False
    assert RESTORE_LLM_FALLBACK is False
    assert RESTORE_SPELL_CORRECTION_FALLBACK is False
    assert "pipeline(" not in _source("unmark/baselines/restore/model.py")


def test_10_protocol_train_and_dev_identities_are_frozen():
    expectations = preg1_identity_expectations()
    assert expectations["protocol_train"] == {
        "rows": RESTORE_PROTOCOL_TRAIN_ROWS,
        "ordered_id_digest": RESTORE_PROTOCOL_TRAIN_ORDERED_ID_DIGEST,
        "label_digest": RESTORE_PROTOCOL_TRAIN_LABEL_DIGEST,
    }
    assert expectations["protocol_dev"] == {
        "rows": RESTORE_PROTOCOL_DEV_ROWS,
        "ordered_id_digest": RESTORE_PROTOCOL_DEV_ORDERED_ID_DIGEST,
        "label_digest": RESTORE_PROTOCOL_DEV_LABEL_DIGEST,
    }
    assert RESTORE_PROTOCOL_TRAIN_ROWS == 9139
    assert RESTORE_PROTOCOL_DEV_ROWS == 2285


def test_11_restore_uses_the_same_frozen_phobert_identity():
    assert RESTORE_PHOBERT_CHECKPOINT == "vinai/phobert-base"
    assert RESTORE_PHOBERT_REVISION == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert RESTORE_RESTORER_FROZEN is True


def test_12_restore_stage2_tokenization_and_pooling_semantics_match_audit063():
    assert RESTORE_STAGE2_MAX_LENGTH == 256
    assert RESTORE_STAGE2_TRUNCATION is True
    assert RESTORE_STAGE2_PADDING == "max_length"
    assert RESTORE_PHOBERT_POOLING == "FIRST_TOKEN"
    assert RESTORE_PHOBERT_HIDDEN_SIZE == 768
    assert RESTORE_PHOBERT_DTYPE == "torch.float32"


def test_13_restore_representation_namespace_is_distinct_from_vanilla_and_unmark():
    assert _rep_key(Preg1Role.PROTOCOL_TRAIN).representation_namespace == "RESTORE"
    with pytest.raises(EvaluationContractViolation, match="namespace"):
        _rep_key(Preg1Role.PROTOCOL_TRAIN, namespace="VANILLA")
    with pytest.raises(EvaluationContractViolation, match="namespace"):
        _rep_key(Preg1Role.PROTOCOL_TRAIN, namespace="UNMARK")


def test_14_five_exact_restore_head_seeds_are_frozen():
    assert RESTORE_STAGE2_HEAD_SEEDS == (53148, 59945, 42941, 720, 9428)


def test_15_restore_head_artifacts_require_all_30_epochs():
    artifact = _head_artifact()
    validate_restore_head_artifact(artifact, expected_seed=RESTORE_STAGE2_HEAD_SEEDS[0])
    short = dict(artifact)
    short["history"] = artifact["history"][:-1]
    with pytest.raises(EvaluationContractViolation, match="30 epochs"):
        validate_restore_head_artifact(short, expected_seed=RESTORE_STAGE2_HEAD_SEEDS[0])


def test_16_clean_protocol_dev_only_controls_within_head_epoch_selection():
    artifact = _head_artifact()
    official = dict(artifact)
    official["selection_role"] = Preg1Role.OFFICIAL_VALIDATION.value
    with pytest.raises(EvaluationContractViolation, match="wrong role"):
        validate_restore_head_artifact(official, expected_seed=RESTORE_STAGE2_HEAD_SEEDS[0])

    corrupted = dict(artifact)
    corrupted["selection_condition"] = "P50"
    with pytest.raises(EvaluationContractViolation, match="non-clean"):
        validate_restore_head_artifact(corrupted, expected_seed=RESTORE_STAGE2_HEAD_SEEDS[0])


def test_17_official_validation_cannot_be_read_before_heads_freeze(tmp_path):
    with pytest.raises(EvaluationContractViolation, match="before the five RESTORE heads freeze"):
        load_restore_validation_split(
            official_validation=tmp_path / "missing.csv",
            text_column="text",
            label_column="label",
            id_column="id",
            heads_frozen=False,
        )


def test_18_corruption_seed_is_exactly_19225():
    assert RESTORE_CORRUPTION_SEED == 19225


def test_19_six_restore_conditions_are_exact_and_ordered():
    assert RESTORE_CONDITIONS == ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")


def test_20_changed_row_counts_are_exact_and_fail_closed():
    streams = {
        condition: _validation_stream(condition, changed)
        for condition, changed in RESTORE_EXPECTED_CHANGED_ROW_COUNTS.items()
    }
    require_changed_row_counts(streams)
    drifted = dict(streams)
    drifted["P25"] = _validation_stream("P25", RESTORE_EXPECTED_CHANGED_ROW_COUNTS["P25"] - 1)
    with pytest.raises(EvaluationContractViolation, match="changed-row counts drifted"):
        require_changed_row_counts(drifted)


def test_21_six_validation_restored_text_caches_are_separate_and_identity_bound(tmp_path):
    ids = ("v-1", "v-2")
    texts = ("toi yeu ngon ngu", "du lieu sach")
    for condition in RESTORE_CONDITIONS:
        request = RestoreTextCacheRequest(
            repository_head=HEAD,
            role=Preg1Role.OFFICIAL_VALIDATION.value,
            condition=condition,
            corruption_seed=RESTORE_CORRUPTION_SEED,
            input_row_count=len(ids),
            ordered_id_digest=ordered_id_digest(ids),
            input_text_digest=semantic_text_digest(texts),
        )
        records = [
            RestoreTextRecord(ids[0], texts[0], texts[0] + "|r"),
            RestoreTextRecord(ids[1], texts[1], texts[1] + "|r"),
        ]
        cache = RestoreTextCache(tmp_path / condition)
        cache.save(request, records)
        manifest = cache.read_manifest()
        assert manifest["request"]["condition"] == condition
        assert manifest["request"]["restore_model_id"] == RESTORE_MODEL_ID
        assert manifest["request"]["restore_model_revision"] == RESTORE_MODEL_REVISION
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(RESTORE_CONDITIONS)


def test_22_six_validation_representation_caches_have_restore_keys(tmp_path):
    caches = {}
    for condition in RESTORE_CONDITIONS:
        key = _rep_key(
            Preg1Role.OFFICIAL_VALIDATION,
            condition=condition,
            corruption_seed=RESTORE_CORRUPTION_SEED,
        )
        cache = RestoreRepresentationCache(tmp_path / condition)
        caches[condition] = (cache, key)
        assert key.representation_namespace == "RESTORE"
        assert cache.manifest_path.parent.name == condition
    assert set(caches) == set(RESTORE_CONDITIONS)


def test_23_restore_measurement_requires_exactly_30_score_units():
    scores = _score_grid()
    aggregate_restore_scores(scores)
    with pytest.raises(EvaluationContractViolation, match="30 score units"):
        aggregate_restore_scores(scores[:-1])
    assert RESTORE_SCORE_UNITS == 30


def test_24_restore_aggregates_with_sample_standard_deviation():
    scores = _score_grid()
    aggregate = aggregate_restore_scores(scores)
    full_values = [score.macro_f1 for score in scores if score.condition == "FULL"]
    assert aggregate["conditions"]["FULL"]["macro_f1_sample_std"] == statistics.stdev(full_values)
    assert aggregate["sample_sd_aggregation"] is True


def test_25_grr_formula_has_no_epsilon_or_clipping():
    restore = {"conditions": _condition_table(1.0, (0.6, 0.6, 0.6, 0.6, 0.6))}
    vanilla = {"conditions": _condition_table(1.0, (0.2, 0.2, 0.2, 0.2, 0.2))}
    grr = compute_grr(restore_aggregate=restore, vanilla_evidence=vanilla)
    assert grr["formula"] == "(S_RESTORE - S_FLOOR) / (S_UPPER - S_FLOOR)"
    assert grr["no_epsilon"] is True
    assert grr["no_clipping"] is True
    assert grr["conditions"]["P25"]["macro_f1_grr"] == pytest.approx(0.5)


def test_26_headline_grr_averages_scores_first_then_applies_ratio_once():
    restore = {"conditions": _condition_table(1.0, (0.5, 0.8, 0.9, 0.5, 0.8))}
    vanilla = {"conditions": _condition_table(1.0, (0.2, 0.7, 0.8, 0.4, 0.6))}
    grr = compute_grr(restore_aggregate=restore, vanilla_evidence=vanilla)
    mean_restore = statistics.fmean([0.5, 0.8, 0.9, 0.5, 0.8])
    mean_floor = statistics.fmean([0.2, 0.7, 0.8, 0.4, 0.6])
    expected = (mean_restore - mean_floor) / (1.0 - mean_floor)
    condition_average = statistics.fmean(
        (r - f) / (1.0 - f)
        for r, f in zip((0.5, 0.8, 0.9, 0.5, 0.8), (0.2, 0.7, 0.8, 0.4, 0.6))
    )
    assert grr["headline_degraded"]["macro_f1_grr"] == pytest.approx(expected)
    assert grr["headline_degraded"]["macro_f1_grr"] != pytest.approx(condition_average)
    assert grr["headline_degraded"]["condition_grrs_averaged"] is False


def test_27_restoration_diagnostics_are_descriptive_only():
    assert RESTORE_DIAGNOSTICS_ARE_SELECTION_INPUT is False
    report = restore_diagnostics(
        [
            RestoreDiagnosticRow(
                sample_id="x",
                observed_text="toi",
                restored_text="toi",
                clean_gold_text="toi",
            )
        ],
        condition="FULL",
    )
    assert report["diagnostics_only"] is True
    assert report["selection_input"] is False
    assert "fraction_restore_leaves_clean_sentence_unchanged" in report
    assert "fraction_restore_changes_clean_sentence" in report


def test_28_vanilla_and_unmark_evidence_helpers_are_read_only(tmp_path):
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps({"conditions": _condition_table(1.0, (0.2, 0.2, 0.2, 0.2, 0.2))}), encoding="utf-8")
    before = file_sha256(source)
    payload = authenticate_read_only_evidence(source, before)
    assert payload["conditions"]["FULL"]["macro_f1"] == 1.0
    assert file_sha256(source) == before
    with pytest.raises(EvaluationContractViolation, match="SHA256 mismatch"):
        authenticate_read_only_evidence(source, "0" * 64)


def test_29_restore_namespace_cannot_collide_with_historical_namespaces(tmp_path):
    root = restore_artifact_root(tmp_path, HEAD)
    require_restore_namespace(root, tmp_path)
    assert "stage2-baselines" in root.parts
    assert "restore" in root.parts
    with pytest.raises(EvaluationContractViolation, match="outside"):
        require_restore_namespace(tmp_path / "stage2-measurement" / "x", tmp_path)


def test_30_resume_reuses_exact_completed_phase_artifacts(tmp_path):
    registry = RestorePhaseRegistry(tmp_path / "phase-state")
    calls = {"count": 0}
    identity = {"phase": "RESTORE_MODEL_PREFLIGHT", "head": HEAD}

    def action():
        calls["count"] += 1
        return {"ok": True}

    first = registry.run_once("RESTORE_MODEL_PREFLIGHT", identity, action)
    second = registry.run_once("RESTORE_MODEL_PREFLIGHT", identity, action)
    assert calls["count"] == 1
    assert first == second


def test_31_conflicting_completed_phase_artifacts_fail_closed(tmp_path):
    registry = RestorePhaseRegistry(tmp_path / "phase-state")
    registry.run_once("RESTORE_MODEL_PREFLIGHT", {"head": HEAD}, lambda: {"ok": True})
    with pytest.raises(EvaluationContractViolation, match="identity"):
        registry.is_complete("RESTORE_MODEL_PREFLIGHT", {"head": "b" * 40})


def test_32_interrupted_later_phase_preserves_earlier_valid_phase(tmp_path):
    registry = RestorePhaseRegistry(tmp_path / "phase-state")
    early_identity = {"phase": "RESTORE_CLEAN_TEXT", "head": HEAD}
    later_identity = {"phase": "RESTORE_GRR", "head": HEAD}
    registry.run_once("RESTORE_CLEAN_TEXT", early_identity, lambda: {"ok": True})
    registry.start("RESTORE_GRR", later_identity)
    assert registry.is_complete("RESTORE_CLEAN_TEXT", early_identity)
    assert registry.complete_path("RESTORE_CLEAN_TEXT").is_file()
    assert registry.in_progress_path("RESTORE_GRR").is_file()


def test_32a_phase_identity_binds_runtime_tokenizer_generation_and_batch_settings(tmp_path):
    module = _load_runner_module()

    class Args:
        drive_root = str(tmp_path)
        restore_batch_size = 3
        phobert_batch_size = 5

    runner = module.RestoreRunner(Args(), execution_head=HEAD)
    identity = runner.phase_identity("RESTORE_MODEL_PREFLIGHT")
    assert identity["restore_batch_size"] == 3
    assert identity["phobert_batch_size"] == 5
    assert identity["generation_config"] == RESTORE_GENERATION_CONFIG.to_dict()
    assert identity["tokenizer_contract"] == RESTORE_TOKENIZER_CONTRACT.to_dict()


def test_33_official_test_is_structurally_unreachable_from_restore_runner():
    module = _load_runner_module()
    parser = module.build_parser()
    options = [option for action in parser._actions for option in action.option_strings]
    assert not any(option.startswith("--test") for option in options)
    assert not any("TEST" in role.name or "test" in role.value for role in Preg1Role)
    assert RESTORE_OFFICIAL_TEST_READ is False


def test_34_run_all_uses_real_runner_without_production_mock_path(tmp_path, monkeypatch, capsys):
    module = _load_runner_module()
    parser = module.build_parser()
    options = [option for action in parser._actions for option in action.option_strings]
    assert "--mock-runtime" not in options
    assert not hasattr(module, "run_mock_runtime")

    seen = {}

    class FakeRunner:
        def __init__(self, args, *, execution_head: str) -> None:
            seen["drive_root"] = args.drive_root
            seen["execution_head"] = execution_head

        def run_all(self):
            seen["run_all"] = True
            return {
                "path": str(tmp_path / "restore-final-evidence.json"),
                "sha256": "0" * 64,
                "score_units": RESTORE_SCORE_UNITS,
                "official_test_read": False,
            }

    monkeypatch.setattr(module, "require_clean_git_head", lambda expected: expected)
    monkeypatch.setattr(module, "RestoreRunner", FakeRunner)
    code = module.main(
        [
            "--drive-root",
            str(tmp_path),
            "--run-all",
            "--expected-head",
            HEAD,
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert seen == {"drive_root": str(tmp_path), "execution_head": HEAD, "run_all": True}
    assert "RESTORE_BASELINE=PASS" in captured.out


def test_35_official_validation_gate_validates_head_closeout_before_loader(tmp_path, monkeypatch):
    module = _load_runner_module()

    class Args:
        drive_root = str(tmp_path)
        derived_train = None
        official_validation = None
        split_dir = None
        text_column = "text"
        label_column = "label"
        id_column = "id"
        restore_batch_size = 1
        phobert_batch_size = 1
        device = "cpu"

    runner = module.RestoreRunner(Args(), execution_head=HEAD)
    seed = RESTORE_STAGE2_HEAD_SEEDS[0]
    store = runner.artifact_root / "heads" / f"seed-{seed}"
    store.mkdir(parents=True)
    (store / "restore-head-artifact.json").write_text("{}", encoding="utf-8")
    (store / "restore-selected-head.pt").write_bytes(b"not-a-real-head")
    monkeypatch.setattr(
        module,
        "load_restore_validation_split",
        lambda **_: (_ for _ in ()).throw(AssertionError("validation loader called")),
    )
    with pytest.raises(EvaluationContractViolation, match="missing"):
        runner.validation_split()


def test_validation_identity_expectations_are_frozen_for_restore_boundary():
    expectations = validation_identity_expectations()
    assert expectations["rows"] == RESTORE_VALIDATION_ROWS == 1583
    assert expectations["class_counts"] == RESTORE_VALIDATION_CLASS_COUNTS
    assert expectations["ordered_id_digest"] == RESTORE_VALIDATION_ORDERED_ID_DIGEST
    assert expectations["label_digest"] == RESTORE_VALIDATION_LABEL_DIGEST
