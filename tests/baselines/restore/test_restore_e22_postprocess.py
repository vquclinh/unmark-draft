"""Audit064 e22 post-measurement recovery executor contracts."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics
from typing import Any

import pytest

from unmark.baselines.restore.cache import file_sha256, read_json
from unmark.baselines.restore.config import (
    RESTORE_BASELINE_SCHEMA_VERSION,
    RESTORE_BEST_SEED_SELECTION,
    RESTORE_CONDITIONS,
    RESTORE_DEGRADED_CONDITIONS,
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_PROTOCOL_VERSION,
    RESTORE_STAGE2_HEAD_SEEDS,
)
from unmark.baselines.restore.evidence import extract_condition_metrics
from unmark.evaluation.contracts import EvaluationContractViolation


REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "baselines" / "restore_e22_postprocess.py"
REPAIR_HEAD = "f" * 40
SECOND_REPAIR_HEAD = "d" * 40

RESTORE_VALUES = {
    "FULL": (0.98, 0.99),
    "P25": (0.50, 0.60),
    "P50": (0.30, 0.50),
    "P75": (0.50, 0.70),
    "P100": (0.70, 0.80),
    "STRIP_ALL": (0.85, 0.95),
}
VANILLA_VALUES = {
    "FULL": (1.00, 1.00),
    "P25": (0.00, 0.10),
    "P50": (0.20, 0.30),
    "P75": (0.40, 0.50),
    "P100": (0.60, 0.70),
    "STRIP_ALL": (0.80, 0.90),
}
UNMARK_A_VALUES = {
    "FULL": (0.90, 0.92),
    "P25": (0.70, 0.72),
    "P50": (0.68, 0.70),
    "P75": (0.66, 0.68),
    "P100": (0.64, 0.66),
    "STRIP_ALL": (0.62, 0.64),
}


def _load_postprocess_module():
    spec = importlib.util.spec_from_file_location("restore_e22_postprocess_for_tests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _restore_measurement(values: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    values = values or RESTORE_VALUES
    conditions = {
        condition: {
            "macro_f1_mean": macro,
            "macro_f1_sample_std": 0.0,
            "accuracy_mean": accuracy,
            "accuracy_sample_std": 0.0,
            "per_class_f1_mean": [macro, macro, macro],
        }
        for condition, (macro, accuracy) in values.items()
    }
    return {
        "schema_version": RESTORE_BASELINE_SCHEMA_VERSION,
        "protocol_version": RESTORE_PROTOCOL_VERSION,
        "model_id": RESTORE_MODEL_ID,
        "model_revision": RESTORE_MODEL_REVISION,
        "score_units": len(RESTORE_STAGE2_HEAD_SEEDS) * len(RESTORE_CONDITIONS),
        "expected_score_units": len(RESTORE_STAGE2_HEAD_SEEDS) * len(RESTORE_CONDITIONS),
        "seeds": list(RESTORE_STAGE2_HEAD_SEEDS),
        "conditions": conditions,
        "per_seed": [
            {
                "seed": seed,
                "condition": condition,
                "macro_f1": conditions[condition]["macro_f1_mean"],
                "accuracy": conditions[condition]["accuracy_mean"],
                "per_class_f1": conditions[condition]["per_class_f1_mean"],
            }
            for seed in RESTORE_STAGE2_HEAD_SEEDS
            for condition in RESTORE_CONDITIONS
        ],
        "degraded_equal_weight": {
            "macro_f1_mean": statistics.fmean(values[c][0] for c in RESTORE_DEGRADED_CONDITIONS),
            "accuracy_mean": statistics.fmean(values[c][1] for c in RESTORE_DEGRADED_CONDITIONS),
        },
        "sample_sd_aggregation": True,
        "best_seed_selection": RESTORE_BEST_SEED_SELECTION,
    }


def _vanilla_evidence(values: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    values = values or VANILLA_VALUES
    return {
        "schema_version": "vanilla-upper-floor-final-v1",
        "conditions": list(RESTORE_CONDITIONS),
        "aggregate": {
            condition: {
                "macro_f1": {"mean": macro, "sample_sd": 0.0},
                "accuracy": {"mean": accuracy, "sample_sd": 0.0},
            }
            for condition, (macro, accuracy) in values.items()
        },
    }


def _unmark_a_evidence(values: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    values = values or UNMARK_A_VALUES
    return {
        "schema_version": "stage2-corrected-measurement-v3-final",
        "aggregate_report": {
            "schema_version": "stage2-head-campaign-v1",
            "arms": {
                "UNMARK-A": {
                    "conditions": {
                        condition: {
                            "macro_f1_mean": macro,
                            "macro_f1_std": 0.0,
                            "accuracy_mean": accuracy,
                            "accuracy_std": 0.0,
                        }
                        for condition, (macro, accuracy) in values.items()
                    }
                }
            },
        },
        "ab_selection_performed": False,
        "winner": None,
    }


def _install_evidence(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    measurement: dict[str, Any] | None = None,
    vanilla: dict[str, Any] | None = None,
    unmark: dict[str, Any] | None = None,
) -> pathlib.Path:
    drive_root = tmp_path / "drive"
    measurement_path = module.e22_measurement_path(drive_root)
    vanilla_path = drive_root / module.RESTORE_VANILLA_ANCHOR_EVIDENCE_RELATIVE
    unmark_path = drive_root / module.RESTORE_UNMARK_A_EVIDENCE_RELATIVE
    _write_json(measurement_path, measurement or _restore_measurement())
    _write_json(vanilla_path, vanilla or _vanilla_evidence())
    _write_json(unmark_path, unmark or _unmark_a_evidence())
    monkeypatch.setattr(module, "E22_RESTORE_MEASUREMENT_SHA256", file_sha256(measurement_path))
    monkeypatch.setattr(module, "RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256", file_sha256(vanilla_path))
    monkeypatch.setattr(module, "RESTORE_UNMARK_A_EVIDENCE_SHA256", file_sha256(unmark_path))
    return drive_root


def test_e22_postprocess_binds_real_production_identities():
    module = _load_postprocess_module()

    assert module.SOURCE_E22_HEAD == "e22c5ea8dbcbea4060193dca3f40fc97694fbb4a"
    assert module.E22_RESTORE_MEASUREMENT_SHA256 == (
        "c175d492834d5bb32e9423fd8f14fbd65874f1c22906f014dbd6727a0ba7b38f"
    )
    assert module.RESTORE_SCORE_UNITS == 30
    assert module.RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256 == (
        "d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862"
    )
    assert module.RESTORE_UNMARK_A_EVIDENCE_SHA256 == (
        "8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2"
    )


def test_e22_postprocess_script_has_no_scientific_executor_surface():
    module = _load_postprocess_module()
    source = SCRIPT.read_text(encoding="utf-8")

    forbidden_source = {
        "from unmark.baselines.restore.model",
        "from unmark.baselines.restore.stage2",
        "load_frozen_restorer",
        "load_restore_phobert_components",
        "train_or_load_restore_heads",
        "extract_or_load_restore_representations",
        "load_restore_protocol_splits",
        "load_restore_validation_split",
        "read_csv_restore_split",
        "load_derived_pool",
        "--run-all",
        "--derived-train",
        "--split-dir",
        "--official-validation",
        "--text-column",
        "--label-column",
        "--id-column",
        "--device",
    }
    for forbidden in forbidden_source:
        assert forbidden not in source

    option_strings = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    assert "--run-e22-postprocess" in option_strings
    assert "--expected-repair-head" in option_strings
    assert "--drive-root" in option_strings
    assert not {
        "--run-all",
        "--derived-train",
        "--split-dir",
        "--official-validation",
        "--text-column",
        "--label-column",
        "--id-column",
        "--device",
    } & option_strings


def test_e22_postprocess_writes_only_separate_postprocess_json_namespace(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    measurement = _restore_measurement()
    drive_root = _install_evidence(tmp_path, monkeypatch, module, measurement=measurement)
    inprogress = module.e22_scientific_namespace(drive_root) / "phase-state" / "RESTORE_GRR.inprogress"
    _write_json(inprogress, {"phase": "RESTORE_GRR"})

    result = module.run_postprocess(
        drive_root=drive_root,
        repair_head=REPAIR_HEAD,
        include_unmark_a=True,
    )

    output_root = pathlib.Path(result["output_root"])
    assert output_root == (
        drive_root
        / "stage2-baselines"
        / "restore-postprocess"
        / REPAIR_HEAD[:12]
        / "audit064-e22-closeout-v1"
    )
    assert {child.name for child in output_root.iterdir()} == {"comparison", "evidence", "grr"}
    for forbidden in module.POSTPROCESS_FORBIDDEN_DIRS:
        assert not (output_root / forbidden).exists()
    assert not (module.e22_scientific_namespace(drive_root) / "grr").exists()
    assert not (module.e22_scientific_namespace(drive_root) / "evidence").exists()
    assert inprogress.is_file()

    closeout = read_json(result["closeout"]["path"])
    assert closeout["scientific_measurement_generated_by_head"] == module.SOURCE_E22_HEAD
    assert closeout["postprocessing_generated_by_head"] == REPAIR_HEAD
    assert closeout["cross_head_scientific_execution_reuse"] == "NO"
    assert closeout["historical_measurement_consumed_read_only"] == "YES"
    assert closeout["restore_measurement"]["sha256"] == module.E22_RESTORE_MEASUREMENT_SHA256
    assert closeout["restore_measurement"]["score_units"] == 30
    assert closeout["restore_measurement"]["payload"] == measurement
    assert closeout["boundaries"]["model_loading_reachable"] == "NO"
    assert closeout["boundaries"]["phobert_loading_reachable"] == "NO"
    assert closeout["boundaries"]["head_training_reachable"] == "NO"
    assert closeout["boundaries"]["dataset_reading_reachable"] == "NO"
    assert closeout["boundaries"]["measurement_recomputed"] == "NO"


def test_e22_postprocess_refuses_exact_measurement_sha_mismatch(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    drive_root = _install_evidence(tmp_path, monkeypatch, module)
    monkeypatch.setattr(module, "E22_RESTORE_MEASUREMENT_SHA256", "0" * 64)

    with pytest.raises(EvaluationContractViolation, match="measurement SHA256 mismatch"):
        module.run_postprocess(
            drive_root=drive_root,
            repair_head=REPAIR_HEAD,
            include_unmark_a=True,
        )
    assert not module.postprocess_namespace(drive_root, REPAIR_HEAD).exists()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.__setitem__("score_units", 29), "score_units"),
        (lambda payload: payload["conditions"].pop("P100"), "conditions"),
        (
            lambda payload: payload["conditions"].__setitem__(
                "EXTRA",
                {"macro_f1_mean": 0.1, "accuracy_mean": 0.1},
            ),
            "conditions",
        ),
        (lambda payload: payload.__setitem__("repository_head", "0" * 40), "head mismatch"),
    ],
)
def test_e22_postprocess_refuses_bad_measurement_identity(
    tmp_path,
    monkeypatch,
    mutator,
    message,
):
    module = _load_postprocess_module()
    measurement = _restore_measurement()
    mutator(measurement)
    drive_root = _install_evidence(tmp_path, monkeypatch, module, measurement=measurement)

    with pytest.raises(EvaluationContractViolation, match=message):
        module.run_postprocess(
            drive_root=drive_root,
            repair_head=REPAIR_HEAD,
            include_unmark_a=True,
        )
    assert not module.postprocess_namespace(drive_root, REPAIR_HEAD).exists()


def test_e22_postprocess_refuses_vanilla_sha_mismatch(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    drive_root = _install_evidence(tmp_path, monkeypatch, module)
    monkeypatch.setattr(module, "RESTORE_VANILLA_ANCHOR_EVIDENCE_SHA256", "0" * 64)

    with pytest.raises(EvaluationContractViolation, match="evidence SHA256 mismatch"):
        module.run_postprocess(
            drive_root=drive_root,
            repair_head=REPAIR_HEAD,
            include_unmark_a=True,
        )
    assert not module.postprocess_namespace(drive_root, REPAIR_HEAD).exists()


def test_e22_postprocess_refuses_unmark_a_sha_mismatch_when_context_is_enabled(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    drive_root = _install_evidence(tmp_path, monkeypatch, module)
    monkeypatch.setattr(module, "RESTORE_UNMARK_A_EVIDENCE_SHA256", "0" * 64)

    with pytest.raises(EvaluationContractViolation, match="evidence SHA256 mismatch"):
        module.run_postprocess(
            drive_root=drive_root,
            repair_head=REPAIR_HEAD,
            include_unmark_a=True,
        )
    assert not module.postprocess_namespace(drive_root, REPAIR_HEAD).exists()


def test_e22_postprocess_refuses_running_from_source_scientific_head(monkeypatch):
    module = _load_postprocess_module()

    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return module.SOURCE_E22_HEAD
        return ""

    monkeypatch.setattr(module, "_git", fake_git)
    with pytest.raises(EvaluationContractViolation, match="committed repair HEAD"):
        module.require_repair_head(module.SOURCE_E22_HEAD)


def test_e22_postprocess_grr_uses_hand_calculated_scores_and_headline_rule(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    drive_root = _install_evidence(tmp_path, monkeypatch, module)

    result = module.run_postprocess(
        drive_root=drive_root,
        repair_head=REPAIR_HEAD,
        include_unmark_a=True,
    )
    grr = read_json(result["grr"]["path"])

    assert grr["conditions"]["P25"]["macro_f1_grr"] == pytest.approx(0.5)
    expected_headline = (
        statistics.fmean(RESTORE_VALUES[c][0] for c in RESTORE_DEGRADED_CONDITIONS)
        - statistics.fmean(VANILLA_VALUES[c][0] for c in RESTORE_DEGRADED_CONDITIONS)
    ) / (
        VANILLA_VALUES["FULL"][0]
        - statistics.fmean(VANILLA_VALUES[c][0] for c in RESTORE_DEGRADED_CONDITIONS)
    )
    condition_grr_mean = statistics.fmean(
        grr["conditions"][condition]["macro_f1_grr"]
        for condition in RESTORE_DEGRADED_CONDITIONS
    )
    assert grr["headline_degraded"]["macro_f1_grr"] == pytest.approx(expected_headline)
    assert abs(grr["headline_degraded"]["macro_f1_grr"] - condition_grr_mean) > 1e-6
    assert grr["headline_degraded"]["condition_grrs_averaged"] is False


def test_e22_postprocess_preserves_undefined_zero_denominator(tmp_path, monkeypatch):
    module = _load_postprocess_module()
    vanilla_values = dict(VANILLA_VALUES)
    vanilla_values["P25"] = vanilla_values["FULL"]
    drive_root = _install_evidence(
        tmp_path,
        monkeypatch,
        module,
        vanilla=_vanilla_evidence(vanilla_values),
    )

    result = module.run_postprocess(
        drive_root=drive_root,
        repair_head=SECOND_REPAIR_HEAD,
        include_unmark_a=False,
    )
    grr = read_json(result["grr"]["path"])

    assert grr["conditions"]["P25"]["macro_f1_grr"] == "UNDEFINED"
    assert grr["conditions"]["P25"]["accuracy_grr"] == "UNDEFINED"


def test_e22_postprocess_parser_supports_restore_vanilla_and_unmark_a_schemas():
    assert extract_condition_metrics(_restore_measurement()) == {
        condition: {
            "macro_f1": RESTORE_VALUES[condition][0],
            "accuracy": RESTORE_VALUES[condition][1],
        }
        for condition in RESTORE_CONDITIONS
    }
    assert extract_condition_metrics(_vanilla_evidence()) == {
        condition: {
            "macro_f1": VANILLA_VALUES[condition][0],
            "accuracy": VANILLA_VALUES[condition][1],
        }
        for condition in RESTORE_CONDITIONS
    }
    assert extract_condition_metrics(_unmark_a_evidence()) == {
        condition: {
            "macro_f1": UNMARK_A_VALUES[condition][0],
            "accuracy": UNMARK_A_VALUES[condition][1],
        }
        for condition in RESTORE_CONDITIONS
    }
