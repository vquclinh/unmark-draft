import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from viunmark.assets import verify_model_assets
from viunmark.provenance import load_reproduction_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_completes_frozen_model_asset_contract():
    manifest = load_reproduction_manifest()
    assets = manifest["model_assets"]
    assert manifest["required_model_asset_count"] == 22
    assert manifest["stage1_checkpoint_count"] == 2
    assert manifest["stage2_head_count"] == 20
    assert len(assets) == 22
    assert sum(item["role"] == "stage1_checkpoint" for item in assets) == 2
    heads = [item for item in assets if item["role"] == "stage2_readout_head"]
    assert len(heads) == 20
    assert Counter(item["stream"] for item in heads) == {
        "gate_robust_readout": 5,
        "scale_unweighted_readout": 5,
        "scale_weighted_readout": 5,
        "phobert_readout": 5,
    }
    for item in assets:
        assert len(item["sha256"]) == 64
        assert item["present_in_public_repo"] is False
    for item in heads:
        assert isinstance(item["selected_seed"], int)
        assert isinstance(item["selected_boundary"], int)


def test_asset_verifier_reports_missing_and_mismatched_files(tmp_path):
    assets = [
        {
            "name": "missing",
            "relative_path": "missing.pt",
            "sha256": "0" * 64,
            "role": "stage1_checkpoint",
        },
        {
            "name": "bad-sha",
            "relative_path": "bad.pt",
            "sha256": "1" * 64,
            "role": "stage2_readout_head",
        },
    ]
    (tmp_path / "bad.pt").write_text("wrong", encoding="utf-8")
    report = verify_model_assets(assets, tmp_path)
    assert report["status"] == "missing_or_mismatched_assets"
    assert report["missing"][0]["expected_path"].endswith("missing.pt")
    assert report["missing"][0]["sha256"] == "0" * 64
    assert report["mismatched"][0]["name"] == "bad-sha"
    assert "actual_sha256" in report["mismatched"][0]


def test_missing_result_artifacts_are_not_silent_metric_sources():
    payload = json.loads((ROOT / "artifacts/results/uit_vsfc/missing_result_artifacts.json").read_text())
    assert payload["status"] == "canonical_numeric_result_files_not_present_in_current_git_repository"
    assert payload["artifacts"]
    for item in payload["artifacts"]:
        assert item["source_file_available_in_current_repository"] is False
        assert "metric_values" not in item
        assert item["known_sha256"]


def test_score_runner_requires_prediction_seal():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_predictions.py",
            "--predictions",
            "predictions.csv",
            "--labels",
            "labels.csv",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--prediction-manifest" in result.stderr


def test_score_runner_rejects_unsealed_manifest(tmp_path):
    predictions = tmp_path / "predictions.csv"
    labels = tmp_path / "labels.csv"
    seal = tmp_path / "seal.json"
    predictions.write_text("sample_id,condition,prediction\nx,clean,1\n", encoding="utf-8")
    labels.write_text("sample_id,label\nx,1\n", encoding="utf-8")
    seal.write_text(json.dumps({"sealed": False, "prediction_sha256": "0" * 64}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_predictions.py",
            "--predictions",
            str(predictions),
            "--prediction-manifest",
            str(seal),
            "--labels",
            str(labels),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "sealed=true" in result.stderr
