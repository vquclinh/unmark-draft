"""Measurement contract, torch-free (Audit 030 §U).

Its companion `test_stage1_validation_measurement.py` proves the real forwards
and parameter hashing and needs torch, so it is Colab-gated. **These run in the
ML-free venv**, and they are the ones that stop the §T defect returning: a
report that claims validation without having done any must fail here, on every
run, with no model anywhere.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.stage1_pretrain_measurements import (
    SAMPLING_SCHEME_VERSION,
    describe,
    profile,
    validation_failures,
)
from unmark.stage1.protocol import (
    BATCH_SIZE,
    EVAL_EVERY_UPDATES,
    VALIDATION_CONDITIONS,
    VALIDATION_CORRUPTION_SEED,
)

SOURCE = pathlib.Path("scripts/stage1_pretrain_measurements.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def function(name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(TREE)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node: ast.AST) -> set[str]:
    return {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(node) if isinstance(n, ast.Call)}


# ---------------------------------------------------------------------------
# It must reach the authoritative evaluator, and not reimplement validation
# ---------------------------------------------------------------------------
def test_validation_calls_the_authoritative_evaluate():
    """§T's defect exactly: `evaluate` imported and never called."""
    called = calls(function("validation_timing"))
    assert "evaluate" in called, "evaluate must be CALLED, not merely imported"
    assert "prepare_condition_batch" in called, "the real condition prep must be used"


def test_validation_does_not_reimplement_the_metric():
    """No second distance, pooling or aggregation inside the measurement tool."""
    for forbidden in ("representation_distance", "collate_stage1_batch",
                      "masked_mean", "cosine_similarity"):
        assert forbidden not in calls(TREE), f"{forbidden} must come from the evaluator"


def test_no_training_path_is_reachable():
    for forbidden in ("backward", "step", "zero_grad", "AdamW", "build_optimizer",
                      "train_run", "execute_stage", "save_training_checkpoint"):
        assert forbidden not in calls(TREE), f"measurement must not reach {forbidden}()"


def test_the_dead_placeholder_mode_is_gone():
    assert "condition_preparation_only_NOT_validation_wall_clock" not in SOURCE
    assert "--reference" not in SOURCE and "--collapse" not in SOURCE, (
        "flags must not be advertised unless implemented"
    )


def test_it_uses_the_locked_constants_rather_than_literals():
    for locked in ("BATCH_SIZE", "VALIDATION_CONDITIONS", "VALIDATION_CORRUPTION_SEED",
                   "EVAL_EVERY_UPDATES", "PRECISION"):
        assert f"{locked} =" not in SOURCE, f"{locked} must not be redefined here"
        assert locked in SOURCE


def test_cuda_timing_is_synchronised_and_memory_is_gpu_not_rss():
    source = ast.unparse(function("validation_timing"))
    assert "synchronize" in source
    assert "reset_peak_memory_stats" in source
    assert "max_memory_allocated" in source and "max_memory_reserved" in source
    assert "statm" not in source, "process RSS must never be reported as GPU memory"


def test_one_time_setup_is_separated_from_recurring_work():
    source = ast.unparse(function("validation_timing"))
    assert "one_time_condition_setup" in source
    assert "recurring_validation_total" in source
    assert "is NOT multiplied" in SOURCE, "the projection must exclude one-time setup"


# ---------------------------------------------------------------------------
# The report cannot claim success without the work (B7)
# ---------------------------------------------------------------------------
def healthy() -> dict:
    return {
        "forward_passes": 40,
        "conditions_executed": sorted(VALIDATION_CONDITIONS),
        "environment": {"device": "cuda:0", "cuda_synchronized_around_timing": True},
        "no_update_boundary": {
            "optimizer_constructed": False, "backward_calls": 0, "optimizer_steps": 0,
            "grad_enabled_during_forward": False, "outputs_requiring_grad": 0,
            "parameters_identical": True,
        },
    }


def test_a_healthy_report_passes():
    assert validation_failures(healthy()) == []


def test_zero_forwards_cannot_masquerade_as_validation():
    report = healthy(); report["forward_passes"] = 0
    assert any("no forward pass" in f for f in validation_failures(report))


@pytest.mark.parametrize("dropped", list(VALIDATION_CONDITIONS))
def test_a_missing_condition_fails_closed(dropped):
    report = healthy()
    report["conditions_executed"] = [c for c in VALIDATION_CONDITIONS if c != dropped]
    assert any("not executed" in f for f in validation_failures(report))


def test_parameter_mutation_fails_closed():
    report = healthy(); report["no_update_boundary"]["parameters_identical"] = False
    assert any("PARAMETERS CHANGED" in f for f in validation_failures(report))


def test_grad_enabled_fails_closed():
    report = healthy(); report["no_update_boundary"]["grad_enabled_during_forward"] = True
    assert any("no_grad was not active" in f for f in validation_failures(report))


def test_an_output_requiring_grad_fails_closed():
    report = healthy(); report["no_update_boundary"]["outputs_requiring_grad"] = 1
    assert validation_failures(report)


@pytest.mark.parametrize("field,value", [
    ("optimizer_constructed", True), ("optimizer_steps", 1), ("backward_calls", 1),
])
def test_any_update_evidence_fails_closed(field, value):
    report = healthy(); report["no_update_boundary"][field] = value
    assert validation_failures(report), field


def test_unsynchronised_cuda_timing_fails_closed():
    report = healthy(); report["environment"]["cuda_synchronized_around_timing"] = False
    assert any("not synchronized" in f for f in validation_failures(report))


def test_cpu_does_not_require_cuda_synchronisation():
    report = healthy()
    report["environment"] = {"device": "cpu", "cuda_synchronized_around_timing": False}
    assert validation_failures(report) == []


# ---------------------------------------------------------------------------
# Task D -- partition-aware deterministic sampling
# ---------------------------------------------------------------------------
def write_corpus(tmp_path, train: int, dev: int) -> pathlib.Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(train):
        rows.append({"chunk_id": f"t-{i:07d}#0", "document_id": f"t-{i // 3:07d}",
                     "partition": "train", "chunk_index": 0, "text": "a" * (10 + i % 40),
                     "source_start": 0, "source_end": 1, "source_shard": "train.parquet"})
    for i in range(dev):
        rows.append({"chunk_id": f"d-{i:07d}#0", "document_id": f"d-{i // 3:07d}",
                     "partition": "dev", "chunk_index": 0, "text": "b" * (10 + i % 40),
                     "source_start": 0, "source_end": 1, "source_shard": "train.parquet"})
    # Interleaved, so a shared counter would starve dev exactly as v1 did.
    rows.sort(key=lambda r: r["chunk_id"][2:])
    (prepared / "chunks.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return prepared


def test_a_dev_population_below_the_cap_is_taken_whole(tmp_path):
    """The real case: dev is 11 443 < 20 000, so every dev chunk is profiled."""
    prepared = write_corpus(tmp_path, train=5_000, dev=200)
    result = profile(prepared, sample=1_000)
    dev = result["sample_selection"]["per_partition"]["dev"]
    assert dev["population_count"] == 200
    assert dev["obtained_count"] == 200
    assert dev["complete_population"] is True
    assert dev["stride"] == 1


def test_train_takes_exactly_the_cap_and_spreads_over_its_population(tmp_path):
    prepared = write_corpus(tmp_path, train=5_000, dev=200)
    result = profile(prepared, sample=1_000)
    train = result["sample_selection"]["per_partition"]["train"]
    assert train["population_count"] == 5_000
    assert train["obtained_count"] == 1_000
    assert train["stride"] == 5
    ids = result["_sampled_ids"]["train"]
    # Spread, not clustered: the last selection is near the end of the partition.
    assert ids[0].endswith("#0") and len(ids) == 1_000
    assert ids[-1] > ids[len(ids) // 2] > ids[0]


def test_sampling_is_deterministic_across_repeated_runs(tmp_path):
    prepared = write_corpus(tmp_path, train=1_500, dev=90)
    first = profile(prepared, sample=100)
    second = profile(prepared, sample=100)
    assert first["_sampled_ids"] == second["_sampled_ids"]
    assert first["partitions"] == second["partitions"]


def test_sampling_is_partition_aware_not_a_shared_counter(tmp_path):
    """v1's defect: dev starved because one counter spanned both partitions."""
    prepared = write_corpus(tmp_path, train=10_000, dev=100)
    result = profile(prepared, sample=500)
    per = result["sample_selection"]["per_partition"]
    assert per["dev"]["obtained_count"] == 100, "dev must not be starved by train's volume"
    assert per["train"]["obtained_count"] == 500


def test_selection_does_not_depend_on_the_data(tmp_path):
    """Same shape, different text -> same selected positions."""
    a = write_corpus(tmp_path / "a", train=600, dev=60)
    b = write_corpus(tmp_path / "b", train=600, dev=60)
    (b / "chunks.jsonl").write_text(
        (b / "chunks.jsonl").read_text(encoding="utf-8").replace("aaaa", "zzzz"),
        encoding="utf-8",
    )
    first = profile(a, sample=50)["sample_selection"]["per_partition"]
    second = profile(b, sample=50)["sample_selection"]["per_partition"]
    assert first == second


def test_the_output_records_the_full_selection_identity(tmp_path):
    prepared = write_corpus(tmp_path, train=600, dev=60)
    selection = profile(prepared, sample=50)["sample_selection"]
    assert selection["sampling_scheme_version"] == SAMPLING_SCHEME_VERSION
    assert selection["deterministic"] is True
    assert selection["seed"] is None
    assert selection["data_independent"] is True
    for part in ("train", "dev"):
        entry = selection["per_partition"][part]
        for field in ("partition", "population_count", "requested_count",
                      "obtained_count", "stride"):
            assert field in entry, (part, field)


def test_no_python_hash_is_used_for_selection():
    """AST, not text: the docstring legitimately says "no `hash()`"."""
    assert "hash" not in calls(function("profile")), (
        "hash() is salted per process and is not reproducible"
    )


def test_token_lengths_stay_labelled_as_recomputed():
    assert "recomputed_not_recorded" in SOURCE
    assert "are NOT persisted in chunks.jsonl" in SOURCE
    token = function("token_profile")
    source = ast.unparse(token)
    for field in ("tokenizer_checkpoint", "tokenizer_revision", "transformers_version",
                  "pathways", "sampling_scheme_version"):
        assert field in source, field


def test_the_profile_still_streams_rather_than_materialising():
    assert "load_prepared_chunks" not in calls(function("profile"))


def test_percentiles_are_sane():
    summary = describe(list(range(1, 101)))
    assert summary["p50"] == 50 and summary["p99"] == 99 and summary["max"] == 100
