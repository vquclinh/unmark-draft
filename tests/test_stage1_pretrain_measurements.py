"""The measurement tool must measure and change nothing (Audit 030 §S)."""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SOURCE = pathlib.Path("scripts/stage1_pretrain_measurements.py").read_text(encoding="utf-8")


def test_it_cannot_update_a_parameter():
    tree = ast.parse(SOURCE)
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    for forbidden in ("backward", "step", "zero_grad", "AdamW", "build_optimizer",
                      "train_run", "execute_stage"):
        assert forbidden not in called, f"a measurement tool must not call {forbidden}()"


def test_it_changes_no_locked_constant():
    """It must READ the locked values, never redefine them."""
    for locked in ("BATCH_SIZE", "VALIDATION_CONDITIONS", "MAX_LENGTH",
                   "ENCODER_REVISION", "ENCODER_CHECKPOINT"):
        assert f"from unmark.stage1.protocol import" in SOURCE or locked in SOURCE
        assert f"{locked} =" not in SOURCE, f"{locked} must not be redefined here"


def test_it_verifies_the_corpus_before_measuring():
    assert "verify_prepared_corpus" in SOURCE


def test_it_streams_the_payload_rather_than_materialising_it():
    """The profile must not reintroduce the F4 pattern it exists to measure."""
    tree = ast.parse(SOURCE)
    profile = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "profile")
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(profile) if isinstance(n, ast.Call)
    }
    assert "load_prepared_chunks" not in called, "the profile must stream, not materialise"


def test_the_percentile_helper_is_correct():
    from scripts.stage1_pretrain_measurements import describe, percentile

    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile([], 50) == 0
    summary = describe([1, 2, 3, 4])
    assert summary["count"] == 4 and summary["max"] == 4
    assert summary["fraction_le_32"] == 1.0


def test_the_recomputed_token_profile_is_labelled_as_recomputed():
    """It must never be presented as a recorded length -- it is not persisted."""
    assert "recomputed_not_recorded" in SOURCE
    assert "are NOT persisted in chunks.jsonl" in SOURCE
