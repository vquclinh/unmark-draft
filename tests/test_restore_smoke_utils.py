"""Unit tests for the G-1 smoke-test support code.

These tests never download the 220M checkpoint and never import torch or
transformers. That is a policy, not a convenience: heavy ML work belongs on
Colab (see the README), and the local `.venv` installs only
`requirements/dev.txt`. Several tests below actively enforce that separation.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from unmark.gates import g_minus1 as g1
from unmark.orthography import base_signature

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "g_minus1_restore_smoke.py"


def _load_script_module():
    """Import the CLI script by path, as `python scripts/...` would."""
    spec = importlib.util.spec_from_file_location("g_minus1_restore_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Environment policy: the lightweight path must stay lightweight
# ---------------------------------------------------------------------------
# Modules that the offline test suite imports, and which must therefore work with
# `requirements/dev.txt` alone. This list is deliberately explicit: it is a
# statement about THESE MODULES, not a rule about the `unmark` package. Later
# phases add genuinely ML-shaped code (proposal 8.1 plans unmark/modules/,
# unmark/training/, unmark/baselines/) and that code is free to import PyTorch
# normally -- it simply does not go on this list.
LIGHTWEIGHT_MODULES = (
    "unmark.orthography.signature",
    "unmark.gates.g_minus1",
)

HEAVY_MODULES = ("torch", "transformers", "sentencepiece", "safetensors", "accelerate")


def _import_in_clean_subprocess(import_statement: str) -> list[str]:
    """Run `import_statement` in a fresh interpreter; return heavy modules loaded.

    Uses `sys.executable` -- the repo `.venv` when the suite runs there -- and a
    clean process, so nothing another test already imported can mask a result.
    """
    probe = (
        "import sys, json\n"
        f"{import_statement}\n"
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", LIGHTWEIGHT_MODULES)
def test_lightweight_module_imports_without_loading_the_heavy_stack(module):
    """Behavioural check: importing this module must not drag in the ML stack.

    Behavioural rather than a source scan, so it constrains only what it names.
    A future `unmark/modules/fusion.py` importing torch is fine and must stay
    fine; it is simply not on LIGHTWEIGHT_MODULES.
    """
    loaded = _import_in_clean_subprocess(f"import {module}")
    assert loaded == [], f"importing {module} loaded heavy modules: {loaded}"


def test_lightweight_modules_are_actually_usable_not_merely_importable():
    """Guards against the probe passing vacuously on a broken import."""
    probe = (
        "import unmark.orthography.signature as s, unmark.gates.g_minus1 as g; "
        "assert s.base_signature('Tôi') == 'Toi'; assert len(g.SMOKE_CASES) > 0"
    )
    assert _import_in_clean_subprocess(probe) == []


def test_cli_script_imports_without_loading_the_heavy_stack():
    """The CLI may *use* torch, but importing it must not require or load it."""
    statement = (
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('g1cli', {str(SCRIPT_PATH)!r}); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "assert mod.Restorer and mod.require_experiment_dependencies"
    )
    assert _import_in_clean_subprocess(statement) == []


def test_no_package_wide_source_scan_bans_ml_imports():
    """Regression guard for audit 001 / issue B1.

    A previous revision asserted that no file under `unmark/` may contain a torch
    or transformers import at any indentation. That banned future ML modules, and
    even lazy imports inside functions. If such a scan reappears, this test is
    the thing that should explain why it must not.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    def _is_source_tree_walk(node: ast.AST) -> bool:
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr in ("rglob", "glob"):  # pathlib
            return True
        # os.walk, but not ast.walk (which this very test uses)
        return node.attr == "walk" and isinstance(node.value, ast.Name) and node.value.id == "os"

    walks = [node for node in ast.walk(tree) if _is_source_tree_walk(node)]
    assert not walks, (
        "this test module walks the package source tree again; use the behavioural "
        "LIGHTWEIGHT_MODULES probe instead, so that future unmark/modules, "
        "unmark/training and unmark/baselines remain free to import torch"
    )


def test_script_imports_torch_and_transformers_only_lazily():
    """Scoped to the single file that must import without the ML stack present."""
    for lineno, line in enumerate(SCRIPT_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("import torch", "from torch", "import transformers", "from transformers")):
            assert line.startswith((" ", "\t")), f"{SCRIPT_PATH}:{lineno} heavy import at module level: {stripped}"


def test_script_module_exposes_the_experiment_guard():
    """Importing the CLI in-process must not require the experiment stack."""
    module = _load_script_module()
    assert hasattr(module, "require_experiment_dependencies")
    assert hasattr(module, "Restorer")


def test_missing_experiment_dependencies_fail_with_a_clear_message(monkeypatch):
    module = _load_script_module()
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(module.ExperimentDependenciesMissing) as excinfo:
        module.require_experiment_dependencies()
    message = str(excinfo.value)
    assert "G-1 model inference requires the experiment environment." in message
    assert "requirements/experiment.txt" in message
    assert "Google Colab" in message


def test_experiment_deps_message_does_not_offer_to_install_anything():
    """Nothing may be installed automatically, and the message must not imply it."""
    message = g1.EXPERIMENT_DEPS_MESSAGE
    assert "Nothing will be installed automatically." in message
    assert "requirements/experiment.txt" in message
    assert ".venv-colab" in message


def test_requirements_split_keeps_heavy_libraries_out_of_dev():
    dev = (REPO_ROOT / "requirements" / "dev.txt").read_text(encoding="utf-8")
    experiment = (REPO_ROOT / "requirements" / "experiment.txt").read_text(encoding="utf-8")
    for heavy in ("torch", "transformers", "sentencepiece", "safetensors", "accelerate"):
        for line in dev.splitlines():
            requirement = line.split("#")[0].strip()
            assert not requirement.startswith(heavy), f"{heavy} must not be a local dev dependency"
    assert "transformers==4.57.6" in experiment, "the incompatible-5.x finding must stay pinned"


def test_gitignore_excludes_environments_caches_and_results():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".venv/", ".venv-colab/", "venv/", ".hf-cache/", "results/**", "__pycache__/", ".pytest_cache/"):
        assert pattern in gitignore, f"missing .gitignore entry: {pattern}"
    # Order matters: git will not descend into an excluded directory, so the
    # directory must be re-included before the .gitkeep negation can match.
    assert "!results/**/" in gitignore
    assert "!results/**/.gitkeep" in gitignore
    assert gitignore.index("!results/**/") < gitignore.index("!results/**/.gitkeep")


# ---------------------------------------------------------------------------
# built-in suite integrity
# ---------------------------------------------------------------------------
def test_suite_is_well_formed():
    g1.validate_suite()


def test_suite_covers_every_required_category():
    required = {
        "full_strip",
        "partial_strip",
        "already_clean",
        "ambiguity_short",
        "ambiguity_context",
        "proper_names",
        "mixed_script",
        "punctuation_numbers",
        "url_email",
        "simple_emoji",
    }
    assert required <= {c.category for c in g1.SMOKE_CASES}


def test_full_strip_cases_really_carry_no_diacritics():
    for case in g1.SMOKE_CASES:
        if case.category == "full_strip":
            assert base_signature(case.text) == case.text, case.id


def test_partial_strip_cases_are_genuinely_partial():
    for case in g1.SMOKE_CASES:
        if case.category == "partial_strip":
            assert base_signature(case.text) != case.text, f"{case.id} has no diacritics at all"
            assert any(w == base_signature(w) for w in case.text.split()), f"{case.id} has no bare word"


def test_already_clean_cases_are_in_nfc():
    for case in g1.SMOKE_CASES:
        if case.category == "already_clean":
            assert case.text == unicodedata.normalize("NFC", case.text), case.id


def test_validate_suite_rejects_duplicate_ids():
    dup = (g1.SmokeCase("x", "full_strip", "a"), g1.SmokeCase("x", "full_strip", "b"))
    with pytest.raises(ValueError, match="duplicate"):
        g1.validate_suite(dup)


def test_validate_suite_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown categories"):
        g1.validate_suite((g1.SmokeCase("x", "not_a_category", "a"),))


def test_select_cases_filters_by_category():
    assert {c.category for c in g1.select_cases("full_strip,simple_emoji")} == {"full_strip", "simple_emoji"}


def test_select_cases_rejects_an_unknown_category():
    with pytest.raises(SystemExit):
        g1.select_cases("no_such_category")


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def _good_config():
    return {
        "model_id": "nrl-ai/vn-diacritic-vit5-base",
        "revision": "30ea5a9e4a0b9436e18915fd4dbb5876eaee7325",
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 256},
        "repeats": 3,
    }


def test_validate_config_accepts_the_locked_shape():
    assert g1.validate_config(_good_config())["repeats"] == 3


def test_shipped_config_file_is_valid_and_pinned():
    cfg = g1.load_config(REPO_ROOT / "configs" / "restore" / "nrl_vit5_base.yaml")
    assert cfg["model_id"] == "nrl-ai/vn-diacritic-vit5-base"
    assert cfg["revision"] == "30ea5a9e4a0b9436e18915fd4dbb5876eaee7325"
    assert cfg["generation"]["do_sample"] is False
    assert cfg["generation"]["num_beams"] == 1
    assert cfg["generation"]["max_new_tokens"] == 256
    assert cfg["repeats"] == 3


def test_validate_config_rejects_sampling():
    cfg = _good_config()
    cfg["generation"]["do_sample"] = True
    with pytest.raises(ValueError, match="greedy"):
        g1.validate_config(cfg)


def test_validate_config_rejects_beam_search():
    cfg = _good_config()
    cfg["generation"]["num_beams"] = 4
    with pytest.raises(ValueError, match="greedy"):
        g1.validate_config(cfg)


def test_validate_config_rejects_unbounded_generation():
    cfg = _good_config()
    del cfg["generation"]["max_new_tokens"]
    with pytest.raises(ValueError, match="bound"):
        g1.validate_config(cfg)


def test_validate_config_rejects_too_few_repeats():
    cfg = _good_config()
    cfg["repeats"] = 1
    with pytest.raises(ValueError, match="repeats"):
        g1.validate_config(cfg)


def test_validate_config_rejects_missing_revision():
    cfg = _good_config()
    del cfg["revision"]
    with pytest.raises(ValueError, match="revision"):
        g1.validate_config(cfg)


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------
def _case(cid="c1", category="full_strip", text="toi di hoc"):
    return g1.SmokeCase(cid, category, text)


def test_build_record_marks_identical_repeats_deterministic():
    rec = g1.build_record(_case(), ["tôi đi học"] * 3, [10.0, 12.0, 14.0], None)
    assert rec["deterministic"] is True
    assert rec["base_preserved"] is True
    assert rec["final_output"] == "tôi đi học"
    assert rec["latency_ms"] == pytest.approx(12.0)
    assert rec["error"] is None
    assert rec["clean_exact_preserved"] is None


def test_build_record_detects_non_determinism():
    rec = g1.build_record(_case(), ["tôi đi học", "tôi đi học", "tôi đi hoc"], [1.0] * 3, None)
    assert rec["deterministic"] is False


def test_build_record_flags_lexical_rewriting_and_keeps_the_evidence():
    rec = g1.build_record(_case(), ["tôi đi làm"] * 3, [1.0] * 3, None)
    assert rec["base_preserved"] is False
    assert rec["base_diff"][0]["input_words"] == ["hoc"]
    assert rec["base_diff"][0]["output_words"] == ["lam"]
    assert rec["first_divergence_char_index"] is not None
    assert rec["input_base_signature"] == "toi di hoc"
    assert rec["output_base_signature"] == "toi di lam"


def test_build_record_reports_whitespace_only_difference_without_hiding_it():
    rec = g1.build_record(_case(), ["tôi  đi học"] * 3, [1.0] * 3, None)
    assert rec["base_preserved"] is True
    assert rec["base_preserved_strict"] is False
    assert rec["whitespace_only_difference"] is True


def test_build_record_clean_exact_preservation():
    clean = _case("ac", "already_clean", "Tôi đi học.")
    assert g1.build_record(clean, ["Tôi đi học."] * 3, [1.0] * 3, None)["clean_exact_preserved"] is True
    assert g1.build_record(clean, ["Tôi đi làm."] * 3, [1.0] * 3, None)["clean_exact_preserved"] is False


def test_build_record_clean_exact_compares_against_nfc_of_input():
    nfd_text = unicodedata.normalize("NFD", "Tôi đi học.")
    clean = _case("ac", "already_clean", nfd_text)
    rec = g1.build_record(clean, [unicodedata.normalize("NFC", nfd_text)] * 3, [1.0] * 3, None)
    assert rec["clean_exact_preserved"] is True


def test_build_record_on_error_leaves_every_metric_null():
    rec = g1.build_record(_case(), [], [], "RuntimeError: boom")
    assert rec["error"] == "RuntimeError: boom"
    assert rec["final_output"] is None
    assert rec["deterministic"] is None
    assert rec["base_preserved"] is None
    assert rec["clean_exact_preserved"] is None
    assert rec["latency_ms"] is None
    assert rec["input_base_signature"] == "toi di hoc"


def test_build_record_has_every_field_the_gate_requires():
    rec = g1.build_record(_case(), ["tôi đi học"] * 3, [1.0] * 3, None)
    required = {
        "id", "category", "input", "outputs", "final_output", "deterministic",
        "input_base_signature", "output_base_signature", "base_preserved",
        "clean_exact_preserved", "error", "latency_ms",
    }
    assert required <= set(rec)


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------
def _records(spec):
    """spec: list of (category, outputs, error)."""
    out = []
    for i, (category, outputs, error) in enumerate(spec):
        text = "Tôi đi học." if category == "already_clean" else "toi di hoc"
        out.append(g1.build_record(_case(f"c{i}", category, text), outputs, [10.0] * len(outputs), error))
    return out


def test_summary_counts_and_rates():
    records = _records(
        [
            ("full_strip", ["tôi đi học"] * 3, None),
            ("full_strip", ["tôi đi làm"] * 3, None),  # base changed
            ("full_strip", ["tôi đi học", "tôi đi học", "tôi đi hoc"], None),  # non-deterministic
            ("full_strip", [], "RuntimeError: boom"),  # error
        ]
    )
    summary = g1.summarize(records, "m", "r")
    assert summary["num_cases"] == 4
    assert summary["num_success"] == 3
    assert summary["num_errors"] == 1
    assert summary["deterministic_rate"] == pytest.approx(2 / 3)
    assert summary["base_preservation_rate"] == pytest.approx(2 / 3)
    assert summary["model_id"] == "m"
    assert summary["revision"] == "r"


def test_summary_rates_exclude_errored_cases_from_the_denominator():
    summary = g1.summarize(_records([("full_strip", ["tôi đi học"] * 3, None), ("full_strip", [], "boom")]), "m", "r")
    assert summary["deterministic_rate"] == 1.0
    assert summary["base_preservation_rate"] == 1.0
    assert summary["num_errors"] == 1


def test_summary_uses_none_not_zero_for_empty_denominators():
    summary = g1.summarize(_records([("full_strip", ["tôi đi học"] * 3, None)]), "m", "r")
    assert summary["clean_exact_preservation_rate"] is None


def test_summary_all_errors_gives_null_rates():
    summary = g1.summarize(_records([("full_strip", [], "boom"), ("full_strip", [], "boom")]), "m", "r")
    assert summary["deterministic_rate"] is None
    assert summary["base_preservation_rate"] is None
    assert summary["mean_latency_ms"] is None
    assert summary["median_latency_ms"] is None


def test_summary_clean_exact_rate_uses_only_clean_cases():
    records = _records(
        [
            ("full_strip", ["tôi đi học"] * 3, None),
            ("already_clean", ["Tôi đi học."] * 3, None),
            ("already_clean", ["Tôi đi làm."] * 3, None),
        ]
    )
    summary = g1.summarize(records, "m", "r")
    assert summary["clean_exact_preservation_rate"] == pytest.approx(0.5)
    assert summary["rates_by_category"]["already_clean"]["n_clean_applicable"] == 2
    assert summary["rates_by_category"]["full_strip"]["clean_exact_preservation_rate"] is None


def test_summary_latency_statistics():
    records = [
        g1.build_record(_case("a"), ["tôi đi học"] * 2, [10.0, 20.0], None),
        g1.build_record(_case("b"), ["tôi đi học"] * 2, [30.0, 50.0], None),
    ]
    summary = g1.summarize(records, "m", "r")
    assert summary["mean_latency_ms"] == pytest.approx(27.5)  # mean of 15 and 40
    assert summary["median_latency_ms"] == pytest.approx(27.5)


def test_summary_categories_follow_the_declared_order():
    records = _records([("simple_emoji", ["a"] * 2, None), ("full_strip", ["tôi đi học"] * 2, None)])
    assert list(g1.summarize(records, "m", "r")["rates_by_category"]) == ["full_strip", "simple_emoji"]


# ---------------------------------------------------------------------------
# engineering status
# ---------------------------------------------------------------------------
def _eng():
    return g1.engineering_settings({})


def test_engineering_status_passes_on_a_clean_run():
    records = _records([("full_strip", ["tôi đi học"] * 3, None), ("already_clean", ["Tôi đi học."] * 3, None)])
    status, checks = g1.engineering_status(g1.summarize(records, "m", "r"), _eng(), model_loaded=True)
    assert status == g1.STATUS_PASS
    assert all(c["passed"] for c in checks)


def test_engineering_status_fails_when_the_model_did_not_load():
    summary = g1.summarize(_records([("full_strip", [], "boom")]), "m", "r")
    status, checks = g1.engineering_status(summary, _eng(), model_loaded=False)
    assert status == g1.STATUS_FAIL
    assert {c["check"] for c in checks if not c["passed"]} >= {"model_loaded"}


def test_engineering_status_fails_on_non_determinism():
    summary = g1.summarize(_records([("full_strip", ["tôi đi học", "tôi đi học", "tôi đi hoc"], None)]), "m", "r")
    status, checks = g1.engineering_status(summary, _eng(), model_loaded=True)
    assert status == g1.STATUS_FAIL
    assert any(c["check"] == "greedy_decoding_deterministic" and not c["passed"] for c in checks)


def test_engineering_status_fails_on_catastrophic_rewriting():
    summary = g1.summarize(_records([("full_strip", ["tôi đi làm"] * 3, None)] * 3), "m", "r")
    status, checks = g1.engineering_status(summary, _eng(), model_loaded=True)
    assert status == g1.STATUS_FAIL
    assert any(c["check"] == "no_catastrophic_lexical_rewriting" and not c["passed"] for c in checks)


def test_advisory_category_failures_do_not_flip_the_status():
    records = _records(
        [
            ("full_strip", ["tôi đi học"] * 3, None),
            ("simple_emoji", [], "RuntimeError: emoji exploded"),
            ("mixed_script", ["hoàn toàn khác"] * 3, None),
            ("url_email", ["cái gì đó"] * 3, None),
        ]
    )
    status, _ = g1.engineering_status(g1.summarize(records, "m", "r"), _eng(), model_loaded=True)
    assert status == g1.STATUS_PASS


def test_core_category_error_flips_the_status():
    records = _records([("full_strip", ["tôi đi học"] * 3, None), ("proper_names", [], "boom")])
    status, checks = g1.engineering_status(g1.summarize(records, "m", "r"), _eng(), model_loaded=True)
    assert status == g1.STATUS_FAIL
    assert any(c["check"] == "core_inference_completed" and not c["passed"] for c in checks)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def test_cases_jsonl_round_trips(tmp_path):
    records = _records([("full_strip", ["tôi đi học"] * 3, None), ("full_strip", [], "boom")])
    path = tmp_path / "cases.jsonl"
    g1.write_jsonl(path, records)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert g1.read_jsonl(path) == records


def test_jsonl_keeps_vietnamese_readable_not_escaped(tmp_path):
    path = tmp_path / "cases.jsonl"
    g1.write_jsonl(path, _records([("full_strip", ["tôi đi học"] * 3, None)]))
    assert "tôi đi học" in path.read_text(encoding="utf-8")


def test_summary_json_round_trips(tmp_path):
    summary = g1.summarize(_records([("full_strip", ["tôi đi học"] * 3, None)]), "m", "r")
    path = tmp_path / "summary.json"
    g1.write_json(path, summary)
    assert json.loads(path.read_text(encoding="utf-8")) == summary


def _run_config(**overrides):
    base = {
        "run_id": "20260819T000000Z",
        "timestamp_utc": "2026-08-19T00:00:00+00:00",
        "script_version": g1.SCRIPT_VERSION,
        "repeats": 3,
        "max_input_tokens": 256,
        "model": {"model_id": "m", "revision": "r", "metadata": {"num_parameters": 1}},
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 256},
        "environment": {"python_version": "3.11.0", "device": "cpu", "platform": "linux"},
    }
    base.update(overrides)
    return base


def test_write_artifacts_creates_all_four_files(tmp_path):
    records = _records([("full_strip", ["tôi đi học"] * 3, None), ("already_clean", ["Tôi đi học."] * 3, None)])
    summary = g1.summarize(records, "m", "r")
    eng = _eng()
    status, checks = g1.engineering_status(summary, eng, model_loaded=True)
    report = g1.render_report(_run_config(), summary, records, status, checks, eng)
    run_dir = tmp_path / "20260819T000000Z"
    g1.write_artifacts(run_dir, _run_config(), records, summary, report)
    for name in ("config.json", "cases.jsonl", "summary.json", "report.md"):
        assert (run_dir / name).is_file(), name
    assert "G-1 Assessment" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_contains_the_required_sections():
    records = _records(
        [
            ("full_strip", ["tôi đi làm"] * 3, None),
            ("already_clean", ["Tôi đi làm."] * 3, None),
            ("ambiguity_short", ["bàn"] * 3, None),
        ]
    )
    summary = g1.summarize(records, "m", "r")
    eng = _eng()
    status, checks = g1.engineering_status(summary, eng, model_loaded=True)
    report = g1.render_report(_run_config(), summary, records, status, checks, eng)
    for heading in (
        "## Environment",
        "## Model",
        "## Generation settings",
        "## Overall summary",
        "## Summary by category",
        "## Failed / lexically rewritten examples",
        "## Formatting-only differences",
        "## Short ambiguous inputs",
        "## G-1 Assessment",
    ):
        assert heading in report, heading
    assert "toi di hoc" in report  # the evidence for the base-signature mismatch


def test_report_does_not_decide_the_gate():
    """The script may report an engineering status; it must not claim to settle G-1."""
    records = _records([("full_strip", ["tôi đi học"] * 3, None)])
    summary = g1.summarize(records, "m", "r")
    eng = _eng()
    status, checks = g1.engineering_status(summary, eng, model_loaded=True)
    report = g1.render_report(_run_config(), summary, records, status, checks, eng)
    assert "The G-1 decision is the researcher's" in report


def test_console_table_has_the_expected_columns():
    records = _records([("full_strip", ["tôi đi học"] * 3, None), ("already_clean", ["Tôi đi học."] * 3, None)])
    table = g1.format_console_table(g1.summarize(records, "m", "r"))
    header = table.splitlines()[0]
    for column in ("Category", "N", "Errors", "Deterministic", "Base preserved", "Clean exact"):
        assert column in header
    assert "full_strip" in table
    assert "TOTAL" in table


# ---------------------------------------------------------------------------
# run-directory helpers
# ---------------------------------------------------------------------------
def test_run_id_is_a_utc_timestamp():
    from datetime import datetime, timezone

    assert g1.make_run_id(datetime(2026, 8, 19, 7, 5, 3, tzinfo=timezone.utc)) == "20260819T070503Z"


def test_unique_run_dir_never_overwrites_an_existing_run(tmp_path):
    first = g1.unique_run_dir(tmp_path, "run")
    first.mkdir(parents=True)
    second = g1.unique_run_dir(tmp_path, "run")
    assert second != first
    assert second.name == "run-1"


# ---------------------------------------------------------------------------
# Audit 001 / B2: the engineering check must key on the lexical metric
# ---------------------------------------------------------------------------
AUDIT_B2_INPUT = "hom nay thoi tiet o thanh pho ho chi minh rat dep"
AUDIT_B2_FORMATTED = "Hôm nay thời tiết ở Thành phố Hồ Chí Minh rất đẹp."


def test_record_carries_both_signatures_and_both_verdicts():
    rec = g1.build_record(_case("fs", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None)
    for field in (
        "input_base_signature",
        "output_base_signature",
        "base_preserved",
        "input_rewrite_signature",
        "output_rewrite_signature",
        "rewrite_preserved",
    ):
        assert field in rec, field
    assert rec["base_preserved"] is False  # strict measurement is unchanged
    assert rec["rewrite_preserved"] is True  # engineering measurement tolerates it
    assert rec["formatting_only_difference"] is True


def test_strict_signature_evidence_survives_for_formatting_only_cases():
    """The relaxation must not erase the record of what actually changed."""
    rec = g1.build_record(_case("fs", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None)
    assert rec["base_diff"], "strict word-level diff must still be recorded"
    assert rec["rewrite_diff"] is None  # nothing lexical to report
    assert rec["input_base_signature"] != rec["output_base_signature"]


def test_genuine_rewrite_populates_the_lexical_diff():
    rec = g1.build_record(_case("fs", "full_strip", "toi dang hoc AI"), ["Tôi đang nghiên cứu AI."] * 3, [1.0] * 3, None)
    assert rec["rewrite_preserved"] is False
    assert rec["formatting_only_difference"] is False
    assert rec["rewrite_diff"], "lexical diff must be recorded when words change"
    ops = {c["op"] for c in rec["rewrite_diff"]}
    assert ops <= {"replace", "insert", "delete"} and ops


def test_engineering_status_passes_on_capitalised_restorations():
    """The exact audit-001 false failure: this must no longer fail the gate."""
    records = [
        g1.build_record(_case(f"c{i}", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None)
        for i in range(6)
    ]
    summary = g1.summarize(records, "m", "r")
    assert summary["base_preservation_rate"] == 0.0  # strict rate is still honest
    assert summary["rewrite_preservation_rate"] == 1.0
    status, checks = g1.engineering_status(summary, _eng(), model_loaded=True)
    assert status == g1.STATUS_PASS
    rewriting = next(c for c in checks if c["check"] == "no_catastrophic_lexical_rewriting")
    assert rewriting["passed"] is True
    assert "strict base-preservation rate" in rewriting["detail"], "the strict rate must stay visible"


def test_engineering_status_still_fails_on_real_lexical_rewriting():
    records = [
        g1.build_record(_case(f"c{i}", "full_strip", "toi dang hoc AI"), ["Tôi đang nghiên cứu AI."] * 3, [1.0] * 3, None)
        for i in range(3)
    ]
    status, checks = g1.engineering_status(g1.summarize(records, "m", "r"), _eng(), model_loaded=True)
    assert status == g1.STATUS_FAIL
    assert any(c["check"] == "no_catastrophic_lexical_rewriting" and not c["passed"] for c in checks)


def test_threshold_key_governs_the_lexical_metric():
    eng = _eng()
    assert "min_core_rewrite_preservation_rate" in eng
    assert "min_core_base_preservation_rate" not in eng, "the strict metric must not be thresholded"


def test_summary_reports_both_rates_and_the_formatting_gap():
    records = [
        g1.build_record(_case("a", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None),
        g1.build_record(_case("b", "full_strip", "toi di hoc"), ["tôi đi học"] * 3, [1.0] * 3, None),
    ]
    summary = g1.summarize(records, "m", "r")
    assert summary["base_preservation_rate"] == pytest.approx(0.5)
    assert summary["rewrite_preservation_rate"] == 1.0
    assert summary["overall_counts"]["num_formatting_only_difference"] == 1
    assert summary["rates_by_category"]["full_strip"]["num_rewrite_preserved"] == 2


def test_console_table_shows_both_preservation_columns():
    records = [g1.build_record(_case("a", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None)]
    table = g1.format_console_table(g1.summarize(records, "m", "r"))
    assert "Base preserved" in table
    assert "Lexical kept" in table
    assert "strict" in table  # the legend distinguishing them


def test_report_separates_lexical_failures_from_formatting_differences():
    records = [
        g1.build_record(_case("fmt", "full_strip", AUDIT_B2_INPUT), [AUDIT_B2_FORMATTED] * 3, [1.0] * 3, None),
        g1.build_record(_case("rew", "full_strip", "toi dang hoc AI"), ["Tôi đang nghiên cứu AI."] * 3, [1.0] * 3, None),
    ]
    summary = g1.summarize(records, "m", "r")
    eng = _eng()
    status, checks = g1.engineering_status(summary, eng, model_loaded=True)
    report = g1.render_report(_run_config(), summary, records, status, checks, eng)
    failures = report.index("## Failed / lexically rewritten examples")
    formatting = report.index("## Formatting-only differences")
    assert report.index("`rew`", failures) < formatting, "the genuine rewrite belongs in the failures section"
    assert "`fmt`" in report[formatting:], "the formatting-only case must be listed, not hidden"
    assert "not merged" in report, "the report must explain that the two metrics differ"


# ---------------------------------------------------------------------------
# Audit 001 / N1, N7: honest field naming
# ---------------------------------------------------------------------------
def test_expected_dtype_is_declared_as_a_diagnostic_not_a_control():
    cfg = g1.load_config(REPO_ROOT / "configs" / "restore" / "nrl_vit5_base.yaml")
    assert cfg["expected_dtype"] == "float32"
    assert "dtype" not in cfg, "a bare 'dtype' key would read as a runtime control"


def test_script_never_passes_a_dtype_to_from_pretrained():
    """expected_dtype must stay an expectation: nothing may cast the checkpoint."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for line in source.splitlines():
        if "from_pretrained" in line:
            assert "dtype" not in line, f"from_pretrained must not receive a dtype: {line.strip()}"


def test_adjudication_field_is_named_honestly():
    rec = g1.build_record(_case("a", "full_strip", "toi di hoc"), ["tôi đi học"] * 3, [1.0] * 3, None)
    assert rec["humanly_adjudicable"] is True
    assert "has_ground_truth" not in rec, "the harness holds no ground truth; the name must not imply one"
    ambiguous = g1.build_record(_case("b", "ambiguity_short", "ban"), ["bàn"] * 3, [1.0] * 3, None)
    assert ambiguous["humanly_adjudicable"] is False


def test_no_fabricated_ground_truth_anywhere_in_the_suite():
    assert all(not hasattr(c, "expected") for c in g1.SMOKE_CASES)
    rec = g1.build_record(_case("a", "full_strip", "toi di hoc"), ["tôi đi học"] * 3, [1.0] * 3, None)
    assert not any("expected" in k for k in rec if k != "expected_dtype")


# ---------------------------------------------------------------------------
# Audit 001 / N6: checkpoint patterns
# ---------------------------------------------------------------------------
def _is_git_work_tree() -> bool:
    """These checks ask git itself, so they need a real checkout.

    A source tarball or a Docker `COPY` without `.git` is a legitimate way to run
    the suite; skipping there beats failing for a reason unrelated to the code.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


requires_git = pytest.mark.skipif(not _is_git_work_tree(), reason="not a git work tree")


def _is_ignored(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True, text=True
        ).returncode
        == 0
    )


@requires_git
@pytest.mark.parametrize(
    "path",
    ["a.safetensors", "a.pt", "a.pth", "a.ckpt", "pytorch_model.bin", "pytorch_model-00001-of-00002.bin", "model.bin"],
)
def test_checkpoint_artifacts_are_gitignored(path):
    assert _is_ignored(path), f"{path} would be committable"


@requires_git
@pytest.mark.parametrize("path", ["tests/fixtures/data.bin", "docs/assets/font.bin", "results/g_minus1/.gitkeep"])
def test_unrelated_files_are_not_swallowed_by_the_checkpoint_patterns(path):
    assert not _is_ignored(path), f"{path} is ignored but should stay trackable"
