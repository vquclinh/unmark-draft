"""The Audit-043 diagnostic scanner. **Torch-free; no model, no training.**

The scanner's first real Colab execution died *after* loading the 2.2 GB
prepared corpus and walking the deterministic sampler:

    File "scripts/stage1_length_contract_scan.py", line 87,
        in authoritative_base_length
        from unmark.linguistics import canon
    ImportError: cannot import name 'canon' from 'unmark.linguistics'

`canon` is exported from `unmark.orthography`, not `unmark.linguistics`. The
import sat inside a function body, so nothing at import time could catch it, and
the local checks only exercised `--help` and the fail-closed path — neither of
which reaches that line.

The first test below is the one that would have caught it: it statically
resolves **every** `from X import Y` in the scanner against the real modules,
including the ones nested inside functions. The rest prove the scanner measures
the same object Stage-6 enforced, using Stage-6's own function.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SCANNER = REPO / "scripts" / "stage1_length_contract_scan.py"

import stage1_length_contract_scan as scanner  # noqa: E402

from unmark.stage1.lengths import PHOBERT_RUN, build_length_functions  # noqa: E402
from unmark.stage1.protocol import MAX_LENGTH  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Import contract -- the check that would have caught the Colab failure
# ---------------------------------------------------------------------------
def scanner_imports() -> list[tuple[str, str]]:
    """Every `from MODULE import NAME` in the scanner, function bodies included."""
    tree = ast.parse(SCANNER.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                found.append((node.module, alias.name))
    return found


COLAB_ONLY_MODULES = {"transformers"}
"""Absent from the deliberate ML-free local venv; resolved on the CUDA host."""


def test_every_from_import_in_the_scanner_resolves():
    """Statically resolve each import against the REAL module.

    Nested-in-function imports are exactly where the Colab failure hid, so they
    are included rather than skipped.
    """
    unresolved = []
    for module_name, symbol in scanner_imports():
        if module_name.split(".")[0] in COLAB_ONLY_MODULES:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:  # pragma: no cover - a real defect
            unresolved.append(f"{module_name} (module missing: {error})")
            continue
        if not hasattr(module, symbol):
            unresolved.append(f"{module_name}.{symbol}")
    assert not unresolved, (
        f"the scanner imports names that do not exist: {unresolved}. This is the "
        "Audit 043 §8a failure mode -- an ImportError inside a function body, "
        "reached only after minutes of corpus loading."
    )


def test_the_original_bad_import_is_detected_by_this_test():
    """Mutation check: the exact Colab defect must fail the check above."""
    assert not hasattr(importlib.import_module("unmark.linguistics"), "canon"), (
        "unmark.linguistics does not export canon -- that was the bad guess"
    )
    assert hasattr(importlib.import_module("unmark.orthography"), "canon"), (
        "canon really lives in unmark.orthography"
    )


def test_the_scanner_no_longer_normalises_anything_itself():
    """Zero duplicated canon/decompose logic: it delegates to Stage-6."""
    tree = ast.parse(SCANNER.read_text(encoding="utf-8"))
    imported = {name for _module, name in scanner_imports()}
    assert "canon" not in imported, "the scanner must not import canon at all"
    assert "decompose" not in imported, "the scanner must not import decompose"
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "canon" not in called and "decompose" not in called
    assert "build_length_functions" in imported, (
        "the authoritative length must come from Stage-6's own function"
    )


def test_the_scanner_loads_no_model_and_steps_no_optimizer():
    """AST, not substring.

    An earlier draft grepped for "backward" and matched the scanner's own
    docstring, which merely *promises* that no backward is called -- the same
    prose-matching defect this repository keeps rediscovering. What matters is
    what the code imports and calls.
    """
    tree = ast.parse(SCANNER.read_text(encoding="utf-8"))

    modules = {module for module, _name in scanner_imports()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert not any(m.split(".")[0] == "torch" for m in modules), modules

    imported = {name for _module, name in scanner_imports()}
    for forbidden in ("build_optimizer", "train_run", "execute_stage",
                      "AutoModel", "AutoModelForMaskedLM"):
        assert forbidden not in imported, f"a diagnostic must not import {forbidden}"

    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called.add(getattr(node.func, "id", None) or getattr(node.func, "attr", None))
    for forbidden in ("step", "backward", "zero_grad", "train_run", "execute_stage",
                      "build_optimizer"):
        assert forbidden not in called, f"a diagnostic must not call {forbidden}()"


# ---------------------------------------------------------------------------
# 2-4. The scanner measures Stage-6's object, through Stage-6's function
# ---------------------------------------------------------------------------
class RunUnitTokenizer:
    """A minimal tokenizer that decomposes exactly as PhoBERT documents.

    `RunLengthComposer` needs `tokenize`, `convert_tokens_to_ids` and
    `build_inputs_with_special_tokens`. This stub supplies the real *shape* of
    the contract so the delegation can be tested without `transformers`; the
    real-tokenizer numbers are a Colab matter (see the module docstring of the
    scanner and Audit 043 §12).
    """

    unk_token_id = 3

    def tokenize(self, text):
        pieces = []
        for match in PHOBERT_RUN.finditer(text):
            run = match.group(0)
            core = run.rstrip("\n")
            parts = [core[i:i + 2] for i in range(0, len(core), 2)] or [core]
            if run.endswith("\n") and len(parts) >= 2:
                parts = parts[:-2] + ["".join(parts[-2:])]
            pieces.extend(parts)
        return pieces

    def convert_tokens_to_ids(self, tokens):
        return [abs(hash(t)) % 1000 + 10 for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


SYNTHETIC = [
    "xin chao ban",
    "xin chao\nban",                 # newline-bearing
    "mot hai\nba bon\nnam",          # several newlines
    "a\nb\nc\nd",
    "khong co ky tu xuong dong",
]


@pytest.mark.parametrize("text", SYNTHETIC)
def test_authoritative_length_executes_through_the_real_production_path(text):
    """It runs, and it returns the Stage-6 number for this text."""
    tokenizer = RunUnitTokenizer()
    _reference, base_length, _transforms = scanner.stage6_length_functions(tokenizer)
    value = scanner.authoritative_base_length(base_length, text)
    assert isinstance(value, int) and value > 0


@pytest.mark.parametrize("text", SYNTHETIC)
def test_scanner_authoritative_equals_stage6_base_length(text):
    """Independent construction of Stage-6's functions must agree exactly."""
    tokenizer = RunUnitTokenizer()
    _r1, scanner_base, _t1 = scanner.stage6_length_functions(tokenizer)
    _r2, stage6_base, _t2 = build_length_functions(RunUnitTokenizer())
    assert scanner.authoritative_base_length(scanner_base, text) == stage6_base(text)


def test_newline_bearing_texts_are_covered():
    assert any("\n" in t for t in SYNTHETIC), "the defect only appears with newlines"
    assert sum("\n" in t for t in SYNTHETIC) >= 3


def test_stage6_length_functions_returns_the_documented_triple():
    tokenizer = RunUnitTokenizer()
    result = scanner.stage6_length_functions(tokenizer)
    assert len(result) == 3
    reference_length, base_length, transforms = result
    assert callable(reference_length) and callable(base_length)
    assert hasattr(transforms, "base") and hasattr(transforms, "canonical")


# ---------------------------------------------------------------------------
# Scanner plumbing
# ---------------------------------------------------------------------------
def test_a_missing_corpus_fails_closed_before_a_tokenizer_is_loaded(tmp_path):
    """The Colab run spent minutes before failing; a bad path must cost nothing."""
    with pytest.raises(SystemExit, match="REFUSED"):
        scanner.require_corpus(tmp_path / "nope")


def test_stable_id_never_returns_the_text():
    text = "xin chao ban"
    identifier = scanner.stable_id(text)
    assert text not in identifier
    assert len(identifier) == 16
    assert scanner.stable_id(text) == identifier


def test_the_cli_offers_both_diagnostic_modes():
    parser = scanner.build_parser()
    args = parser.parse_args(["--prepared-corpus", "/x", "--reproduce", "--seed", "21230"])
    assert args.reproduce is True and args.seed == 21230
    scan_args = parser.parse_args(["--prepared-corpus", "/x", "--partition", "train"])
    assert scan_args.reproduce is False and scan_args.partition == "train"


def test_the_scanner_reports_test_as_unused():
    source = SCANNER.read_text(encoding="utf-8")
    assert '"official_test_used": False' in source


def test_the_bound_is_the_locked_max_length():
    assert MAX_LENGTH == 256
    assert "MAX_LENGTH" in SCANNER.read_text(encoding="utf-8")
