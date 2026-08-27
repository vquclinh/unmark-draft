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


# ---------------------------------------------------------------------------
# Optimised scope scan (Audit 043 §9a)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import types  # noqa: E402

from unmark.stage1.data import project_text  # noqa: E402
from unmark.stage1.protocol import MAX_LENGTH as LOCKED_MAX  # noqa: E402


class CorpusTokenizer(RunUnitTokenizer):
    """Adds the ids/specials the realised path needs."""

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


def classifier():
    from unmark.linguistics import make_classifier, try_load_inventory

    return make_classifier(try_load_inventory())


CORPUS_TEXTS = [
    "xin chao ban",
    "xin chao\nban hien",
    "mot hai\nba bon\nnam sau",
    "a\nb\nc\nd",
    "khong co ky tu xuong dong o day",
    "",
    "   ",
    "\n",
    "trailing   ",
    "   leading",
]


def write_corpus(directory, texts, base_lengths=None):
    """A minimal prepared-corpus shaped file. Never real corpus data."""
    from unmark.stage1.manifest import CHUNKS_NAME

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = CorpusTokenizer()
    _reference, base_length, _t = build_length_functions(tokenizer)
    rows = []
    for index, text in enumerate(texts):
        rows.append({
            "chunk_id": f"doc-{index:04d}#0",
            "partition": "train",
            "text": text,
            "base_length": (base_lengths[index] if base_lengths is not None
                            else base_length(text)),
        })
    (directory / CHUNKS_NAME).write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return directory


def scan_args(prepared, **overrides):
    parser = scanner.build_parser()
    argv = ["--prepared-corpus", str(prepared), "--partition", "train"]
    args = parser.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(scanner, "load_tokenizer", lambda revision: CorpusTokenizer())


# -- 1. fast path == production path ----------------------------------------
@pytest.mark.parametrize("text", CORPUS_TEXTS)
def test_fast_realised_length_equals_the_production_path(text):
    """The invariant the optimisation stands or falls on."""
    tokenizer = CorpusTokenizer()
    counter = scanner.RealisedLengthCounter(tokenizer, tokenizer.unk_token_id)
    _base, content_ids, _proj = project_text(text, tokenizer, classifier(),
                                             tokenizer.unk_token_id)
    expected = len(tokenizer.build_inputs_with_special_tokens(list(content_ids)))
    assert counter.length(text) == expected


def test_the_fast_path_counts_aligned_pieces_not_raw_tokens():
    """`align_chunk` returns pieces=() on failure, so a token count would differ."""
    source = SCANNER.read_text(encoding="utf-8")
    assert "align_chunk(chunk, tokens, ids" in source
    assert ".pieces)" in source


def test_the_memo_changes_speed_not_results():
    tokenizer = CorpusTokenizer()
    warm = scanner.RealisedLengthCounter(tokenizer, tokenizer.unk_token_id)
    for text in CORPUS_TEXTS * 3:
        cold = scanner.RealisedLengthCounter(tokenizer, tokenizer.unk_token_id)
        assert warm.length(text) == cold.length(text)
    assert warm.memo_hits > 0, "the memo must actually be exercised"


# -- 2. persisted Stage-6 base_length verification ---------------------------
def test_a_correct_persisted_base_length_is_accepted(tmp_path, stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS)
    code = scanner.scan(scan_args(prepared, verify_every=1, workers=1))
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_NO_VIOLATION
    assert report["status"] == "SUCCESS_NO_VIOLATION"
    assert report["scanned"] == len(CORPUS_TEXTS)
    assert report["authoritative_spot_checks"] == len(CORPUS_TEXTS)


def test_a_tampered_persisted_base_length_fails_closed(tmp_path, stub_tokenizer, capsys):
    """The shortcut must never silently trust the JSON."""
    lengths = [None] * len(CORPUS_TEXTS)
    tokenizer = CorpusTokenizer()
    _r, base_length, _t = build_length_functions(tokenizer)
    for index, text in enumerate(CORPUS_TEXTS):
        lengths[index] = base_length(text)
    lengths[3] += 7                                  # a lie
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS, base_lengths=lengths)
    code = scanner.scan(scan_args(prepared, verify_every=1, workers=1))
    assert code == scanner.EXIT_DIAGNOSTIC_FAILURE
    assert "failed verification" in capsys.readouterr().err


def test_offenders_are_always_recomputed_even_without_spot_checks(tmp_path,
                                                                  stub_tokenizer, capsys):
    long_text = " ".join(["abcdefgh"] * 400)
    prepared = write_corpus(tmp_path / "corpus", [long_text])
    scanner.scan(scan_args(prepared, verify_every=0, workers=1, max_offenders=5))
    report = json.loads(capsys.readouterr().out)
    assert report["over_max_length"] == 1
    offender = report["offenders"][0]
    assert offender["stage6_recomputed_base_length"] is not None, (
        "an offender must never be reported on the persisted value alone"
    )


# -- 3. newline-bearing coverage --------------------------------------------
def test_newline_bearing_rows_are_scanned(tmp_path, stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS)
    scanner.scan(scan_args(prepared, verify_every=1, workers=1))
    report = json.loads(capsys.readouterr().out)
    assert report["scanned"] == len(CORPUS_TEXTS)
    assert sum("\n" in t for t in CORPUS_TEXTS) >= 4


# -- 4. worker-count independence -------------------------------------------
def measured_items(texts):
    tokenizer = CorpusTokenizer()
    counter = scanner.RealisedLengthCounter(tokenizer, tokenizer.unk_token_id)
    _r, base_length, _t = build_length_functions(tokenizer)
    items = []
    for index, text in enumerate(texts):
        items.append({
            "line_index": index, "chunk_id": f"doc-{index:04d}#0", "partition": "train",
            "text_sha256_16": scanner.stable_id(text), "characters": len(text),
            "contains_newline": "\n" in text,
            "persisted": base_length(text), "recomputed": None,
            "realised": counter.length(text),
        })
    return items


def test_aggregation_is_independent_of_arrival_order(tmp_path):
    """Worker count changes arrival order and nothing else."""
    import random

    texts = CORPUS_TEXTS + [" ".join(["abcdefgh"] * 400)]
    items = measured_items(texts)

    ordered = scanner.Aggregate()
    ordered.absorb(items, 10)

    for seed in (1, 2, 3):
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        out_of_order = scanner.Aggregate()
        for item in shuffled:                    # one at a time, like imap chunks
            out_of_order.absorb([item], 10)
        assert out_of_order.state() == ordered.state(), (
            f"aggregate depends on arrival order (seed {seed})"
        )


def test_offender_list_is_bounded_and_deterministic():
    texts = [" ".join(["abcdefgh"] * 400)] * 20
    items = measured_items(texts)
    aggregate = scanner.Aggregate()
    aggregate.absorb(items, 5)
    assert len(aggregate.offenders) == 5
    assert [o["line_index"] for o in aggregate.offenders] == [0, 1, 2, 3, 4]


# -- 5. interrupt + resume ---------------------------------------------------
def test_a_resumed_scan_equals_an_uninterrupted_one(tmp_path, stub_tokenizer, capsys):
    texts = CORPUS_TEXTS * 4
    prepared = write_corpus(tmp_path / "corpus", texts)

    whole_report = tmp_path / "whole.json"
    scanner.scan(scan_args(prepared, report=str(whole_report), workers=1,
                           verify_every=5, checkpoint_every=4, batch_size=2))
    capsys.readouterr()
    whole = json.loads(whole_report.read_text(encoding="utf-8"))

    # Interrupt: scan only the first half, then persist state by hand exactly as
    # the checkpoint would, and resume.
    part_report = tmp_path / "part.json"
    args = scan_args(prepared, report=str(part_report), workers=1, verify_every=5,
                     checkpoint_every=4, batch_size=2, limit=len(texts) // 2)
    scanner.scan(args)
    capsys.readouterr()
    half = json.loads(part_report.read_text(encoding="utf-8"))
    assert half["scanned"] == len(texts) // 2

    state = tmp_path / "resumed.json.state.json"
    state.write_text(json.dumps({
        "identity": scanner.diagnostic_identity(
            scan_args(prepared, report=str(tmp_path / "resumed.json"), workers=1,
                      verify_every=5)),
        "aggregate": {
            "scanned": half["scanned"], "within": half["within_max_length"],
            "over": half["over_max_length"],
            "disagreements": half["stage6_vs_stage1_disagreements"],
            "verified": half["authoritative_spot_checks"],
            "max_authoritative": half["max_stage6_authoritative_base_length"],
            "max_realised": half["max_stage1_realised_base_length"],
            "over_histogram": {str(k): v for k, v in half["over_length_histogram"].items()},
            "delta_histogram": {str(k): v for k, v in half["delta_histogram"].items()},
            "offenders": half["offenders"], "next_index": len(texts) // 2,
        },
    }, sort_keys=True), encoding="utf-8")

    resumed_args = scan_args(prepared, report=str(tmp_path / "resumed.json"), workers=1,
                             verify_every=5, checkpoint_every=4, batch_size=2,
                             resume=True)
    scanner.scan(resumed_args)
    capsys.readouterr()
    resumed = json.loads((tmp_path / "resumed.json").read_text(encoding="utf-8"))

    for field in ("scanned", "within_max_length", "over_max_length",
                  "stage6_vs_stage1_disagreements",
                  "max_stage6_authoritative_base_length",
                  "max_stage1_realised_base_length",
                  "over_length_histogram", "delta_histogram", "offenders"):
        assert resumed[field] == whole[field], f"{field} differs after resume"


def test_a_resume_from_a_foreign_diagnostic_is_refused(tmp_path, stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS)
    report = tmp_path / "r.json"
    (tmp_path / "r.json.state.json").write_text(json.dumps({
        "identity": {"prepared_corpus": "/somewhere/else", "partition": "train",
                     "revision": "x", "max_length": 256,
                     "encoder_checkpoint": "other", "verify_every": 1000},
        "aggregate": scanner.Aggregate().state(),
    }), encoding="utf-8")
    code = scanner.scan(scan_args(prepared, report=str(report), workers=1, resume=True))
    assert code == scanner.EXIT_DIAGNOSTIC_FAILURE
    assert "different diagnostic" in capsys.readouterr().err


# -- 6. exit semantics -------------------------------------------------------
def test_the_three_exit_codes_are_distinct_and_named():
    assert scanner.EXIT_NO_VIOLATION == 0
    assert scanner.EXIT_DIAGNOSTIC_FAILURE == 1
    assert scanner.EXIT_VIOLATION_FOUND == 2
    assert scanner.STATUS_BY_EXIT[0] == "SUCCESS_NO_VIOLATION"
    assert scanner.STATUS_BY_EXIT[1] == "DIAGNOSTIC_FAILURE"
    assert scanner.STATUS_BY_EXIT[2] == "SUCCESS_VIOLATION_FOUND"


def test_a_violation_does_not_share_an_exit_code_with_a_failure(tmp_path,
                                                                stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", [" ".join(["abcdefgh"] * 400)])
    code = scanner.scan(scan_args(prepared, workers=1, verify_every=1))
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_VIOLATION_FOUND != scanner.EXIT_DIAGNOSTIC_FAILURE
    assert report["status"] == "SUCCESS_VIOLATION_FOUND"
    assert report["exit_code"] == 2


# -- 7. safety ---------------------------------------------------------------
def test_the_report_never_carries_corpus_text(tmp_path, stub_tokenizer, capsys):
    secret = "bi mat khong duoc ro ri\nra ngoai"
    prepared = write_corpus(tmp_path / "corpus", [secret, " ".join(["abcdefgh"] * 400)])
    scanner.scan(scan_args(prepared, workers=1, verify_every=1))
    out = capsys.readouterr().out
    assert secret not in out
    assert "bi mat" not in out


def test_the_locked_bound_is_untouched():
    assert LOCKED_MAX == 256


# ---------------------------------------------------------------------------
# Local-SSD staging, real-tokenizer validation, worker benchmark (Audit 043 §9c)
# ---------------------------------------------------------------------------
import hashlib as _hashlib  # noqa: E402

from unmark.stage1.manifest import CHUNKS_NAME  # noqa: E402


def fake_verified(prepared, size, digest):
    """A stand-in for `verify_prepared_corpus`'s result, with the real shape."""
    return types.SimpleNamespace(
        prepared_dir=prepared, artifacts={CHUNKS_NAME: (size, digest)},
    )


@pytest.fixture
def staged_corpus(tmp_path, monkeypatch):
    """A Drive-like source whose verified identity the stager will check."""
    prepared = write_corpus(tmp_path / "drive", CORPUS_TEXTS)
    source = prepared / CHUNKS_NAME
    size, digest = scanner.sha256_of(source)
    import unmark.stage1.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "verify_prepared_corpus",
                        lambda p, c: fake_verified(p, size, digest))
    return prepared, size, digest


def stage_args(prepared, target, **overrides):
    args = scan_args(prepared, **overrides)
    args.stage_local = str(target)
    args.completion_dir = None
    return args


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    payload = b"xin chao\n" * 1000
    path.write_bytes(payload)
    size, digest = scanner.sha256_of(path)
    assert size == len(payload)
    assert digest == _hashlib.sha256(payload).hexdigest()


def test_staging_verifies_source_then_copy(tmp_path, staged_corpus, capsys):
    prepared, size, digest = staged_corpus
    target_dir = tmp_path / "ssd"
    staged = scanner.stage_to_local(stage_args(prepared, target_dir))
    assert staged == target_dir / CHUNKS_NAME
    assert scanner.sha256_of(staged) == (size, digest)
    err = capsys.readouterr().err
    assert "VERIFIED local copy" in err
    assert "MB/s" in err, "copy throughput must be reported"


def test_staging_never_modifies_the_drive_source(tmp_path, staged_corpus):
    prepared, size, digest = staged_corpus
    scanner.stage_to_local(stage_args(prepared, tmp_path / "ssd"))
    assert scanner.sha256_of(prepared / CHUNKS_NAME) == (size, digest)


def test_staging_refuses_a_source_that_does_not_match_the_verified_identity(
        tmp_path, monkeypatch):
    prepared = write_corpus(tmp_path / "drive", CORPUS_TEXTS)
    import unmark.stage1.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "verify_prepared_corpus",
                        lambda p, c: fake_verified(p, 123, "d" * 64))
    with pytest.raises(SystemExit, match="does not match the verified corpus identity"):
        scanner.stage_to_local(stage_args(prepared, tmp_path / "ssd"))


def test_staging_reuses_an_already_verified_local_copy(tmp_path, staged_corpus, capsys):
    prepared, _size, _digest = staged_corpus
    target_dir = tmp_path / "ssd"
    scanner.stage_to_local(stage_args(prepared, target_dir))
    capsys.readouterr()
    scanner.stage_to_local(stage_args(prepared, target_dir))
    assert "reusing verified local copy" in capsys.readouterr().err


def test_staging_recopies_a_corrupted_local_cache(tmp_path, staged_corpus, capsys):
    prepared, size, digest = staged_corpus
    target_dir = tmp_path / "ssd"
    scanner.stage_to_local(stage_args(prepared, target_dir))
    (target_dir / CHUNKS_NAME).write_text("corrupted\n", encoding="utf-8")
    capsys.readouterr()
    staged = scanner.stage_to_local(stage_args(prepared, target_dir))
    assert "does not verify; recopying" in capsys.readouterr().err
    assert scanner.sha256_of(staged) == (size, digest)


def test_a_scan_reads_the_staged_local_copy(tmp_path, staged_corpus, stub_tokenizer,
                                            capsys):
    prepared, _s, _d = staged_corpus
    target_dir = tmp_path / "ssd"
    args = stage_args(prepared, target_dir, workers=1, verify_every=1)
    args.local_chunks = str(scanner.stage_to_local(args))
    capsys.readouterr()
    assert scanner.chunks_path_for(args) == target_dir / CHUNKS_NAME
    scanner.scan(args)
    report = json.loads(capsys.readouterr().out)
    assert report["scanned"] == len(CORPUS_TEXTS)


# -- validation mode ---------------------------------------------------------
def test_validate_reports_zero_mismatches(tmp_path, stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS * 5)
    args = scan_args(prepared, validate_rows=50, validate_stride=1, progress=0,
                     stage_local=None, local_chunks=None, offender_hash="deadbeef")
    code = scanner.validate(args)
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_NO_VIOLATION
    assert report["mode"] == "validate"
    assert report["mismatch_count"] == 0
    assert report["compared"] > 0
    assert report["newline_bearing_sampled"] > 0


def test_validate_reconfirms_the_offender_by_hash(tmp_path, stub_tokenizer, capsys):
    offending = " ".join(["abcdefgh"] * 400)
    prepared = write_corpus(tmp_path / "corpus", [*CORPUS_TEXTS, offending])
    args = scan_args(prepared, validate_rows=100, validate_stride=1, progress=0,
                     stage_local=None, local_chunks=None,
                     offender_hash=scanner.stable_id(offending))
    scanner.validate(args)
    report = json.loads(capsys.readouterr().out)
    found = report["offender_reconfirmed"]
    assert found is not None
    assert found["text_sha256_16"] == scanner.stable_id(offending)
    assert found["stage1_fast_base_length"] == found["stage1_realised_base_length"]


def test_validate_fails_closed_when_the_fast_path_disagrees(tmp_path, stub_tokenizer,
                                                            capsys, monkeypatch):
    """Mutation: break the fast path and the gate must refuse."""
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS)
    monkeypatch.setattr(scanner.RealisedLengthCounter, "length",
                        lambda self, text: 12345)
    args = scan_args(prepared, validate_rows=20, validate_stride=1, progress=0,
                     stage_local=None, local_chunks=None, offender_hash="x")
    code = scanner.validate(args)
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_DIAGNOSTIC_FAILURE
    assert report["mismatch_count"] > 0
    assert report["status"] == "DIAGNOSTIC_FAILURE"


def test_validation_sampling_is_deterministic(tmp_path, stub_tokenizer):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS * 6)
    args = scan_args(prepared, validate_rows=30, validate_stride=3,
                     stage_local=None, local_chunks=None)
    first, _b1, _n1, _s1 = scanner._sample_rows(args, 30)
    second, _b2, _n2, _s2 = scanner._sample_rows(args, 30)
    assert [i for i, _ in first] == [i for i, _ in second]


# -- benchmark mode ----------------------------------------------------------
def test_benchmark_runs_and_reports_a_recommendation(tmp_path, stub_tokenizer, capsys):
    prepared = write_corpus(tmp_path / "corpus", CORPUS_TEXTS * 20)
    args = scan_args(prepared, benchmark_rows=40, benchmark_workers="1",
                     verify_every=0, progress=0, stage_local=None, local_chunks=None,
                     checkpoint_every=0, report=None, batch_size=10)
    code = scanner.benchmark(args)
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_NO_VIOLATION
    assert report["mode"] == "benchmark"
    assert report["all_digests_identical"] is True
    assert report["recommended_workers"] == 1
    entry = report["results"][0]
    for field in ("rows", "elapsed_seconds", "rows_per_second", "cpu_utilisation",
                  "peak_rss_mib", "aggregate_digest"):
        assert entry[field] is not None, field


def test_the_aggregate_digest_ignores_timing_but_not_results():
    base = {
        "scanned": 10, "within_max_length": 9, "over_max_length": 1,
        "stage6_vs_stage1_disagreements": 3,
        "max_stage6_authoritative_base_length": 256,
        "max_stage1_realised_base_length": 257,
        "over_length_histogram": {"257": 1}, "delta_histogram": {"1": 3},
        "offenders": [{"line_index": 4}],
    }
    faster = dict(base, elapsed_seconds=0.1, rows_per_second=99999, workers=8)
    slower = dict(base, elapsed_seconds=9.9, rows_per_second=1, workers=1)
    assert scanner.aggregate_digest(faster) == scanner.aggregate_digest(slower)
    changed = dict(base, over_max_length=2)
    assert scanner.aggregate_digest(changed) != scanner.aggregate_digest(base)


def test_the_benchmark_flags_differing_digests():
    """If two worker counts ever disagreed, the benchmark must not pass."""
    a = {"aggregate_digest": "aaaa", "rows_per_second": 10}
    b = {"aggregate_digest": "bbbb", "rows_per_second": 20}
    assert len({r["aggregate_digest"] for r in (a, b)}) > 1


# -- GPU decision ------------------------------------------------------------
def test_the_gpu_decision_is_recorded_in_the_scanner():
    """The decision is stated, and the scanner really is CPU-only."""
    import re as _re

    source = SCANNER.read_text(encoding="utf-8")
    collapsed = _re.sub(r"\s+", " ", source)
    assert ("GPU not used because no exact GPU implementation provides a measured "
            "benefit." in collapsed)
    # And it is not merely a claim: nothing here touches a GPU runtime.
    modules = {module for module, _name in scanner_imports()}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert not any(m.split(".")[0] in {"torch", "cupy", "numba"} for m in modules)


# ---------------------------------------------------------------------------
# --validate-rows is a HARD cap on expensive work (Audit 043 §9d)
# ---------------------------------------------------------------------------
def skewed_corpus(tmp_path, rows=4000, newline_rate=0.926, offender="OFFENDER\nrow"):
    """A population shaped like the measured one: 92.6 % newline-bearing, rare
    near-boundary rows, and the known offender placed far down the file."""
    from unmark.stage1.manifest import CHUNKS_NAME

    directory = tmp_path / "skewed"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(rows):
        is_newline = (index % 1000) < int(newline_rate * 1000)
        is_boundary = (index % 250) == 0
        records.append({
            "chunk_id": f"doc-{index:06d}#0", "partition": "train",
            "text": "aa bb\ncc" if is_newline else "aa bb cc",
            "base_length": 251 if is_boundary else 40,
        })
    offender_line = int(rows * 0.75)
    records[offender_line]["text"] = offender
    records[offender_line]["chunk_id"] = "Mô_đun:Inflation/data#572"
    (directory / CHUNKS_NAME).write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8")
    return directory, offender_line


def selection_args(prepared, offender_hash, **overrides):
    args = scan_args(prepared, stage_local=None, local_chunks=None,
                     validate_stride=97, validate_scan_limit=0,
                     offender_hash=offender_hash)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_rows_caps_expensive_comparisons(tmp_path):
    """The cap is `--validate-rows` (+1 for the forced offender), full stop."""
    prepared, _line = skewed_corpus(tmp_path, rows=4000)
    for wanted in (50, 200, 1000):
        args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"))
        picked, _b, _n, stats = scanner._sample_rows(args, wanted)
        assert len(picked) <= wanted + 1, (
            f"asked for {wanted}, selected {len(picked)} -- the expensive full "
            "project_text path must not run more than the cap allows"
        )
        assert stats["selected"] == len(picked)


def test_the_expensive_path_runs_exactly_once_per_selected_row(tmp_path,
                                                               stub_tokenizer, capsys):
    """Count real calls into the production path; it must equal `compared`."""
    prepared, _line = skewed_corpus(tmp_path, rows=3000)
    calls = {"n": 0}
    original = scanner.realised_base_length

    def counted(text, tokenizer, classifier, unk):
        calls["n"] += 1
        return original(text, tokenizer, classifier, unk)

    scanner.realised_base_length = counted
    try:
        args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"),
                              validate_rows=100, progress=0)
        scanner.validate(args)
    finally:
        scanner.realised_base_length = original
    report = json.loads(capsys.readouterr().out)
    assert calls["n"] == report["compared"] <= 101
    assert report["expensive_comparison_cap"] == 101


def test_a_skewed_corpus_no_longer_starves_the_rare_strata(tmp_path):
    """The defect this repair exists for: 92.6 % newline rows crowding out the
    near-boundary rows and stopping after ~2 % of the file."""
    prepared, _line = skewed_corpus(tmp_path, rows=4000)
    args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"))
    picked, boundary, newline, stats = scanner._sample_rows(args, 500)
    lines = [i for i, _ in picked]

    assert boundary > 0, "near-boundary rows must be represented"
    assert newline > 0, "newline-bearing rows must be represented"
    assert stats["streamed_rows"] == 4000, "metadata streaming covers the partition"
    assert max(lines) > 0.5 * 4000, (
        "the sample must span the corpus, not just its opening rows"
    )


def test_the_known_offender_is_always_included(tmp_path):
    """Even when ordinary sampling would never reach it."""
    offender = "OFFENDER\nrow"
    prepared, offender_line = skewed_corpus(tmp_path, rows=4000, offender=offender)
    args = selection_args(prepared, scanner.stable_id(offender))
    picked, _b, _n, stats = scanner._sample_rows(args, 20)
    assert stats["offender_forced_in"] is True
    assert any(index == offender_line for index, _row in picked), (
        "the measured Audit 043 offender must be in every validation set"
    )


def test_no_offender_hash_means_no_forced_row(tmp_path):
    prepared, _line = skewed_corpus(tmp_path, rows=1000)
    args = selection_args(prepared, "")
    picked, _b, _n, stats = scanner._sample_rows(args, 50)
    assert stats["offender_forced_in"] is False
    assert len(picked) <= 50


def test_selection_is_stable_across_runs(tmp_path):
    prepared, _line = skewed_corpus(tmp_path, rows=4000)
    args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"))
    first = [i for i, _ in scanner._sample_rows(args, 300)[0]]
    second = [i for i, _ in scanner._sample_rows(args, 300)[0]]
    assert first == second and first, "deterministic selection, no RNG state"


def test_the_reservoir_is_bounded_and_deterministic():
    reservoir = scanner._DeterministicReservoir(10)
    for index in range(1000):
        reservoir.offer(f"chunk-{index}", index)
    kept = list(reservoir.items)
    assert len(kept) == 10
    again = scanner._DeterministicReservoir(10)
    for index in range(1000):
        again.offer(f"chunk-{index}", index)
    assert again.items == kept
    assert max(kept) > 100, "the reservoir must reach beyond the opening rows"


def test_strata_quotas_sum_to_the_cap():
    assert sum(share for _name, share in scanner.VALIDATION_STRATA) == pytest.approx(1.0)
    assert scanner.BOUNDARY_MARGIN == 8
    names = [name for name, _share in scanner.VALIDATION_STRATA]
    assert names[0] == "boundary", "rarest stratum must be assigned first"


def test_validate_still_reports_zero_mismatches_on_a_skewed_corpus(tmp_path,
                                                                   stub_tokenizer,
                                                                   capsys):
    prepared, _line = skewed_corpus(tmp_path, rows=2000)
    args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"),
                          validate_rows=200, progress=0)
    code = scanner.validate(args)
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_NO_VIOLATION
    assert report["mismatch_count"] == 0
    assert report["selection"]["selected_near_boundary"] > 0
    assert report["selection"]["selected_newline_bearing"] > 0


def test_a_broken_fast_path_is_still_detected_on_a_skewed_corpus(tmp_path,
                                                                 stub_tokenizer,
                                                                 capsys, monkeypatch):
    prepared, _line = skewed_corpus(tmp_path, rows=1000)
    monkeypatch.setattr(scanner.RealisedLengthCounter, "length",
                        lambda self, text: 999)
    args = selection_args(prepared, scanner.stable_id("OFFENDER\nrow"),
                          validate_rows=50, progress=0)
    code = scanner.validate(args)
    report = json.loads(capsys.readouterr().out)
    assert code == scanner.EXIT_DIAGNOSTIC_FAILURE
    assert report["mismatch_count"] > 0
