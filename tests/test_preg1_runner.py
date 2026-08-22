"""The executable pre-G1 diagnostic runner (`scripts/preg1_head_diagnostic.py`).

Tests the **wiring**, not the protocol — the protocol has its own tests. Every
test here is ML-free: the module must import without torch, and nothing below
loads a model, reads a corpus, trains a head or produces a score.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import pytest

from unmark.evaluation.contracts import EvaluationContractViolation, SystemPathway
from unmark.evaluation.preg1_head import Preg1Role, SplitMembership
from unmark.evaluation.preg1_protocol import (
    BATCH_SIZE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EPOCHS,
    LR_GRID,
    MAX_LENGTH,
    PADDING,
    PREG1_POOLING,
    TRUNCATION,
    TUNING_SEEDS,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "preg1_head_diagnostic.py"


def load_cli():
    """Import the runner. It must not need torch to be importable."""
    spec = importlib.util.spec_from_file_location("preg1_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_imports_without_torch():
    before = "torch" in sys.modules
    load_cli()
    assert before or "torch" not in sys.modules, "torch must be imported lazily"


# ---------------------------------------------------------------------------
# The schedule: exactly 5 x 3, and derived from the protocol
# ---------------------------------------------------------------------------
def test_exactly_five_by_three_runs_are_scheduled():
    schedule = load_cli().tuning_schedule()
    assert len(schedule) == len(LR_GRID) * len(TUNING_SEEDS) == 15
    assert {lr for lr, _ in schedule} == set(LR_GRID)
    assert {seed for _, seed in schedule} == set(TUNING_SEEDS)
    assert len(set(schedule)) == 15, "every (lr, seed) pair must be distinct"


def test_the_schedule_follows_the_protocol_rather_than_a_typed_count():
    """AST: the schedule is built from LR_GRID and TUNING_SEEDS, not `range(15)`."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "tuning_schedule")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert {"LR_GRID", "TUNING_SEEDS"} <= names
    numbers = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}
    assert not numbers, f"the schedule must not hard-code counts, found {numbers}"


def test_no_scientific_value_is_restated_in_the_runner():
    """Every locked number is imported; none is typed into the script."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    numbers = {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    }
    forbidden = {30, 128, 256, 0.01, 1e-8, 0.9, 0.999,
                 5509, 19422, 11800, 53148, 59945, 42941, 720, 9428,
                 0.0001, 0.0003, 0.001, 0.003}
    assert not numbers & forbidden, f"restated locked values: {sorted(numbers & forbidden)}"


# ---------------------------------------------------------------------------
# Boundaries: pathway, roles, official validation, official TEST
# ---------------------------------------------------------------------------
def test_tuning_is_vanilla_only():
    cli = load_cli()
    assert cli.TUNING_PATHWAY is SystemPathway.VANILLA


def test_base_only_never_appears_in_the_tune_path():
    """AST over `run_tune` and its helpers: `BASE_ONLY` is not reachable."""
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    tune_fns = {"run_tune", "materialise_split", "representation_key",
                "tuning_schedule", "tuning_artifact", "extract_or_load"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in tune_fns:
            body = ast.unparse(node)
            assert "BASE_ONLY" not in body, f"{node.name} references BASE_ONLY"


def test_tune_may_only_touch_protocol_train_and_protocol_dev():
    cli = load_cli()
    assert cli.TUNING_ROLES == (Preg1Role.PROTOCOL_TRAIN, Preg1Role.PROTOCOL_DEV)
    assert Preg1Role.OFFICIAL_VALIDATION not in cli.TUNING_ROLES


def test_materialise_split_refuses_official_validation():
    cli = load_cli()

    class Pool:
        records = (("a", "x", 0),)

    with pytest.raises(EvaluationContractViolation, match="may only materialise"):
        cli.materialise_split(Pool(), ["a"], Preg1Role.OFFICIAL_VALIDATION)


def test_tune_has_no_official_validation_argument():
    """`measure` takes it; `tune` structurally cannot."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    tune_flags, measure_flags, current = set(), set(), None
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets and targets[0] in ("tune", "measure"):
                current = targets[0]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.func.value, ast.Name)):
            (tune_flags if node.func.value.id == "tune" else
             measure_flags if node.func.value.id == "measure" else set()
             ).add(node.args[0].value)
    assert "--official-validation" not in tune_flags
    assert "--official-validation" in measure_flags


def test_official_test_remains_unreachable():
    """Structural, not substring — the docstring legitimately *names* the absence.

    A grep for "OFFICIAL_TEST" matches the paragraph explaining that no such
    role exists, which proves nothing. What matters is that no attribute access
    reaches it, no CLI flag carries it, and the enum has no such member.
    """
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "OFFICIAL_TEST" not in attributes

    flags = {
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert not {"--test", "--official-test", "--test-split"} & flags
    assert "OFFICIAL_TEST" not in {role.name for role in Preg1Role}


def test_no_cli_override_for_a_locked_scientific_value():
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    flags = {
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert not flags & {
        "--seeds", "--tuning-seeds", "--measurement-seeds", "--grid",
        "--learning-rate", "--epochs", "--batch-size", "--max-length",
        "--padding", "--truncation", "--pooling",
    }
    # runtime-only resource paths are fine
    assert {"--derived-train", "--split-dir", "--cache-root", "--output-dir"} <= flags


def test_measure_still_requires_a_frozen_lr():
    cli = load_cli()
    with pytest.raises(SystemExit):
        cli.main(["measure", "--split-dir", "x", "--derived-train", "x",
                  "--text-column", "t", "--label-column", "l", "--id-column", "i",
                  "--cache-root", "c", "--output-dir", "o",
                  "--official-validation", "v"])  # no --frozen-lr


def test_measure_does_not_execute_a_measurement():
    """The first downstream number is deliberately not wired here."""
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    assert "train_head" not in body and "score_measurement" not in body
    assert "NOT EXECUTED" in body


# ---------------------------------------------------------------------------
# Representation provenance and reuse
# ---------------------------------------------------------------------------
def test_representation_key_carries_the_locked_contract():
    cli = load_cli()
    key = cli.representation_key(
        Preg1Role.PROTOCOL_DEV, ["train:00001", "train:00002"], "a" * 64, 768
    )
    assert key.role is Preg1Role.PROTOCOL_DEV
    assert key.pathway is SystemPathway.VANILLA
    assert key.tokenizer_id == ENCODER_CHECKPOINT
    assert key.model_revision == ENCODER_REVISION
    assert key.max_length == MAX_LENGTH
    assert key.padding == PADDING
    assert key.truncation == TRUNCATION
    assert key.pooling == PREG1_POOLING.value
    assert key.dtype == "torch.float32"
    assert key.count == 2 and key.hidden_size == 768


def test_representation_key_pins_the_membership_order():
    cli = load_cli()
    a = cli.representation_key(Preg1Role.PROTOCOL_DEV, ["x", "y"], "a" * 64, 8)
    b = cli.representation_key(Preg1Role.PROTOCOL_DEV, ["y", "x"], "a" * 64, 8)
    assert a.ordered_id_digest != b.ordered_id_digest


def test_representations_are_extracted_once_per_role_and_then_reused():
    """AST: `extract_or_load` returns the cache before touching the encoder."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "extract_or_load")
    # Compare the CALLS, not the first textual occurrence: the lazy
    # `from ... import extract_representations` sits above `cache.load`, so a
    # naive substring order check measures the import and proves nothing.
    def call_line(name, attr=False):
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                func = node.func
                if attr and isinstance(func, ast.Attribute) and func.attr == name:
                    return node.lineno
                if not attr and isinstance(func, ast.Name) and func.id == name:
                    return node.lineno
        return None

    load_at = call_line("load", attr=True)
    extract_at = call_line("extract_representations")
    save_at = call_line("save", attr=True)
    assert load_at is not None and extract_at is not None and save_at is not None
    assert load_at < extract_at, "a cache hit must short-circuit extraction"

    # ... and the early return has to actually be there.
    returns = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert any(r < extract_at for r in returns), "cache hit must return before extracting"
    body = ast.unparse(fn)

    tune = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "run_tune")
    tune_body = ast.unparse(tune)
    assert tune_body.count("extract_or_load(") == 1, (
        "extraction must happen in one loop over TUNING_ROLES, not per head run"
    )
    train_at = tune_body.index("extract_or_load(")
    schedule_at = tune_body.index("tuning_schedule()")
    assert train_at < schedule_at, "representations must exist before the 15 runs"


def test_the_runner_reuses_committed_apis_and_defines_no_second_trainer():
    source = CLI.read_text(encoding="utf-8")
    for name in ("load_membership", "load_derived_pool", "train_head",
                 "select_learning_rate", "freeze_learning_rate",
                 "RepresentationCache", "pathway_text"):
        assert name in source, f"{name} must be reused, not reimplemented"
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"train_head", "select_learning_rate", "macro_f1",
                          "accuracy", "select_checkpoint"}


# ---------------------------------------------------------------------------
# The tuning artifact
# ---------------------------------------------------------------------------
def fake_run(pathway, lr, seed, f1=0.5, acc=0.5):
    from unmark.evaluation.preg1_head import EpochScore, HeadRun

    return HeadRun(pathway=pathway, learning_rate=lr, seed=seed,
                   scores=tuple(EpochScore(e, f1, acc) for e in range(1, EPOCHS + 1)),
                   hidden_size=768)


def build_artifact():
    cli = load_cli()
    from unmark.evaluation.preg1_head import LrCandidate, freeze_learning_rate, select_learning_rate

    candidates = [
        LrCandidate(lr, tuple(fake_run(SystemPathway.VANILLA, lr, s,
                                       0.9 if lr == LR_GRID[2] else 0.5)
                              for s in TUNING_SEEDS))
        for lr in LR_GRID
    ]
    winner = select_learning_rate(candidates)
    membership = SplitMembership(("t1", "t2"), ("d1",), assignment_digest="deadbeef")
    keys = {
        role.value: cli.representation_key(role, ["a", "b"], "a" * 64, 768)
        for role in cli.TUNING_ROLES
    }
    return cli.tuning_artifact(
        repository_head="cafe1234", membership=membership, source_sha256="a" * 64,
        keys=keys, candidates=candidates, winner=winner,
        frozen=freeze_learning_rate(winner),
    ), winner


def test_the_artifact_records_all_fifteen_runs():
    artifact, _ = build_artifact()
    assert len(artifact["runs"]) == 15
    assert artifact["schedule"]["planned_runs"] == 15
    assert len(artifact["per_learning_rate"]) == len(LR_GRID)
    assert {run["pathway"] for run in artifact["runs"]} == {"VANILLA"}
    assert {run["scored_on"] for run in artifact["runs"]} == {"protocol-dev"}


def test_the_artifact_selection_comes_from_the_committed_selector():
    artifact, winner = build_artifact()
    assert artifact["selection"]["selected_learning_rate"] == winner.learning_rate
    assert artifact["selection"]["frozen"]["selected_on"] == "VANILLA"
    assert artifact["selection"]["rule"], "the aggregation rule must be recorded"


def test_the_artifact_records_provenance_and_boundaries():
    artifact, _ = build_artifact()
    assert artifact["repository_head"] == "cafe1234"
    assert set(artifact["representations"]) == {"protocol-train", "protocol-dev"}
    assert artifact["official_validation_used"] is False
    assert artifact["official_test_used"] is False
    assert artifact["boundaries"]["downstream_score"] is None
    assert artifact["boundaries"]["raw_text_persisted"] is False
    assert artifact["pathway"] == "VANILLA"


def test_the_artifact_is_json_safe_and_deterministic():
    first, _ = build_artifact()
    second, _ = build_artifact()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_artifact_persists_no_raw_text():
    artifact, _ = build_artifact()
    blob = json.dumps(artifact)
    marked = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    assert not set(blob.lower()) & marked

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert "text" not in set(keys(artifact))


def test_the_artifact_carries_no_runtime_varying_field():
    artifact, _ = build_artifact()

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    banned = {"timestamp", "created_at", "generated_at", "hostname", "cwd", "elapsed"}
    assert not set(keys(artifact)) & banned


# ---------------------------------------------------------------------------
# Runtime guards
# ---------------------------------------------------------------------------
def test_tune_refuses_a_revision_other_than_the_pinned_one(tmp_path, capsys):
    cli = load_cli()
    status = cli.main([
        "tune", "--split-dir", str(tmp_path), "--derived-train", str(tmp_path / "x.csv"),
        "--text-column", "t", "--label-column", "l", "--id-column", "i",
        "--cache-root", str(tmp_path / "c"), "--output-dir", str(tmp_path / "o"),
        "--revision", "0" * 40,
    ])
    assert status == 2
    assert "pinned diagnostic revision" in capsys.readouterr().err


def test_tune_refuses_an_existing_output_directory(tmp_path):
    """Tuning artifacts are immutable, like the split membership."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_tune")
    body = ast.unparse(fn)
    assert "already exists" in body and "output_dir.exists()" in body
