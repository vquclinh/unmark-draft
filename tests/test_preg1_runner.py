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
    """`BASE_ONLY` is unreachable from `tune` — checked on CODE, not prose.

    `representation_key` legitimately *documents* that a Vanilla cache cannot be
    reloaded as Base-only, so a substring scan would fail on its own docstring.
    What matters is that the tune path never *evaluates* `BASE_ONLY`.
    """
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    tune_fns = {"run_tune", "materialise_split", "tuning_schedule",
                "tuning_artifact", "extract_or_load"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in tune_fns:
            attributes = {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
            assert "BASE_ONLY" not in attributes, f"{node.name} evaluates BASE_ONLY"

    # `representation_key` may mention it in prose but must DEFAULT to VANILLA.
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "representation_key")
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                        fn.args.defaults)) if fn.args.defaults else {}
    assert "pathway" in defaults
    assert ast.unparse(defaults["pathway"]) == "TUNING_PATHWAY"

    cli = load_cli()
    assert cli.representation_key(
        Preg1Role.PROTOCOL_DEV, ["a"], "a" * 64, 8
    ).pathway is SystemPathway.VANILLA


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


def test_measure_reuses_the_committed_trainer_and_scorer():
    """`measure` is now wired; it must reuse, not reimplement."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    for name in ("train_head", "score_measurement", "PairedSeedResult",
                 "PairedDiagnostic", "build_head", "load_derived_pool",
                 "load_membership", "pathway_text", "extract_or_load"):
        assert name in body, f"{name} must be reused"
    # the checkpoint choice comes from the committed selector, not a local scan
    assert "run.selected" in body
    assert "max(" not in body and "argmax" not in body.replace("argmax(dim=1)", "")


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


# ---------------------------------------------------------------------------
# The paired measurement (Audit 026)
# ---------------------------------------------------------------------------
def test_exactly_five_paired_seeds_and_ten_head_runs():
    from unmark.evaluation.preg1_protocol import MEASUREMENT_SEEDS

    assert len(MEASUREMENT_SEEDS) == 5
    assert len(set(MEASUREMENT_SEEDS)) == 5
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    # one loop over the seeds, one inner loop over the two pathways => 5 x 2
    assert "for index, seed in enumerate(MEASUREMENT_SEEDS" in body
    assert "for pathway in (SystemPathway.VANILLA, SystemPathway.BASE_ONLY)" in body
    assert body.count("train_head(") == 1, "one call site, driven by the loops"


def write_tuning(tmp_path, lr=0.01, selected_on="VANILLA", validation_used=False):
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({
        "repository_head": "d10aaae",
        "official_validation_used": validation_used,
        "selection": {
            "selected_learning_rate": lr,
            "frozen": {"learning_rate": lr, "selected_on": selected_on},
            "rule": ["highest MEAN selected-checkpoint Macro-F1 across the tuning seeds"],
        },
    }), encoding="utf-8")
    return path


def test_the_frozen_lr_must_equal_the_tuning_artifact_selection(tmp_path):
    cli = load_cli()
    artifact = write_tuning(tmp_path, lr=0.01)
    assert cli.load_tuning_artifact(artifact, 0.01)["selection"]["selected_learning_rate"] == 0.01
    with pytest.raises(EvaluationContractViolation, match="does not match"):
        cli.load_tuning_artifact(artifact, 0.003)


def test_a_caller_cannot_substitute_another_lr(tmp_path):
    """Every other grid rate is refused against a 0.01 artifact."""
    cli = load_cli()
    artifact = write_tuning(tmp_path, lr=0.01)
    for other in (lr for lr in LR_GRID if lr != 0.01):
        with pytest.raises(EvaluationContractViolation, match="not a caller choice"):
            cli.load_tuning_artifact(artifact, other)


def test_the_lr_must_have_been_selected_on_vanilla(tmp_path):
    """Base-only cannot retroactively become the selecting pathway."""
    cli = load_cli()
    artifact = write_tuning(tmp_path, selected_on="BASE_ONLY")
    with pytest.raises(EvaluationContractViolation, match="must be\\s+selected on VANILLA|selected on"):
        cli.load_tuning_artifact(artifact, 0.01)


def test_a_tuning_run_that_touched_official_validation_is_refused(tmp_path):
    cli = load_cli()
    artifact = write_tuning(tmp_path, validation_used=True)
    with pytest.raises(EvaluationContractViolation, match="official_validation_used"):
        cli.load_tuning_artifact(artifact, 0.01)


def test_a_missing_or_malformed_tuning_artifact_is_refused(tmp_path):
    cli = load_cli()
    with pytest.raises(EvaluationContractViolation, match="not found"):
        cli.load_tuning_artifact(tmp_path / "absent.json", 0.01)
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="malformed"):
        cli.load_tuning_artifact(bad, 0.01)


def test_official_validation_is_gated_on_its_own_locked_identity():
    """It is loaded with its own SHA, rows and label counts — not by filename."""
    from unmark.evaluation.preg1_protocol import (
        DERIVED_VALIDATION_CSV_SHA256, PUBLISHED_LABEL_COUNTS, PUBLISHED_SPLIT_SIZES,
    )

    assert DERIVED_VALIDATION_CSV_SHA256 == (
        "9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0"
    )
    assert PUBLISHED_SPLIT_SIZES["validation"] == 1583
    assert PUBLISHED_LABEL_COUNTS["validation"] == {
        "negative": 705, "neutral": 73, "positive": 805
    }
    body = ast.unparse(next(
        n for n in ast.walk(ast.parse(CLI.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "run_measure"
    ))
    assert "DERIVED_VALIDATION_CSV_SHA256" in body
    assert "PUBLISHED_SPLIT_SIZES['validation']" in body


def test_official_validation_never_selects_a_checkpoint_or_an_lr():
    """`train_head` is called with protocol-train/dev only; scoring is separate."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "train_head")
    supplied = ast.unparse(call)
    assert "PROTOCOL_TRAIN" in supplied and "PROTOCOL_DEV" in supplied
    assert "OFFICIAL_VALIDATION" not in supplied, (
        "official validation must never reach the trainer or the selector"
    )
    scorer = next(n for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "score_measurement")
    assert "OFFICIAL_VALIDATION" in ast.unparse(scorer)
    # and no LR selection happens here at all
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in {"select_learning_rate", "freeze_learning_rate"}]


def test_the_same_seed_drives_both_arms():
    """One seed variable feeds both pathways, so the paired init guarantee holds."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    assert "seed=seed" in body
    assert "build_head(hidden_size, seed)" in body

    # NESTING, checked on the tree. A textual index would find the *extraction*
    # loop over pathways, which legitimately precedes the seed loop.
    seed_loop = next(
        node for node in ast.walk(fn)
        if isinstance(node, ast.For) and "MEASUREMENT_SEEDS" in ast.unparse(node.iter)
    )
    inner = [
        node for node in ast.walk(seed_loop)
        if isinstance(node, ast.For) and "SystemPathway.BASE_ONLY" in ast.unparse(node.iter)
    ]
    assert inner, "the pathway loop must be nested inside the seed loop"
    trainer = [
        node for node in ast.walk(inner[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "train_head"
    ]
    assert trainer, "both arms must be trained inside one seed iteration"


def test_representation_keys_separate_the_two_pathways():
    cli = load_cli()
    vanilla = cli.representation_key(
        Preg1Role.OFFICIAL_VALIDATION, ["a"], "a" * 64, 768, SystemPathway.VANILLA)
    base = cli.representation_key(
        Preg1Role.OFFICIAL_VALIDATION, ["a"], "a" * 64, 768, SystemPathway.BASE_ONLY)
    assert vanilla.pathway is SystemPathway.VANILLA
    assert base.pathway is SystemPathway.BASE_ONLY
    with pytest.raises(EvaluationContractViolation, match="pathway"):
        base.require_compatible(vanilla)


def test_provenance_mismatch_fails_closed_rather_than_reusing_a_cache():
    """`extract_or_load` calls `cache.load(key)`, which compares every field."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "extract_or_load")
    body = ast.unparse(fn)
    assert "cache.load(key)" in body, "the loaded key must be the one we require"


def test_measure_records_no_significance_machinery(tmp_path):
    """The report shape comes from the committed PairedDiagnostic."""
    from unmark.evaluation.preg1_head import (
        FrozenLearningRate, PairedDiagnostic, PairedSeedResult,
    )
    from unmark.evaluation.preg1_protocol import MEASUREMENT_SEEDS

    report = PairedDiagnostic(
        learning_rate=FrozenLearningRate(0.01),
        results=tuple(PairedSeedResult(s, 0.70, 0.80, 0.60, 0.72) for s in MEASUREMENT_SEEDS),
    ).to_dict()

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert not set(keys(report)) & {
        "p_value", "significant", "threshold", "confidence_interval",
        "test_statistic", "verdict", "passed",
    }
    assert len(report["per_seed"]) == 5
    for arm in ("vanilla", "base_only", "delta_vanilla_minus_base_only"):
        assert "mean_macro_f1" in report[arm] and "sample_stdev_macro_f1" in report[arm]


def test_measure_persists_no_raw_text():
    """The artifact is built from digests, keys and floats only."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    assert "report['representations'] = {name: key.to_dict()" in body
    assert "'raw_text_persisted': False" in body
    # texts are used for tokenization only, never written
    assert "json.dumps(report" in body and "texts" not in body.split("json.dumps(report")[1]


def test_measure_refuses_an_existing_output_directory():
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_measure")
    body = ast.unparse(fn)
    assert "output_dir.exists()" in body and "already exists" in body


def test_measure_refuses_a_revision_other_than_the_pinned_one(tmp_path, capsys):
    cli = load_cli()
    status = cli.main([
        "measure", "--split-dir", str(tmp_path), "--derived-train", str(tmp_path / "x.csv"),
        "--text-column", "t", "--label-column", "l", "--id-column", "i",
        "--cache-root", str(tmp_path / "c"), "--output-dir", str(tmp_path / "o"),
        "--official-validation", str(tmp_path / "v.csv"),
        "--frozen-lr", "0.01", "--tuning-artifact", str(tmp_path / "tuning.json"),
        "--revision", "0" * 40,
    ])
    assert status == 2
    assert "pinned diagnostic revision" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Checkpoint snapshots must be independent of later training (Audit 026 review)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the environment
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover - the normal local state
    TORCH = False

requires_torch = pytest.mark.skipif(not TORCH, reason="torch is not installed locally")


def test_the_runner_clones_every_checkpoint_tensor():
    """Structural, runs locally: a bare `state_dict()` would alias live params.

    `Module.state_dict()` hands back tensors that share storage with the live
    parameters, so `saved[epoch] = head.state_dict()` would leave all 30 entries
    tracking the optimizer and every "checkpoint" would end up holding the
    final-epoch weights. The runner must clone. This catches a regression
    without needing torch; `test_checkpoint_snapshots_survive_later_training`
    proves the behaviour.
    """
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    capture = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "capture")
    body = ast.unparse(capture)
    assert "state_dict()" in body
    assert ".clone()" in body, "each checkpoint tensor must be cloned, not aliased"
    # and the aliasing form must not be what is stored
    assert "_store[epoch] = head.state_dict()" not in body


@requires_torch
def test_checkpoint_snapshots_survive_later_training():
    """Behavioural: an early checkpoint must not drift to the final weights.

    Three assertions, in order of what they rule out:

    1. a **bare** `state_dict()` captured in the same run *does* alias — so the
       hazard is real and this test is not vacuous;
    2. the runner's cloned epoch-1 snapshot still differs from epoch 30;
    3. restoring epoch 1 yields exactly epoch 1, not the final weights.
    """
    import torch

    from unmark.evaluation.preg1_head import (
        BoundRepresentations, build_head, train_head,
    )
    from unmark.evaluation.preg1_protocol import EPOCHS

    cli = load_cli()
    hidden, n_train, n_dev = 8, 40, 12
    train_key = cli.representation_key(
        Preg1Role.PROTOCOL_TRAIN, [f"train:{i:05d}" for i in range(n_train)],
        "a" * 64, hidden)
    dev_key = cli.representation_key(
        Preg1Role.PROTOCOL_DEV, [f"train:{i:05d}" for i in range(n_dev)],
        "a" * 64, hidden)
    torch.manual_seed(0)
    train = BoundRepresentations(torch.randn(n_train, hidden), train_key)
    dev = BoundRepresentations(torch.randn(n_dev, hidden), dev_key)
    train_labels = [i % 3 for i in range(n_train)]
    dev_labels = [i % 3 for i in range(n_dev)]

    cloned: dict[int, dict] = {}
    aliased: dict[int, dict] = {}

    def capture(epoch, score, head):
        # exactly what the runner does
        cloned[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        # the unsafe form, kept only to prove the hazard is real
        aliased[epoch] = head.state_dict()

    seed = 53148
    run = train_head(
        train, train_labels, dev, dev_labels,
        learning_rate=0.01, seed=seed, epochs=EPOCHS, on_checkpoint=capture,
    )
    assert len(cloned) == EPOCHS

    # 1. the hazard is real: the aliased epoch-1 entry now holds final weights.
    assert torch.equal(aliased[1]["weight"], aliased[EPOCHS]["weight"]), (
        "a bare state_dict() no longer aliases live parameters — the runner's "
        "clone may be redundant, but this assumption must be reviewed, not assumed"
    )

    # 2. the cloned snapshot did NOT drift.
    assert not torch.equal(cloned[1]["weight"], cloned[EPOCHS]["weight"])
    assert not torch.equal(cloned[1]["weight"], aliased[1]["weight"])

    # 3. restoring epoch 1 returns exactly epoch 1.
    head = build_head(hidden, seed)
    head.load_state_dict(cloned[1])
    assert torch.equal(head.weight, cloned[1]["weight"])
    assert torch.equal(head.bias, cloned[1]["bias"])
    assert not torch.equal(head.weight, cloned[EPOCHS]["weight"]), (
        "restoring an early checkpoint must not yield the final-epoch weights"
    )

    # and the selector's choice is restorable for any eligible epoch
    assert run.selected.epoch in cloned


@requires_torch
def test_every_epoch_snapshot_is_a_distinct_object_with_its_own_storage():
    """No two cloned checkpoints may share a tensor."""
    import torch

    from unmark.evaluation.preg1_head import BoundRepresentations, train_head
    from unmark.evaluation.preg1_protocol import EPOCHS

    cli = load_cli()
    hidden = 8
    key_t = cli.representation_key(
        Preg1Role.PROTOCOL_TRAIN, [f"train:{i:05d}" for i in range(20)], "a" * 64, hidden)
    key_d = cli.representation_key(
        Preg1Role.PROTOCOL_DEV, [f"train:{i:05d}" for i in range(9)], "a" * 64, hidden)
    torch.manual_seed(0)
    train = BoundRepresentations(torch.randn(20, hidden), key_t)
    dev = BoundRepresentations(torch.randn(9, hidden), key_d)

    snaps: dict[int, dict] = {}

    def capture(epoch, score, head):
        snaps[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}

    train_head(train, [i % 3 for i in range(20)], dev, [i % 3 for i in range(9)],
               learning_rate=0.01, seed=720, epochs=EPOCHS, on_checkpoint=capture)

    pointers = {epoch: snaps[epoch]["weight"].data_ptr() for epoch in snaps}
    assert len(set(pointers.values())) == EPOCHS, "checkpoints share tensor storage"


# ===========================================================================
# SECONDARY OWN-LR SENSITIVITY (Audit 027)
#
# The secondary analysis was precommitted in `preg1_protocol` before any
# primary result existed. These tests check the wiring that keeps it SECONDARY:
# same grid, same seeds, Base-only retuned, Vanilla neither retuned nor rerun,
# and an artifact that names itself rather than passing as the primary.
# ===========================================================================
def sensitivity_fn():
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    return tree, next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_sensitivity"
    )


def test_the_secondary_analysis_was_precommitted_not_invented_after_the_fact():
    """The precommitment must live in the protocol, not in Audit 027's code."""
    from unmark.evaluation.preg1_protocol import (
        PRIMARY_LR_CAVEAT,
        SECONDARY_SENSITIVITY,
    )

    assert "SECONDARY sensitivity analysis" in SECONDARY_SENSITIVITY
    assert "same grid" in SECONDARY_SENSITIVITY
    assert "MUST NOT replace" in SECONDARY_SENSITIVITY
    # and the primary must already have disclaimed being a bound
    assert "does NOT make Vanilla an upper bound" in PRIMARY_LR_CAVEAT


def test_sensitivity_retunes_base_only_and_nothing_else():
    cli = load_cli()
    assert cli.SENSITIVITY_PATHWAY is SystemPathway.BASE_ONLY
    assert cli.TUNING_PATHWAY is SystemPathway.VANILLA, "the primary is untouched"

    _, fn = sensitivity_fn()
    # Every pathway-carrying call in the sensitivity body names the constant,
    # so no call site can quietly encode the other arm.
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {
            "pathway_text", "representation_key"
        }:
            rendered = ast.unparse(node)
            assert "SENSITIVITY_PATHWAY" in rendered, rendered
            assert "VANILLA" not in rendered, rendered


def test_vanilla_is_neither_retuned_nor_rerun():
    """The comparator is READ from the primary artifact, never recomputed."""
    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    assert "primary_vanilla_by_seed" in body or "primary_vanilla[" in body
    # no Vanilla representations are built here at all
    assert "SystemPathway.VANILLA" not in body, (
        "run_sensitivity must not construct a Vanilla arm; it reads the primary"
    )


def test_phase_a_is_exactly_the_precommitted_fifteen_runs():
    cli = load_cli()
    assert len(cli.tuning_schedule()) == len(LR_GRID) * len(TUNING_SEEDS) == 15

    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    # the SAME schedule helper the primary uses -- not a second schedule
    assert "tuning_schedule()" in body
    assert "for index, (lr, seed) in enumerate(schedule" in body


def test_phase_b_is_five_runs_not_ten():
    """The secondary trains ONE arm per seed; the other arm is already measured."""
    from unmark.evaluation.preg1_protocol import MEASUREMENT_SEEDS

    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    assert "for index, seed in enumerate(MEASUREMENT_SEEDS" in body
    assert "for pathway in" not in body, (
        "a pathway loop would retrain Vanilla and double Phase B to 10 runs"
    )
    # two train_head call sites: one for Phase A, one for Phase B
    assert body.count("train_head(") == 2
    assert len(MEASUREMENT_SEEDS) == 5


def test_the_grid_is_not_expanded_post_hoc():
    """`require_full_grid` is never disabled, anywhere in the runner."""
    source = CLI.read_text(encoding="utf-8")
    assert "require_full_grid=False" not in source
    assert "expected_seeds=None" not in source

    tree, fn = sensitivity_fn()
    # the candidates the selector sees are built over LR_GRID itself
    body = ast.unparse(fn)
    assert "for lr in LR_GRID" in body
    # and no numeric LR literal is introduced in the sensitivity path
    literals = {
        n.value for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, float)
    }
    assert not literals - set(LR_GRID), f"stray LR literal(s): {literals}"


def test_official_validation_is_opened_only_after_the_base_only_lr_is_frozen():
    """Ordering, not just intent: the file is untouched during selection."""
    _, fn = sensitivity_fn()
    freeze_at = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "freeze_learning_rate"
    ]
    validation_at = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr == "official_validation"
    ]
    assert freeze_at and validation_at
    assert min(validation_at) > max(freeze_at), (
        "official validation must not be read, encoded or cached during Phase A"
    )


def test_official_test_stays_unreachable_from_the_sensitivity_path():
    assert not hasattr(Preg1Role, "OFFICIAL_TEST")
    _, fn = sensitivity_fn()
    roles = {
        n.attr for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "Preg1Role"
    }
    assert roles <= {"PROTOCOL_TRAIN", "PROTOCOL_DEV", "OFFICIAL_VALIDATION"}, roles


def test_sensitivity_exposes_no_scientific_overrides():
    """Runtime paths only. A flag that could move a precommitted value is the hole."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    flags = {
        n.args[0].value for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and getattr(getattr(n.func, "attr", None), "__str__", str)() == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant)
    }
    for forbidden in ("--learning-rate", "--lr", "--seeds", "--seed", "--grid",
                      "--epochs", "--batch-size", "--max-length", "--pooling",
                      "--base-only-lr"):
        assert forbidden not in flags, f"{forbidden} could override a locked value"


def write_primary_measurement(tmp_path, *, lr=0.01, selected_on="VANILLA",
                              seeds=None, analysis=None, base_only_lr=None,
                              measured_on="official-validation"):
    from unmark.evaluation.preg1_protocol import MEASUREMENT_SEEDS

    seeds = MEASUREMENT_SEEDS if seeds is None else seeds
    report = {
        "repository_head": "929f80e",
        "measured_on": measured_on,
        "learning_rate": {"learning_rate": lr, "selected_on": selected_on},
        "per_seed": [
            {"seed": seed, "vanilla_macro_f1": 0.74, "vanilla_accuracy": 0.90,
             "base_only_macro_f1": 0.66, "base_only_accuracy": 0.82}
            for seed in seeds
        ],
        "vanilla": {"mean_macro_f1": 0.74},
        "base_only": {"mean_macro_f1": 0.66},
        "delta_vanilla_minus_base_only": {"mean_macro_f1": 0.08},
        "boundaries": {"official_test_used": False, "encoder_trained": False},
    }
    if analysis is not None:
        report["analysis"] = analysis
    if base_only_lr is not None:
        report["base_only_learning_rate"] = {"learning_rate": base_only_lr}
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_the_primary_comparator_is_verified_not_trusted(tmp_path):
    cli = load_cli()
    good = cli.load_primary_measurement(write_primary_measurement(tmp_path))
    assert len(good["per_seed"]) == 5

    by_seed = cli.primary_vanilla_by_seed(good)
    from unmark.evaluation.preg1_protocol import MEASUREMENT_SEEDS
    assert sorted(by_seed) == sorted(MEASUREMENT_SEEDS)
    assert by_seed[MEASUREMENT_SEEDS[0]] == (0.74, 0.90)


@pytest.mark.parametrize("kwargs, message", [
    ({"selected_on": "BASE_ONLY"}, "selected on VANILLA"),
    ({"analysis": "SECONDARY OWN-LR SENSITIVITY"}, "PRIMARY shared-LR"),
    ({"base_only_lr": 0.003}, "not the primary"),
    ({"seeds": (53148, 59945)}, "needs all of"),
    ({"measured_on": "protocol-dev"}, "reports on"),
    ({"lr": 5e-3}, "not in the precommitted grid"),
])
def test_a_non_primary_artifact_is_refused_as_the_comparator(tmp_path, kwargs, message):
    cli = load_cli()
    path = write_primary_measurement(tmp_path, **kwargs)
    with pytest.raises(EvaluationContractViolation, match=message):
        cli.load_primary_measurement(path)


def test_a_missing_or_malformed_primary_is_refused(tmp_path):
    cli = load_cli()
    with pytest.raises(EvaluationContractViolation, match="not found"):
        cli.load_primary_measurement(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="malformed"):
        cli.load_primary_measurement(bad)


def test_the_two_primary_artifacts_are_cross_checked_against_each_other():
    """The measurement's LR must be the one the tuning sweep actually selected.

    `run_sensitivity` passes the LR it read from the measurement into
    `load_tuning_artifact`, so a mismatched pair fails closed. Nothing supplies
    the LR on the command line.
    """
    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    assert "load_tuning_artifact(Path(args.primary_tuning), primary_lr_value)" in body
    assert "args.frozen_lr" not in body, "the secondary takes no LR from the caller"


def test_the_sensitivity_artifact_names_itself_secondary():
    from unmark.evaluation.preg1_protocol import (
        PRIMARY_ANALYSIS_LABEL,
        SECONDARY_ANALYSIS_LABEL,
    )

    assert SECONDARY_ANALYSIS_LABEL == "SECONDARY OWN-LR SENSITIVITY"
    assert PRIMARY_ANALYSIS_LABEL == "PRIMARY SHARED-LR"
    for word in ("upper bound", "lower bound", "significan", "corrected"):
        assert word not in SECONDARY_ANALYSIS_LABEL.lower()

    # It is `base_only_learning_rate=` that makes the report label itself
    # SECONDARY -- printing the constant in a banner would not. Assert the
    # artifact-producing call actually passes it.
    _, fn = sensitivity_fn()
    diagnostic = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "PairedDiagnostic"
    )
    supplied = {kw.arg for kw in diagnostic.keywords}
    assert "base_only_learning_rate" in supplied, supplied
    assert ast.unparse(
        next(kw for kw in diagnostic.keywords if kw.arg == "base_only_learning_rate")
    ).endswith("base_frozen"), "the retuned Base-only LR, not the primary one"
    assert ast.unparse(
        next(kw for kw in diagnostic.keywords if kw.arg == "learning_rate")
    ).endswith("primary_frozen"), "the Vanilla arm stays the primary shared LR"

    body = ast.unparse(fn)
    assert "SECONDARY_ANALYSIS_LABEL" in body
    # the primary numbers are carried through for reference, clearly labelled
    assert "primary_reference_burden" in body
    assert "primary_provenance" in body


def test_the_sensitivity_artifact_reports_no_significance_machinery():
    _, fn = sensitivity_fn()
    body = ast.unparse(fn).lower()
    for banned in ("p_value", "p-value", "pvalue", "threshold", "significant",
                   "passes", "fails", "reject"):
        assert banned not in body, f"{banned} has no place in a descriptive report"


def test_sensitivity_refuses_an_existing_output_directory():
    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    assert "output_dir.exists()" in body and "already exists" in body


def test_sensitivity_reuses_the_committed_apis():
    """No second trainer, selector, metric, checkpoint rule or protocol."""
    _, fn = sensitivity_fn()
    body = ast.unparse(fn)
    for name in ("train_head", "select_learning_rate", "freeze_learning_rate",
                 "LrCandidate", "PairedSeedResult", "PairedDiagnostic",
                 "score_measurement", "build_head", "extract_or_load"):
        assert name in body, f"{name} must be reused, not reimplemented"
    # and it defines none of them itself
    tree, _ = sensitivity_fn()
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("train_head", "select_learning_rate", "score_measurement",
                 "freeze_learning_rate", "build_head"):
        assert name not in defined, f"the runner must not define its own {name}"
