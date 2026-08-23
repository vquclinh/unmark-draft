"""CLI/API boundaries, validation contract, manifest and model contract.

ML-free: AST and contract assertions, no torch, no model, no corpus.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "stage1_runner.py"
STAGE1 = REPO / "unmark" / "stage1"


def load_cli():
    spec = importlib.util.spec_from_file_location("stage1_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_of(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------
def test_the_runner_imports_without_torch():
    import sys

    before = "torch" in sys.modules
    load_cli()
    assert before or "torch" not in sys.modules, "torch must be imported lazily"


@pytest.mark.parametrize("module", [
    "protocol.py", "corpus.py", "chunking.py", "sampler.py",
    "selection.py", "optim.py", "manifest.py", "trainer.py", "execute.py",
])
def test_stage1_modules_do_not_import_torch_at_module_scope(module):
    tree = tree_of(STAGE1 / module)
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "torch" for a in node.names), module
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "torch", module


# ---------------------------------------------------------------------------
# Boundaries: no TEST, no scientific overrides
# ---------------------------------------------------------------------------
def test_there_is_no_official_test_argument_anywhere():
    main = next(
        n for n in ast.walk(tree_of(CLI))
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    flags = {
        n.args[0].value
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant)
    }
    assert flags, "no flags were discovered; the check would be vacuous"
    for flag in flags:
        assert "test" not in flag.lower(), f"{flag} could route to official TEST"


def test_no_scientific_value_can_be_overridden_from_the_cli():
    main = next(
        n for n in ast.walk(tree_of(CLI))
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    flags = {
        n.args[0].value
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant)
    }
    for forbidden in (
        "--lr", "--learning-rate", "--r", "--lambda-align", "--lambda-clean",
        "--batch-size", "--epochs", "--max-updates", "--pi-strip", "--scope",
        "--corruption-scope", "--seed", "--seeds", "--dev-documents", "--max-length",
        "--eval-every", "--weight-decay", "--precision", "--amp",
    ):
        assert forbidden not in flags, f"{forbidden} overrides a locked scientific value"


def test_the_only_uitvsfc_inputs_are_the_two_already_opened_ones():
    source = CLI.read_text(encoding="utf-8")
    tree = tree_of(CLI)
    flags = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant)
    }
    uitvsfc = {f for f in flags if "uitvsfc" in f}
    assert uitvsfc == {"--uitvsfc-derived-train", "--uitvsfc-official-validation"}


def test_prepare_corpus_is_the_only_command_taking_uitvsfc_paths():
    """Structural: resolve which subparser each --uitvsfc flag is added to.

    A substring scan over `ast.unparse` would be fragile (quote normalisation)
    and would not actually prove *which* parser owns the flag.
    """
    fn = next(
        n for n in ast.walk(tree_of(CLI))
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    # variable name -> subcommand it was created for
    subcommand_of: dict[str, str] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "attr", None) == "add_parser"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.targets[0], ast.Name)
        ):
            subcommand_of[node.targets[0].id] = node.value.args[0].value
    assert subcommand_of, "no subparsers discovered; the check would be vacuous"

    owners = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args and isinstance(node.args[0], ast.Constant)
            and "uitvsfc" in str(node.args[0].value)
            and isinstance(node.func.value, ast.Name)
        ):
            owners.add(subcommand_of.get(node.func.value.id, node.func.value.id))
    assert owners == {"prepare-corpus"}, (
        f"UIT-VSFC paths reachable from {sorted(owners)}; only prepare-corpus may take them"
    )


# ---------------------------------------------------------------------------
# smoke cannot step
# ---------------------------------------------------------------------------
def test_smoke_is_structurally_incapable_of_an_optimizer_step():
    fn = next(
        n for n in ast.walk(tree_of(STAGE1 / "execute.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "smoke_check"
    )
    called = {
        getattr(n.func, "attr", None) or getattr(n.func, "id", None)
        for n in ast.walk(fn) if isinstance(n, ast.Call)
    }
    for forbidden in ("backward", "step", "build_optimizer", "AdamW", "zero_grad"):
        assert forbidden not in called, f"smoke_check calls {forbidden}()"


# ---------------------------------------------------------------------------
# Validation contract
# ---------------------------------------------------------------------------
def test_validation_grid_and_seed_are_the_locked_ones():
    from unmark.stage1.validation import VALIDATION_CONTRACT, condition_for
    from unmark.stage1.protocol import VALIDATION_CONDITIONS, VALIDATION_CORRUPTION_SEED

    assert tuple(VALIDATION_CONTRACT["conditions"]) == VALIDATION_CONDITIONS
    assert VALIDATION_CONTRACT["conditions"] == ["FULL", "P50", "P100", "STRIP_ALL"]
    assert VALIDATION_CONTRACT["corruption_seed"] == VALIDATION_CORRUPTION_SEED == 19225
    assert VALIDATION_CONTRACT["labels_used"] is False
    assert VALIDATION_CONTRACT["downstream_task_used"] is False
    assert VALIDATION_CONTRACT["training_seed_affects_validation_corruption"] is False

    expected = {
        "FULL": ("NONE", 0.0), "P50": ("TONE", 0.5),
        "P100": ("TONE", 1.0), "STRIP_ALL": ("TONE_AND_LETTER", 1.0),
    }
    for name, (scope, p) in expected.items():
        condition = condition_for(name)
        assert condition.scope.value == scope and condition.probability == p


def test_off_grid_validation_conditions_are_refused():
    from unmark.stage1.validation import ValidationContractViolation, condition_for

    for name in ("P25", "P75", "VARIANT"):
        with pytest.raises(ValidationContractViolation, match="not in the locked validation grid"):
            condition_for(name)


def test_the_training_seed_cannot_change_validation_corruption():
    """`prepare_condition_batch` must key on the validation seed, not a run seed."""
    fn = next(
        n for n in ast.walk(tree_of(STAGE1 / "validation.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_condition_batch"
    )
    body = ast.unparse(fn)
    assert "corruption_seed=VALIDATION_CORRUPTION_SEED" in body
    assert "run_seed" not in body and "provenance" not in body


def test_validation_reuses_the_training_preparation_path():
    """One preparation implementation, so "validation sees the training pipeline"
    is a fact rather than a claim."""
    body = (STAGE1 / "validation.py").read_text(encoding="utf-8")
    assert "prepare_with_condition" in body
    tree = tree_of(STAGE1 / "validation.py")
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "prepare_example" not in defined and "corrupt" not in defined


# ---------------------------------------------------------------------------
# Optimizer / model contract
# ---------------------------------------------------------------------------
def test_weight_decay_groups_follow_the_locked_policy():
    from unmark.stage1.optim import parameter_group_plan
    from unmark.stage1.protocol import WEIGHT_DECAY_EXEMPT, WEIGHT_DECAY_WEIGHTS

    names = [
        "adapter.fusion.weight", "adapter.fusion.bias",
        "adapter.gate.weight", "adapter.gate.bias",
        "adapter.layernorm.weight", "adapter.layernorm.bias",
        "adapter.tone_embedding.weight", "adapter.letter_embedding.weight",
    ]
    plan = parameter_group_plan(names)
    assert plan["decay"]["names"] == ["adapter.fusion.weight", "adapter.gate.weight"]
    assert plan["decay"]["weight_decay"] == WEIGHT_DECAY_WEIGHTS == 0.01
    assert plan["exempt"]["weight_decay"] == WEIGHT_DECAY_EXEMPT == 0.0
    for exempt in ("bias", "layernorm", "tone_embedding", "letter_embedding"):
        assert any(exempt in n for n in plan["exempt"]["names"])


def test_the_optimizer_refuses_frozen_parameters():
    """The encoder must never be handed to the optimizer, and silently filtering
    it out would hide a wiring error instead of surfacing it."""
    fn = next(
        n for n in ast.walk(tree_of(STAGE1 / "optim.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "build_optimizer"
    )
    body = ast.unparse(fn)
    assert "requires_grad" in body and "frozen" in body


def test_stage1_pooling_is_masked_mean_not_first_token():
    from unmark.stage1.protocol import STAGE1_POOLING

    assert STAGE1_POOLING == "attention_masked_mean_non_special"
    objective = (REPO / "unmark" / "stage1" / "objective.py").read_text(encoding="utf-8")
    assert "masked_mean_non_special" in objective
    assert "FIRST_TOKEN" not in objective, "FIRST_TOKEN was pre-G1 only"
    for module in ("trainer.py", "validation.py", "execute.py"):
        assert "FIRST_TOKEN" not in (STAGE1 / module).read_text(encoding="utf-8")


def test_no_mixed_precision_is_introduced():
    from unmark.stage1.protocol import PRECISION

    assert PRECISION == "fp32"
    for module in ("trainer.py", "execute.py", "validation.py"):
        source = (STAGE1 / module).read_text(encoding="utf-8")
        tree = tree_of(STAGE1 / module)
        called = {
            getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        for forbidden in ("autocast", "GradScaler", "half", "bfloat16"):
            assert forbidden not in called, f"{module} uses {forbidden}"


def test_adapter_capacity_is_unchanged():
    from unmark.stage1.protocol import ADAPTER_TRAINABLE_PARAMETERS, HIDDEN_SIZE

    assert HIDDEN_SIZE == 768
    assert ADAPTER_TRAINABLE_PARAMETERS == 3_551_232 == 6 * 768**2 + 16 * 768


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def manifest_dict(**overrides):
    from unmark.stage1.manifest import MANIFEST_SCHEMA_VERSION
    from unmark.stage1.protocol import (
        CONTAMINATION_METHOD, CORPUS_REVISION, DEV_DOCUMENTS,
        ENCODER_REVISION, MAX_LENGTH, SPLIT_SEED, STAGE1_PROTOCOL_VERSION,
    )

    base = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "official_test_used": False,
        "source": {"revision": CORPUS_REVISION},
        "chunking": {
            "max_length": MAX_LENGTH, "split_before_chunk": True,
            "chunks_inherit_parent_partition": True, "truncation": False,
            "tokenizer_revision": ENCODER_REVISION,
        },
        "partition": {"seed": SPLIT_SEED, "dev_documents": DEV_DOCUMENTS},
        "counts": {"parents_spanning_both_partitions": 0, "overflow_count": 0},
        "contamination": {"method": CONTAMINATION_METHOD, "official_test_screened": False},
    }
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            base[section] = {**base[section], field: value}
        else:
            base[section] = value
    return base


def test_a_compliant_manifest_is_accepted():
    from unmark.stage1.manifest import require_compatible

    require_compatible(manifest_dict())


@pytest.mark.parametrize("override, message", [
    ({"source__revision": "deadbeef"}, "not the pinned revision"),
    ({"chunking__max_length": 512}, "max_length"),
    ({"chunking__split_before_chunk": False}, "split_before_chunk"),
    ({"chunking__chunks_inherit_parent_partition": False}, "inherit"),
    ({"chunking__truncation": True}, "does not truncate"),
    ({"chunking__tokenizer_revision": "abc"}, "not the pinned"),
    ({"partition__seed": 1}, "split seed"),
    ({"partition__dev_documents": 1000}, "dev document count"),
    ({"counts__parents_spanning_both_partitions": 3}, "parents spanning"),
    ({"counts__overflow_count": 1}, "zero overflow"),
    ({"contamination__method": "fuzzy"}, "fuzzy or semantic"),
    ({"contamination__official_test_screened": True}, "SEALED"),
    ({"official_test_used": True}, "official_test_used=false"),
    ({"protocol_version": "other"}, "not comparable"),
])
def test_an_off_protocol_manifest_is_refused(override, message):
    from unmark.stage1.manifest import ManifestViolation, require_compatible

    with pytest.raises(ManifestViolation, match=message):
        require_compatible(manifest_dict(**override))


# ---------------------------------------------------------------------------
# The Colab tokenizer micro-probe (Audit 029 §S)
# ---------------------------------------------------------------------------
PROBE = REPO / "scripts" / "stage1_tokenizer_probe.py"


def test_probe_help_exits_zero_without_loading_the_tokenizer():
    """Revision 3a: `--help` previously executed the probe and failed."""
    import subprocess

    result = subprocess.run(
        [__import__("sys").executable, str(PROBE), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "transformers" not in result.stdout.lower() or "--help" in result.stdout


def test_probe_imports_transformers_only_inside_main():
    """`--help` must not touch the tokenizer, so the import cannot be top-level."""
    tree = tree_of(PROBE)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "transformers" not in node.module, "transformers imported at module scope"
        if isinstance(node, ast.Import):
            assert all("transformers" not in a.name for a in node.names)


def test_probe_loads_no_encoder_and_cannot_step():
    source = PROBE.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    calls = {
        getattr(n.func, "attr", None) or getattr(n.func, "id", None)
        for n in ast.walk(tree_of(PROBE)) if isinstance(n, ast.Call)
    }
    for forbidden in ("backward", "step", "AdamW", "zero_grad"):
        assert forbidden not in calls, f"probe calls {forbidden}()"


def test_probe_compares_against_the_authoritative_length_definition():
    """It must check optimized == authoritative, not a guessed composition."""
    source = PROBE.read_text(encoding="utf-8")
    assert "build_inputs_with_special_tokens" in source
    assert "convert_tokens_to_ids" in source
    assert "removed_shortcut_would_have_given" in source, (
        "the diagnostic must show what the falsified shortcut would have produced"
    )
