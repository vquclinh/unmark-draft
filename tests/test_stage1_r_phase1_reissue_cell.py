"""Fail-closed contract of the resource-bounded r-phase1 Colab reissue cell.

The cell is a script that mounts Drive and rewrites a scientific handoff
artifact, so it cannot be imported here. These tests therefore read its source:
the commit guard is extracted and executed in isolation, and the identity
direction is checked structurally with the AST rather than by substring, because
a grep would match the cell's own explanatory comments.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.protocol import ENCODER_CHECKPOINT, ENCODER_REVISION  # noqa: E402


CELL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "colab"
    / "regenerate_r_phase1_resource_bounded_author_override_cell.py"
)

PLACEHOLDER = "REPLACE_WITH_FULL_40_HEX_COMMIT_SHA"
SOURCE_R_PHASE1_HEAD = "3bb2944e6f71865d5a37fe403b78ea640f8a3f1d"


def source() -> str:
    return CELL.read_text(encoding="utf-8")


def tree() -> ast.Module:
    return ast.parse(source())


def assignments(name: str) -> list[ast.expr]:
    """Every top-level value assigned to `name`."""
    return [
        node.value
        for node in tree().body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == name
    ]


def guard():
    """Extract `require_immutable_commit` and run it against a raising `fail`."""
    function = next(
        node
        for node in tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "require_immutable_commit"
    )

    def fail(msg):
        raise RuntimeError(str(msg))

    namespace: dict = {"re": re, "fail": fail}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CELL), "exec"), namespace)
    return namespace["require_immutable_commit"]


# ==========================================================================================
# Finding 3 - no mutable ref may reach the checkout.
# ==========================================================================================

def test_the_cell_exists_and_parses():
    assert CELL.is_file(), CELL
    assert tree().body


@pytest.mark.parametrize(
    "rejected",
    [
        "origin/main",
        "main",
        "master",
        "HEAD",
        "v1.0.0",
        "refs/heads/main",
        "49f8c68",
        "49f8c68afea98f8440fdf330cdcd5629e96071e",       # 39 hex
        "49f8c68afea98f8440fdf330cdcd5629e96071eff",     # 41 hex
        "49f8c68afea98f8440fdf330cdcd5629e96071eg",      # non-hex
        "",
        "   ",
        PLACEHOLDER,
        None,
    ],
)
def test_the_commit_guard_rejects_everything_that_can_move(rejected):
    with pytest.raises(RuntimeError):
        guard()(rejected)


def test_the_commit_guard_accepts_a_full_40_hex_sha():
    sha = "49f8c68afea98f8440fdf330cdcd5629e96071ef"
    assert guard()(sha) == sha
    assert guard()(sha.upper()) == sha
    assert guard()(f"  {sha}  ") == sha


def test_the_cell_ships_with_an_unresolved_placeholder():
    """The committed cell must never carry a runnable default."""
    values = assignments("IMPLEMENTATION_COMMIT")
    assert values, "IMPLEMENTATION_COMMIT is not assigned at top level"
    literal = values[0]
    assert isinstance(literal, ast.Constant) and literal.value == PLACEHOLDER


def test_the_environment_override_has_no_default():
    """`os.environ.get(...)` may only fall back to the placeholder, never to a ref."""
    getters = [
        value
        for value in assignments("IMPLEMENTATION_COMMIT")
        if isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
    ]
    assert getters, "no environment override found"
    for call in getters:
        assert len(call.args) == 2, "environ.get must fall back to the placeholder"
        fallback = call.args[1]
        # Either the placeholder literal or the name still holding it.
        assert isinstance(fallback, (ast.Name, ast.Constant))
        if isinstance(fallback, ast.Constant):
            assert fallback.value == PLACEHOLDER


def git_commands() -> list[list[ast.expr]]:
    """Every `run(...)`/`capture(...)` argument list whose first element is "git"."""
    found = []
    for node in ast.walk(tree()):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"run", "capture", "check_output"} or not node.args:
            continue
        argv = node.args[0]
        if not isinstance(argv, ast.List) or not argv.elts:
            continue
        head = argv.elts[0]
        if isinstance(head, ast.Constant) and head.value == "git":
            found.append(argv.elts)
    return found


def test_no_mutable_ref_reaches_fetch_or_checkout():
    """`git fetch`/`git checkout` may only name the verified immutable commit."""
    assert "REPO_REF" not in source()
    seen = 0
    for argv in git_commands():
        words = [e.value for e in argv if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if not ({"fetch", "checkout"} & set(words)):
            continue
        seen += 1
        for element in argv[1:]:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                assert element.value not in {
                    "origin/main", "main", "master", "HEAD", "origin/HEAD", "FETCH_HEAD",
                }, f"{element.value!r} reaches a fetch/checkout"
            elif isinstance(element, ast.Name):
                assert element.id == "IMPLEMENTATION_COMMIT", (
                    f"fetch/checkout target is {element.id!r}, not the verified commit"
                )
    assert seen >= 2, "expected both a fetch and a checkout"


def test_the_checkout_is_verified_against_the_requested_commit():
    body = source()
    assert "if HEAD != IMPLEMENTATION_COMMIT:" in body


# ==========================================================================================
# Finding 4 - identity is established from evidence, not from the artifact.
# ==========================================================================================

def test_identity_is_built_from_verified_inputs_not_from_the_artifact():
    calls = [
        node
        for node in ast.walk(tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_inputs"
    ]
    assert calls, "the cell must build identity with CampaignIdentity.from_inputs"

    keywords = {kw.arg for call in calls for kw in call.keywords}
    assert {"repository_head", "corpus_manifest_digest", "encoder_revision", "inventory"} <= keywords


def test_identity_is_never_reconstructed_from_a_loaded_artifact():
    """`CampaignIdentity(**artifact["identity"])` is the circular pattern."""
    for node in ast.walk(tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "CampaignIdentity":
            for kw in node.keywords:
                if kw.arg is None:  # **something
                    pytest.fail(
                        "identity is splatted from a loaded artifact; it must come "
                        "from independently verified runtime inputs"
                    )


def test_the_canonical_corpus_verifier_is_used():
    imported = set()
    for node in ast.walk(tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert "unmark.stage1.checkpoint.verify_prepared_corpus" in imported, (
        "the cell must reuse the repository's own prepared-corpus verifier"
    )


def test_the_expected_scientific_identity_is_pinned():
    body = source()
    for expected in (
        "aa49785eadcb",                                                        # corpus key
        "2198412593",                                                          # chunks bytes
        "5e4c5e0c77e7677e188501723651e0923d072a31a9048a7d04042ff7b290cad6",    # chunks sha
        "2878",                                                                # manifest bytes
        "6f33c2aa51b63a4dc68e238594acbec581b2a1f6b0f7be42e002dfb10a02ef62",    # manifest sha
        "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6",    # membership
        "135a4d9716e49a981624474156d6f247b9b46f6a",                            # inventory rev
        "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",    # inventory sha
        "116290",                                                              # inventory size
        ENCODER_CHECKPOINT,
        ENCODER_REVISION,
        "stage1-protocol-v1",
    ):
        assert expected in body, f"cell does not pin {expected!r}"


# ==========================================================================================
# Preserved Audit 047 guarantees.
# ==========================================================================================

def test_the_cell_only_touches_the_backup_campaign_root():
    body = source()
    assert "/content/drive/MyDrive/UNMARK/UNMARK-BACKUP" in body
    assert "UNMARK-REAL" not in body


def test_the_source_execution_head_stays_distinct_from_the_reissue_head():
    body = source()
    assert SOURCE_R_PHASE1_HEAD in body
    assert "source_r_phase1_repository_head=SOURCE_R_PHASE1_HEAD" in body
    assert "reissued_under_repository_head=HEAD" in body


def test_the_previous_artifact_is_backed_up_before_replacement():
    body = source()
    assert "shutil.copy2(R_PHASE1_ARTIFACT, backup)" in body
    assert "before-resource-bounded-r-override" in body


def test_the_cell_never_deletes_scientific_evidence():
    """No rmtree/unlink/remove anywhere: checkpoints, telemetry and W&B state stay."""
    forbidden = {"rmtree", "unlink", "remove", "rmdir"}
    for node in ast.walk(tree()):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in forbidden:
                pytest.fail(f"cell calls {name}(); it must not delete evidence")
    assert "wandb_run_ids.json" not in source()


def test_the_cell_cannot_launch_final_main():
    """It reports suggested paths and stops; nothing may execute a training stage."""
    body = source()
    assert "This cell did not start final-main." in body
    for node in ast.walk(tree()):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value
        if "stage1_runner.py" in text and "final-main" in text:
            pytest.fail("cell embeds a final-main runner invocation")
    assert "final-main" not in _executed_command_strings()


def _executed_command_strings() -> str:
    """Concatenate every string literal that reaches run()/capture()/subprocess."""
    parts: list[str] = []
    for node in ast.walk(tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in {"run", "capture", "check_output", "Popen"}:
            continue
        for argument in ast.walk(node):
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                parts.append(argument.value)
    return "\n".join(parts)


def test_the_pinned_decision_is_recorded_in_the_cell():
    body = source()
    assert "selected_r=1.0" in body
    assert "fail(f\"LR handoff winner is {lr_winner.learning_rate!r}, not 0.0001\")" in body
    assert "0.0003" not in body, "the historical locked-rule LR must not be reintroduced"


# ==========================================================================================
# The LR helper is exec()'d by the r cell, so it must obey the same commit rule.
# A branch checkout there would move the repository off the pinned commit mid-run.
# ==========================================================================================

LR_CELL = CELL.with_name("regenerate_lr_pilot_author_override_cell.py")


def lr_source() -> str:
    return LR_CELL.read_text(encoding="utf-8")


def lr_tree() -> ast.Module:
    return ast.parse(lr_source())


def test_the_lr_helper_also_refuses_a_mutable_ref():
    body = lr_source()
    assert "REPO_REF" not in body
    assert "origin/main" not in body.replace(
        "``origin/main``", ""  # the guard's own docstring may name what it rejects
    )
    assert 'run(["git", "fetch", "--quiet", "origin", IMPLEMENTATION_COMMIT], cwd=REPO)' in body
    assert "if current_head != IMPLEMENTATION_COMMIT:" in body


def test_the_lr_helper_accepts_the_commit_injected_by_the_r_cell():
    """The r cell pins and verifies once; the LR helper must reuse that value."""
    assert "INJECTED_IMPLEMENTATION_COMMIT" in lr_source()
    assert '"INJECTED_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,' in source()


def test_the_lr_helper_ships_with_an_unresolved_placeholder():
    assert PLACEHOLDER in lr_source()


def test_the_lr_helper_still_selects_0001():
    body = lr_source()
    assert "selected_learning_rate" in body or "lr=0.0001" in body
    assert "author_lr_override_after_validation_curve_review" in body
