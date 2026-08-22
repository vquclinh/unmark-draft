"""Cross-module import contract for the pre-G1 runtime.

**Why this file exists.** Audit 027 shipped a `sensitivity` subcommand whose
modules import `SECONDARY_ANALYSIS_LABEL` / `PRIMARY_ANALYSIS_LABEL` from
`preg1_protocol`. Every local test passed — because the local suite imports the
**working tree**, where those constants were defined. The GPU environment does
not run the working tree: it runs `git clone`, i.e. the **committed** tree. The
commit that added the importers left the file that defines them behind, so
Colab died with `ImportError` before a single scientific run.

No test that only exercises the working tree can catch that. The test below
therefore checks the committed tree's **internal** consistency: for one and the
same commit, every protocol symbol its importers name must be defined in its
protocol module. It never compares HEAD against the working tree, so the
project's normal "leave everything unstaged" workflow does not trip it — it
fails only when a commit is itself inconsistent, which is exactly the defect.

ML-free: nothing here imports torch, loads a model or reads a corpus.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "unmark/evaluation/preg1_protocol.py"

# Every module that names symbols from the protocol. Discovered once and pinned
# here so that adding a new importer without extending this list is visible.
IMPORTERS = (
    "unmark/evaluation/preg1_head.py",
    "unmark/evaluation/preg1_split.py",
    "unmark/evaluation/__init__.py",
    "scripts/preg1_head_diagnostic.py",
    "scripts/preg1_dataset_profile.py",
    "scripts/materialize_preg1_split.py",
)


def protocol_imports(source: str) -> set[str]:
    """Names this source pulls out of `preg1_protocol`, at any import depth."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[-1] == "preg1_protocol":
                names |= {alias.name for alias in node.names}
    return names


def module_level_names(source: str) -> set[str]:
    """Names `preg1_protocol` actually binds at module scope."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
    return names


def git_blob(path: str, rev: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO, capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"{path} is not present at {rev}: {result.stderr.strip()}")
    return result.stdout


def test_the_importer_list_is_complete():
    """A new importer must be added above, or the contract silently narrows."""
    found = {
        str(path.relative_to(REPO))
        for directory in ("unmark", "scripts")
        for path in (REPO / directory).rglob("*.py")
        if protocol_imports(path.read_text(encoding="utf-8"))
    }
    assert found == set(IMPORTERS), (
        f"importer list is stale; missing {sorted(found - set(IMPORTERS))}, "
        f"stale {sorted(set(IMPORTERS) - found)}"
    )


@pytest.mark.parametrize("importer", IMPORTERS)
def test_working_tree_resolves_every_protocol_symbol(importer):
    defined = module_level_names((REPO / PROTOCOL).read_text(encoding="utf-8"))
    wanted = protocol_imports((REPO / importer).read_text(encoding="utf-8"))
    assert wanted <= defined, (
        f"{importer} imports {sorted(wanted - defined)} from the protocol, "
        "which does not define them"
    )


@pytest.mark.parametrize("importer", IMPORTERS)
def test_committed_tree_resolves_every_protocol_symbol(importer):
    """THE regression test for the Audit 027 runtime blocker.

    Colab runs the committed tree, so the committed tree must be importable on
    its own terms. Both sides are read from the SAME commit: this asserts that
    commit's internal consistency, not that the working tree has been committed.
    """
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    defined = module_level_names(git_blob(PROTOCOL))
    wanted = protocol_imports(git_blob(importer))
    missing = sorted(wanted - defined)
    assert not missing, (
        f"COMMITTED {importer} imports {missing} from {PROTOCOL}, but the "
        f"COMMITTED {PROTOCOL} does not define them. A `git clone` of this "
        "commit raises ImportError before any scientific run. The file that "
        "defines these symbols was omitted from the commit — commit it."
    )


def test_the_secondary_label_is_single_sourced_in_the_protocol():
    """The repair is option 1: define it once, in the protocol. Not inlined."""
    from unmark.evaluation.preg1_protocol import (
        PRIMARY_ANALYSIS_LABEL,
        SECONDARY_ANALYSIS_LABEL,
    )

    assert SECONDARY_ANALYSIS_LABEL == "SECONDARY OWN-LR SENSITIVITY"
    assert PRIMARY_ANALYSIS_LABEL == "PRIMARY SHARED-LR"

    # No consumer may hard-code the string instead of importing the constant.
    for importer in ("unmark/evaluation/preg1_head.py",
                     "scripts/preg1_head_diagnostic.py"):
        source = (REPO / importer).read_text(encoding="utf-8")
        assert f'"{SECONDARY_ANALYSIS_LABEL}"' not in source, (
            f"{importer} hard-codes the label; it must import the constant"
        )

    # And exactly one definition exists repository-wide.
    definitions = [
        path for directory in ("unmark", "scripts")
        for path in (REPO / directory).rglob("*.py")
        if f'SECONDARY_ANALYSIS_LABEL = "' in path.read_text(encoding="utf-8")
    ]
    assert len(definitions) == 1, f"expected one definition, found {definitions}"
