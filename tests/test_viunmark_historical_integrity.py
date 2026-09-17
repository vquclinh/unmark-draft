"""The ViUnMark public layer is additive: historical files stay byte-identical.

Checked against git, not against a list maintained by hand. Every tracked file
under the historical research paths must be unmodified relative to the commit
this layer was built on. New files are allowed; modifications, deletions,
renames and type changes are not. Torch-free.
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "93227e134b31b850763d9c1dd33ccc9abf3c9ae7"

HISTORICAL_PATHS = ("unmark", "scripts", "configs", "docs/spec", "docs/audits")
NEW_LAYER_EXCLUSIONS = (
    ":(exclude)unmark/viunmark",
    ":(exclude)docs/spec/viunmark-final-system-v1.json",
    ":(exclude)docs/spec/viunmark-historical-aliases-v1.json",
)

AUDIT_072 = "docs/audits/072-sa-vlsp2016-external-evaluation-repository-reconnaissance.md"
AUDIT_072_SHA256 = "6f0266d7bcd74602ba81d11292be416e3dcd165b701658d264070a5ce8fc7281"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def require_baseline() -> None:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    if git("cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}").returncode != 0:
        pytest.skip(f"baseline commit {BASELINE_COMMIT} is not available (shallow clone?)")


def test_no_historical_file_is_modified_deleted_or_renamed():
    require_baseline()
    result = git(
        "diff", "--name-status", "--diff-filter=DMRT", BASELINE_COMMIT, "--",
        *HISTORICAL_PATHS, *NEW_LAYER_EXCLUSIONS,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        "historical research files changed relative to the ViUnMark baseline:\n" + result.stdout
    )


def test_frozen_protocol_specs_are_byte_identical_to_the_baseline():
    require_baseline()
    for name in (
        "docs/spec/stage1-final-freeze.json",
        "docs/spec/stage1-adapter-finalists.json",
        "docs/spec/stage2-dual-finalist-protocol.json",
        "docs/spec/stage2-v2-scf-posthoc-protocol.json",
        "docs/spec/restore-baseline-protocol.json",
        "docs/spec/decisions.md",
    ):
        committed = git("show", f"{BASELINE_COMMIT}:{name}")
        assert committed.returncode == 0, name
        assert (REPO / name).read_text(encoding="utf-8") == committed.stdout, name


def test_audit_072_is_preserved_exactly():
    path = REPO / AUDIT_072
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == AUDIT_072_SHA256


def test_the_accepted_adapter_implementation_is_reused_not_copied():
    """The public layer builds adapters through Stage-I dispatch, never by hand."""
    system = (REPO / "unmark/viunmark/system.py").read_text(encoding="utf-8")
    assert "reconstruct_adapter" in system and "fresh_adapter" in system
    for path in (REPO / "unmark/viunmark").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "OrthographyInputAdapter(" not in text, path
        assert "def scale_calibrated_fusion" not in text, path
        assert ".norm(" not in text, path
