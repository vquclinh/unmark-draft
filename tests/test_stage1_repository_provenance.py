"""Stage-1 repository provenance is DERIVED, never supplied. **Torch-free.**

Audit 031 B5 / Audit 032 MAJ2. Stage-6 always derived its HEAD from Git, and
`resolve_repository_head`'s own docstring said there was "deliberately no CLI
flag and no environment override", because "a caller-claimed HEAD would let a
checkpoint written by commit A resume under commit B while asserting it did
not".

Stage-1 training did exactly that. `--repository-head` defaulted to `None` and
whatever arrived was recorded, so a run could either:

* record **no** repository identity at all -- and because
  `RunProvenance.require_match` then compared `None` against `None` and passed,
  the resume-blocking HEAD gate introduced in Audit 030 SS V was vacuous; or
* record a commit it was not running, undetectably.

No test here runs a Git *write*. Every case is either a read-only query against
the real repository or a canned `git` result injected with monkeypatch, so the
working tree is never modified, nothing is staged, and no repository is created.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1 import checkpoint as checkpoint_module  # noqa: E402
from unmark.stage1.checkpoint import (  # noqa: E402
    EXECUTION_RELEVANT_PATHS,
    CheckpointViolation,
    repository_execution_modifications,
    require_clean_execution_tree,
    resolve_asserted_repository_head,
    resolve_repository_head,
)
from unmark.stage1.trainer import RunProvenance  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def actual_head() -> str:
    """Read-only. The authority this module is asserting against."""
    completed = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def clean_tree(monkeypatch):
    """Report a clean working tree without touching the real one."""
    monkeypatch.setattr(
        checkpoint_module, "repository_execution_modifications", lambda root=None: ()
    )


# ---------------------------------------------------------------------------
# The head is derived
# ---------------------------------------------------------------------------
def test_the_derived_head_is_the_real_full_sha():
    head = resolve_repository_head(REPO)
    assert FULL_SHA.match(head), head
    assert head == actual_head()


def test_an_omitted_assertion_yields_the_actual_head_not_none(clean_tree):
    """The defect: omitting the flag used to record `None`."""
    head = resolve_asserted_repository_head(None, root=REPO)
    assert head == actual_head()
    assert head is not None


def test_a_correct_assertion_is_accepted(clean_tree):
    head = actual_head()
    assert resolve_asserted_repository_head(head, root=REPO) == head
    assert resolve_asserted_repository_head(head.upper(), root=REPO) == head


def test_a_false_assertion_is_refused(clean_tree):
    with pytest.raises(CheckpointViolation, match="assertion, not an override"):
        resolve_asserted_repository_head("b" * 40, root=REPO)


def test_an_abbreviated_assertion_is_refused(clean_tree):
    with pytest.raises(CheckpointViolation, match="assertion, not an override"):
        resolve_asserted_repository_head(actual_head()[:12], root=REPO)


def test_git_resolution_failure_fails_closed(tmp_path):
    """A directory that is not a repository cannot answer, so nothing runs."""
    with pytest.raises(CheckpointViolation, match="cannot resolve the repository HEAD"):
        resolve_repository_head(tmp_path)
    with pytest.raises(CheckpointViolation):
        resolve_asserted_repository_head(None, root=tmp_path)


def test_a_missing_git_binary_fails_closed(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("git: command not found")

    monkeypatch.setattr(subprocess, "run", explode)
    with pytest.raises(CheckpointViolation, match="cannot resolve the repository HEAD"):
        resolve_repository_head(REPO)


def test_a_branch_name_is_not_an_identity(monkeypatch):
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="main\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(CheckpointViolation, match="not a full 40-character"):
        resolve_repository_head(REPO)


# ---------------------------------------------------------------------------
# The clean-tree rule
# ---------------------------------------------------------------------------
def _porcelain(monkeypatch, output: str):
    """Inject a canned `git status --porcelain`, leaving `rev-parse` real.

    Only the status query is faked, so the HEAD these tests see is still the
    repository's actual HEAD -- the point being that a dirty tree is refused
    even though the head resolves perfectly well.
    """
    real_head = actual_head()

    def fake_run(cmd, *args, **kwargs):
        if "status" in cmd:
            assert "--porcelain" in cmd, cmd
            return types.SimpleNamespace(returncode=0, stdout=output, stderr="")
        assert "rev-parse" in cmd, cmd
        return types.SimpleNamespace(returncode=0, stdout=real_head + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_untracked_and_ignored_files_do_not_make_a_tree_dirty(monkeypatch):
    """.venv, caches, prepared corpora and run outputs must not block a run."""
    _porcelain(monkeypatch, "\n".join([
        "?? results/lr-pilot/",
        "?? docs/audits/033-consolidated-material-repair.md",
        "!! .venv/",
        "?? prepared-corpus/chunks.jsonl",
        "",
    ]))
    assert repository_execution_modifications(REPO) == ()
    require_clean_execution_tree(REPO)


@pytest.mark.parametrize("line", [
    " M unmark/stage1/trainer.py",
    "M  unmark/stage1/execute.py",
    "A  scripts/stage1_runner.py",
    " D unmark/stage1/selection.py",
    "R  configs/a.yaml -> configs/b.yaml",
    "UU unmark/stage1/checkpoint.py",
])
def test_modified_tracked_execution_code_fails_closed(monkeypatch, line):
    _porcelain(monkeypatch, line + "\n")
    assert repository_execution_modifications(REPO) == (line,)
    with pytest.raises(CheckpointViolation, match="tracked execution code is modified"):
        require_clean_execution_tree(REPO)


def test_a_dirty_tree_blocks_the_asserted_head(monkeypatch):
    _porcelain(monkeypatch, " M unmark/stage1/trainer.py\n")
    # The head still resolves; it is the *claim* that is refused.
    with pytest.raises(CheckpointViolation, match="tracked execution code is modified"):
        resolve_asserted_repository_head(None, root=REPO)


def test_the_clean_check_only_asks_about_execution_relevant_paths(monkeypatch):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    repository_execution_modifications(REPO)
    for path in EXECUTION_RELEVANT_PATHS:
        assert path in seen["cmd"], f"{path} must be inspected"
    assert "docs" not in seen["cmd"], "documentation is not execution-relevant"
    assert "tests" not in seen["cmd"], "a modified test does not change the training math"


def test_a_git_failure_during_the_clean_check_fails_closed(monkeypatch):
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=128, stdout="", stderr="fatal: not a repo")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(CheckpointViolation, match="cannot inspect the working tree"):
        repository_execution_modifications(REPO)


def test_the_real_repository_reports_only_tracked_execution_lines():
    """Integration, read-only, and stable whatever the tree currently holds."""
    for line in repository_execution_modifications(REPO):
        assert line[:2] not in ("??", "!!"), line
        assert any(p in line for p in EXECUTION_RELEVANT_PATHS), line


def test_smoke_may_run_on_a_dirty_tree_but_not_claim_a_false_head(monkeypatch):
    """Smoke is a no-update diagnostic used *while* code is being changed.

    It produces no scientific artifact, so the clean-tree rule is relaxed for
    it -- but it still cannot record a commit it is not running.
    """
    expected = actual_head()
    _porcelain(monkeypatch, " M unmark/stage1/trainer.py\n")
    assert resolve_asserted_repository_head(
        None, root=REPO, require_clean=False
    ) == expected
    with pytest.raises(CheckpointViolation, match="assertion, not an override"):
        resolve_asserted_repository_head("c" * 40, root=REPO, require_clean=False)


# ---------------------------------------------------------------------------
# The runner wires the derived head into the real execution call
# ---------------------------------------------------------------------------
def test_the_runner_passes_the_derived_head_not_the_flag(tmp_path, monkeypatch, clean_tree):
    """The REAL `_execute`, with only `execute_stage` itself stubbed."""
    import unmark.stage1.execute as execute_module
    from scripts.stage1_runner import _execute

    captured = {}

    def fake_execute_stage(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(execute_module, "execute_stage", fake_execute_stage)

    args = types.SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        prepared_corpus=str(tmp_path / "prepared"),
        cache_root=str(tmp_path / "cache"),
        revision="01daacda68afe13d83023d16ec647239e344a1e6",
        repository_head=None,
        resume=False,
    )
    verified = types.SimpleNamespace(chunk_membership_digest="d" * 64, manifest={})

    assert _execute(args, [], "lr_pilot", verified) == 0
    assert captured["repository_head"] == actual_head(), (
        "the runner must record the derived HEAD, never the flag"
    )


def test_the_runner_refuses_a_false_asserted_head(tmp_path, monkeypatch, clean_tree):
    import unmark.stage1.execute as execute_module
    from scripts.stage1_runner import _execute

    monkeypatch.setattr(execute_module, "execute_stage", lambda **k: 0)
    args = types.SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        prepared_corpus=str(tmp_path / "prepared"),
        cache_root=str(tmp_path / "cache"),
        revision="01daacda68afe13d83023d16ec647239e344a1e6",
        repository_head="b" * 40,
        resume=False,
    )
    verified = types.SimpleNamespace(chunk_membership_digest="d" * 64, manifest={})
    with pytest.raises(CheckpointViolation, match="assertion, not an override"):
        _execute(args, [], "lr_pilot", verified)


# ---------------------------------------------------------------------------
# The consequence for resume matching
# ---------------------------------------------------------------------------
def test_the_head_gate_is_no_longer_vacuous():
    """`None == None` used to pass, so the SS V resume gate did not gate."""
    head = actual_head()
    mine = RunProvenance(
        run_seed=1, init_seed=2, corruption_seed=3, learning_rate=1e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head=head,
    )
    foreign = RunProvenance(
        run_seed=1, init_seed=2, corruption_seed=3, learning_rate=1e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head="b" * 40,
    )
    mine.require_match(mine.to_dict())
    with pytest.raises(Exception, match="repository_head"):
        mine.require_match(foreign.to_dict())


def test_the_repository_head_flag_is_documented_as_an_assertion():
    from scripts.stage1_runner import build_parser

    parser = build_parser()
    text = parser.format_help()
    assert "--repository-head" not in text or True  # subparser help lives below
    action = None
    for sub in parser._subparsers._group_actions:  # noqa: SLF001
        for name, subparser in sub.choices.items():
            for candidate in subparser._actions:  # noqa: SLF001
                if candidate.dest == "repository_head" and candidate.help:
                    action = candidate
    assert action is not None, "the flag must still exist for operator assertions"
    assert "assertion" in action.help.lower(), action.help


# ---------------------------------------------------------------------------
# The structural backstop inside execute_stage itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, "", "abc", "a" * 39, "a" * 41, 123, "z" * 40])
def test_execute_stage_refuses_a_head_that_is_not_a_commit(bad, tmp_path):
    """Reached before any lazy import, so it runs in the ML-free venv too.

    The runner is the authority, but a second caller could always be added.
    A `None` head is what made the SS V resume gate vacuous, so `execute_stage`
    refuses one itself rather than trusting every future call site.
    """
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.execute import execute_stage

    with pytest.raises(Stage1ContractViolation, match="40-character commit sha"):
        execute_stage(
            stage="lr_pilot", schedule=[], prepared_corpus=tmp_path,
            verified=None, output_dir=tmp_path, cache_root=tmp_path,
            revision="01daacda68afe13d83023d16ec647239e344a1e6",
            repository_head=bad,
        )


def test_execute_stage_accepts_a_real_head_and_proceeds_past_the_guard(tmp_path):
    """A well-formed head passes the guard and the function moves on.

    It then fails on something else entirely (torch, or the absent corpus),
    which is the proof that the guard itself did not reject it.
    """
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.execute import execute_stage

    try:
        execute_stage(
            stage="lr_pilot", schedule=[], prepared_corpus=tmp_path,
            verified=None, output_dir=tmp_path, cache_root=tmp_path,
            revision="01daacda68afe13d83023d16ec647239e344a1e6",
            repository_head=actual_head(),
        )
    except Stage1ContractViolation as error:
        assert "40-character commit sha" not in str(error), error
    except Exception:
        pass  # torch/transformers/corpus absent -- past the guard, which is the point
