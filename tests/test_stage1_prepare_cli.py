"""End-to-end `prepare-corpus` through the REAL parser. ML-free.

**Why this file exists.** Revision 3c crashed on the real corpus with

    AttributeError: 'Namespace' object has no attribute 'repository_head'

immediately after `[6/6] deterministic pre-chunking`. Every existing Stage-1
runner test was either AST-only or built `CheckpointIdentity` objects directly,
so nothing ever parsed a real `prepare-corpus` command line and entered
`run_prepare_corpus`. A test that constructs `argparse.Namespace(...)` by hand
would have reproduced that same blind spot -- it supplies the very attribute
whose absence was the bug.

These tests therefore go through `build_parser()`, the same path `main()` uses,
and drive the whole Stage-6 pipeline with only the expensive I/O injected:
the corpus pin, the shard reader and the tokenizer. Chunking, checkpointing,
finalisation and the completion marker all run for real.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "stage1_runner.py"

from unmark.stage1.checkpoint import (  # noqa: E402
    COMPLETE_NAME,
    STATE_NAME,
    CheckpointViolation,
    resolve_repository_head,
)
from unmark.stage1.corpus import CorpusDocument, CorpusPin, ShardPin  # noqa: E402
from unmark.stage1.manifest import CHUNKS_NAME, MANIFEST_NAME  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    CORPUS_DATASET,
    CORPUS_REVISION,
    CORPUS_SHARD_ORDER,
)

PHOBERT_RUN = re.compile(r"\S+\n?")


def load_cli():
    spec = importlib.util.spec_from_file_location("stage1_cli_e2e", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubTokenizer:
    """Whitespace/BPE-shaped stand-in. Never downloads anything."""

    all_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

    def get_added_vocab(self):
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    def bpe(self, token):
        return " ".join(token[i:i + 4] for i in range(0, len(token), 4)) or token

    def tokenize(self, text):
        out = []
        for run in PHOBERT_RUN.findall(text):
            out.extend(self.bpe(run).split(" "))
        return out

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def build_inputs_with_special_tokens(self, ids):
        return ["<s>", *ids, "</s>"]


WORDS = "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương".split()


def documents(n=5_200):
    """Must exceed the locked `DEV_DOCUMENTS = 5000` -- that constant is
    scientific and is not adjusted for a test. Documents are kept short so the
    end-to-end run stays fast with the stub tokenizer."""
    import random

    rng = random.Random(51733)
    return [
        CorpusDocument(
            f"doc-{i:05d}",
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 24))),
            "train.parquet", i,
        )
        for i in range(n)
    ]


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """The CLI with only pin/reader/tokenizer injected."""
    cli = load_cli()
    docs = documents()
    per_shard = {
        "train.parquet": docs[:5_000],
        "validation.parquet": docs[5_000:5_100],
        "test.parquet": docs[5_100:],
    }
    pin = CorpusPin(
        dataset=CORPUS_DATASET, revision=CORPUS_REVISION,
        files=tuple(ShardPin(name, 10, f"{i}" * 64)
                    for i, name in enumerate(CORPUS_SHARD_ORDER)),
        concatenation_order=CORPUS_SHARD_ORDER,
        schema_version="stage1-corpus-pin-v1",
    )
    monkeypatch.setattr(cli, "load_pin", lambda *a, **k: pin)
    monkeypatch.setattr(cli, "verify_corpus_root", lambda root, p=None: {
        "dataset": CORPUS_DATASET, "revision": CORPUS_REVISION,
        "concatenation_order": list(CORPUS_SHARD_ORDER),
        "files": [{"name": f.name, "bytes": f.bytes, "sha256": f.sha256} for f in pin.files],
        "shard_labels_are_a_split": False,
    })
    monkeypatch.setattr(cli, "read_shard", lambda path, name: per_shard[name])
    monkeypatch.setattr(cli, "_load_tokenizer", lambda revision: StubTokenizer())

    # Operational only: a small interval so a commit actually happens within the
    # fixture. The interval changes no output (Audit 029 §U.3).
    real_checkpoint = cli.PrepareCheckpoint
    monkeypatch.setattr(
        cli, "PrepareCheckpoint",
        lambda root, identity, total, **kw: real_checkpoint(
            root, identity, total, interval=kw.pop("interval", 500), **kw
        ),
    )
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    return cli, corpus_root, docs


def argv(corpus_root, output, checkpoint=None):
    args = ["prepare-corpus", "--corpus-root", str(corpus_root),
            "--output-dir", str(output)]
    if checkpoint:
        args += ["--checkpoint-dir", str(checkpoint)]
    return args


def run(cli, arguments):
    """Parse with the REAL parser, then dispatch exactly as `main` does."""
    parsed = cli.build_parser().parse_args(arguments)
    return parsed, cli.run_prepare_corpus(parsed)


# ---------------------------------------------------------------------------
# The regression the real probe found
# ---------------------------------------------------------------------------
def test_parsed_prepare_corpus_namespace_has_no_repository_head(wired, tmp_path):
    """The exact shape of the crash: the flag does not exist on this command."""
    cli, corpus_root, _ = wired
    parsed = cli.build_parser().parse_args(argv(corpus_root, tmp_path / "out"))
    assert not hasattr(parsed, "repository_head"), (
        "prepare-corpus must NOT gain a --repository-head flag; the HEAD is "
        "derived from the repository"
    )
    assert parsed.command == "prepare-corpus"


def test_start_path_runs_end_to_end_without_attribute_error(wired, tmp_path):
    """This is the test that would have caught the real crash."""
    cli, corpus_root, docs = wired
    output = tmp_path / "out"
    parsed, status = run(cli, argv(corpus_root, output))
    assert status == 0
    assert (output / CHUNKS_NAME).is_file() and (output / MANIFEST_NAME).is_file()

    seen = []
    with open(output / CHUNKS_NAME, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not seen or seen[-1] != record["document_id"]:
                seen.append(record["document_id"])
    assert seen == [d.document_id for d in docs]


def test_the_checkpoint_records_the_real_repository_head(wired, tmp_path):
    cli, corpus_root, _ = wired
    output = tmp_path / "out"
    run(cli, argv(corpus_root, output))
    marker = json.loads(
        (output / "_checkpoint" / COMPLETE_NAME).read_text(encoding="utf-8")
    )
    head = marker["identity"]["repository_head"]
    assert head == resolve_repository_head()
    assert re.fullmatch(r"[0-9a-f]{40}", head), "must be a full 40-char sha"


def test_already_complete_short_circuits_and_still_validates_identity(wired, tmp_path):
    cli, corpus_root, _ = wired
    output = tmp_path / "out"
    run(cli, argv(corpus_root, output))
    first = (output / CHUNKS_NAME).read_bytes()

    parsed, status = run(cli, argv(corpus_root, output))     # second invocation
    assert status == 0
    assert (output / CHUNKS_NAME).read_bytes() == first, "must not recompute or alter"


def test_resume_path_runs_without_attribute_error(wired, tmp_path, monkeypatch):
    """Kill Stage 6 mid-corpus, then resume through the real parser again."""
    cli, corpus_root, docs = wired
    output = tmp_path / "out"

    real_chunk = cli.chunk_document
    calls = {"n": 0}

    def dying(document, partition, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1_700:
            raise KeyboardInterrupt("simulated runtime death")
        return real_chunk(document, partition, **kwargs)

    monkeypatch.setattr(cli, "chunk_document", dying)
    with pytest.raises(KeyboardInterrupt):
        run(cli, argv(corpus_root, output))
    assert (output / "_checkpoint" / STATE_NAME).is_file()

    monkeypatch.setattr(cli, "chunk_document", real_chunk)
    parsed, status = run(cli, argv(corpus_root, output))
    assert status == 0
    assert not hasattr(parsed, "repository_head")

    seen = []
    with open(output / CHUNKS_NAME, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not seen or seen[-1] != record["document_id"]:
                seen.append(record["document_id"])
    assert seen == [d.document_id for d in docs], "resume must not skip or repeat"


def test_a_checkpoint_from_another_head_cannot_resume(wired, tmp_path, monkeypatch):
    """HEAD stays a fail-closed identity field."""
    cli, corpus_root, _ = wired
    output = tmp_path / "out"

    real_chunk = cli.chunk_document
    calls = {"n": 0}

    def dying(document, partition, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1_700:
            raise KeyboardInterrupt
        return real_chunk(document, partition, **kwargs)

    monkeypatch.setattr(cli, "chunk_document", dying)
    with pytest.raises(KeyboardInterrupt):
        run(cli, argv(corpus_root, output))
    monkeypatch.setattr(cli, "chunk_document", real_chunk)

    monkeypatch.setattr(cli, "resolve_repository_head", lambda *a, **k: "b" * 40)
    with pytest.raises(CheckpointViolation, match="identity mismatch"):
        run(cli, argv(corpus_root, output))


def test_pre_checkpoint_failure_leaves_nothing_that_looks_like_progress(
    wired, tmp_path, monkeypatch
):
    """The real f9c23fe crash: died at next_document_index=0, before any commit."""
    cli, corpus_root, _ = wired
    output = tmp_path / "out"

    def explode(*a, **k):
        raise AttributeError("simulating the f9c23fe wiring crash")

    real_identity = cli.CheckpointIdentity
    monkeypatch.setattr(cli, "CheckpointIdentity", explode)
    with pytest.raises(AttributeError):
        run(cli, argv(corpus_root, output))

    # Nothing is created before the identity is built: no mkdir, no begin(), no
    # write happens earlier in run_prepare_corpus. So the failed START leaves
    # NOTHING that could masquerade as progress.
    checkpoint = output / "_checkpoint"
    assert not output.exists(), "the failed START must not leave an output directory"
    assert not (checkpoint / STATE_NAME).exists(), "no state means no claimed progress"
    assert not (checkpoint / COMPLETE_NAME).exists()
    assert not (output / CHUNKS_NAME).exists()

    # and a re-run starts cleanly from document 0
    monkeypatch.setattr(cli, "CheckpointIdentity", real_identity)
    parsed, status = run(cli, argv(corpus_root, output))
    assert status == 0
    assert (output / CHUNKS_NAME).is_file()


def test_a_stale_output_directory_without_a_checkpoint_is_still_refused(wired, tmp_path):
    """The immutable-output contract is unchanged: only a real checkpoint or a
    completion marker licenses reusing a directory."""
    cli, corpus_root, _ = wired
    output = tmp_path / "out"
    output.mkdir(parents=True)
    (output / "leftover.txt").write_text("from some earlier run", encoding="utf-8")
    parsed, status = run(cli, argv(corpus_root, output))
    assert status == 2, "a directory with no state and no COMPLETE must be refused"
    assert (output / "leftover.txt").exists(), "user data is never deleted"


def test_help_is_side_effect_free():
    result = subprocess.run(
        [__import__("sys").executable, str(CLI), "prepare-corpus", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "--checkpoint-dir" in result.stdout
    assert "--repository-head" not in result.stdout


# ---------------------------------------------------------------------------
# The repository-identity helper
# ---------------------------------------------------------------------------
def test_head_is_a_full_forty_character_sha():
    head = resolve_repository_head()
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    actual = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    assert head == actual


@pytest.mark.parametrize("output, message", [
    ("main", "not a full 40-character"),
    ("f9c23fe", "not a full 40-character"),
    ("", "not a full 40-character"),
    ("Z" * 40, "not a full 40-character"),
])
def test_a_non_sha_result_is_rejected(monkeypatch, output, message):
    import unmark.stage1.checkpoint as checkpoint_module

    class Result:
        returncode = 0
        stdout = output
        stderr = ""

    import subprocess as sp

    monkeypatch.setattr(sp, "run", lambda *a, **k: Result())
    with pytest.raises(CheckpointViolation, match=message):
        resolve_repository_head()


def test_a_git_failure_fails_closed(monkeypatch):
    import subprocess as sp

    class Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(sp, "run", lambda *a, **k: Failed())
    with pytest.raises(CheckpointViolation, match="git exited 128"):
        resolve_repository_head()


def test_a_missing_git_binary_fails_closed(monkeypatch):
    import subprocess as sp

    def missing(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(sp, "run", missing)
    with pytest.raises(CheckpointViolation, match="cannot resolve the repository HEAD"):
        resolve_repository_head()


def test_head_is_never_defaulted_or_environment_supplied(monkeypatch):
    """No env var may stand in for the repository's own answer."""
    import ast

    source = (REPO / "unmark" / "stage1" / "checkpoint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_repository_head")
    # Structural, not a substring scan over prose: the docstring legitimately
    # explains that no environment override exists, so scanning the source text
    # would match this function's own documentation.
    code = [n for n in fn.body if not (isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    body = "\n".join(ast.unparse(n) for n in code)
    assert "environ" not in body and "getenv" not in body, body
    assert "unknown" not in body.lower()
    assert "rev-parse" in body and "HEAD" in body
