"""Deterministic parallel Stage-1 preparation (Audit 030 §AG).

§AF measured the training path as **preparation-bound** — 79 % of each step is
`prepare_example`. Preparation now runs across a persistent spawn pool. That is a
pure engineering change **only if** every prepared example is byte-identical to
the serial path's, so this file's job is to prove exactly that, and to prove the
main process kept everything scientific.

These tests really start worker processes. They use a tiny injected tokenizer, so
**no PhoBERT is downloaded**, and they gate on the git-ignored pinned inventory
because workers verify it for real.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.contracts import (
    CorruptionRatePolicy,
    OverflowBehaviour,
    TruncationPolicy,
)
from unmark.stage1.preparation import (
    MULTIPROCESSING_START_METHOD,
    PREPARATION_WORKERS,
    PreparationContractViolation,
    PreparationPool,
    pinned_tokenizer,
    prepare_serially,
    preparation_provenance,
    worker_config,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
TRUNCATION = TruncationPolicy(max_length=256, on_overflow=OverflowBehaviour.FAIL)
POLICY = CorruptionRatePolicy(seed=35422)

provisioned = pytest.mark.skipif(
    not (REPO / ".resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt").is_file(),
    reason="workers verify the pinned inventory for real; it is not provisioned here",
)


# ---------------------------------------------------------------------------
# A tiny picklable tokenizer, resolved by reference in the spawned child
# ---------------------------------------------------------------------------
class TinyTokenizer:
    unk_token_id = 3
    pad_token_id = 1

    def tokenize(self, text):
        out = []
        for chunk in text.split():
            out += [chunk[:3] + "@@", chunk[3:]] if len(chunk) > 4 else [chunk]
        return out

    def convert_tokens_to_ids(self, tokens):
        return [7 + (len(t) % 5) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


def tiny_tokenizer_factory(checkpoint, revision):
    """Module-level so `spawn` can pickle it by reference."""
    return TinyTokenizer()


def config(workers_factory=tiny_tokenizer_factory):
    return worker_config(
        encoder_checkpoint="tiny", encoder_revision="tiny-rev",
        corruption_policy=POLICY, truncation=TRUNCATION,
        unk_token_id=TinyTokenizer.unk_token_id, tokenizer_factory=workers_factory,
    )


def fixture_tasks():
    """Deliberately varied: scopes, marks, punctuation, mixed script, lengths."""
    texts = [
        "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương",
        "toi dung Python va PyTorch de huan luyen mo hinh",          # unmarked
        "Chào bạn! Số 123, e-mail: a@b.com — thử nghiệm (dấu câu).",  # punctuation
        "machine learning và trí tuệ nhân tạo ở Việt Nam",            # mixed script
        "cà phê sữa đá",                                              # short
        "Hà Nội " * 40,                                               # long
        "Đường Trường Sơn đông, đường Trường Sơn tây",
        "a",                                                          # minimal
    ]
    tasks = []
    for visit in (0, 1, 2):
        for i, text in enumerate(texts):
            tasks.append((f"doc-{i:04d}#0", visit, text))
    return tasks


def serial(tasks):
    from unmark.linguistics import load_inventory, make_classifier

    return prepare_serially(
        tasks, TinyTokenizer(), corruption_policy=POLICY, truncation=TRUNCATION,
        classifier=make_classifier(load_inventory()),
        unk_token_id=TinyTokenizer.unk_token_id,
    )


def identical(a, b) -> bool:
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    return all(getattr(a, f) == getattr(b, f) for f in a.__dataclass_fields__)


# ---------------------------------------------------------------------------
# A. SERIAL vs PARALLEL — exact equality, no tolerance
# ---------------------------------------------------------------------------
@provisioned
def test_parallel_preparation_is_byte_identical_to_serial():
    tasks = fixture_tasks()
    reference = serial(tasks)
    with PreparationPool(config(), workers=4) as pool:
        parallel = pool.prepare(tasks)

    assert len(parallel) == len(reference) == len(tasks)
    for index, (got, want) in enumerate(zip(parallel, reference)):
        assert identical(got, want), f"example {index} ({tasks[index][0]}, visit {tasks[index][1]})"


# ---------------------------------------------------------------------------
# B. ORDER PRESERVATION — completion order must not leak into scientific order
# ---------------------------------------------------------------------------
@provisioned
def test_results_come_back_in_input_order_not_completion_order():
    """The long task is submitted FIRST, so it finishes last. Order must hold."""
    # Long enough to finish last, still under MAX_LENGTH so it must not fail closed.
    tasks = [("doc-slow#0", 0, "Hà Nội " * 90)] + [
        (f"doc-{i:04d}#0", 0, "cà phê") for i in range(1, 12)
    ]
    reference = serial(tasks)
    with PreparationPool(config(), workers=4) as pool:
        parallel = pool.prepare(tasks)

    assert [p.sample_id for p in parallel] == [t[0] for t in tasks]
    for got, want in zip(parallel, reference):
        assert identical(got, want)


# ---------------------------------------------------------------------------
# C. WORKER-COUNT INVARIANCE
# ---------------------------------------------------------------------------
@provisioned
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_worker_count_does_not_change_the_prepared_output(workers):
    tasks = fixture_tasks()
    reference = serial(tasks)
    with PreparationPool(config(), workers=workers) as pool:
        parallel = pool.prepare(tasks)
    for index, (got, want) in enumerate(zip(parallel, reference)):
        assert identical(got, want), (workers, index)


def test_the_production_worker_count_is_the_locked_operational_constant():
    assert PREPARATION_WORKERS == 8
    assert preparation_provenance() == {
        "preparation_backend": "multiprocessing_spawn",
        "preparation_workers": 8,
        "order_preserving": True,
        "prefetch": False,
    }


# ---------------------------------------------------------------------------
# D. FAILURE PROPAGATION — no partial batch may reach training
# ---------------------------------------------------------------------------
def exploding_factory(checkpoint, revision):
    raise RuntimeError("tokenizer identity could not be established")


@provisioned
def test_a_worker_failure_aborts_and_never_falls_back_to_serial():
    with pytest.raises(Exception) as caught:
        with PreparationPool(config(exploding_factory), workers=2) as pool:
            pool.prepare(fixture_tasks())
    assert "serial" not in str(caught.value).lower() or "will not fall back" in str(caught.value)


@provisioned
def test_a_preparation_error_propagates_rather_than_returning_a_short_batch():
    tasks = fixture_tasks()[:4] + [("doc-bad#0", 0, None)]   # None text -> raises inside
    with PreparationPool(config(), workers=2) as pool:
        with pytest.raises(PreparationContractViolation) as caught:
            pool.prepare(tasks)
    assert "will not fall back to serial" in str(caught.value)


def test_using_the_pool_outside_its_context_is_refused():
    pool = PreparationPool(config(), workers=2)
    with pytest.raises(PreparationContractViolation, match="not running"):
        pool.prepare(fixture_tasks()[:1])


# ---------------------------------------------------------------------------
# E/F. OWNERSHIP and NO LOOK-AHEAD — structural, on the real call graph
# ---------------------------------------------------------------------------
def function(source: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node) -> list[str]:
    return [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(node) if isinstance(n, ast.Call)]


def test_workers_can_neither_own_nor_advance_a_sampler():
    from unmark.stage1 import preparation

    source = inspect.getsource(preparation)
    for forbidden in ("next_batch", "DeterministicSampler", "build_optimizer",
                      "save_training_checkpoint", "evaluate", "backward", "step",
                      "AutoModel", "batch_to_device", "collate_stage1_batch"):
        assert forbidden not in calls(ast.parse(source)), forbidden
    # AST, not text: the module docstring legitimately explains why fork is unsafe
    # on a CUDA parent, and a substring search would trip over that prose.
    touched = {ast.unparse(n.func) for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.Call)}
    assert not [c for c in touched if "cuda" in c.lower()], (
        f"workers must not touch CUDA: {[c for c in touched if 'cuda' in c.lower()]}"
    )
    imported = {a.name for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("torch" in m for m in imported), f"preparation must not import torch: {imported}"


def test_the_main_process_still_owns_sampler_update_and_checkpoint():
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = function(source, "train_run")
    called = calls(node)
    for owned in ("next_batch", "collate_stage1_batch", "batch_to_device",
                  "save_training_checkpoint", "backward", "step", "zero_grad"):
        assert owned in called, f"train_run must still own {owned}"


def test_exactly_one_sampler_batch_is_consumed_per_step_and_nothing_reads_ahead():
    """No prefetch: one `next_batch` in the loop, and no look-ahead buffering."""
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = function(source, "train_run")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.While))
    consumed = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "next_batch"]
    assert len(consumed) == 1, f"{len(consumed)} sampler consumptions per step"

    from unmark.stage1 import preparation

    assert preparation.PREFETCH_ENABLED is False
    for forbidden in ("submit", "as_completed", "Queue", "Thread"):
        assert forbidden not in calls(ast.parse(inspect.getsource(preparation))), forbidden


def test_global_update_increments_once_after_the_step():
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = function(source, "train_run")
    bumps = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign)
             and getattr(n.target, "id", None) == "global_update"]
    assert len(bumps) == 1


# ---------------------------------------------------------------------------
# G. CUDA SAFETY — spawn, never fork
# ---------------------------------------------------------------------------
def test_the_production_start_method_is_spawn_not_fork():
    assert MULTIPROCESSING_START_METHOD == "spawn"


def test_no_fork_context_can_be_introduced_by_a_later_refactor():
    """Source-level guard: a refactor must not silently switch to fork.

    Forking a CUDA-initialised parent is unsupported and can deadlock rather than
    fail cleanly, which is exactly what the production path would do.
    """
    from unmark.stage1 import preparation

    source = inspect.getsource(preparation)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get_context":
            assert len(node.args) == 1
            argument = node.args[0]
            assert isinstance(argument, ast.Name) and argument.id == "MULTIPROCESSING_START_METHOD", (
                f"get_context({ast.unparse(argument)}) must use the locked constant"
            )
    assert '"fork"' not in source and "'fork'" not in source.replace("fork_rng", "")


def test_execute_stage_uses_the_pinned_tokenizer_factory_not_a_stub():
    source = (REPO / "unmark/stage1/execute.py").read_text(encoding="utf-8")
    node = function(source, "execute_stage")
    assert "worker_config" in calls(node)
    assert "tokenizer_factory" not in ast.unparse(node), (
        "execute_stage must take the pinned default, never inject a factory"
    )


def test_every_train_run_call_receives_the_preparation_pool():
    """No silent serial degradation on the scientific path.

    `train_run` keeps a serial branch for tests and diagnostics, and it is *not* a
    failure fallback — a broken pool raises. But a future `execute_stage` edit that
    forgot to pass the pool would silently run ~7x slower with no error, which is
    exactly the parser/handler drift class this repository has been bitten by
    twice (§AA, §AD). Checked structurally so it cannot drift.
    """
    source = (REPO / "unmark/stage1/execute.py").read_text(encoding="utf-8")
    node = function(source, "execute_stage")
    train_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) == "train_run"]
    assert train_calls, "execute_stage must call train_run"
    for call in train_calls:
        keyword = next((k for k in call.keywords if k.arg == "preparation_pool"), None)
        assert keyword is not None, (
            f"train_run call at line {call.lineno} does not pass preparation_pool; "
            "it would silently prepare serially"
        )
        assert ast.unparse(keyword.value) == "preparation_pool", ast.unparse(keyword.value)


def test_the_serial_branch_is_not_a_failure_fallback():
    """A pool failure must abort, never degrade: no try/except around `prepare`."""
    source = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    node = function(source, "train_run")
    for handler in (n for n in ast.walk(node) if isinstance(n, ast.Try)):
        body = ast.unparse(handler)
        assert "prepare_serially" not in body or "preparation_pool" not in body, (
            "a serial fallback inside an exception handler would hide a broken pool"
        )


def test_the_pinned_factory_refuses_a_foreign_revision():
    with pytest.raises(PreparationContractViolation, match="not the locked"):
        pinned_tokenizer("vinai/phobert-base", "deadbeef")
