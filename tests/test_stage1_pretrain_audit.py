"""Regressions for concrete defects found by the PRE-TRAIN audit (Audit 030).

Three findings, each pinned so it cannot regress or be misread later. None of
these tests duplicates an implementation; each fails on a real defect class the
audit identified by tracing code.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# F2 -- pi_strip is defined twice, as two independent literals
# ---------------------------------------------------------------------------
def test_pi_strip_has_one_value_even_though_it_has_two_definitions():
    """`contracts.PI_STRIP` governs corruption; `protocol.PI_STRIP` is recorded.

    They are **separate literals** -- `contracts.py` does not import
    `protocol.py`. Today both are 0.25, so the science is correct and the
    manifest is honest. If one were ever edited alone, the corruption engine
    would draw against one value while the manifest recorded the other, and
    nothing else in the repository would notice. This is that noticer.
    """
    from unmark.stage1.contracts import PI_STRIP as governing
    from unmark.stage1.protocol import PI_STRIP as recorded

    assert governing == recorded, (
        f"corruption uses pi_strip={governing} but the manifest records "
        f"{recorded}; a run's provenance would misdescribe its own corruption"
    )


def test_the_corruption_policy_default_is_the_governing_constant():
    """The default must come from the constant, not be re-typed as a literal."""
    from unmark.stage1.contracts import PI_STRIP, CorruptionRatePolicy

    policy = CorruptionRatePolicy(seed=1)
    assert policy.pi_strip == PI_STRIP


# ---------------------------------------------------------------------------
# F1 -- transform counters are blind under multiprocessing
# ---------------------------------------------------------------------------
class _Tokenizer:
    all_special_tokens = ["<s>", "</s>"]

    def get_added_vocab(self):
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    def tokenize(self, text):
        return [c for c in text if not c.isspace()]

    def convert_tokens_to_ids(self, tokens):
        return [1] * len(tokens)

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


def test_a_freshly_built_length_pair_reports_exactly_two_authoritative_queries():
    """Pins the signature the real full run printed, so it is never misread.

    The `aa49785` full prepare reported `length queries 0, BPE run evaluations 0,
    run-cache hits 0, incremental appends 0, full fallbacks 0, authoritative
    verifications 2`. That is **not** evidence the authoritative length contract
    was bypassed: it is exactly what an unused pair of composers reports, because
    each `RunLengthComposer.__init__` derives its special-token count with one
    `authoritative_length("")` call. Under `--prepare-workers 16` the chunking
    ran in workers, each with its **own** length functions and its own
    fail-closed verifier, so the main process's counters could not see it.

    If this ever reports something other than 2, the constructor changed and the
    §AB reading of that run's counters needs revisiting.
    """
    from unmark.stage1.lengths import build_length_functions

    _, _, transforms = build_length_functions(_Tokenizer())
    counters = transforms.counters.to_dict()

    assert counters["authoritative_queries"] == 2, counters
    for blind in ("length_queries", "bpe_run_evaluations", "run_cache_hits",
                  "incremental_appends", "full_fallbacks"):
        assert counters[blind] == 0, (blind, counters[blind])


def test_each_worker_builds_its_own_length_functions():
    """Structural: the worker initialiser must construct its own composers.

    This is *why* the main-process counters read zero, and also why the
    authoritative verifier still ran 16 times over. Asserted on the call graph so
    a refactor that silently shared main's functions with workers -- which would
    make the counters meaningful but the tokenizer state shared across processes
    -- is caught.
    """
    from unmark.stage1 import parallel

    tree = ast.parse(inspect.getsource(parallel))
    initialiser = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_initialise_worker"
    )
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(initialiser) if isinstance(n, ast.Call)
    }
    assert "build_length_functions" in called, called


# ---------------------------------------------------------------------------
# F3 -- training checkpointing is implemented but never invoked
# ---------------------------------------------------------------------------
def test_training_checkpoints_are_now_persisted():
    """REPLACED after the F3 hardening — it previously asserted the gap.

    Audit 030 recorded that `checkpoint_payload` existed but was called from
    nowhere and `execute_stage` hard-coded `resume=None`, and the original test
    said in as many words that it "will fail the moment persistence is wired, at
    which point resume equivalence needs its own evidence". Persistence is now
    wired and that evidence exists
    (`test_stage1_training_resume_state.py`, and the torch half in
    `test_stage1_training_resume.py`), so the assertion is inverted rather than
    deleted: the gap must not silently reopen.
    """
    import unmark.stage1.execute as execute_module
    import unmark.stage1.trainer as trainer_module

    trainer_source = inspect.getsource(trainer_module)
    assert "def save_training_checkpoint" in trainer_source
    assert "def load_training_checkpoint" in trainer_source

    # The loop publishes at the validation boundary.
    tree = ast.parse(trainer_source)
    train_run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_run"
    )
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(train_run) if isinstance(n, ast.Call)
    }
    assert "save_training_checkpoint" in called, called
    assert "checkpoint_payload" in called, called

    # And the orchestrator both supplies a checkpoint directory and can resume.
    execute_source = inspect.getsource(execute_module)
    assert "checkpoint_dir=run_checkpoints" in execute_source
    assert "load_training_checkpoint(run_checkpoints)" in execute_source


def test_execute_stage_no_longer_hard_codes_resume_none():
    """The companion fact, inverted: resume is now explicit and operator-driven."""
    import unmark.stage1.execute as execute_module

    source = inspect.getsource(execute_module)
    # `resume=... if resume else None` is the correct conditional form; what must
    # not come back is the UNCONDITIONAL hard-code both train_run call sites had.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("resume=None"):
            raise AssertionError(
                f"execute_stage hard-codes resume unconditionally again: {stripped!r}; "
                "an interrupted stage would silently restart from zero"
            )
    assert "resume: bool = False" in source, "resume must be an explicit opt-in"
    assert "if resume else None" in source, "resume must be operator-driven, not automatic"


def test_the_continuation_preserves_optimizer_and_sampler_state():
    """The 20k->40k leg must continue the SAME run, not start a new one.

    Found during the F3 hardening: the continuation passed `resume=None`, so it
    rebuilt the optimizer and restarted the sampler at visit 0 while the locked
    budget rule requires preserving "adapter, optimizer, visit, cursor and
    streams". It now resumes from the checkpoint the first leg wrote at `cap`.
    """
    import unmark.stage1.execute as execute_module

    tree = ast.parse(inspect.getsource(execute_module))
    stage = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_stage"
    )
    # AST, not a source split. The old body split the source on the literal
    # `"if result.continued:"`, so rewording the guard broke the test without
    # any behaviour changing -- and a source split cannot see whether `resume`
    # is actually passed or merely mentioned nearby.
    train_calls = [
        n for n in ast.walk(stage)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "train_run"
    ]
    assert len(train_calls) == 2, (
        f"expected the initial leg and exactly one continuation, found {len(train_calls)}"
    )
    for call in train_calls:
        keywords = {k.arg: k.value for k in call.keywords}
        assert "resume" in keywords, "every leg must be resume-capable"
        value = keywords["resume"]
        assert not (isinstance(value, ast.Constant) and value.value is None), (
            "a continuation that passes resume=None rebuilds the optimizer and "
            "restarts the sampler: a continuation in name only"
        )
        assert "cap" in keywords, "every leg must state the budget it runs under"
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(stage) if isinstance(n, ast.Call)
    }
    assert "load_training_checkpoint" in called
    assert "resume_cap" in called, (
        "the resumed leg's cap must come from the checkpoint, not from a default"
    )


# ---------------------------------------------------------------------------
# Standing guarantees this audit re-verified
# ---------------------------------------------------------------------------
def test_official_test_has_no_cli_route():
    """No flag anywhere in the runner may reach official TEST."""
    source = pathlib.Path("scripts/stage1_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags = {
        arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }
    assert flags, "expected to find CLI flags"
    for flag in flags:
        assert "test" not in flag.lower(), f"{flag} could reach official TEST"


def test_the_contamination_screen_refuses_any_unlisted_source():
    from unmark.stage1.corpus import CorpusContractViolation, screen_contamination

    with pytest.raises(CorpusContractViolation) as caught:
        screen_contamination([], {"uitvsfc_official_test": ["anything"]})
    assert "SEALED" in str(caught.value)
