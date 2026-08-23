"""The REAL validation preparation path and its truncation contract (Audit 030 §X).

The third real no-update smoke died here:

    prepare_with_condition(...) -> truncation.check(len(reference_ids), ...)
    AttributeError: 'NoneType' object has no attribute 'check'

because the measurement tool passed `truncation=None`. Twenty-two measurement
tests had passed moments earlier, because **none of them ever executed
`prepare_condition_batch`** -- one asserted it appeared in `validation_timing`'s
call graph, and the runtime fixture substituted hand-built integers for the whole
preparation stage and started at `evaluate`, downstream of the defect.

These tests run the real `prepare_condition_batch` and the real
`prepare_with_condition` with the real authoritative `TruncationPolicy`. Torch-free
and PhoBERT-free, so they run in the ML-free venv on every run.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.linguistics import load_inventory, make_classifier
from unmark.stage1.contracts import (
    OverflowBehaviour,
    Stage1ContractViolation,
    TruncationPolicy,
)
from unmark.stage1.data import prepare_with_condition
from unmark.stage1.execute import TRUNCATION
from unmark.stage1.protocol import MAX_LENGTH, VALIDATION_CONDITIONS
from unmark.stage1.validation import HeldOutExample, prepare_condition_batch

REPO = pathlib.Path(__file__).resolve().parents[1]

provisioned = pytest.mark.skipif(
    not (REPO / ".resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt").is_file(),
    reason="the git-ignored inventory cache is not provisioned in this runtime",
)


class StubTokenizer:
    """Whitespace/BPE-shaped stand-in. No PhoBERT, no download.

    `pieces_per_word` lets a test push a sequence past 256 tokens without needing
    a 256-word string, so the overflow case exercises the same code path.
    """

    unk_token_id = 3
    pad_token_id = 1

    def __init__(self, pieces_per_word: int = 1) -> None:
        self.pieces_per_word = pieces_per_word

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for chunk in text.split():
            if self.pieces_per_word == 1:
                out.append(chunk)
            else:
                out += [chunk + "@@"] * (self.pieces_per_word - 1) + [chunk]
        return out

    def convert_tokens_to_ids(self, tokens):
        return [7 + (len(t) % 5) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + list(ids) + [2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


@pytest.fixture(scope="module")
def classifier():
    """The REAL resolved scientific classifier -- not a stub."""
    return make_classifier(load_inventory())


def held_out(n_words: int, count: int = 3):
    text = " ".join(["tiếng", "việt", "không", "dấu"][i % 4] for i in range(n_words))
    return [HeldOutExample(f"doc-{i:04d}#0", text) for i in range(count)]


# ---------------------------------------------------------------------------
# 1. The authoritative object, and that None was never valid
# ---------------------------------------------------------------------------
def test_the_authoritative_policy_is_the_locked_science():
    assert TRUNCATION.max_length == MAX_LENGTH == 256
    assert TRUNCATION.on_overflow is OverflowBehaviour.FAIL
    assert TRUNCATION.is_enabled


def test_truncation_is_a_required_non_optional_argument():
    """Positive contract: the parameter is typed and has no default."""
    import inspect

    for function in (prepare_condition_batch, prepare_with_condition):
        parameter = inspect.signature(function).parameters["truncation"]
        assert parameter.default is inspect.Parameter.empty, function.__name__
        assert parameter.annotation in ("TruncationPolicy", TruncationPolicy), function.__name__


def test_none_cannot_satisfy_the_contract(classifier):
    """The exact third-smoke failure, pinned so it cannot be reintroduced."""
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'check'"):
        prepare_condition_batch(
            held_out(4), StubTokenizer(), "FULL", truncation=None, classifier=classifier
        )


# ---------------------------------------------------------------------------
# 2. The real path: <= 256 succeeds, all four conditions
# ---------------------------------------------------------------------------
@provisioned
@pytest.mark.parametrize("condition", list(VALIDATION_CONDITIONS))
def test_every_condition_prepares_under_the_real_policy(condition, classifier):
    prepared = prepare_condition_batch(
        held_out(8), StubTokenizer(), condition, truncation=TRUNCATION, classifier=classifier
    )
    assert len(prepared) == 3
    for item in prepared:
        assert len(item.reference_input_ids) <= MAX_LENGTH
        assert len(item.base_input_ids) <= MAX_LENGTH
        assert item.metadata["validation_condition"] == condition


@provisioned
def test_nothing_is_truncated_when_it_fits(classifier):
    """Lengths must be whatever the tokenizer produced -- never clipped to 256."""
    tokenizer = StubTokenizer()
    example = held_out(9, count=1)[0]
    expected = len(tokenizer.build_inputs_with_special_tokens(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text))
    ))
    prepared = prepare_condition_batch(
        [example], tokenizer, "FULL", truncation=TRUNCATION, classifier=classifier
    )
    assert len(prepared[0].reference_input_ids) == expected
    assert expected < MAX_LENGTH, "this fixture must fit, or it tests the wrong branch"


# ---------------------------------------------------------------------------
# 3. > 256 fails closed, for BOTH checked streams
# ---------------------------------------------------------------------------
@provisioned
def test_an_overlong_reference_fails_closed(classifier):
    """The first of the two `truncation.check` calls: the reference stream."""
    with pytest.raises(Stage1ContractViolation) as caught:
        prepare_condition_batch(
            held_out(MAX_LENGTH + 50, count=1), StubTokenizer(), "FULL",
            truncation=TRUNCATION, classifier=classifier,
        )
    message = str(caught.value)
    assert "reference sequence" in message
    assert "exceeds max_length 256" in message
    assert "does not truncate" in message


@provisioned
def test_an_overlong_base_stream_is_also_checked(classifier):
    """The second call guards `base_ids` -- the RAW_BASE grid the adapter reads.

    Asserted structurally as well, because a fixture that trips the reference
    check first would leave the base check unexercised and nobody would notice.
    """
    import ast
    import inspect

    source = inspect.getsource(prepare_with_condition)
    checks = [
        ast.literal_eval(node.args[1])
        for node in ast.walk(ast.parse(source.lstrip()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "check"
        and isinstance(node.args[1], ast.Constant)
    ]
    assert checks == ["reference sequence", "base sequence"], checks


@provisioned
def test_overflow_raises_rather_than_silently_skipping(classifier):
    """FAIL, not SKIP: an overlong held-out chunk must never vanish from the set."""
    skipping = TruncationPolicy(max_length=8, on_overflow=OverflowBehaviour.SKIP)
    # SKIP would return None, which the validation layer converts into a hard error
    # rather than a shorter batch.
    from unmark.stage1.validation import ValidationContractViolation

    with pytest.raises(ValidationContractViolation) as caught:
        prepare_condition_batch(
            held_out(40, count=1), StubTokenizer(), "FULL",
            truncation=skipping, classifier=classifier,
        )
    assert "overflowed" in str(caught.value)


@provisioned
def test_the_scientific_classifier_is_resolved_on_this_path(classifier):
    from unmark.corruption.eligibility import EligibilityPolicy, active_eligibility_policy

    assert active_eligibility_policy() is EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY
    prepared = prepare_condition_batch(
        held_out(6, count=1), StubTokenizer(), "P50",
        truncation=TRUNCATION, classifier=classifier,
    )
    assert prepared[0].corruption_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. The measurement tool must pass exactly this object
# ---------------------------------------------------------------------------
def test_validation_timing_cannot_pass_truncation_none():
    """AST, not text: the *argument* must be the production `TRUNCATION` name.

    Deliberately not a source-string search -- the tool's own comment explains the
    old `truncation=None` defect, and a text match would trip over the explanation
    instead of the code. This reads the actual keyword argument of the actual call.
    """
    import ast

    source = pathlib.Path(
        REPO / "scripts/stage1_pretrain_measurements.py"
    ).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "validation_timing")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
             == "prepare_condition_batch"]
    assert calls, "validation_timing must call prepare_condition_batch"
    for call in calls:
        keyword = next((k for k in call.keywords if k.arg == "truncation"), None)
        assert keyword is not None, "truncation must be passed explicitly"
        assert isinstance(keyword.value, ast.Name), ast.unparse(keyword.value)
        assert keyword.value.id == "TRUNCATION", (
            f"validation_timing passes truncation={ast.unparse(keyword.value)}; it must "
            "pass the production policy imported from unmark.stage1.execute"
        )


def test_the_measurement_tool_builds_no_policy_of_its_own():
    """No second MAX_LENGTH and no measurement-specific TruncationPolicy."""
    import ast

    source = pathlib.Path(
        REPO / "scripts/stage1_pretrain_measurements.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    constructed = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
                   for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "TruncationPolicy" not in constructed, (
        "the measurement tool must reuse the production policy, not build its own"
    )
    assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert "MAX_LENGTH" not in assigned and "TRUNCATION" not in assigned, (
        "a second length/truncation definition must not appear here"
    )
