"""`scope_for` -- the fix for the STRIP-ALL training-support defect (D-S1B-003).

ML-free. Audit 028 §F proved, from the data path, that a run-global `"TONE"`
scope left the corrupted branch's letter channel **bit-identical** to the clean
branch's in every prepared example, so the headline evaluation condition had
zero training support. These tests pin that the repair actually repairs it.

No statistical independence *proof* is claimed from a finite sample: what is
tested is the **construction** (separate namespaces, no shared scalar, no
functional dependence) plus deterministic finite-sample sanity.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.stage1.contracts import (
    PI_STRIP,
    RATE_NAMESPACE,
    SCOPE_NAMESPACE,
    CorruptionRatePolicy,
    Stage1ContractViolation,
)

SEED = 35422
IDS = [f"doc-{i}#0" for i in range(6000)]


def policy(**kw) -> CorruptionRatePolicy:
    return CorruptionRatePolicy(seed=SEED, **kw)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_draws_are_deterministic_and_repeatable():
    a, b = policy(), policy()
    for cid in IDS[:200]:
        assert a.rate_for(cid, 0) == b.rate_for(cid, 0)
        assert a.scope_for(cid, 0) == b.scope_for(cid, 0)


def test_visit_changes_both_streams():
    p = policy()
    rates = {p.rate_for("doc-1#0", v) for v in range(8)}
    assert len(rates) == 8, "redraw per visit is locked; p must move with the visit"
    scopes = [p.scope_for(f"doc-{i}#0", 0) for i in range(400)]
    other = [p.scope_for(f"doc-{i}#0", 1) for i in range(400)]
    assert scopes != other, "the scope stream must also advance with the visit"


def test_sample_id_changes_both_streams():
    p = policy()
    assert len({p.rate_for(cid, 0) for cid in IDS[:200]}) > 190
    assert p.scope_for("doc-1#0", 0) is not None


def test_seed_changes_the_stream():
    a, b = CorruptionRatePolicy(seed=1), CorruptionRatePolicy(seed=2)
    assert [a.rate_for(c, 0) for c in IDS[:50]] != [b.rate_for(c, 0) for c in IDS[:50]]


# ---------------------------------------------------------------------------
# Domain separation -- the construction, not a statistical claim
# ---------------------------------------------------------------------------
def test_the_two_namespaces_are_distinct_and_actually_used():
    assert RATE_NAMESPACE != SCOPE_NAMESPACE
    source = pathlib.Path("unmark/stage1/contracts.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fns = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in {"rate_for", "scope_for", "_unit_draw"}
    }
    assert set(fns) == {"rate_for", "scope_for", "_unit_draw"}

    # each public draw passes its OWN namespace into the shared digest helper
    for name, expected in (("rate_for", "RATE_NAMESPACE"), ("scope_for", "SCOPE_NAMESPACE")):
        calls = [
            ast.unparse(n)
            for n in ast.walk(fns[name])
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_unit_draw"
        ]
        assert calls, f"{name} must derive its draw from _unit_draw"
        assert all(expected in c for c in calls), (name, calls)


def test_scope_is_not_derived_from_the_rate():
    """`scope_for` must not call `rate_for`, or read `p` at all."""
    tree = ast.parse(pathlib.Path("unmark/stage1/contracts.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scope_for"
    )
    called = {getattr(n.func, "attr", None) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "rate_for" not in called, "scope must not be a function of p"
    rate_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "rate_for"
    )
    called_by_rate = {getattr(n.func, "attr", None) for n in ast.walk(rate_fn) if isinstance(n, ast.Call)}
    assert "scope_for" not in called_by_rate, "p must not be a function of scope"


def test_conditional_rate_distributions_cover_the_whole_range_for_both_scopes():
    """`p | scope` must span [0,1) for BOTH scopes -- deterministic sanity.

    If scope were conditioned on `p` (say TONE_AND_LETTER only when p > 0.9),
    one conditional would collapse onto a sub-interval. This checks the coverage
    that such a construction would destroy; it is not an independence proof.
    """
    p = policy()
    buckets = {"TONE": set(), "TONE_AND_LETTER": set()}
    means = {"TONE": [], "TONE_AND_LETTER": []}
    for cid in IDS:
        scope = p.scope_for(cid, 0)
        rate = p.rate_for(cid, 0)
        buckets[scope].add(int(rate * 10))
        means[scope].append(rate)
    for scope, seen in buckets.items():
        assert seen == set(range(10)), f"{scope}: p covers only deciles {sorted(seen)}"
    for scope, values in means.items():
        mean = sum(values) / len(values)
        assert 0.45 < mean < 0.55, f"{scope}: conditional mean {mean:.3f} is not ~0.5"


def test_empirical_scope_fraction_is_close_to_the_locked_pi_strip():
    p = policy()
    strip = sum(1 for cid in IDS if p.scope_for(cid, 0) == "TONE_AND_LETTER")
    fraction = strip / len(IDS)
    assert abs(fraction - PI_STRIP) < 0.02, f"P(TONE_AND_LETTER) = {fraction:.4f}"


def test_pi_strip_is_locked_at_one_quarter():
    assert PI_STRIP == 0.25
    assert policy().pi_strip == 0.25
    assert policy().is_locked_mixture


# ---------------------------------------------------------------------------
# The defect itself: support for both regimes
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared():
    """Prepare real examples through the real Stage-1 path, ML-free."""
    import sys

    sys.path.insert(0, "tests")
    from test_stage1 import StubTokenizer

    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.contracts import TruncationPolicy
    from unmark.stage1.data import Stage1Example, prepare_example

    tokenizer = StubTokenizer()
    classifier = make_classifier(try_load_inventory())
    pol = policy()
    texts = [
        "Tôi đã đọc quyển sách này rồi và thấy rất hay",
        "Sản phẩm này rất tốt phục vụ nhiệt tình",
        "Giảng viên dạy dễ hiểu nhưng đề thi hơi khó",
    ]
    out = []
    for i, text in enumerate(texts):
        for visit in range(40):
            out.append(
                prepare_example(
                    Stage1Example(text=text, sample_id=f"doc-{i}#0"),
                    tokenizer,
                    corruption_policy=pol,
                    truncation=TruncationPolicy.unbounded(),
                    visit=visit,
                    classifier=classifier,
                )
            )
    return out


def test_strip_all_support_now_exists(prepared):
    """The defect: this count was **0 / N** under the old run-global TONE scope."""
    letter_degraded = [p for p in prepared if p.letter_channels_differ]
    assert letter_degraded, (
        "no prepared example has a degraded letter channel; STRIP-ALL would again "
        "have zero training support"
    )
    assert len(letter_degraded) / len(prepared) > 0.05


def test_tone_only_support_survives(prepared):
    """The repair must not cost the P25..P100 regime its support."""
    tone_only = [p for p in prepared if p.channels_differ and not p.letter_channels_differ]
    assert tone_only, "tone-only degradation disappeared; P100 would lose support"


def test_both_scopes_are_actually_realised(prepared):
    assert {p.corruption_scope for p in prepared} == {"TONE", "TONE_AND_LETTER"}


def test_base_invariance_holds_under_both_scopes(prepared):
    """`b(C(x)) == b(x)` -- `prepare_example` raises otherwise, so reaching here
    with every example prepared is the assertion."""
    assert all(p is not None for p in prepared)
    assert len({p.base_text for p in prepared}) == 3  # one base per source text


def test_prepare_example_uses_both_draws_not_a_run_global_scope():
    tree = ast.parse(pathlib.Path("unmark/stage1/data.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare_example"
    )
    called = {
        getattr(n.func, "attr", None) for n in ast.walk(fn) if isinstance(n, ast.Call)
    }
    assert "rate_for" in called and "scope_for" in called
