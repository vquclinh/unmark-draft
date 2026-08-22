"""Stage-1 objective and data path.

Three tiers: static/AST guards, torch-free data tests (the preparation layer
genuinely does not need torch), and torch-gated numerics that skip cleanly in the
ML-free local environment.

**Nothing here trains.** One synthetic backward runs in a torch-gated test to
verify gradient routing; no optimizer is constructed and no parameter is updated.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.linguistics import load_inventory, make_classifier
from unmark.stage1 import (
    LOCKED_STAGE1_VALUES,
    OPEN_STAGE1_VALUES,
    SCIENTIFIC_REQUIRED_VALUES,
    BaseInvarianceViolation,
    CorruptionRatePolicy,
    ObjectiveWeights,
    OverflowBehaviour,
    Stage1Branch,
    Stage1ContractViolation,
    Stage1Example,
    Stage1Purpose,
    Stage1RunConfig,
    TruncationPolicy,
    UnresolvedStage1Value,
    padded_stage1_batch,
    prepare_example,
    require_resolved,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE1_MODULES = (
    "unmark/stage1/contracts.py",
    "unmark/stage1/data.py",
    "unmark/stage1/objective.py",
    "unmark/stage1/__init__.py",
)
OBJECTIVE = "unmark/stage1/objective.py"
DATA = "unmark/stage1/data.py"


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    return ast.parse(source(name))


def called_names(name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


def imported(name: str) -> set[str]:
    parsed = tree(name)
    return {
        (n.module or "").split(".")[0]
        for n in ast.walk(parsed)
        if isinstance(n, ast.ImportFrom)
    } | {
        a.name.split(".")[0]
        for n in ast.walk(parsed)
        if isinstance(n, ast.Import)
        for a in n.names
    }


class StubTokenizer:
    """Whitespace-chunk stand-in.

    `fragment_marked` makes diacritized text fragment more than its stripped
    base, which is what a real BPE does -- so the reference and base branches
    genuinely get different sequence lengths.
    """

    unk_token_id = 3
    pad_token_id = 1

    def __init__(self, fragment_marked: bool = False) -> None:
        self.fragment_marked = fragment_marked

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for chunk in text.split():
            pieces = [chunk[:3] + "@@", chunk[3:]] if len(chunk) > 4 else [chunk]
            if self.fragment_marked and any(ord(c) > 127 for c in chunk):
                pieces = [p + "@@" for p in pieces[:-1]] + [pieces[-1]] + ["~"]
            out += pieces
        return out

    def convert_tokens_to_ids(self, tokens):
        return [7 + (len(t) % 5) for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + list(ids) + [2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


@pytest.fixture(scope="module")
def classifier():
    return make_classifier(load_inventory())


def prepared(text, sample_id, classifier, *, seed=20260820, tokenizer=None, **kwargs):
    """Test helper. `truncation` and `visit` are explicit -- they have no
    defaults in the real API, and the helper must not reintroduce one."""
    kwargs.setdefault("truncation", TruncationPolicy.unbounded())
    kwargs.setdefault("visit", 0)
    return prepare_example(
        Stage1Example(text, sample_id),
        tokenizer or StubTokenizer(),
        corruption_policy=CorruptionRatePolicy(seed=seed),
        classifier=classifier,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1-4. Branch separation and base invariance
# ---------------------------------------------------------------------------
def test_reference_text_is_kept_distinct_from_base_text(classifier):
    example = prepared("Tôi học", "s1", classifier)
    assert example.canonical_text == "Tôi học"
    assert example.base_text == "Toi hoc"
    assert example.canonical_text != example.base_text


def test_branches_are_named_and_distinct():
    assert len(Stage1Branch) == 3
    assert not Stage1Branch.REFERENCE_CLEAN.uses_adapter
    assert Stage1Branch.ADAPTED_CLEAN.uses_adapter
    assert Stage1Branch.ADAPTED_CORRUPT.uses_adapter
    assert not Stage1Branch.REFERENCE_CLEAN.requires_gradient
    assert Stage1Branch.ADAPTED_CLEAN.requires_gradient


def test_adapted_branches_share_the_base_grid(classifier):
    example = prepared("Tôi đang học nghiên cứu tại Đại học", "s2", classifier)
    assert example.clean_encoded().input_ids == example.corrupt_encoded().input_ids
    assert (
        example.clean_encoded().special_tokens_mask
        == example.corrupt_encoded().special_tokens_mask
    )


def test_base_mismatch_fails_loud(classifier, monkeypatch):
    """If b(C(x)) != b(x) the preparation refuses rather than repairing."""
    import unmark.stage1.data as data_module

    real = data_module.project_text
    calls = {"n": 0}

    def sabotage(text, tokenizer, classifier_, unk):
        calls["n"] += 1
        base, ids, projections = real(text, tokenizer, classifier_, unk)
        if calls["n"] == 2:  # the corrupted branch
            base = base + "X"
        return base, ids, projections

    monkeypatch.setattr(data_module, "project_text", sabotage)
    with pytest.raises(BaseInvarianceViolation, match=r"b\(C\(x\)\) != b\(x\)"):
        prepared("Tôi học", "s3", classifier)


def test_clean_and_corrupt_channels_may_differ_while_base_ids_match(classifier):
    """The whole point: corruption is channel-level inside UNMARK."""
    example = prepared(
        "Tôi đang học nghiên cứu tại Đại học Quốc gia", "s4", classifier, seed=20260820
    )
    assert example.channels_differ, "expected corruption to change the tone channel"
    assert example.clean_encoded().input_ids == example.corrupt_encoded().input_ids


# ---------------------------------------------------------------------------
# 5-7. Lengths, padding domains, masks
# ---------------------------------------------------------------------------
def test_reference_and_base_lengths_may_differ(classifier):
    example = prepared(
        "Tôi đang học nghiên cứu", "s5", classifier, tokenizer=StubTokenizer(fragment_marked=True)
    )
    assert example.reference_length != example.base_length


def test_padding_domains_are_independent(classifier):
    tokenizer = StubTokenizer(fragment_marked=True)
    examples = [
        prepared("Tôi đang học nghiên cứu tại Đại học", "s6", classifier, tokenizer=tokenizer),
        prepared("Chào", "s7", classifier, tokenizer=tokenizer),
    ]
    batch = padded_stage1_batch(examples, pad_token_id=1)
    reference_width = len(batch["reference_input_ids"][0])
    base_width = len(batch["base_input_ids"][0])
    assert reference_width != base_width, "reference was padded to the base width"
    assert all(len(row) == reference_width for row in batch["reference_input_ids"])
    assert all(len(row) == base_width for row in batch["base_input_ids"])


def test_every_pooled_branch_has_a_special_tokens_mask(classifier):
    batch = padded_stage1_batch([prepared("Tôi học", "s8", classifier)], pad_token_id=1)
    assert "reference_special_tokens_mask" in batch
    assert "base_special_tokens_mask" in batch
    assert batch["reference_special_tokens_mask"][0][0] == 1
    assert batch["base_special_tokens_mask"][0][0] == 1


def test_padding_is_marked_special_and_unattended(classifier):
    tokenizer = StubTokenizer(fragment_marked=True)
    examples = [
        prepared("Tôi đang học nghiên cứu tại Đại", "s9", classifier, tokenizer=tokenizer),
        prepared("Chào", "s10", classifier, tokenizer=tokenizer),
    ]
    batch = padded_stage1_batch(examples, pad_token_id=1)
    short = batch["reference_attention_mask"][1]
    special = batch["reference_special_tokens_mask"][1]
    for index, attended in enumerate(short):
        if not attended:
            assert special[index] == 1, "padding must also be excluded as special"


# ---------------------------------------------------------------------------
# 8-11. Channel semantics carried through unchanged
# ---------------------------------------------------------------------------
def test_tone_na_uses_sentinel_and_mask(classifier):
    example = prepared("Tôi học.", "s11", classifier)
    assert example.clean_tone_ids[0] == -1
    assert example.clean_tone_mask[0] is False
    assert example.clean_tone_ids[-1] == -1


def test_unmarked_is_a_real_row_not_na(classifier):
    from unmark.modeling import OBSERVABLE_TONE_IDS

    example = prepared("Toi hoc", "s12", classifier)  # already unmarked
    live = [i for i, m in zip(example.clean_tone_ids, example.clean_tone_mask) if m]
    assert live, "expected at least one live tone position"
    assert all(0 <= i < 7 for i in live)
    assert OBSERVABLE_TONE_IDS["UNMARKED"] in live


def test_none_participates_in_letter_contributors(classifier):
    from unmark.modeling import LETTER_LABEL_IDS

    example = prepared("hoc", "s13", classifier)
    contributors = [row for row in example.clean_letter_ids if row]
    assert contributors
    assert LETTER_LABEL_IDS["NONE"] in contributors[0]


def test_letter_na_is_excluded_from_contributors(classifier):
    example = prepared("hoc.", "s14", classifier)
    joined = [i for row in example.clean_letter_ids for i in row]
    assert all(0 <= i < 5 for i in joined), "an NA sentinel leaked into contributors"


def test_special_tokens_have_no_letter_contributors(classifier):
    example = prepared("Tôi học", "s15", classifier)
    assert example.clean_letter_ids[0] == ()
    assert example.clean_letter_ids[-1] == ()


# ---------------------------------------------------------------------------
# 12-13. Self-supervision and stable identity
# ---------------------------------------------------------------------------
def test_no_downstream_label_field_exists():
    """Stage-1 is self-supervised: the target is h(x), not a task label."""
    banned = {"label", "labels", "target_label", "y", "gold", "class_id"}
    for name in STAGE1_MODULES:
        for node in ast.walk(tree(name)):
            if isinstance(node, ast.ClassDef):
                for statement in node.body:
                    if isinstance(statement, ast.AnnAssign) and isinstance(
                        statement.target, ast.Name
                    ):
                        assert statement.target.id not in banned, f"{name}: {node.name}"


def test_stage1_does_not_import_downstream_or_baseline_code():
    for name in STAGE1_MODULES:
        body = source(name)
        for banned in ("restore", "baseline", "classification_head", "task_head"):
            assert f"import {banned}" not in body.lower()
        assert "unmark.gates" not in body


def test_stable_sample_id_is_required():
    with pytest.raises(Stage1ContractViolation, match="stable identity"):
        Stage1Example("xin chao", "")
    with pytest.raises(Stage1ContractViolation):
        Stage1Example("xin chao", None)  # type: ignore[arg-type]


def test_corruption_is_reproducible_from_its_key(classifier):
    first = prepared("Tôi đang học nghiên cứu", "s16", classifier)
    second = prepared("Tôi đang học nghiên cứu", "s16", classifier)
    assert first.corruption_rate == second.corruption_rate
    assert first.corrupted_text == second.corrupted_text


def test_reordering_does_not_change_a_sample_corruption(classifier):
    """Corruption is keyed on sample_id, not on position in the corpus."""
    a = prepared("Tôi đang học", "alpha", classifier)
    b = prepared("Tôi đang học", "beta", classifier)
    assert a.corruption_rate != b.corruption_rate
    assert prepared("Tôi đang học", "alpha", classifier).corruption_rate == a.corruption_rate


# ---------------------------------------------------------------------------
# 14-16. OPEN values are not silently defaulted
# ---------------------------------------------------------------------------
def test_lambdas_have_no_defaults():
    with pytest.raises(TypeError):
        ObjectiveWeights()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ObjectiveWeights(lambda_align=1.0)  # type: ignore[call-arg]


def test_lambdas_are_registered_open():
    assert "lambda_align" in OPEN_STAGE1_VALUES
    assert "lambda_clean" in OPEN_STAGE1_VALUES
    for name in ("lambda_align", "lambda_clean"):
        with pytest.raises(UnresolvedStage1Value, match=name):
            require_resolved(name)


def test_lambda_validation_rejects_nonsense():
    with pytest.raises(Stage1ContractViolation):
        ObjectiveWeights(lambda_align=-1.0, lambda_clean=1.0)
    with pytest.raises(Stage1ContractViolation):
        ObjectiveWeights(lambda_align=float("nan"), lambda_clean=1.0)
    with pytest.raises(Stage1ContractViolation, match="identically zero"):
        ObjectiveWeights(lambda_align=0.0, lambda_clean=0.0)


def test_max_length_is_registered_open():
    assert "max_length" in OPEN_STAGE1_VALUES
    assert "truncation_behaviour" in OPEN_STAGE1_VALUES
    with pytest.raises(UnresolvedStage1Value):
        require_resolved("max_length")
    with pytest.raises(UnresolvedStage1Value):
        require_resolved("truncation_behaviour")


def test_truncation_policy_cannot_be_constructed_without_arguments():
    """`TruncationPolicy()` would have selected 'unbounded, fail' for an
    experiment without anyone choosing it."""
    with pytest.raises(TypeError):
        TruncationPolicy()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TruncationPolicy(max_length=64)  # type: ignore[call-arg]


def test_prepare_example_requires_an_explicit_policy_and_visit():
    """Signature inspection: no experiment-facing scientific argument may have
    an omitted default."""
    import inspect

    parameters = inspect.signature(prepare_example).parameters
    for name in ("truncation", "visit", "corruption_policy"):
        assert parameters[name].default is inspect.Parameter.empty, (
            f"{name} has a default; omission would silently select a policy"
        )


def test_explicit_unbounded_is_a_statement_not_a_default():
    policy = TruncationPolicy.unbounded()
    assert policy.max_length is None
    assert policy.on_overflow is OverflowBehaviour.NOT_APPLICABLE
    assert not policy.is_enabled
    assert policy.check(10_000, "seq") is True


def test_explicit_numeric_max_length_is_accepted():
    policy = TruncationPolicy(max_length=8, on_overflow=OverflowBehaviour.FAIL)
    assert policy.is_enabled and policy.max_length == 8
    assert policy.check(8, "seq") is True


def test_skip_is_never_selected_by_omission():
    """SKIP changes the corpus distribution; it must be chosen, never inherited."""
    assert TruncationPolicy.unbounded().on_overflow is not OverflowBehaviour.SKIP
    chosen = TruncationPolicy(max_length=4, on_overflow=OverflowBehaviour.SKIP)
    assert chosen.check(5, "seq") is False


def test_fail_is_never_selected_by_omission():
    assert TruncationPolicy.unbounded().on_overflow is not OverflowBehaviour.FAIL
    chosen = TruncationPolicy(max_length=4, on_overflow=OverflowBehaviour.FAIL)
    with pytest.raises(Stage1ContractViolation, match="does not truncate"):
        chosen.check(5, "seq")


def test_inconsistent_length_and_overflow_combinations_fail_loud():
    with pytest.raises(Stage1ContractViolation, match="nothing can overflow"):
        TruncationPolicy(max_length=None, on_overflow=OverflowBehaviour.FAIL)
    with pytest.raises(Stage1ContractViolation, match="nothing can overflow"):
        TruncationPolicy(max_length=None, on_overflow=OverflowBehaviour.SKIP)
    with pytest.raises(Stage1ContractViolation, match="must state FAIL or SKIP"):
        TruncationPolicy(max_length=8, on_overflow=OverflowBehaviour.NOT_APPLICABLE)


def test_truncation_is_never_implemented():
    assert not hasattr(OverflowBehaviour, "TRUNCATE")
    assert {m.name for m in OverflowBehaviour} == {"FAIL", "SKIP", "NOT_APPLICABLE"}
    for name in STAGE1_MODULES:
        assert "def truncate" not in source(name)


def test_overlong_example_is_skipped_not_trimmed(classifier):
    result = prepare_example(
        Stage1Example("Tôi đang học nghiên cứu tại Đại học Quốc gia", "s17"),
        StubTokenizer(),
        corruption_policy=CorruptionRatePolicy(seed=1),
        truncation=TruncationPolicy(max_length=3, on_overflow=OverflowBehaviour.SKIP),
        visit=0,
        classifier=classifier,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Diagnostic-only values cannot become scientific configuration
# ---------------------------------------------------------------------------
def test_a_scientific_config_cannot_be_built_while_values_are_open():
    """The strongest available guarantee that a diagnostic number does not drift
    into a training run: the scientific configuration does not construct."""
    with pytest.raises(UnresolvedStage1Value, match="lambda_align"):
        Stage1RunConfig(
            purpose=Stage1Purpose.SCIENTIFIC,
            weights=ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
            truncation=TruncationPolicy.unbounded(),
            corruption=CorruptionRatePolicy(seed=1),
        )


def test_a_diagnostic_config_is_allowed_and_labelled():
    config = Stage1RunConfig(
        purpose=Stage1Purpose.DIAGNOSTIC,
        weights=ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
        truncation=TruncationPolicy.unbounded(),
        corruption=CorruptionRatePolicy(seed=1),
        note="real-model wiring dry run",
    )
    payload = config.to_dict()
    assert config.is_diagnostic_only
    assert payload["purpose"] == "DIAGNOSTIC"
    assert payload["diagnostic_only"] is True
    assert payload["values_are_scientific"] is False
    assert payload["resolved_values"] == []


def test_purpose_has_no_default():
    import inspect

    parameters = inspect.signature(Stage1RunConfig).parameters
    assert parameters["purpose"].default is inspect.Parameter.empty
    assert parameters["weights"].default is inspect.Parameter.empty
    assert parameters["truncation"].default is inspect.Parameter.empty


def test_resolved_values_must_name_real_open_items():
    with pytest.raises(Stage1ContractViolation, match="not in the OPEN register"):
        Stage1RunConfig(
            purpose=Stage1Purpose.DIAGNOSTIC,
            weights=ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
            truncation=TruncationPolicy.unbounded(),
            corruption=CorruptionRatePolicy(seed=1),
            resolved_values=frozenset({"not_a_real_open_item"}),
        )


def test_scientific_requirements_are_a_subset_of_the_open_register():
    assert set(SCIENTIFIC_REQUIRED_VALUES) <= set(OPEN_STAGE1_VALUES)
    for name in ("lambda_align", "lambda_clean", "corpus", "max_length"):
        assert name in SCIENTIFIC_REQUIRED_VALUES


def test_corpus_is_not_hardcoded():
    """Structural: no dataset import, no download call, no URL *literal*.

    Checked over the AST rather than the raw text, because the modules
    legitimately *discuss* datasets in prose when recording what is still open.
    """
    for name in STAGE1_MODULES:
        assert not imported(name) & {"datasets", "urllib", "requests", "huggingface_hub"}, name
        assert not called_names(name) & {"load_dataset", "urlopen", "get", "hf_hub_download"} - {"get"}, name
        for node in ast.walk(tree(name)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "http://" not in node.value and "https://" not in node.value, name
    assert "corpus" in OPEN_STAGE1_VALUES


def test_corruption_schedule_and_scope_policy_are_now_LOCKED():
    """Both were OPEN until D-S1B-003/004 decided them.

    The old assertion that they are OPEN is not loosened, it is **replaced**:
    the register must now say they are locked, and must say what to.
    """
    from unmark.stage1.contracts import LOCKED_STAGE1_VALUES

    assert "corruption_redraw_schedule" not in OPEN_STAGE1_VALUES
    assert "letter_dropout_rate" not in OPEN_STAGE1_VALUES
    assert "per visit" in LOCKED_STAGE1_VALUES["corruption_redraw_schedule"]
    scope_policy = LOCKED_STAGE1_VALUES["corruption_scope_policy"]
    assert "0.25" in scope_policy and "domain-separated" in scope_policy
    # the corpus revision pin is the remaining corpus-side OPEN item
    assert "corpus_revision_pin" in OPEN_STAGE1_VALUES


def test_locked_values_are_recorded_as_locked():
    assert "p ~ U(0,1)" in LOCKED_STAGE1_VALUES["corruption_rate_distribution"]
    assert "cosine" in LOCKED_STAGE1_VALUES["distance"]
    assert "corruption_rate_distribution" not in OPEN_STAGE1_VALUES


# ---------------------------------------------------------------------------
# 17. Corruption sampling: locked distribution, explicit key
# ---------------------------------------------------------------------------
def test_rate_is_drawn_from_a_keyed_digest_not_a_global_rng():
    body = source("unmark/stage1/contracts.py")
    assert "blake2b" in body
    assert "import random" not in body
    calls = called_names("unmark/stage1/contracts.py")
    assert not calls & {"seed", "uniform", "random", "randint"}


def test_rate_is_in_the_unit_interval_and_spreads():
    policy = CorruptionRatePolicy(seed=11)
    rates = [policy.rate_for(f"sample-{i}") for i in range(200)]
    assert all(0.0 <= r < 1.0 for r in rates)
    assert min(rates) < 0.1 and max(rates) > 0.9, "draw does not span the unit interval"
    assert abs(sum(rates) / len(rates) - 0.5) < 0.1


def test_visit_is_explicit_with_no_implied_schedule():
    policy = CorruptionRatePolicy(seed=11)
    assert policy.rate_for("s", 0) != policy.rate_for("s", 1)
    assert policy.rate_for("s", 0) == policy.rate_for("s", 0)


def test_the_default_policy_is_the_locked_mixture_not_a_pinned_scope():
    """Replaces the old "TONE by default" assertion, which encoded the defect.

    A run-global `TONE` scope is exactly what left STRIP-ALL with zero training
    support, so the default is now the locked mixture and there is no
    run-global scope to read at all.
    """
    from unmark.stage1.contracts import PI_STRIP

    policy = CorruptionRatePolicy(seed=1)
    assert policy.is_locked_mixture
    assert policy.pi_strip == PI_STRIP == 0.25
    assert policy.forced_scope is None

    # asking for a single run-global scope is refused rather than answered
    with pytest.raises(Stage1ContractViolation, match="no run-global scope"):
        policy.scope

    # both scopes are actually produced
    scopes = {policy.scope_for(f"d{i}", 0) for i in range(200)}
    assert scopes == {"TONE", "TONE_AND_LETTER"}

    with pytest.raises(Stage1ContractViolation, match="unsupported corruption scope"):
        CorruptionRatePolicy(seed=1, forced_scope="EVERYTHING")


def test_a_pinned_scope_is_diagnostic_only_and_cannot_go_scientific():
    from unmark.stage1.contracts import (
        ObjectiveWeights,
        OverflowBehaviour,
        Stage1Purpose,
        Stage1RunConfig,
        TruncationPolicy,
    )

    pinned = CorruptionRatePolicy(seed=1, forced_scope="TONE")
    assert pinned.scope == "TONE"          # a pinned policy may be read
    assert not pinned.is_locked_mixture

    with pytest.raises(Stage1ContractViolation, match="locked corruption mixture"):
        Stage1RunConfig(
            purpose=Stage1Purpose.SCIENTIFIC,
            weights=ObjectiveWeights(1.0, 1.0),
            truncation=TruncationPolicy(256, OverflowBehaviour.FAIL),
            corruption=pinned,
            resolved_values=frozenset(OPEN_STAGE1_VALUES),
        )


# ---------------------------------------------------------------------------
# 18-20. Static guards: no training, no duplicated machinery
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", STAGE1_MODULES)
def test_no_optimizer_or_training_loop(name):
    calls = called_names(name)
    assert not calls & {"step", "zero_grad", "save_pretrained", "save", "backward"}
    assert not imported(name) & {"optim", "datasets", "wandb", "tensorboard"}
    attributes = {n.attr for n in ast.walk(tree(name)) if isinstance(n, ast.Attribute)}
    assert not attributes & {"AdamW", "SGD", "lr_scheduler", "GradScaler"}


@pytest.mark.parametrize("name", STAGE1_MODULES)
def test_no_second_pooling_or_position_implementation(name):
    body = source(name)
    assert "def masked_mean" not in body, f"{name} defines a second pooling"
    assert "def create_position_ids" not in body
    assert "cumsum" not in body, f"{name} looks like a second position-id rule"


def test_objective_reuses_the_locked_pooling():
    assert "masked_mean_non_special" in source(OBJECTIVE)
    assert "from unmark.modeling.pooling import" in source(OBJECTIVE)


def test_objective_omits_position_ids_so_the_wrapper_derives_them():
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name == "adapted_representation":
            body = ast.unparse(node)
            assert "position_ids" not in body.replace("`position_ids`", "")


@pytest.mark.parametrize("name", STAGE1_MODULES)
def test_no_hardcoded_backbone_constants(name):
    literals = [
        node.value
        for node in ast.walk(tree(name))
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    ]
    assert 768 not in literals, f"{name} hardcodes a hidden size"
    assert "pad_token_id = 1" not in source(name)
    assert "vncorenlp" not in source(name).lower()


def test_no_family_wide_roberta_permission_is_reintroduced():
    for name in STAGE1_MODULES:
        body = source(name).lower()
        assert "verified_position_families" not in body
        assert 'model_type == "roberta"' not in body


def test_no_token_level_hidden_state_matching():
    """§4.6 defers per-token alignment; the branches do not share a token grid."""
    body = source(OBJECTIVE)
    for banned in ("[:, :min", "min(l_ref", "truncate", "align_tokens"):
        assert banned not in body
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name == "representation_distance":
            unparsed = ast.unparse(node)
            assert "dim=-1" in unparsed


# ---------------------------------------------------------------------------
# 21. Gradient contract, statically
# ---------------------------------------------------------------------------
def _no_grad_blocks(function: ast.FunctionDef) -> list[ast.With]:
    """`with torch.no_grad():` statements inside a function.

    Structural, so a docstring saying the branch is *not* under `no_grad` cannot
    trip the check -- an earlier version of this test matched raw text and
    flagged its own documentation.
    """
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any("no_grad" in ast.unparse(item.context_expr) for item in node.items)
    ]


def test_only_the_reference_branch_uses_no_grad():
    seen = {}
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "adapted_representation", "reference_representation", "forward"
        }:
            seen[node.name] = _no_grad_blocks(node)
    assert not seen["adapted_representation"], "the adapted branch is under no_grad"
    assert not seen["forward"], "the objective forward is under no_grad"
    assert seen["reference_representation"], "the reference target must not build a graph"


def test_no_adapted_representation_is_detached():
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "adapted_representation",
            "forward",
        }:
            body = ast.unparse(node)
            assert ".detach()" not in body.replace("self.loss.detach()", ""), node.name


def test_only_diagnostics_detach():
    """`to_dict` may detach for logging; the loss path may not."""
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name == "to_dict":
            assert "detach" in ast.unparse(node)


def test_objective_delegates_train_mode_to_the_wrapper():
    for node in ast.walk(tree(OBJECTIVE)):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            body = ast.unparse(node)
            assert "self.unmark_encoder.train(mode)" in body
            assert "return self" in body


# ---------------------------------------------------------------------------
# 22. Torch-gated
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


def build_stack(d: int = 16):
    from torch import nn

    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.config import AdapterConfig
    from unmark.stage1.objective import Stage1Objective

    import types as _types

    class RobertaLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(64, d, padding_idx=1)
            self.proj = nn.Linear(d, d)
            self.embeddings = _types.SimpleNamespace(padding_idx=1, word_embeddings=self.embed)
            self.config = _types.SimpleNamespace(
                model_type="roberta", pad_token_id=1, _name_or_path="vinai/phobert-base"
            )

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                    position_ids=None, **_):
            hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
            return self.proj(hidden)

    RobertaLike.__name__ = "RobertaModel"
    torch.manual_seed(0)
    encoder = RobertaLike()
    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=d))
    wrapper = UnmarkEncoder(encoder, adapter)
    objective = Stage1Objective(wrapper, ObjectiveWeights(lambda_align=0.7, lambda_clean=0.3))
    return encoder, adapter, wrapper, objective


def synthetic_batch(reference_len: int = 7, base_len: int = 5, batch: int = 2):
    """Deliberately unequal reference and base lengths.

    **The orthography channels follow `base_len`, not a hard-coded 5.**
    C24-1-R2B, Group D: they previously stayed at length 5 while `base_len`
    moved, so `synthetic_batch(base_len=4)` produced a batch that violated the
    adapted-grid contract -- base ids at L=4 beside tone/letter channels at
    L=5 -- and the adapter failed inside `torch.cat`. The test then appeared to
    say "unequal reference/base lengths do not work", which is the opposite of
    the Stage-1 contract.

    The contract these two lengths express:

    * `reference_len` is independent -- the clean reference branch may tokenize
      to a different length than the adapted branch, and that is the point;
    * everything on the **adapted/base grid** -- base ids, tone ids, tone mask,
      letter ids, letter mask -- must share one length.
    """
    if base_len < 3:
        raise ValueError("base_len must leave room for two special tokens and content")

    def base_grid_tone(fill: int) -> list[int]:
        return [-1] + [fill] * (base_len - 2) + [-1]

    def base_grid_mask() -> list[bool]:
        return [False] + [True] * (base_len - 2) + [False]

    def base_grid_letter(fill: int) -> list[list[int]]:
        return [[-1]] + [[fill]] * (base_len - 2) + [[-1]]

    def base_grid_letter_mask() -> list[list[bool]]:
        return [[False]] + [[True]] * (base_len - 2) + [[False]]

    return {
        "reference_input_ids": torch.randint(4, 60, (batch, reference_len)),
        "reference_attention_mask": torch.ones(batch, reference_len, dtype=torch.long),
        "reference_special_tokens_mask": torch.tensor(
            [[1] + [0] * (reference_len - 2) + [1]] * batch
        ),
        "base_input_ids": torch.randint(4, 60, (batch, base_len)),
        "base_attention_mask": torch.ones(batch, base_len, dtype=torch.long),
        "base_special_tokens_mask": torch.tensor([[1] + [0] * (base_len - 2) + [1]] * batch),
        "clean_tone_ids": torch.tensor([base_grid_tone(1)] * batch),
        "clean_tone_mask": torch.tensor([base_grid_mask()] * batch),
        "clean_letter_ids": torch.tensor([base_grid_letter(2)] * batch),
        "clean_letter_mask": torch.tensor([base_grid_letter_mask()] * batch),
        "corrupt_tone_ids": torch.tensor([base_grid_tone(5)] * batch),
        "corrupt_tone_mask": torch.tensor([base_grid_mask()] * batch),
        "corrupt_letter_ids": torch.tensor([base_grid_letter(0)] * batch),
        "corrupt_letter_mask": torch.tensor([base_grid_letter_mask()] * batch),
    }


@requires_torch
def test_runtime_distance_shape_and_extremes():
    from unmark.stage1.objective import representation_distance

    a = torch.randn(4, 8)
    assert representation_distance(a, a.clone()).shape == (4,)
    assert torch.allclose(representation_distance(a, a.clone()), torch.zeros(4), atol=1e-6)
    assert torch.allclose(representation_distance(a, -a), torch.full((4,), 2.0), atol=1e-6)


@requires_torch
def test_runtime_distance_rejects_token_level_tensors():
    from unmark.stage1.objective import representation_distance

    with pytest.raises(Stage1ContractViolation, match=r"\[B, d\]"):
        representation_distance(torch.randn(2, 5, 8), torch.randn(2, 5, 8))


@requires_torch
def test_runtime_loss_components_are_scalars_and_combine_correctly():
    _, _, _, objective = build_stack()
    result = objective(synthetic_batch())
    assert result.loss.shape == () and result.loss_align.shape == () and result.loss_clean.shape == ()
    assert result.distance_align_per_example.shape == (2,)
    expected = 0.7 * result.loss_align + 0.3 * result.loss_clean
    assert torch.allclose(result.loss, expected)


@requires_torch
def test_runtime_aggregation_is_mean_not_sum():
    _, _, _, objective = build_stack()
    result = objective(synthetic_batch(batch=2))
    assert torch.allclose(result.loss_align, result.distance_align_per_example.mean())
    assert not torch.allclose(result.loss_align, result.distance_align_per_example.sum())


@requires_torch
def test_runtime_unequal_reference_and_base_lengths_work():
    _, _, _, objective = build_stack()
    result = objective(synthetic_batch(reference_len=11, base_len=4))
    assert torch.isfinite(result.loss)


@requires_torch
def test_runtime_reference_does_not_require_grad_but_adapted_do():
    _, _, _, objective = build_stack()
    batch = synthetic_batch()
    h_ref = objective.reference_representation(
        batch["reference_input_ids"],
        batch["reference_attention_mask"],
        batch["reference_special_tokens_mask"],
    )
    assert not h_ref.requires_grad
    h_clean = objective.adapted_representation(
        batch["base_input_ids"], batch["base_attention_mask"], batch["base_special_tokens_mask"],
        batch["clean_tone_ids"], batch["clean_tone_mask"],
        batch["clean_letter_ids"], batch["clean_letter_mask"],
    )
    h_corrupt = objective.adapted_representation(
        batch["base_input_ids"], batch["base_attention_mask"], batch["base_special_tokens_mask"],
        batch["corrupt_tone_ids"], batch["corrupt_tone_mask"],
        batch["corrupt_letter_ids"], batch["corrupt_letter_mask"],
    )
    assert h_clean.requires_grad and h_corrupt.requires_grad


@requires_torch
def test_runtime_channels_actually_influence_the_two_adapted_branches():
    """If the channels were ignored, both adapted branches would coincide."""
    _, _, _, objective = build_stack()
    batch = synthetic_batch()
    h_clean = objective.adapted_representation(
        batch["base_input_ids"], batch["base_attention_mask"], batch["base_special_tokens_mask"],
        batch["clean_tone_ids"], batch["clean_tone_mask"],
        batch["clean_letter_ids"], batch["clean_letter_mask"],
    )
    h_corrupt = objective.adapted_representation(
        batch["base_input_ids"], batch["base_attention_mask"], batch["base_special_tokens_mask"],
        batch["corrupt_tone_ids"], batch["corrupt_tone_mask"],
        batch["corrupt_letter_ids"], batch["corrupt_letter_mask"],
    )
    assert not torch.allclose(h_clean, h_corrupt)


@requires_torch
def test_runtime_loss_backpropagates_into_the_adapter_only():
    encoder, adapter, _, objective = build_stack()
    objective.train()
    result = objective(synthetic_batch())
    result.loss.backward()

    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), name
    assert any(float(p.grad.abs().sum()) != 0 for p in adapter.parameters())
    assert all(p.grad is None for p in encoder.parameters()), "frozen encoder got gradients"
    assert all(not p.requires_grad for p in encoder.parameters())


@requires_torch
def test_runtime_encoder_stays_eval_when_the_objective_trains():
    encoder, adapter, _, objective = build_stack()
    objective.train()
    assert objective.training and adapter.training
    assert not encoder.training, "Stage-1 train() reactivated frozen-encoder dropout"
    objective.eval()
    assert not adapter.training and not encoder.training
    objective.train()
    assert adapter.training and not encoder.training


@requires_torch
def test_runtime_zero_content_branch_fails_loud():
    from unmark.modeling.contracts import Stage1PoolingError

    _, _, _, objective = build_stack()
    batch = synthetic_batch()
    batch["reference_special_tokens_mask"] = torch.ones_like(
        batch["reference_special_tokens_mask"]
    )
    with pytest.raises(Stage1PoolingError):
        objective(batch)


@requires_torch
def test_runtime_missing_batch_field_fails_loud():
    _, _, _, objective = build_stack()
    batch = synthetic_batch()
    del batch["corrupt_tone_ids"]
    with pytest.raises(Stage1ContractViolation, match="missing fields"):
        objective(batch)


@requires_torch
def test_runtime_non_finite_component_fails_loud(monkeypatch):
    import unmark.stage1.objective as objective_module

    _, _, _, objective = build_stack()
    monkeypatch.setattr(
        objective_module,
        "representation_distance",
        lambda a, b: torch.full((a.shape[0],), float("nan")),
    )
    with pytest.raises(Stage1ContractViolation, match="not finite"):
        objective(synthetic_batch())


@requires_torch
def test_runtime_result_dict_carries_no_raw_text():
    _, _, _, objective = build_stack()
    payload = objective(synthetic_batch()).to_dict()
    assert set(payload) >= {"loss", "loss_align", "loss_clean", "lambda_align", "lambda_clean"}
    assert all(isinstance(v, (int, float)) for v in payload.values())


@requires_torch
def test_runtime_collation_produces_a_usable_batch(classifier):
    from unmark.stage1.data import collate_stage1_batch

    tokenizer = StubTokenizer(fragment_marked=True)
    examples = [
        prepared("Tôi đang học nghiên cứu", "c1", classifier, tokenizer=tokenizer),
        prepared("Chào bạn", "c2", classifier, tokenizer=tokenizer),
    ]
    batch = collate_stage1_batch(examples, pad_token_id=1)
    assert batch["reference_input_ids"].shape[0] == 2
    assert batch["base_input_ids"].shape[0] == 2
    assert batch["clean_letter_ids"].dim() == 3
    assert batch["clean_tone_mask"].dtype == torch.bool
    assert torch.equal(batch["base_input_ids"], batch["base_input_ids"])
    assert batch["sample_ids"] == ["c1", "c2"]
