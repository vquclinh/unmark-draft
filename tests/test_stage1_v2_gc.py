"""**V2-GC** -- the Stage-1 grid-consistency objective candidate.

Three tiers, matching `test_stage1.py`:

* static/AST guards that run in the ML-free local venv and pin the *structural*
  claims -- where the stop-gradient is, that no forward was added, that no model
  parameter was added, that nothing reaches checkpoint selection;
* torch-free contract tests for the locked weight and the provenance identity,
  which is where "an old checkpoint cannot silently resume as V2-GC" actually
  lives;
* torch-gated numerics that skip cleanly here and run on Colab.

**Nothing here trains.** Two synthetic backwards run in torch-gated tests to
verify gradient routing; no optimizer is constructed and no parameter is updated.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.contracts import (
    GRID_CONSISTENCY_OBJECTIVE,
    HISTORICAL_OBJECTIVE,
    GridConsistencyWeights,
    ObjectiveIdentity,
    ObjectiveWeights,
    Stage1ContractViolation,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORRUPTION_SEED,
    FUSION_IDS,
    GRID_CONSISTENCY_OBJECTIVE_ID,
    HISTORICAL_FUSION_ID,
    HISTORICAL_OBJECTIVE_ID,
    RELATION_LOSS,
    RELATION_METRIC,
    RELATION_SPACE,
    INITIAL_MAX_UPDATES,
    LAMBDA_GRID,
    OBJECTIVE_IDS,
    V2_GC_LEARNING_RATE,
    V2_GC_MAX_UPDATES,
    V2_GC_R,
    V2_GC_RUN_SEED,
    V2_GC_STAGE,
    adapter_init_seed,
    lambda_grid_for,
    lambdas_for_r,
)
from unmark.stage1.selection import ValidationPoint, v2_gc_schedule, select_checkpoint
from unmark.stage1.trainer import (
    RunProvenance,
    TrainerContractViolation,
    checkpoint_payload,
    loss_telemetry,
    verify_checkpoint,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
GRID = "unmark/stage1/objective_grid.py"
OBJECTIVE = "unmark/stage1/objective.py"
VALIDATION = "unmark/stage1/validation.py"
SELECTION = "unmark/stage1/selection.py"


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    return ast.parse(source(name))


def function(name: str, module: str = GRID) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree(module))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def calls_in(node: ast.AST) -> list[str]:
    """Every called name, WITH multiplicity -- "how many times" is the contract."""
    return [
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
    ]


LOCKED_IDENTITY_TOKENS = frozenset(
    {*OBJECTIVE_IDS, *FUSION_IDS, RELATION_SPACE, RELATION_METRIC, RELATION_LOSS}
)
"""Every string a loss payload is allowed to contain.

The payload's ONLY legitimate strings are locked identity tokens. Anything else
-- a chunk id, a corpus fragment, a file path, a tokenizer artifact -- is raw
text that must never travel into a log, so the check is a closed vocabulary
rather than `isinstance(value, str)`.
"""

JSON_SCALARS = (str, int, float, bool, type(None))
"""What a serialized Stage-1 payload may hold. `bool` is listed explicitly even
though it is an `int` subclass, so the permitted set is readable."""


def assert_json_safe(value, path: str = "payload") -> None:
    """Recursively refuse anything a JSON artifact could not faithfully carry.

    Explicitly rejects `torch.Tensor` and any other runtime object: a payload
    that holds a live tensor keeps an autograd graph and a CUDA allocation alive
    for as long as the log line does, and silently fails to serialise.

    Strings are additionally constrained to `LOCKED_IDENTITY_TOKENS`, which is
    what keeps the "no raw text" guarantee from degrading into "no non-strings".
    """
    if torch is not None and isinstance(value, torch.Tensor):
        raise AssertionError(f"{path} is a torch.Tensor; payloads carry numbers, not tensors")
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), f"{path} has a non-string key {key!r}"
            assert_json_safe(item, f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_json_safe(item, f"{path}[{index}]")
        return
    assert isinstance(value, JSON_SCALARS), (
        f"{path} is {type(value).__name__}, which is not a JSON scalar"
    )
    if isinstance(value, float):
        assert value == value and value not in (float("inf"), float("-inf")), (
            f"{path} is {value!r}: a payload must not carry NaN or Inf"
        )
    if isinstance(value, str):
        assert value in LOCKED_IDENTITY_TOKENS, (
            f"{path} carries the string {value!r}, which is not a locked identity "
            "token. Payloads must never contain raw text."
        )


def code_only(name: str) -> str:
    """Module source with every comment and string literal removed.

    Prose that *describes* a rule must not be able to break -- or satisfy -- a
    test that checks the rule. Without this, a comment saying "no downstream
    Macro-F1 enters here" trips a grep for "macro-f1", and a docstring saying
    "there is deliberately no --lambda-grid flag" trips a grep for the flag.
    The same trap `_no_grad_blocks` in `test_stage1.py` was rewritten to avoid.
    """
    import io
    import tokenize

    kept = []
    readline = io.StringIO(source(name)).readline
    for token in tokenize.generate_tokens(readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def cli_options(name: str) -> set[str]:
    """Every `--flag` the argparse tree in `name` actually declares.

    Structural: the option strings handed to `add_argument`, not a grep. A
    docstring promising that a flag does not exist is not evidence that it does
    not exist -- and a grep for the flag would be tripped by that very promise.
    """
    options: set[str] = set()
    for node in ast.walk(tree(name)):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("-"):
                        options.add(argument.value)
    return options


def body_without_docstring(node: ast.FunctionDef) -> str:
    """Unparsed source with the docstring removed.

    Prose that *describes* a rule must not be able to satisfy a test that checks
    the rule -- the same trap `_no_grad_blocks` in `test_stage1.py` was rewritten
    to avoid.
    """
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(statement) for statement in body)


# ===========================================================================
# 1. The locked weight: lambda_grid is not a hyperparameter
# ===========================================================================
def test_lambda_grid_is_locked_at_one():
    assert LAMBDA_GRID == 1.0
    assert lambda_grid_for(GRID_CONSISTENCY_OBJECTIVE_ID) == 1.0


def test_the_historical_objective_has_no_grid_term_at_all():
    """`None`, never `0.0`: absence, not a term that was weighted away."""
    assert lambda_grid_for(HISTORICAL_OBJECTIVE_ID) is None
    assert HISTORICAL_OBJECTIVE.to_dict()["lambda_grid"] is None


def test_the_weights_refuse_any_other_lambda_grid():
    for rejected in (0.0, 0.5, 2.0, 1.0000001):
        with pytest.raises(Stage1ContractViolation, match="LOCKED"):
            GridConsistencyWeights(
                lambda_align=1.0, lambda_clean=1.0, lambda_grid=rejected
            )


def test_lambda_grid_is_not_a_command_line_flag():
    """It cannot be swept, because no flag accepts it and nothing would take one."""
    declared = cli_options("scripts/stage1_runner.py")
    for forbidden in ("--lambda-grid", "--lambda_grid", "--grid-weight",
                      "--objective", "--objective-id", "--lr", "--r"):
        assert forbidden not in declared, f"the runner declares {forbidden}"
    # And the V2-GC subcommand takes exactly the shared corpus-consumer options
    # every historical stage takes -- no scientific argument of its own.
    assert "--prepared-corpus" in declared and "--output-dir" in declared


def test_the_objective_id_set_is_closed():
    """Closed, and the grid objective is exactly one member of it."""
    assert OBJECTIVE_IDS[:2] == (HISTORICAL_OBJECTIVE_ID, GRID_CONSISTENCY_OBJECTIVE_ID)
    assert GRID_CONSISTENCY_OBJECTIVE_ID in OBJECTIVE_IDS
    assert len(set(OBJECTIVE_IDS)) == len(OBJECTIVE_IDS)
    with pytest.raises(Stage1ContractViolation, match="closed set"):
        ObjectiveIdentity("grid-consistency-v2")


def test_only_the_grid_objective_carries_a_grid_weight():
    """Every other registered objective must report `None`, never 0.0."""
    for objective_id in OBJECTIVE_IDS:
        weight = ObjectiveIdentity(objective_id).lambda_grid
        if objective_id == GRID_CONSISTENCY_OBJECTIVE_ID:
            assert weight == 1.0
        else:
            assert weight is None, objective_id


def test_the_v2_gc_weights_are_the_historical_ones_at_r_one():
    """`lambda_align = lambda_clean = 1.0`: the closed campaign's selected ratio."""
    lambda_align, lambda_clean = lambdas_for_r(V2_GC_R)
    assert (lambda_align, lambda_clean) == (1.0, 1.0)
    weights = GridConsistencyWeights(lambda_align=lambda_align, lambda_clean=lambda_clean)
    assert weights.to_dict() == {
        "lambda_align": 1.0,
        "lambda_clean": 1.0,
        "lambda_grid": 1.0,
    }


# ===========================================================================
# 2. The run plan: every value is an already-closed one, none retuned
# ===========================================================================
def test_the_v2_gc_plan_is_exactly_the_locked_values():
    (planned,) = v2_gc_schedule()
    assert planned.stage == V2_GC_STAGE == "v2_gc"
    assert planned.learning_rate == V2_GC_LEARNING_RATE == 1e-4
    assert planned.r == V2_GC_R == 1.0
    assert planned.seed == V2_GC_RUN_SEED == 36930
    assert adapter_init_seed(planned.seed) == 51800
    assert CORRUPTION_SEED == 35422
    assert V2_GC_MAX_UPDATES == INITIAL_MAX_UPDATES == 20_000


def test_v2_gc_is_one_run_and_does_not_grow_the_locked_campaign():
    from unmark.stage1.protocol import TOTAL_NOMINAL_RUNS
    from unmark.stage1.selection import total_planned_runs

    assert len(v2_gc_schedule()) == 1
    assert total_planned_runs() == TOTAL_NOMINAL_RUNS == 11


def test_v2_gc_takes_no_scientific_arguments():
    """A candidate whose values can be passed in is a candidate nobody locked."""
    node = function("v2_gc_schedule", SELECTION)
    assert not node.args.args and not node.args.kwonlyargs


# ===========================================================================
# 3. Structure: zero new parameters, zero new encoder forwards
# ===========================================================================
def test_the_candidate_defines_no_model_parameters():
    body = source(GRID)
    for forbidden in (
        "nn.Parameter", "nn.Embedding", "nn.Linear", "nn.LayerNorm",
        "register_parameter", "register_buffer",
    ):
        assert forbidden not in body, f"V2-GC must add no parameters, found {forbidden}"


def test_the_adapter_size_is_untouched():
    assert ADAPTER_TRAINABLE_PARAMETERS == 3_551_232


def test_the_candidate_does_not_touch_the_adapter_or_the_backbone():
    """One isolated change. The model code is not in the diff surface at all."""
    import re

    body = code_only(GRID)
    for forbidden in (
        "OrthographyInputAdapter", "AdapterConfig", "AutoModel", "AutoTokenizer",
        "gate", "tone_embedding", "letter_embedding", "position_ids",
        "reset_gate_parameters", "convex_combination",
    ):
        assert not re.search(rf"\b{re.escape(forbidden)}\b", body), (
            f"V2-GC must not reach {forbidden}"
        )


def test_the_forward_runs_exactly_three_branches_statically():
    """One reference forward and two adapted forwards. Counted, not described."""
    node = function("forward")
    called = calls_in(node)
    assert called.count("reference_representation") == 1
    assert called.count("adapted_branch") == 2
    # The pooled-only accessor would be a SECOND forward of the same branch.
    assert "adapted_representation" not in called


def test_the_grid_term_runs_no_forward_of_its_own():
    node = function("token_grid_distance")
    called = calls_in(node)
    for forbidden in ("adapted_branch", "adapted_representation",
                      "reference_representation", "unmark_encoder", "encoder"):
        assert forbidden not in called, f"L_grid must not call {forbidden}"


def test_the_historical_adapted_branch_still_omits_position_ids():
    """The wrapper derives them (D-B4B-002); Stage-1 does not reimplement that."""
    for name in ("adapted_branch", "adapted_representation"):
        assert "position_ids" not in body_without_docstring(function(name, OBJECTIVE))


# ===========================================================================
# 4. The stop-gradient, structurally
# ===========================================================================
def test_the_clean_target_is_detached_in_the_loss_path():
    body = body_without_docstring(function("token_grid_distance"))
    assert "hidden_clean_target.detach()" in body, (
        "L_grid must detach the clean hidden states; without it the term could be "
        "minimised by pulling the clean branch toward the corrupted one"
    )


def test_the_corrupt_branch_is_never_detached():
    for name in ("token_grid_distance", "forward"):
        body = body_without_docstring(function(name))
        assert "hidden_corrupt.detach()" not in body, name
        assert "h_adapt_corrupt.detach()" not in body, name


def test_only_diagnostics_detach_elsewhere():
    """`to_dict` may detach for logging; the loss path may not."""
    node = function("to_dict")
    assert "detach" in ast.unparse(node)
    assert "detach" not in body_without_docstring(function("_require_shared_grid"))


def test_the_grid_term_never_touches_the_vanilla_reference_branch():
    """§4.6 stands: the reference has its own tokenization and no shared grid."""
    node = function("forward")
    for statement in ast.walk(node):
        if isinstance(statement, ast.Call):
            name = getattr(statement.func, "id", None) or getattr(
                statement.func, "attr", None
            )
            if name == "token_grid_distance":
                passed = ast.unparse(statement)
                assert "reference" not in passed, (
                    "L_grid was handed a reference-branch tensor; those branches do "
                    "not share a token grid"
                )
                assert "h_ref" not in passed


def test_the_grid_term_reuses_the_locked_content_mask():
    """attention_mask == 1 AND special_tokens_mask == 0, defined once (D-B4A-006)."""
    body = source(GRID)
    assert "from unmark.modeling.pooling import content_mask" in body
    assert "def content_mask" not in body, "a second definition of a content token"
    assert "def masked_mean" not in body


def test_the_grid_term_reuses_the_locked_cosine_epsilon():
    body = source(GRID)
    assert "COSINE_EPS" in body
    assert "eps=COSINE_EPS" in body
    assert "1e-8" not in body, "the epsilon must be imported, never retyped"


# ===========================================================================
# 5. Nothing reaches checkpoint selection
# ===========================================================================
@pytest.mark.parametrize("module", [VALIDATION, SELECTION])
def test_the_grid_term_never_reaches_validation_or_selection(module):
    """D-S1B-001/004 unchanged: the held-out UNLABELED distance selects, alone."""
    body = code_only(module)
    for forbidden in ("loss_grid", "token_grid_distance", "objective_grid",
                      "GridConsistencyObjective", "mean_distance_grid"):
        assert forbidden not in body, f"{module} must not reach {forbidden}"


def test_the_validation_point_schema_is_unchanged():
    fields = tuple(f.name for f in dataclasses.fields(ValidationPoint))
    assert fields == ("update", "distances", "d_clean")


def test_the_selection_rule_is_unchanged():
    points = [
        ValidationPoint(0, {"FULL": .5, "P50": .5, "P100": .5, "STRIP_ALL": .9}, .5),
        ValidationPoint(500, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .4}, .1),
        ValidationPoint(1000, {"FULL": .1, "P50": .2, "P100": .3, "STRIP_ALL": .8}, .1),
    ]
    assert select_checkpoint(points).update == 500


def test_no_downstream_label_or_score_enters_the_candidate():
    """D-S1B-001: Stage-1 selection reads held-out UNLABELED signals only."""
    for module in (GRID, SELECTION, VALIDATION, "unmark/stage1/execute.py"):
        body = code_only(module).lower()
        for forbidden in ("uitvsfc", "uit_vsfc", "macro_f1", "f1_score", "labels"):
            assert forbidden not in body, f"{module} reaches {forbidden}"


# ===========================================================================
# 6. Provenance: V2-GC cannot masquerade as, or resume from, UNMARK-A/B
# ===========================================================================
def historical_provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=V2_GC_RUN_SEED,
        init_seed=adapter_init_seed(V2_GC_RUN_SEED),
        corruption_seed=CORRUPTION_SEED,
        learning_rate=V2_GC_LEARNING_RATE,
        r=V2_GC_R,
        corpus_manifest_digest="d" * 64,
        repository_head="a" * 40,
    )
    base.update(overrides)
    return RunProvenance(**base)


def payload_for(provenance: RunProvenance) -> dict:
    return checkpoint_payload(
        provenance=provenance, adapter_state={}, optimizer_state={}, global_update=500,
        sampler_state={"cursor": 3, "visit": 1}, cap=INITIAL_MAX_UPDATES,
        budget_limited=False, points=[],
    )


def test_the_default_objective_is_the_historical_one():
    assert historical_provenance().objective is HISTORICAL_OBJECTIVE


def test_the_objective_is_recorded_in_the_artifact():
    """HEAD, objective identity and lambda_grid all recoverable from the payload."""
    v2gc = historical_provenance(objective=GRID_CONSISTENCY_OBJECTIVE)
    recorded = payload_for(v2gc)["provenance"]
    assert recorded["objective"]["objective_id"] == "grid-consistency-v1"
    assert recorded["objective"]["lambda_grid"] == 1.0
    # C3 has no relational term; those fields must be honest absence, not zero.
    assert recorded["objective"]["lambda_grd"] is None
    assert recorded["objective"]["relational"] is None
    assert recorded["repository_head"] == "a" * 40
    assert recorded["lambda_align"] == recorded["lambda_clean"] == 1.0


def test_a_v2_gc_checkpoint_cannot_be_read_as_historical():
    v2gc = payload_for(historical_provenance(objective=GRID_CONSISTENCY_OBJECTIVE))
    with pytest.raises(TrainerContractViolation, match="objective"):
        verify_checkpoint(v2gc, historical_provenance())


def test_v2_gc_cannot_resume_from_a_historical_checkpoint():
    """Same seeds, same LR, same r, same corpus, same HEAD. Different experiment."""
    historical = payload_for(historical_provenance())
    with pytest.raises(TrainerContractViolation, match="objective"):
        verify_checkpoint(historical, historical_provenance(
            objective=GRID_CONSISTENCY_OBJECTIVE
        ))


def test_a_legacy_checkpoint_without_the_field_still_verifies_as_historical():
    """The frozen UNMARK-A/B payloads predate `objective`; the gate still passes them."""
    legacy = payload_for(historical_provenance())
    del legacy["provenance"]["objective"]
    verify_checkpoint(legacy, historical_provenance())  # must not raise


def test_a_legacy_checkpoint_still_cannot_resume_as_v2_gc():
    """Absence reads as HISTORICAL -- which is a refusal, not a free pass."""
    legacy = payload_for(historical_provenance())
    del legacy["provenance"]["objective"]
    with pytest.raises(TrainerContractViolation, match="objective"):
        verify_checkpoint(legacy, historical_provenance(
            objective=GRID_CONSISTENCY_OBJECTIVE
        ))


def test_a_payload_naming_v2_gc_with_the_wrong_weight_is_refused():
    """Hand-edited or corrupted: the id and the weight are compared together."""
    v2gc = historical_provenance(objective=GRID_CONSISTENCY_OBJECTIVE)
    payload = payload_for(v2gc)
    payload["provenance"]["objective"] = {
        "objective_id": "grid-consistency-v1",
        "lambda_grid": 0.5,
    }
    with pytest.raises(TrainerContractViolation, match="objective"):
        verify_checkpoint(payload, v2gc)


def test_the_finalist_verifier_covers_the_new_field():
    from unmark.stage1.finalists import VERIFIED_PROVENANCE_FIELDS

    assert "objective" in VERIFIED_PROVENANCE_FIELDS


def test_the_historical_provenance_dict_is_otherwise_unchanged():
    """Exactly one key was added. Nothing existing was renamed or dropped."""
    recorded = set(historical_provenance().to_dict())
    assert recorded == {
        "run_seed", "init_seed", "corruption_seed", "learning_rate", "r",
        "lambda_align", "lambda_clean", "corpus_manifest_digest", "repository_head",
        "backbone_checkpoint", "backbone_revision", "protocol_version", "precision",
        "inventory", "objective", "fusion",
    }


# ===========================================================================
# 7. Telemetry: the old path is byte-identical, the new path adds two keys
# ===========================================================================
class _Scalar:
    """A loss-like stand-in. `loss_telemetry` is torch-free by construction."""

    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self):
        return self

    def mean(self):
        return self

    def __float__(self) -> float:
        return float(self.value)


class _HistoricalResult:
    loss = _Scalar(0.30)
    loss_align = _Scalar(0.20)
    loss_clean = _Scalar(0.10)


class _GridResult(_HistoricalResult):
    loss_grid = _Scalar(0.05)
    distance_grid_per_example = _Scalar(0.05)


def test_the_old_telemetry_path_is_unchanged():
    assert list(loss_telemetry(_HistoricalResult()).items()) == [
        ("loss", 0.30), ("loss_align", 0.20), ("loss_clean", 0.10)
    ]


def test_v2_gc_telemetry_exposes_the_required_series():
    emitted = loss_telemetry(_GridResult())
    assert set(emitted) >= {
        "loss", "loss_align", "loss_clean", "loss_grid", "mean_distance_grid"
    }
    # Appended, never interleaved: an existing consumer reads the first three.
    assert list(emitted)[:3] == ["loss", "loss_align", "loss_clean"]
    assert emitted["loss_grid"] == 0.05
    assert emitted["mean_distance_grid"] == 0.05


def test_the_monitor_maps_the_new_series_without_touching_the_old_ones():
    body = source("scripts/stage1_wandb_monitor.py")
    assert '"train/loss": event.get("loss")' in body
    assert '"train/loss_align": event.get("loss_align")' in body
    assert '"train/loss_clean": event.get("loss_clean")' in body
    assert '"loss_grid", "mean_distance_grid"' in body


# ===========================================================================
# 8. Torch-gated numerics
# ===========================================================================
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)

if torch is not None:  # pragma: no cover - exercised only where torch exists
    from test_stage1 import build_stack, synthetic_batch


def build_grid_stack(d: int = 16):
    """The same synthetic stack the historical objective is tested on.

    Reused deliberately: V2-GC must run on exactly the model the historical
    objective runs on, so anything that had to change about the stack would be a
    finding.
    """
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.objective_grid import GridConsistencyObjective

    encoder, adapter, wrapper, historical = build_stack(d)
    objective = GridConsistencyObjective(
        UnmarkEncoder(encoder, adapter),
        GridConsistencyWeights(lambda_align=1.0, lambda_clean=1.0),
    )
    return encoder, adapter, objective, historical


def hidden(rows: list[list[list[float]]]):
    return torch.tensor(rows, dtype=torch.float32)


# -- the distance itself ----------------------------------------------------
@requires_torch
def test_runtime_token_grid_distance_matches_hand_computed_cosines():
    """Expected values derived analytically, not read off the implementation."""
    from unmark.stage1.objective_grid import token_grid_distance

    # One example, L = 4: [special, content, content, special].
    corrupt = hidden([[[9., 9.], [1., 0.], [1., 1.], [9., 9.]]])
    clean = hidden([[[8., 8.], [1., 0.], [0., 1.], [8., 8.]]])
    attention = torch.ones(1, 4, dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 1]])

    # cos([1,0],[1,0]) = 1            -> d = 0
    # cos([1,1],[0,1]) = 1/sqrt(2)    -> d = 1 - 0.70710678 = 0.29289322
    # mean over the 2 valid tokens    -> 0.14644661
    out = token_grid_distance(corrupt, clean, attention, special)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.14644661, abs=1e-6)


@requires_torch
def test_runtime_identical_states_give_zero_and_opposed_states_give_two():
    from unmark.stage1.objective_grid import token_grid_distance

    states = torch.randn(3, 5, 8)
    attention = torch.ones(3, 5, dtype=torch.long)
    special = torch.zeros(3, 5, dtype=torch.long)
    same = token_grid_distance(states, states.clone(), attention, special)
    opposed = token_grid_distance(states, -states.clone(), attention, special)
    assert torch.allclose(same, torch.zeros(3), atol=1e-6)
    assert torch.allclose(opposed, torch.full((3,), 2.0), atol=1e-6)


@requires_torch
def test_runtime_padding_tokens_are_excluded():
    """A padded position may hold anything; it must not move the loss."""
    from unmark.stage1.objective_grid import token_grid_distance

    corrupt = hidden([[[1., 0.], [1., 0.], [0., 0.]]])
    clean = hidden([[[1., 0.], [1., 0.], [0., 0.]]])
    attention = torch.tensor([[1, 1, 0]])
    special = torch.zeros(1, 3, dtype=torch.long)
    baseline = token_grid_distance(corrupt, clean, attention, special)
    assert baseline.item() == pytest.approx(0.0, abs=1e-6)

    poisoned = corrupt.clone()
    poisoned[0, 2] = torch.tensor([-1.0, 0.0])  # would score d = 2 if counted
    after = token_grid_distance(poisoned, clean, attention, special)
    assert after.item() == pytest.approx(0.0, abs=1e-6)


@requires_torch
def test_runtime_special_tokens_are_excluded():
    """`<s>`/`</s>` are attended but are not content (D-B4A-006)."""
    from unmark.stage1.objective_grid import token_grid_distance

    corrupt = hidden([[[1., 0.], [1., 0.], [1., 0.]]])
    clean = hidden([[[-1., 0.], [1., 0.], [-1., 0.]]])  # specials fully opposed
    attention = torch.ones(1, 3, dtype=torch.long)
    special = torch.tensor([[1, 0, 1]])
    out = token_grid_distance(corrupt, clean, attention, special)
    assert out.item() == pytest.approx(0.0, abs=1e-6), (
        "special tokens leaked into L_grid"
    )


@requires_torch
def test_runtime_the_mean_is_per_example_then_per_batch():
    """Not a flat mean over all tokens: unequal lengths would re-weight examples."""
    from unmark.stage1.objective_grid import token_grid_distance

    # Example 0: ONE valid token, distance 0.
    # Example 1: THREE valid tokens, each distance 2.
    corrupt = hidden([
        [[1., 0.], [1., 0.], [1., 0.], [1., 0.]],
        [[1., 0.], [1., 0.], [1., 0.], [1., 0.]],
    ])
    clean = hidden([
        [[1., 0.], [0., 0.], [0., 0.], [0., 0.]],
        [[-1., 0.], [-1., 0.], [-1., 0.], [0., 0.]],
    ])
    attention = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]])
    special = torch.zeros(2, 4, dtype=torch.long)

    per_example = token_grid_distance(corrupt, clean, attention, special)
    assert per_example.tolist() == pytest.approx([0.0, 2.0], abs=1e-6)
    # Per-example then batch: (0 + 2) / 2 = 1.0.
    assert per_example.mean().item() == pytest.approx(1.0, abs=1e-6)
    # A flat token mean would be (0 + 2 + 2 + 2) / 4 = 1.5. It is not that.
    assert per_example.mean().item() != pytest.approx(1.5, abs=1e-3)


# -- fail-closed ------------------------------------------------------------
@requires_torch
@pytest.mark.parametrize(
    "corrupt_shape, clean_shape, mask_shape",
    [
        ((2, 5, 8), (2, 4, 8), (2, 5)),   # different sequence lengths
        ((2, 5, 8), (2, 5, 4), (2, 5)),   # different hidden sizes
        ((2, 5, 8), (3, 5, 8), (2, 5)),   # different batch sizes
        ((2, 5, 8), (2, 5, 8), (2, 4)),   # mask off the grid
    ],
)
def test_runtime_grid_mismatch_fails_closed(corrupt_shape, clean_shape, mask_shape):
    """Nothing is padded, trimmed or reshaped to make a mismatch line up."""
    from unmark.stage1.objective_grid import token_grid_distance

    with pytest.raises(Stage1ContractViolation):
        token_grid_distance(
            torch.randn(*corrupt_shape),
            torch.randn(*clean_shape),
            torch.ones(*mask_shape, dtype=torch.long),
            torch.zeros(*mask_shape, dtype=torch.long),
        )


@requires_torch
def test_runtime_pooled_shaped_input_fails_closed():
    from unmark.stage1.objective_grid import token_grid_distance

    with pytest.raises(Stage1ContractViolation, match=r"\[B, L, d\]"):
        token_grid_distance(
            torch.randn(2, 8), torch.randn(2, 8),
            torch.ones(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long),
        )


@requires_torch
def test_runtime_an_example_with_no_valid_tokens_fails_closed():
    from unmark.stage1.objective_grid import token_grid_distance

    with pytest.raises(Stage1ContractViolation, match="no valid base-grid tokens"):
        token_grid_distance(
            torch.randn(2, 4, 8), torch.randn(2, 4, 8),
            torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long),
        )


# -- gradient routing -------------------------------------------------------
@requires_torch
def test_runtime_the_clean_target_receives_no_gradient_from_l_grid():
    """THE stop-gradient, measured rather than read off the source."""
    from unmark.stage1.objective_grid import token_grid_distance

    corrupt = torch.randn(2, 5, 8, requires_grad=True)
    clean = torch.randn(2, 5, 8, requires_grad=True)
    attention = torch.ones(2, 5, dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 0, 1]] * 2)

    token_grid_distance(corrupt, clean, attention, special).mean().backward()
    assert corrupt.grad is not None and float(corrupt.grad.abs().sum()) > 0
    assert clean.grad is None, (
        "L_grid back-propagated into the clean branch; it could then be minimised "
        "by degrading the clean path toward the corrupted one"
    )


@requires_torch
def test_runtime_l_clean_still_sends_gradient_through_the_clean_branch():
    """The detach is scoped to L_grid. `L_clean` is untouched."""
    _, adapter, objective, _ = build_grid_stack()
    batch = synthetic_batch()
    objective.train()

    result = objective(batch)
    result.loss_clean.backward()
    reached = {n for n, p in adapter.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert reached, "L_clean no longer reaches the adapter through the clean branch"


@requires_torch
def test_runtime_the_corrupted_branch_receives_gradient_from_l_grid_alone():
    _, adapter, objective, _ = build_grid_stack()
    batch = synthetic_batch()
    objective.train()

    result = objective(batch)
    result.loss_grid.backward()
    reached = {n for n, p in adapter.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert reached, "L_grid reaches no adapter parameter at all"


@requires_torch
def test_runtime_the_frozen_encoder_receives_no_parameter_gradients():
    encoder, adapter, objective, _ = build_grid_stack()
    objective.train()
    objective(synthetic_batch()).loss.backward()

    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.grad is None for p in encoder.parameters()), (
        "the frozen backbone received gradients under V2-GC"
    )
    assert any(float(p.grad.abs().sum()) != 0 for p in adapter.parameters())


@requires_torch
def test_runtime_the_frozen_encoder_stays_in_eval():
    encoder, adapter, objective, _ = build_grid_stack()
    objective.train()
    assert objective.training and adapter.training and not encoder.training


# -- composition ------------------------------------------------------------
@requires_torch
def test_runtime_the_three_term_composition_is_exact():
    _, _, objective, _ = build_grid_stack()
    result = objective(synthetic_batch())
    expected = result.loss_align + result.loss_clean + 1.0 * result.loss_grid
    assert torch.allclose(result.loss, expected, atol=0, rtol=0), (
        "lambda_align = lambda_clean = lambda_grid = 1.0 must compose exactly"
    )
    assert result.weights.lambda_grid == 1.0


@requires_torch
def test_runtime_each_term_is_a_mean_over_examples_not_a_sum():
    _, _, objective, _ = build_grid_stack()
    result = objective(synthetic_batch(batch=3))
    for loss, distances in (
        (result.loss_align, result.distance_align_per_example),
        (result.loss_clean, result.distance_clean_per_example),
        (result.loss_grid, result.distance_grid_per_example),
    ):
        assert torch.allclose(loss, distances.mean())
        assert not torch.allclose(loss, distances.sum())


@requires_torch
def test_runtime_the_two_pooled_terms_equal_the_historical_objectives():
    """V2-GC adds a term; it does not redefine the two that were there."""
    _, _, objective, historical = build_grid_stack()
    batch = synthetic_batch()
    with torch.no_grad():
        new = objective(batch)
        old = historical(batch)
    assert torch.allclose(new.loss_align, old.loss_align, atol=0, rtol=0)
    assert torch.allclose(new.loss_clean, old.loss_clean, atol=0, rtol=0)


@requires_torch
def test_runtime_the_result_dict_carries_the_identity_and_no_raw_text():
    """The payload states WHICH objective ran, and carries nothing unsafe.

    The value-type assertion used to read `isinstance(v, (int, float, str))`.
    That was written when `ObjectiveIdentity.to_dict()` held two keys; since C2
    it durably carries `lambda_grd` and the nested `relational` specification,
    both legitimately `None` for a non-relational objective. `None` is valid JSON
    and correct provenance, so the stale check was testing the wrong invariant --
    the real one is JSON-safety plus a closed string vocabulary.
    """
    _, _, objective, _ = build_grid_stack()
    payload = objective(synthetic_batch()).to_dict()

    # 1. The scientific identity this result belongs to.
    assert payload["objective_id"] == "grid-consistency-v1"
    assert payload["lambda_grid"] == 1.0
    assert payload["lambda_grd"] is None, "C3 must not claim a relational weight"
    assert payload["relational"] is None, "C3 must not claim a relational spec"
    assert payload["lambda_align"] == payload["lambda_clean"] == 1.0

    # 2. The terms and diagnostics a V2-GC run must report.
    assert set(payload) >= {
        "loss", "loss_align", "loss_clean", "loss_grid",
        "mean_distance_align", "mean_distance_clean", "mean_distance_grid",
        "batch_size",
    }
    assert payload["batch_size"] == 2

    # 3. Nothing unsafe: no tensors, no runtime objects, no NaN, no raw text.
    assert_json_safe(payload)
    assert json.loads(json.dumps(payload, allow_nan=False, sort_keys=True)) == payload


# -- zero new parameters, three encoder forwards ----------------------------
@requires_torch
def test_runtime_v2_gc_adds_zero_model_parameters():
    _, _, objective, historical = build_grid_stack()
    mine = [(n, p.numel()) for n, p in objective.named_parameters()]
    theirs = [(n, p.numel()) for n, p in historical.named_parameters()]
    assert mine == theirs, "V2-GC changed the parameter set"
    assert sorted(objective.state_dict()) == sorted(historical.state_dict())


@requires_torch
def test_runtime_v2_gc_runs_exactly_three_encoder_forwards():
    """Measured on the frozen encoder itself, so no branch can hide a forward."""
    encoder, _, objective, historical = build_grid_stack()

    def counted(module):
        calls = []
        original = module.forward

        def wrapper(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        module.forward = wrapper
        return calls, original

    batch = synthetic_batch()
    calls, original = counted(encoder)
    try:
        objective(batch)
        v2gc_forwards = len(calls)
        calls.clear()
        historical(batch)
        historical_forwards = len(calls)
    finally:
        encoder.forward = original

    assert v2gc_forwards == 3, f"V2-GC ran {v2gc_forwards} encoder forwards, not 3"
    assert v2gc_forwards == historical_forwards, (
        f"V2-GC introduced extra encoder forwards: {v2gc_forwards} vs "
        f"{historical_forwards} for the historical objective"
    )


@requires_torch
def test_runtime_the_two_adapted_branches_run_on_the_same_base_grid():
    """The invariance L_grid depends on, asserted on the real forward."""
    _, _, objective, _ = build_grid_stack()
    batch = synthetic_batch()
    hidden_clean, _ = objective.adapted_branch(
        batch["base_input_ids"], batch["base_attention_mask"],
        batch["base_special_tokens_mask"],
        batch["clean_tone_ids"], batch["clean_tone_mask"],
        batch["clean_letter_ids"], batch["clean_letter_mask"],
    )
    hidden_corrupt, _ = objective.adapted_branch(
        batch["base_input_ids"], batch["base_attention_mask"],
        batch["base_special_tokens_mask"],
        batch["corrupt_tone_ids"], batch["corrupt_tone_mask"],
        batch["corrupt_letter_ids"], batch["corrupt_letter_mask"],
    )
    assert hidden_clean.shape == hidden_corrupt.shape
    assert not torch.allclose(hidden_clean, hidden_corrupt), (
        "the channels did not reach the encoder; L_grid would be identically zero"
    )


@requires_torch
def test_runtime_adapted_branch_pooled_half_is_the_historical_value():
    """The refactor is a refactor: same forward, same pooling, same number."""
    _, _, _, historical = build_grid_stack()
    batch = synthetic_batch()
    args = (
        batch["base_input_ids"], batch["base_attention_mask"],
        batch["base_special_tokens_mask"],
        batch["clean_tone_ids"], batch["clean_tone_mask"],
        batch["clean_letter_ids"], batch["clean_letter_mask"],
    )
    with torch.no_grad():
        _, pooled = historical.adapted_branch(*args)
        direct = historical.adapted_representation(*args)
    assert torch.allclose(pooled, direct, atol=0, rtol=0)


@requires_torch
def test_runtime_plain_weights_are_refused_by_the_v2_gc_objective():
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.objective_grid import GridConsistencyObjective

    encoder, adapter, _, _ = build_grid_stack()
    with pytest.raises(Stage1ContractViolation, match="GridConsistencyWeights"):
        GridConsistencyObjective(
            UnmarkEncoder(encoder, adapter),
            ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
        )


@requires_torch
def test_runtime_a_missing_batch_field_fails_loud():
    _, _, objective, _ = build_grid_stack()
    batch = synthetic_batch()
    del batch["corrupt_tone_ids"]
    with pytest.raises(Stage1ContractViolation, match="missing fields"):
        objective(batch)


# ===========================================================================
# 9. The historical objective is not disturbed
# ===========================================================================
def test_the_historical_forward_still_composes_exactly_two_terms():
    """`Stage1Objective` is the historical objective, unchanged. Structurally."""
    body = body_without_docstring(function("forward", OBJECTIVE))
    assert (
        "self.weights.lambda_align * loss_align + self.weights.lambda_clean * loss_clean"
        in body
    )
    for forbidden in ("lambda_grid", "loss_grid", "token_grid_distance"):
        assert forbidden not in body, f"the historical objective gained {forbidden}"


def test_the_historical_objective_module_knows_nothing_about_v2_gc():
    body = code_only(OBJECTIVE)
    for forbidden in ("GridConsistencyObjective", "GridConsistencyWeights",
                      "token_grid_distance", "lambda_grid", "objective_grid"):
        assert forbidden not in body, f"objective.py reaches {forbidden}"


def test_the_historical_forward_still_runs_three_branches():
    called = calls_in(function("forward", OBJECTIVE))
    assert called.count("reference_representation") == 1
    assert called.count("adapted_representation") == 2


def test_the_historical_loss_result_schema_is_unchanged():
    node = next(
        n for n in ast.walk(tree(OBJECTIVE))
        if isinstance(n, ast.ClassDef) and n.name == "Stage1LossResult"
    )
    fields = [
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    ]
    assert fields == [
        "loss", "loss_align", "loss_clean",
        "distance_align_per_example", "distance_clean_per_example", "weights",
    ]


def test_only_the_v2_gc_stage_gets_the_grid_objective():
    """Keyed on the STAGE NAME through the candidate register, not on an argument.

    Executed against the real register rather than asserted about source text:
    every registered stage is resolved and its objective checked.
    """
    from unmark.stage1.candidates import CANDIDATES, candidate_for_stage

    for candidate in CANDIDATES:
        resolved = candidate_for_stage(candidate.stage)
        if candidate.stage == V2_GC_STAGE:
            assert resolved.objective is GRID_CONSISTENCY_OBJECTIVE
        else:
            assert resolved.objective is not GRID_CONSISTENCY_OBJECTIVE, candidate.stage
            assert resolved.objective.lambda_grid is None, candidate.stage
    # Exactly one registered stage trains the grid objective.
    assert [c.stage for c in CANDIDATES
            if c.objective is GRID_CONSISTENCY_OBJECTIVE] == [V2_GC_STAGE]

    # `execute_stage` takes no objective argument at all, so the stage name is
    # the only thing that can choose one.
    stage_fn = function("execute_stage", "unmark/stage1/execute.py")
    parameters = {a.arg for a in stage_fn.args.args + stage_fn.args.kwonlyargs}
    assert "objective" not in parameters and "objective_identity" not in parameters

    # And it resolves the candidate exactly once, from `stage`.
    resolutions = [
        node for node in ast.walk(stage_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "candidate_for_stage"
    ]
    assert len(resolutions) == 1
    assert ast.unparse(resolutions[0]) == "candidate_for_stage(stage)"


def test_the_v2_gc_weights_come_from_the_provenance_the_artifact_records():
    """One source for the two lambdas, so loss and checkpoint cannot disagree.

    `build_candidate_objective` upgrades the weights it is HANDED; it never
    derives its own. `execute_stage` hands it `provenance.weights` -- the same
    object the checkpoint records -- so the objective cannot optimise one pair of
    lambdas while the artifact claims another.
    """
    node = next(
        n for n in ast.walk(tree("unmark/stage1/execute.py"))
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "GridConsistencyWeights"
    )
    passed = ast.unparse(node)
    assert "weights.lambda_align" in passed
    assert "weights.lambda_clean" in passed
    assert "lambda_grid" not in passed, "lambda_grid is locked; it is never passed in"
    # The builder derives nothing of its own.
    builder = function("build_candidate_objective", "unmark/stage1/execute.py")
    assert "lambdas_for_r" not in calls_in(builder)

    # And the run hands it exactly the provenance weights.
    construction = next(
        n for n in ast.walk(function("execute_stage", "unmark/stage1/execute.py"))
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "build_candidate_objective"
    )
    assert "provenance.weights" in ast.unparse(construction)


# ===========================================================================
# 10. The real-model smoke dispatch (Audit 065 BLOCKER 4)
# ===========================================================================
EXECUTE = "unmark/stage1/execute.py"


def test_smoke_resolves_the_candidate_through_the_shared_register():
    """The SAME mechanism the run uses, not a second stage->objective table."""
    called = calls_in(function("smoke_check", EXECUTE))
    assert "candidate_for_stage" in called, (
        "smoke_check must resolve its candidate through the shared register"
    )
    assert "build_candidate_objective" in called, (
        "smoke_check must build the objective through the shared constructor"
    )


def test_smoke_does_not_build_the_historical_objective_directly():
    """Audit 065 BLOCKER 4: it used to call the class handed back by build_objective."""
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "objective_cls(" not in body, (
        "smoke_check constructs a hard-wired objective class again"
    )
    assert "Stage1Objective(" not in body
    assert "GridConsistencyObjective(" not in body, (
        "smoke_check must not special-case a candidate; the register dispatches"
    )


def test_smoke_takes_the_candidate_as_a_stage_name():
    node = function("smoke_check", EXECUTE)
    parameters = {a.arg for a in node.args.args + node.args.kwonlyargs}
    assert "stage" in parameters
    # It cannot be handed an objective or weights directly.
    for forbidden in ("objective", "objective_id", "weights", "lambda_grid"):
        assert forbidden not in parameters, f"smoke_check accepts {forbidden}"


def test_smoke_still_cannot_train():
    called = calls_in(function("smoke_check", EXECUTE))
    for forbidden in ("backward", "step", "zero_grad", "build_optimizer", "AdamW",
                      "train_run", "save_training_checkpoint", "select_checkpoint",
                      "resolve_budget", "evaluate"):
        assert forbidden not in called, f"smoke_check reaches {forbidden}()"
    assert "no_grad" in called, "the single forward must be under no_grad"


def test_smoke_asserts_every_loss_term_is_finite():
    """A smoke that prints NaN and returns 0 is not a check."""
    body = body_without_docstring(function("smoke_check", EXECUTE))
    assert "loss_grid" in body and "mean_distance_grid" in body, (
        "the smoke does not verify the grid quantities a V2-GC run produces"
    )
    assert "lambda_grid is not None" in body, (
        "the grid terms must be required exactly when the candidate has a grid term"
    )
    assert "Stage1ContractViolation" in body, "a non-finite term must fail closed"


def test_smoke_default_candidate_is_the_historical_objective():
    """A smoke that names no candidate behaves exactly as it did before."""
    from unmark.stage1.candidates import candidate_for_stage as resolve
    from unmark.stage1.execute import HISTORICAL_SMOKE_STAGE

    assert resolve(HISTORICAL_SMOKE_STAGE).objective is HISTORICAL_OBJECTIVE


def test_the_shared_constructor_dispatches_on_the_objective_identity():
    """Not on the stage string, so C1/C2 need no new branch at this call site."""
    builder = function("build_candidate_objective", EXECUTE)
    body = body_without_docstring(builder)
    assert "candidate.objective is GRID_CONSISTENCY_OBJECTIVE" in body
    assert "V2_GC_STAGE" not in body, "the constructor compares stage names"
    assert "V2_SCF_STAGE" not in body, "the constructor compares stage names"

    # The dispatch CONDITION is the objective identity and nothing else. The
    # stage name may appear only inside the fail-closed guard's message.
    conditions = [
        ast.unparse(node.test) for node in ast.walk(builder) if isinstance(node, ast.If)
    ]
    assert "candidate.objective is GRID_CONSISTENCY_OBJECTIVE" in conditions
    assert not any("candidate.stage" in c for c in conditions), (
        "the objective dispatch branches on a stage name"
    )


@requires_torch
def test_runtime_the_shared_constructor_builds_each_candidates_objective():
    """One dispatch, exercised for every registered candidate.

    Each candidate is given an adapter built for ITS OWN fusion, through the
    authoritative candidate-aware path (`fresh_adapter(..., fusion_id)`). The
    earlier version reused one historical-fusion adapter for every candidate,
    which C1 made invalid: production correctly refuses a `v2_scf` candidate
    holding a `historical-fusion-v1` adapter, and that refusal is the guard --
    covered below by its own negative test -- not something to work around here.
    """
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.candidates import CANDIDATES
    from unmark.stage1.execute import build_candidate_objective
    from unmark.stage1.initialisation import fresh_adapter
    from unmark.stage1.objective import Stage1Objective
    from unmark.stage1.objective_grid import GridConsistencyObjective
    from unmark.stage1.objective_relational import RelationalDistillationObjective

    expected = {
        "lr_pilot": (HISTORICAL_FUSION_ID, Stage1Objective),
        "r_phase1": (HISTORICAL_FUSION_ID, Stage1Objective),
        "final_main": (HISTORICAL_FUSION_ID, Stage1Objective),
        "v2_scf": ("scale-calibrated-fusion-v1", Stage1Objective),
        "v2_grd": (HISTORICAL_FUSION_ID, RelationalDistillationObjective),
        "v2_gc": (HISTORICAL_FUSION_ID, GridConsistencyObjective),
    }
    assert {c.stage for c in CANDIDATES} == set(expected), (
        "a candidate was registered without an expectation here"
    )

    encoder, _, _, _ = build_grid_stack()
    weights = ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0)
    for candidate in CANDIDATES:
        fusion_id, objective_type = expected[candidate.stage]
        assert candidate.fusion.fusion_id == fusion_id, candidate.stage

        # The candidate's OWN architecture, from the candidate's own identity.
        adapter = fresh_adapter(16, 51800, candidate.fusion.fusion_id)
        assert adapter.config.fusion_id == fusion_id
        wrapper = UnmarkEncoder(encoder, adapter)

        built = build_candidate_objective(wrapper, candidate, weights)
        assert type(built) is objective_type, (
            f"{candidate.stage}: built {type(built).__name__}, expected "
            f"{objective_type.__name__}"
        )
        if candidate.objective is GRID_CONSISTENCY_OBJECTIVE:
            assert built.weights.lambda_grid == 1.0
        else:
            assert not isinstance(built, GridConsistencyObjective), candidate.stage
        # Whichever was built, the parameter set is identical: no candidate adds
        # a parameter, whatever its fusion or its objective.
        assert sum(p.numel() for p in built.parameters()) == sum(
            p.numel() for p in wrapper.parameters()
        )


@requires_torch
def test_runtime_a_mismatched_fusion_fails_closed():
    """The production guard the stale test above was tripping over. It stays.

    A C1 candidate handed a historical-fusion adapter must be refused: the two
    adapters are shape-compatible, so building the objective anyway would train
    `v2_scf`'s provenance over the historical mixture rule and the artifact would
    describe a model that was never run.
    """
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.candidates import candidate_for_stage
    from unmark.stage1.execute import build_candidate_objective
    from unmark.stage1.initialisation import fresh_adapter

    encoder, _, _, _ = build_grid_stack()
    weights = ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0)
    historical_adapter = fresh_adapter(16, 51800, HISTORICAL_FUSION_ID)

    with pytest.raises(Stage1ContractViolation, match="implements fusion"):
        build_candidate_objective(
            UnmarkEncoder(encoder, historical_adapter),
            candidate_for_stage("v2_scf"),
            weights,
        )

    # ... and the reverse: a scale-calibrated adapter under a historical stage.
    calibrated = fresh_adapter(16, 51800, "scale-calibrated-fusion-v1")
    with pytest.raises(Stage1ContractViolation, match="implements fusion"):
        build_candidate_objective(
            UnmarkEncoder(encoder, calibrated),
            candidate_for_stage("lr_pilot"),
            weights,
        )


@requires_torch
def test_runtime_the_smoke_objective_produces_the_grid_telemetry():
    """What `smoke_check` asserts finite: every term, plus the grid diagnostics."""
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.candidates import candidate_for_stage as resolve
    from unmark.stage1.execute import build_candidate_objective

    encoder, adapter, _, _ = build_grid_stack()
    objective = build_candidate_objective(
        UnmarkEncoder(encoder, adapter),
        resolve(V2_GC_STAGE),
        ObjectiveWeights(lambda_align=1.0, lambda_clean=1.0),
    )
    with torch.no_grad():
        payload = objective(synthetic_batch()).to_dict()
    for key in ("loss", "loss_align", "loss_clean", "loss_grid", "mean_distance_grid"):
        assert isinstance(payload[key], float), key
        assert payload[key] == payload[key], f"{key} is NaN"
    assert payload["objective_id"] == "grid-consistency-v1"
