"""B4A: the torch-free neural adapter contract.

Local-only. No torch, no transformers, no model downloads, no network. These
tests guard a *specification*, not an implementation: they check that the
contract matches the proposal and that OPEN items cannot acquire a default by
accident.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.alignment import TokenToneLabel
from unmark.modeling import (
    APPLICABLE_LETTER_LABELS,
    FUSION_INPUT_MULTIPLIER,
    FUSION_IS_CONVEX,
    GATE_INIT_BIAS,
    GATE_INIT_TARGET,
    GATE_INIT_WEIGHT,
    GATE_IS_PROJECTION,
    GATE_TRANSFORM,
    GATE_ZERO_IS_ATTAINABLE,
    GATE_ZERO_IS_WIRING_TEST_ONLY,
    GATE_ZERO_RECOVERS,
    LETTER_EMPTY_IS_ZERO_VECTOR,
    LETTER_LABEL_IDS,
    LETTER_NA_SENTINEL,
    LETTER_TABLE_ROWS,
    MARKED_TONE_LABELS,
    OBSERVABLE_TONE_IDS,
    PARAMETER_FORMULA,
    STAGE1_POOLING,
    TONE_NA_IS_ZERO_VECTOR,
    TONE_NA_SENTINEL,
    TONE_TABLE_ROWS,
    AdapterConfig,
    GateContract,
    GateInit,
    LetterChannelContract,
    LetterEmptyTreatment,
    LockedContractViolation,
    Stage1PoolingContract,
    Stage1PoolingError,
    ToneChannelContract,
    ToneNaTreatment,
    TonePolicy,
    UnresolvedAdapterContract,
    fusion_equation,
    h4_equalized,
    logit,
    sigmoid,
)
from unmark.orthography import LetterDiacritic, ObservedTone, Tone

REPO = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_MODULES = ("unmark/modeling/contracts.py", "unmark/modeling/config.py")


def resolved(hidden_size: int = 768, **kwargs) -> AdapterConfig:
    """A config at the locked B4A settings.

    Every former OPEN item is now the *default*, so this is just `AdapterConfig`
    with a hidden size. The helper survives because many tests read better with
    the name.
    """
    return AdapterConfig(hidden_size=hidden_size, **kwargs)


# ---------------------------------------------------------------------------
# 1. Tone cardinality and mapping
# ---------------------------------------------------------------------------
def test_tone_table_has_seven_slots():
    """Proposal §4.3/§5.1: 5 marked + 2 policy slots."""
    assert TONE_TABLE_ROWS == 7
    assert len(MARKED_TONE_LABELS) == 5


def test_seven_slots_are_five_marked_plus_two_policy_slots():
    """The repository's 7 deploy labels and the proposal's 7 slots are different
    sevens. This test pins the proposal's composition (D-B4A-002)."""
    assert TONE_TABLE_ROWS == len(MARKED_TONE_LABELS) + 2
    assert len(TokenToneLabel) == 7
    # ... but NA is not one of the proposal's slots.
    assert "NA" not in OBSERVABLE_TONE_IDS


def test_observable_ids_are_deterministic_and_contiguous():
    assert OBSERVABLE_TONE_IDS == {
        "SAC": 0, "HUYEN": 1, "HOI": 2, "NGA": 3, "NANG": 4, "UNMARKED": 5,
    }
    assert sorted(OBSERVABLE_TONE_IDS.values()) == list(range(len(OBSERVABLE_TONE_IDS)))
    assert max(OBSERVABLE_TONE_IDS.values()) < TONE_TABLE_ROWS


def test_policy_slots_match_the_proposal_table():
    assert TonePolicy.OBSERVABLE.slot_a == "UNMARKED"
    assert TonePolicy.OBSERVABLE.slot_b is None
    assert TonePolicy.FORCED_NGANG.slot_a == "NGANG"
    assert TonePolicy.FORCED_NGANG.slot_b is None
    assert TonePolicy.ORACLE.slot_a == "NGANG"
    assert TonePolicy.ORACLE.slot_b == "MISSING"


def test_oracle_is_not_deployable():
    assert not TonePolicy.ORACLE.is_deployable
    assert TonePolicy.OBSERVABLE.is_deployable and TonePolicy.FORCED_NGANG.is_deployable


# ---------------------------------------------------------------------------
# 2. UNMARKED != NA, and no lexical NGANG on the deploy path
# ---------------------------------------------------------------------------
def test_unmarked_is_not_na():
    assert TokenToneLabel.UNMARKED is not TokenToneLabel.NA
    assert TokenToneLabel.UNMARKED.value != TokenToneLabel.NA.value


def test_lexical_ngang_absent_from_deployable_labels():
    """§4.3: genuine ngang and stripped tone both map to UNMARKED."""
    assert "NGANG" not in OBSERVABLE_TONE_IDS
    assert Tone.NGANG.name not in {label.name for label in TokenToneLabel}
    assert "NGANG" not in {label.name for label in ObservedTone}


def test_ngang_appears_only_under_non_observable_policies():
    """FORCED-NGANG and ORACLE use ngang in slot A -- but neither is UNMARK."""
    assert TonePolicy.OBSERVABLE.slot_a != "NGANG"
    for policy in (TonePolicy.FORCED_NGANG, TonePolicy.ORACLE):
        assert policy.slot_a == "NGANG"


# ---------------------------------------------------------------------------
# 3. Letter labels
# ---------------------------------------------------------------------------
def test_letter_none_is_a_label_and_na_is_not():
    assert "NONE" in APPLICABLE_LETTER_LABELS
    assert "NA" not in APPLICABLE_LETTER_LABELS
    assert LetterDiacritic.NONE is not LetterDiacritic.NA


def test_applicable_letter_set_matches_the_orthography_enum():
    assert set(APPLICABLE_LETTER_LABELS) == {
        m.name for m in LetterDiacritic if m is not LetterDiacritic.NA
    }


def test_letter_ids_are_deterministic_and_contiguous():
    assert LETTER_LABEL_IDS == {
        "NONE": 0, "BREVE": 1, "CIRCUMFLEX": 2, "HORN": 3, "STROKE": 4,
    }
    assert sorted(LETTER_LABEL_IDS.values()) == list(range(len(LETTER_LABEL_IDS)))


def test_none_is_included_and_na_excluded_from_pooling():
    contract = LetterChannelContract(empty_treatment=LetterEmptyTreatment.ZERO_VECTOR)
    assert contract.include_none_in_pool
    assert contract.exclude_na_from_pool
    assert contract.pooling == "mean"


# ---------------------------------------------------------------------------
# 4. Dimension validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -768])
def test_non_positive_hidden_size_is_rejected(bad):
    with pytest.raises(ValueError):
        AdapterConfig(hidden_size=bad)


@pytest.mark.parametrize("bad", [768.0, "768", None, True])
def test_non_integer_hidden_size_is_rejected(bad):
    with pytest.raises(TypeError):
        AdapterConfig(hidden_size=bad)


def test_hidden_size_has_no_default():
    """D-B3B0-002 is OPEN, so no backbone dimension may be assumed."""
    with pytest.raises(TypeError):
        AdapterConfig()


def test_unfreezing_the_encoder_requires_a_logged_decision():
    with pytest.raises(ValueError, match="frozen"):
        AdapterConfig(hidden_size=768, encoder_frozen=False)


def test_fusion_input_is_three_d():
    assert AdapterConfig(hidden_size=512).fusion_input_size == 1536
    assert FUSION_INPUT_MULTIPLIER == 3


# ---------------------------------------------------------------------------
# 5. Symbolic shapes
# ---------------------------------------------------------------------------
def test_symbolic_shapes_are_backbone_parameterized():
    shapes = AdapterConfig(hidden_size=1024).tensor_shapes()
    assert shapes["input_ids"] == ("B", "L")
    assert shapes["attention_mask"] == ("B", "L")
    assert shapes["tone_ids"] == ("B", "L")
    assert shapes["base_embeddings"] == ("B", "L", 1024)
    assert shapes["concatenated_channels"] == ("B", "L", 3072)
    assert shapes["fusion_weight"] == (1024, 3072)
    assert shapes["gate_weight"] == (1024, 3072)
    assert shapes["gate"] == ("B", "L", 1024)
    assert shapes["inputs_embeds"] == ("B", "L", 1024)
    assert shapes["stage1_pooled"] == ("B", 1024)


def test_letter_contributor_axis_is_ragged_not_a_fixed_constant():
    """No maximum K is invented: it stays symbolic."""
    shapes = AdapterConfig(hidden_size=768).tensor_shapes()
    assert shapes["letter_contributor_ids"] == ("B", "L", "K")
    assert shapes["letter_contributor_mask"] == ("B", "L", "K")


def test_gate_and_fusion_have_identical_projection_shapes():
    shapes = AdapterConfig(hidden_size=768).tensor_shapes()
    assert shapes["gate_weight"] == shapes["fusion_weight"]


# ---------------------------------------------------------------------------
# 6. Parameter count
# ---------------------------------------------------------------------------
def test_parameter_count_formula():
    """|phi| = 6d^2 + (4 + n_tau + n_lambda) d"""
    d = 768
    config = resolved(d)
    counts = config.parameter_count()
    n_tau, n_lambda = config.tone.embedding_rows, config.letter.embedding_rows
    assert counts.total == 6 * d * d + (4 + n_tau + n_lambda) * d


def test_parameter_terms_are_separate():
    counts = resolved(768).parameter_count()
    assert counts.fusion_weight == 3 * 768 * 768
    assert counts.gate_weight == 3 * 768 * 768
    assert counts.fusion_bias == 768 and counts.gate_bias == 768
    assert counts.layernorm == 2 * 768
    assert counts.tone_embedding == 7 * 768
    assert counts.letter_embedding == 5 * 768


def test_count_reproduces_the_proposal_budget():
    """§4.7 totals ~3.6M with a 3d->d GATE PROJECTION. A raw gate vector could
    not reach it -- this is the arithmetic behind D-B4A-001."""
    counts = resolved(768).parameter_count()
    assert 3.5e6 < counts.total < 3.6e6
    without_gate = resolved(768, use_gate=False).parameter_count()
    assert without_gate.total < 1.8e6
    assert counts.total - without_gate.total == 3 * 768 * 768 + 768


def test_gate_projection_dominates_a_hypothetical_gate_vector():
    """A per-dimension gate VECTOR would cost d; the locked gate costs 3d^2+d."""
    d = 768
    counts = resolved(d).parameter_count()
    assert counts.gate_weight + counts.gate_bias == 3 * d * d + d
    assert counts.gate_weight + counts.gate_bias > 1000 * d


def test_stage2_head_is_excluded():
    """§8.3 trains the head in a separate stage; §4.7 lists no head."""
    fields = resolved(768).parameter_count().to_dict()
    assert not any("head" in key for key in fields)


def test_h4_policies_receive_identical_capacity():
    """§4.3's whole reason for 7 slots."""
    config = resolved(768)
    assert h4_equalized(config)
    totals = {
        config.with_policy(p).parameter_count().total for p in TonePolicy
    }
    assert len(totals) == 1


def test_a_learned_na_row_is_rejected_by_the_locked_contract():
    """D-B4A-002 option (b), now rejected rather than merely described.

    An eighth learned row would give ORACLE 8 rows against the others' 7 and
    defeat the H4 equalization the 7-slot table exists to provide.
    """
    with pytest.raises(LockedContractViolation, match="D-B4A-002"):
        ToneChannelContract(na_treatment=ToneNaTreatment.EXTRA_ROW)


def test_reusing_slot_b_for_na_is_rejected():
    """D-B4A-002 option (a): the oracle needs slot B for MISSING."""
    with pytest.raises(LockedContractViolation, match="D-B4A-002"):
        ToneChannelContract(na_treatment=ToneNaTreatment.SLOT_B_ROW)


def test_a_tone_table_of_any_other_size_is_rejected():
    for rows in (6, 8):
        with pytest.raises(LockedContractViolation, match="7 trainable rows"):
            ToneChannelContract(rows=rows)


def test_a_learned_letter_na_row_is_rejected():
    with pytest.raises(LockedContractViolation, match="D-B4A-005"):
        LetterChannelContract(empty_treatment=LetterEmptyTreatment.LEARNED_NA_ROW)


def test_masking_the_letter_channel_out_is_rejected():
    """MASKED_OUT changes the concatenation width, breaking a fixed W_f."""
    with pytest.raises(LockedContractViolation, match="concatenation width"):
        LetterChannelContract(empty_treatment=LetterEmptyTreatment.MASKED_OUT)


def test_letter_table_of_any_other_size_is_rejected():
    for rows in (4, 6, 10):
        with pytest.raises(LockedContractViolation, match="D-B4A-007"):
            LetterChannelContract(rows=rows)


def test_dropping_none_or_admitting_na_to_the_pool_is_rejected():
    with pytest.raises(LockedContractViolation, match="D-B3B1C-001"):
        LetterChannelContract(include_none_in_pool=False)
    with pytest.raises(LockedContractViolation, match="D-B3B1C-001"):
        LetterChannelContract(exclude_na_from_pool=False)


def test_mlp_ablation_is_not_silently_costed():
    with pytest.raises(UnresolvedAdapterContract, match="MLP"):
        resolved(768, fusion_kind="mlp").parameter_count()


# ---------------------------------------------------------------------------
# 7. The gate-zero contract
# ---------------------------------------------------------------------------
def test_gate_zero_recovers_the_base_only_pathway():
    """§4.5, not the clean-text pathway."""
    assert GATE_ZERO_RECOVERS == "BASE_ONLY_PATHWAY"


def test_gate_zero_is_recorded_as_unattainable():
    """sigma maps onto the OPEN interval (0,1): g = 0 is a limit, not a value."""
    assert GATE_ZERO_IS_ATTAINABLE is False
    assert GateContract().zero_is_attainable is False


def test_gate_is_a_projection_with_sigmoid():
    assert GATE_TRANSFORM == "sigmoid"
    assert GATE_IS_PROJECTION is True


def test_fusion_is_convex_not_residual():
    assert FUSION_IS_CONVEX is True
    equation = fusion_equation()
    assert "z_i = g_i * f_i + (1 - g_i) * e_i" in equation
    assert "LN( W_f [ e_i ; t_i ; l_i ] + c_f )" in equation
    assert "sigma( W_g [ e_i ; t_i ; l_i ] + c_g )" in equation


# ---------------------------------------------------------------------------
# 8. The former OPEN items are locked, and the evidence chain survives
# ---------------------------------------------------------------------------
def test_a_default_config_embodies_the_locked_decisions():
    config = AdapterConfig(hidden_size=768)
    assert config.tone.na_is_zero_vector
    assert config.letter.empty_is_zero_vector
    assert config.gate.initialisation is GateInit.NEAR_ZERO_LOGIT
    assert config.resolved_decisions == (
        "D-B4A-002", "D-B4A-003", "D-B4A-004", "D-B4A-005", "D-B4A-006", "D-B4A-007",
    )


def test_rejected_alternatives_are_retained_not_deleted():
    """Audit 014 found these ambiguities; the enums keep the evidence chain.

    Deleting the rejected members would make a later reader rediscover them as
    plausible options rather than seeing them named and refused.
    """
    assert {m.name for m in ToneNaTreatment} == {"ZERO_VECTOR", "SLOT_B_ROW", "EXTRA_ROW"}
    assert {m.name for m in LetterEmptyTreatment} == {
        "ZERO_VECTOR", "LEARNED_NA_ROW", "MASKED_OUT",
    }
    assert {m.name for m in GateInit} == {"NEAR_ZERO_LOGIT", "ZERO_BIAS", "POSITIVE_BIAS"}


@pytest.mark.parametrize(
    "factory,marker",
    [
        (lambda: ToneChannelContract(na_treatment=ToneNaTreatment.EXTRA_ROW), "D-B4A-002"),
        (lambda: LetterChannelContract(empty_treatment=LetterEmptyTreatment.MASKED_OUT), "D-B4A-005"),
        (lambda: GateContract(initialisation=GateInit.ZERO_BIAS), "D-B4A-003"),
    ],
)
def test_each_violation_names_its_decision(factory, marker):
    with pytest.raises(LockedContractViolation, match=marker):
        factory()


def test_a_hand_written_gate_bias_must_match_the_locked_value():
    with pytest.raises(LockedContractViolation, match="logit"):
        GateContract(init_bias=0.0)
    with pytest.raises(LockedContractViolation, match="W_g = 0"):
        GateContract(init_weight=0.1)


def test_shapes_do_not_depend_on_the_resolved_items():
    AdapterConfig(hidden_size=768).tensor_shapes()


# ---------------------------------------------------------------------------
# 9. D-B3B0-002 must not have been closed by accident
# ---------------------------------------------------------------------------
def test_backbone_decision_is_still_open():
    decisions = (REPO / "docs/spec/decisions.md").read_text(encoding="utf-8")
    assert "D-B3B0-002" in decisions
    assert "D-B3B0-002](#d-b3b0-002): REMAINS OPEN" in decisions or (
        "D-B3B0-002 (backbone checkpoint not locked)" in decisions
    )


def test_no_backbone_hidden_size_is_hardcoded_in_the_contract():
    """768 may appear in tests and docs as an arithmetic check, never as a
    default in the contract modules."""
    for name in CONTRACT_MODULES:
        source = (REPO / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, int)
        ]
        assert 768 not in literals, f"{name} hardcodes a backbone dimension"


# ---------------------------------------------------------------------------
# 10. Hygiene: the contract modules import no ML dependency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CONTRACT_MODULES + ("unmark/modeling/__init__.py",))
def test_contract_modules_import_no_ml_dependency(name):
    source = (REPO / name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not modules & {
        "torch", "transformers", "sentencepiece", "datasets", "numpy", "py_vncorenlp"
    }


def test_pure_data_contract_modules_define_no_nn_module():
    """The *contract* modules stay pure data.

    B4A asserted this of the whole `unmark/modeling` package, when B4B's
    `nn.Module` did not exist yet. B4B added `adapter.py`, `pooling.py` and
    `collate.py`, which legitimately use torch; the assertion is therefore
    rescoped to the modules that must never depend on it. `test_neural_adapter.py`
    guards the neural side.

    Structural, not textual: the docstrings legitimately *mention* `nn.Module`
    when saying it is B4B's job. What must not exist here is a class that
    subclasses one, or a `forward` method.
    """
    for name in CONTRACT_MODULES + ("unmark/modeling/__init__.py",):
        path = REPO / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(base) for base in node.bases]
                assert not any("Module" in base for base in bases), f"{name}: {node.name}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name != "forward", f"{name} defines forward()"


# ---------------------------------------------------------------------------
# 11. Tone NA is a zero vector outside the table (D-B4A-002)
# ---------------------------------------------------------------------------
def test_tone_table_has_exactly_seven_trainable_rows_for_every_policy():
    for policy in TonePolicy:
        contract = ToneChannelContract(policy=policy)
        assert contract.embedding_rows == 7
        assert contract.trainable


def test_tone_na_is_outside_the_table():
    assert "NA" not in OBSERVABLE_TONE_IDS
    assert TONE_NA_SENTINEL not in OBSERVABLE_TONE_IDS.values()
    assert TONE_NA_SENTINEL < 0, "the NA sentinel must not be a valid row index"
    assert all(0 <= i < TONE_TABLE_ROWS for i in OBSERVABLE_TONE_IDS.values())


def test_tone_na_maps_to_the_exact_zero_vector():
    assert TONE_NA_IS_ZERO_VECTOR is True
    assert ToneChannelContract().na_is_zero_vector


def test_unmarked_is_a_learned_row_and_na_is_not():
    """UNMARKED is an observable tone state with a learned row; NA is structural
    non-applicability with no row at all."""
    assert OBSERVABLE_TONE_IDS["UNMARKED"] == 5
    assert 0 <= OBSERVABLE_TONE_IDS["UNMARKED"] < TONE_TABLE_ROWS
    assert "NA" not in OBSERVABLE_TONE_IDS
    assert TokenToneLabel.UNMARKED is not TokenToneLabel.NA


def test_the_na_sentinel_is_documented_as_never_indexing_a_table():
    source = (REPO / "unmark/modeling/contracts.py").read_text(encoding="utf-8")
    assert "Never index an embedding table with this" in source
    assert "nn.Embedding" in source


def test_unused_policy_slot_still_costs_a_row():
    """OBSERVABLE never indexes slot B, but the row is still allocated -- dropping
    it would give the policies different capacity."""
    assert not TonePolicy.OBSERVABLE.uses_slot_b
    assert TonePolicy.ORACLE.uses_slot_b
    assert ToneChannelContract(policy=TonePolicy.OBSERVABLE).embedding_rows == 7


# ---------------------------------------------------------------------------
# 12. Letter table is exactly five rows (D-B4A-005, D-B4A-007)
# ---------------------------------------------------------------------------
def test_letter_table_has_exactly_five_rows():
    assert LETTER_TABLE_ROWS == 5
    assert len(APPLICABLE_LETTER_LABELS) == 5
    assert LetterChannelContract().embedding_rows == 5


def test_none_is_a_valid_learned_letter_row():
    assert LETTER_LABEL_IDS["NONE"] == 0
    assert 0 <= LETTER_LABEL_IDS["NONE"] < LETTER_TABLE_ROWS


def test_letter_na_is_outside_the_table():
    assert "NA" not in LETTER_LABEL_IDS
    assert LETTER_NA_SENTINEL < 0
    assert LETTER_NA_SENTINEL not in LETTER_LABEL_IDS.values()


def test_empty_contributor_set_maps_to_the_exact_zero_vector():
    assert LETTER_EMPTY_IS_ZERO_VECTOR is True
    assert LetterChannelContract().empty_is_zero_vector


def test_zero_over_zero_is_explicitly_forbidden():
    """A clamped denominator is allowed only if the output is then forced to
    exact zero -- otherwise the zero arrives by accident."""
    source = (REPO / "unmark/modeling/contracts.py").read_text(encoding="utf-8")
    assert "clamp the denominator" in source
    assert "forced to exact zero" in source


# ---------------------------------------------------------------------------
# 13. Parameter formula is 6d^2 + 16d (D-B4A-007)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [128, 256, 512, 768, 1024, 4096])
def test_parameter_formula_is_six_d_squared_plus_sixteen_d(d):
    assert resolved(d).parameter_count().total == 6 * d * d + 16 * d


def test_formula_string_matches_the_arithmetic():
    assert PARAMETER_FORMULA == "6*d**2 + 16*d"
    d = 512
    assert eval(PARAMETER_FORMULA, {"d": d}) == resolved(d).parameter_count().total


def test_d_768_gives_the_expected_total_without_768_being_a_default():
    assert resolved(768).parameter_count().total == 3_551_232
    with pytest.raises(TypeError):
        AdapterConfig()


def test_n_tau_and_n_lambda_enter_the_formula_as_seven_and_five():
    config = resolved(1000)
    counts = config.parameter_count()
    assert counts.tone_embedding == 7 * 1000
    assert counts.letter_embedding == 5 * 1000


# ---------------------------------------------------------------------------
# 14. Gate initialisation (D-B4A-003)
# ---------------------------------------------------------------------------
def test_gate_weight_is_initialised_to_zero():
    assert GATE_INIT_WEIGHT == 0.0
    assert GateContract().init_weight == 0.0
    assert resolved(768).initialisation_plan()["gate_weight"] == 0.0


def test_gate_bias_is_logit_of_one_percent():
    assert GATE_INIT_TARGET == 0.01
    assert GATE_INIT_BIAS == pytest.approx(logit(0.01))
    assert GATE_INIT_BIAS == pytest.approx(-4.59511985013459, abs=1e-12)
    assert GateContract().init_bias == pytest.approx(GATE_INIT_BIAS)


def test_planned_initial_gate_is_exactly_one_percent():
    assert GateContract().initial_gate_value == pytest.approx(0.01, abs=1e-12)
    assert sigmoid(GATE_INIT_BIAS) == pytest.approx(0.01, abs=1e-12)


def test_zero_weight_makes_the_initial_gate_input_independent():
    """W_g = 0 means the concatenated channels drop out of the gate on step zero,
    so every position and dimension starts at the same value."""
    plan = resolved(768).initialisation_plan()
    assert plan["gate_weight"] == 0.0
    assert plan["initial_gate_value"] == pytest.approx(0.01, abs=1e-12)


def test_initial_gate_keeps_a_usable_derivative():
    """A gate driven to machine zero could not learn: g(1-g) would vanish."""
    derivative = GateContract().initial_gate_derivative
    assert derivative == pytest.approx(0.01 * 0.99, abs=1e-12)
    assert derivative > 1e-3


def test_zero_bias_would_have_started_at_one_half():
    """The rejected alternative, shown rather than asserted."""
    assert sigmoid(0.0) == pytest.approx(0.5)
    with pytest.raises(LockedContractViolation):
        GateContract(initialisation=GateInit.ZERO_BIAS)


def test_initialisation_does_not_claim_base_only_equality():
    plan = resolved(768).initialisation_plan()
    assert plan["initial_gate_value"] != 0.0
    assert "not equal to it" in plan["note"]


# ---------------------------------------------------------------------------
# 15. Gate-zero is a wiring test only (D-B4A-004)
# ---------------------------------------------------------------------------
def test_forced_gate_zero_is_a_wiring_test_not_a_mode():
    assert GATE_ZERO_IS_WIRING_TEST_ONLY is True
    assert GATE_ZERO_IS_ATTAINABLE is False
    assert resolved(768).initialisation_plan()["gate_zero_is_wiring_test_only"] is True


def test_wiring_test_is_distinct_from_the_initialised_model():
    """g := 0 (test override) and g = 0.01 (initialisation) are different things
    and must not be conflated."""
    plan = resolved(768).initialisation_plan()
    assert plan["initial_gate_value"] == pytest.approx(0.01)
    assert plan["gate_zero_is_attainable"] is False


def test_convex_combination_at_g_zero_is_exactly_the_base_embedding():
    """The wiring identity the forced override checks: z = g*f + (1-g)*e with
    g = 0 gives z = e exactly, for any f."""
    for e, f in ((1.5, -3.25), (0.0, 7.0), (-2.0, 0.125)):
        g = 0.0
        assert g * f + (1 - g) * e == e


def test_no_public_gate_zero_flag_is_exposed():
    """A casual production 'gate zero mode' could silently enter an experiment."""
    config = resolved(768)
    names = set(dir(config)) | set(config.initialisation_plan())
    assert not any(
        name in names for name in ("gate_zero", "force_gate_zero", "use_gate_zero", "zero_gate")
    )


# ---------------------------------------------------------------------------
# 16. Stage-1 pooling (D-B4A-006)
# ---------------------------------------------------------------------------
def test_stage1_pooling_is_masked_mean_over_non_special_content():
    assert STAGE1_POOLING == "masked_mean_over_non_special_content_tokens"
    assert Stage1PoolingContract().kind == STAGE1_POOLING


def test_stage1_pooling_excludes_special_tokens():
    pooling = Stage1PoolingContract()
    assert pooling.exclude_special_tokens
    # <s> tok tok </s>
    mask = pooling.content_mask([1, 1, 1, 1], [1, 0, 0, 1])
    assert mask == [0, 1, 1, 0]
    assert pooling.require_content(mask) == 2


def test_stage1_pooling_excludes_padding():
    pooling = Stage1PoolingContract()
    assert pooling.exclude_padding
    # <s> tok </s> <pad> <pad>
    mask = pooling.content_mask([1, 1, 1, 0, 0], [1, 0, 1, 1, 1])
    assert mask == [0, 1, 0, 0, 0]
    assert pooling.require_content(mask) == 1


def test_stage1_pooling_permits_branch_lengths_to_differ():
    """h(x) runs the encoder's own tokenization of clean text; h'(.) runs the
    base grid. They do not share L, and no per-token correspondence is assumed."""
    pooling = Stage1PoolingContract()
    assert not pooling.requires_equal_branch_lengths
    clean = pooling.content_mask([1] * 6, [1, 0, 0, 0, 0, 1])
    adapted = pooling.content_mask([1] * 9, [1, 0, 0, 0, 0, 0, 0, 0, 1])
    assert len(clean) != len(adapted)
    assert pooling.require_content(clean) == 4
    assert pooling.require_content(adapted) == 7


def test_zero_content_pooling_fails_loud():
    pooling = Stage1PoolingContract()
    only_special = pooling.content_mask([1, 1], [1, 1])
    assert only_special == [0, 0]
    with pytest.raises(Stage1PoolingError, match="no content positions"):
        pooling.require_content(only_special)


def test_zero_content_does_not_fall_back_silently():
    source = (REPO / "unmark/modeling/contracts.py").read_text(encoding="utf-8")
    assert "fails loud" in source
    assert "<s>, an unmasked mean, or a zero vector" in source


def test_mismatched_mask_lengths_are_rejected():
    with pytest.raises(ValueError, match="mask lengths differ"):
        Stage1PoolingContract().content_mask([1, 1, 1], [1, 0])


def test_masked_mean_matches_the_specified_formula():
    """h = sum_i m_i H_i / sum_i m_i, on a scalar stand-in for H."""
    pooling = Stage1PoolingContract()
    hidden = [10.0, 2.0, 4.0, 99.0]          # <s>, tok, tok, <pad>
    mask = pooling.content_mask([1, 1, 1, 0], [1, 0, 0, 1])
    count = pooling.require_content(mask)
    pooled = sum(m * h for m, h in zip(mask, hidden)) / count
    assert mask == [0, 1, 1, 0]
    assert pooled == pytest.approx(3.0)      # padding and <s> excluded


# ---------------------------------------------------------------------------
# 17. The zero channels do not make the adapter inactive
# ---------------------------------------------------------------------------
def test_zero_channels_still_pass_through_fusion():
    """[e_i ; 0 ; 0] is still a full-width input to W_f and W_g. Special tokens
    get zero CONTRIBUTIONS, not a bypass."""
    shapes = resolved(768).tensor_shapes()
    assert shapes["concatenated_channels"] == ("B", "L", 3 * 768)
    assert shapes["fusion_weight"] == (768, 3 * 768)


def test_no_special_token_bypass_is_specified():
    source = (REPO / "unmark/modeling/config.py").read_text(encoding="utf-8")
    assert "bypass" not in source.lower()


# ---------------------------------------------------------------------------
# 18. tone_mask is part of the tensor contract (D-B4A-002)
# ---------------------------------------------------------------------------
def test_tone_mask_is_in_the_tensor_contract():
    shapes = AdapterConfig(hidden_size=768).tensor_shapes()
    assert shapes["tone_mask"] == ("B", "L")
    assert shapes["tone_ids"] == ("B", "L")


def test_tone_mask_and_tone_ids_share_a_shape():
    """One mask entry per id entry: the mask says whether that id indexes a row."""
    shapes = AdapterConfig(hidden_size=512).tensor_shapes()
    assert shapes["tone_mask"] == shapes["tone_ids"]


def test_tone_mask_semantics_are_documented():
    doc = AdapterConfig.tensor_shapes.__doc__
    assert "sentinel" in doc and "safe placeholder" in doc


def test_special_tokens_mask_is_in_the_tensor_contract():
    assert AdapterConfig(hidden_size=768).tensor_shapes()["special_tokens_mask"] == ("B", "L")


# ---------------------------------------------------------------------------
# 19. The two masks are not interchangeable
# ---------------------------------------------------------------------------
def test_attention_mask_excludes_padding_and_special_mask_excludes_special_tokens():
    pooling = Stage1PoolingContract()
    # <s> tok tok </s> <pad>
    attention = [1, 1, 1, 1, 0]
    special = [1, 0, 0, 1, 1]
    assert pooling.content_mask(attention, special) == [0, 1, 1, 0, 0]
    # Attention alone would keep the special tokens ...
    assert pooling.content_mask(attention, [0] * 5) == [1, 1, 1, 1, 0]
    # ... and the special mask alone would keep the padding.
    assert pooling.content_mask([1] * 5, special) == [0, 1, 1, 0, 0]


def test_padding_is_never_counted_as_content():
    pooling = Stage1PoolingContract()
    mask = pooling.content_mask([1, 1, 0, 0], [0, 0, 0, 0])
    assert mask[2:] == [0, 0]
    assert pooling.require_content(mask) == 2


# ---------------------------------------------------------------------------
# 20. Document consistency: no stale current-state claims
# ---------------------------------------------------------------------------
B4A_DOCS = (
    "docs/audits/014-b4a-neural-adapter-contract-preflight.md",
    "docs/spec/neural-adapter.md",
    "docs/spec/decisions.md",
)

STALE_CURRENT_STATE = (
    "pooling OPEN",
    "Pooling representation: OPEN",
    "**Marked OPEN rather than chosen.**",
    "**Blocks B4B.**",
    "**Blocks B4B**:",
    "OPEN — RESEARCHER DECISION REQUIRED",
    "**[OPEN]**",
)


def b4a_region(name: str) -> str:
    """The part of a document that talks about B4A.

    `decisions.md` accumulates every phase, and later phases legitimately have
    open items -- so the whole file must not be scanned for "still open"
    phrasing. Only the B4A block is in scope here.
    """
    text = (REPO / name).read_text(encoding="utf-8")
    if name != "docs/spec/decisions.md":
        return text
    start = text.index("## B4A — neural adapter contract preflight")
    remainder = text.index("\n## ", start + 1)
    return text[start:remainder]


@pytest.mark.parametrize("name", B4A_DOCS)
@pytest.mark.parametrize("phrase", STALE_CURRENT_STATE)
def test_no_stale_current_state_claims_in_b4a_docs(name, phrase):
    """The B4A items were open when first found and are resolved now.

    History stays documented, but no sentence may read as saying a **B4A** item
    is *currently* open. Two such sentences survived the first revision of Audit
    014 -- this guards the class rather than the instances.
    """
    assert phrase not in b4a_region(name), (
        f"{name} still asserts a resolved B4A item is open: {phrase!r}"
    )


@pytest.mark.parametrize("name", B4A_DOCS)
def test_d_b4a_006_is_described_as_resolved(name):
    text = (REPO / name).read_text(encoding="utf-8")
    if "D-B4A-006" not in text:
        pytest.skip(f"{name} does not mention D-B4A-006")
    assert "masked mean" in text or "attention-masked mean" in text


def test_the_audit_verdict_is_pass():
    text = (REPO / B4A_DOCS[0]).read_text(encoding="utf-8")
    assert "**PASS — B4B NEURAL IMPLEMENTATION READY**" in text
    # The historical verdict is still recorded, but only as history.
    assert "first returned **CONDITIONAL PASS" in text


def test_audit_records_the_locked_values():
    text = (REPO / B4A_DOCS[0]).read_text(encoding="utf-8")
    for claim in (
        "Exactly 7 trainable rows",
        "Exactly 5 rows",
        "logit(0.01)",
        "wiring test only",
        "non-special content tokens",
        "6d² + 16d",
        "D-B3B0-002 remains OPEN",
    ):
        assert claim in text, f"audit no longer states: {claim}"


def test_position_id_question_is_still_empirical_b4b_work():
    """Resolved items are resolved; this one is deferred to real-model checking
    and must not be presented as verified."""
    for name in (B4A_DOCS[0], B4A_DOCS[1]):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "position id" in text.lower()
        assert "B4B" in text
    audit = (REPO / B4A_DOCS[0]).read_text(encoding="utf-8")
    assert "not verified" in audit or "must not be \"fixed\" in\n   pure-data" in audit


def test_stage1_pooling_formula_appears_in_the_audit():
    text = (REPO / B4A_DOCS[0]).read_text(encoding="utf-8")
    assert "content_mask = attention_mask AND NOT special_tokens_mask" in text
    assert "sum(content_mask * H) / sum(content_mask)" in text
    assert "FAIL LOUD" in text
