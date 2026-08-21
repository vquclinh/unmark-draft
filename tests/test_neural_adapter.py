"""B4B: the neural adapter, as far as an ML-free environment can check it.

Three tiers:

* **Static** — AST/source checks on `unmark/modeling/adapter.py`,
  `pooling.py` and `collate.py`. These run everywhere and guard the locked
  contract against edits.
* **Torch-free runtime** — the collator's metadata layer genuinely does not need
  torch, so its semantics are exercised for real.
* **Torch-gated runtime** — numerics. Skipped cleanly when torch is absent,
  which is the normal state of the local `.venv`.

No package installation, no network, no model weights, no training.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.alignment import (
    OrthographicRegion,
    TokenToneLabel,
    align_chunk,
    character_letter_labels,
    overlay_orthography,
    project_piece,
    whitespace_chunks,
)
from unmark.modeling import (
    GATE_INIT_BIAS,
    LETTER_LABEL_IDS,
    LETTER_NA_SENTINEL,
    LETTER_TABLE_ROWS,
    OBSERVABLE_TONE_IDS,
    TONE_NA_SENTINEL,
    TONE_TABLE_ROWS,
    AdapterConfig,
)
from unmark.modeling.collate import (
    EncodedExample,
    build_example,
    letter_contributor_ids,
    padded_batch,
    tone_id_and_mask,
)
from unmark.orthography import Eligibility, ObservedTone, decompose

REPO = pathlib.Path(__file__).resolve().parents[1]
NEURAL_MODULES = (
    "unmark/modeling/adapter.py",
    "unmark/modeling/pooling.py",
    "unmark/modeling/collate.py",
)
PROBE = "scripts/b4b_phobert_adapter_probe.py"


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    return ast.parse(source(name))


def called_names(name: str) -> set[str]:
    """Every function/method name actually *called* in the module.

    Structural, so prose in a docstring saying what the module does **not** do
    cannot trip the check -- an earlier version of these tests matched raw
    strings and flagged its own documentation.
    """
    parsed = tree(name)
    out: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


def imported_modules(name: str) -> set[str]:
    parsed = tree(name)
    return {
        (node.module or "").split(".")[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


# ---------------------------------------------------------------------------
# 1. The ML-free package must stay importable
# ---------------------------------------------------------------------------
def test_modeling_package_does_not_import_torch():
    """`unmark.modeling` is imported by ML-free code; the neural modules are
    opt-in and must not be pulled in by the package `__init__`."""
    assert "torch" not in imported_modules("unmark/modeling/__init__.py")
    # Structural: the docstring legitimately *shows* how to import the neural
    # modules explicitly. What matters is that __init__ does not import them.
    actually_imported = {
        node.module
        for node in tree("unmark/modeling/__init__.py").body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for module in ("adapter", "pooling", "collate"):
        assert f"unmark.modeling.{module}" not in actually_imported, (
            f"{module} imports torch and must not be re-exported from __init__"
        )


def test_collate_metadata_layer_is_torch_free():
    """`import torch` in collate.py must live inside the tensor-packing function
    only, so the metadata semantics stay locally testable."""
    parsed = tree("unmark/modeling/collate.py")
    top_level = {
        alias.name.split(".")[0]
        for node in parsed.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_level


# ---------------------------------------------------------------------------
# 2. Table cardinality is not re-declared in the neural code
# ---------------------------------------------------------------------------
def test_embedding_tables_are_built_from_the_locked_constants():
    """Seven tone rows and five letter rows, taken from the contract rather than
    written as literals that could drift from it."""
    body = source("unmark/modeling/adapter.py")
    assert "nn.Embedding(TONE_TABLE_ROWS," in body
    assert "nn.Embedding(LETTER_TABLE_ROWS," in body
    assert TONE_TABLE_ROWS == 7 and LETTER_TABLE_ROWS == 5


def test_no_literal_table_size_in_the_neural_module():
    parsed = tree("unmark/modeling/adapter.py")
    calls = [
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Embedding"
    ]
    assert calls, "expected embedding tables"
    for call in calls:
        first = call.args[0]
        assert not isinstance(first, ast.Constant), (
            "embedding cardinality must come from the contract, not a literal"
        )


def test_no_eighth_tone_row_anywhere():
    body = source("unmark/modeling/adapter.py")
    assert "TONE_TABLE_ROWS + 1" not in body
    assert "LETTER_TABLE_ROWS + 1" not in body


# ---------------------------------------------------------------------------
# 3. The NA sentinel can never index a table
# ---------------------------------------------------------------------------
def test_both_channels_replace_the_sentinel_before_lookup():
    body = source("unmark/modeling/adapter.py")
    assert body.count("torch.where(mask, ") >= 2, "tone and letter must both guard the lookup"
    assert "safe_ids" in body


def test_lookup_is_never_called_on_raw_ids():
    """The embedding tables must be called with `safe_ids`, never the raw ids."""
    parsed = tree("unmark/modeling/adapter.py")
    for node in ast.walk(parsed):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"tone_embedding", "letter_embedding"}
        ):
            assert node.args and isinstance(node.args[0], ast.Name)
            assert node.args[0].id == "safe_ids", (
                f"{node.func.attr} called with {ast.unparse(node.args[0])}, not safe_ids"
            )


def test_sentinels_are_negative_and_outside_both_tables():
    assert TONE_NA_SENTINEL < 0 and LETTER_NA_SENTINEL < 0
    assert TONE_NA_SENTINEL not in OBSERVABLE_TONE_IDS.values()
    assert LETTER_NA_SENTINEL not in LETTER_LABEL_IDS.values()


def test_masked_positions_are_zeroed_after_lookup():
    body = source("unmark/modeling/adapter.py")
    assert "mask.unsqueeze(-1).to(embedded.dtype)" in body


# ---------------------------------------------------------------------------
# 4. No public gate-zero flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEURAL_MODULES + (PROBE,))
def test_no_public_force_gate_zero_flag(name):
    body = source(name).lower()
    for banned in ("force_gate_zero", "gate_zero_mode", "use_gate_zero", "zero_gate"):
        assert banned not in body, f"{name} exposes {banned}"


def test_adapter_config_has_no_gate_zero_field():
    config = AdapterConfig(hidden_size=64)
    assert not any("zero" in field.lower() for field in vars(config))


def test_wiring_identity_is_tested_on_a_free_function():
    """`convex_combination` takes the gate as an argument, so a test can force
    `g := 0` without the module carrying a flag that could reach an experiment."""
    parsed = tree("unmark/modeling/adapter.py")
    names = {n.name for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef)}
    assert "convex_combination" in names


# ---------------------------------------------------------------------------
# 5. The neural module does not duplicate deterministic B1A/B2/B3 logic
# ---------------------------------------------------------------------------
def test_adapter_does_not_import_the_deterministic_pipeline():
    modules = imported_modules("unmark/modeling/adapter.py")
    body = source("unmark/modeling/adapter.py")
    assert "unmark.orthography" not in body
    assert "unmark.corruption" not in body
    assert "unmark.linguistics" not in body
    assert "unmark.alignment" not in body
    assert modules <= {"__future__", "dataclasses", "typing", "torch", "unmark"}


@pytest.mark.parametrize("name", NEURAL_MODULES)
def test_no_vncorenlp_or_tokenizer_dependency_in_neural_modules(name):
    modules = imported_modules(name)
    assert not modules & {"py_vncorenlp", "vncorenlp", "transformers", "datasets"}
    assert "unicodedata" not in modules, "no second Unicode implementation"


def test_collate_consumes_projections_rather_than_text():
    """The collator is the seam: metadata in, tensors out.

    Structural: it imports the projection type, and calls no tokenization,
    decomposition or corruption routine.
    """
    assert "TokenOrthographyProjection" in {
        alias.name
        for node in tree("unmark/modeling/collate.py").body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    calls = called_names("unmark/modeling/collate.py")
    assert not calls & {"tokenize", "decompose", "canon", "corrupt", "encode", "align_chunk"}


# ---------------------------------------------------------------------------
# 6. Nothing trains
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEURAL_MODULES + (PROBE,))
def test_no_optimizer_or_training_loop(name):
    calls = called_names(name)
    assert not calls & {"step", "zero_grad_and_step", "save_pretrained", "save"}, (
        f"{name} calls an optimizer/checkpoint routine"
    )
    assert "optim" not in imported_modules(name)
    parsed = tree(name)
    attributes = {
        node.attr for node in ast.walk(parsed) if isinstance(node, ast.Attribute)
    }
    assert not attributes & {"lr_scheduler", "AdamW", "SGD"}


def test_probe_performs_at_most_one_backward():
    body = source(PROBE)
    assert body.count(".backward()") == 1
    assert "training_performed" in body


# ---------------------------------------------------------------------------
# 7. Gate initialisation is the locked one
# ---------------------------------------------------------------------------
def test_gate_initialisation_uses_the_locked_constants():
    body = source("unmark/modeling/adapter.py")
    assert "self.gate.weight.fill_(GATE_INIT_WEIGHT)" in body
    assert "self.gate.bias.fill_(GATE_INIT_BIAS)" in body
    assert GATE_INIT_BIAS == pytest.approx(-4.59511985013459, abs=1e-12)


def test_fusion_and_layernorm_keep_pytorch_defaults():
    """No invented initialisation for W_f or the adapter LayerNorm: the proposal
    locks none, and adding one would be a new hyperparameter."""
    body = source("unmark/modeling/adapter.py")
    parsed = tree("unmark/modeling/adapter.py")
    initialisers = {
        node.func.attr
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"xavier_uniform_", "kaiming_normal_", "normal_", "trunc_normal_"} & initialisers
    assert "conventional defaults" in body


# ---------------------------------------------------------------------------
# 8. Parameter formula agreement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [64, 256, 768, 1024])
def test_parameter_formula_matches_the_module_shapes(d):
    """Derived from the declared module shapes rather than restated."""
    config = AdapterConfig(hidden_size=d)
    expected = (
        TONE_TABLE_ROWS * d          # tone table
        + LETTER_TABLE_ROWS * d      # letter table
        + 3 * d * d + d              # W_f + c_f
        + 3 * d * d + d              # W_g + c_g
        + 2 * d                      # adapter LayerNorm
    )
    assert expected == 6 * d * d + 16 * d
    assert config.parameter_count().total == expected


def test_probe_derives_hidden_size_from_the_model():
    body = source(PROBE)
    assert "model.config.hidden_size" in body
    assert "768" not in body, "the probe must not hardcode a backbone dimension"


# ---------------------------------------------------------------------------
# 9. Stage-1 pooling contract stays locked
# ---------------------------------------------------------------------------
def test_pooling_excludes_padding_and_special_tokens():
    body = source("unmark/modeling/pooling.py")
    assert "keep & ~special" in body
    assert "Stage1PoolingError" in body


def test_pooling_fails_loud_on_zero_content():
    body = source("unmark/modeling/pooling.py")
    assert "counts == 0" in body
    assert "falling back to <s>, an unmasked mean, or a zero vector" in body


def test_pooling_has_no_loss_or_training_code():
    """Only the pooling utility. The Stage-1 objective belongs to a later phase.

    Checked structurally: no loss function is defined, and nothing calls
    `backward` or a distance/optimizer routine. Docstring prose describing what
    the module does not do is not evidence either way.
    """
    parsed = tree("unmark/modeling/pooling.py")
    defined = {
        node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)
    }
    assert defined == {"content_mask", "masked_mean_non_special"}
    calls = called_names("unmark/modeling/pooling.py")
    assert not calls & {"backward", "cosine_similarity", "step", "cross_entropy"}


# ---------------------------------------------------------------------------
# 10. D-B3B0-002 is still open
# ---------------------------------------------------------------------------
def test_backbone_decision_remains_open():
    decisions = source("docs/spec/decisions.md")
    assert "D-B3B0-002" in decisions
    assert "REMAINS OPEN" in decisions or "remains OPEN" in decisions


def test_probe_records_the_revision_as_a_probe_revision():
    body = source(PROBE)
    assert "D-B3B0-002 remains OPEN" in body
    assert "B4B_PROBE_REVISION" in body


# ---------------------------------------------------------------------------
# 11. Collator metadata layer -- real behaviour, no torch
# ---------------------------------------------------------------------------
def test_tone_na_becomes_the_sentinel_with_a_false_mask():
    assert tone_id_and_mask(TokenToneLabel.NA) == (TONE_NA_SENTINEL, False)
    for label in TokenToneLabel:
        identifier, live = tone_id_and_mask(label)
        if label is TokenToneLabel.NA:
            continue
        assert live and 0 <= identifier < TONE_TABLE_ROWS


def test_unmarked_is_a_real_row_not_the_sentinel():
    identifier, live = tone_id_and_mask(TokenToneLabel.UNMARKED)
    assert live and identifier == OBSERVABLE_TONE_IDS["UNMARKED"]


def projections_for(text: str):
    parts = decompose(text)
    labels = character_letter_labels(parts)
    base = parts.base_text
    regions, cursor = [], 0
    for span in parts.syllables:
        if span.base_start > cursor:
            regions.append(
                OrthographicRegion(
                    len(regions), base[cursor : span.base_start], cursor, span.base_start,
                    Eligibility.NOT_APPLICABLE, is_syllable=False,
                )
            )
        regions.append(
            OrthographicRegion(
                len(regions), span.base_text, span.base_start, span.base_end,
                Eligibility.VIETNAMESE_CANDIDATE,
            )
        )
        cursor = span.base_end
    if cursor < len(base):
        regions.append(
            OrthographicRegion(
                len(regions), base[cursor:], cursor, len(base),
                Eligibility.NOT_APPLICABLE, is_syllable=False,
            )
        )
    tones = {
        r.index: s.observed_tone
        for r in regions if r.is_syllable
        for s in parts.syllables if s.base_start == r.start
    }
    out = []
    for chunk in whitespace_chunks(base):
        alignment = align_chunk(chunk, [chunk.text], [7])
        overlays = overlay_orthography(alignment.pieces, regions)
        for piece, overlay in zip(alignment.pieces, overlays):
            out.append(project_piece(len(out), piece, overlay, base, labels, regions, tones))
    return out


def test_build_example_places_projections_in_non_special_slots():
    projections = projections_for("học tập")
    example = build_example([0, 11, 12, 2], [1, 0, 0, 1], projections)
    assert example.length == 4
    assert example.tone_mask == [False, True, True, False]
    assert example.tone_ids[0] == TONE_NA_SENTINEL
    assert example.tone_ids[-1] == TONE_NA_SENTINEL
    assert example.tone_ids[1] == OBSERVABLE_TONE_IDS[ObservedTone.NANG.value]


def test_special_tokens_get_na_in_both_channels():
    projections = projections_for("học")
    example = build_example([0, 11, 2], [1, 0, 1], projections)
    assert example.letter_ids[0] == [] and example.letter_ids[-1] == []
    assert not example.tone_mask[0] and not example.tone_mask[-1]


def test_projection_count_mismatch_fails_loud():
    projections = projections_for("học")
    with pytest.raises(ValueError, match="disagree"):
        build_example([0, 11, 12, 2], [1, 0, 0, 1], projections)


def test_letter_contributors_include_none_and_exclude_na():
    projections = projections_for("học.")
    ids = letter_contributor_ids(projections[0])
    assert LETTER_LABEL_IDS["NONE"] in ids
    assert all(0 <= i < LETTER_TABLE_ROWS for i in ids)
    # "hoc." is four characters but only three are applicable.
    assert len(ids) == 3


def test_padding_rows_are_na_in_both_channels():
    long_example = build_example([0, 11, 12, 2], [1, 0, 0, 1], projections_for("học tập"))
    short_example = build_example([0, 11, 2], [1, 0, 1], projections_for("học"))
    batch = padded_batch([long_example, short_example], pad_token_id=1)
    assert batch["attention_mask"][1] == [1, 1, 1, 0]
    assert batch["special_tokens_mask"][1] == [1, 0, 1, 1]
    assert batch["tone_ids"][1][-1] == TONE_NA_SENTINEL
    assert batch["tone_mask"][1][-1] is False
    assert not any(batch["letter_mask"][1][-1])


def test_k_axis_is_the_batch_maximum_not_a_constant():
    short = build_example([0, 11, 2], [1, 0, 1], projections_for("hạ"))
    long = build_example([0, 11, 2], [1, 0, 1], projections_for("nghiêng"))
    small = padded_batch([short], pad_token_id=1)
    big = padded_batch([short, long], pad_token_id=1)
    assert len(small["letter_ids"][0][1]) < len(big["letter_ids"][0][1])


def test_collating_an_empty_batch_fails():
    with pytest.raises(ValueError, match="empty batch"):
        padded_batch([], pad_token_id=1)


def test_inconsistent_example_lengths_fail():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        EncodedExample([1, 2], [0, 0], [0], [True, True], [[], []])


# ---------------------------------------------------------------------------
# 12. Torch-gated numerics -- skipped cleanly without torch
# ---------------------------------------------------------------------------
try:  # torch is absent from the local .venv by design
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


def build_adapter(d: int = 16):
    from unmark.modeling.adapter import OrthographyInputAdapter

    torch.manual_seed(0)
    return OrthographyInputAdapter(AdapterConfig(hidden_size=d))


@requires_torch
def test_runtime_initial_gate_is_one_percent():
    adapter = build_adapter()
    e = torch.randn(2, 3, 16)
    tone_ids = torch.zeros(2, 3, dtype=torch.long)
    tone_mask = torch.ones(2, 3, dtype=torch.bool)
    letter_ids = torch.zeros(2, 3, 2, dtype=torch.long)
    letter_mask = torch.ones(2, 3, 2, dtype=torch.bool)
    gate = adapter.gate_values(e, tone_ids, tone_mask, letter_ids, letter_mask)
    assert torch.allclose(gate, torch.full_like(gate, 0.01), atol=1e-7)


@requires_torch
def test_runtime_tone_na_is_exactly_zero():
    adapter = build_adapter()
    tone_ids = torch.tensor([[-1, 3]])
    tone_mask = torch.tensor([[False, True]])
    t = adapter.tone_channel(tone_ids, tone_mask)
    assert torch.equal(t[0, 0], torch.zeros(16))
    assert not torch.equal(t[0, 1], torch.zeros(16))


@requires_torch
def test_runtime_empty_letter_channel_is_exactly_zero():
    adapter = build_adapter()
    letter_ids = torch.tensor([[[-1, -1], [0, 2]]])
    letter_mask = torch.tensor([[[False, False], [True, True]]])
    l = adapter.letter_channel(letter_ids, letter_mask)
    assert torch.equal(l[0, 0], torch.zeros(16))
    assert torch.isfinite(l).all(), "0/0 must not produce NaN"


@requires_torch
def test_runtime_letter_mean_includes_none_and_excludes_masked():
    adapter = build_adapter()
    rows = adapter.letter_embedding.weight
    letter_ids = torch.tensor([[[0, 2, -1]]])
    letter_mask = torch.tensor([[[True, True, False]]])
    pooled = adapter.letter_channel(letter_ids, letter_mask)
    assert torch.allclose(pooled[0, 0], (rows[0] + rows[2]) / 2, atol=1e-6)


@requires_torch
def test_runtime_out_of_range_unmasked_id_fails_loud():
    from unmark.modeling.adapter import ChannelContractViolation

    adapter = build_adapter()
    with pytest.raises(ChannelContractViolation, match=r"outside \[0, 7\)"):
        adapter.tone_channel(torch.tensor([[7]]), torch.tensor([[True]]))
    with pytest.raises(ChannelContractViolation, match=r"outside \[0, 5\)"):
        adapter.letter_channel(torch.tensor([[[5]]]), torch.tensor([[[True]]]))


@requires_torch
def test_runtime_masked_sentinel_cannot_change_the_result():
    adapter = build_adapter()
    mask = torch.tensor([[False, True]])
    a = adapter.tone_channel(torch.tensor([[-1, 2]]), mask)
    b = adapter.tone_channel(torch.tensor([[0, 2]]), mask)
    assert torch.equal(a, b), "the substituted row must not leak through the mask"


@requires_torch
def test_runtime_forced_gate_zero_wiring_identity():
    from unmark.modeling.adapter import convex_combination

    e = torch.randn(2, 4, 16)
    f = torch.randn(2, 4, 16)
    assert torch.equal(convex_combination(torch.zeros_like(f), f, e), e)


@requires_torch
def test_runtime_initialised_adapter_is_not_identity():
    adapter = build_adapter()
    e = torch.randn(2, 3, 16)
    z = adapter(
        e,
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, 2, dtype=torch.long),
        torch.ones(2, 3, 2, dtype=torch.bool),
    )
    assert z.shape == e.shape
    assert not torch.allclose(z, e), "g = 0.01 is close to base-only, not equal to it"


@requires_torch
def test_runtime_parameter_count_matches_the_formula():
    for d in (16, 32):
        adapter = build_adapter(d)
        assert adapter.trainable_parameter_count() == 6 * d * d + 16 * d
        assert adapter.trainable_parameter_count() == adapter.expected_parameter_count()


@requires_torch
def test_runtime_pooling_excludes_padding_and_specials():
    from unmark.modeling.pooling import masked_mean_non_special

    hidden = torch.tensor([[[10.0], [2.0], [4.0], [99.0]]])
    attention = torch.tensor([[1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1]])
    pooled = masked_mean_non_special(hidden, attention, special)
    assert pooled.shape == (1, 1)
    assert pooled.item() == pytest.approx(3.0)


@requires_torch
def test_runtime_pooling_fails_loud_on_zero_content():
    from unmark.modeling.contracts import Stage1PoolingError
    from unmark.modeling.pooling import masked_mean_non_special

    hidden = torch.randn(2, 3, 4)
    attention = torch.tensor([[1, 1, 1], [1, 1, 1]])
    special = torch.tensor([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(Stage1PoolingError, match=r"examples \[1\]"):
        masked_mean_non_special(hidden, attention, special)


@requires_torch
def test_runtime_pooling_handles_unequal_branch_lengths():
    from unmark.modeling.pooling import masked_mean_non_special

    short = masked_mean_non_special(
        torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.long),
        torch.tensor([[1, 0, 0, 0, 1]]),
    )
    long = masked_mean_non_special(
        torch.randn(1, 9, 8), torch.ones(1, 9, dtype=torch.long),
        torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 1]]),
    )
    assert short.shape == long.shape == (1, 8)


# ---------------------------------------------------------------------------
# 13. Gap 1 — the gradient path must traverse the frozen encoder
# ---------------------------------------------------------------------------
def _function_node(name: str, module: str) -> ast.FunctionDef:
    for node in ast.walk(tree(module)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module} does not define {name}()")


def test_adapted_forward_does_not_detach_z():
    """A `detach()` between adapter and encoder would sever Stage-1's gradient
    path while every local numeric test still passed."""
    for name in ("forward", "adapted_embeddings"):
        node = _function_node(name, "unmark/modeling/adapter.py")
        detaches = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "detach"
        ]
        assert not detaches, f"UnmarkEncoder.{name} detaches the graph"


def test_adapter_module_never_uses_no_grad():
    """The adapter is the trainable component; nothing in it may be run under
    `no_grad`. `reset_gate_parameters` legitimately uses it for initialisation
    only, which is not part of any forward path."""
    for node in ast.walk(tree("unmark/modeling/adapter.py")):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "forward", "adapted_embeddings", "tone_channel", "letter_channel"
        }:
            withs = [
                item
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.With)
                for item in stmt.items
                if "no_grad" in ast.unparse(item.context_expr)
            ]
            assert not withs, f"{node.name} runs under no_grad"


def test_probe_gradient_loss_comes_from_encoder_output_not_z():
    """The repaired probe: the backward pass must start downstream of the real
    encoder. `z.sum().backward()` would pass even with a severed graph."""
    body = source(PROBE)
    assert "z_grad.sum().backward()" not in body, "the superseded z-only probe is back"
    assert "diagnostic_loss.backward()" in body
    assert "grad_outputs.last_hidden_state" in body
    assert 'gradient_loss_source": "encoder_final_hidden_state"' in body
    assert '"gradient_path_includes_encoder": True' in body


def test_probe_gradient_path_uses_the_real_integration_wrapper():
    """Not a hand-rolled forward: the path Stage-1 will actually use."""
    body = source(PROBE)
    assert "grad_outputs = wrapped(" in body


def test_probe_gradient_backward_is_not_inside_no_grad():
    """Structural: walk the enclosing statements of the backward call and assert
    no `no_grad` context wraps it."""
    parsed = tree(PROBE)
    for node in ast.walk(parsed):
        if isinstance(node, ast.With):
            uses_no_grad = any(
                "no_grad" in ast.unparse(item.context_expr) for item in node.items
            )
            if not uses_no_grad:
                continue
            inner = ast.unparse(node)
            assert "backward()" not in inner, "the backward pass is inside a no_grad block"
            assert "grad_outputs = wrapped(" not in inner, (
                "the gradient-routing forward is inside a no_grad block"
            )


def test_equivalence_control_may_still_use_no_grad():
    """The frozen-model control is inference-only; `no_grad` there is correct.
    This test exists so a future edit does not over-correct Gap 1 by banning
    `no_grad` from the probe entirely."""
    body = source(PROBE)
    assert "with torch.no_grad():" in body
    assert "model.eval()" in body


def test_probe_checks_are_computed_not_hardcoded():
    body = source(PROBE)
    assert "complete = all(checks.values())" in body
    assert "len(summary['checks'])" in body, "the report should state the actual check count"


# ---------------------------------------------------------------------------
# 14. Gap 2 — the frozen encoder must stay in eval across mode changes
# ---------------------------------------------------------------------------
def test_unmark_encoder_overrides_train():
    node = _function_node("train", "unmark/modeling/adapter.py")
    assert node.args.args[1].arg == "mode"
    body = ast.unparse(node)
    assert "super().train(mode)" in body
    assert "self.encoder.eval()" in body
    assert "return self" in body, "nn.Module.train returns self"


def test_train_override_does_not_touch_requires_grad():
    """Mode and freezing are different contracts; the override must not blur
    them by also flipping requires_grad."""
    node = _function_node("train", "unmark/modeling/adapter.py")
    body = ast.unparse(node)
    assert "requires_grad" not in body


def test_train_override_has_no_config_flag():
    """A frozen representation encoder running dropout is not something anyone
    should be able to select by accident."""
    node = _function_node("train", "unmark/modeling/adapter.py")
    assert len(node.args.args) == 2, "train(self, mode) only -- no escape hatch"


def test_probe_records_mode_invariants():
    body = source(PROBE)
    assert "mode_snapshot(" in body
    for step in ("constructed", "wrapper.train()", "wrapper.eval()", "wrapper.train() again"):
        assert step in body
    assert "encoder_always_eval" in body
    assert "adapter_follows_mode" in body


# ---------------------------------------------------------------------------
# Test-only position-profile seam (Audit 024 / C24-1-R2B, Group A)
# ---------------------------------------------------------------------------
def grant_test_only_position_profile(monkeypatch, encoder_class_name: str):
    """Let ONE synthetic encoder past the verified-profile gate, for this test only.

    Production refuses any backbone whose `inputs_embeds` position semantics were
    never measured (D-B4B-002; D-B3B0-002 is OPEN), and that fail-closed rule is
    **not** relaxed here: the profile below is constructed locally and is
    deliberately **never added to `VERIFIED_POSITION_PROFILES`**. Only the
    resolver is patched, and only inside the test that calls this.

    The properties these tests name -- train/eval mode, encoder freezing, gradient
    connectivity -- are independent of *which* backbone is permitted. The
    permission check has its own dedicated tests, which still run unpatched:
    `test_runtime_wrapper_refuses_an_unverified_backbone` and friends.
    """
    from unmark.modeling import adapter as adapter_module

    profile = adapter_module.VerifiedPositionProfile(
        checkpoint="TEST_ONLY_synthetic",
        model_type="roberta",
        model_class=encoder_class_name,
        position_rule="roberta_input_ids_offset",
        evidence="TEST-ONLY fixture. Not a measured backbone. Never registered.",
    )
    monkeypatch.setattr(adapter_module, "resolve_position_profile", lambda _enc: profile)
    return profile


def tiny_encoder_config():
    """A minimal `config`/`embeddings` surface, so padding-index detection is
    genuinely exercised rather than patched away."""
    import types

    return types.SimpleNamespace(model_type="roberta", pad_token_id=1)


@requires_torch
def test_runtime_train_mode_invariants(monkeypatch):
    from torch import nn

    from unmark.modeling.adapter import UnmarkEncoder

    class TinyEncoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, d, padding_idx=1)
            self.dropout = nn.Dropout(0.5)
            self.config = tiny_encoder_config()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, **_):
            return self.dropout(inputs_embeds)

    grant_test_only_position_profile(monkeypatch, "TinyEncoder")
    encoder = TinyEncoder(16)
    wrapper = UnmarkEncoder(encoder, build_adapter(16))

    assert encoder.training is False, "frozen encoder must start in eval"

    returned = wrapper.train()
    assert returned is wrapper, "train() must return self"
    assert wrapper.training is True
    assert wrapper.adapter.training is True
    assert encoder.training is False, "wrapper.train() reactivated encoder dropout"

    wrapper.eval()
    assert wrapper.training is False
    assert wrapper.adapter.training is False
    assert encoder.training is False

    # Repeated transitions must preserve the invariant.
    for _ in range(3):
        wrapper.train()
        assert encoder.training is False and wrapper.adapter.training is True
        wrapper.eval()
        assert encoder.training is False and wrapper.adapter.training is False


@requires_torch
def test_runtime_encoder_stays_frozen_across_mode_changes(monkeypatch):
    from torch import nn

    from unmark.modeling.adapter import UnmarkEncoder, trainable_parameters

    class TinyEncoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, d, padding_idx=1)
            self.config = tiny_encoder_config()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, **_):
            return inputs_embeds

    grant_test_only_position_profile(monkeypatch, "TinyEncoder")
    encoder = TinyEncoder(16)
    adapter = build_adapter(16)
    wrapper = UnmarkEncoder(encoder, adapter)

    for _ in range(2):
        wrapper.train()
        wrapper.eval()
    assert trainable_parameters(encoder) == 0
    assert all(not p.requires_grad for p in encoder.parameters())
    assert trainable_parameters(adapter) == 6 * 16 * 16 + 16 * 16


@requires_torch
def test_runtime_gradients_reach_the_adapter_through_a_stand_in_encoder(monkeypatch):
    """Connectivity check with a trivial frozen encoder.

    The real check runs on Colab against PhoBERT; this proves the *wiring* --
    that a loss taken from the encoder's output reaches every adapter component.
    """
    from torch import nn

    from unmark.modeling.adapter import UnmarkEncoder, trainable_parameters

    class TinyEncoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, d, padding_idx=1)
            self.proj = nn.Linear(d, d)
            self.config = tiny_encoder_config()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, **_):
            return self.proj(inputs_embeds)

    grant_test_only_position_profile(monkeypatch, "TinyEncoder")
    encoder = TinyEncoder(16)
    adapter = build_adapter(16)
    wrapper = UnmarkEncoder(encoder, adapter)
    wrapper.train()

    hidden = wrapper(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        tone_ids=torch.tensor([[0, 1, -1]]),
        tone_mask=torch.tensor([[True, True, False]]),
        letter_ids=torch.tensor([[[0, 1], [2, -1], [-1, -1]]]),
        letter_mask=torch.tensor([[[True, True], [True, False], [False, False]]]),
    )
    assert hidden.requires_grad, "the encoder output must carry a graph to A_phi"
    hidden.sum().backward()

    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient at {name}"
    assert any(float(p.grad.abs().sum()) != 0 for p in adapter.parameters())

    assert trainable_parameters(encoder) == 0
    assert all(p.grad is None for p in encoder.parameters()), "frozen encoder accumulated grads"


@requires_torch
def test_runtime_gate_weight_gradient_exists_despite_zero_init(monkeypatch):
    """`W_g` starts at zero, but its gradient must still be able to exist."""
    from torch import nn

    from unmark.modeling.adapter import UnmarkEncoder

    class TinyEncoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, d, padding_idx=1)
            self.config = tiny_encoder_config()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, **_):
            return inputs_embeds

    grant_test_only_position_profile(monkeypatch, "TinyEncoder")
    adapter = build_adapter(16)
    wrapper = UnmarkEncoder(TinyEncoder(16), adapter)
    wrapper.train()
    out = wrapper(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        tone_ids=torch.tensor([[0, 1]]),
        tone_mask=torch.tensor([[True, True]]),
        letter_ids=torch.tensor([[[0], [1]]]),
        letter_mask=torch.tensor([[[True], [True]]]),
    )
    out.sum().backward()
    assert adapter.gate.weight.grad is not None
    assert adapter.gate.bias.grad is not None
    assert torch.isfinite(adapter.gate.weight.grad).all()


# ---------------------------------------------------------------------------
# Adapted-grid agreement (Audit 024 / C24-1-R2B, Group D)
# ---------------------------------------------------------------------------
@requires_torch
@pytest.mark.parametrize("channel", ["tone_ids", "tone_mask", "letter_ids", "letter_mask"])
def test_runtime_adapted_channels_must_share_the_base_token_grid(channel):
    """A channel off the base grid fails closed, naming the grids.

    Previously this reached `torch.cat` and produced a raw size error, which
    reads as though unequal reference/base lengths were unsupported — the exact
    misreading that hid a fixture defect (§P, Group D).
    """
    from unmark.modeling.adapter import ChannelContractViolation

    adapter = build_adapter(16)
    good = dict(
        base_embeddings=torch.randn(1, 4, 16),
        tone_ids=torch.tensor([[-1, 0, 1, -1]]),
        tone_mask=torch.tensor([[False, True, True, False]]),
        letter_ids=torch.tensor([[[-1], [0], [2], [-1]]]),
        letter_mask=torch.tensor([[[False], [True], [True], [False]]]),
    )
    adapter(**good)  # the aligned case must still work

    bad = dict(good)
    wrong = {
        "tone_ids": torch.tensor([[-1, 0, 1, 2, -1]]),
        "tone_mask": torch.tensor([[False, True, True, True, False]]),
        "letter_ids": torch.tensor([[[-1], [0], [2], [3], [-1]]]),
        "letter_mask": torch.tensor([[[False], [True], [True], [True], [False]]]),
    }[channel]
    bad[channel] = wrong
    with pytest.raises(ChannelContractViolation, match="different token grid"):
        adapter(**bad)
