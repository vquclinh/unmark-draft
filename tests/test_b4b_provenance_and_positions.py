"""B4B repair: model-revision provenance and authoritative position ids.

The first real PhoBERT run passed 21 of 22 checks. The one failure was the
provenance collector, not the model: it asked `model.name_or_path` for a cache
snapshot path, and for a model that attribute is just the repo id. An
independent offline diagnostic confirmed the requested revision *was* loaded.

These tests cover the repair. Local environment stays ML-free; the probe module
imports torch only inside `main()`, so its pure helpers are genuinely testable
here rather than only statically.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBE_PATH = REPO / "scripts" / "b4b_phobert_adapter_probe.py"
ADAPTER_PATH = REPO / "unmark" / "modeling" / "adapter.py"

REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
OTHER_REVISION = "0" * 40
SNAPSHOT = f"/content/unmark-draft/.hf-cache/hub/models--vinai--phobert-base/snapshots/{REVISION}"
BLOB = (
    "/content/unmark-draft/.hf-cache/hub/models--vinai--phobert-base/blobs/"
    "a0b0f0912c710147fbaac015b0a4011216a0061a56c03b840b639e40d3bb49cc"
)


def load_probe() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("b4b_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_probe()


def source(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(source(path))


def stub_model(commit_hash: str | None = REVISION, name_or_path: str = "vinai/phobert-base"):
    config = types.SimpleNamespace(model_type="roberta", pad_token_id=1)
    if commit_hash is not None:
        config._commit_hash = commit_hash
    return types.SimpleNamespace(config=config, name_or_path=name_or_path)


def with_cache(monkeypatch, mapping: dict[tuple[str, str], str | None]) -> None:
    """Simulate the HF cache: (filename, revision) -> raw path or None."""
    monkeypatch.setattr(
        probe, "cached_artifact_path", lambda ckpt, filename, rev: mapping.get((filename, rev))
    )
    monkeypatch.setattr(probe, "read_main_ref", lambda ckpt: OTHER_REVISION)


def full_cache(revision: str = REVISION) -> dict[tuple[str, str], str]:
    snapshot = SNAPSHOT if revision == REVISION else SNAPSHOT.replace(REVISION, revision)
    return {
        ("config.json", revision): f"{snapshot}/config.json",
        ("pytorch_model.bin", revision): f"{snapshot}/pytorch_model.bin",
    }


# ---------------------------------------------------------------------------
# 1-4. Revision extraction uses RAW snapshot paths
# ---------------------------------------------------------------------------
def test_revision_is_extracted_from_a_raw_snapshot_path():
    assert probe.extract_snapshot_revision(f"{SNAPSHOT}/pytorch_model.bin") == REVISION
    assert probe.extract_snapshot_revision(f"{SNAPSHOT}/config.json") == REVISION


def test_a_resolved_blob_path_yields_no_revision():
    """`Path.resolve()` follows the snapshot symlink into `blobs/`, which is
    content-addressed and carries no revision. Resolving first would destroy the
    only evidence being collected."""
    assert probe.extract_snapshot_revision(BLOB) is None


def test_the_verifier_never_resolves_before_extracting():
    """Structural: `realpath`/`resolve` may appear only for forensic recording,
    never feeding `extract_snapshot_revision`."""
    for node in ast.walk(tree(PROBE_PATH)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "extract_snapshot_revision"
        ):
            argument = ast.unparse(node.args[0])
            assert "realpath" not in argument and "resolve()" not in argument, (
                f"revision extracted from a resolved path: {argument}"
            )


def test_repo_id_alone_is_not_a_snapshot_path():
    assert probe.extract_snapshot_revision("vinai/phobert-base") is None


# ---------------------------------------------------------------------------
# 5-6. Config and weight artifacts must belong to the requested snapshot
# ---------------------------------------------------------------------------
def test_verification_passes_with_config_and_weight_in_the_snapshot(monkeypatch):
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["verified"] is True
    assert evidence["resolved_revision"] == REVISION
    assert evidence["cached_config_revision"] == REVISION
    assert evidence["cached_weight_revision"] == REVISION
    assert evidence["cached_weight_kind"] == "pytorch_model.bin"
    assert evidence["failure_reasons"] == []


def test_missing_weight_provenance_fails_verification(monkeypatch):
    """Config alone does not prove the *weights* came from that revision."""
    with_cache(monkeypatch, {("config.json", REVISION): f"{SNAPSHOT}/config.json"})
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["verified"] is False
    assert evidence["cached_weight_raw_path"] is None
    assert any("weight artifact" in reason for reason in evidence["failure_reasons"])


def test_missing_config_provenance_fails_verification(monkeypatch):
    with_cache(monkeypatch, {("pytorch_model.bin", REVISION): f"{SNAPSHOT}/pytorch_model.bin"})
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["verified"] is False
    assert any("cached config" in reason for reason in evidence["failure_reasons"])


def test_safetensors_and_sharded_layouts_are_accepted(monkeypatch):
    for kind in ("model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json"):
        with_cache(
            monkeypatch,
            {
                ("config.json", REVISION): f"{SNAPSHOT}/config.json",
                (kind, REVISION): f"{SNAPSHOT}/{kind}",
            },
        )
        evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
        assert evidence["verified"] is True, kind
        assert evidence["cached_weight_kind"] == kind


# ---------------------------------------------------------------------------
# 7. A disagreeing config commit hash fails loudly
# ---------------------------------------------------------------------------
def test_config_commit_hash_disagreement_fails(monkeypatch):
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(
        stub_model(commit_hash=OTHER_REVISION), "vinai/phobert-base", REVISION
    )
    assert evidence["verified"] is False
    assert evidence["config_commit_hash_matches"] is False
    assert any("_commit_hash" in reason for reason in evidence["failure_reasons"])


def test_absent_commit_hash_does_not_fail_verification(monkeypatch):
    """`_commit_hash` is private and transformers may drop it; its *absence* is
    not evidence of anything, while a *disagreement* is."""
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(
        stub_model(commit_hash=None), "vinai/phobert-base", REVISION
    )
    assert evidence["config_commit_hash_present"] is False
    assert evidence["verified"] is True


# ---------------------------------------------------------------------------
# 8. refs/main is recorded but never required
# ---------------------------------------------------------------------------
def test_refs_main_is_not_a_verification_condition(monkeypatch):
    """Upstream `main` may legitimately move while the pinned snapshot stays
    correct. Requiring a match would fail a valid pinned run for an unrelated
    reason."""
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["refs_main"] == OTHER_REVISION
    assert evidence["refs_main_is_required"] is False
    assert evidence["verified"] is True


# ---------------------------------------------------------------------------
# 9-10. name_or_path, and a wrong requested revision
# ---------------------------------------------------------------------------
def test_name_or_path_alone_is_insufficient(monkeypatch):
    """The exact defect that failed the first real run."""
    with_cache(monkeypatch, {})
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["name_or_path"] == "vinai/phobert-base"
    assert evidence["name_or_path_is_revision_evidence"] is False
    assert evidence["verified"] is False


def test_the_old_collector_is_not_used_for_the_model():
    body = source(PROBE_PATH)
    assert 'observe_revision(model, ("name_or_path",))' not in body
    assert "verify_model_revision(model," in body


def test_requesting_a_different_revision_fails(monkeypatch):
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(
        stub_model(commit_hash=REVISION), "vinai/phobert-base", OTHER_REVISION
    )
    assert evidence["verified"] is False


def test_blob_path_is_recorded_but_flagged_as_non_evidence(monkeypatch):
    with_cache(monkeypatch, full_cache())
    evidence = probe.verify_model_revision(stub_model(), "vinai/phobert-base", REVISION)
    assert evidence["weight_blob_is_not_revision_evidence"] is True


def test_hub_cache_root_is_not_hf_home_itself():
    """`HF_HOME=/x` means the hub cache is `/x/hub`; passing `/x` makes
    transformers look for `/x/models--...` and silently find nothing."""
    body = source(PROBE_PATH)
    assert "HF_HUB_CACHE" in body
    assert '"hub"' in body


# ---------------------------------------------------------------------------
# 11-13. D-B4B-002 closure and the padding index
# ---------------------------------------------------------------------------
def test_d_b4b_002_is_closed_in_the_decision_log():
    decisions = source(REPO / "docs" / "spec" / "decisions.md")
    assert "D-B4B-002" in decisions
    marker = decisions[decisions.index("### D-B4B-002") : decisions.index("### D-B4B-003")]
    assert "CLOSED" in marker
    assert "explicit" in marker.lower() and "position_ids" in marker


def test_padding_index_is_read_from_the_model():
    pytest.importorskip("torch", reason="adapter imports torch")
    from unmark.modeling import adapter as adapter_module  # noqa: PLC0415

    embeddings = types.SimpleNamespace(padding_idx=7)
    encoder = types.SimpleNamespace(embeddings=embeddings, config=types.SimpleNamespace(pad_token_id=1))
    assert adapter_module.detect_padding_index(encoder) == 7


def test_padding_index_is_not_hardcoded_anywhere():
    """The module-level helper must derive it; the wrapper method delegates."""
    checked = False
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name == "authoritative_position_ids":
            body = ast.unparse(node)
            arguments = {a.arg for a in node.args.args}
            if "encoder" in arguments:  # the module-level helper
                assert "detect_padding_index" in body
                checked = True
            assert "padding_idx=1" not in body
    assert checked, "the module-level authoritative_position_ids helper was not found"


def test_hidden_size_is_not_hardcoded_in_the_adapter():
    literals = [
        node.value
        for node in ast.walk(tree(ADAPTER_PATH))
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    ]
    assert 768 not in literals


# ---------------------------------------------------------------------------
# 14-17. The generic adapter stays position-agnostic
# ---------------------------------------------------------------------------
def test_orthography_adapter_contains_no_position_logic():
    """Position semantics are checkpoint-specific; the orthography adapter is
    backbone-independent and must stay that way (D-B3B0-002 is OPEN)."""
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.ClassDef) and node.name == "OrthographyInputAdapter":
            body = ast.unparse(node)
            for banned in (
                "position_ids", "padding_idx", "roberta", "position_embeddings",
                "phobert", "checkpoint", "VerifiedPositionProfile",
            ):
                assert banned not in body, f"OrthographyInputAdapter mentions {banned}"


def test_position_helpers_live_in_the_integration_layer():
    names = {n.name for n in ast.walk(tree(ADAPTER_PATH)) if isinstance(n, ast.FunctionDef)}
    assert {
        "detect_padding_index",
        "detect_model_family",
        "detect_checkpoint",
        "resolve_position_profile",
        "roberta_position_ids_from_input_ids",
        "authoritative_position_ids",
    } <= names


def test_only_the_measured_checkpoint_is_claimed_verified():
    """Exactly one profile, because exactly one was measured.

    An earlier version trusted `model_type == "roberta"` wholesale, which
    asserted an empirical result never obtained and contradicted this audit's own
    statement that a second backbone needs its own measurement.
    """
    pytest.importorskip("torch", reason="adapter imports torch")
    from unmark.modeling import adapter as adapter_module  # noqa: PLC0415

    assert not hasattr(adapter_module, "VERIFIED_POSITION_FAMILIES")
    profiles = adapter_module.VERIFIED_POSITION_PROFILES
    assert len(profiles) == 1
    only = profiles[0]
    assert only.checkpoint == "vinai/phobert-base"
    assert only.model_type == "roberta"
    assert only.model_class == "RobertaModel"


def test_a_profile_requires_checkpoint_type_and_class_together():
    pytest.importorskip("torch", reason="adapter imports torch")
    from unmark.modeling.adapter import PHOBERT_BASE_POSITION_PROFILE as profile

    assert profile.matches("vinai/phobert-base", "roberta", "RobertaModel")
    # A different checkpoint of the same family is NOT covered.
    assert not profile.matches("roberta-base", "roberta", "RobertaModel")
    assert not profile.matches("vinai/phobert-large", "roberta", "RobertaModel")
    assert not profile.matches("vinai/phobert-base", "bert", "BertModel")
    assert not profile.matches("vinai/phobert-base", "roberta", "SomeCustomModel")
    assert not profile.matches(None, "roberta", "RobertaModel")


def test_unverified_backbone_fails_loud_in_source():
    body = source(ADAPTER_PATH)
    assert "UnsupportedPositionSemantics" in body
    assert "no verified inputs_embeds position profile" in body
    assert "A shared model_type is NOT sufficient evidence" in body


def test_checkpoint_identity_is_separate_from_revision_verification():
    """`name_or_path` identifies the checkpoint but never the revision
    (D-B4B-006). Neither layer substitutes for the other."""
    body = source(ADAPTER_PATH)
    assert "detect_checkpoint" in body
    assert "not the revision" in body
    probe_body = source(PROBE_PATH)
    assert "verify_model_revision" in probe_body


def test_backbone_decision_remains_open():
    decisions = source(REPO / "docs" / "spec" / "decisions.md")
    assert "D-B3B0-002" in decisions
    assert "remains OPEN" in decisions or "REMAINS OPEN" in decisions


# ---------------------------------------------------------------------------
# 18-19. Positions passed once, never baked into z
# ---------------------------------------------------------------------------
def test_position_ids_are_passed_to_the_encoder_exactly_once():
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            if "inputs_embeds" not in body:
                continue
            assert body.count("encoder_kwargs['position_ids']") == 1
            assert "self.encoder(inputs_embeds=z" in body


def test_no_positional_embedding_is_added_into_z():
    """§4.5: adding position encodings inside the module double-counts them, and
    the failure is silent."""
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name in {"forward", "adapted_embeddings"}:
            body = ast.unparse(node)
            assert "position_embeddings" not in body
            assert "+ position" not in body


def test_wrapper_derives_positions_when_the_caller_omits_them():
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            if "inputs_embeds" not in body:
                continue
            assert "if position_ids is None:" in body
            assert "self.authoritative_position_ids(input_ids)" in body


def test_wrapper_validates_rather_than_trusts_supplied_positions():
    """A supplied tensor is checked, not honoured. Honouring it would reopen the
    hole D-B4B-002 closed."""
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            if "inputs_embeds" not in body:
                continue
            assert "_require_authoritative_positions" in body
    body = source(ADAPTER_PATH)
    assert "class PositionContractViolation" in body


def test_position_comparison_is_exact_integer_equality():
    """Indices, not values -- no floating tolerance."""
    for node in ast.walk(tree(ADAPTER_PATH)):
        if isinstance(node, ast.FunctionDef) and node.name == "_require_authoritative_positions":
            body = ast.unparse(node)
            assert "torch.equal" in body
            assert "shape" in body
            for banned in ("allclose", "atol", "rtol", "isclose"):
                assert banned not in body, f"tolerance-based comparison: {banned}"


def test_probe_path_c_uses_the_raw_encoder():
    """Path C intentionally compares arbitrary explicit position tensors, so it
    calls the frozen model directly rather than weakening the wrapper."""
    body = source(PROBE_PATH)
    compare = body[body.index("def compare_paths(") : body.index("def tensor_diff(")]
    assert "model(" in compare
    assert "wrapped(" not in compare and "UnmarkEncoder" not in compare


def test_probe_checks_the_wrapper_rejects_a_wrong_override():
    body = source(PROBE_PATH)
    assert "wrong_override_rejected" in body
    assert "PositionContractViolation" in body
    assert "wrapper_passes_authoritative_ids" in body


def test_probe_omits_position_ids_on_the_gradient_path():
    """The gradient forward exercises the wrapper's own derivation, which is the
    production-safety property."""
    body = source(PROBE_PATH)
    grad_call = body[body.index("grad_outputs = wrapped(") : body.index("grad_hidden =")]
    assert "position_ids=" not in grad_call


def test_probe_checks_the_derived_ids_against_the_model():
    body = source(PROBE_PATH)
    assert "derived_matches_model" in body
    assert "roberta_position_ids_from_input_ids" in body


# ---------------------------------------------------------------------------
# 20-22. Regression: earlier repairs stay intact
# ---------------------------------------------------------------------------
def test_gradient_routing_repair_is_intact():
    body = source(PROBE_PATH)
    assert "diagnostic_loss.backward()" in body
    assert "z_grad.sum().backward()" not in body
    assert 'gradient_loss_source": "encoder_final_hidden_state"' in body


def test_frozen_encoder_mode_enforcement_is_intact():
    body = source(ADAPTER_PATH)
    assert "super().train(mode)" in body
    assert "self.encoder.eval()" in body


def test_no_optimizer_or_training_was_introduced():
    for path in (PROBE_PATH, ADAPTER_PATH):
        parsed = tree(path)
        calls = {
            node.func.attr
            for node in ast.walk(parsed)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "step" not in calls and "save_pretrained" not in calls
    backwards = sum(
        1
        for node in ast.walk(tree(PROBE_PATH))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    )
    assert backwards == 1


# ---------------------------------------------------------------------------
# 23. Torch-gated: the position rule reproduces the measured evidence
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:  # pragma: no cover - the normal local state
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is not installed (ML-free local .venv); runs on Colab"
)


@requires_torch
def test_runtime_roberta_position_ids_match_the_measured_evidence():
    """The real run recorded `2, 3, 4, 5, 6, 1, 1, 1` for a right-padded
    sequence, against the sequential `2, 3, 4, 5, 6, 7, 8, 9` the implicit
    inputs_embeds path produced."""
    from unmark.modeling.adapter import roberta_position_ids_from_input_ids

    input_ids = torch.tensor([[0, 10, 11, 12, 2, 1, 1, 1]])
    positions = roberta_position_ids_from_input_ids(input_ids, padding_idx=1)
    assert positions.tolist() == [[2, 3, 4, 5, 6, 1, 1, 1]]


@requires_torch
def test_runtime_unequal_lengths_get_independent_numbering():
    from unmark.modeling.adapter import roberta_position_ids_from_input_ids

    input_ids = torch.tensor([[0, 10, 11, 2], [0, 10, 2, 1]])
    positions = roberta_position_ids_from_input_ids(input_ids, padding_idx=1)
    assert positions.tolist() == [[2, 3, 4, 5], [2, 3, 4, 1]]


@requires_torch
def test_runtime_wrapper_rejects_an_unverified_backbone():
    from torch import nn

    from unmark.modeling.adapter import UnmarkEncoder, UnsupportedPositionSemantics
    from unmark.modeling.config import AdapterConfig
    from unmark.modeling.adapter import OrthographyInputAdapter

    class MysteryEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, 16)
            self.config = types.SimpleNamespace(model_type="mystery-net", pad_token_id=1)

        def get_input_embeddings(self):
            return self.embed

    with pytest.raises(UnsupportedPositionSemantics, match="mystery-net"):
        UnmarkEncoder(MysteryEncoder(), OrthographyInputAdapter(AdapterConfig(hidden_size=16)))


@requires_torch
def test_runtime_wrapper_supplies_positions_automatically():
    from torch import nn

    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.config import AdapterConfig

    seen: dict[str, object] = {}

    class RobertaLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(20, 16, padding_idx=1)
            self.embeddings = types.SimpleNamespace(padding_idx=1, word_embeddings=self.embed)
            self.config = types.SimpleNamespace(model_type="roberta", pad_token_id=1)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **_):
            seen["position_ids"] = None if position_ids is None else position_ids.tolist()
            return inputs_embeds

    wrapper = UnmarkEncoder(RobertaLike(), OrthographyInputAdapter(AdapterConfig(hidden_size=16)))
    wrapper(
        input_ids=torch.tensor([[0, 10, 11, 2, 1]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        tone_ids=torch.tensor([[-1, 0, 1, -1, -1]]),
        tone_mask=torch.tensor([[False, True, True, False, False]]),
        letter_ids=torch.zeros(1, 5, 1, dtype=torch.long),
        letter_mask=torch.zeros(1, 5, 1, dtype=torch.bool),
    )
    assert seen["position_ids"] == [[2, 3, 4, 5, 1]], "wrapper did not supply authoritative ids"


# ---------------------------------------------------------------------------
# 24. Torch-gated: profile matching and supplied-position validation
# ---------------------------------------------------------------------------
def roberta_like(checkpoint: str, model_type: str = "roberta", class_name: str = "RobertaModel"):
    """A minimal stand-in whose class name is configurable."""
    from torch import nn

    base = type(
        class_name,
        (nn.Module,),
        {
            "__init__": lambda self: (
                nn.Module.__init__(self),
                setattr(self, "embed", nn.Embedding(20, 16, padding_idx=1)),
                setattr(self, "config", types.SimpleNamespace(
                    model_type=model_type, pad_token_id=1, _name_or_path=checkpoint
                )),
                None,
            )[-1],
            "get_input_embeddings": lambda self: self.embed,
            "forward": lambda self, inputs_embeds=None, attention_mask=None, position_ids=None, **_: (
                setattr(self, "seen_positions", None if position_ids is None else position_ids.tolist()),
                inputs_embeds,
            )[-1],
        },
    )
    return base()


def make_wrapper(encoder):
    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.config import AdapterConfig

    return UnmarkEncoder(encoder, OrthographyInputAdapter(AdapterConfig(hidden_size=16)))


def sample_batch():
    return dict(
        input_ids=torch.tensor([[0, 10, 11, 2, 1]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        tone_ids=torch.tensor([[-1, 0, 1, -1, -1]]),
        tone_mask=torch.tensor([[False, True, True, False, False]]),
        letter_ids=torch.zeros(1, 5, 1, dtype=torch.long),
        letter_mask=torch.zeros(1, 5, 1, dtype=torch.bool),
    )


@requires_torch
def test_runtime_phobert_profile_is_accepted():
    wrapper = make_wrapper(roberta_like("vinai/phobert-base"))
    assert wrapper.position_profile.checkpoint == "vinai/phobert-base"


@requires_torch
def test_runtime_other_roberta_checkpoint_is_rejected():
    """The scope repair: a shared model_type is not evidence."""
    from unmark.modeling.adapter import UnsupportedPositionSemantics

    for checkpoint in ("roberta-base", "vinai/phobert-large", "xlm-roberta-base"):
        with pytest.raises(UnsupportedPositionSemantics, match="no verified"):
            make_wrapper(roberta_like(checkpoint))


@requires_torch
def test_runtime_other_family_is_rejected():
    from unmark.modeling.adapter import UnsupportedPositionSemantics

    with pytest.raises(UnsupportedPositionSemantics):
        make_wrapper(roberta_like("vinai/phobert-base", model_type="bert", class_name="BertModel"))


@requires_torch
def test_runtime_unexpected_model_class_is_rejected():
    from unmark.modeling.adapter import UnsupportedPositionSemantics

    with pytest.raises(UnsupportedPositionSemantics):
        make_wrapper(roberta_like("vinai/phobert-base", class_name="PatchedRobertaModel"))


@requires_torch
def test_runtime_correct_supplied_positions_are_accepted():
    encoder = roberta_like("vinai/phobert-base")
    wrapper = make_wrapper(encoder)
    batch = sample_batch()
    derived = wrapper.authoritative_position_ids(batch["input_ids"])
    wrapper(**batch, position_ids=derived)
    assert encoder.seen_positions == [[2, 3, 4, 5, 1]]


@requires_torch
def test_runtime_wrong_supplied_positions_fail_loud():
    from unmark.modeling.adapter import PositionContractViolation

    wrapper = make_wrapper(roberta_like("vinai/phobert-base"))
    batch = sample_batch()
    wrong = wrapper.authoritative_position_ids(batch["input_ids"]) + 1
    with pytest.raises(PositionContractViolation, match="do not match the authoritative"):
        wrapper(**batch, position_ids=wrong)


@requires_torch
def test_runtime_sequential_fallback_ids_are_rejected():
    """The exact tensor the implicit inputs_embeds path would have produced."""
    from unmark.modeling.adapter import PositionContractViolation

    wrapper = make_wrapper(roberta_like("vinai/phobert-base"))
    batch = sample_batch()
    sequential = torch.tensor([[2, 3, 4, 5, 6]])  # numbering straight through padding
    with pytest.raises(PositionContractViolation):
        wrapper(**batch, position_ids=sequential)


@requires_torch
def test_runtime_shape_mismatched_positions_fail_loud():
    from unmark.modeling.adapter import PositionContractViolation

    wrapper = make_wrapper(roberta_like("vinai/phobert-base"))
    batch = sample_batch()
    with pytest.raises(PositionContractViolation, match="shape"):
        wrapper(**batch, position_ids=torch.tensor([[2, 3, 4]]))


@requires_torch
def test_runtime_wrapper_never_lets_the_encoder_default_kick_in():
    """`position_ids` must always reach the encoder, on every path."""
    encoder = roberta_like("vinai/phobert-base")
    wrapper = make_wrapper(encoder)
    wrapper(**sample_batch())
    assert encoder.seen_positions is not None
    assert encoder.seen_positions == [[2, 3, 4, 5, 1]]


# ---------------------------------------------------------------------------
# 25. Documentation consistency for Audit 016
# ---------------------------------------------------------------------------
AUDIT_016 = REPO / "docs" / "audits" / "016-b4b-real-model-provenance-and-position-repair.md"


def test_audit_016_verdict_and_test_count_agree_across_sections():
    """Section A's headline count must match the pytest block in Section P.

    Both were stale once: A kept a pre-repair count while P was updated. A
    summary that disagrees with its own evidence is worse than no summary.
    """
    import re

    text = AUDIT_016.read_text(encoding="utf-8")
    evidence = re.search(r"(\d+) passed, (\d+) skipped", text)
    headline = re.search(r"\*\*(\d+)\s*\n?tests pass, (\d+) skip\*\*", text)
    assert evidence, "Section P no longer records a pytest result"
    assert headline, "Section A no longer states a test count"
    assert headline.groups() == evidence.groups(), (
        f"Section A says {headline.groups()}, Section P says {evidence.groups()}"
    )


def test_audit_016_claims_no_family_wide_position_permission():
    """No current-state sentence may imply family-wide authorization.

    Historical discussion of the repaired behaviour is fine and is kept; these
    phrasings are the ones that would read as *currently* true.
    """
    text = AUDIT_016.read_text(encoding="utf-8")
    for phrase in (
        "only `roberta` claimed verified",
        "unverified families failing loud",
        "Measured families",
        'VERIFIED_POSITION_FAMILIES = frozenset({"roberta"})` — **measured',
    ):
        assert phrase not in text, f"Audit 016 still claims family-wide permission: {phrase!r}"


def test_audit_016_keeps_the_locked_current_state():
    text = AUDIT_016.read_text(encoding="utf-8")
    assert "**PASS — B4B REAL-MODEL RERUN REQUIRED**" in text
    assert "all 27 checks true" in text
    assert "D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains OPEN" in text
    assert "nothing was trained" in text.lower()
    assert "VerifiedPositionProfile" in text


# ---------------------------------------------------------------------------
# 26. The experiment record keeps both runs honestly
# ---------------------------------------------------------------------------
B4B_RESULT = REPO / "docs" / "experiments" / "b4b-phobert-adapter-integration-result.md"


def test_experiment_record_states_the_final_complete_status():
    text = B4B_RESULT.read_text(encoding="utf-8")
    assert "B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE" in text
    assert "27 of 27 checks" in text
    assert "20260820T081554Z" in text
    assert "7f6e26c80c0acfa3cdf9168a9b0e2981e6ae1491" in text


def test_experiment_record_preserves_the_first_run_failure():
    """The 21/22 run is not rewritten as COMPLETE. A record that erases its own
    failures is not evidence."""
    text = B4B_RESULT.read_text(encoding="utf-8")
    assert "B4B_PHOBERT_ADAPTER_INTEGRATION_INCOMPLETE" in text
    assert "21 of 22 checks" in text
    assert "# Run 1" in text and "# Run 2" in text


def test_experiment_record_does_not_overclaim():
    text = B4B_RESULT.read_text(encoding="utf-8")
    assert "integration evidence, not linguistic coverage" in text
    assert "remains **OPEN**" in text or "remains OPEN" in text


def test_b4b_closure_decision_scopes_what_complete_means():
    decisions = (REPO / "docs" / "spec" / "decisions.md").read_text(encoding="utf-8")
    assert "### D-B4B-007 — B4B is COMPLETE" in decisions
    closure = decisions[decisions.index("### D-B4B-007") :]
    assert "Stage-1 objective is not implemented" in closure
    assert "Nothing has been trained" in closure
    assert "is still OPEN" in closure
    assert "PRE-TRAIN audit" in closure
