"""Stage-2 dual-finalist UNMARK infrastructure acceptance tests.

No real downstream rows, no measurement-dev, no official TEST, no model
downloads, and no Stage-2 head training. Torch-dependent checks use synthetic
checkpoints and synthetic modules and skip cleanly when torch is unavailable.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import sys
import zlib
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.evaluation.stage2_dual_finalist as s2  # noqa: E402
from unmark.corruption import (  # noqa: E402
    CorruptionCondition,
    CorruptionPurpose,
    CorruptionScope,
    EligibilityPolicy,
    get_condition,
)
from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import Preg1Role  # noqa: E402
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    MAX_LENGTH,
    PRIMARY_NUM_LABELS,
)
from unmark.modeling.contracts import LETTER_NA_SENTINEL, TONE_NA_SENTINEL  # noqa: E402
from unmark.orthography import Eligibility  # noqa: E402
from unmark.stage1.finalists import (  # noqa: E402
    ADAPTER_TENSOR_COUNT,
    FINALIST_A,
    FINALIST_B,
    INVENTORY_SHA256,
    INVENTORY_SIZE_BYTES,
    INVENTORY_SOURCE_REVISION,
    PRECISION,
    expected_run_provenance,
)
from unmark.stage1.protocol import ADAPTER_TRAINABLE_PARAMETERS, HIDDEN_SIZE  # noqa: E402
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULE = "unmark/evaluation/stage2_dual_finalist.py"

try:  # pragma: no cover - depends on the environment
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover - the normal ML-free path
    TORCH = False

requires_torch = pytest.mark.skipif(
    not TORCH, reason="torch is not installed locally; tensor checks run where available"
)


class StubTokenizer:
    """Character-level tokenizer over the already-stripped base chunks."""

    pad_token_id = 1
    unk_token_id = 3
    name_or_path = ENCODER_CHECKPOINT
    _commit_hash = ENCODER_REVISION

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        span = 400
        return [7 + zlib.crc32(token.encode("utf-8")) % span for token in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + list(ids) + [2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


class FakeInventoryIdentity:
    """Enough of D-S1A-008's identity for the verifier's existing contract."""

    def to_dict(self):
        return {
            "inventory_schema_version": "vn-syllables-v1",
            "source_name": "synthetic-test-inventory-identity",
            "source_author": "tests",
            "source_revision": INVENTORY_SOURCE_REVISION,
            "sha256": INVENTORY_SHA256,
            "size_bytes": INVENTORY_SIZE_BYTES,
            "license_status": "NO_EXPLICIT_LICENSE",
        }


INVENTORY = FakeInventoryIdentity()
TEXT = "Tôi đang học ở trường đại học Việt Nam hôm nay"


def source(name: str = MODULE) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str = MODULE) -> ast.Module:
    return ast.parse(source(name))


def called_names(name: str = MODULE) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


def defined_names(name: str = MODULE) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree(name))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def top_level_imports(name: str = MODULE) -> set[str]:
    parsed = tree(name)
    modules: set[str] = set()
    for node in parsed.body:
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def vi_classifier(_text: str) -> Eligibility:
    return Eligibility.VIETNAMESE_CANDIDATE


def evidence(finalist):
    return {
        "kind": "stage1_finalist_checkpoint_evidence",
        "finalist_key": finalist.key,
        "role": finalist.role,
        "source_stage": finalist.source_stage,
        "run_seed": finalist.run_seed,
        "update": finalist.update,
        "checkpoint_sha256": finalist.checkpoint_sha256,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": ADAPTER_TENSOR_COUNT,
        "adapter_trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        "adapter_dtype": PRECISION,
        "all_finite": True,
    }


# ---------------------------------------------------------------------------
# A. Frozen identities and structural hygiene
# ---------------------------------------------------------------------------
def test_exactly_two_stage2_unmark_arms():
    assert s2.STAGE2_UNMARK_ARM_NAMES == ("UNMARK-A", "UNMARK-B")
    assert {arm.value for arm in s2.Stage2UnmarkArm} == {"UNMARK-A", "UNMARK-B"}
    assert s2.finalist_for_arm("UNMARK-A") == FINALIST_A
    assert s2.finalist_for_arm("UNMARK-B") == FINALIST_B
    s2.require_dual_finalist_state()


def test_unknown_arm_and_ensemble_are_refused():
    for bad in ("UNMARK-C", "UNMARK_A", "unmark-a", "ENSEMBLE", "A+B"):
        with pytest.raises(EvaluationContractViolation):
            s2.require_stage2_unmark_arm(bad)
    assert "ENSEMBLE" not in {arm.name for arm in s2.Stage2UnmarkArm}


def test_binding_refuses_swapped_arm_evidence():
    s2.bind_verified_checkpoint("UNMARK-A", "a.pt", evidence(FINALIST_A))
    s2.bind_verified_checkpoint("UNMARK-B", "b.pt", evidence(FINALIST_B))
    with pytest.raises(EvaluationContractViolation, match="does not bind"):
        s2.bind_verified_checkpoint("UNMARK-B", "a.pt", evidence(FINALIST_A))
    with pytest.raises(EvaluationContractViolation, match="does not bind"):
        s2.bind_verified_checkpoint("UNMARK-A", "b.pt", evidence(FINALIST_B))


def test_stage2_protocol_artifact_still_says_no_selection_or_training():
    payload = json.loads((REPO / "docs/spec/stage2-dual-finalist-protocol.json").read_text())
    assert payload["schema_version"] == "stage2-dual-finalist-protocol-v1"
    assert payload["arms"]["count"]["value"] == 2
    assert payload["arms"]["ensemble"]["value"] is False
    assert payload["measurement"]["conditions"]["value"] == list(s2.STAGE2_UNMARK_CONDITIONS)
    assert payload["measurement"]["excluded_conditions"]["value"] == ["VARIANT"]
    assert payload["state"]["stage2_started"]["value"] is False
    assert payload["state"]["final_adapter_selected"]["value"] is False
    assert payload["state"]["downstream_may_select_a_vs_b"]["value"] is False
    assert payload["state"]["downstream_results_seen"]["value"] is False
    assert payload["state"]["downstream_test"]["value"] == "SEALED"
    assert payload["reporting"]["winner_rule"]["value"] is None
    assert payload["reporting"]["tie_break"]["value"] is None
    assert payload["reporting"]["summaries_may_select_arm"]["value"] is False


def test_module_import_boundary_stays_ml_free():
    modules = top_level_imports()
    assert "torch" not in modules
    assert "transformers" not in modules


def test_no_stage2_training_selection_or_test_pathway_is_added():
    assert s2.STAGE2_HEAD_TRAINING_IMPLEMENTED is False
    assert s2.STAGE2_A_B_SELECTION_IMPLEMENTED is False
    assert s2.STAGE2_MEASUREMENT_DEV_SELECTION_IMPLEMENTED is False
    assert s2.STAGE2_OFFICIAL_TEST_REACHABLE is False
    assert s2.STAGE2_TRAINING_STARTED is False
    assert "OFFICIAL_TEST" not in {role.name for role in Preg1Role}
    assert Preg1Role.OFFICIAL_VALIDATION.may_select is False
    assert not any("winner" in name.lower() for name in defined_names())
    assert not called_names() & {"select_checkpoint", "select_learning_rate", "train_head"}


def test_loader_requires_explicit_checkpoint_path_and_no_drive_literal():
    parameters = inspect.signature(s2.load_frozen_unmark_pathway).parameters
    assert parameters["checkpoint_path"].default is inspect.Parameter.empty
    assert "/content/drive" not in source().lower()
    assert "MyDrive" not in source()


def test_loader_delegates_to_authoritative_finalist_verifier():
    assert "verify_finalist_checkpoint" in called_names()


def test_stage2_loader_uses_the_pinned_encoder_and_revision():
    body = source()
    assert "AutoTokenizer.from_pretrained" in body
    assert "AutoModel.from_pretrained" in body
    assert "revision=ENCODER_REVISION" in body
    assert ENCODER_CHECKPOINT in s2.STAGE2_TOKENIZATION
    assert ENCODER_REVISION in s2.STAGE2_TOKENIZATION


# ---------------------------------------------------------------------------
# B. Corruption and base-grid invariance
# ---------------------------------------------------------------------------
def test_supported_conditions_are_exact_and_variant_refuses():
    assert s2.STAGE2_UNMARK_CONDITIONS == ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")
    for name in s2.STAGE2_UNMARK_CONDITIONS:
        assert s2.require_stage2_condition(name) is get_condition(name)
    assert s2.STAGE2_EXCLUDED_CONDITIONS == ("VARIANT",)
    with pytest.raises(EvaluationContractViolation, match="VARIANT"):
        s2.require_stage2_condition("VARIANT")
    with pytest.raises(EvaluationContractViolation):
        s2.require_stage2_condition(
            CorruptionCondition("P10", CorruptionScope.TONE, 0.10, "not frozen")
        )


def test_condition_grid_preserves_one_base_token_grid():
    grid = s2.prepare_stage2_unmark_condition_grid(
        text=TEXT,
        sample_id="sample-1",
        tokenizer=StubTokenizer(),
        corruption_seed=91,
        classifier=vi_classifier,
        corruption_purpose=CorruptionPurpose.SELF_CHECK,
        eligibility_policy=EligibilityPolicy.UNRESOLVED,
        max_length=96,
    )
    assert [item.condition for item in grid] == list(s2.STAGE2_UNMARK_CONDITIONS)
    s2.require_same_base_grid(grid)
    assert len({item.input_ids for item in grid}) == 1
    assert len({item.special_tokens_mask for item in grid}) == 1

    by_condition = {item.condition: item for item in grid}
    assert by_condition["FULL"].corrupted_text == by_condition["FULL"].canonical_text
    assert by_condition["P100"].tone_ids != by_condition["FULL"].tone_ids
    assert by_condition["P100"].letter_ids == by_condition["FULL"].letter_ids
    assert by_condition["STRIP_ALL"].letter_ids != by_condition["FULL"].letter_ids


def test_downstream_truncation_keeps_all_conditions_on_the_same_grid():
    grid = s2.prepare_stage2_unmark_condition_grid(
        text=TEXT,
        sample_id="sample-truncated",
        tokenizer=StubTokenizer(),
        corruption_seed=91,
        classifier=vi_classifier,
        corruption_purpose=CorruptionPurpose.SELF_CHECK,
        eligibility_policy=EligibilityPolicy.UNRESOLVED,
        max_length=12,
    )
    assert all(item.length == 12 for item in grid)
    assert all(item.corruption_metadata["truncated"] is True for item in grid)
    assert len({item.input_ids for item in grid}) == 1


def test_base_grid_violation_is_refused(monkeypatch):
    real = s2.project_text
    calls = {"n": 0}

    def sabotage(text, tokenizer, classifier, unk_token_id):
        calls["n"] += 1
        base, ids, projections = real(text, tokenizer, classifier, unk_token_id)
        if calls["n"] == 2:
            return base + "x", ids, projections
        return base, ids, projections

    monkeypatch.setattr(s2, "project_text", sabotage)
    with pytest.raises(EvaluationContractViolation, match=r"b\(C\(x\)\) != b\(x\)"):
        s2.prepare_stage2_unmark_input(
            text="Tôi học",
            sample_id="sample-bad-grid",
            tokenizer=StubTokenizer(),
            condition="P100",
            corruption_seed=1,
            classifier=vi_classifier,
            corruption_purpose=CorruptionPurpose.SELF_CHECK,
            eligibility_policy=EligibilityPolicy.UNRESOLVED,
        )


def test_stage2_corruption_is_keyed_by_sample_id_not_row_order():
    first = s2.apply_stage2_corruption(
        TEXT, "P50", seed=7, sample_id="alpha", purpose=CorruptionPurpose.SELF_CHECK
    )
    second = s2.apply_stage2_corruption(
        TEXT, "P50", seed=7, sample_id="alpha", purpose=CorruptionPurpose.SELF_CHECK
    )
    assert first.to_dict() == second.to_dict()

    samples = [("alpha", TEXT), ("beta", TEXT)]
    forward = {
        sample_id: s2.apply_stage2_corruption(
            text, "P50", seed=7, sample_id=sample_id, purpose=CorruptionPurpose.SELF_CHECK
        ).to_dict()
        for sample_id, text in samples
    }
    backward = {
        sample_id: s2.apply_stage2_corruption(
            text, "P50", seed=7, sample_id=sample_id, purpose=CorruptionPurpose.SELF_CHECK
        ).to_dict()
        for sample_id, text in reversed(samples)
    }
    assert forward == backward


def test_every_supported_condition_reaches_the_existing_corruption_api():
    for condition in s2.STAGE2_UNMARK_CONDITIONS:
        result = s2.apply_stage2_corruption(
            TEXT,
            condition,
            seed=17,
            sample_id="api-sample",
            purpose=CorruptionPurpose.SELF_CHECK,
        )
        assert result.condition is get_condition(condition)


def test_empty_sample_id_refuses_before_row_order_can_be_identity():
    with pytest.raises(EvaluationContractViolation, match="stable identity"):
        s2.prepare_stage2_unmark_input(
            text="Tôi học",
            sample_id="",
            tokenizer=StubTokenizer(),
            condition="FULL",
            corruption_seed=1,
            classifier=vi_classifier,
            corruption_purpose=CorruptionPurpose.SELF_CHECK,
        )


# ---------------------------------------------------------------------------
# C. Torch-gated checkpoint, freezing, representation, and collation tests
# ---------------------------------------------------------------------------
def _adapter_state(torch_module, *, dtype=None, drop=None, extra=None):
    shapes = {
        "tone_embedding.weight": (7, 768),
        "letter_embedding.weight": (5, 768),
        "fusion.weight": (768, 2304),
        "fusion.bias": (768,),
        "gate.weight": (768, 2304),
        "gate.bias": (768,),
        "layer_norm.weight": (768,),
        "layer_norm.bias": (768,),
    }
    dtype = dtype or torch_module.float32
    state = {key: torch_module.zeros(shape, dtype=dtype) for key, shape in shapes.items()}
    if drop:
        del state[drop]
    if extra:
        state[extra] = torch_module.zeros((1,), dtype=dtype)
    return state


def _payload(torch_module, finalist, **overrides):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": expected_run_provenance(finalist, inventory=INVENTORY).to_dict(),
        "adapter_state": _adapter_state(torch_module),
        "optimizer_state": {},
        "global_update": finalist.update,
        "sampler_state": {},
        "cap": 20000,
        "points": [],
        "execution": {},
    }
    payload.update(overrides)
    return payload


def _write_checkpoint(torch_module, tmp_path, payload, name="checkpoint.pt"):
    path = tmp_path / name
    torch_module.save(payload, path)
    return path


def _mock_digest(monkeypatch, digest: str):
    import unmark.stage1.finalists as finalists_module

    monkeypatch.setattr(finalists_module, "sha256_file", lambda _path: digest)


def make_encoder(hidden: int = HIDDEN_SIZE, revision: str = ENCODER_REVISION):
    import torch
    from torch import nn

    class Embeddings:
        def __init__(self, word_embeddings):
            self.word_embeddings = word_embeddings

    class RobertaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(512, hidden, padding_idx=1)
            self.embeddings = Embeddings(self.word_embeddings)
            self.name_or_path = ENCODER_CHECKPOINT
            self.config = SimpleNamespace(
                hidden_size=hidden,
                model_type="roberta",
                pad_token_id=1,
                _commit_hash=revision,
            )

        def get_input_embeddings(self):
            return self.word_embeddings

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None):
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            batch, length, dim = inputs_embeds.shape
            positions = torch.arange(
                length, dtype=torch.float32, device=inputs_embeds.device
            ).view(1, length, 1)
            offsets = torch.arange(
                batch, dtype=torch.float32, device=inputs_embeds.device
            ).view(batch, 1, 1) * 1000.0
            hidden_state = positions.expand(batch, length, dim) + offsets
            return SimpleNamespace(last_hidden_state=hidden_state)

    return RobertaModel()


def make_frozen_pathway(arm=s2.Stage2UnmarkArm.UNMARK_A):
    from unmark.modeling.adapter import OrthographyInputAdapter
    from unmark.modeling.config import AdapterConfig

    resolved = s2.require_stage2_unmark_arm(arm)
    encoder = make_encoder()
    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=HIDDEN_SIZE))
    for module in (encoder, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    binding = s2.bind_verified_checkpoint(
        resolved, "synthetic.pt", evidence(s2.finalist_for_arm(resolved))
    )
    return s2.FrozenUnmarkPathway(
        arm=resolved,
        checkpoint_binding=binding,
        encoder=encoder,
        adapter=adapter,
    )


@requires_torch
@pytest.mark.parametrize(
    ("arm", "finalist"),
    [(s2.Stage2UnmarkArm.UNMARK_A, FINALIST_A), (s2.Stage2UnmarkArm.UNMARK_B, FINALIST_B)],
)
def test_correct_finalist_checkpoint_identity_loads(tmp_path, monkeypatch, arm, finalist):
    import torch

    path = _write_checkpoint(torch, tmp_path, _payload(torch, finalist))
    _mock_digest(monkeypatch, finalist.checkpoint_sha256)
    pathway = s2.load_frozen_unmark_pathway(
        arm, path, encoder=make_encoder(), inventory=INVENTORY
    )

    assert pathway.arm is arm
    assert pathway.checkpoint_binding.finalist_key == finalist.key
    assert pathway.checkpoint_binding.source_stage == finalist.source_stage
    assert pathway.checkpoint_sha256 == finalist.checkpoint_sha256
    assert all(not parameter.requires_grad for parameter in pathway.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in pathway.adapter.parameters())
    assert not pathway.encoder.training
    assert not pathway.adapter.training


@requires_torch
def test_a_checkpoint_under_b_identity_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    path = _write_checkpoint(torch, tmp_path, _payload(torch, FINALIST_A))
    _mock_digest(monkeypatch, FINALIST_A.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="sha256 mismatch"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_B, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_b_checkpoint_under_a_identity_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    path = _write_checkpoint(torch, tmp_path, _payload(torch, FINALIST_B))
    _mock_digest(monkeypatch, FINALIST_B.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="sha256 mismatch"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_checkpoint_digest_mismatch_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    path = _write_checkpoint(torch, tmp_path, _payload(torch, FINALIST_A))
    _mock_digest(monkeypatch, "0" * 64)
    with pytest.raises(FinalistFreezeViolation, match="sha256 mismatch"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_malformed_stage1_provenance_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    payload = _payload(torch, FINALIST_A)
    payload["provenance"]["run_seed"] = 999
    path = _write_checkpoint(torch, tmp_path, payload)
    _mock_digest(monkeypatch, FINALIST_A.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="provenance"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_malformed_adapter_state_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    payload = _payload(torch, FINALIST_A, adapter_state=_adapter_state(torch, drop="gate.bias"))
    path = _write_checkpoint(torch, tmp_path, payload)
    _mock_digest(monkeypatch, FINALIST_A.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="adapter contract"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_encoder_revision_mismatch_is_refused():
    with pytest.raises(EvaluationContractViolation, match="revision"):
        s2.require_stage2_encoder_identity(make_encoder(revision="0" * 40))


@requires_torch
def test_frozen_pathway_guard_refuses_training_modes_and_trainable_parameters():
    pathway = make_frozen_pathway()
    pathway.require_frozen()

    pathway.encoder.train()
    with pytest.raises(EvaluationContractViolation, match="encoder must be in eval"):
        pathway.require_frozen()
    pathway.encoder.eval()

    pathway.adapter.train()
    with pytest.raises(EvaluationContractViolation, match="adapter must be in eval"):
        pathway.require_frozen()
    pathway.adapter.eval()

    next(pathway.adapter.parameters()).requires_grad_(True)
    with pytest.raises(EvaluationContractViolation, match="adapter parameter"):
        pathway.require_frozen()


@requires_torch
def test_future_head_parameter_enumeration_excludes_encoder_and_adapter():
    import torch
    from torch import nn

    pathway = make_frozen_pathway()
    head = nn.Linear(HIDDEN_SIZE, PRIMARY_NUM_LABELS, bias=True)
    params = s2.stage2_head_trainable_parameters(pathway, head)

    assert {id(parameter) for parameter in params} == {
        id(parameter) for parameter in head.parameters()
    }
    frozen_ids = {
        id(parameter)
        for module in (pathway.encoder, pathway.adapter)
        for parameter in module.parameters()
    }
    assert not ({id(parameter) for parameter in params} & frozen_ids)
    assert any(parameter.requires_grad for parameter in params)


@requires_torch
def test_first_token_representation_uses_exactly_position_zero():
    import torch

    hidden = torch.randn(3, 5, HIDDEN_SIZE)
    pooled = s2.stage2_first_token_representation(hidden)
    assert torch.equal(pooled, hidden[:, 0, :])
    with pytest.raises(EvaluationContractViolation, match=r"\[batch, length, hidden\]"):
        s2.stage2_first_token_representation(torch.zeros(3, HIDDEN_SIZE))
    with pytest.raises(EvaluationContractViolation, match="768-dimensional"):
        s2.stage2_first_token_representation(torch.zeros(3, 5, 8))


@requires_torch
def test_extract_representations_are_fp32_detached_and_not_masked_mean():
    import torch

    items = s2.prepare_stage2_unmark_condition_grid(
        text="Tôi học",
        sample_id="representation-sample",
        tokenizer=StubTokenizer(),
        corruption_seed=1,
        classifier=vi_classifier,
        corruption_purpose=CorruptionPurpose.SELF_CHECK,
        eligibility_policy=EligibilityPolicy.UNRESOLVED,
        max_length=16,
    )[:2]
    batch = s2.collate_stage2_unmark_batch(
        items, pad_token_id=StubTokenizer.pad_token_id, max_length=16
    )
    features = s2.extract_stage2_unmark_representations(make_frozen_pathway(), batch)
    repeated = s2.extract_stage2_unmark_representations(make_frozen_pathway(), batch)

    assert features.shape == (2, HIDDEN_SIZE)
    assert features.dtype is torch.float32
    assert not features.requires_grad
    assert features.grad_fn is None
    assert torch.equal(features, repeated)
    assert torch.equal(features[0], torch.zeros(HIDDEN_SIZE))
    assert torch.equal(features[1], torch.full((HIDDEN_SIZE,), 1000.0))

    function = next(
        node for node in ast.walk(tree())
        if isinstance(node, ast.FunctionDef)
        and node.name == "extract_stage2_unmark_representations"
    )
    assert "masked_mean" not in ast.unparse(function)


@requires_torch
def test_collation_pads_to_max_length_and_marks_padding_na():
    inputs = s2.prepare_stage2_unmark_condition_grid(
        text="Tôi học",
        sample_id="collate-sample",
        tokenizer=StubTokenizer(),
        corruption_seed=1,
        classifier=vi_classifier,
        corruption_purpose=CorruptionPurpose.SELF_CHECK,
        eligibility_policy=EligibilityPolicy.UNRESOLVED,
        max_length=16,
    )[:2]
    batch = s2.collate_stage2_unmark_batch(
        inputs, pad_token_id=StubTokenizer.pad_token_id, max_length=16
    )

    assert tuple(batch["input_ids"].shape) == (2, 16)
    assert tuple(batch["tone_ids"].shape) == (2, 16)
    assert tuple(batch["letter_ids"].shape)[:2] == (2, 16)
    pad_index = inputs[0].length
    assert batch["attention_mask"][0, pad_index].item() == 0
    assert batch["special_tokens_mask"][0, pad_index].item() == 1
    assert batch["tone_ids"][0, pad_index].item() == TONE_NA_SENTINEL
    assert batch["tone_mask"][0, pad_index].item() is False
    assert batch["letter_ids"][0, pad_index, 0].item() == LETTER_NA_SENTINEL
    assert batch["letter_mask"][0, pad_index, 0].item() is False


@requires_torch
def test_non_fp32_adapter_checkpoint_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    payload = _payload(torch, FINALIST_A, adapter_state=_adapter_state(torch, dtype=torch.float16))
    path = _write_checkpoint(torch, tmp_path, payload)
    _mock_digest(monkeypatch, FINALIST_A.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="not float32"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )


@requires_torch
def test_non_finite_adapter_checkpoint_is_refused(tmp_path, monkeypatch):
    import torch
    from unmark.stage1.finalists import FinalistFreezeViolation

    state = _adapter_state(torch)
    state["layer_norm.weight"][0] = float("nan")
    path = _write_checkpoint(torch, tmp_path, _payload(torch, FINALIST_A, adapter_state=state))
    _mock_digest(monkeypatch, FINALIST_A.checkpoint_sha256)
    with pytest.raises(FinalistFreezeViolation, match="NaN or Inf"):
        s2.load_frozen_unmark_pathway(
            s2.Stage2UnmarkArm.UNMARK_A, path, encoder=make_encoder(), inventory=INVENTORY
        )
