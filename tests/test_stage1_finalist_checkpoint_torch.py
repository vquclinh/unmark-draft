"""Tensor-level verification of the frozen Stage-1 finalist adapters (Audit 048).

Needs **real torch**: these tests build checkpoint payloads and inspect adapter
tensors. Kept apart from `test_stage1_finalist_freeze.py` so the module-level
`importorskip` below cannot skip that file's torch-free contract coverage.

Nothing here trains, and no downstream data is read.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.finalists import (  # noqa: E402
    ADAPTER_TENSOR_COUNT,
    CORPUS_MANIFEST_DIGEST,
    FINALIST_A,
    INVENTORY_SHA256,
    SHA256_PENDING,
    FinalistFreezeViolation,
    FinalistIdentity,
    expected_run_provenance,
    resolve_inventory,
    verify_finalist_checkpoint,
)
from unmark.stage1.protocol import (  # noqa: E402
    ADAPTER_TRAINABLE_PARAMETERS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    PRECISION,
    STAGE1_PROTOCOL_VERSION,
)
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402

def _adapter_state(torch, *, dtype=None, drop=None, extra=None):
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
    dtype = dtype or torch.float32
    state = {k: torch.zeros(v, dtype=dtype) for k, v in shapes.items()}
    if drop:
        del state[drop]
    if extra:
        state[extra] = torch.zeros((1,), dtype=dtype)
    return state


INVENTORY = resolve_inventory()


def _provenance(finalist):
    """The full RunProvenance the authoritative gate compares, derived not retyped."""
    return expected_run_provenance(finalist, inventory=INVENTORY).to_dict()


def _payload(torch, finalist, **overrides):
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": _provenance(finalist),
        "adapter_state": _adapter_state(torch),
        "optimizer_state": {},
        "global_update": finalist.update,
        "sampler_state": {},
        "cap": 20000,
        "points": [],
        "execution": {},
    }
    body.update(overrides)
    return body


def _write(torch, tmp_path, payload, name="checkpoint.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _unbound(finalist):
    """A finalist whose digest is not yet bound, so file content is what is tested."""
    return FinalistIdentity(**{**finalist.__dict__, "checkpoint_sha256": SHA256_PENDING})





torch = pytest.importorskip(
    "torch", reason="adapter tensor inspection needs real torch; run where checkpoints live"
)


def test_a_correct_checkpoint_verifies(tmp_path):
    finalist = _unbound(FINALIST_A)
    path = _write(torch, tmp_path, _payload(torch, finalist))
    evidence = verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)
    assert evidence["adapter_trainable_parameters"] == ADAPTER_TRAINABLE_PARAMETERS
    assert evidence["adapter_tensor_count"] == ADAPTER_TENSOR_COUNT
    assert evidence["adapter_dtype"] == PRECISION
    assert evidence["all_finite"] is True
    assert len(evidence["checkpoint_sha256"]) == 64
    assert evidence["digest_was_bound_before_verification"] is False


def test_a_non_mapping_payload_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(FinalistFreezeViolation, match="did not deserialise to a mapping"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_wrong_checkpoint_schema_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    path = _write(torch, tmp_path, _payload(torch, finalist, schema_version="stage1-checkpoint-v1"))
    with pytest.raises(FinalistFreezeViolation, match="schema"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_wrong_global_update_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    path = _write(torch, tmp_path, _payload(torch, finalist, global_update=17000))
    with pytest.raises(FinalistFreezeViolation, match="global_update"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Every scientific-identity field RunProvenance.require_match compares.
        ("run_seed", 7309),
        ("init_seed", 45833),                       # the OTHER final-main seed's init
        ("corruption_seed", 12345),
        ("learning_rate", 3e-4),
        ("r", 2.0),
        ("corpus_manifest_digest", "0" * 64),       # trained on different data
        ("backbone_checkpoint", "vinai/phobert-large"),
        ("backbone_revision", "0" * 40),
        ("protocol_version", "stage1-protocol-v2"),
        ("precision", "bf16"),
        ("repository_head", "f" * 40),
        ("inventory", None),                        # D-S1A-008
    ],
)
def test_a_wrong_provenance_field_is_refused(tmp_path, field, value):
    """These are refused by the authoritative `require_match`, not by a local copy."""
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["provenance"][field] = value
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match=f"mismatch on '{field}'"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize("field", ["lambda_align", "lambda_clean"])
def test_an_objective_weight_inconsistent_with_r_is_refused(tmp_path, field):
    """The derived weights must agree with the r they claim to come from."""
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["provenance"][field] = 0.5
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="internally inconsistent"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize("field", ["lambda_align", "lambda_clean"])
def test_a_missing_derived_weight_is_refused(tmp_path, field):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    del payload["provenance"][field]
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="missing the derived key"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "0" * 64),
        ("source_revision", "0" * 40),
        ("size_bytes", 1),
    ],
)
def test_a_wrong_inventory_pin_in_the_checkpoint_is_refused(tmp_path, field, value):
    """Named individually so the operator is told which inventory field is wrong."""
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    inventory = dict(payload["provenance"]["inventory"])
    inventory[field] = value
    payload["provenance"]["inventory"] = inventory
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize("missing", ["cap", "sampler_state", "optimizer_state", "points"])
def test_a_checkpoint_missing_a_required_key_is_refused(tmp_path, missing):
    """`verify_checkpoint` enforces the whole REQUIRED_CHECKPOINT_KEYS set."""
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    del payload[missing]
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="missing"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_the_evidence_record_carries_the_execution_fingerprint(tmp_path):
    """Operational, not identity: reported for later comparison, not enforced."""
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["execution"] = {"backend": "cuda", "gpu_name": "NVIDIA A100-SXM4-40GB"}
    path = _write(torch, tmp_path, payload)
    evidence = verify_finalist_checkpoint(
        path, finalist, require_bound_digest=False, inventory=INVENTORY
    )
    assert evidence["execution_fingerprint"]["gpu_name"] == "NVIDIA A100-SXM4-40GB"
    assert evidence["corpus_manifest_digest"] == CORPUS_MANIFEST_DIGEST
    assert evidence["inventory"]["sha256"] == INVENTORY_SHA256
    assert evidence["init_seed"] == 51800
    assert evidence["lambda_align"] == 1.0 and evidence["lambda_clean"] == 1.0
    # A differing fingerprint is reported, never a refusal.
    payload["execution"] = {"backend": "cuda", "gpu_name": "Tesla T4"}
    other = _write(torch, tmp_path, payload, name="t4.pt")
    assert verify_finalist_checkpoint(
        other, finalist, require_bound_digest=False, inventory=INVENTORY
    )["execution_fingerprint"]["gpu_name"] == "Tesla T4"


def test_a_missing_provenance_block_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    del payload["provenance"]
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="no provenance"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_missing_adapter_state_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    del payload["adapter_state"]
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="no adapter_state"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_dropped_adapter_tensor_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["adapter_state"] = _adapter_state(torch, drop="gate.bias")
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="adapter contract"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_an_extra_adapter_tensor_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["adapter_state"] = _adapter_state(torch, extra="head.weight")
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="adapter contract"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_wrong_parameter_count_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    state = _adapter_state(torch)
    state["fusion.bias"] = torch.zeros((767,), dtype=torch.float32)
    payload["adapter_state"] = state
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="parameters, not the locked"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_non_fp32_adapter_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    payload["adapter_state"] = _adapter_state(torch, dtype=torch.float16)
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="not float32"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_a_non_tensor_adapter_entry_is_refused(tmp_path):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    state = _adapter_state(torch)
    state["gate.bias"] = [0.0] * 768
    payload["adapter_state"] = state
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="not a tensor"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_adapter_is_refused(tmp_path, bad):
    finalist = _unbound(FINALIST_A)
    payload = _payload(torch, finalist)
    state = _adapter_state(torch)
    state["layer_norm.weight"][0] = float(bad)
    payload["adapter_state"] = state
    path = _write(torch, tmp_path, payload)
    with pytest.raises(FinalistFreezeViolation, match="NaN or Inf"):
        verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)


def test_the_verifier_never_modifies_the_checkpoint(tmp_path):
    finalist = _unbound(FINALIST_A)
    path = _write(torch, tmp_path, _payload(torch, finalist))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    verify_finalist_checkpoint(path, finalist, require_bound_digest=False, inventory=INVENTORY)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_a_bound_digest_must_match_the_real_file(tmp_path):
    """End to end: bind the true digest, then a one-byte change is refused."""
    from unmark.stage1.checkpoint import sha256_file

    finalist = _unbound(FINALIST_A)
    path = _write(torch, tmp_path, _payload(torch, finalist))
    bound = FinalistIdentity(**{**finalist.__dict__, "checkpoint_sha256": sha256_file(path)})
    evidence = verify_finalist_checkpoint(path, bound, inventory=INVENTORY)
    assert evidence["digest_was_bound_before_verification"] is True

    other = _write(torch, tmp_path, _payload(torch, finalist, cap=40000), name="other.pt")
    with pytest.raises(FinalistFreezeViolation, match="sha256 mismatch"):
        verify_finalist_checkpoint(other, bound, inventory=INVENTORY)
