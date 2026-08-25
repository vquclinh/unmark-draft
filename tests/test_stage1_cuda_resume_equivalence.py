"""CUDA interrupted-vs-uninterrupted resume equivalence (Audit 030 §AF.4 / §AG).

§AF.4 established this once with an out-of-tree fixture and recorded, honestly,
that **the repository had no persistent regression test for it**. This file is
that test.

**TEST-ONLY throughout.** A tiny synthetic model, never the real backbone and
never the prepared corpus. It takes synthetic `optimizer.step`s because
populated optimizer state is precisely what must be compared — that is the whole
point of the contract — and it writes no scientific artifact.

CUDA-gated: skips cleanly without a GPU, and is authoritative only on one.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="needs torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA host")

from unmark.stage1.initialisation import (  # noqa: E402
    fresh_adapter,
    trainable_state,
    trainable_state_hash,
)
from unmark.stage1.data import module_device  # noqa: E402
from unmark.stage1.optim import build_optimizer  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    TrainerContractViolation,
    checkpoint_payload,
    load_training_checkpoint,
    require_optimizer_parameter_identity,
    require_optimizer_state_device,
    save_training_checkpoint,
    verify_checkpoint,
)

TINY = 8
INIT_SEED = 3203
TOTAL, INTERRUPT_AT = 16, 8


def provenance():
    from unmark.stage1.preflight import InventoryIdentity
    from unmark.stage1.trainer import RunProvenance

    return RunProvenance(
        run_seed=36930, init_seed=51800, corruption_seed=35422, learning_rate=3e-4,
        r=1.0, corpus_manifest_digest="d" * 64, repository_head="a" * 40,
        inventory=InventoryIdentity(
            inventory_schema_version="vn-syllables-v1",
            source_name="all-vietnamese-syllables.txt", source_author="hieuthi",
            source_revision="135a4d9716e49a981624474156d6f247b9b46f6a",
            sha256="78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",
            size_bytes=116_290, license_status="NO_EXPLICIT_LICENSE",
        ),
    )


def build(requested_device):
    """Deterministic CPU init, then placement — the production order.

    Returns `(adapter, optimizer, actual_device)`.

    **`requested_device` and `actual_device` are not the same thing.**
    `torch.device("cuda")` is a *logical alias* with no index; once a module is
    placed, its parameters live on a *concrete* device such as `cuda:0`, and
    `torch.device("cuda") != torch.device("cuda:0")`. Production never confuses
    the two: `train_run` passes `module_device(objective)` —
    `next(module.parameters()).device` — so its postcondition is always the
    concrete placed device. This helper derives the same thing the same way, so
    the test asserts what production asserts.
    """
    adapter = fresh_adapter(TINY, INIT_SEED).to(requested_device)
    optimizer = build_optimizer(list(adapter.named_parameters()), 3e-4)
    require_optimizer_parameter_identity(optimizer, adapter)
    return adapter, optimizer, module_device(adapter)


def synthetic_step(adapter, optimizer, update: int, device) -> None:
    """TEST-ONLY. A fixed, update-dependent loss: no data, no corpus, no science."""
    optimizer.zero_grad(set_to_none=True)
    driver = torch.full((TINY,), float(update + 1), device=device)
    loss = sum((p * driver.sum()).sum() for p in adapter.parameters())
    loss.backward()
    optimizer.step()


def optimizer_state_snapshot(optimizer):
    return {
        index: {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in state.items()}
        for index, state in optimizer.state_dict()["state"].items()
    }


def equal_state(a, b) -> bool:
    if set(a) != set(b):
        return False
    for index in a:
        if set(a[index]) != set(b[index]):
            return False
        for key in a[index]:
            x, y = a[index][key], b[index][key]
            if torch.is_tensor(x):
                if not torch.equal(x, y):
                    return False
            elif x != y:
                return False
    return True


@needs_cuda
def test_cuda_interrupted_then_resumed_equals_uninterrupted(tmp_path):
    requested_device = torch.device("cuda")   # a logical alias, NOT a postcondition

    # --- uninterrupted reference -------------------------------------------
    adapter, optimizer, device = build(requested_device)
    assert device.type == "cuda" and device.index is not None, (
        f"expected a concrete placed device, got {device}"
    )
    sampler_state = {"cursor": 0, "visit": 0}
    for update in range(TOTAL):
        synthetic_step(adapter, optimizer, update, device)
        sampler_state = {"cursor": (update + 1) * 3 % 7, "visit": (update + 1) // 7}
    uninterrupted_hash = trainable_state_hash(trainable_state(adapter))
    uninterrupted_optimizer = optimizer_state_snapshot(optimizer)
    uninterrupted_sampler = dict(sampler_state)

    # --- interrupted at update 8, checkpointed ------------------------------
    adapter_a, optimizer_a, _ = build(requested_device)
    sampler_state = {"cursor": 0, "visit": 0}
    for update in range(INTERRUPT_AT):
        synthetic_step(adapter_a, optimizer_a, update, device)
        sampler_state = {"cursor": (update + 1) * 3 % 7, "visit": (update + 1) // 7}
    points = [{"update": 0, "score": 1.0, "d_clean": 0.5,
               "distances": {"FULL": 1.0, "P50": 1.0, "P100": 1.0, "STRIP_ALL": 1.0}}]
    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=provenance(), adapter_state=adapter_a.state_dict(),
        optimizer_state=optimizer_a.state_dict(), global_update=INTERRUPT_AT,
        sampler_state=dict(sampler_state), cap=TOTAL, budget_limited=False,
        points=points, execution=None,
    ))
    del adapter_a, optimizer_a

    # --- fresh reconstruction, then resume ----------------------------------
    payload = load_training_checkpoint(tmp_path)
    verify_checkpoint(payload, provenance())
    adapter_b, optimizer_b, resumed_device = build(requested_device)  # CPU init -> placed
    adapter_b.load_state_dict(payload["adapter_state"], strict=True)
    optimizer_b.load_state_dict(payload["optimizer_state"])
    require_optimizer_parameter_identity(optimizer_b, adapter_b)
    # The authoritative postcondition, against the device THIS adapter is on --
    # derived from the freshly reconstructed module, exactly as production does.
    assert resumed_device.type == "cuda" and resumed_device.index is not None
    assert resumed_device == device, "single-GPU test: both placements agree"
    require_optimizer_state_device(optimizer_b, resumed_device)

    resumed_sampler = dict(payload["sampler_state"])
    global_update = int(payload["global_update"])
    for update in range(global_update, TOTAL):
        synthetic_step(adapter_b, optimizer_b, update, device)
        resumed_sampler = {"cursor": (update + 1) * 3 % 7, "visit": (update + 1) // 7}
        global_update += 1

    # --- exact equality ------------------------------------------------------
    assert trainable_state_hash(trainable_state(adapter_b)) == uninterrupted_hash
    assert equal_state(optimizer_state_snapshot(optimizer_b), uninterrupted_optimizer)
    assert resumed_sampler == uninterrupted_sampler
    assert global_update == TOTAL
    assert [dict(p) for p in payload["points"]] == points


@needs_cuda
def test_populated_adam_state_lands_on_cuda_with_scalar_step_left_on_cpu(tmp_path):
    """Real AdamW semantics (Audit 030 §AF.3), asserted rather than assumed.

    `exp_avg` / `exp_avg_sq` must be on concrete CUDA; the zero-dimensional
    scalar `step` legitimately stays on CPU and **must not** be forced across.
    """
    requested_device = torch.device("cuda")   # logical alias, not a postcondition
    adapter, optimizer, device = build(requested_device)
    for update in range(3):
        synthetic_step(adapter, optimizer, update, device)

    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=provenance(), adapter_state=adapter.state_dict(),
        optimizer_state=optimizer.state_dict(), global_update=3,
        sampler_state={"cursor": 1, "visit": 0}, cap=TOTAL, budget_limited=False,
        points=[], execution=None,
    ))
    payload = load_training_checkpoint(tmp_path)
    assert all(not torch.is_tensor(v) or v.device.type == "cpu"
               for s in payload["optimizer_state"]["state"].values()
               for v in s.values()), "map_location='cpu' must bring the payload back on CPU"

    adapter_b, optimizer_b, resumed_device = build(requested_device)
    adapter_b.load_state_dict(payload["adapter_state"], strict=True)
    optimizer_b.load_state_dict(payload["optimizer_state"])

    # Concrete placed device, derived from the reconstructed adapter itself.
    assert resumed_device.type == "cuda" and resumed_device.index is not None
    require_optimizer_state_device(optimizer_b, resumed_device)

    # And the whole point of the strict postcondition: the LOGICAL alias is not
    # the concrete device, and production is right to reject it.
    with pytest.raises(TrainerContractViolation, match="not the training device"):
        require_optimizer_state_device(optimizer_b, torch.device("cuda"))

    seen_moment = seen_step = False
    for state in optimizer_b.state_dict()["state"].values():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if key in ("exp_avg", "exp_avg_sq"):
                assert value.device.type == "cuda", f"{key} on {value.device}"
                seen_moment = True
            elif key == "step" and value.dim() == 0:
                seen_step = True                            # CPU here is correct
    assert seen_moment, "expected populated Adam moments"
    assert seen_step, "expected a scalar Adam step"


@needs_cuda
def test_the_checkpoint_carries_no_frozen_encoder_key(tmp_path):
    adapter, optimizer, device = build(torch.device("cuda"))
    synthetic_step(adapter, optimizer, 0, device)
    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=provenance(), adapter_state=adapter.state_dict(),
        optimizer_state=optimizer.state_dict(), global_update=1,
        sampler_state={}, cap=TOTAL, budget_limited=False, points=[], execution=None,
    ))
    keys = set(load_training_checkpoint(tmp_path)["adapter_state"])
    assert keys and not any(k.startswith("encoder.") for k in keys), keys
