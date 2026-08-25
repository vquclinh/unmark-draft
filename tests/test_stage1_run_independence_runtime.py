"""Nominal-run independence and initialisation — runtime half (§AE).

Executes the contracts its torch-free companion asserts structurally. Tiny
fixtures throughout: **no PhoBERT is downloaded**. The CUDA-gated tests skip off
a GPU and are authoritative only in the fresh CUDA runtime.

No synthetic optimizer step is taken anywhere in this file. (An earlier draft of
this docstring referred to a CUDA resume-equivalence test that was never added
here; that coverage now lives in `test_stage1_cuda_resume_equivalence.py` — see
Audit 030 §AF.4 and §AG.)
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the runtime half needs torch")

from unmark.stage1.initialisation import (  # noqa: E402
    expected_fresh_init_hash,
    fresh_adapter,
    module_state_hash,
    trainable_state,
    trainable_state_hash,
)
from unmark.stage1.protocol import (  # noqa: E402
    ADAPTER_TRAINABLE_PARAMETERS,
    HIDDEN_SIZE,
    adapter_init_seed,
)
from unmark.stage1.trainer import (  # noqa: E402
    TrainerContractViolation,
    require_optimizer_parameter_identity,
    require_optimizer_state_device,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA host")
TINY = 8


# ---------------------------------------------------------------------------
# 1. Deterministic, CPU-only, RNG-isolated initialisation (D-S1B-016)
# ---------------------------------------------------------------------------
def test_the_same_init_seed_reproduces_byte_identical_state():
    a = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    b = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    assert a == b


def test_a_different_init_seed_gives_a_different_state():
    a = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    b = trainable_state_hash(trainable_state(fresh_adapter(TINY, 51800)))
    assert a != b


def test_the_adapter_is_initialised_on_cpu():
    for parameter in fresh_adapter(TINY, 3203).parameters():
        assert parameter.device.type == "cpu"


def test_ambient_rng_consumption_cannot_perturb_initialisation():
    """Prior RNG use in the process must not change a run's starting point."""
    torch.manual_seed(1234)
    first = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    torch.manual_seed(999)
    _ = torch.randn(50)            # unrelated consumption
    second = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    assert first == second


def test_initialisation_restores_the_ambient_cpu_rng_byte_for_byte():
    """`fork_rng`, not a bare seed: the CPU generator state must come back exactly."""
    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    fresh_adapter(TINY, 3203)
    assert torch.equal(before, torch.get_rng_state()), "CPU RNG state was not restored"


def test_initialisation_does_not_change_the_ambient_stream():
    torch.manual_seed(4242)
    expected = torch.randn(4)

    torch.manual_seed(4242)
    fresh_adapter(TINY, 3203)
    observed = torch.randn(4)
    assert torch.equal(expected, observed), "adapter construction perturbed ambient RNG"


@needs_cuda
def test_initialisation_leaves_every_cuda_generator_byte_identical():
    """The §AE.4 correction, proven.

    `torch.manual_seed` seeds **all devices**, and `fork_rng(devices=[])` snapshots
    only the CPU generator — so the original pairing perturbed CUDA RNG and never
    restored it. `torch.default_generator.manual_seed` cannot: this asserts it on
    the real states.
    """
    torch.cuda.init()
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    fresh_adapter(TINY, 3203)
    after = torch.cuda.get_rng_state_all()

    assert len(before) == len(after)
    for index, (was, now) in enumerate(zip(before, after)):
        assert torch.equal(was, now), f"CUDA generator {index} was perturbed"


@needs_cuda
def test_initialisation_does_not_depend_on_cuda_rng():
    """Hardware independence: seeding CUDA differently changes nothing."""
    torch.cuda.manual_seed_all(1)
    first = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    torch.cuda.manual_seed_all(2)
    second = trainable_state_hash(trainable_state(fresh_adapter(TINY, 3203)))
    assert first == second


def test_initialisation_does_not_initialise_cuda_as_a_side_effect():
    """A pure CPU construction must not bring up a CUDA context, nor queue a
    deferred `manual_seed_all` that would fire when one is created later."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        pytest.skip("CUDA was already initialised by an earlier test")
    fresh_adapter(TINY, 3203)
    assert not torch.cuda.is_initialized()


def test_the_hash_survives_a_device_move():
    """An expected hash computed on CPU must match a placed model's state."""
    adapter = fresh_adapter(TINY, 3203)
    before = trainable_state_hash(trainable_state(adapter))
    adapter.to(torch.device("cpu"))
    assert trainable_state_hash(trainable_state(adapter)) == before


def test_expected_fresh_init_hash_matches_a_real_construction():
    assert expected_fresh_init_hash(TINY, 21230) == trainable_state_hash(
        trainable_state(fresh_adapter(TINY, adapter_init_seed(21230)))
    )


def test_the_locked_adapter_still_has_the_locked_parameter_count():
    adapter = fresh_adapter(HIDDEN_SIZE, 3203)
    total = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert total == ADAPTER_TRAINABLE_PARAMETERS


# ---------------------------------------------------------------------------
# 2. Storage independence — hash equality proves nothing (D-S1B-017)
# ---------------------------------------------------------------------------
def test_two_adapters_from_one_seed_share_no_object_or_storage():
    """The eight selection candidates DO share values; they must not share memory."""
    a = fresh_adapter(TINY, 3203)
    b = fresh_adapter(TINY, 3203)

    assert trainable_state_hash(trainable_state(a)) == trainable_state_hash(trainable_state(b))
    assert a is not b

    pairs = list(zip(a.named_parameters(), b.named_parameters()))
    assert pairs
    for (name_a, pa), (name_b, pb) in pairs:
        assert name_a == name_b
        assert pa is not pb, name_a
        assert pa.data_ptr() != pb.data_ptr(), f"{name_a} shares storage"


def test_mutating_one_adapter_leaves_the_other_byte_identical():
    """Positive proof of independence. TEST-ONLY mutation; not training."""
    a = fresh_adapter(TINY, 3203)
    b = fresh_adapter(TINY, 3203)
    before_a = trainable_state_hash(trainable_state(a))
    before_b = trainable_state_hash(trainable_state(b))
    assert before_a == before_b

    with torch.no_grad():
        next(iter(a.parameters())).add_(1.0)

    assert trainable_state_hash(trainable_state(a)) != before_a
    assert trainable_state_hash(trainable_state(b)) == before_b


def test_a_mutated_candidate_cannot_change_the_next_candidates_start():
    """The §AD leakage, inverted: candidate N+1 starts from its expected hash."""
    expected = expected_fresh_init_hash(TINY, 21230)
    first = fresh_adapter(TINY, adapter_init_seed(21230))
    with torch.no_grad():
        for parameter in first.parameters():
            parameter.add_(3.0)          # stands in for "candidate 1 trained"
    assert trainable_state_hash(trainable_state(first)) != expected

    for _ in range(2):                    # candidates 2 and 3
        nxt = fresh_adapter(TINY, adapter_init_seed(21230))
        assert trainable_state_hash(trainable_state(nxt)) == expected
        assert nxt is not first


# ---------------------------------------------------------------------------
# 3. Optimizer object identity and state device
# ---------------------------------------------------------------------------
def optimizer_for(adapter):
    from unmark.stage1.optim import build_optimizer

    return build_optimizer(list(adapter.named_parameters()), 3e-4)


def test_a_freshly_built_optimizer_satisfies_the_identity_contract():
    adapter = fresh_adapter(TINY, 3203)
    require_optimizer_parameter_identity(optimizer_for(adapter), adapter)


def test_an_optimizer_bound_to_a_previous_adapter_is_rejected():
    """The classic resume failure: forward uses the new adapter, optimizer the old."""
    stale = fresh_adapter(TINY, 3203)
    current = fresh_adapter(TINY, 3203)          # identical VALUES, different objects
    with pytest.raises(TrainerContractViolation, match="not parameters of the current"):
        require_optimizer_parameter_identity(optimizer_for(stale), current)


def test_a_missing_parameter_is_rejected():
    adapter = fresh_adapter(TINY, 3203)
    partial = list(adapter.named_parameters())[:-1]
    from unmark.stage1.optim import build_optimizer

    with pytest.raises(TrainerContractViolation, match="absent from the optimizer"):
        require_optimizer_parameter_identity(build_optimizer(partial, 3e-4), adapter)


def test_optimizer_state_device_is_asserted_recursively():
    adapter = fresh_adapter(TINY, 3203)
    optimizer = optimizer_for(adapter)
    loss = sum(p.sum() for p in adapter.parameters())
    loss.backward()
    optimizer.step()                              # TEST-ONLY: tiny fixture, no science
    require_optimizer_state_device(optimizer, torch.device("cpu"))

    for state in optimizer.state.values():
        if "exp_avg" in state:
            state["exp_avg"] = state["exp_avg"].to("meta")
            break
    with pytest.raises(TrainerContractViolation, match="optimizer state"):
        require_optimizer_state_device(optimizer, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 4. Checkpoint: adapter-only, strict, and the H0/Hc continuation contract
# ---------------------------------------------------------------------------
def test_the_adapter_state_contains_no_frozen_encoder_key():
    adapter = fresh_adapter(TINY, 3203)
    keys = set(trainable_state(adapter))
    assert keys, "expected adapter state"
    assert not any(k.startswith("encoder.") for k in keys), keys
    assert keys == {name for name, _ in adapter.named_parameters()}, (
        "adapter.state_dict() must be exactly the trainable parameter set"
    )


def test_a_strict_restore_rejects_a_missing_key():
    adapter = fresh_adapter(TINY, 3203)
    state = dict(trainable_state(adapter))
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError, match="Missing key"):
        adapter.load_state_dict(state, strict=True)


def test_a_strict_restore_rejects_an_unexpected_key():
    adapter = fresh_adapter(TINY, 3203)
    state = dict(trainable_state(adapter))
    state["invented.weight"] = torch.zeros(2)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        adapter.load_state_dict(state, strict=True)


def test_a_strict_restore_rejects_a_wrong_shape():
    adapter = fresh_adapter(TINY, 3203)
    state = dict(trainable_state(adapter))
    name = next(iter(state))
    state[name] = torch.zeros(3, 3, 3)
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        adapter.load_state_dict(state, strict=True)


def test_continuation_restores_hc_and_not_h0():
    """H0 vs Hc (§AD.6(D)). No scientific optimizer step is needed."""
    adapter = fresh_adapter(TINY, 3203)
    h0 = trainable_state_hash(trainable_state(adapter))

    with torch.no_grad():                          # TEST-ONLY stand-in for training
        for parameter in adapter.parameters():
            parameter.add_(0.5)
    hc = trainable_state_hash(trainable_state(adapter))
    checkpoint = {k: v.clone() for k, v in trainable_state(adapter).items()}
    assert hc != h0

    # a continuation deterministically rebuilds the adapter, then restores
    rebuilt = fresh_adapter(TINY, 3203)
    assert trainable_state_hash(trainable_state(rebuilt)) == h0
    rebuilt.load_state_dict(checkpoint, strict=True)

    assert trainable_state_hash(trainable_state(rebuilt)) == hc
    assert trainable_state_hash(trainable_state(rebuilt)) != h0
    require_optimizer_parameter_identity(optimizer_for(rebuilt), rebuilt)


# ---------------------------------------------------------------------------
# 5. The shared frozen encoder stays immutable
# ---------------------------------------------------------------------------
class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(TINY, TINY)
        self.register_buffer("running", torch.zeros(TINY))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()


def test_the_full_state_dict_hash_covers_buffers_not_only_parameters():
    """Parameters alone would miss a mutated buffer (§AD.6(A))."""
    backbone = TinyBackbone()
    before = module_state_hash(backbone)
    with torch.no_grad():
        backbone.running.add_(1.0)          # a BUFFER, not a parameter
    assert module_state_hash(backbone) != before


def test_the_backbone_guard_accepts_an_untouched_encoder_and_rejects_a_changed_one():
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.execute import require_frozen_backbone_unchanged

    backbone = TinyBackbone()
    reference = module_state_hash(backbone)
    require_frozen_backbone_unchanged(backbone, reference, "candidate-1")

    with torch.no_grad():
        backbone.linear.weight.add_(1.0)
    with pytest.raises(Stage1ContractViolation, match="CHANGED"):
        require_frozen_backbone_unchanged(backbone, reference, "candidate-2")


def test_the_backbone_guard_rejects_a_thawed_encoder():
    from unmark.stage1.contracts import Stage1ContractViolation
    from unmark.stage1.execute import require_frozen_backbone_unchanged

    backbone = TinyBackbone()
    reference = module_state_hash(backbone)
    backbone.linear.weight.requires_grad_(True)
    with pytest.raises(Stage1ContractViolation, match="trainable"):
        require_frozen_backbone_unchanged(backbone, reference, "candidate-1")


# ---------------------------------------------------------------------------
# 6. The runtime eval guard at the encoder forward boundaries
# ---------------------------------------------------------------------------
def test_the_encoder_eval_guard_raises_when_the_encoder_is_in_train_mode():
    from unmark.modeling.adapter import UnmarkEncoder

    guard = UnmarkEncoder.require_frozen_encoder_eval
    holder = type("Holder", (), {"encoder": TinyBackbone()})()
    guard(holder)                                  # eval -> fine

    holder.encoder.train()
    with pytest.raises(RuntimeError, match="TRAIN mode"):
        guard(holder)


# ---------------------------------------------------------------------------
# 7. Device / numerical policy
# ---------------------------------------------------------------------------
def test_scientific_training_fails_closed_without_cuda(monkeypatch):
    from unmark.stage1.device import DeviceContractViolation, resolve_scientific_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceContractViolation, match="requires CUDA"):
        resolve_scientific_device()


def test_a_perturbed_precision_policy_fails_closed(monkeypatch):
    """A global change elsewhere must not reach a run claiming fp32."""
    from unmark.stage1.device import DeviceContractViolation, verify_numerical_policy

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False, raising=False)
    with pytest.raises(DeviceContractViolation, match="TF32|matmul precision"):
        verify_numerical_policy()


def test_a_missing_cublas_workspace_config_fails_closed(monkeypatch):
    from unmark.stage1.device import DeviceContractViolation, verify_numerical_policy

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False, raising=False)
    with pytest.raises(DeviceContractViolation, match="CUBLAS_WORKSPACE_CONFIG"):
        verify_numerical_policy()


def test_a_late_cublas_configuration_fails_closed(monkeypatch):
    """Setting it after CUDA init would be silently ineffective."""
    from unmark.stage1.device import DeviceContractViolation, require_deterministic_cublas_workspace

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(DeviceContractViolation, match="already initialised"):
        require_deterministic_cublas_workspace()


def test_the_fingerprint_blocks_a_numerically_different_continuation():
    from unmark.stage1.device import DeviceContractViolation, ExecutionFingerprint

    def make(**overrides):
        base = dict(
            backend="cuda", device="cuda", gpu_name="A", compute_capability="9.0",
            torch_version="2.11.0", cuda_version="12.8", cudnn_version=91002,
            deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
            cublas_workspace_config=":4096:8", float32_matmul_precision="highest",
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
        )
        base.update(overrides)
        return ExecutionFingerprint(**base)

    mine = make()
    mine.require_compatible(make().to_dict())                    # identical -> fine
    # The LOGICAL index is renumbered freely by CUDA_VISIBLE_DEVICES: not blocking.
    mine.require_compatible(make(device="cuda:1").to_dict())

    for field, value in (("backend", "cpu"), ("compute_capability", "8.0"),
                         ("torch_version", "2.10.0"), ("float32_matmul_precision", "high"),
                         ("deterministic_algorithms", False),
                         ("cublas_workspace_config", None),
                         ("cuda_version", "12.4"), ("cudnn_version", 90000),
                         ("cudnn_deterministic", False), ("cudnn_benchmark", True),
                         ("cuda_matmul_allow_tf32", True), ("cudnn_allow_tf32", True)):
        with pytest.raises(DeviceContractViolation, match=field):
            mine.require_compatible(make(**{field: value}).to_dict())


def test_a_different_gpu_model_blocks_a_continuation():
    """Conservative until proven otherwise: nothing here has demonstrated that two
    models sharing a compute capability train byte-identically."""
    from unmark.stage1.device import DeviceContractViolation, ExecutionFingerprint

    base = dict(
        backend="cuda", device="cuda", gpu_name="RTX PRO 6000", compute_capability="9.0",
        torch_version="2.11.0", cuda_version="12.8", cudnn_version=91002,
        deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
        cublas_workspace_config=":4096:8", float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    )
    mine = ExecutionFingerprint(**base)
    other = dict(base, gpu_name="H100")          # same capability, different model
    with pytest.raises(DeviceContractViolation, match="gpu_name"):
        mine.require_compatible(other)
