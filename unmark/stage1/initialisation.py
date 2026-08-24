"""Deterministic, hardware-independent adapter initialisation (D-S1B-016/017).

Audit 030 §AC.7 found that nothing seeded torch, so a published `run_seed` could
not reproduce its own run. §AD found the more fundamental defect: nominal runs
2..N were never initialised **at all**, because one `UnmarkEncoder` was built
before the schedule loop and every candidate trained the same adapter in place.

This module supplies the two things that repair requires:

* `fresh_adapter` -- a **new** adapter, initialised on **CPU** inside an isolated
  RNG scope, so the values depend on `init_seed` and on nothing else;
* `trainable_state_hash` -- one canonical hash of trainable state, so "this run
  started from the state its seed says it should" is checkable rather than
  assumed.

**Why CPU.** CUDA has a different RNG backend, so initialising there would make a
run's scientific starting point depend on the accelerator it happened to get.
Initialisation is a scientific identity; it must be hardware-independent.
"""

from __future__ import annotations

import hashlib
from typing import Any

from unmark.stage1.protocol import adapter_init_seed


def fresh_adapter(hidden_size: int, init_seed: int) -> Any:
    """A new `OrthographyInputAdapter` on CPU, initialised from `init_seed` alone.

    The RNG is **forked**, not merely seeded: `torch.random.fork_rng(devices=[])`
    snapshots and restores the **CPU** generator on exit, so constructing an
    adapter cannot perturb any other stream and no prior RNG consumption in the
    process can perturb the adapter.

    **`torch.manual_seed` must NOT be used here.** Its documented contract is to
    seed "all devices": it calls `torch.cuda.manual_seed_all` before seeding the
    CPU generator. `fork_rng(devices=[])` snapshots only the CPU state, so that
    combination would perturb every CUDA generator *without restoring it* -- and
    when CUDA is not yet initialised, `manual_seed_all` defers through
    `_lazy_call`, queueing a seed that fires later, at CUDA init. Either way the
    ambient accelerator RNG is silently altered by an operation that is supposed
    to be a pure CPU construction.

    `torch.default_generator` **is** the CPU default generator -- the one every
    `nn.Linear`/`nn.Embedding`/`nn.LayerNorm` `reset_parameters` draws from for a
    CPU tensor -- so seeding it directly is both exactly sufficient and strictly
    confined. No CUDA generator is read, written, or initialised.
    """
    import torch

    from unmark.modeling.adapter import OrthographyInputAdapter
    from unmark.modeling.config import AdapterConfig

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(init_seed))
        adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=hidden_size))
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    return adapter


def trainable_state(adapter: Any) -> dict[str, Any]:
    """The authoritative trainable state of one adapter.

    `state_dict()` of the adapter module itself -- not of the wrapper, which would
    drag in the frozen encoder (Audit 030 §AC.9). Asserted elsewhere to be exactly
    the trainable parameter set.
    """
    return adapter.state_dict()


def trainable_state_hash(state: dict[str, Any]) -> str:
    """Canonical SHA-256 over a trainable state dict.

    Canonical ordering is **sorted key name**, and each entry contributes its
    name, dtype, shape and raw bytes, so a renamed, re-typed, reshaped or
    re-valued tensor all change the digest. Tensors are moved to CPU and made
    contiguous first, so the same state hashes identically whether it currently
    lives on CPU or CUDA -- that is what lets an expected fresh-init hash computed
    on CPU be compared against a model already placed on an accelerator.
    """
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:  # pragma: no cover - the adapter state is all tensors today
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def expected_fresh_init_hash(hidden_size: int, run_seed: int) -> str:
    """The hash a fresh nominal run with this `run_seed` must start from.

    Pure function of `(hidden_size, run_seed)`. Because all eight
    hyperparameter-selection candidates share `SELECTION_SEED`, they share this
    value -- deliberately (D-S1B-016). Equal hashes are therefore **expected** and
    prove nothing about object identity; storage independence is a separate
    contract, proven by mutation isolation.
    """
    return trainable_state_hash(
        trainable_state(fresh_adapter(hidden_size, adapter_init_seed(run_seed)))
    )


def module_state_hash(module: Any) -> str:
    """Canonical hash of a module's **full** `state_dict` -- parameters *and*
    persistent buffers. Used for the frozen encoder's immutability check, where
    parameters alone would miss a mutated buffer (Audit 030 §AD.6(A))."""
    return trainable_state_hash(module.state_dict())
