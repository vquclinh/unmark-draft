"""Scientific Stage-1 execution: device, numerics, fingerprint (D-S1B-015).

Audit 030 §AC.4 established that training ran **silently end-to-end on CPU** and
that no artifact recorded the backend. §AC.11 established that `fp32` was
asserted nowhere on CUDA. This module is the one authoritative resolver for both.

**Scope.** Only the scientific training entry points -- `lr-pilot`, `r-phase1`,
`final-main` -- require CUDA. `smoke`, the measurement tool and the test suite
are explicitly exempt and remain CPU-capable; they never call
`resolve_scientific_device`.

**Ordering matters and is enforced, not assumed.** `CUBLAS_WORKSPACE_CONFIG` is
read by cuBLAS when its handle is created, which happens at the *first* cuBLAS
call, after CUDA initialisation. Setting it afterwards is silently ineffective,
so `require_deterministic_cublas_workspace` refuses when CUDA is already
initialised and the variable is absent or wrong. The runner sets it before torch
touches CUDA; if that ordering was missed, this fails closed rather than
producing a run that merely *claims* determinism.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"
"""One of the two configurations PyTorch documents as deterministic for cuBLAS
(the other is `:16:8`, which is slower). Required for deterministic GEMMs on
CUDA >= 10.2, which is every version this project can run on."""

SCIENTIFIC_DEVICE_BACKEND = "cuda"
"""The single literal name for the required scientific training backend."""

FLOAT32_MATMUL_PRECISION = "highest"
"""True fp32 matmul. `"high"`/`"medium"` would permit TF32 or bf16x3 internally
while every artifact still recorded `precision: fp32` -- exactly the silent
weakening §AC.11 warned about. The scientific constant is unchanged; this makes
the *effective* arithmetic match what the artifact claims."""


class DeviceContractViolation(RuntimeError):
    """Scientific execution cannot proceed under the requested environment."""


def require_deterministic_cublas_workspace() -> str:
    """Ensure `CUBLAS_WORKSPACE_CONFIG` is set, and set *early enough*.

    Returns the effective value. Fails closed rather than setting it too late.
    """
    import torch

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    accepted = {DETERMINISTIC_CUBLAS_WORKSPACE, ":16:8"}
    if current in accepted:
        return current

    if torch.cuda.is_initialized():
        raise DeviceContractViolation(
            f"CUDA is already initialised and CUBLAS_WORKSPACE_CONFIG={current!r}. "
            f"cuBLAS reads this when its handle is created, so setting it now would "
            f"be silently ineffective and the run would claim a determinism it does "
            f"not have. Restart the process with "
            f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE} set before "
            "torch touches CUDA."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE
    return DETERMINISTIC_CUBLAS_WORKSPACE


def resolve_scientific_device() -> Any:
    """The logical CUDA device scientific Stage-1 training must run on.

    `torch.device("cuda")` -- the logical default *visible* device, which honours
    `CUDA_VISIBLE_DEVICES`. No physical index is ever hardcoded, so restricting
    visibility selects the GPU without touching this code.
    """
    import torch

    if not torch.cuda.is_available():
        raise DeviceContractViolation(
            "scientific Stage-1 training requires CUDA and there is none available "
            "(D-S1B-015). There is no CPU fallback: proposal §8.4 budgets Stage 1 at "
            "'a few hours' on one GPU, and a CPU run would silently take a scale of "
            "time nothing in the protocol contemplates. `smoke`, the measurement tool "
            "and the tests remain CPU-capable; scientific training does not."
        )
    return torch.device(SCIENTIFIC_DEVICE_BACKEND)


def enforce_numerical_policy() -> None:
    """Establish the deterministic, true-fp32 policy. Idempotent.

    Enforced **globally** for the process. Note the distinction the audit asked
    for: enabling the cuDNN deterministic policy is a *policy* statement, and this
    architecture happens to exercise no convolution, so that particular flag
    guards nothing today. It is set anyway so a future operator change cannot
    quietly introduce nondeterminism.
    """
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def verify_numerical_policy() -> None:
    """Assert the policy actually holds. Fail closed if anything perturbed it.

    Separate from `enforce_` on purpose: a global setting changed elsewhere in the
    process -- a notebook cell, an imported library -- must not be able to reach a
    scientific run while the artifact still claims `fp32` and 'deterministic'.
    """
    import torch

    problems = []
    if not torch.are_deterministic_algorithms_enabled():
        problems.append("torch.use_deterministic_algorithms is not enabled")
    if not torch.backends.cudnn.deterministic:
        problems.append("torch.backends.cudnn.deterministic is False")
    if torch.backends.cudnn.benchmark:
        problems.append("torch.backends.cudnn.benchmark is True")
    precision = torch.get_float32_matmul_precision()
    if precision != FLOAT32_MATMUL_PRECISION:
        problems.append(
            f"float32 matmul precision is {precision!r}, not {FLOAT32_MATMUL_PRECISION!r}: "
            "fp32 matmuls could run with TF32 internal precision while the artifact "
            "still records precision=fp32"
        )
    if getattr(getattr(torch.backends.cuda, "matmul", None), "allow_tf32", False):
        problems.append("torch.backends.cuda.matmul.allow_tf32 is True")
    if getattr(torch.backends.cudnn, "allow_tf32", False):
        problems.append("torch.backends.cudnn.allow_tf32 is True")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {DETERMINISTIC_CUBLAS_WORKSPACE, ":16:8"}:
        problems.append(f"CUBLAS_WORKSPACE_CONFIG={workspace!r} is not a deterministic value")
    if problems:
        raise DeviceContractViolation(
            "the scientific numerical policy does not hold:\n  - " + "\n  - ".join(problems)
        )


@dataclass(frozen=True)
class ExecutionFingerprint:
    """The effective numerical environment. **Operational, not scientific identity.**

    Kept deliberately separate from `RunProvenance`: `init_seed` determines the
    science and lives there; a GPU's name does not. Only the fields in
    `RESUME_BLOCKING` gate a continuation -- a crash-resume onto a different
    logical device index or a different physical card of the same architecture is
    legitimate, while a change of backend, architecture or numerical policy is not.
    """

    backend: str
    device: str
    gpu_name: str | None
    compute_capability: str | None
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool | None
    cudnn_allow_tf32: bool | None

    RESUME_BLOCKING: tuple[str, ...] = (
        "backend",
        "gpu_name",
        "compute_capability",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    )
    """Everything that can change the arithmetic.

    **`gpu_name` blocks, conservatively.** Compute capability is the property one
    would *expect* to govern kernel selection, but nothing in this repository has
    demonstrated that two different GPU models sharing a capability produce
    byte-identical interrupted-vs-uninterrupted training, and the project's
    reproducibility claim is too strong to rest on that assumption. Until a CUDA
    experiment proves cross-model identity, a continuation stays on the same model.
    Relaxing this later is an explicit decision, not a default.

    **Excluded on purpose:** `device`, a logical index that `CUDA_VISIBLE_DEVICES`
    freely renumbers, and the physical GPU UUID, which is not recorded here at all —
    neither changes the arithmetic, so neither may block a legitimate crash-resume
    onto the same model of card."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cublas_workspace_config": self.cublas_workspace_config,
            "float32_matmul_precision": self.float32_matmul_precision,
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
        }

    def require_compatible(self, other: dict[str, Any]) -> None:
        """Refuse to continue a run across a numerically different environment."""
        mine = self.to_dict()
        for key in self.RESUME_BLOCKING:
            if other.get(key) != mine[key]:
                raise DeviceContractViolation(
                    f"execution fingerprint mismatch on {key!r}: the checkpoint was "
                    f"written under {other.get(key)!r}, this environment is {mine[key]!r}. "
                    "A continuation may not silently cross a numerically different "
                    "execution contract (D-S1B-015)."
                )


def current_fingerprint(device: Any) -> ExecutionFingerprint:
    """Capture the effective environment. Call after `enforce_numerical_policy`."""
    import torch

    on_cuda = getattr(device, "type", str(device)) == "cuda"
    name = capability = None
    if on_cuda:
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        capability = f"{major}.{minor}"
    cudnn_version = torch.backends.cudnn.version() if on_cuda else None
    return ExecutionFingerprint(
        backend=SCIENTIFIC_DEVICE_BACKEND if on_cuda else "cpu",
        device=str(device),
        gpu_name=name,
        compute_capability=capability,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=cudnn_version,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=getattr(
            getattr(torch.backends.cuda, "matmul", None), "allow_tf32", None
        ),
        cudnn_allow_tf32=getattr(torch.backends.cudnn, "allow_tf32", None),
    )
