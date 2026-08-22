"""Stage-1 optimizer construction. **The grouping logic is torch-free.**

`parameter_group_plan` decides decay purely from parameter *names*, so the
locked weight-decay policy is testable without building a model, and
`build_optimizer` is a thin wrapper that applies the plan.

Locked (D-S1B-004): AdamW, betas `(0.9, 0.999)`, eps `1e-8`, amsgrad `False`,
constant LR, no warmup, accumulation `1`, **no clipping initially**,
`0.01` on fusion/gate weight matrices and `0.0` on biases, LayerNorm, and
**both embedding tables**.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    ADAMW_BETAS,
    ADAMW_EPS,
    AMSGRAD,
    WEIGHT_DECAY_EXEMPT,
    WEIGHT_DECAY_WEIGHTS,
)

NO_DECAY_MARKERS: tuple[str, ...] = (
    ".bias",
    "layernorm",
    "layer_norm",
    "tone_embedding",
    "letter_embedding",
    "tone.weight",
    "letter.weight",
)
"""Substrings that exempt a parameter from weight decay.

The **embedding** entries are the scientifically motivated ones: decaying the
tone/letter tables would shrink channel information toward zero, the opposite of
Stage-1's purpose. The bias/LayerNorm entries follow the audited pre-G1 house
style.
"""


def is_decayed(name: str) -> bool:
    """Whether weight decay applies to a parameter, by name alone."""
    lowered = name.lower()
    return not any(marker in lowered for marker in NO_DECAY_MARKERS)


def parameter_group_plan(names: Sequence[str]) -> dict[str, Any]:
    """Split parameter names into the two locked decay groups.

    Pure data, so the policy can be asserted without torch and without a model.
    """
    if not names:
        raise Stage1ContractViolation("no trainable parameter names supplied")
    decay = sorted(n for n in names if is_decayed(n))
    exempt = sorted(n for n in names if not is_decayed(n))
    if not decay:
        raise Stage1ContractViolation(
            f"every parameter was exempted from weight decay: {exempt}. The fusion and "
            "gate weight matrices must be decayed."
        )
    return {
        "decay": {"names": decay, "weight_decay": WEIGHT_DECAY_WEIGHTS},
        "exempt": {"names": exempt, "weight_decay": WEIGHT_DECAY_EXEMPT},
    }


def build_optimizer(named_parameters: Iterable[tuple[str, Any]], learning_rate: float):
    """AdamW over the adapter only. **Imports torch lazily.**

    Every supplied parameter must require grad: the frozen encoder's parameters
    must never reach this function, and silently filtering them out would hide a
    wiring error instead of surfacing it.
    """
    import torch

    pairs = list(named_parameters)
    if not pairs:
        raise Stage1ContractViolation("no parameters supplied to the optimizer")
    frozen = [n for n, p in pairs if not p.requires_grad]
    if frozen:
        raise Stage1ContractViolation(
            f"{len(frozen)} supplied parameter(s) do not require grad, e.g. {frozen[:5]}. "
            "The encoder is frozen and must not be handed to the optimizer at all."
        )
    plan = parameter_group_plan([n for n, _ in pairs])
    by_name = dict(pairs)
    groups = [
        {
            "params": [by_name[n] for n in plan["decay"]["names"]],
            "weight_decay": WEIGHT_DECAY_WEIGHTS,
        },
        {
            "params": [by_name[n] for n in plan["exempt"]["names"]],
            "weight_decay": WEIGHT_DECAY_EXEMPT,
        },
    ]
    return torch.optim.AdamW(
        groups, lr=learning_rate, betas=ADAMW_BETAS, eps=ADAMW_EPS, amsgrad=AMSGRAD
    )
