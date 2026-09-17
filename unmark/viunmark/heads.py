"""The robust MLP classification head. **Torch is imported lazily.**

    LayerNorm(input_dim)
    Linear(input_dim, 256)
    GELU
    Dropout(p=0.1)
    Linear(256, num_labels)

The layer order, the widths and the dropout rate are fixed. `robust_mlp_layers`
and `RobustMLPHeadConfig.parameter_count` describe the head without torch;
`build_robust_mlp_head` builds it.

`build_robust_mlp_head` defines the architecture only.
`initialize_robust_mlp_head` applies the recovered initialisation policy:
LayerNorm weight 1 and bias 0, Linear Xavier-uniform weight and zero bias.
`build_robust_mlp_optimizer` applies the recovered AdamW policy, with weight decay
on matrix weights only.

How the random stream that draws the Xavier weights is seeded is not part of the
recovered policy. `initialize_robust_mlp_head` draws from the caller's current
torch RNG state and does not seed it.
"""

from __future__ import annotations

from typing import Any

from unmark.viunmark.config import (
    ROBUST_MLP_DROPOUT,
    ROBUST_MLP_HIDDEN_DIM,
    ROBUST_MLP_LAYERS,
    RobustMLPHeadConfig,
    ViUnMarkContractError,
)
from unmark.viunmark.training import (
    LAYERNORM_BIAS_INIT,
    LAYERNORM_WEIGHT_INIT,
    LINEAR_BIAS_INIT,
    ROBUST_MLP_OPTIMIZER_POLICY,
    RobustMLPOptimizerPolicy,
)


def robust_mlp_layers(config: RobustMLPHeadConfig) -> tuple[tuple[str, Any], ...]:
    """The layer sequence as `(layer, arguments)` pairs. Torch-free."""
    if not isinstance(config, RobustMLPHeadConfig):
        raise ViUnMarkContractError("config must be a RobustMLPHeadConfig")
    d, h, n = config.input_dim, ROBUST_MLP_HIDDEN_DIM, config.num_labels
    layers = (
        ("LayerNorm", (d,)),
        ("Linear", (d, h)),
        ("GELU", ()),
        ("Dropout", (ROBUST_MLP_DROPOUT,)),
        ("Linear", (h, n)),
    )
    if tuple(name for name, _ in layers) != ROBUST_MLP_LAYERS:  # pragma: no cover - guard
        raise AssertionError("robust MLP layer order drifted from ROBUST_MLP_LAYERS")
    return layers


def build_robust_mlp_head(config: RobustMLPHeadConfig) -> Any:
    """Build the head as a `torch.nn.Sequential` in the fixed layer order."""
    from torch import nn

    robust_mlp_layers(config)
    head = nn.Sequential(
        nn.LayerNorm(config.input_dim),
        nn.Linear(config.input_dim, ROBUST_MLP_HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(p=ROBUST_MLP_DROPOUT),
        nn.Linear(ROBUST_MLP_HIDDEN_DIM, config.num_labels),
    )
    built = sum(parameter.numel() for parameter in head.parameters())
    if built != config.parameter_count:
        raise ViUnMarkContractError(
            f"built head has {built} parameters, the architecture defines "
            f"{config.parameter_count}"
        )
    return head


def initialize_robust_mlp_head(head: Any) -> Any:
    """Apply the recovered initialisation to every LayerNorm and Linear. Returns `head`."""
    import torch
    from torch import nn

    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, LAYERNORM_WEIGHT_INIT)
                nn.init.constant_(module.bias, LAYERNORM_BIAS_INIT)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, LINEAR_BIAS_INIT)
    return head


def robust_mlp_parameter_groups(
    head: Any, policy: RobustMLPOptimizerPolicy = ROBUST_MLP_OPTIMIZER_POLICY
) -> list[dict[str, Any]]:
    """Two AdamW groups: matrix weights (decayed) and everything else (not decayed)."""
    matrices = [p for p in head.parameters() if p.requires_grad and p.dim() >= 2]
    others = [p for p in head.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": matrices, "weight_decay": policy.weight_decay_for(2)},
        {"params": others, "weight_decay": policy.weight_decay_for(1)},
    ]


def build_robust_mlp_optimizer(
    head: Any, policy: RobustMLPOptimizerPolicy = ROBUST_MLP_OPTIMIZER_POLICY
) -> Any:
    """AdamW over the head's parameters under the recovered policy."""
    import torch

    if policy.name != "AdamW":
        raise ViUnMarkContractError(f"unsupported optimizer {policy.name!r}")
    return torch.optim.AdamW(
        robust_mlp_parameter_groups(head, policy),
        lr=policy.learning_rate,
        betas=policy.betas,
        eps=policy.eps,
    )


__all__ = [
    "build_robust_mlp_head",
    "build_robust_mlp_optimizer",
    "initialize_robust_mlp_head",
    "robust_mlp_layers",
    "robust_mlp_parameter_groups",
]
