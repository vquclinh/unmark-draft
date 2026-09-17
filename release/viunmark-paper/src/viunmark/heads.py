"""Robust MLP readout head."""

from __future__ import annotations

from typing import Any

from viunmark.config import (
    ROBUST_MLP_DROPOUT,
    ROBUST_MLP_HIDDEN_DIM,
    ROBUST_MLP_LAYERS,
    RobustMLPHeadConfig,
    ViUnMarkContractError,
)
from viunmark.training import (
    LAYERNORM_BIAS_INIT,
    LAYERNORM_WEIGHT_INIT,
    LINEAR_BIAS_INIT,
    ROBUST_MLP_OPTIMIZER_POLICY,
    RobustMLPOptimizerPolicy,
)


def robust_mlp_layers(config: RobustMLPHeadConfig) -> tuple[tuple[str, Any], ...]:
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
    if tuple(name for name, _ in layers) != ROBUST_MLP_LAYERS:
        raise AssertionError("robust MLP layer order drifted")
    return layers


def build_robust_mlp_head(config: RobustMLPHeadConfig) -> Any:
    from torch import nn

    robust_mlp_layers(config)
    head = nn.Sequential(
        nn.LayerNorm(config.input_dim),
        nn.Linear(config.input_dim, ROBUST_MLP_HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(p=ROBUST_MLP_DROPOUT),
        nn.Linear(ROBUST_MLP_HIDDEN_DIM, config.num_labels),
    )
    if sum(parameter.numel() for parameter in head.parameters()) != config.parameter_count:
        raise ViUnMarkContractError("built head parameter count disagrees with config")
    return head


def initialize_robust_mlp_head(head: Any) -> Any:
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
    matrices = [p for p in head.parameters() if p.requires_grad and p.dim() >= 2]
    others = [p for p in head.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": matrices, "weight_decay": policy.weight_decay_for(2)},
        {"params": others, "weight_decay": policy.weight_decay_for(1)},
    ]


def build_robust_mlp_optimizer(
    head: Any, policy: RobustMLPOptimizerPolicy = ROBUST_MLP_OPTIMIZER_POLICY
) -> Any:
    import torch

    if policy.name != "AdamW":
        raise ViUnMarkContractError(f"unsupported optimizer {policy.name!r}")
    return torch.optim.AdamW(
        robust_mlp_parameter_groups(head, policy),
        lr=policy.learning_rate,
        betas=policy.betas,
        eps=policy.eps,
    )
