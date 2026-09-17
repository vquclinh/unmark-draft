import math

import pytest

from viunmark.config import (
    CONCAT_ORDER,
    GATE_INIT_BIAS,
    ReadoutKind,
    RobustMLPHeadConfig,
    ViUnMarkGateConfig,
    ViUnMarkScaleConfig,
)
from viunmark.losses import sqrt_inverse_frequency_weights


def test_adapter_identity_and_gate_initialization():
    gate = ViUnMarkGateConfig()
    scale = ViUnMarkScaleConfig()
    assert (gate.tone_rows, gate.letter_rows) == (7, 5)
    assert gate.gate_weight_initial == 0.0
    assert GATE_INIT_BIAS == pytest.approx(math.log(0.01 / 0.99))
    assert gate.fusion_equation == "z = g * f + (1 - g) * e"
    assert scale.scale_epsilon == 1e-8


def test_readout_and_head_shapes():
    assert CONCAT_ORDER == (ReadoutKind.FIRST_TOKEN, ReadoutKind.MASKED_MEAN)
    assert ReadoutKind.CONCAT.feature_dim(768) == 1536
    assert RobustMLPHeadConfig(768, 3).parameter_count == 199171
    assert RobustMLPHeadConfig(1536, 3).parameter_count == 397315


def test_weighted_ce_rule_and_uit_vsfc_vector():
    assert sqrt_inverse_frequency_weights((4259, 366, 4514), arithmetic="float32") == (
        0.5573523044586182,
        1.9012669324874878,
        0.5413808226585388,
    )
    assert sqrt_inverse_frequency_weights((10, 10, 10)) == pytest.approx((1.0, 1.0, 1.0))
