from viunmark.config import CalibrationConfig, LogitStream, ViUnMarkConfig
from viunmark.fusion import (
    BEST_SEED_SELECTION,
    PARENT_BIASES_STACKED,
    WITHIN_BRANCH_LOGIT_ENSEMBLE,
    HeadLogits,
    ensemble_branch,
    expanded_viunmark_weights,
    viunmark_fusion,
)

SEEDS = (53148, 59945, 42941, 720, 9428)
ROWS = ("a", "b")


def heads(stream, base):
    return [HeadLogits(stream, seed, ROWS, [[base + i, 0.0, 1.0], [0.0, base, float(i)]]) for i, seed in enumerate(SEEDS)]


def branch(stream, base):
    return ensemble_branch(heads(stream, base), expected_seeds=SEEDS)


def test_final_weights_and_ensemble_policy():
    assert WITHIN_BRANCH_LOGIT_ENSEMBLE == "MEAN"
    assert BEST_SEED_SELECTION is False
    assert PARENT_BIASES_STACKED is False
    assert expanded_viunmark_weights() == {
        LogitStream.PHOBERT_READOUT: 0.5,
        LogitStream.GATE_ROBUST_READOUT: 0.25,
        LogitStream.SCALE_UNWEIGHTED_READOUT: 0.125,
        LogitStream.SCALE_WEIGHTED_READOUT: 0.125,
    }
    assert ViUnMarkConfig(num_labels=3, head_seeds=SEEDS).head_count == 20


def test_viunmark_calibrates_once_after_raw_fusion():
    gate = branch(LogitStream.GATE_ROBUST_READOUT, 1.0)
    scale_u = branch(LogitStream.SCALE_UNWEIGHTED_READOUT, 2.0)
    scale_w = branch(LogitStream.SCALE_WEIGHTED_READOUT, 4.0)
    phobert = branch(LogitStream.PHOBERT_READOUT, 8.0)
    final = viunmark_fusion(gate, scale_u, scale_w, phobert, CalibrationConfig(1, 1.25, "neutral"))
    raw = viunmark_fusion(gate, scale_u, scale_w, phobert, CalibrationConfig.identity())
    assert final.values == tuple((row[0], row[1] + 1.25, row[2]) for row in raw.values)
