from viunmark.diagnostics import (
    COMPUTABLE_FACTORS,
    FACTORIZATION_ORDER,
    UNRESOLVED_FACTORS,
    cosine_distances,
    matched_recipe_streams,
    prediction_agreement,
    validate_scale_pathway_config,
)
from viunmark.config import LogitStream


def test_descriptive_diagnostics():
    assert validate_scale_pathway_config()["pathway"] == "ViUnMark-Scale"
    assert set(matched_recipe_streams()) == {
        LogitStream.GATE_MATCHED_RECIPE_READOUT,
        LogitStream.SCALE_UNWEIGHTED_READOUT,
        LogitStream.PHOBERT_READOUT,
    }
    assert UNRESOLVED_FACTORS == FACTORIZATION_ORDER[:3]
    assert COMPUTABLE_FACTORS == FACTORIZATION_ORDER[3:]


def test_decision_geometry_helpers():
    assert cosine_distances([[1.0, 0.0]], [[1.0, 0.0]]) == (0.0,)
    assert prediction_agreement([[2.0, 1.0], [0.0, 1.0]], [[3.0, 0.0], [2.0, 1.0]]) == 0.5
