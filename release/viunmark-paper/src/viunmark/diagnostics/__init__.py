"""Descriptive diagnostic interfaces."""

from viunmark.diagnostics.complementarity import (
    MatchedRecipeAverage,
    matched_recipe_average,
    matched_recipe_streams,
)
from viunmark.diagnostics.decision_geometry import cosine_distances, prediction_agreement
from viunmark.diagnostics.gain_factorization import (
    COMPUTABLE_FACTORS,
    FACTORIZATION_ORDER,
    UNRESOLVED_FACTORS,
    RobustnessFactor,
    final_system_factors,
)
from viunmark.diagnostics.pooling_bridge import same_forward_pooling_pair
from viunmark.diagnostics.pooling_comparison import MatchedPoolingComparisonPlan
from viunmark.diagnostics.scale_pathway_preflight import validate_scale_pathway_config

__all__ = [
    "COMPUTABLE_FACTORS",
    "FACTORIZATION_ORDER",
    "MatchedPoolingComparisonPlan",
    "MatchedRecipeAverage",
    "RobustnessFactor",
    "UNRESOLVED_FACTORS",
    "cosine_distances",
    "final_system_factors",
    "matched_recipe_average",
    "matched_recipe_streams",
    "prediction_agreement",
    "same_forward_pooling_pair",
    "validate_scale_pathway_config",
]
