"""ViUnMark analyses, under descriptive names. **Torch-free at import.**

    pooling_bridge       Checkpoint-Free Pooling Bridge
    pooling_comparison   Matched-Head Pooling Comparison
    decision_geometry    Native-Adapted Decision Geometry
    complementarity      Matched-Recipe Complementarity Analysis
    factorization        Robustness Gain Factorization

Each module implements only the parts of its analysis whose calculation is fully
specified, and says in its docstring what it leaves out and why.
"""

from unmark.viunmark.diagnostics.complementarity import (
    MatchedRecipeAverage,
    matched_recipe_average,
    matched_recipe_streams,
)
from unmark.viunmark.diagnostics.decision_geometry import cosine_distances, prediction_agreement
from unmark.viunmark.diagnostics.factorization import (
    COMPUTABLE_FACTORS,
    FACTORIZATION_ORDER,
    UNRESOLVED_FACTORS,
    RobustnessFactor,
    final_system_factors,
)
from unmark.viunmark.diagnostics.pooling_bridge import same_forward_pooling_pair
from unmark.viunmark.diagnostics.pooling_comparison import MatchedPoolingComparisonPlan

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
]
