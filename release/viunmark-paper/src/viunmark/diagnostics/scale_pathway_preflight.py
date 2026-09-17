"""Scale pathway preflight checks."""

from viunmark.config import ViUnMarkScaleConfig


def validate_scale_pathway_config() -> dict[str, object]:
    config = ViUnMarkScaleConfig()
    return {
        "pathway": config.pathway.value,
        "fusion_id": config.fusion_id,
        "scale_epsilon": config.scale_epsilon,
        "hidden_size": config.hidden_size,
        "tone_rows": config.tone_rows,
        "letter_rows": config.letter_rows,
    }
