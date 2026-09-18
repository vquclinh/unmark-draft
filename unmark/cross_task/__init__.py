"""Cross-task transfer experiments for frozen ViUnMark pathways.

The package stays import-light. Torch and Transformers are imported lazily by
the runner paths only, so local contract tests do not need model downloads.
"""

from unmark.cross_task.phoner_transfer import (
    PHONER_CROSS_TASK_PROTOCOL_VERSION,
    PHONER_CORRUPTION_SEED,
    PHONER_GATE_SHA256,
    PHONER_REPRESENTATION,
    PHONER_SCALE_SHA256,
    PHONER_TASK_ID,
    CORRUPTION_PARITY_CONCLUSION,
    PhoNERCrossTaskConfig,
    PhoNERDatasetIdentity,
    PhoNERExample,
    PhoNERPathway,
    EntityF1,
    FrozenTokenProbe,
    align_word_labels,
    audit_condition_invariant_coverage,
    build_phoner_dataset_identity,
    corrupt_phoner_example,
    cross_pathway_corrupted_inputs,
    parse_phoner_split,
    require_stage_access,
)

__all__ = [
    "PHONER_CROSS_TASK_PROTOCOL_VERSION",
    "PHONER_CORRUPTION_SEED",
    "PHONER_GATE_SHA256",
    "PHONER_REPRESENTATION",
    "PHONER_SCALE_SHA256",
    "PHONER_TASK_ID",
    "CORRUPTION_PARITY_CONCLUSION",
    "EntityF1",
    "FrozenTokenProbe",
    "PhoNERCrossTaskConfig",
    "PhoNERDatasetIdentity",
    "PhoNERExample",
    "PhoNERPathway",
    "align_word_labels",
    "audit_condition_invariant_coverage",
    "build_phoner_dataset_identity",
    "corrupt_phoner_example",
    "cross_pathway_corrupted_inputs",
    "parse_phoner_split",
    "require_stage_access",
]
