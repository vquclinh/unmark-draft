"""Tokenizer input-contract structures (B3B-0).

Preparation for PhoBERT alignment, not the alignment itself. No transformers, no
torch, no tokenizer: this package only names and measures the alternatives so the
input contract can be settled empirically. See `docs/spec/decisions.md` D-B3B0-001.
"""

from unmark.alignment.contracts import (
    PROBE_CONDITIONS,
    REPO_LOCAL_HF_CACHE,
    AlignmentStatus,
    OffsetAvailability,
    PathAvailability,
    PreprocessingPath,
    SegmenterContract,
    TokenizerContract,
)
from unmark.alignment.probe_models import PathObservation, compare_paths, grid_invariance, path_summary
from unmark.alignment.spans import (
    TokenSpan,
    alignment_status,
    character_coverage,
    syllable_token_map,
    tokens_for_span,
    validate_offsets,
)

__all__ = [
    "PROBE_CONDITIONS",
    "REPO_LOCAL_HF_CACHE",
    "AlignmentStatus",
    "OffsetAvailability",
    "PathAvailability",
    "PathObservation",
    "PreprocessingPath",
    "SegmenterContract",
    "TokenSpan",
    "TokenizerContract",
    "alignment_status",
    "character_coverage",
    "compare_paths",
    "grid_invariance",
    "path_summary",
    "syllable_token_map",
    "tokens_for_span",
    "validate_offsets",
]
