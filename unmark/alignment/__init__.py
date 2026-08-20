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
from unmark.alignment.manual import (
    CONTINUATION_MARKER,
    AlignmentFailureReason,
    PieceAlignment,
    SequenceAlignment,
    SpanAlignment,
    SpanAlignmentStatus,
    align_span,
    characters_for_piece,
    compare_sequences,
    pieces_for_character,
    piece_surface,
    reconstruct_surface,
    summarize_alignments,
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
    "CONTINUATION_MARKER",
    "PROBE_CONDITIONS",
    "REPO_LOCAL_HF_CACHE",
    "AlignmentFailureReason",
    "AlignmentStatus",
    "OffsetAvailability",
    "PathAvailability",
    "PathObservation",
    "PieceAlignment",
    "SequenceAlignment",
    "SpanAlignment",
    "SpanAlignmentStatus",
    "PreprocessingPath",
    "SegmenterContract",
    "TokenSpan",
    "TokenizerContract",
    "align_span",
    "alignment_status",
    "character_coverage",
    "characters_for_piece",
    "compare_paths",
    "compare_sequences",
    "grid_invariance",
    "path_summary",
    "piece_surface",
    "pieces_for_character",
    "reconstruct_surface",
    "summarize_alignments",
    "syllable_token_map",
    "tokens_for_span",
    "validate_offsets",
]
