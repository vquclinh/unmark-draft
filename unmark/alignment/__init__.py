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
    AlignmentStatusB,
    Chunk,
    ChunkAlignment,
    OrthographicRegion,
    PieceAlignment,
    PieceContribution,
    PieceOverlay,
    ToneOwnership,
    align_chunk,
    characters_for_piece,
    compose,
    overlay_orthography,
    piece_surface,
    reconstruct_surface,
    summarize_chunk_alignments,
    verify_token_grid,
    whitespace_chunks,
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
    "AlignmentFailureReason",
    "AlignmentStatus",
    "AlignmentStatusB",
    "CONTINUATION_MARKER",
    "Chunk",
    "ChunkAlignment",
    "OffsetAvailability",
    "OrthographicRegion",
    "PROBE_CONDITIONS",
    "PathAvailability",
    "PathObservation",
    "PieceAlignment",
    "PieceContribution",
    "PieceOverlay",
    "PreprocessingPath",
    "REPO_LOCAL_HF_CACHE",
    "SegmenterContract",
    "TokenSpan",
    "TokenizerContract",
    "ToneOwnership",
    "align_chunk",
    "alignment_status",
    "character_coverage",
    "characters_for_piece",
    "compare_paths",
    "compose",
    "grid_invariance",
    "overlay_orthography",
    "path_summary",
    "piece_surface",
    "reconstruct_surface",
    "summarize_chunk_alignments",
    "syllable_token_map",
    "tokens_for_span",
    "validate_offsets",
    "verify_token_grid",
    "whitespace_chunks",
]
