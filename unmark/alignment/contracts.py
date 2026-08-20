"""Input-contract vocabulary for the frozen tokenizer (B3B-0).

Nothing here imports transformers or torch. It is the typed vocabulary the Colab
probe reports in, and the local analysis reasons over, so that the tokenizer
question can be settled with evidence rather than assumption.

The open question
-----------------
Proposal v1.3 §4.4 writes the token grid as::

    T(b(x))

-- the frozen tokenizer applied directly to the base (stripped) text -- and §4.3
records the base stream as "fully stripped letters; tokenized by the frozen
tokenizer". Neither mentions word segmentation.

PhoBERT's published usage contract does mention it: its input is expected to be
Vietnamese **word-segmented** text (underscore-joined compounds such as
`nghiên_cứu`), produced by the RDRSegmenter/VnCoreNLP preprocessing used during
pretraining. Operationally that is::

    T(S(b(x)))

for some segmentation function `S`. Whether `S` belongs in the pipeline, and if
so where, is a scientific question -- it decides what distribution the frozen
encoder sees, whether the base grid stays corruption-invariant, and whether a
segmenter that reads diacritics smuggles in a restoration signal.

**B3B-0 does not answer it.** This module only names the alternatives precisely
enough to measure them. See `docs/spec/decisions.md` D-B3B0-001.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PreprocessingPath(Enum):
    """Candidate pipelines between clean text and tokenizer input.

    Deliberately enumerated without a preferred value: the probe measures them,
    the researcher chooses.
    """

    RAW_BASE = "RAW_BASE"
    """`T(b(x))` -- tokenize the stripped base directly, no segmentation.
    Matches the proposal's notation exactly; ignores PhoBERT's stated
    preprocessing contract."""

    CLEAN_SEGMENT_THEN_BASE = "CLEAN_SEGMENT_THEN_BASE"
    """Segment the canonical *clean* text, then strip. Segmentation sees
    diacritics, which is a deployability problem (clean text is unavailable at
    inference) and a hidden-restoration risk."""

    BASE_THEN_SEGMENT = "BASE_THEN_SEGMENT"
    """Strip to the invariant base, then segment the base text. Deployable, but
    the segmenter is being run out of its training distribution."""

    PRESEGMENTED_DATASET = "PRESEGMENTED_DATASET"
    """Use segmentation shipped with the dataset. Reproducible, but ties the
    pipeline to per-dataset preprocessing."""

    OBSERVED_SEGMENT_THEN_BASE = "OBSERVED_SEGMENT_THEN_BASE"
    """Segment whatever text is actually observed at inference (possibly
    corrupted), then strip. Deployable and honest, but segmentation output then
    depends on the corruption level, which threatens grid invariance."""


class PathAvailability(Enum):
    """Whether a path could be exercised in a given probe run."""

    OK = "OK"
    UNAVAILABLE_SEGMENTER = "UNAVAILABLE_SEGMENTER"
    """VnCoreNLP / py_vncorenlp or its model resource was not present. Reported
    as unavailable, never faked."""
    UNAVAILABLE_TOKENIZER = "UNAVAILABLE_TOKENIZER"
    ERROR = "ERROR"


class OffsetAvailability(Enum):
    """How usable the tokenizer's offset mapping is.

    §4.4 step 2 assigns channel labels "by tracking character offsets through
    tokenization". That presupposes offsets exist and mean what we need. Neither
    is guaranteed for a given tokenizer class, so it is measured.
    """

    NATIVE_EXACT = "NATIVE_EXACT"
    """`offset_mapping` present, and every non-special token's slice reproduces
    its surface form."""

    NATIVE_INEXACT = "NATIVE_INEXACT"
    """Offsets present but at least one token's slice does not match its surface
    form -- normalisation happened inside the tokenizer."""

    NATIVE_MALFORMED = "NATIVE_MALFORMED"
    """Offsets present but structurally unusable: wrong length, reversed,
    out of range, or overlapping in a way that breaks span assignment."""

    ABSENT = "ABSENT"
    """No `offset_mapping` at all -- typical of slow (non-fast) tokenizers."""

    NOT_PROBED = "NOT_PROBED"


class AlignmentStatus(Enum):
    """Whether base characters could be mapped to tokens for this observation."""

    ALIGNED = "ALIGNED"
    PARTIAL = "PARTIAL"
    UNALIGNED = "UNALIGNED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass(frozen=True)
class TokenizerContract:
    """What a probe run observed about the tokenizer itself.

    Recorded per run so a later result can never be attributed to the wrong
    tokenizer version.
    """

    checkpoint: str

    revision_requested: str | None
    """The full immutable commit SHA the caller asked for. Supplying it is
    necessary but NOT sufficient: `revision=` is an argument, not a
    verification."""

    revision_observed: str | None = None
    """The commit actually resolved, read back from the loaded tokenizer's own
    files after loading. `None` when it could not be determined -- never
    fabricated from the request."""

    revision_verified: bool = False
    """True only when `revision_observed` was recoverable AND equals
    `revision_requested`."""

    revision_evidence: tuple[str, ...] = ()
    """The resolved file paths the observed revision was read from."""

    revision_evidence_source: str = ""
    """How the revision was recovered, or why it could not be."""

    tokenizer_class: str = ""
    is_fast: bool = False
    vocab_size: int | None = None
    unk_token: str | None = None
    special_tokens: tuple[str, ...] = ()
    word_segmentation_expected: bool | None = None
    """Whether the model's documentation states that input must be
    word-segmented. Recorded as observed, not inferred."""
    transformers_version: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision_requested": self.revision_requested,
            "revision_observed": self.revision_observed,
            "revision_verified": self.revision_verified,
            "revision_evidence": list(self.revision_evidence),
            "revision_evidence_source": self.revision_evidence_source,
            "tokenizer_class": self.tokenizer_class,
            "is_fast": self.is_fast,
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "special_tokens": list(self.special_tokens),
            "word_segmentation_expected": self.word_segmentation_expected,
            "transformers_version": self.transformers_version,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SegmenterContract:
    """What produced the word segmentation, if anything did."""

    available: bool
    name: str | None = None
    package: str | None = None
    package_version: str | None = None
    model_resource: str | None = None
    model_version: str | None = None
    jar_name: str | None = None
    """The jar actually loaded. Equal to `required_jar` or the load was refused."""
    required_jar: str | None = None
    """The jar named by the committed pin. Never discovered by globbing."""
    other_jars_present: tuple[str, ...] = ()
    """Any other VnCoreNLP jars in the directory, reported but never substituted."""
    manifest_path: str | None = None

    manifest_revision: str | None = None
    """The Git revision the committed pin names."""
    observed_revision: str | None = None
    """`git rev-parse HEAD` of the provisioned checkout, or None when .git is absent."""
    revision_verified: bool = False
    """True only when the observed revision equals the pinned one. Never inferred."""
    observed_tags_at_head: tuple[str, ...] = ()
    """Diagnostic only. Tag text alone never constitutes verification."""

    expected_hashes: dict[str, str] = field(default_factory=dict)
    """Digests the pin requires."""
    resource_hashes: dict[str, str] = field(default_factory=dict)
    """SHA-256 of every required resource file, as observed by this run."""
    hashes_verified: bool = False
    """True only when every observed digest equalled the pinned one."""
    pinned: bool = False
    """True ONLY when this run verified every observed resource against externally
    supplied hashes. Files merely existing never counts as pinned -- the first
    Colab run was invalidated precisely because provenance was assumed."""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "name": self.name,
            "package": self.package,
            "package_version": self.package_version,
            "model_resource": self.model_resource,
            "model_version": self.model_version,
            "jar_name": self.jar_name,
            "required_jar": self.required_jar,
            "other_jars_present": list(self.other_jars_present),
            "manifest_path": self.manifest_path,
            "manifest_revision": self.manifest_revision,
            "observed_revision": self.observed_revision,
            "revision_verified": self.revision_verified,
            "observed_tags_at_head": list(self.observed_tags_at_head),
            "expected_hashes": dict(self.expected_hashes),
            "resource_hashes": dict(self.resource_hashes),
            "hashes_verified": self.hashes_verified,
            "pinned": self.pinned,
            "notes": self.notes,
        }


# Conditions the probe compares for one clean example. FULL first so it can act
# as the reference grid.
PROBE_CONDITIONS: tuple[str, ...] = ("FULL", "P25", "P50", "P75", "P100", "STRIP_ALL")

REPO_LOCAL_HF_CACHE = ".hf-cache"
"""Colab must keep model downloads inside the checkout, as G-1 established."""
