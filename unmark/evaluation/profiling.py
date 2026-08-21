"""Data-only dataset profiler for the pre-G1 burden diagnostic.

**No torch. No network. No model.** Everything here operates on strings and the
repository's authoritative `canon` / base-text implementations -- it never
reimplements stripping rules and never runs a restorer.

What it answers: is the selected dataset clean enough, duplicate-free enough, and
short enough to support a *descriptive* Vanilla-vs-Base-only measurement? It
answers nothing about downstream performance.

**A vocabulary rule this module enforces on itself.** A text with no observed
tone mark is **base-equivalent**, not "missing diacritics". Unmarked Vietnamese
is observationally ambiguous -- §4.3: genuine *ngang* and a stripped mark are
indistinguishable at inference. Every field name here says what was *observed*.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.orthography import LetterDiacritic, ObservedTone, canon, decompose

PROFILE_SCHEMA_VERSION = "preg1-profile-v1"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
class DatasetAccess(Enum):
    """How a copy of the dataset was obtained.

    Replaces an earlier boolean `authorisation_established`, which encoded an
    SA-VLSP-specific assumption: that a scientific run requires a signed user
    agreement. That is true for some corpora and false for others, and the
    boolean wrongly classified an **officially and publicly distributed** corpus
    as unusable simply because no agreement exists.
    """

    OFFICIAL_PUBLIC_DISTRIBUTION = "OFFICIAL_PUBLIC_DISTRIBUTION"
    """Obtained from the official distributor's public download, with no user
    agreement required. UIT-VSFC v1.0 is distributed this way."""

    OFFICIAL_AGREEMENT_AUTHORISED = "OFFICIAL_AGREEMENT_AUTHORISED"
    """Obtained from the official distributor under a signed user agreement."""

    MIRROR = "MIRROR"
    """A third-party mirror. Usable for profiling; **must not** be reported as
    the official distribution."""

    UNKNOWN = "UNKNOWN"
    """Provenance not established. Never usable for a scientific run."""

    @property
    def is_official(self) -> bool:
        return self in (
            DatasetAccess.OFFICIAL_PUBLIC_DISTRIBUTION,
            DatasetAccess.OFFICIAL_AGREEMENT_AUTHORISED,
        )


LICENSE_NOT_ESTABLISHED = "NOT_ESTABLISHED"
"""No explicit license metadata was found from an authoritative source.

**Official public distribution and an identified license are different facts**
and are recorded separately. A dataset can be officially and publicly
downloadable while carrying no machine-readable license statement, and this
module makes no legal claim beyond the evidence it was given.
"""


@dataclass(frozen=True)
class FileProvenance:
    """One dataset file: its identity and exact bytes."""

    path: str
    sha256: str
    size_bytes: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class DatasetProvenance:
    """Everything needed to identify the exact data a profile describes.

    `access` is required and has no default: a run that cannot say how it got
    the data should not imply anything about it.
    """

    dataset_name: str
    dataset_version: str
    task: str
    access: DatasetAccess
    source_name: str
    label_mapping: dict[str, int]
    columns: tuple[str, ...]
    source_revision: str | None = None
    source_url: str | None = None
    files: tuple[FileProvenance, ...] = ()
    license_status: str = LICENSE_NOT_ESTABLISHED
    notes: str = ""
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.access, DatasetAccess):
            raise EvaluationContractViolation(
                "access must be a DatasetAccess; it has no default"
            )
        if not self.label_mapping:
            raise EvaluationContractViolation("label_mapping must not be empty")

    @property
    def is_official(self) -> bool:
        return self.access.is_official

    @property
    def usable_for_scientific_run(self) -> bool:
        """Any **official** distribution qualifies -- public or agreement-based.

        A mirror does not, and neither does UNKNOWN provenance. License status
        is deliberately **not** part of this test: it is a separate fact, and
        conflating the two would be a legal claim beyond the evidence.
        """
        return self.is_official

    @property
    def license_established(self) -> bool:
        return self.license_status != LICENSE_NOT_ESTABLISHED

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "access": self.access.value,
            "is_official": self.is_official,
            "usable_for_scientific_run": self.usable_for_scientific_run,
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "source_url": self.source_url,
            "license_status": self.license_status,
            "license_established": self.license_established,
            "label_mapping": dict(self.label_mapping),
            "columns": list(self.columns),
            "files": [f.to_dict() for f in self.files],
            "notes": self.notes,
            "schema_version": self.schema_version,
        }


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def text_digest(text: str) -> str:
    """Stable identity of a text, for duplicate grouping without storing it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
PERCENTILES: tuple[int, ...] = (25, 50, 75, 90, 95, 99)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on a sorted copy. Deterministic, no numpy."""
    if not values:
        raise EvaluationContractViolation("cannot take a percentile of no values")
    ordered = sorted(values)
    if q <= 0:
        return float(ordered[0])
    if q >= 100:
        return float(ordered[-1])
    rank = max(1, min(len(ordered), int(-(-q / 100 * len(ordered) // 1))))
    return float(ordered[rank - 1])


def distribution(values: Sequence[float]) -> dict[str, float]:
    """min / percentiles / max / mean, as a plain dict."""
    if not values:
        return {"count": 0}
    out: dict[str, float] = {"count": len(values), "min": float(min(values))}
    for q in PERCENTILES:
        out[f"p{q}"] = percentile(values, q)
    out["max"] = float(max(values))
    out["mean"] = sum(values) / len(values)
    return out


# ---------------------------------------------------------------------------
# Text-noise descriptives
# ---------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_REPEATED_RUN = re.compile(r"(.)\1{2,}")
_DIGIT_HEAVY = re.compile(r"\b\w*\d\w*\b")


def _is_emoji_or_symbol(ch: str) -> bool:
    return ord(ch) > 0xFFFF or unicodedata.category(ch) in {"So", "Sk"}


def noise_descriptives(text: str) -> dict[str, int]:
    """Observable surface artifacts. **Descriptive only.**

    Nothing here is normalised away, and no "teencode corrector" is implied.
    These are counts of things a reader can point at, not judgements about
    whether the text is correct.
    """
    return {
        "urls": len(_URL.findall(text)),
        "mentions": len(_MENTION.findall(text)),
        "hashtags": len(_HASHTAG.findall(text)),
        "emoji_or_symbols": sum(1 for c in text if _is_emoji_or_symbol(c)),
        "repeated_char_runs": len(_REPEATED_RUN.findall(text)),
        "digit_bearing_tokens": len(_DIGIT_HEAVY.findall(text)),
    }


# ---------------------------------------------------------------------------
# Orthographic observables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrthographyObservation:
    """What is *observable* in one text. No inference about what was lost."""

    canonical: str
    base: str
    canon_changed: bool
    """`canon(x) != x` -- an NFC/placement normalisation happened."""
    base_equivalent: bool
    """`canon(x) == b(canon(x))`: **no observed mark at all**.

    This is **not** "missing diacritics". An unmarked Vietnamese syllable is
    observationally ambiguous (§4.3), and a text may be legitimately mark-free.
    """
    changed_characters: int
    changed_units: int
    """Character units whose observed orthography differs from the base."""
    units_with_observed_tone: int
    units_with_observed_letter: int
    total_units: int
    syllables: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "canon_changed": self.canon_changed,
            "base_equivalent": self.base_equivalent,
            "changed_characters": self.changed_characters,
            "changed_units": self.changed_units,
            "units_with_observed_tone": self.units_with_observed_tone,
            "units_with_observed_letter": self.units_with_observed_letter,
            "total_units": self.total_units,
            "syllables": self.syllables,
        }


def observe_orthography(text: str) -> OrthographyObservation:
    """Profile one text with the **authoritative** `canon` and base decomposition.

    Deliberately delegates: stripping rules live in `unmark.orthography` and are
    not reimplemented here, so a profile can never disagree with the pipeline it
    is profiling.
    """
    canonical = canon(text)
    parts = decompose(canonical)
    base = parts.base_text

    changed_characters = sum(
        1 for a, b in zip(canonical, base) if a != b
    ) + abs(len(canonical) - len(base))

    tone_units = sum(
        1 for u in parts.units if u.observed_tone is not ObservedTone.UNMARKED
    )
    letter_units = sum(
        1
        for u in parts.units
        if u.letter_diacritic not in (LetterDiacritic.NONE, LetterDiacritic.NA)
    )
    return OrthographyObservation(
        canonical=canonical,
        base=base,
        canon_changed=canonical != text,
        base_equivalent=canonical == base,
        changed_characters=changed_characters,
        changed_units=tone_units + letter_units,
        units_with_observed_tone=tone_units,
        units_with_observed_letter=letter_units,
        total_units=len(parts.units),
        syllables=len(parts.syllables),
    )


# ---------------------------------------------------------------------------
# Split profile
# ---------------------------------------------------------------------------
@dataclass
class SplitProfile:
    """The profile of one split. Counts and hashes only -- no raw text."""

    split_name: str
    examples: int
    labels: dict[str, int] = field(default_factory=dict)
    empty_or_invalid: int = 0
    canon_changed: int = 0
    exact_duplicate_texts: int = 0
    canonical_duplicate_texts: int = 0
    conflicting_label_groups: int = 0
    base_equivalent: int = 0
    with_observed_tone: int = 0
    with_observed_letter: int = 0
    changed_unit_distribution: dict[str, float] = field(default_factory=dict)
    character_length_distribution: dict[str, float] = field(default_factory=dict)
    noise: dict[str, int] = field(default_factory=dict)

    @property
    def label_proportions(self) -> dict[str, float]:
        total = sum(self.labels.values())
        return {k: v / total for k, v in self.labels.items()} if total else {}

    @property
    def base_equivalent_rate(self) -> float:
        """Fraction with **no observed mark**. Not a missing-diacritic rate."""
        return self.base_equivalent / self.examples if self.examples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "examples": self.examples,
            "labels": dict(self.labels),
            "label_proportions": self.label_proportions,
            "empty_or_invalid": self.empty_or_invalid,
            "canon_changed": self.canon_changed,
            "exact_duplicate_texts": self.exact_duplicate_texts,
            "canonical_duplicate_texts": self.canonical_duplicate_texts,
            "conflicting_label_groups": self.conflicting_label_groups,
            "base_equivalent": self.base_equivalent,
            "base_equivalent_rate": self.base_equivalent_rate,
            "with_observed_tone": self.with_observed_tone,
            "with_observed_letter": self.with_observed_letter,
            "changed_unit_distribution": self.changed_unit_distribution,
            "character_length_distribution": self.character_length_distribution,
            "noise": dict(self.noise),
        }


def profile_split(
    split_name: str, records: Iterable[tuple[str, str, Any]]
) -> tuple[SplitProfile, dict[str, list[tuple[str, Any]]]]:
    """Profile one split.

    Args:
        records: `(sample_id, text, label)` triples.

    Returns:
        The profile, and a canonical-digest -> `[(sample_id, label), …]` index
        used for duplicate and cross-split analysis. The index stores **hashes,
        not text**.
    """
    records = list(records)
    profile = SplitProfile(split_name=split_name, examples=len(records))
    exact_seen: Counter[str] = Counter()
    canonical_index: dict[str, list[tuple[str, Any]]] = {}
    changed_units: list[float] = []
    lengths: list[float] = []
    noise_total: Counter[str] = Counter()

    for sample_id, text, label in records:
        profile.labels[str(label)] = profile.labels.get(str(label), 0) + 1
        if not isinstance(text, str) or not text.strip():
            profile.empty_or_invalid += 1
            continue

        observed = observe_orthography(text)
        if observed.canon_changed:
            profile.canon_changed += 1
        if observed.base_equivalent:
            profile.base_equivalent += 1
        if observed.units_with_observed_tone:
            profile.with_observed_tone += 1
        if observed.units_with_observed_letter:
            profile.with_observed_letter += 1

        changed_units.append(observed.changed_units)
        lengths.append(len(observed.canonical))
        exact_seen[text_digest(text)] += 1
        canonical_index.setdefault(text_digest(observed.canonical), []).append(
            (sample_id, label)
        )
        for key, value in noise_descriptives(text).items():
            noise_total[key] += value

    profile.exact_duplicate_texts = sum(c - 1 for c in exact_seen.values() if c > 1)
    profile.canonical_duplicate_texts = sum(
        len(v) - 1 for v in canonical_index.values() if len(v) > 1
    )
    profile.conflicting_label_groups = sum(
        1 for v in canonical_index.values() if len({str(l) for _, l in v}) > 1
    )
    profile.changed_unit_distribution = distribution(changed_units)
    profile.character_length_distribution = distribution(lengths)
    profile.noise = dict(noise_total)
    return profile, canonical_index


# ---------------------------------------------------------------------------
# Duplicates and leakage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DuplicateReport:
    """Duplicate and cross-split findings. Hashes and ids only."""

    canonical_duplicate_groups: int
    conflicting_label_groups: tuple[dict[str, Any], ...]
    cross_split_groups: tuple[dict[str, Any], ...]

    @property
    def has_conflicting_labels(self) -> bool:
        return bool(self.conflicting_label_groups)

    @property
    def has_cross_split_leakage(self) -> bool:
        return bool(self.cross_split_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_duplicate_groups": self.canonical_duplicate_groups,
            "conflicting_label_group_count": len(self.conflicting_label_groups),
            "conflicting_label_groups": list(self.conflicting_label_groups),
            "cross_split_group_count": len(self.cross_split_groups),
            "cross_split_groups": list(self.cross_split_groups),
        }


def analyse_duplicates(
    indexes: dict[str, dict[str, list[tuple[str, Any]]]]
) -> DuplicateReport:
    """Cross-split duplicate analysis over canonical-digest indexes.

    A canonical duplicate that carries **different labels** is not a
    de-duplication problem, it is a labelling problem -- reported, never
    silently dropped or relabelled. The handling decision is the researcher's.
    """
    merged: dict[str, list[tuple[str, str, Any]]] = {}
    for split_name, index in indexes.items():
        for digest, entries in index.items():
            for sample_id, label in entries:
                merged.setdefault(digest, []).append((split_name, sample_id, label))

    conflicting, cross_split, groups = [], [], 0
    for digest, entries in merged.items():
        if len(entries) < 2:
            continue
        groups += 1
        labels = {str(label) for _, _, label in entries}
        splits = {split for split, _, _ in entries}
        record = {
            "canonical_digest": digest,
            "size": len(entries),
            "splits": sorted(splits),
            "labels": sorted(labels),
            "sample_ids": sorted(sample_id for _, sample_id, _ in entries),
        }
        if len(labels) > 1:
            conflicting.append(record)
        if len(splits) > 1:
            cross_split.append(record)
    return DuplicateReport(
        canonical_duplicate_groups=groups,
        conflicting_label_groups=tuple(conflicting),
        cross_split_groups=tuple(cross_split),
    )


# ---------------------------------------------------------------------------
# max_length selection rule
# ---------------------------------------------------------------------------
LENGTH_REPORT_THRESHOLDS: tuple[int, ...] = (64, 128, 256)
"""Coverage thresholds the profiler reports. **Descriptive only.**

An earlier revision used these as a *selection* rule -- the smallest candidate
covering 99% of train on both pathways. That rule is **superseded**:
`max_length` is now fixed at 256 (D-PREG1-008b), so these numbers characterise
the corpus rather than choosing a protocol value.
"""

FIXED_MAX_LENGTH = 256
"""The locked `max_length` for this pre-G1 diagnostic.

Not derived from the data. PhoBERT's pretrained positional capacity is 256 for a
task sequence, and the diagnostic wants to **minimise truncation** rather than
optimise inference cost -- compute is not a constraint here. Fixing it removes
an otherwise data-dependent protocol decision from the measurement.
"""


@dataclass(frozen=True)
class LengthCoverage:
    """Coverage at one threshold, on both pathways. Reported, never selecting."""

    threshold: int
    vanilla_coverage: float
    base_only_coverage: float

    @property
    def joint_coverage(self) -> float:
        return min(self.vanilla_coverage, self.base_only_coverage)

    @property
    def vanilla_overflow(self) -> float:
        return 1.0 - self.vanilla_coverage

    @property
    def base_only_overflow(self) -> float:
        return 1.0 - self.base_only_coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "vanilla_coverage": self.vanilla_coverage,
            "base_only_coverage": self.base_only_coverage,
            "joint_coverage": self.joint_coverage,
            "vanilla_overflow_rate": self.vanilla_overflow,
            "base_only_overflow_rate": self.base_only_overflow,
        }


def length_coverage(
    vanilla_lengths: Sequence[int],
    base_only_lengths: Sequence[int],
    thresholds: Sequence[int] = LENGTH_REPORT_THRESHOLDS,
) -> tuple[LengthCoverage, ...]:
    """Coverage and overflow at each threshold, for both pathways.

    Purely descriptive: the profiler must still report how much of the corpus
    fits, including the overflow rate at the fixed 256, so truncation is a known
    quantity rather than an invisible one.
    """
    if not vanilla_lengths or not base_only_lengths:
        raise EvaluationContractViolation(
            "both pathway length samples are required for a coverage report"
        )
    return tuple(
        LengthCoverage(
            threshold=threshold,
            vanilla_coverage=sum(1 for n in vanilla_lengths if n <= threshold)
            / len(vanilla_lengths),
            base_only_coverage=sum(1 for n in base_only_lengths if n <= threshold)
            / len(base_only_lengths),
        )
        for threshold in sorted(thresholds)
    )


# ---------------------------------------------------------------------------
# Deterministic, group-aware, stratified splitter
# ---------------------------------------------------------------------------
def stratified_group_split(
    records: Sequence[tuple[str, str, Any]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Split `(sample_id, text, label)` into named parts. Returns sample ids.

    Four properties, all load-bearing:

    * **group-aware by canonical text** -- every canonical duplicate lands in one
      part, so a duplicate cannot straddle protocol-train and protocol-dev;
    * **label-stratified** as closely as grouping allows, using each group's
      majority label;
    * **deterministic** -- ordering comes from a keyed digest of the seed and the
      canonical digest, not from `random`, so it is stable across processes and
      reruns;
    * **independent of any downstream score.**

    The generic mechanism is implemented here; it is **not run on real data** in
    this phase, because conflicting-label groups must be inspected first.
    """
    if not records:
        raise EvaluationContractViolation("cannot split an empty record set")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise EvaluationContractViolation(f"fractions must sum to 1.0, got {fractions}")

    groups: dict[str, list[tuple[str, Any]]] = {}
    for sample_id, text, label in records:
        groups.setdefault(text_digest(canon(text)), []).append((sample_id, label))

    def group_label(entries: list[tuple[str, Any]]) -> str:
        return Counter(str(label) for _, label in entries).most_common(1)[0][0]

    def order_key(digest: str) -> str:
        return hashlib.blake2b(
            f"{seed}|{digest}".encode("utf-8"), digest_size=16
        ).hexdigest()

    by_label: dict[str, list[str]] = {}
    for digest, entries in groups.items():
        by_label.setdefault(group_label(entries), []).append(digest)

    names = list(fractions)
    assignment: dict[str, list[str]] = {name: [] for name in names}
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=order_key)
        total = len(ordered)
        start = 0
        for index, name in enumerate(names):
            take = total - start if index == len(names) - 1 else round(total * fractions[name])
            for digest in ordered[start : start + take]:
                assignment[name].extend(sample_id for sample_id, _ in groups[digest])
            start += take
    return {name: sorted(ids) for name, ids in assignment.items()}


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------
SEED_DERIVATION_RULE = (
    "SHA-256 of the ASCII tag, read as successive 2-byte big-endian words: "
    "seed_i = int.from_bytes(sha256(tag)[2i:2i+2], 'big'). Fully determined by "
    "the tag, so the seeds cannot have been selected after seeing any result."
)


def derive_seeds(tag: str, count: int) -> tuple[int, ...]:
    """Deterministic seeds from a precommitted tag.

    The point is falsifiability: anyone can recompute these from the tag string
    alone, so a reader can verify they were not chosen to flatter a result.
    """
    if count <= 0:
        raise EvaluationContractViolation("count must be positive")
    digest = hashlib.sha256(tag.encode("ascii")).digest()
    if count * 2 > len(digest):
        raise EvaluationContractViolation(
            f"tag yields at most {len(digest) // 2} seeds, {count} requested"
        )
    return tuple(
        int.from_bytes(digest[i * 2 : i * 2 + 2], "big") for i in range(count)
    )
