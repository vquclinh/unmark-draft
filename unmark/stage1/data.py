"""Stage-1 data path: clean text -> three prepared branches -> batched tensors.

**Torch is imported lazily, inside `collate_stage1_batch` only.** Everything up
to tensor packing is pure Python and therefore genuinely testable in the ML-free
local environment.

Three branches, deliberately kept in separate, differently-named fields so a
wiring error is visually obvious rather than a silent swap:

* `reference_*` -- the frozen tokenizer on the **clean** text. No adapter, no
  channels, no `b(x)`.
* `base_*` -- `T(b(x))`, the authoritative adapted grid, **shared** by the two
  adapted branches.
* `clean_*` / `corrupt_*` -- the orthographic channels on that shared grid.

The reference and base sequences may have different lengths. That is expected
and is why §4.6 aligns pooled representations rather than tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from unmark.alignment import (
    OrthographicRegion,
    align_chunk,
    character_letter_labels,
    overlay_orthography,
    project_piece,
    whitespace_chunks,
)
from unmark.corruption import CorruptionCondition, CorruptionScope, corrupt
from unmark.modeling.collate import EncodedExample, build_example
from unmark.orthography import Eligibility, canon, decompose
from unmark.stage1.contracts import (
    STAGE1_SCHEMA_VERSION,
    BaseInvarianceViolation,
    CorruptionRatePolicy,
    Stage1ContractViolation,
    TruncationPolicy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor


@dataclass(frozen=True)
class Stage1Example:
    """One clean training example.

    `sample_id` must be a **stable** identity -- not row order. B2's per-unit
    decision is keyed on it, so reordering a corpus must not change any
    example's corruption (§5.3).
    """

    text: str
    sample_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise Stage1ContractViolation(
                "sample_id must be a non-empty stable identity; row order is not an "
                "identity and would make corruption depend on corpus ordering"
            )
        if not isinstance(self.text, str):
            raise Stage1ContractViolation("text must be a string")


@dataclass(frozen=True)
class PreparedStage1Example:
    """One example prepared for all three branches.

    The two adapted branches **share** `base_input_ids` and
    `base_special_tokens_mask`: the base grid is corruption-invariant, and
    `prepare_example` proves that before sharing rather than assuming it.
    """

    sample_id: str
    canonical_text: str
    corrupted_text: str
    base_text: str
    corruption_rate: float
    corruption_scope: str

    reference_input_ids: tuple[int, ...]
    reference_special_tokens_mask: tuple[int, ...]

    base_input_ids: tuple[int, ...]
    base_special_tokens_mask: tuple[int, ...]

    clean_tone_ids: tuple[int, ...]
    clean_tone_mask: tuple[bool, ...]
    clean_letter_ids: tuple[tuple[int, ...], ...]

    corrupt_tone_ids: tuple[int, ...]
    corrupt_tone_mask: tuple[bool, ...]
    corrupt_letter_ids: tuple[tuple[int, ...], ...]

    schema_version: str = STAGE1_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_length(self) -> int:
        return len(self.reference_input_ids)

    @property
    def base_length(self) -> int:
        return len(self.base_input_ids)

    @property
    def channels_differ(self) -> bool:
        """Whether corruption actually changed the observed tone channel."""
        return self.clean_tone_ids != self.corrupt_tone_ids

    @property
    def letter_channels_differ(self) -> bool:
        """Whether corruption removed letter-diacritic information.

        Under the old run-global `"TONE"` scope this was **never** true, which
        is precisely how STRIP-ALL ended up with no training support. It is a
        monitored quantity now, not an assumption.
        """
        return self.clean_letter_ids != self.corrupt_letter_ids

    def clean_encoded(self) -> EncodedExample:
        return EncodedExample(
            input_ids=list(self.base_input_ids),
            special_tokens_mask=list(self.base_special_tokens_mask),
            tone_ids=list(self.clean_tone_ids),
            tone_mask=list(self.clean_tone_mask),
            letter_ids=[list(row) for row in self.clean_letter_ids],
        )

    def corrupt_encoded(self) -> EncodedExample:
        return EncodedExample(
            input_ids=list(self.base_input_ids),
            special_tokens_mask=list(self.base_special_tokens_mask),
            tone_ids=list(self.corrupt_tone_ids),
            tone_mask=list(self.corrupt_tone_mask),
            letter_ids=[list(row) for row in self.corrupt_letter_ids],
        )


# ---------------------------------------------------------------------------
# Deterministic preparation
# ---------------------------------------------------------------------------
def _regions(base_text: str, parts) -> list[OrthographicRegion]:
    """Syllable spans plus the gaps between them, covering every character.

    Duplicated from the B3B probe scripts rather than imported: those scripts are
    frozen evidence-producing artifacts and are not edited. Recorded as an
    extraction candidate in Audit 018.
    """
    regions: list[OrthographicRegion] = []
    cursor = 0
    for span in parts.syllables:
        if span.base_start > cursor:
            regions.append(
                OrthographicRegion(
                    len(regions), base_text[cursor : span.base_start], cursor, span.base_start,
                    Eligibility.NOT_APPLICABLE, is_syllable=False,
                )
            )
        regions.append(
            OrthographicRegion(
                len(regions), span.base_text, span.base_start, span.base_end, span.eligibility
            )
        )
        cursor = span.base_end
    if cursor < len(base_text):
        regions.append(
            OrthographicRegion(
                len(regions), base_text[cursor:], cursor, len(base_text),
                Eligibility.NOT_APPLICABLE, is_syllable=False,
            )
        )
    return regions


def project_text(
    text: str,
    tokenizer: Any,
    classifier: Callable[[str], Eligibility] | None,
    unk_token_id: int | None,
):
    """Run B1A/B3A/B3B on one string. Returns `(base_text, content_ids, projections)`.

    Deterministic only: no torch, no adapter, no model.
    """
    parts = decompose(canon(text), eligibility_classifier=classifier)
    base_text = parts.base_text
    labels = character_letter_labels(parts)
    regions = _regions(base_text, parts)
    tones = {
        region.index: span.observed_tone
        for region in regions
        if region.is_syllable
        for span in parts.syllables
        if span.base_start == region.start
    }

    projections, content_ids = [], []
    for chunk in whitespace_chunks(base_text):
        tokens = tuple(tokenizer.tokenize(chunk.text))
        ids = tuple(tokenizer.convert_tokens_to_ids(list(tokens)))
        alignment = align_chunk(chunk, tokens, ids, unk_token_id=unk_token_id)
        overlays = overlay_orthography(alignment.pieces, regions)
        for piece, overlay in zip(alignment.pieces, overlays):
            projections.append(
                project_piece(len(projections), piece, overlay, base_text, labels, regions, tones)
            )
            content_ids.append(piece.token_id)
    return base_text, content_ids, projections


def _with_special_tokens(tokenizer: Any, content_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Wrap content ids in the model's own special tokens.

    Identity and order come from the tokenizer, never guessed here.
    """
    full = tokenizer.build_inputs_with_special_tokens(list(content_ids))
    mask = tokenizer.get_special_tokens_mask(list(content_ids), already_has_special_tokens=False)
    return list(full), list(mask)


def prepare_example(
    example: Stage1Example,
    tokenizer: Any,
    *,
    corruption_policy: CorruptionRatePolicy,
    truncation: TruncationPolicy,
    visit: int,
    classifier: Callable[[str], Eligibility] | None = None,
    unk_token_id: int | None = None,
) -> PreparedStage1Example | None:
    """Prepare one example for all three branches.

    The reference branch tokenizes `canon(x)`, not the raw string: corruption is
    defined on `canon(x)` (§5.3), so two inputs differing only in NFC/NFD form or
    tone placement are *the same example* and must share one reference target.
    Using the raw string would make the target depend on incoming spelling
    variation, which is the separate `VARIANT` axis (§6.3).

    `truncation` and `visit` are **required**, without defaults. Omitting either
    would silently select a scientific policy: an implicit `TruncationPolicy()`
    would choose "unbounded, fail", and an implicit `visit=0` would choose "never
    redraw" -- and both of those questions are OPEN. `TruncationPolicy.unbounded()`
    states the unbounded choice explicitly at the call site.

    Returns `None` only when the truncation policy is `SKIP` and the example
    overflows.

    Raises:
        BaseInvarianceViolation: if `b(C(x)) != b(x)`, or if the corrupted text
            does not reproduce the clean base token grid.
    """
    # Two INDEPENDENT, domain-separated draws (D-S1B-003). The scope is drawn per
    # example, never fixed for the run: a run-global "TONE" scope is exactly what
    # left STRIP-ALL with zero training support.
    rate = corruption_policy.rate_for(example.sample_id, visit)
    scope = corruption_policy.scope_for(example.sample_id, visit)
    condition = CorruptionCondition(
        name=f"stage1-{scope.lower()}-p{rate:.6f}",
        scope=CorruptionScope[scope],
        probability=rate,
        description="Stage-1 structured channel dropout, p ~ U(0,1) per example (§4.6)",
    )
    return prepare_with_condition(
        example,
        tokenizer,
        condition=condition,
        corruption_seed=corruption_policy.seed,
        truncation=truncation,
        visit=visit,
        classifier=classifier,
        unk_token_id=unk_token_id,
        extra_metadata={"corruption_pi_strip": corruption_policy.pi_strip},
    )


def prepare_with_condition(
    example: Stage1Example,
    tokenizer: Any,
    *,
    condition: CorruptionCondition,
    corruption_seed: int,
    truncation: TruncationPolicy,
    visit: int,
    classifier: Callable[[str], Eligibility] | None = None,
    unk_token_id: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> PreparedStage1Example | None:
    """Prepare one example under an **explicitly supplied** corruption condition.

    The single implementation of the three-branch preparation. `prepare_example`
    calls it with the drawn training condition; the held-out evaluator calls it
    with each fixed condition of the locked validation grid. Keeping one
    implementation is what makes "validation sees exactly the training pipeline"
    a fact rather than a claim.
    """
    canonical = canon(example.text)

    # --- PATH R: clean reference. No adapter, no channels, no b(x). ---------
    reference_content = tokenizer.convert_tokens_to_ids(list(tokenizer.tokenize(canonical)))
    reference_ids, reference_special = _with_special_tokens(tokenizer, reference_content)

    # --- PATH C: adapted clean -------------------------------------------
    clean_base, clean_content_ids, clean_projections = project_text(
        canonical, tokenizer, classifier, unk_token_id
    )

    # --- PATH K: adapted corrupted ---------------------------------------
    corrupted_text = corrupt(
        canonical, condition, seed=corruption_seed, sample_id=example.sample_id
    ).corrupted_text
    corrupt_base, corrupt_content_ids, corrupt_projections = project_text(
        corrupted_text, tokenizer, classifier, unk_token_id
    )

    # --- the load-bearing invariant, verified rather than assumed ---------
    if clean_base != corrupt_base:
        raise BaseInvarianceViolation(
            f"b(C(x)) != b(x) for sample {example.sample_id!r}: "
            f"{corrupt_base!r} vs {clean_base!r}. The deterministic phase established "
            "this equality (D-B3B2-001); it is not repaired heuristically here."
        )
    if list(clean_content_ids) != list(corrupt_content_ids):
        raise BaseInvarianceViolation(
            f"authoritative base token ids differ between the clean and corrupted "
            f"branches for sample {example.sample_id!r} despite an identical base string"
        )
    if len(clean_projections) != len(corrupt_projections):
        raise BaseInvarianceViolation(
            f"projection counts differ ({len(clean_projections)} vs "
            f"{len(corrupt_projections)}) for sample {example.sample_id!r}"
        )

    base_ids, base_special = _with_special_tokens(tokenizer, clean_content_ids)
    clean_encoded = build_example(base_ids, base_special, clean_projections)
    corrupt_encoded = build_example(base_ids, base_special, corrupt_projections)

    if not truncation.check(len(reference_ids), "reference sequence"):
        return None
    if not truncation.check(len(base_ids), "base sequence"):
        return None

    return PreparedStage1Example(
        sample_id=example.sample_id,
        canonical_text=canonical,
        corrupted_text=corrupted_text,
        base_text=clean_base,
        corruption_rate=condition.probability,
        corruption_scope=condition.scope.value,
        reference_input_ids=tuple(reference_ids),
        reference_special_tokens_mask=tuple(reference_special),
        base_input_ids=tuple(base_ids),
        base_special_tokens_mask=tuple(base_special),
        clean_tone_ids=tuple(clean_encoded.tone_ids),
        clean_tone_mask=tuple(clean_encoded.tone_mask),
        clean_letter_ids=tuple(tuple(row) for row in clean_encoded.letter_ids),
        corrupt_tone_ids=tuple(corrupt_encoded.tone_ids),
        corrupt_tone_mask=tuple(corrupt_encoded.tone_mask),
        corrupt_letter_ids=tuple(tuple(row) for row in corrupt_encoded.letter_ids),
        metadata={
            "corruption_condition": condition.name,
            "corruption_scope": condition.scope.value,
            "corruption_seed": corruption_seed,
            "visit": visit,
            **(extra_metadata or {}),
        },
    )


# ---------------------------------------------------------------------------
# Collation -- two independent padding domains
# ---------------------------------------------------------------------------
def padded_stage1_batch(
    examples: Sequence[PreparedStage1Example], pad_token_id: int
) -> dict[str, Any]:
    """Right-pad a batch as nested Python lists. Torch-free.

    **Reference and base get separate padding domains.** Their lengths differ in
    general, and padding both to a common maximum would inflate one branch with
    positions that exist only to match the other -- pooling excludes padding, so
    it would be waste at best and a masking bug at worst.
    """
    if not examples:
        raise Stage1ContractViolation("cannot collate an empty batch")

    reference_width = max(e.reference_length for e in examples)
    base_width = max(e.base_length for e in examples)
    depth = max(
        (len(row) for e in examples for row in (e.clean_letter_ids + e.corrupt_letter_ids)),
        default=0,
    )
    depth = max(depth, 1)

    from unmark.modeling.collate import padded_batch

    reference_ids, reference_attention, reference_special = [], [], []
    for example in examples:
        pad = reference_width - example.reference_length
        reference_ids.append(list(example.reference_input_ids) + [pad_token_id] * pad)
        reference_attention.append([1] * example.reference_length + [0] * pad)
        reference_special.append(list(example.reference_special_tokens_mask) + [1] * pad)

    clean = padded_batch([e.clean_encoded() for e in examples], pad_token_id=pad_token_id)
    corrupted = padded_batch([e.corrupt_encoded() for e in examples], pad_token_id=pad_token_id)

    if clean["input_ids"] != corrupted["input_ids"]:
        raise BaseInvarianceViolation(
            "collated base ids differ between the clean and corrupted branches"
        )
    if clean["special_tokens_mask"] != corrupted["special_tokens_mask"]:
        raise BaseInvarianceViolation(
            "collated base special-token masks differ between the branches"
        )

    def pad_letters(rows: list[list[list[int]]]) -> list[list[list[int]]]:
        return [[row + [-1] * (depth - len(row)) for row in ex] for ex in rows]

    def pad_letter_masks(rows: list[list[list[bool]]]) -> list[list[list[bool]]]:
        return [[row + [False] * (depth - len(row)) for row in ex] for ex in rows]

    return {
        "reference_input_ids": reference_ids,
        "reference_attention_mask": reference_attention,
        "reference_special_tokens_mask": reference_special,
        "base_input_ids": clean["input_ids"],
        "base_attention_mask": clean["attention_mask"],
        "base_special_tokens_mask": clean["special_tokens_mask"],
        "clean_tone_ids": clean["tone_ids"],
        "clean_tone_mask": clean["tone_mask"],
        "clean_letter_ids": pad_letters(clean["letter_ids"]),
        "clean_letter_mask": pad_letter_masks(clean["letter_mask"]),
        "corrupt_tone_ids": corrupted["tone_ids"],
        "corrupt_tone_mask": corrupted["tone_mask"],
        "corrupt_letter_ids": pad_letters(corrupted["letter_ids"]),
        "corrupt_letter_mask": pad_letter_masks(corrupted["letter_mask"]),
        "sample_ids": [e.sample_id for e in examples],
        "corruption_rates": [e.corruption_rate for e in examples],
        "corruption_scopes": [e.corruption_scope for e in examples],
    }


_BOOL_FIELDS = frozenset(
    {"clean_tone_mask", "clean_letter_mask", "corrupt_tone_mask", "corrupt_letter_mask"}
)
_NON_TENSOR_FIELDS = frozenset({"sample_ids", "corruption_rates", "corruption_scopes"})


def collate_stage1_batch(
    examples: Sequence[PreparedStage1Example], pad_token_id: int
) -> dict[str, Any]:
    """`padded_stage1_batch`, wrapped in tensors. **Imports torch lazily.**"""
    import torch

    rows = padded_stage1_batch(examples, pad_token_id)
    out: dict[str, Any] = {}
    for key, value in rows.items():
        if key in _NON_TENSOR_FIELDS:
            out[key] = value
        else:
            out[key] = torch.tensor(
                value, dtype=torch.bool if key in _BOOL_FIELDS else torch.long
            )
    return out
