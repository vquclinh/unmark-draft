"""B3B-1C: deterministic orthographic channel projection.

Local-only. No torch, no transformers, no network, no model weights. Token
sequences are the *real* pieces recorded by the corrected B3B-1B probe run
(20260820T035339Z) or minimal constructions in the same fastBPE format.

28 categories, in order.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from unmark.alignment import (
    LETTER_POOLING_RULE,
    CharacterContribution,
    LetterProjection,
    OrthographicRegion,
    TokenOrthographyProjection,
    TokenToneLabel,
    ToneOwnership,
    ToneProjection,
    align_chunk,
    character_letter_labels,
    overlay_orthography,
    project_piece,
    project_special_token,
    summarize_projections,
    whitespace_chunks,
)
from unmark.corruption import CorruptionPurpose, corrupt
from unmark.orthography import (
    Eligibility,
    LetterDiacritic,
    ObservedTone,
    Tone,
    decompose,
)

VN = Eligibility.VIETNAMESE_CANDIDATE
NA = Eligibility.NOT_APPLICABLE
CONDITIONS = ["FULL", "P25", "P50", "P75", "P100", "STRIP_ALL"]


def regions_from(text: str, *, eligible: set[int] | None = None) -> list[OrthographicRegion]:
    """Build regions from a real decomposition: syllables are candidates, the
    gaps between them are not."""
    parts = decompose(text)
    regions: list[OrthographicRegion] = []
    cursor = 0
    for order, span in enumerate(parts.syllables):
        if span.base_start > cursor:
            regions.append(
                OrthographicRegion(
                    len(regions),
                    parts.base_text[cursor : span.base_start],
                    cursor,
                    span.base_start,
                    NA,
                    is_syllable=False,
                )
            )
        decided = VN if (eligible is None or order in eligible) else NA
        regions.append(
            OrthographicRegion(
                len(regions), span.base_text, span.base_start, span.base_end, decided
            )
        )
        cursor = span.base_end
    if cursor < len(parts.base_text):
        regions.append(
            OrthographicRegion(
                len(regions), parts.base_text[cursor:], cursor, len(parts.base_text), NA,
                is_syllable=False,
            )
        )
    return regions


def region_tones(text: str, regions) -> dict[int, ObservedTone]:
    parts = decompose(text)
    by_start = {s.base_start: s.observed_tone for s in parts.syllables}
    return {r.index: by_start[r.start] for r in regions if r.start in by_start}


def project_all(text: str, tokens, ids, *, eligible=None):
    """Project a single-chunk text end to end using the real pipeline."""
    parts = decompose(text)
    labels = character_letter_labels(parts)
    regions = regions_from(text, eligible=eligible)
    tones = region_tones(text, regions)
    chunks = whitespace_chunks(parts.base_text)
    assert len(chunks) == 1, "helper handles one chunk"
    alignment = align_chunk(chunks[0], list(tokens), list(ids))
    overlays = overlay_orthography(alignment.pieces, regions)
    return [
        project_piece(i, p, o, parts.base_text, labels, regions, tones)
        for i, (p, o) in enumerate(zip(alignment.pieces, overlays))
    ]


# ---------------------------------------------------------------------------
# 1. Character labels are read from the canonical decomposition
# ---------------------------------------------------------------------------
def test_character_labels_are_indexed_by_base_text():
    parts = decompose("Tôi học")
    labels = character_letter_labels(parts)
    assert len(labels) == len(parts.base_text)
    assert parts.base_text == "Toi hoc"
    assert labels[1] is LetterDiacritic.CIRCUMFLEX  # ô -> o + circumflex
    assert labels[3] is LetterDiacritic.NA  # the space


def test_labels_come_from_units_not_from_a_token_string():
    """The label of a character never depends on how BPE happened to cut it."""
    parts = decompose("đường")
    labels = character_letter_labels(parts)
    whole = project_all("đường", ["duong"], [1])
    split = project_all("đường", ["du@@", "ong"], [1, 2])
    assert [c.letter_diacritic for c in whole[0].letter.contributions] == list(labels)
    assert (
        [c.letter_diacritic for p in split for c in p.letter.contributions] == list(labels)
    )


# ---------------------------------------------------------------------------
# 2. Units whose base form spans several characters
# ---------------------------------------------------------------------------
def test_every_character_of_a_multi_character_unit_is_labelled():
    parts = decompose("naïve")
    labels = character_letter_labels(parts)
    assert len(labels) == len(parts.base_text)
    assert all(isinstance(label, LetterDiacritic) for label in labels)


# ---------------------------------------------------------------------------
# 3. NONE and NA are different states
# ---------------------------------------------------------------------------
def test_none_and_na_are_not_interchangeable():
    parts = decompose("hoc.")
    labels = character_letter_labels(parts)
    assert labels[0] is LetterDiacritic.NONE  # a letter with no letter diacritic
    assert labels[3] is LetterDiacritic.NA  # punctuation: the channel is undefined
    assert LetterDiacritic.NONE is not LetterDiacritic.NA


# ---------------------------------------------------------------------------
# 4. TokenToneLabel covers every observed tone
# ---------------------------------------------------------------------------
def test_token_tone_label_projects_every_observed_tone():
    for tone in ObservedTone:
        assert TokenToneLabel.from_observed_tone(tone).value == tone.value
    assert len(TokenToneLabel) == len(ObservedTone) + 1  # the extra state is NA


# ---------------------------------------------------------------------------
# 5. The deploy pathway never sees lexical NGANG
# ---------------------------------------------------------------------------
def test_no_lexical_ngang_in_the_token_channel():
    assert "NGANG" not in {label.name for label in TokenToneLabel}
    assert Tone.NGANG.name not in {label.name for label in TokenToneLabel}
    unmarked = project_all("hoc", ["hoc"], [1])[0]
    assert unmarked.tone.label is TokenToneLabel.UNMARKED
    assert unmarked.tone.label is not TokenToneLabel.NA


def test_unmarked_is_distinct_from_not_applicable():
    """`UNMARKED` says "a Vietnamese syllable with no readable mark".
    `NA` says "no Vietnamese syllable here at all". Collapsing them would tell
    the adapter a comma is a ngang-looking syllable."""
    vietnamese = project_all("hoc", ["hoc"], [1])[0]
    punctuation = project_all("...", ["..."], [1])[0]
    assert vietnamese.tone.label is TokenToneLabel.UNMARKED
    assert punctuation.tone.label is TokenToneLabel.NA


# ---------------------------------------------------------------------------
# 6. A single candidate owns the tone
# ---------------------------------------------------------------------------
def test_single_candidate_propagates_its_tone():
    projection = project_all("học", ["hoc"], [1])[0]
    assert projection.tone.ownership is ToneOwnership.SINGLE_CANDIDATE
    assert projection.tone.label is TokenToneLabel.NANG
    assert projection.tone.source_region_index == 0
    assert projection.tone.carries_tone


def test_every_piece_of_a_split_syllable_carries_the_same_tone():
    """The tone belongs to the syllable, not to the piece that happens to hold
    the marked vowel."""
    pieces = project_all("nghiên", ["nghi@@", "en"], [1, 2])
    assert [p.tone.label for p in pieces] == [TokenToneLabel.UNMARKED] * 2
    marked = project_all("cứu", ["c@@", "uu"], [1, 2])
    assert [p.tone.label for p in marked] == [TokenToneLabel.SAC] * 2


# ---------------------------------------------------------------------------
# 7. One candidate plus punctuation still owns the tone (the real probe case)
# ---------------------------------------------------------------------------
def test_candidate_mixed_with_punctuation_keeps_the_tone():
    """`en-` in the probe: "en" from candidate `tuyen`, plus a hyphen.

    Punctuation does not compete for the tone label, so there is nothing
    ambiguous to protect against.
    """
    text = "tuyến-hoc"
    parts = decompose(text)
    labels = character_letter_labels(parts)
    regions = regions_from(text)
    tones = region_tones(text, regions)
    alignment = align_chunk(whitespace_chunks(parts.base_text)[0], ["tuy@@", "en-@@", "hoc"], [1, 2, 3])
    overlays = overlay_orthography(alignment.pieces, regions)
    pieces = [
        project_piece(i, p, o, parts.base_text, labels, regions, tones)
        for i, (p, o) in enumerate(zip(alignment.pieces, overlays))
    ]
    assert pieces[1].token == "en-@@"
    assert pieces[1].tone.ownership is ToneOwnership.SINGLE_CANDIDATE
    assert pieces[1].tone.label is TokenToneLabel.SAC
    assert pieces[1].tone.candidate_region_indices == (0,)


def test_trailing_period_does_not_suppress_the_tone():
    pieces = project_all("phố.", ["pho."], [1])
    assert pieces[0].tone.ownership is ToneOwnership.SINGLE_CANDIDATE
    assert pieces[0].tone.label is TokenToneLabel.SAC


# ---------------------------------------------------------------------------
# 8. Zero candidates
# ---------------------------------------------------------------------------
def test_no_candidate_means_not_applicable():
    projection = project_all("2026", ["2026"], [1])[0]
    assert projection.tone.ownership is ToneOwnership.NOT_APPLICABLE
    assert projection.tone.label is TokenToneLabel.NA
    assert projection.tone.candidate_region_indices == ()


def test_ineligible_syllable_is_not_a_candidate():
    projection = project_all("hoc", ["hoc"], [1], eligible=set())[0]
    assert projection.tone.ownership is ToneOwnership.NOT_APPLICABLE
    assert projection.tone.label is TokenToneLabel.NA


# ---------------------------------------------------------------------------
# 9. Two or more candidates: ambiguous, never resolved
# ---------------------------------------------------------------------------
def test_two_candidates_yield_na_and_record_every_contributor():
    pieces = project_all("học-cứu", ["hoc-cuu"], [1])
    tone = pieces[0].tone
    assert tone.ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS
    assert tone.label is TokenToneLabel.NA
    assert tone.source_region_index is None
    assert tone.candidate_region_indices == (0, 2)
    assert tone.is_ambiguous and not tone.carries_tone


def test_ambiguity_is_not_resolved_by_majority_length():
    """`nghiên` is six characters and `hạ` is two. Length must not decide."""
    pieces = project_all("nghiên-hạ", ["nghien-ha"], [1])
    assert pieces[0].tone.label is TokenToneLabel.NA
    reversed_order = project_all("hạ-nghiên", ["ha-nghien"], [1])
    assert reversed_order[0].tone.label is TokenToneLabel.NA


def test_ambiguity_is_not_resolved_by_first_or_last():
    pieces = project_all("học-cứu", ["hoc-cuu"], [1])
    first_tone = region_tones("học-cứu", regions_from("học-cứu"))
    assert set(first_tone.values()) == {ObservedTone.NANG, ObservedTone.SAC}
    assert pieces[0].tone.label not in {TokenToneLabel.NANG, TokenToneLabel.SAC}


def test_agreeing_candidates_are_still_ambiguous():
    """Two candidates carrying the same tone share a value, not a source."""
    pieces = project_all("học-học", ["hoc-hoc"], [1])
    assert pieces[0].tone.ownership is ToneOwnership.MULTI_CANDIDATE_AMBIGUOUS
    assert pieces[0].tone.label is TokenToneLabel.NA


def test_categorical_tone_ids_are_never_averaged():
    """No arithmetic on tone labels: the projected label is always a member of
    the enum, and always either NA or one contributor's actual tone."""
    pieces = project_all("cứu-hạ-học", ["cuu-ha-hoc"], [1])
    assert isinstance(pieces[0].tone.label, TokenToneLabel)
    assert pieces[0].tone.label is TokenToneLabel.NA


# ---------------------------------------------------------------------------
# 10. Unresolved eligibility
# ---------------------------------------------------------------------------
def test_undecided_eligibility_yields_no_tone():
    parts = decompose("toi")
    regions = [OrthographicRegion(0, "toi", 0, 3, Eligibility.UNDECIDED)]
    alignment = align_chunk(whitespace_chunks("toi")[0], ["toi"], [1])
    overlay = overlay_orthography(alignment.pieces, regions)[0]
    projection = project_piece(
        0, alignment.pieces[0], overlay, "toi", character_letter_labels(parts), regions, {}
    )
    assert projection.tone.ownership is ToneOwnership.UNRESOLVED
    assert projection.tone.label is TokenToneLabel.NA


# ---------------------------------------------------------------------------
# 11. Special tokens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token", ["<s>", "</s>", "<pad>", "<mask>", "<unk>"])
def test_special_tokens_carry_no_orthography(token):
    projection = project_special_token(0, token, 1)
    assert projection.is_special
    assert projection.tone.label is TokenToneLabel.NA
    assert projection.letter.token_label is LetterDiacritic.NA
    assert projection.letter.contributions == ()


def test_special_tokens_never_fabricate_a_source_range():
    projection = project_special_token(3, "<s>", 0)
    assert projection.global_start is None and projection.global_end is None
    assert not projection.has_source_range


# ---------------------------------------------------------------------------
# 12. Whitespace never becomes a token
# ---------------------------------------------------------------------------
def test_whitespace_produces_no_model_token():
    text = "hoc tap"
    parts = decompose(text)
    chunks = whitespace_chunks(parts.base_text)
    assert [c.text for c in chunks] == ["hoc", "tap"]
    covered = {i for c in chunks for i in range(c.start, c.end)}
    assert parts.base_text.index(" ") not in covered


def test_no_projection_contributes_a_whitespace_character():
    text = "học tập"
    parts = decompose(text)
    labels = character_letter_labels(parts)
    regions = regions_from(text)
    tones = region_tones(text, regions)
    projections = []
    for chunk, tokens, ids in zip(whitespace_chunks(parts.base_text), (["hoc"], ["tap"]), ([1], [2])):
        alignment = align_chunk(chunk, tokens, ids)
        overlays = overlay_orthography(alignment.pieces, regions)
        projections += [
            project_piece(len(projections), p, o, parts.base_text, labels, regions, tones)
            for p, o in zip(alignment.pieces, overlays)
        ]
    assert len(projections) == 2
    assert not any(
        c.character.isspace() for p in projections for c in p.letter.contributions
    )


# ---------------------------------------------------------------------------
# 13. NONE participates in the letter channel
# ---------------------------------------------------------------------------
def test_plain_letters_are_applicable_contributors():
    projection = project_all("hoc", ["hoc"], [1])[0]
    assert projection.letter.applicable_labels == (LetterDiacritic.NONE,) * 3
    assert projection.letter.has_applicable_contributors


# ---------------------------------------------------------------------------
# 14. NA is excluded from the letter channel
# ---------------------------------------------------------------------------
def test_punctuation_is_recorded_but_not_applicable():
    projection = project_all("hoc.", ["hoc."], [1])[0]
    assert len(projection.letter.contributions) == 4
    assert len(projection.letter.applicable) == 3
    assert projection.letter.contributions[3].letter_diacritic is LetterDiacritic.NA
    assert not projection.letter.contributions[3].is_applicable


def test_letter_marks_survive_projection():
    projection = project_all("đường", ["duong"], [1])[0]
    labels = projection.letter.applicable_labels
    assert LetterDiacritic.STROKE in labels
    assert LetterDiacritic.HORN in labels


# ---------------------------------------------------------------------------
# 15. Zero applicable contributors
# ---------------------------------------------------------------------------
def test_token_with_no_applicable_letters_is_na():
    projection = project_all("...", ["..."], [1])[0]
    assert projection.letter.applicable == ()
    assert not projection.letter.has_applicable_contributors
    assert projection.letter.token_label is LetterDiacritic.NA


# ---------------------------------------------------------------------------
# 16. The pooling rule is recorded, not implemented
# ---------------------------------------------------------------------------
def test_pooling_rule_is_persisted_on_every_projection():
    projection = project_all("đường", ["duong"], [1])[0]
    assert projection.letter.pooling_rule == LETTER_POOLING_RULE
    assert "mean" in LETTER_POOLING_RULE
    assert "NONE is" in LETTER_POOLING_RULE and "NA contributors are excluded" in LETTER_POOLING_RULE


def test_pooling_is_not_performed_here():
    """This module publishes the contributors; the adapter pools them. Nothing
    here computes an embedding-space mean."""
    source = pathlib.Path("unmark/alignment/channels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"mean", "average", "fmean"}
    assert "import torch" not in source and "import numpy" not in source


# ---------------------------------------------------------------------------
# 17. Contribution order and identity
# ---------------------------------------------------------------------------
def test_contributions_are_in_source_order():
    projection = project_all("đường", ["duong"], [1])[0]
    indices = [c.char_index for c in projection.letter.contributions]
    assert indices == sorted(indices)
    assert indices == list(range(projection.global_start, projection.global_end))


def test_contribution_characters_match_the_base_text():
    text = "nghiên cứu"
    parts = decompose(text)
    labels = character_letter_labels(parts)
    regions = regions_from(text)
    for chunk, tokens, ids in zip(whitespace_chunks(parts.base_text), (["nghi@@", "en"], ["cuu"]), ([1, 2], [3])):
        alignment = align_chunk(chunk, tokens, ids)
        overlays = overlay_orthography(alignment.pieces, regions)
        for i, (p, o) in enumerate(zip(alignment.pieces, overlays)):
            projection = project_piece(i, p, o, parts.base_text, labels, regions, {})
            for contribution in projection.letter.contributions:
                assert contribution.character == parts.base_text[contribution.char_index]


# ---------------------------------------------------------------------------
# 18. Ranges are inherited, never recomputed
# ---------------------------------------------------------------------------
def test_projection_ranges_equal_the_piece_ranges():
    text = "nghiên"
    parts = decompose(text)
    alignment = align_chunk(whitespace_chunks(parts.base_text)[0], ["nghi@@", "en"], [1, 2])
    overlays = overlay_orthography(alignment.pieces, regions_from(text))
    for i, (piece, overlay) in enumerate(zip(alignment.pieces, overlays)):
        projection = project_piece(
            i, piece, overlay, parts.base_text, character_letter_labels(parts), regions_from(text), {}
        )
        assert projection.global_start == piece.global_start
        assert projection.global_end == piece.global_end


# ---------------------------------------------------------------------------
# 19. Unknown vocabulary ids are reported, not treated as failures
# ---------------------------------------------------------------------------
def test_unknown_token_id_is_reported_and_still_projects():
    """The `khut` finding: an OOV id does not make the surface unrecoverable."""
    text = "khút"
    parts = decompose(text)
    alignment = align_chunk(whitespace_chunks(parts.base_text)[0], ["khut"], [3], unk_token_id=3)
    regions = regions_from(text)
    projection = project_piece(
        0,
        alignment.pieces[0],
        overlay_orthography(alignment.pieces, regions)[0],
        parts.base_text,
        character_letter_labels(parts),
        regions,
        region_tones(text, regions),
    )
    assert projection.has_unknown_token_id, "the OOV id must be reported"
    assert projection.tone.label is TokenToneLabel.SAC, "and must not suppress the channels"
    assert projection.letter.has_applicable_contributors
    assert summarize_projections([projection])["tokens_with_unknown_token_id"] == 1


# ---------------------------------------------------------------------------
# 20-25. Corruption invariance, against the real B2 engine
# ---------------------------------------------------------------------------
CORRUPTION_TEXT = "Tôi học nghiên cứu đường phố."


def corrupt_text(condition: str) -> str:
    return corrupt(
        CORRUPTION_TEXT,
        condition,
        seed=7,
        sample_id="b3b1c",
        purpose=CorruptionPurpose.SELF_CHECK,
    ).corrupted_text


# 20. b(x) is invariant -- this is what lets one token grid serve every condition
@pytest.mark.parametrize("condition", CONDITIONS)
def test_base_text_is_invariant_under_corruption(condition):
    assert decompose(corrupt_text(condition)).base_text == decompose(CORRUPTION_TEXT).base_text


# 21. and therefore so is the chunking
@pytest.mark.parametrize("condition", CONDITIONS)
def test_whitespace_chunks_are_invariant_under_corruption(condition):
    base = decompose(corrupt_text(condition)).base_text
    reference = decompose(CORRUPTION_TEXT).base_text
    assert [(c.text, c.start, c.end) for c in whitespace_chunks(base)] == [
        (c.text, c.start, c.end) for c in whitespace_chunks(reference)
    ]


# 22. contribution ranges do not move
@pytest.mark.parametrize("condition", CONDITIONS)
def test_contribution_ranges_are_invariant_under_corruption(condition):
    def ranges(text: str):
        parts = decompose(text)
        labels = character_letter_labels(parts)
        regions = regions_from(text)
        out = []
        for chunk in whitespace_chunks(parts.base_text):
            alignment = align_chunk(chunk, [chunk.text], [1])
            overlays = overlay_orthography(alignment.pieces, regions)
            for i, (p, o) in enumerate(zip(alignment.pieces, overlays)):
                projection = project_piece(i, p, o, parts.base_text, labels, regions, {})
                out.append(
                    (
                        projection.global_start,
                        projection.global_end,
                        tuple(c.char_index for c in projection.letter.contributions),
                    )
                )
        return out

    assert ranges(corrupt_text(condition)) == ranges(CORRUPTION_TEXT)


# 23. the tone channel is what degrades
def test_tone_channel_degrades_monotonically():
    counts = []
    for condition in CONDITIONS:
        text = corrupt_text(condition)
        parts = decompose(text)
        labels = character_letter_labels(parts)
        regions = regions_from(text)
        tones = region_tones(text, regions)
        marked = 0
        for chunk in whitespace_chunks(parts.base_text):
            alignment = align_chunk(chunk, [chunk.text], [1])
            overlays = overlay_orthography(alignment.pieces, regions)
            for i, (p, o) in enumerate(zip(alignment.pieces, overlays)):
                projection = project_piece(i, p, o, parts.base_text, labels, regions, tones)
                if projection.tone.label not in {TokenToneLabel.NA, TokenToneLabel.UNMARKED}:
                    marked += 1
        counts.append(marked)
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0, "FULL must retain marked tones"
    assert counts[-1] == 0, "STRIP_ALL must leave no readable tone"


# 24. and STRIP_ALL leaves UNMARKED, never NA, on real syllables
def test_strip_all_leaves_unmarked_not_not_applicable():
    text = corrupt_text("STRIP_ALL")
    parts = decompose(text)
    regions = regions_from(text)
    labels = character_letter_labels(parts)
    tones = region_tones(text, regions)
    seen = set()
    for chunk in whitespace_chunks(parts.base_text):
        alignment = align_chunk(chunk, [chunk.text], [1])
        overlays = overlay_orthography(alignment.pieces, regions)
        for i, (p, o) in enumerate(zip(alignment.pieces, overlays)):
            projection = project_piece(i, p, o, parts.base_text, labels, regions, tones)
            seen.add(projection.tone.label)
    assert TokenToneLabel.UNMARKED in seen
    assert not seen & {TokenToneLabel.SAC, TokenToneLabel.HUYEN, TokenToneLabel.NANG}


# 25. the letter channel degrades too, and only under conditions that touch it
def test_letter_channel_is_untouched_by_tone_only_conditions():
    def letters(text: str) -> list[LetterDiacritic]:
        parts = decompose(text)
        return [
            label for label in character_letter_labels(parts) if label is not LetterDiacritic.NA
        ]

    assert letters(corrupt_text("P50")) == letters(CORRUPTION_TEXT)
    assert letters(corrupt_text("STRIP_ALL")) != letters(CORRUPTION_TEXT)


# ---------------------------------------------------------------------------
# 26. Determinism
# ---------------------------------------------------------------------------
def test_projection_is_deterministic():
    first = [p.to_dict() for p in project_all("nghiên cứu".split()[0], ["nghi@@", "en"], [1, 2])]
    second = [p.to_dict() for p in project_all("nghiên cứu".split()[0], ["nghi@@", "en"], [1, 2])]
    assert first == second


# ---------------------------------------------------------------------------
# 27. Serialisation
# ---------------------------------------------------------------------------
def test_projections_serialise_stably():
    projection = project_all("đường.", ["duong."], [1])[0]
    payload = projection.to_dict()
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert json.loads(encoded) == payload
    assert payload["tone"]["ownership"] == "SINGLE_CANDIDATE"
    assert payload["tone"]["label"] == "HUYEN"
    assert payload["letter"]["applicable_labels"].count("NONE") >= 1
    assert payload["has_source_range"] is True


def test_summary_counts_match_the_projections():
    projections = [project_special_token(0, "<s>", 0)] + project_all("học-cứu", ["hoc-cuu"], [1])
    summary = summarize_projections(projections)
    assert summary["total_tokens"] == 2
    assert summary["special_tokens"] == 1
    assert summary["tokens_tone_ambiguous"] == 1
    assert summary["tokens_with_tone"] == 0
    assert summary["tone_label_counts"]["NA"] == 2


# ---------------------------------------------------------------------------
# 28. Hygiene: no second Unicode implementation, no ML in the projection layer
# ---------------------------------------------------------------------------
def test_alignment_does_not_reimplement_unicode_decomposition():
    """Character structure has exactly one source of truth: `unmark.orthography`."""
    source = pathlib.Path("unmark/alignment/channels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "unicodedata" not in imported
    assert "unmark.orthography" in imported
    for banned in ("NFD", "NFC", "combining(", "\\u0300"):
        assert banned not in source, f"{banned} suggests a second decomposition"


def test_projection_layer_imports_no_ml_dependencies():
    source = pathlib.Path("unmark/alignment/channels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not modules & {"torch", "transformers", "sentencepiece", "datasets", "numpy"}
