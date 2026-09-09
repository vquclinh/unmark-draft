"""Stage-2 tone-channel regression: tone-only corruption must reach `tone_ids`.

**Why this file exists.** A real Stage-2 measurement run produced degraded
representations bit-identical to `FULL` for `P25`..`P100`, while the corrupted
*text* differed on 1576/1583 rows. The tone channel was silently dead: with no
syllable-inventory classifier, every orthographic region carries
`Eligibility.UNDECIDED`, `overlay_orthography` returns `ToneOwnership.UNRESOLVED`
for every piece, and every tone label collapses to `NA`. Nothing else notices --
the base grid matches, `b(C(x)) == b(x)` holds, and the letter channel still
changes under `STRIP_ALL`, which is exactly why the failure looked like a
head-training problem rather than a projection one.

**Torch-free.** Everything here is deterministic projection over a stub
tokenizer. Nothing trains, reads a downstream row, or names official TEST.
"""

from __future__ import annotations

import pathlib
import sys
import zlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.alignment import ToneOwnership  # noqa: E402
from unmark.alignment.channels import TokenToneLabel  # noqa: E402
from unmark.corruption import CorruptionPurpose, EligibilityPolicy  # noqa: E402
from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    prepare_stage2_unmark_input,
    require_resolved_tone_channel,
)
from unmark.modeling.collate import OBSERVABLE_TONE_IDS  # noqa: E402
from unmark.modeling.contracts import TONE_NA_SENTINEL  # noqa: E402
from unmark.orthography import Eligibility  # noqa: E402

UNMARKED_ID = OBSERVABLE_TONE_IDS[TokenToneLabel.UNMARKED.value]
SEED = 19225

# "Tôi đã học" -- three syllables, two carrying a visible tone mark (nga, nang),
# and one genuinely unmarked. "đ" additionally carries a letter-forming stroke,
# so the letter channel has something to lose under STRIP_ALL and nothing to
# lose under a tone-only condition.
MARKED = "Tôi đã học"


class StubTokenizer:
    """Character-level tokenizer over the already-stripped base chunks."""

    pad_token_id = 1
    unk_token_id = 3

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return [7 + zlib.crc32(t.encode("utf-8")) % 400 for t in tokens]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + list(ids) + [2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


def vietnamese(_text: str) -> Eligibility:
    return Eligibility.VIETNAMESE_CANDIDATE


def not_applicable(_text: str) -> Eligibility:
    return Eligibility.NOT_APPLICABLE


def prepare(condition, *, text=MARKED, classifier=vietnamese, sample_id="regression"):
    return prepare_stage2_unmark_input(
        text=text,
        sample_id=sample_id,
        tokenizer=StubTokenizer(),
        condition=condition,
        corruption_seed=SEED,
        classifier=classifier,
        corruption_purpose=CorruptionPurpose.SELF_CHECK,
        eligibility_policy=EligibilityPolicy.UNRESOLVED,
        max_length=64,
    )


def content_tones(prepared):
    """Tone ids at non-special positions."""
    return [
        tone
        for tone, special in zip(prepared.tone_ids, prepared.special_tokens_mask)
        if not special
    ]


# ==========================================================================================
# The guard itself: the exact condition that made the real run silently wrong
# ==========================================================================================

def test_an_unresolved_tone_channel_is_refused_rather_than_silently_dead():
    """Without a classifier every tone label is NA; this must fail closed.

    Before the repair this returned a `Stage2UnmarkInput` whose `tone_ids` were
    uniformly the NA sentinel in **every** condition, so `P25`..`P100` were
    bit-identical to `FULL`.
    """
    with pytest.raises(EvaluationContractViolation, match="unresolved tone channel"):
        prepare("FULL", classifier=None)
    with pytest.raises(EvaluationContractViolation, match="unresolved tone channel"):
        prepare("P100", classifier=None)


def test_the_guard_names_the_remedy_and_does_not_special_case_a_condition():
    with pytest.raises(EvaluationContractViolation) as caught:
        prepare("P100", classifier=None)
    message = str(caught.value)
    assert "classifier" in message
    assert "inventory" in message
    for condition in ("P25", "P50", "P75", "STRIP_ALL"):
        assert condition not in message, "the guard must not be condition-specific"


def test_the_guard_reads_only_projection_ownership():
    class Piece:
        def __init__(self, index, ownership):
            self.token_index = index
            self.tone = type("T", (), {"ownership": ownership})()

    require_resolved_tone_channel(
        [Piece(0, ToneOwnership.SINGLE_CANDIDATE), Piece(1, ToneOwnership.NOT_APPLICABLE)],
        sample_id="s", what="x",
    )
    with pytest.raises(EvaluationContractViolation, match="unresolved tone channel"):
        require_resolved_tone_channel(
            [Piece(0, ToneOwnership.UNRESOLVED)], sample_id="s", what="x"
        )


# ==========================================================================================
# A. FULL vs P100 -- tone-only corruption reaches tone_ids
# ==========================================================================================

def test_full_versus_p100_changes_tone_ids_and_nothing_else():
    full, p100 = prepare("FULL"), prepare("P100")

    assert full.corrupted_text != p100.corrupted_text
    assert full.base_text == p100.base_text
    assert full.input_ids == p100.input_ids
    assert full.special_tokens_mask == p100.special_tokens_mask

    assert full.tone_ids != p100.tone_ids, (
        "tone-only corruption must change the tone channel; identical tone_ids is "
        "the exact defect this file regresses"
    )
    changed = [i for i, (a, b) in enumerate(zip(full.tone_ids, p100.tone_ids)) if a != b]
    assert changed, "at least one content tone_id must differ"

    assert full.letter_ids == p100.letter_ids, (
        "P100 removes tone state only; letter-forming diacritics must survive"
    )


def test_every_tone_removed_by_p100_becomes_the_observable_unmarked_state():
    full, p100 = prepare("FULL"), prepare("P100")
    moved = 0
    for before, after in zip(full.tone_ids, p100.tone_ids):
        if before == after:
            continue
        assert before not in (UNMARKED_ID, TONE_NA_SENTINEL), (
            "only a visibly marked tone may change"
        )
        assert after == UNMARKED_ID, (
            "a stripped tone becomes the observable UNMARKED state, never a new state"
        )
        moved += 1
    assert moved >= 1


def test_p100_introduces_no_new_tone_state():
    full, p100 = prepare("FULL"), prepare("P100")
    assert set(p100.tone_ids) <= set(full.tone_ids) | {UNMARKED_ID}


# ==========================================================================================
# B. Partial tone corruption
# ==========================================================================================

@pytest.mark.parametrize("condition", ["P25", "P50", "P75"])
def test_partial_conditions_are_deterministic_and_keep_the_base_grid(condition):
    first, second = prepare(condition), prepare(condition)
    assert first.corrupted_text == second.corrupted_text
    assert first.tone_ids == second.tone_ids

    full = prepare("FULL")
    assert first.base_text == full.base_text
    assert first.input_ids == full.input_ids
    assert first.letter_ids == full.letter_ids


@pytest.mark.parametrize("condition", ["P25", "P50", "P75"])
def test_a_partial_condition_that_removes_a_tone_shows_it_in_tone_ids(condition):
    full, partial = prepare("FULL"), prepare(condition)
    if partial.corrupted_text == full.corrupted_text:
        pytest.skip(f"{condition} removed no tone from this sample under seed {SEED}")
    assert partial.tone_ids != full.tone_ids
    for before, after in zip(full.tone_ids, partial.tone_ids):
        if before != after:
            assert after == UNMARKED_ID


def test_at_least_one_partial_condition_actually_removes_a_tone():
    """Guards the skip above from hiding a universally dead channel."""
    full = prepare("FULL")
    assert any(
        prepare(c).tone_ids != full.tone_ids for c in ("P25", "P50", "P75", "P100")
    )


# ==========================================================================================
# C. STRIP_ALL
# ==========================================================================================

def test_strip_all_changes_both_channels_and_keeps_the_base_grid():
    full, strip = prepare("FULL"), prepare("STRIP_ALL")
    assert strip.base_text == full.base_text
    assert strip.input_ids == full.input_ids
    assert strip.tone_ids != full.tone_ids, "STRIP_ALL removes tone state"
    assert strip.letter_ids != full.letter_ids, "STRIP_ALL removes letter-forming marks"
    for before, after in zip(full.tone_ids, strip.tone_ids):
        if before != after:
            assert after == UNMARKED_ID


# ==========================================================================================
# D. Genuine ngang is indistinguishable from a stripped tone -- no oracle
# ==========================================================================================

def test_a_genuine_unmarked_syllable_and_a_stripped_one_share_one_observable_state():
    """`Toi` was never marked; `hoc` lost its `nang`. Both read UNMARKED."""
    full, p100 = prepare("FULL"), prepare("P100")

    genuinely_unmarked = [
        i for i, (a, b) in enumerate(zip(full.tone_ids, p100.tone_ids))
        if a == b == UNMARKED_ID
    ]
    stripped = [
        i for i, (a, b) in enumerate(zip(full.tone_ids, p100.tone_ids))
        if a != b
    ]
    assert genuinely_unmarked, "the fixture must contain a genuinely unmarked syllable"
    assert stripped, "the fixture must contain a stripped tone"
    for index in stripped:
        assert p100.tone_ids[index] == UNMARKED_ID
    assert {p100.tone_ids[i] for i in genuinely_unmarked} == {UNMARKED_ID}


def test_the_stripped_text_alone_determines_the_tone_state():
    """Deployability: preparing the already-stripped string reproduces P100 tones."""
    p100 = prepare("P100")
    from_stripped = prepare("FULL", text=p100.corrupted_text, sample_id="deployed")
    assert from_stripped.tone_ids == p100.tone_ids, (
        "the tone state must be derivable from the observable input alone; no "
        "knowledge of the original marked string may be required"
    )


# ==========================================================================================
# E. Non-Vietnamese and special tokens
# ==========================================================================================

def test_non_vietnamese_text_keeps_na_tones_and_is_not_refused():
    prepared = prepare("FULL", text="hello world", classifier=not_applicable)
    assert set(prepared.tone_ids) == {TONE_NA_SENTINEL}
    assert not any(prepared.tone_mask)


def test_special_token_positions_stay_na_and_masked_out():
    for condition in ("FULL", "P100", "STRIP_ALL"):
        prepared = prepare(condition)
        for tone, mask, special in zip(
            prepared.tone_ids, prepared.tone_mask, prepared.special_tokens_mask
        ):
            if special:
                assert tone == TONE_NA_SENTINEL
                assert mask is False


# ==========================================================================================
# F. The direct Stage-2 regression
# ==========================================================================================

def test_stage2_full_and_p100_are_not_identical_for_an_eligible_marked_syllable():
    """The one-line statement of the bug, as a test."""
    full, p100 = prepare("FULL"), prepare("P100")
    assert full.tone_ids != p100.tone_ids
    assert content_tones(full) != content_tones(p100)
