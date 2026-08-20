"""Vietnamese candidate classification: a pure function of the stripped form.

Proposal v1.3 §4.3:

> An alphabetic span is treated as a **Vietnamese candidate** if it matches the
> Vietnamese syllable inventory after stripping; otherwise both channels are
> `N/A`.
>
> Ambiguous spans are resolved towards Vietnamese, and this is documented as a
> known and deliberate error mode rather than hidden. One property matters more
> than the rule's accuracy: it is a pure function of the *stripped* form, so it
> assigns the same labels to clean and corrupted input and cannot break grid
> invariance.

This module is that rule and nothing more. It is **orthographic and structural,
never semantic**. It does not use, and must never use:

* language identification;
* word or corpus frequency;
* sentence context or neighbouring spans;
* an English dictionary or stop-word list;
* capitalisation heuristics.

The consequence, accepted by the proposal and stated plainly here: an English
word whose letters happen to form a valid stripped Vietnamese syllable is
classified as Vietnamese. `ban`, `the`, `com`, `on`, `in`, `an`, `la`, `co` are
all real stripped Vietnamese syllables, so an English sentence containing them
has those spans marked eligible. Words that are not syllable-shaped -- `machine`,
`learning`, `Python`, `PyTorch` -- are not, because Vietnamese phonotactics
cannot produce them.

The property that matters is invariance, not accuracy: because the rule reads
only the stripped form, `học` and its corrupted form `hoc` classify identically,
so corruption cannot change which units are eligible.
"""

from __future__ import annotations

from typing import Callable

from unmark.linguistics.inventory import SyllableInventory, membership_form
from unmark.orthography import Eligibility


def is_vietnamese_candidate(text: str, inventory: SyllableInventory) -> bool:
    """Whether `text`'s stripped form is in the inventory."""
    return inventory.contains_membership_form(membership_form(text))


def classify_candidate(text: str, inventory: SyllableInventory | None) -> Eligibility:
    """Classify one span.

    Returns `VIETNAMESE_CANDIDATE` when the stripped form is in the inventory,
    `NOT_APPLICABLE` when it is not or the span is not alphabetic at all, and
    `UNDECIDED` only when there is no inventory to consult -- the one state that
    means "cannot be resolved", never "resolved as non-Vietnamese".
    """
    if inventory is None:
        return Eligibility.UNDECIDED
    form = membership_form(text)
    if not form or not form.replace(" ", "").isalpha():
        # Digits, punctuation, symbols and emoji are trivially N/A (§4.3).
        return Eligibility.NOT_APPLICABLE
    if inventory.contains_membership_form(form):
        return Eligibility.VIETNAMESE_CANDIDATE
    return Eligibility.NOT_APPLICABLE


def make_classifier(inventory: SyllableInventory | None) -> Callable[[str], Eligibility]:
    """A single-argument classifier, for passing into `decompose`."""

    def classify(text: str) -> Eligibility:
        return classify_candidate(text, inventory)

    return classify
