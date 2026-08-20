"""Eligibility policy: which syllable spans corruption probabilities apply to.

The scientific question
-----------------------
Proposal v1.3 section 4.3 fixes the intended semantics:

> An alphabetic span is treated as a **Vietnamese candidate** if it matches the
> Vietnamese syllable inventory after stripping; otherwise both channels are
> `N/A`.

Section 6.3 then defines `P25`/`P50`/`P75` as "Tone marks removed from
25/50/75% of **syllables**". Read together, the denominator of `p` is the set of
**eligible Vietnamese syllables**, not every alphabetic run in the string.

The current state
-----------------
That rule needs the Vietnamese syllable inventory, which is not enumerated in
the proposal and does not exist in this repository (GAP-2, see
`docs/spec/orthography.md` D-002). B1A therefore reports every alphabetic span
as `Eligibility.UNDECIDED`, and B2 must not quietly upgrade that to "eligible".

So B2 separates two things that the first implementation conflated:

* **candidate span** -- a maximal alphabetic run. Always computable, structural,
  language-blind. This is what the deterministic engine scores today.
* **eligible Vietnamese syllable** -- a candidate that the resolved eligibility
  policy accepts. **Not computable yet.**

While the policy is `UNRESOLVED`, every count and rate B2 reports is about
*candidates*, and is named accordingly. `CorruptionResult.eligible_units` does
not return a provisional number: it raises, because there is no honest value.

Why this matters, concretely
----------------------------
Under the provisional fallback, `P50` on `toi dung Python va PyTorch` puts
`Python` and `PyTorch` in the denominator, so the realized rate is a fraction of
five spans rather than of three Vietnamese syllables. And `STRIP_ALL` rewrites
`café` to `cafe`, because the acute is a Vietnamese tone codepoint. Neither is
wrong as an *engine* behaviour; both are wrong as a *scientific protocol*, and
the difference must survive into the artifacts.

The guard
---------
`require_resolved_eligibility()` raises while the policy is `UNRESOLVED`, and
`corrupt()` calls it by default. Generating stage-1 training data or final
evaluation corruption therefore fails loudly today; self-check and unit tests
opt in to the provisional mode explicitly.
"""

from __future__ import annotations

from enum import Enum


class EligibilityPolicy(Enum):
    """How a candidate span becomes an eligible Vietnamese syllable."""

    UNRESOLVED = "UNRESOLVED"
    """GAP-2 is open. No candidate can be confirmed eligible; every count and
    rate B2 produces is about *candidate spans*. Valid for implementation
    verification only."""

    VIETNAMESE_SYLLABLE_INVENTORY = "VIETNAMESE_SYLLABLE_INVENTORY"
    """Proposal 4.3's rule: a candidate is eligible when its stripped form is in
    the Vietnamese syllable inventory. **Not implemented.** Reserved so the
    resolved state has a name before it has an implementation."""


def active_eligibility_policy() -> EligibilityPolicy:
    """The policy actually in force right now.

    Resolved when the pinned Vietnamese syllable inventory is present and
    verified, `UNRESOLVED` otherwise -- so deleting the git-ignored cache
    re-arms the guard rather than silently falling back to candidate spans.
    Computed rather than hard-coded for exactly that reason.
    """
    from unmark.linguistics import try_load_inventory

    return (
        EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY
        if try_load_inventory() is not None
        else EligibilityPolicy.UNRESOLVED
    )


class CorruptionPurpose(Enum):
    """What a corruption call is for, which decides whether the guard applies."""

    SCIENTIFIC = "SCIENTIFIC"
    """Stage-1 training data, evaluation conditions, anything whose numbers
    reach a table. Requires a resolved eligibility policy. The default, so that
    the unsafe path is the one you have to ask for."""

    SELF_CHECK = "SELF_CHECK"
    """Implementation verification: unit tests and the B2 self-check. Runs under
    the provisional candidate-span fallback, and every artifact it produces is
    stamped `provisional_eligibility: true`."""


class EligibilityUnresolved(RuntimeError):
    """Raised when scientific corruption is attempted before GAP-2 is closed."""


_UNRESOLVED_MESSAGE = """\
Corruption for {context} requires a resolved eligibility policy, and the current
policy is {policy}.

WHY THIS IS BLOCKED
Proposal v1.3 section 4.3 defines a Vietnamese candidate as an alphabetic span
"[that] matches the Vietnamese syllable inventory after stripping; otherwise both
channels are N/A", and section 6.3 applies the corruption probability to
syllables. The Vietnamese syllable inventory that rule needs is not enumerated in
the proposal and does not exist in this repository. That open item is GAP-2
(docs/spec/orthography.md D-002).

Until it is closed, B2 can only score *candidate spans* -- every maximal
alphabetic run, language-blind. Using that as the scientific denominator would:
  * put English spans into the p25/p50/p75 denominator, so the reported rate is
    not the rate of corrupted Vietnamese syllables;
  * let STRIP_ALL rewrite foreign words whose spelling uses a codepoint that is
    also a Vietnamese tone mark (cafe' -> cafe).

The deterministic engine itself is final and unaffected; only the denominator
and the span filter are open.

WHAT TO DO
  * GAP-2 is closed in code (B3A), but the inventory itself is NOT committed --
    the upstream gist carries no license statement. Fetch and verify it:

        .venv/bin/python scripts/fetch_vietnamese_syllable_inventory.py

    That is almost certainly why you are seeing this message.
  * For implementation verification without the inventory, pass
    purpose=CorruptionPurpose.SELF_CHECK. Results are stamped
    provisional_eligibility=True and their counts are named `candidate_*`,
    never `eligible_*`.
  * See docs/spec/decisions.md D-B2-003 and D-B3A-001.
"""


def require_resolved_eligibility(
    context: str = "stage-1 training or evaluation generation",
    policy: EligibilityPolicy | None = None,
) -> None:
    """Raise unless the eligibility policy is resolved.

    The point is that it is *impossible* to generate scientific corruption data
    under the provisional fallback by accident.
    """
    policy = active_eligibility_policy() if policy is None else policy
    if policy is EligibilityPolicy.UNRESOLVED:
        raise EligibilityUnresolved(_UNRESOLVED_MESSAGE.format(context=context, policy=policy.name))


def is_resolved(policy: EligibilityPolicy | None = None) -> bool:
    policy = active_eligibility_policy() if policy is None else policy
    return policy is not EligibilityPolicy.UNRESOLVED
