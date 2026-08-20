# Decision and deviation log

Every implementation decision that changes, narrows, clarifies or deviates from
`unmark-proposal.md`. Recorded so a later proposal-vs-repository audit can be
mechanical rather than archaeological.

Decisions that are purely about orthography live in
[`orthography.md`](orthography.md) (D-001…D-003) and are referenced here rather
than repeated.

**Status vocabulary**

| Status | Meaning |
|---|---|
| `MATCH` | Implementation follows the proposal exactly; nothing to record beyond the entry |
| `CLARIFIED` | The proposal was underspecified; a concrete reading was chosen and is recorded here |
| `DEVIATED` | The implementation intentionally differs from the proposal |
| `DEFERRED` | The proposal requires something not implemented yet |
| `BLOCKED` | Cannot proceed without a decision the researcher must make |
| `TEMPORARY IMPLEMENTATION FALLBACK` | The code does something the final protocol must **not** do. Safe only for implementation verification, guarded so it cannot reach an experiment, and owned by a named later stage |

A later pre-training proposal-vs-repository audit must be able to tell these
apart at a glance:

| Category | Entries |
|---|---|
| **INTENDED FINAL SPEC** | D-B2-001, D-B2-002, D-B2-004, D-B2-006 |
| **TEMPORARY IMPLEMENTATION FALLBACK** | *(none open)* — D-B2-003 was closed by D-B3A-001 |
| **RESOLVED DECISION** | D-001 (canonical tone placement) |
| **RESOLVED DECISION** (cont.) | D-B3A-001 (Vietnamese eligibility, closes GAP-2) |
| **KNOWN DEFERRED GAP** | D-B2-005 (`VARIANT`) |

---

## B2 — deterministic corruption

Summary: **no DEVIATED entries.** D-B2-003 is now **closed** by D-B3A-001; no temporary
fallback remains open. The deterministic engine follows proposal §5.3
and §6.3 as written. Four clarifications, one deferral, and — after researcher
review of [audit 003](../audits/003-b2-deterministic-corruption.md) — **one
temporary implementation fallback** that must be closed before any training run
(D-B2-003, see [audit 004](../audits/004-b2-eligibility-safety-followup.md)).

### D-B2-001 — the corruption key carries an explicit `sample_id`

| | |
|---|---|
| **Status** | CLARIFIED |
| **Proposal** | §5.3: "Corruption is a deterministic function `C(x, p, s) → x̃` of example, rate, and seed: the same triple must always produce the same corrupted string." |
| **Implemented** | `corrupt(text, condition, seed, sample_id)`. The per-unit key is `(schema_version, seed, sample_id, sha256(canonical clean text), unit_index)`, digested with BLAKE2b. |
| **Why** | The proposal names "example" as part of the key but does not say how an example is identified. Row order is unusable (reordering a dataset would change its noise) and Python's `hash()` is randomised per process (a corpus would not reproduce tomorrow). An explicit `sample_id` makes the proposal's own requirement enforceable. |
| **Affected code** | `unmark/corruption/deterministic.py`, `unmark/corruption/corrupt.py` |
| **Affected experiments** | All corruption conditions. Downstream data layers must supply a stable `sample_id`; if a dataset has no natural id, one must be derived deterministically from the clean sample. That policy is *not* decided here. |
| **Proposal updated** | Yes — §5.3 now states the key explicitly. |
| **PDF stale** | Yes. `unmark-proposal.pdf` was not regenerated. |

### D-B2-002 — independent per-syllable selection, not an exact count

| | |
|---|---|
| **Status** | CLARIFIED |
| **Proposal** | §6.3: "Tone marks removed from 25/50/75% of syllables". §4.6: "set the tone channel to `UNMARKED` for a random `p`-fraction of syllables". |
| **Implemented** | Each syllable is selected independently when its stable score is `< p`. The realized fraction is recorded, not forced to `round(p·N)`. |
| **Why** | "p-fraction" admits two readings. Choosing exactly `round(p·N)` units would make each unit's fate depend on the sentence's length and on every other unit, which contradicts §5.3's determinism requirement in spirit: inserting one syllable would change decisions elsewhere. Independent Bernoulli selection keeps each unit's decision a pure function of its own key. On short sentences the realized fraction therefore differs from `p`, which is expected and reported. |
| **Affected code** | `unmark/corruption/deterministic.py` |
| **Affected experiments** | `P25`, `P50`, `P75`. Reported per example as `requested_probability` and `realized_probability`. |
| **Proposal updated** | No — the wording is compatible with this reading; the clarification lives here. |
| **PDF stale** | n/a |

### D-B2-003 — candidate spans stand in for eligible Vietnamese syllables

| | |
|---|---|
| **Status** | **CLOSED** by [D-B3A-001](#d-b3a-001--vietnamese-syllable-eligibility-resolved) on 2026-08-19. *Was:* TEMPORARY IMPLEMENTATION FALLBACK, reclassified by researcher review of audit 003 which had recorded it as a harmless CLARIFIED entry. |
| **Resolution owner** | B3 / pre-training — **discharged**. See [audit 005](../audits/005-b3a-vietnamese-syllable-eligibility.md). |

**Original proposal wording.** §4.3: "An alphabetic span is treated as a
**Vietnamese candidate** if it matches the Vietnamese syllable inventory after
stripping; otherwise both channels are `N/A`." §6.3: "Tone marks removed from
25/50/75% of **syllables**." Read together, the denominator of `p` is the set of
eligible Vietnamese syllables.

**Temporary implementation state.** The Vietnamese syllable inventory that rule
needs is not enumerated in the proposal and does not exist in this repository
(GAP-2, `orthography.md` D-002). B2 therefore scores **candidate spans** — every
maximal alphabetic run, language-blind. Nothing is labelled eligible:
`Eligibility.UNDECIDED` is carried through to every `UnitDecision`, all counts
are named `candidate_*`, and `CorruptionResult.eligible_units` **raises** rather
than returning the candidate count.

**Why the fallback exists, and why it is not innocuous.** B2's deterministic
engine is complete and correct, and blocking all of B2 on GAP-2 would have
stalled work that does not depend on it. But the fallback is *not* the
scientific protocol, for two concrete reasons:

1. English spans enter the `P25`/`P50`/`P75` denominator, so a reported rate is
   a fraction of alphabetic runs, not of Vietnamese syllables. On
   `toi dung Python va PyTorch` the denominator is 5, not 3.
2. `STRIP_ALL` rewrites a foreign span whose spelling uses a codepoint that is
   also a Vietnamese tone mark: `café` → `cafe`.

Audit 003 recorded these as documented consequences of a clarification. That was
too weak: they change what a corruption rate *means*, so the entry is
reclassified as a temporary fallback with a guard.

**Guard.** `unmark/corruption/eligibility.py` defines `EligibilityPolicy`
(`UNRESOLVED` / `VIETNAMESE_SYLLABLE_INVENTORY`) and `CorruptionPurpose`
(`SCIENTIFIC` / `SELF_CHECK`). `corrupt()` defaults to `SCIENTIFIC` and calls
`require_resolved_eligibility()`, so **generating training or evaluation data
raises today**. Implementation verification must ask for `SELF_CHECK`
explicitly, and every artifact it writes is stamped
`provisional_eligibility: true`.

**What closing it requires.** Enumerate the Vietnamese syllable inventory, apply
it to the *stripped* form (so clean and corrupted input get identical labels and
the base grid stays invariant, §4.3), filter `spans` at the single marked place
in `corrupt.py`, set `ACTIVE_ELIGIBILITY_POLICY`, and re-run every corruption
artifact — old ones are not comparable, because the denominator changes.

| | |
|---|---|
| **Affected code** | `unmark/corruption/eligibility.py`, `corrupt.py`, `models.py`, `scripts/b2_corruption_self_check.py`, `configs/corruption/default.yaml` |
| **Affected metrics** | `candidate_selection_rate` and `candidate_modification_rate` for `P25`/`P50`/`P75`; which spans `STRIP_ALL` touches; anything downstream that counts corrupted syllables |
| **Proposal updated** | No — and deliberately not. §4.3 already states the intended semantics correctly; writing the fallback into the proposal would make a temporary state look normative. |
| **PDF stale** | n/a for this entry |

### D-B2-004 — corruption operates on `canon(x)`

| | |
|---|---|
| **Status** | CLARIFIED |
| **Proposal** | §5.3 gives `C(x, p, s)` without saying whether `x` is raw or canonical. §4.5 requires `b(x) = b(x̃)` and an invariant base grid. |
| **Implemented** | `corrupt` canonicalises first (`canon`, nucleus-based placement per `orthography.md` D-001), then corrupts. The original string is preserved verbatim in `original_text` and never mutated. |
| **Why** | Without it, `hòa` and `hoà` would be different examples with different identities and different corruption decisions, and the base grid would depend on source spelling. Canonicalising first is what makes the proposal's invariance claim true. |
| **Affected code** | `unmark/corruption/corrupt.py` |
| **Affected experiments** | All. Also makes `VARIANT` (§6.3) a strictly separate condition rather than something that leaks into the others. |
| **Proposal updated** | Yes — §5.3 now says corruption operates on the canonical form. |
| **PDF stale** | Yes. |

### D-B2-005 — the `VARIANT` condition is not implemented in B2

| | |
|---|---|
| **Status** | DEFERRED |
| **Proposal** | §6.3 lists `VARIANT`: "Tone-placement variants (hoà/hòa) and NFC/NFD forms". |
| **Implemented** | `FULL`, `P25`, `P50`, `P75`, `P100`, `STRIP_ALL`. `VARIANT` is registered as recognised-but-unimplemented and `get_condition("VARIANT")` raises with the reason. |
| **Why** | Emitting a placement variant requires `TonePlacement.TRADITIONAL`, which B1A deliberately did not implement (`orthography.md` D-001): implementing it means adopting a second spelling convention the project does not otherwise need. Emitting only the NFD half would misrepresent the condition as specified. |
| **Affected code** | `unmark/corruption/conditions.py` |
| **Affected experiments** | The `VARIANT` row of §6.3 cannot be run until this is closed. Every other condition is unaffected. |
| **Proposal updated** | No — the condition remains specified; only its implementation is deferred. |
| **PDF stale** | n/a |

### D-B2-006 — `selected` and `modified` are reported separately

| | |
|---|---|
| **Status** | CLARIFIED |
| **Proposal** | §4.3: "Under corruption, a *ngang* syllable is *invariant*; a toned syllable transitions to `UNMARKED`." |
| **Implemented** | A selected `ngang` syllable is recorded as `selected=True, modified=False`. `realized_probability` counts selections; `realized_modification_rate` counts actual changes. |
| **Why** | Reporting only one of them would either overstate how much text changed or understate the corruption rate that was applied. Both are needed to interpret a `P50` run whose sentence happens to be mostly `ngang`. |
| **Affected code** | `unmark/corruption/models.py` |
| **Affected experiments** | Reporting of corruption rates. |
| **Proposal updated** | No. |
| **PDF stale** | n/a |

### Explicitly *not* deviations

* **`P100` vs `STRIP_ALL`** — the proposal distinguishes them clearly (§6.3: `P100` = "All tone marks removed"; `STRIP_ALL` = "Tone **and** letter diacritics removed"). Implemented as written, with a test asserting they differ. **No B2 deviation from proposal.**
* **Sampling unit** — §6.3 says syllables; `SyllableSpan` is used. **No B2 deviation from proposal.**
* **Determinism requirement** — §5.3 requires the same key to give the same string; implemented and tested across fresh processes and `PYTHONHASHSEED` values. **No B2 deviation from proposal.**
* **Tone semantics** — §4.3's `UNMARKED`-is-not-`NGANG` rule is enforced by the B1A type split and carried through corruption. **No B2 deviation from proposal.**

Not covered by that list: the corruption *denominator*. See D-B2-003 — the
engine matches the proposal, the eligibility policy does not yet.

---

## Cross-references

| Decision | Where |
|---|---|
| D-001 canonical tone placement (MODERN) | [`orthography.md`](orthography.md) |
| D-002 Vietnamese-candidate eligibility (GAP-2, deferred) | [`orthography.md`](orthography.md) |
| D-003 what `canon` does not normalise | [`orthography.md`](orthography.md) |


---

## B3A — Vietnamese syllable eligibility

### D-B3A-001 — Vietnamese syllable eligibility resolved

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — closes GAP-2 and D-B2-003 |
| **Date** | 2026-08-19 |

**Original proposal.** §4.3: "An alphabetic span is treated as a **Vietnamese
candidate** if it matches the Vietnamese syllable inventory after stripping;
otherwise both channels are `N/A`. […] Ambiguous spans are resolved towards
Vietnamese, and this is documented as a known and deliberate error mode rather
than hidden. One property matters more than the rule's accuracy: it is a pure
function of the *stripped* form."

**Previous temporary state.** No syllable inventory existed in the repository, so
B2 treated every maximal alphabetic run as a candidate. That fallback was
`SELF_CHECK`-only and the scientific path was hard-blocked (D-B2-003, audit 004).

**Final implementation.** Membership of a pinned inventory, tested on the
stripped form:

```text
candidate span -> canon() -> strip_to_base() -> casefold() -> inventory lookup
              -> Eligibility.VIETNAMESE_CANDIDATE | NOT_APPLICABLE
```

`Eligibility.VIETNAMESE_CANDIDATE` was added to the B1A enum. `UNDECIDED` now
means only "the inventory is unavailable", never "resolved as non-Vietnamese".

**Source provenance.**

| | |
|---|---|
| author | `hieuthi` |
| gist | `0f5adb7d3f79e7fb67e0e499004bf558` |
| revision | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| bytes | 116,290 |
| entries | 17,974 raw → 17,954 unique canonical → **2,489 unique stripped forms** (15,465 collisions) |
| license | **NO_EXPLICIT_LICENSE** |

The structural enumeration ("all onsets × all rimes") was chosen deliberately
over the author's separate ~7,184-entry *common* list: a frequency list answers
"which syllables occur often", not "which syllables are legal", and §4.3 asks for
the latter.

**License handling.** No LICENSE file, no license field, no statement in the
description. A public gist is not a licence, so the raw list is **not committed**.
It is fetched into the git-ignored `.resources-cache/` by
`scripts/fetch_vietnamese_syllable_inventory.py`, verified by SHA-256, and
everything scientific fails loudly without it. Only provenance is in git.
Changing the pinned revision is a scientific spec change and must be recorded
here.

**Known error mode — accepted, not fixed.** An English word whose letters form a
valid stripped Vietnamese syllable is classified Vietnamese: `ban`, `the`, `com`,
`on`, `in`, `an`, `la`, `co` are all real stripped syllables. The classifier is
orthographic and structural — no language identification, frequency list,
dictionary, capitalisation heuristic or sentence context is consulted, because
any of those would break the pure-function-of-the-stripped-form property that
guarantees `eligibility(clean) == eligibility(corrupted)`. This is exactly the
trade §4.3 makes: "chosen for determinism, not for correctness".

Words that are not syllable-shaped are correctly rejected: `machine`, `learning`,
`python`, `pytorch`, `café`, `google`, `server`, `email`.

**Affected.**

| Area | Effect |
|---|---|
| B1A eligibility metadata | `SyllableSpan.eligibility` now resolves when a classifier is injected; `UNDECIDED` otherwise. Round-trip and channels unchanged. |
| B2 denominator | Only eligible Vietnamese syllables are scored. `toi dung Python va PyTorch`: 5 candidates → 3 eligible. |
| P25/P50/P75 realized rates | `realized_probability = selected / eligible`. The provisional `candidate_selection_rate` remains available and is clearly distinct. |
| STRIP_ALL mixed language | Ineligible spans are untouched. `café` survives, where the fallback stripped it to `cafe`. |
| Stage-1 corruption | Unblocked: `purpose=SCIENTIFIC` now succeeds when the inventory is present. |
| All future evaluation corruption | Same denominator change; artifacts record `eligibility_schema_version` and inventory provenance. |

**Engine unchanged.** `CORRUPTION_SCHEMA_VERSION` stays `b2-v1`. Scores, keys,
`sample_id` semantics and `unit_index` are untouched — `unit_index` is still the
span's index in the full candidate list, so a given unit's score is bit-identical
to the engine audited in 003/004. Filtering changes *which* units are scored,
never *what score* a unit receives. A separate `ELIGIBILITY_SCHEMA_VERSION =
"vn-syllables-v1"` versions the policy, and both appear in every artifact.

**Proposal updated?** **No.** §4.3 already specifies this rule exactly; the
implementation follows it rather than changing it. Nothing needed clarifying.

| | |
|---|---|
| **Affected code** | `unmark/linguistics/{inventory,classify}.py`, `unmark/orthography/{marks,decompose}.py`, `unmark/corruption/{eligibility,corrupt,models}.py`, `scripts/fetch_vietnamese_syllable_inventory.py`, `scripts/b3a_eligibility_check.py`, `configs/linguistics/vietnamese_syllables.yaml` |
| **PDF stale** | Unchanged by this task; still stale from §4.2 and §5.3 edits. |
