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
| **RESOLVED DECISION** (cont.) | D-B3B0-002 — **CLOSED** by D-B3B0-007 (backbone locked to `vinai/phobert-base` @ `01daacda…`) |
| **RESOLVED DECISION** (cont.) | D-B3B1C-001 (manual alignment validated; tone ownership by candidate count) |
| **RESOLVED DECISION** (cont.) | D-B3B2-001 (deterministic B3B COMPLETE), D-B4A-001, D-B4A-007 |
| **RESOLVED DECISION** (cont.) | D-B4A-002 … D-B4A-007 — all six B4A items, resolved by researcher decision; **B4B unblocked** |
| **RESOLVED DECISION** (cont.) | D-S1A-001 … D-S1A-004 (Stage-1 data path and objective) |
| **OPEN — RESEARCHER DECISION REQUIRED** | D-S1A-005 — `lambda_a`, `lambda_c`, Stage-1 corpus, `max_length`, corruption redraw schedule and every training hyperparameter. **Concrete values are now PROPOSED** in [Audit 028](../audits/028-stage1-scientific-config-review.md); the register stays OPEN until the researcher approves them |
| **RESOLVED DECISION** (cont.) | D-PREG1-015 (pre-G1 CLOSED — primary and secondary), D-S1B-001 (UIT-VSFC excluded from Stage-1 selection), D-B3B0-007 (main backbone locked) |
| **RESOLVED DECISION** (cont.) | D-S1B-002 (Stage-1 corpus `undertheseanlp/UVW-2026` + contamination contract), D-S1B-003 (scope mixture, `pi_strip = 0.25`, stream separation), D-S1B-004 (Stage-1 optimizer/training lock) |
| **BLOCKING STAGE-1 TRAINING — DECIDED, NOT IMPLEMENTED** | Stage-1 corruption gives **STRIP-ALL zero training support** ([Audit 028 §F](../audits/028-stage1-scientific-config-review.md)). Mechanism and value are decided by D-S1B-003; **`scope_for` does not exist yet**, so support is still zero until it is implemented and tested |
| **RESOLVED DECISION** (cont.) | D-S1A-008 (syllable-inventory provenance — **blocking** for scientific training), D-S1A-008a (absent historical diagnostic driver — **non-blocking**), D-S1A-009 (revised roadmap) |
| **RESOLVED DECISION** (cont.) | D-G1-001 (pre-G1 burden diagnostic), D-G1-002 (BASE_ONLY implemented without the adapter), D-G1-003 (GRR reconciled and unclamped), D-G1-005 (Stage-2 pooling stays OPEN) |
| **OPEN — RESEARCHER DECISION REQUIRED** (cont.) | D-G1-004 — every classification-head concrete value for the **full §6 grid**; §5 names these as blocking **G1** |
| **SUPERSEDED** | D-PREG1-001 (SA-VLSP2016), D-PREG1-004 (70/15/15), D-PREG1-008 (`max_length` coverage rule) — all superseded **before any downstream result existed** |
| **RESOLVED DECISION** (cont.) | D-PREG1-001b … D-PREG1-010 — the **active** pre-G1 protocol on **UIT-VSFC v1.0**, including the paired reproducibility lock |
| **RESOLVED DECISION** (cont.) | D-B4B-001 (adapter implemented), D-B4B-003 (torch kept out of the package `__init__`), D-B4B-004 (frozen encoder stays in eval), D-B4B-005 (gradient validation via encoder output) — **all confirmed on real PhoBERT, run `20260820T081554Z`, 27/27** |
| **RESOLVED DECISION** (cont.) | D-B4B-002 — **CLOSED** by the real PhoBERT run: explicit authoritative `position_ids` are required; D-B4B-006 (model provenance verifier repaired) |
| **RESOLVED DECISION** (cont.) | D-B3B0-001 (RAW_BASE selected, closed by D-B3B1A-001) |

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


---

## B3B-0 — PhoBERT input contract

### D-B3B0-001 — PhoBERT expects word-segmented input; the proposal writes `T(b(x))`

| | |
|---|---|
| **Status** | **CLOSED** by [D-B3B1A-001](#d-b3b1a-001--raw_base-selected-as-the-main-unmark-base-path). *Was:* OPEN — empirical feasibility probe required. |
| **Owner** | B3B-0 (probe) → B3B (decision) — both discharged |
| **Date raised** | 2026-08-19 |

**Original proposal wording.** §4.4: "The **base stream** defines the token grid.
All positions are indexed by `T(b(x))`." §4.3 records the base stream as "fully
stripped letters; tokenized by the frozen tokenizer", and §5.1 locks it as
"stripped text, frozen tokenizer, frozen embedding table". §4.4 step 2 assigns
channel labels "by tracking character offsets through tokenization". No section
mentions word segmentation.

**Official PhoBERT input requirement.** PhoBERT's published usage contract states
that input must be Vietnamese **word-segmented** before tokenization —
underscore-joined compounds such as `nghiên_cứu` — produced by the
VnCoreNLP/RDRSegmenter preprocessing used during pretraining. §6.1 itself
describes PhoBERT-base as "word/syllable-level BPE", so the proposal is aware of
the tokenizer's granularity without stating where segmentation happens.

Operationally the pipeline is therefore `T(S(b(x)))`, not `T(b(x))`.

**Why this is a specification question, not a coding detail.** `S` decides what
distribution the frozen encoder sees, and it sits between two things the design
depends on. Four constraints pull against each other:

| Constraint | Source |
|---|---|
| **Deployability** — inference cannot require clean text | §1.3, the whole premise |
| **Corruption invariance** — the base token grid must be identical across conditions | §4.5, §4.4 ("corrupting the input changes the tone labels but never the base ids") |
| **PhoBERT compatibility** — input should match the pretraining distribution | PhoBERT model card |
| **No hidden restoration** — segmentation must not become an implicit diacritic restorer | §3.2 N-claims; a segmenter reading diacritics would leak exactly the signal `RESTORE` is supposed to own |
| **Reproducibility** — the segmenter and its model must be version-pinned | §5.3's determinism requirement |

**Possible operational choices**, enumerated in
`unmark/alignment/contracts.py::PreprocessingPath` and all measured by the probe:

| Path | Description | Principal risk |
|---|---|---|
| `RAW_BASE` | `T(b(x))`, no segmentation | ignores PhoBERT's stated contract; possible distribution mismatch and heavy fragmentation |
| `CLEAN_SEGMENT_THEN_BASE` | segment clean text, then strip | **not deployable** (needs clean text at inference) and a hidden-restoration risk |
| `BASE_THEN_SEGMENT` | strip, then segment the base | deployable, but the segmenter runs out of distribution on undiacritized text |
| `OBSERVED_SEGMENT_THEN_BASE` | segment whatever is observed, then strip | deployable and honest, but segmentation output may vary with corruption level, breaking grid invariance |
| `PRESEGMENTED_DATASET` | use dataset-supplied segmentation | reproducible, but ties the pipeline to per-dataset preprocessing |

**Scientific risks if this is decided wrongly.**

* A path whose token ids differ between `FULL` and `STRIP_ALL` silently violates
  §4.5's central claim, and UNMARK's contribution ("the base grid is invariant by
  construction") would be false.
* A path that segments clean text turns word segmentation into an unmeasured
  restoration step, contaminating the comparison against `RESTORE`.
* A path chosen without pinning the segmenter makes every downstream number
  irreproducible.
* If the tokenizer exposes no usable offsets, §4.4 step 2 is not implementable as
  written and a deterministic manual alignment must be designed — which is B3B's
  work, not something to improvise.

**Affected.** Tokenizer contract; §4.4 alignment; base-grid invariance; Stage-1
self-supervised training; Stage-2 head training; and every PhoBERT-based baseline
(`FLOOR`, `RESTORE`, `ALIGN`, `UPPER`), since they share the backbone.

**Proposal source updated?** **NO.** Deliberately not, until the probe returns
and a policy is locked. Writing any of these paths into §4.4 now would make an
unmeasured guess look normative.

**What closes it.** Run `scripts/b3b0_phobert_input_probe.py` on Colab, read the
per-path grid-invariance and offset-availability columns, choose a path against
the five constraints above, record the choice here, and only then amend §4.4.

### D-B3B0-002 — the first backbone checkpoint is not locked

| | |
|---|---|
| **Status** | **CLOSED** by [D-B3B0-007](#d-b3b0-007--the-main-backbone-is-locked). *Was:* OPEN — SPEC LOCK ITEM. |
| **Owner** | B3B / spec lock |

**Original proposal wording.** §6.1: "At least two […] **PhoBERT-base**
(word/syllable-level BPE, trained on formal diacritized text) and **ViSoBERT**".
§5.1 locks the *architecture* around a frozen encoder but names no repository.

**Gap.** No Hugging Face repository id, no checkpoint revision, and no tokenizer
revision is pinned anywhere — and unlike the `RESTORE` checkpoint, the backbone
is **not** listed in §5's open-items table either. It is therefore neither locked
nor tracked as open, which is the worst of the two states.

**Why it matters.** §5.2 requires baselines to be fixed before any UNMARK number
is seen; the backbone is shared by all five systems, so an unpinned backbone makes
the whole comparison unreproducible. G−1 already established the pattern for
`RESTORE`: repo id plus exact revision, verified at load.

**Implemented now.** `scripts/b3b0_phobert_input_probe.py` defaults to
`vinai/phobert-base` with `--revision` available and unset. That default is a
**probe convenience, not a lock** — the probe records whatever it actually loaded.

**Proposal source updated?** **NO.** Adding a checkpoint to §6.1 would be locking
it by side effect. The researcher should either pin it in §5.1 or add it to §5's
open-items table with the gate it blocks.


### D-B3B0-003 — first Colab probe run invalidated; probe repaired

| | |
|---|---|
| **Status** | **IMPLEMENTATION REPAIR** — not a change to the scientific proposal |
| **Owner** | B3B-0 (done) |
| **Date** | 2026-08-19 |

**Observed on the first real Colab probe run.** Two infrastructure bugs that the
local mock tests could not have caught:

1. **The automatic VnCoreNLP downloader was still in the script.** It called
   `py_vncorenlp.download_model()` before constructing the segmenter, so the
   segmentation model was whatever upstream served that day. The researcher had
   already provisioned a pinned VnCoreNLP v1.2 checkout with recorded SHA-256
   hashes; the probe ignored it and fetched its own.
2. **The relative output root drifted.** `py_vncorenlp.VnCoreNLP(save_dir=…)`
   `chdir()`s into its resource directory, and the run directory was built from a
   relative `--output-root` *after* that call. Artifacts landed in
   `.vncorenlp/results/b3b0/<run_id>/` instead of `<repo>/results/b3b0/<run_id>/`.

**Consequence.** That run is an **INVALID PROBE RUN for scientific
decision-making**: the segmentation resource provenance was not guaranteed, so any
per-path measurement depending on segmentation is unattributable. No conclusion is
drawn from it. Its artifacts are left untouched where they are; `.vncorenlp/` is
now git-ignored so they cannot be committed.

**Resolution.**

* The segmenter resource is now **externally provisioned only**. `download_model()`
  is absent from the script and a test asserts it can never reappear. The probe
  requires `--vncorenlp-dir`, checks that `VnCoreNLP-*.jar`,
  `models/wordsegmenter/vi-vocab` and `models/wordsegmenter/wordsegmenter.rdr`
  exist, computes their SHA-256, and refuses on mismatch against
  `--vncorenlp-hashes`. `pinned=true` is recorded **only** when every observed file
  was verified against a supplied hash; existence alone never qualifies.
* Every path is resolved to an absolute `Path` at process start, before any
  dependency runs. `cwd_at_start`, `cwd_after_segmenter_initialization`,
  `repository_root`, `resolved_output_root`, `resolved_vncorenlp_dir` and
  `cwd_changed_by_dependency` are recorded, so a library moving the cwd is
  reported rather than silently relocating artifacts. No `chdir()` is used.
* `--revision` is now **required**: a floating tokenizer revision fails closed.
  `--allow-floating-revision` permits exploration but stamps
  `scientifically_usable: false`.

**Scope.** Probe infrastructure only. No preprocessing policy was chosen, no
alignment implemented, no scientific semantics touched.

**D-B3B0-001 remains OPEN.** The segmentation question is unanswered and requires
a rerun of the repaired probe.

**Proposal source updated?** **NO.**


### D-B3B0-004 — VnCoreNLP dependency pinned by a committed manifest

| | |
|---|---|
| **Status** | **CLOSED** by [D-B3B0-005](#d-b3b0-005--vncorenlp-provenance-closed-and-revision-verified). *Was:* IMPLEMENTATION REPAIR, blocked on researcher provenance. |
| **Owner** | B3B-0 (code) / researcher (values) — both discharged |
| **Date** | 2026-08-19 |

**Original state.** Audit 007 closed the automatic-download bug but left the
segmenter provenance caller-supplied: `--vncorenlp-hashes` pointed at a
Colab-side file, and the jar was chosen by globbing `VnCoreNLP-*.jar` and taking
the first match (audit 007 N1 and N2).

**Final state.** The dependency is represented by a **committed manifest**,
`configs/linguistics/vncorenlp_v1.2.json`, carrying the schema version, source,
repository, release tag, exact Git revision, the exact required jar
(`VnCoreNLP-1.2.jar`) and the SHA-256 of all three required resources. It is the
canonical `--vncorenlp-manifest` provenance source. The jar is **named, never
discovered**: extra jars in the directory are reported but never substituted, and
a missing required jar fails closed.

**Reason.** Scientific runs must be reproducible from repository configuration,
not from notebook-side scratch state. A provenance file that lives only in a
Colab session disappears with the session.

**Blocked on.** The manifest is committed **incomplete**: `revision` and all three
SHA-256 values read `PENDING_RESEARCHER_PROVENANCE`, and `status` is
`AWAITING_RESEARCHER_PROVENANCE`. Those values exist only in the researcher's
Colab provisioning cells and are **not derivable in this repository** —
`.vncorenlp/` is a git-ignored Colab-side runtime directory. Inventing a digest
would defeat the entire purpose of a pin, so none was invented. The loader
refuses any manifest still containing a placeholder, and the probe cannot mark a
run scientifically usable until the values are filled in.

**Conflict policy.** `--vncorenlp-revision` and `--vncorenlp-hashes` are retained
for compatibility, but a disagreement with the committed manifest is an **error**,
not a precedence question: silently preferring one source would let a stale CLI
value override the repository's pin.

**Notebook scratch state.** `.probe_phobert_revision`,
`.probe_vncorenlp_revision` and `.probe_vncorenlp_hashes.txt` are notebook
cell-to-cell state, **not** scientific configuration. The probe never reads them
(asserted by test) and `.probe_*` is git-ignored. VnCoreNLP provenance comes from
the committed manifest; the tokenizer revision comes from an explicit CLI flag.

**Dependency-change policy.** Changing the release tag, Git revision, jar,
vocabulary, RDR model or any SHA-256 is an **experiment dependency change** and
must be recorded here before any run depends on it.

| | |
|---|---|
| **Affected** | B3B-0 Colab probe; the PhoBERT preprocessing path if segmentation is selected; the pre-training audit |
| **Affected code** | `configs/linguistics/vncorenlp_v1.2.json`, `scripts/b3b0_phobert_input_probe.py`, `unmark/alignment/contracts.py`, `.gitignore` |
| **Proposal updated** | **NO.** This is not a scientific method change. |

**D-B3B0-001 remains OPEN** (segmentation vs `T(b(x))`).
**D-B3B0-002 remains OPEN** (backbone checkpoint not locked).


### D-B3B0-005 — VnCoreNLP provenance closed and revision verified

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — closes D-B3B0-004 and audit 008 N1 |
| **Owner** | B3B-0 — discharged |
| **Date** | 2026-08-19 |

**Original state.** Audit 008 created the committed manifest but could not complete
it: the exact Git revision and the three SHA-256 digests existed only in the
researcher's Colab checkout, were absent from this repository, and were forbidden
to be invented. The manifest shipped with `AWAITING_RESEARCHER_PROVENANCE` and the
loader refused it. Audit 008 also left N1 open: nothing checked that the checkout
was actually *at* the pinned revision.

**Researcher-provided provenance**, extracted from the provisioned checkout with
`git rev-parse HEAD`, `git tag --points-at HEAD` and sha256 over the resources:

| | |
|---|---|
| source repository | `https://github.com/vncorenlp/VnCoreNLP.git` |
| release tag at HEAD | `v1.2` |
| revision | `62bbc58fe5d113c898eae112656be97dcf50b3a0` |
| required jar | `VnCoreNLP-1.2.jar` |
| `VnCoreNLP-1.2.jar` | `9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793` |
| `models/wordsegmenter/vi-vocab` | `0a47c5b55bbce163029d37730a67b9479740388695c29c106c112b815613eaa5` |
| `models/wordsegmenter/wordsegmenter.rdr` | `9e62f96bd93e37a24f364238e8d8ae986fa5dad6dbc9f4eae622ab3651b7fa06` |

**Final implementation.**

* The committed manifest is fully pinned, `status: PINNED`, no placeholder left.
* At run time the probe reads `git rev-parse HEAD` of the provisioned checkout
  (a local subprocess, no network) and compares it to the pinned revision.
  **Mismatch fails closed** — it refuses to load, it does not warn.
* If `.git` metadata is absent, resource digests are still verified,
  `observed_revision` is `None`, `revision_verified` is `false`, and `pinned`
  is `false`. No revision verification is ever fabricated.
* `git tag --points-at HEAD` is recorded as `observed_tags_at_head`, a
  **diagnostic only**: a matching tag never rescues a revision mismatch, which a
  test asserts directly.
* `pinned = hashes_verified AND revision_verified`. Exact-jar selection from
  audit 008 is retained; extra jars are reported, never substituted.
* `scientifically_usable = tokenizer revision supplied AND segmenter pinned`.

**Reason.** B3B-0 preprocessing evidence must be reproducible from committed
repository configuration plus a provisioned checkout — not from notebook-side
scratch state that vanishes with the session.

| | |
|---|---|
| **Affected** | the B3B-0 probe only; the PhoBERT preprocessing pipeline if segmentation is later selected; the pre-training proposal-vs-repository audit |
| **Affected code** | `configs/linguistics/vncorenlp_v1.2.json`, `scripts/b3b0_phobert_input_probe.py`, `unmark/alignment/contracts.py` |
| **Proposal updated** | **NO.** Not a scientific method change. |

**Still open, unchanged by this entry:**

* **D-B3B0-001** — whether PhoBERT's word segmentation belongs in the pipeline
  (`T(b(x))` vs `T(S(b(x)))`). **OPEN.** No preprocessing policy has been selected.
* **D-B3B0-002** — the backbone checkpoint is named but not pinned. **OPEN.**


### D-B3B0-006 — tokenizer revision verified after loading, not merely requested

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — closes audit 009 N3 |
| **Owner** | B3B-0 — discharged |
| **Date** | 2026-08-19 |

**Original state.** Audit 009 required an explicit `--revision` and passed it to
`from_pretrained`, recording it as `revision_pinned`. Nothing checked what
actually loaded. A stale or shared Hugging Face cache, or a checkpoint resolved
by some other path, could have produced measurements attributed to the wrong
tokenizer, and the artifact would have claimed the revision was pinned.

**Final state.** After loading, the probe reads the resolved commit back off the
tokenizer's own files and compares it to the request:

* candidate paths are collected from documented attributes (`vocab_file`,
  `merges_file`, `tokenizer_file`, `name_or_path`) and from `init_kwargs` —
  not from a guessed private field;
* the commit is parsed out of the Hugging Face cache layout
  `models--org--name/snapshots/<commit_sha>/…`. The snapshot directory is always
  the *resolved* commit, so this is genuine post-load evidence rather than a
  restatement of the request;
* no second download and no floating lookup: re-resolving the repository would
  either echo the request or introduce exactly the mutability being guarded
  against;
* contradictory evidence (two different snapshot SHAs) yields **no** observed
  revision rather than a pick.

Fields are separated so none of them can be misread:
`revision_requested`, `revision_observed`, `revision_verified`,
`revision_evidence`, `revision_evidence_source`. The ambiguous `revision_pinned`
— which only ever meant "the CLI argument was present" — is gone.

**Fail-closed.** Observed ≠ requested aborts the run (exit 3). Observed
unrecoverable leaves `revision_verified: false`, which makes the run not
scientifically usable. `--revision` must be a **full 40-character lowercase
commit SHA**: branches, tags and abbreviated SHAs are rejected at argument
validation, because each is mutable or ambiguous.

**Final contract.**

```text
scientifically_usable = tokenizer.revision_verified AND segmenter.pinned
segmenter.pinned      = vncorenlp_revision_verified AND vncorenlp_hashes_verified
```

**Reason.** To make the PhoBERT side of provenance as strict as the VnCoreNLP
side already was after D-B3B0-005. Asymmetric rigour is how a pipeline ends up
trusted for the wrong reasons.

| | |
|---|---|
| **Affected** | B3B-0 probe provenance only; the future B3B backbone lock; the pre-training audit |
| **Affected code** | `scripts/b3b0_phobert_input_probe.py`, `unmark/alignment/contracts.py` |
| **Proposal updated** | **NO.** |

The revision used by the probe is **evidence provenance, not the final paper
checkpoint lock**.

**Still open, unchanged:** **D-B3B0-001** (segmentation vs `T(b(x))`) and
**D-B3B0-002** (backbone checkpoint not locked).


---

## B3B-1A — input path locked, alignment preflight

### D-B3B1A-001 — RAW_BASE selected as the main UNMARK base path

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — closes D-B3B0-001 |
| **Owner** | B3B-1A — discharged |
| **Date** | 2026-08-20 |

**Original proposal wording.** §4.4: "The **base stream** defines the token grid.
All positions are indexed by `T(b(x))`." §4.3: base is "fully stripped letters;
tokenized by the frozen tokenizer". §5.1: "stripped text, frozen tokenizer,
frozen embedding table". No section mentioned word segmentation.

**Discovered PhoBERT requirement.** PhoBERT's published usage contract expects
Vietnamese **word-segmented** input (underscore-joined compounds) from the
VnCoreNLP/RDRSegmenter preprocessing used in pretraining — operationally
`T(S(b(x)))`, not `T(b(x))`. D-B3B0-001 recorded this as an open scientific
question rather than resolving it by assumption.

**Scientific probe evidence.** A provenance-verified Colab run
(`20260820T031644Z`, `scientifically_usable: true`) measured four pipelines over
18 cases × 6 conditions; full record in
[`docs/experiments/b3b0-input-contract-result.md`](../experiments/b3b0-input-contract-result.md).

| Path | Grid invariant | Mean fragmentation | Unknown | Offsets |
|---|---|---:|---:|---|
| `RAW_BASE` | 18/18 | 1.5674 | 12 | ABSENT |
| `BASE_THEN_SEGMENT` | 18/18 | 1.5424 | 12 | ABSENT |
| `CLEAN_SEGMENT_THEN_BASE` | 18/18 | 1.6192 | 12 | ABSENT |
| `OBSERVED_SEGMENT_THEN_BASE` | 9/18 | 1.5763 | 12 | ABSENT |

**Selected policy.**

```text
MAIN UNMARK BASE PATH = RAW_BASE
tokenizer_input = b(x)
token_grid      = T(b(x))
```

No VnCoreNLP segmentation sits between `b(x)` and `T` on the main UNMARK base
stream. This makes the proposal's notation literal.

**Reasons.** Deployable; requires no clean text at inference; exactly
corruption-invariant; introduces no hidden restoration or clean-segmentation side
information; minimal preprocessing; free of post-strip segmenter side effects;
and `BASE_THEN_SEGMENT` offered no compelling empirical benefit (1.5424 vs
1.5674 fragmentation, identical unknown-token count, while recovering little of
the segmentation obtainable from clean text — a diagnostic 8 underscores versus
39).

**Path status.**

| Path | Status |
|---|---|
| `RAW_BASE` | **SELECTED** — main UNMARK base/deployment path |
| `BASE_THEN_SEGMENT` | **NOT SELECTED FOR MAIN METHOD.** Retained only as a possible later ablation/diagnostic; no compute spent on it now. |
| `CLEAN_SEGMENT_THEN_BASE` | **DIAGNOSTIC ONLY.** Non-deployable — requires clean text. May serve as an upper-bound preprocessing diagnostic, never as a system. |
| `OBSERVED_SEGMENT_THEN_BASE` | **REJECTED.** Violates base-token-grid invariance (9/18). |

**PhoBERT preprocessing trade-off, stated plainly.** Standard PhoBERT usage
expects pre-word-segmented Vietnamese. UNMARK **intentionally departs** from that
standard preprocessing on its base branch, because every clean or observed
segmentation alternative conflicts with deployability, invariance, or the
empirical probe. This is a deliberate experiment-design choice and **a possible
source of distribution shift**; it is not claimed that `RAW_BASE` matches
PhoBERT's pretraining preprocessing. Clean-reference and baseline preprocessing
is a separate question, to be locked when those pathways are implemented.

| | |
|---|---|
| **Affected** | Stage-1 self-supervised training input; Stage-2 head input; §4.4 alignment; every PhoBERT-based UNMARK measurement |
| **Affected code** | `unmark/alignment/`, `scripts/b3b1_phobert_alignment_probe.py` |
| **Proposal source updated** | **YES** — §4.4 now states the PhoBERT branch explicitly. |
| **Compiled proposal PDF stale** | **YES** |

### D-B3B1A-002 — eligibility metadata was not resolved in the B3B-0 probe

| | |
|---|---|
| **Status** | **IMPLEMENTATION REPAIR** |
| **Owner** | B3B-1A — discharged |

**Observed.** Every one of the 432 artifact observations recorded
`Eligibility.UNDECIDED` (3960 labels; zero `VIETNAMESE_CANDIDATE`, zero
`NOT_APPLICABLE`), contradicting B3A's resolved semantics.

**Actual cause**, reproduced locally rather than assumed: `DEFAULT_MANIFEST` in
`unmark/linguistics/inventory.py` was the *relative* string
`configs/linguistics/vietnamese_syllables.yaml`. `py_vncorenlp.VnCoreNLP()`
`chdir()`s into its resource directory before the probe's case loop, so
`try_load_inventory()` resolved the manifest against the wrong directory,
returned `None`, and no classifier was injected. The same class of bug as audit
007's output-path drift — fixed there for outputs, still latent in the library's
own default.

**Consequence beyond the labels.** The probe's B2 corruption also failed to load
the inventory and therefore ran under the provisional candidate-span policy.
That affects only `OBSERVED_SEGMENT_THEN_BASE`, whose input is corrupted text;
the other three paths tokenize `base_text` or clean text, which is invariant
under both corruption and the eligibility policy. The path decision above is
therefore unaffected, and this is recorded in the experiment record rather than
glossed.

**Fix.** `DEFAULT_MANIFEST` is now anchored to the repository via `__file__`, so
inventory loading — and with it `corrupt()`'s policy resolution — no longer
depends on the caller's working directory. **B3A's scientific eligibility
semantics are unchanged.** The B3B-1 probe additionally calls `load_inventory()`
rather than `try_load_inventory()`, so a missing inventory fails loudly instead
of silently degrading to `UNDECIDED`.

| | |
|---|---|
| **Affected code** | `unmark/linguistics/inventory.py`, `scripts/b3b1_phobert_alignment_probe.py` |
| **Proposal updated** | **NO** |

### D-B3B1A-003 — offsets are absent; manual alignment is the hypothesis

| | |
|---|---|
| **Status** | **SUPERSEDED** by [D-B3B1B-001](#d-b3b1b-001--alignment-runs-over-whitespace-chunks-not-linguistic-spans). The hypothesis was tested and **refuted** at span granularity. |
| **Owner** | B3B-1 |

The scientific run found `offset_availability = ABSENT` for every path: the
authoritative tokenizer is `PhobertTokenizer` (`is_fast = false`) and returns no
`offset_mapping`. Proposal §4.4 step 2 propagates channel labels "by tracking
character offsets through tokenization", which is therefore not implementable as
written for this tokenizer.

**The token-grid authority does not move.** The frozen token ids produced by the
pinned slow tokenizer remain authoritative; no scientific path switches to a fast
implementation merely because it offers offsets.

**Hypothesis to test:** tokenizing each base span independently and stripping the
fastBPE `@@` continuation marker reconstructs the span's exact surface, yielding
deterministic half-open character ranges per piece. Implemented in
`unmark/alignment/manual.py`; **not validated** until
`scripts/b3b1_phobert_alignment_probe.py` runs on Colab against the real
tokenizer.

**Failure policy.** An eligible Vietnamese syllable that produces `<unk>`, or
whose pieces do not reconstruct its surface, is an explicit `ALIGNMENT_FAILURE`
with a reason. Special tokens, punctuation and non-Vietnamese spans are `N/A` in
both channels. `UNDECIDED` eligibility fails rather than being labelled. No span
is ever labelled on a guess.


---

## B3B-1B — whitespace-chunk alignment

### D-B3B1B-001 — alignment runs over whitespace chunks, not linguistic spans

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation); real-tokenizer validation **OPEN** pending the corrected probe |
| **Owner** | B3B-1B |
| **Date** | 2026-08-20 |

**Original B3B-1A hypothesis.** Tokenize each B1A/B3A linguistic span
independently with the frozen slow tokenizer and compose the pieces to recover a
character map.

**Empirical result.** The first real B3B-1 Colab run reported **6/13**
full-sequence agreement, despite every span reconstructing its own surface
exactly. 2488/2489 inventory forms "aligned"; the run was correctly marked
`B3B1_ALIGNMENT_PROBE_INCOMPLETE`.

**Root cause.** PhoBERT's fastBPE operates over **maximal non-whitespace
chunks**, not over linguistic spans. Punctuation, hyphens, URLs and e-mail
addresses change the BPE segmentation of the whole chunk they sit in:

```text
"nhien."   authoritative ["nhi@@", "en@@", "."]   span-composed ["nh@@","ien"]+["."]
"VNU-HCM"  authoritative ["VN@@", "U-@@", "HCM"]
"(VAT"     authoritative ["(@@", "VAT"]
"Viet-Nam" authoritative ["Viet@@", "-@@", "Nam"]
```

URLs and e-mail addresses reconstructed exactly from their raw pieces when taken
as whole chunks.

**Follow-up result.** Splitting each sentence's `base_text` on `\S+` and
tokenizing each whole chunk gave, across the 13 representative sentences:

| | |
|---|---:|
| token sequence matches | **13/13** |
| token-ID matches | **13/13** |
| non-whitespace chunks | 119 |
| chunk surface reconstruction failures | **0** |

The 6/13 result was **wrong granularity**, not an inability to reconstruct
PhoBERT's tokenization.

**Final implementation.**

```text
authoritative token grid = T(b(x))          <- never defined by the alignment code
auxiliary character map  = per-chunk raw-BPE reconstruction, then overlaid with
                           B1A/B3A orthographic spans by character-range overlap
```

Take `b(x)`; split into maximal non-whitespace chunks with exact global ranges;
tokenize each **whole chunk**; use **raw** BPE pieces before any id round trip;
reconstruct and require exact surface equality; derive local then global
half-open ranges; compose and verify against the authoritative tokens **and**
ids; overlay the orthographic spans.

**Linguistic spans are orthographic metadata boundaries, not tokenization
boundaries.** A BPE piece may cover part of a span, several regions, or
punctuation adjacent to text, so attribution is by character-range overlap.

| | |
|---|---|
| **Affected** | §4.4 alignment; Stage-1 and Stage-2 input construction; every later channel-propagation step |
| **Affected code** | `unmark/alignment/manual.py`, `scripts/b3b1_phobert_alignment_probe.py` |
| **Proposal source updated** | **YES** — §4.4 step 2 no longer implies native offsets. |
| **Compiled PDF stale** | **YES** |

**Still open:** validation on the real tokenizer, pending the corrected probe.
**D-B3B0-001 remains CLOSED** (`RAW_BASE`). **D-B3B0-002 remains OPEN** (final
backbone lock).

### D-B3B1B-002 — vocabulary OOV is not alignment failure; mixed pieces stay open

| | |
|---|---|
| **Status** | **RESOLVED** for the OOV policy; **OPEN** for mixed-contributor tone assignment |
| **Owner** | B3B-1B (OOV) / B3B (mixed pieces) |

**OOV finding.** The real tokenizer on `khut`:

```text
tokenizer.tokenize("khut")     -> ["khut"]      raw surface recoverable
tokenizer.bpe("khut")          -> "khut"
raw ids                        -> [3]           the unknown id
convert_ids_to_tokens([3])     -> ["<unk>"]     surface destroyed by the round trip
surface reconstruction exact   -> TRUE
```

B3B-1A conflated **unknown vocabulary id** with **unknown alignment surface** and
reported a failure. They are now separate: such a piece is `ALIGNED` with
`has_unknown_token_id = True`, and may carry orthography channels when the
intersecting orthographic region is resolved and valid. This is a **general
policy**; the string `khut` is nowhere special-cased. `AlignmentFailureReason`
no longer contains `UNKNOWN_TOKEN`.

True alignment failure is only: raw-surface reconstruction mismatch, malformed
continuation, impossible or non-monotonic ranges, an unexplained authoritative
token, or unresolved eligibility where a scientific channel assignment is needed.

**Mixed-contributor pieces — CLOSED** by
[D-B3B1C-001](#d-b3b1c-001--manual-alignment-is-validated-tone-ownership-is-decided-by-candidate-count).
The paragraph below records what was open at the time of writing.

**Mixed-contributor pieces — OPEN (superseded).** A BPE piece can straddle a Vietnamese
candidate span and punctuation or non-Vietnamese text within one chunk. The
alignment records **every** contributing region with its exact overlap range and
marks the piece `ToneOwnership.MIXED`; it does **not** claim the token is
Vietnamese. A deterministic tone-assignment rule for such pieces is **not**
decided here — the evidence does not yet support one, and guessing would attach
a tone label to characters that did not produce it. The corrected probe reports
how often this occurs; B3B decides the rule.

| | |
|---|---|
| **Proposal updated** | **NO** for this entry |

---

## B3B-1C — alignment validated; channel projection

### D-B3B1C-001 — manual alignment is validated; tone ownership is decided by candidate count

| | |
|---|---|
| **Status** | **RESOLVED DECISION**. Closes the validation gap left open by [D-B3B1B-001](#d-b3b1b-001--alignment-runs-over-whitespace-chunks-not-linguistic-spans) and the mixed-piece question left open by [D-B3B1B-002](#d-b3b1b-002--vocabulary-oov-is-not-alignment-failure-mixed-pieces-stay-open). |
| **Owner** | B3B-1C |
| **Evidence** | [`docs/experiments/b3b1-manual-alignment-result.md`](../experiments/b3b1-manual-alignment-result.md), run `20260820T035339Z` |

**Part 1 — alignment is a validated component, not a hypothesis.** The corrected
probe aligned **2,489 / 2,489** sentences against the real pinned
`PhobertTokenizer`, with 13/13 token-sequence consistency, 13/13 token-**id**
consistency, 119/119 exact chunk reconstructions, 42/42 curated sentences and
**0** `UNDECIDED` eligibility labels. Whitespace-chunk manual alignment is
therefore adopted as the B3B channel-propagation mechanism. Proposal §4.4 step 2
("tracking character offsets through tokenization") is satisfied by exact
reconstructed ranges rather than by `offset_mapping`, which this tokenizer does
not provide.

**Part 2 — tone ownership is decided by counting distinct candidate
contributors**, replacing the conservative audit-012 rule:

| Distinct Vietnamese candidates overlapping the piece | Tone |
|---|---|
| 0 | `NA` — `NOT_APPLICABLE` |
| exactly 1 | **that candidate's observed tone**, even when the piece also covers punctuation or non-Vietnamese characters |
| ≥ 2 | `NA` — `MULTI_CANDIDATE_AMBIGUOUS`, every contributor recorded |

*Why the change.* Of 191 overlays, 2 were mixed and **both** mixed a single
candidate with punctuation (`en` + `-`; `.` + `com`). **Zero** pieces spanned two
distinct candidates. The old rule discarded information that was never
ambiguous: when only one candidate is present, nothing competes to own the tone.

*What is never done.* A multi-candidate piece is never resolved — not by
majority overlap length, not by first or last contributor, and never by
averaging categorical tone ids. This holds **even when the candidates carry the
same tone**: sharing a value is not the same as having one source, and a
"they agree, so take it" rule has no principled extension to the disagreeing
case. No URL, e-mail, or literal piece is special-cased anywhere.

**Part 3 — deterministic channel projection** (`unmark/alignment/channels.py`).
Pure data: no torch, no trainable parameters, no model weights, no pooling.

* `TokenToneLabel` projects `ObservedTone` and adds a token-level `NA` that no
  syllable can have. Lexical `NGANG` is **not** in it — the deploy pathway keeps
  an unmarked syllable ambiguous, and `UNMARKED` ≠ `NA`.
* The letter channel publishes every contributing character with its label.
  `NONE` is an applicable contributor ("a letter that could carry a Vietnamese
  letter diacritic and does not"); only `NA` is excluded. A token with zero
  applicable contributors has letter channel `NA`.
* **Letter pooling rule, recorded not implemented:** arithmetic mean *in
  embedding space* over the applicable contributors, `NONE` included, `NA`
  excluded. Persisted as `LETTER_POOLING_RULE` so the adapter implements the
  decided rule rather than re-deciding it.
* Special tokens carry `NA` in both channels and **no fabricated source range**.
  Whitespace never becomes a model token.
* Character structure has exactly one source of truth: labels are read from the
  canonical `unmark.orthography` decomposition, never re-derived from a BPE
  token string, and no second Unicode implementation exists in the alignment
  package. Enforced by AST tests.

**Corruption invariance.** `b(x)` is invariant under all six B2 conditions
(FULL / P25 / P50 / P75 / P100 / STRIP_ALL), so one token grid and one set of
character ranges serve every condition; only the channel *values* degrade.
Verified locally against the real B2 engine.

**Scope.** This entry does **not** lock the paper's backbone checkpoint. The
probe pinned a revision for reproducibility; [D-B3B0-002](#d-b3b0-002) stays
open.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.4 is satisfied as written; the offset mechanism is an implementation detail already recorded in D-B3B1A-003. PDF stale: **YES** (unchanged from B3B-1B) |

---

## B3B-2 — deterministic B3B closure

### D-B3B2-001 — deterministic B3B is COMPLETE

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | B3B-2 |
| **Evidence** | [`docs/experiments/b3b2-channel-projection-result.md`](../experiments/b3b2-channel-projection-result.md), run `20260820T041812Z`, HEAD `c09516e03300e670fc20ac10173d7c346106fd6a`, status `B3B2_CHANNEL_PROJECTION_COMPLETE`; and [`b3b1-manual-alignment-result.md`](../experiments/b3b1-manual-alignment-result.md), run `20260820T035339Z` |

**Deterministic B3B: COMPLETE.** Both closing probes ran against the real pinned
slow tokenizer, with no model weights loaded and nothing trained.

**The validated structural contract.** For the locked corruption conditions,

```
b(C_c(x)) = b(x)      hence      T(b(C_c(x))) = T(b(x))
```

and the deterministic raw-BPE character-range structure is identical. **Only
orthographic channel values change.** Measured across 7 cases × 6 conditions:
42/42 token-id matches against the per-case `FULL` authoritative grid, 42/42
piece-range matches, 42/42 sequence consistency, and 0 multi-candidate pieces.

**The locked contract.**

| Element | Locked value |
|---|---|
| Input path | `RAW_BASE` — no post-strip word segmenter |
| Authoritative grid | `T(b(x))`, from the frozen slow tokenizer |
| Tokenizer authority | slow `PhobertTokenizer`; alignment metadata **never** defines or changes token ids |
| Character map | deterministic raw-BPE range reconstruction over maximal non-whitespace chunks |
| Channel metadata | B1A orthography + B3A eligibility overlay |
| Tone ownership | unique-candidate rule (D-B3B1C-001) |
| Ambiguity | ≥ 2 distinct candidates → tone `NA`, contributors recorded, never resolved |
| Letter channel | `NONE` **included**, `NA` **excluded**, mean in embedding space |
| Structural invariance | grid and ranges corruption-invariant; channel values alone degrade |

**Scope of the monotonic result.** Monotonic marked-tone degradation is an
**observed** result under the locked deterministic B2 protocol and these seven
probe cases — **not** a universal theorem about arbitrary corruption processes.
The structural equality above is the load-bearing invariant; it holds by
construction.

**Special-token integration** is an integration test for B4/B5, **not** an open
B3B scientific blocker. B3B-2 operated on ordinary tokenizer positions.

**D-B3B0-001: CLOSED.** **[D-B3B0-002](#d-b3b0-002): REMAINS OPEN** — the probe
revision is reproducibility evidence, not the final backbone decision.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

---

## B4A — neural adapter contract preflight

Full extraction: [`docs/spec/neural-adapter.md`](neural-adapter.md).
**No `nn.Module` was written; torch is not installed.**

### D-B4A-001 — the gate is a projection, not a vector; fusion is convex, not residual

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — proposal over project history |
| **Owner** | B4A |

Project history summarised the adapter as "a 3d → d fusion projection; LayerNorm;
a per-dimension gate", suggesting a raw trainable `g ∈ R^d` and a residual form
`e + g ⊙ LN(W[e;t;l] + b)`. **The proposal specifies neither.** §4.5:

```
f_i = LN( W_f [ e_i ; t_i ; l_i ] + c_f )
g_i = σ( W_g [ e_i ; t_i ; l_i ] + c_g ) ∈ (0,1)^d
z_i = g_i ⊙ f_i + (1 − g_i) ⊙ e_i
```

The gate is a **second `3d → d` projection** followed by `σ`, input-dependent per
position; §5.1's "per-dimension" describes its **output shape**, not a
position-independent parameter. Combination is **convex**, not residual.

**Decided in favour of the proposal**, on three independent grounds: §4.5 states
it; §5.1 locks `σ(W_g[·])`; and §4.7's budget bills a *"Gate projection (3d → d)
≈ 1.8M"*, which a `d`-sized vector cannot account for. The derived formula
`|φ| = 6d² + (4 + n_τ + n_λ)d` reproduces §4.7's ≈3.6M exactly at `d = 768`,
`n_τ = 7`, `n_λ = 10` — **only** under the projection reading.

The forms are not interchangeable: under the convex form `g → 1` *replaces* the
base embedding; under the residual form the base term is always at full strength.

### D-B4A-002 — tone `NA` is a fixed zero vector outside the 7-slot table

| | |
|---|---|
| **Status** | **RESOLVED DECISION.** Was OPEN and blocking when Audit 014 was written. |
| **Owner** | B4A |
| **Resolved** | 2026-08-20, researcher decision |

**RESOLUTION: `NA` is not a trainable table row. It is the fixed zero vector.**

```
t_i = 0 in R^d
```

at non-Vietnamese positions, special tokens, padding, and multi-candidate
ambiguous pieces. The tone table stays at **exactly 7 trainable rows for every H4
policy**. `UNMARKED` remains a genuine learned observable row, distinct from
`NA`. No learned `NA` row; no eighth row.

**Reason.** `NA` is *structural non-applicability*, not an observable tone state.
Reading (c) below was the only one preserving both the §5.1 7-slot lock and the
H4 equalization; the decision adopts it and makes it explicit rather than
implicit.

**Mathematical consequence.** `NA` costs zero parameters, so `|φ|` is identical
across all three policies and the H4 fairness argument holds exactly. At an `NA`
position the fusion still receives `[e_i ; 0 ; l_i]` — the position is **not
inert**, it is fused with a zero tone contribution. Under `OBSERVABLE` the 7-row
table is still allocated in full even though slot B is never indexed; dropping
the unused row would itself break the equalization.

**Batching contract.** `tone_ids: [B, L]` with valid rows `0..6`, plus
`tone_mask: [B, L]`. `NA` may be carried as an out-of-table sentinel (`-1`).
**A torch implementation must never feed that sentinel to `nn.Embedding`** — it
must use a safe placeholder lookup plus masking, or an equivalent safe mechanism,
and force the result to exact zero.

**Affects.** `unmark/modeling/contracts.py` (`TONE_NA_SENTINEL`,
`TONE_NA_IS_ZERO_VECTOR`, `ToneChannelContract.require_locked`); the B4B tone
embedding lookup; the H4 parameter-equality test. The rejected alternatives are
retained in `ToneNaTreatment` and rejected **by name**, so a later reader sees
them refused rather than rediscovering them as plausible.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.3's 7 slots were already correct; the gap was that `N/A` had no stated representation, which is now recorded here. PDF stale: **YES** |

---

**The original ambiguity, preserved.** §4.3 and §5.1 lock **7 tone slots = 5
marked + 2 policy slots**. §4.3's table assigns slot A and slot B per H4 policy — for `OBSERVABLE`,
slot A = `UNMARKED` and slot B is *unused*; for `ORACLE`, slot A = *ngang* and
slot B = `MISSING`. §4.3 and §4.4 separately require non-Vietnamese subwords to
carry `N/A` in both channels. **No slot is allocated to `N/A`.**

The repository's `TokenToneLabel` also has 7 members, but a different seven:
`5 marked + UNMARKED + NA`. The counts coincide; the compositions do not.

**Competing reasonable choices.**

| Option | Consequence |
|---|---|
| **(a)** `NA` takes unused slot B | Works for `OBSERVABLE`/`FORCED-NGANG`; **breaks `ORACLE`**, which needs slot B for `MISSING` |
| **(b)** `NA` becomes an 8th row | Contradicts the §5.1 7-slot lock; gives `ORACLE` 8 rows against 7 — **defeats the H4 equalization**, whose stated purpose is to remove "any objection that the oracle was granted extra capacity" |
| **(c)** `NA` is not a row: fixed zero vector or masked contribution | Preserves 7 rows **and** H4 equalization for all three policies. The proposal never states it |

**Mathematical consequence.** Under (c) with a zero vector, `t_i = 0` at
non-Vietnamese positions, so `[e_i ; 0 ; l_i]` still passes through `W_f`/`W_g`
and `c_f`/`c_g` — the position is *not* inert, it is fused with a zero tone
contribution. Under (a)/(b) the position gets a **learned** vector, and the model
can allocate capacity to "this is not Vietnamese". These are materially different
inductive biases and different parameter counts.

**This also subsumed the general `NA`-embedding question** for both channels:
learned row, fixed zero, or masked. It **blocked B4B** at the time — no tensor
could be built without it. *(Resolved above: fixed zero vector.)*

### D-B4A-003 — gate initialisation: `W_g = 0`, `c_g = logit(0.01)`

| | |
|---|---|
| **Status** | **RESOLVED DECISION.** Was OPEN and blocking when Audit 014 was written. |
| **Owner** | B4A |
| **Resolved** | 2026-08-20, researcher decision |

**RESOLUTION.**

```
W_g = 0
c_g = logit(0.01) = ln(0.01 / 0.99) ~= -4.59511985013459
```

for every output dimension, so at initialisation `g_i = 0.01` for every token and
every dimension, before any learning. The locked sigmoid-gate architecture is
unchanged.

**Reason.** Start close to the pretrained base-only pathway; avoid `c_g = 0 →
g = 0.5`, which would inject a **randomly initialised** fusion branch at half
weight on step zero; retain a nonzero sigmoid derivative so the gate projection
can still learn.

**Mathematical consequence.** `W_g = 0` makes the gate input-independent at step
zero — the concatenated channels drop out and every position starts at `σ(c_g)`.
The derivative `g(1−g) ≈ 0.0099` stays usable; an initialisation driving `g` to
machine zero would drive the derivative there too and the gate could never open.
**No exact base-only equality is claimed**: `g = 0.01` is close to that pathway,
not equal to it (see D-B4A-004).

**Affects.** `GATE_INIT_WEIGHT`, `GATE_INIT_BIAS`, `GATE_INIT_TARGET`,
`GateContract.initial_gate_value`, `AdapterConfig.initialisation_plan()`; B4B's
module construction, which must log this explicitly; G1's "force the gate towards
identity" criterion, which this initialisation now starts near.

| | |
|---|---|
| **Proposal updated** | **NO** — the proposal locks the transform and is silent on initialisation; recorded here. PDF stale: **YES** |

---

**The original ambiguity, preserved.** The gate **transform** is locked (`σ`).
Its **initialisation** appears nowhere in the proposal.

| Option | Consequence at step 0 |
|---|---|
| `c_g = 0`, `W_g` standard | `g ≈ 0.5` — the adapter starts **halfway**, immediately perturbing every embedding the frozen encoder sees |
| `c_g` strongly negative | `g ≈ 0` — training starts near the **base-only pathway** and must learn to open the gate |
| `c_g` strongly positive | `g ≈ 1` — starts at **full fusion**, base term suppressed |

**Mathematical consequence.** G1's pass criterion is "attach the fusion layer,
train briefly, **force the gate towards identity**, evaluate on `FULL`, pass
within ≈1 point". A `g ≈ 0.5` initialisation starts far from that operating
point; a negative-bias initialisation starts at it. The choice materially affects
whether G1 measures the architecture or the initialisation. It **blocked B4B**
at the time. *(Resolved above: `W_g = 0`, `c_g = logit(0.01)`, initial `g = 0.01`
— the negative-bias family, made exact.)*

Note the interaction with D-B4A-004: "initialised at zero" and "initialised so
its *effect* is zero" are different requests, and under `σ` **only the first is
achievable** — as a bias value, not as a gate output.

### D-B4A-004 — gate-zero recovery is a limit, not an attainable value

| | |
|---|---|
| **Status** | **RESOLVED DECISION.** Was OPEN (test design) when Audit 014 was written. |
| **Owner** | B4A |
| **Resolved** | 2026-08-20, researcher decision |

**RESOLUTION: forced `g := 0` is a wiring test only.** Under an explicit test
override, `z == e` must hold **exactly**, up to ordinary floating-point
arithmetic. It is **not** a trainable parameterization, **not** an experiment
condition, **not** a claim that `σ` attains zero, and **not** evidence that the
initialised module is identity.

**No casual production "gate zero mode" may be exposed**, since such a mode could
silently enter an experiment. B4B should test the fusion-combination primitive or
internal path directly rather than adding a public flag.

**Separately, B4B must measure the real initialised gate at `g = 0.01` against
the base-only pathway and report the difference — expected to be nonzero.**
Reporting zero there would mean something is wrong.

**Mathematical consequence.** The wiring identity is exact and trivial:
`z = g·f + (1−g)·e` at `g = 0` gives `z = e` for any `f`. That is what the
override checks — the plumbing, not the parameterization.

**Affects.** B4B's test suite; `GATE_ZERO_IS_WIRING_TEST_ONLY`. A test asserts no
public gate-zero flag exists on the config.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** |

---

**The original finding, preserved.**

**Confirmed against the proposal:** `g → 0` recovers the **base-only pathway**
(same authoritative `T(b(x))` ids, same special tokens, same attention mask, same
frozen encoder, no tone/letter contribution) — **not** the clean-text pathway
`E_θ(T(x))`. §4.5 corrects an earlier false claim to exactly this effect.

**The exact numerical condition.** `z_i = e_i` requires, per position and
dimension, `g_i[k] · (f_i[k] − e_i[k]) = 0` — so `g_i[k] = 0` **or**
`f_i[k] = e_i[k]`. **Neither is attainable:**

1. `g = σ(·) ∈ (0,1)^d`, an **open** interval; `σ(u) = 0` only as `u → −∞`.
2. `f = LN(·)` is normalised per position; `e_i = Emb_θ(b_i)` is not. Equality
   for all `i` would require the LN affine parameters to invert normalisation for
   every token at once.

**This is not an inconsistency in the proposal**, which writes `g_i → 0` — a
limit — and never claims exact recovery. It is a consequence for the test plan:
the "gate-zero numerical equivalence" check cannot be an exact equality test on
the trained parameterization.

| Option | What it tests |
|---|---|
| Forced override `g := 0` on a test-only path | the **wiring** — that `z = e` reproduces base-only bit-for-bit. Exact, but proves plumbing, not the parameterization |
| Tolerance test, `c_g` driven very negative | approach to the limit. Requires choosing a tolerance |
| Reparameterize the gate to attain `0` exactly | **changes the locked architecture** — researcher's call |

It was **not decided** at the time, and did not block implementation, the
equation being fully specified. *(Resolved above: the forced-override wiring
test.)*

### D-B4A-005 — the empty letter channel is the exact zero vector

| | |
|---|---|
| **Status** | **RESOLVED DECISION.** Was OPEN and blocking when Audit 014 was written. |
| **Owner** | B4A |
| **Resolved** | 2026-08-20, researcher decision |

**RESOLUTION: the exact zero vector.** For a token with applicable contributor
set `A_i`:

```
|A_i| > 0   ->   l_i = (1 / |A_i|) * sum_{j in A_i} W_lambda[label_ij]
|A_i| = 0   ->   l_i = 0 in R^d
```

`NONE` is included in the arithmetic mean; `NA` contributors are excluded;
`NA` is **not** a trainable letter row.

**Reason.** Symmetric with D-B4A-002: non-applicability is structural, not a
learned state. It also keeps `n_lambda = 5` (D-B4A-007).

**Mathematical consequence.** The implementation must **explicitly prevent
`0/0`**. A torch implementation may clamp the denominator for vectorisation, but
**only if the zero-contributor output is then explicitly forced to exact zero** —
clamping alone leaves `sum(empty)/1 = 0` true by accident rather than by
contract, and the accident stops holding the moment the numerator stops being
empty-safe. B3B-2 recorded 25 `NA` positions per condition, so real batches
exercise this on every step; an unguarded `NaN` would poison the batch silently.

**Affects.** `LETTER_EMPTY_IS_ZERO_VECTOR`, `LETTER_NA_SENTINEL`,
`LetterChannelContract.require_locked`; B4B's pooling implementation.
`MASKED_OUT` is rejected by name — it changes the concatenation *width* and is
incompatible with a fixed `W_f in R^(d x 3d)`.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.4 specifies the pooling, not the empty case. PDF stale: **YES** |

---

**The original ambiguity, preserved.** D-B3B1C-001 locks the *label-space*
semantics: zero applicable contributors → token-level letter channel `NA`. It does **not** say what vector `l_i ∈ R^d` the
fusion receives there, and the proposal does not either.

| Option | Consequence |
|---|---|
| Fixed zero vector | `l_i = 0`; position still fused via `W_f`, `c_f` |
| Learned `NA` row of `W_λ` | model can represent "no applicable letter"; couples to D-B4A-002 |
| Masked out of the concatenation | changes the input **width** at those positions — incompatible with a fixed `W_f ∈ R^(d×3d)` |

**Mathematical consequence.** A masked mean over zero applicable contributors is
`0/0`. Left unguarded it yields `NaN` and silently poisons the batch — the same
class of silent failure §4.5 warns about for double position encoding. B3B-2
recorded 25 `NA` tone positions per condition, so **this path is exercised by
real data on every batch**, not a rare edge case. It **blocked B4B** at the time.
*(Resolved above: the exact zero vector, with `0/0` explicitly prevented.)*

### D-B4A-006 — Stage-1 pooling: attention-masked mean over non-special content

| | |
|---|---|
| **Status** | **RESOLVED DECISION.** Was OPEN when Audit 014 was written. |
| **Owner** | B4A |
| **Resolved** | 2026-08-20, researcher decision |

**RESOLUTION: attention-masked mean over non-special content tokens**, from the
final encoder hidden state. For `H in R^[B, L, d]`:

```
m_i = attention_mask_i * (1 - special_tokens_mask_i)

h   = sum_i m_i H_i / sum_i m_i
```

computed **independently for each branch**. Excludes `<s>`, `</s>`, `<pad>` and
every other tokenizer/model special token.

**An example with zero content positions after masking FAILS LOUD.** No silent
fallback to `<s>`, to an unmasked mean, or to a zero vector.

**Reason.** The proposal locks pooled-representation alignment but not a
particular pool. Mean pooling is defined across **unequal branch lengths**, which
is the actual situation. Excluding padding prevents a bias that varies with batch
composition. Excluding special tokens prevents the alignment objective from
receiving an artificially easy shared signal: those positions are near-invariant
between branches, so a cosine objective including them partly measures agreement
that was never in question.

**Mathematical consequence.** The two branches map to `R^d` independently and **no
per-token correspondence is assumed** — consistent with §4.6 deferring per-token
alignment. A zero denominator is an error rather than a defined value.

**Affects.** `Stage1PoolingContract`, `Stage1PoolingError`; Stage-1 training when
it is implemented; the `L_align` / `L_clean` objective inputs.

| | |
|---|---|
| **Proposal updated** | **YES** — §4.6 in v1.4 now states the pooling, because this is a scientific decision rather than an implementation detail. PDF stale: **YES** |

---

**The original ambiguity, preserved.** §4.6 locked *"pooled representations only,
with `D` the cosine distance"* but never said **which** pooling. §5.2 lists head
"pooling" among values pinned during spec lock, and §13 item 4 left it open.

Options were: `CLS`/`<s>` vector; mean over all positions; **attention-masked**
mean. They differ materially under padding — an unmasked mean averages `<pad>`
embeddings into the representation, which for a cosine objective is a silent
systematic bias that varies with batch composition.

It was marked open rather than chosen. A structural constraint on any choice: the
reference branch `h(x)` and the adapted branch `h′(·)` do **not** share a
sequence length (§4.6 defers per-token alignment for exactly this reason), so the
pooling must map both to `R^d` independently. *(Resolved above: attention-masked
mean over non-special content tokens.)*

### D-B4A-007 — letter-table cardinality, and the superseded §8.2 sketch

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — confirmed by researcher decision 2026-08-20 |
| **Owner** | B4A |

**RESOLUTION: `n_lambda = 5`.** The trainable letter table contains exactly
`NONE`, `BREVE`, `CIRCUMFLEX`, `HORN`, `STROKE`. `NA` is not a row.

**Updated symbolic parameter count.** With `n_tau = 7` and `n_lambda = 5`:

```
|phi| = 6 d^2 + (4 + n_tau + n_lambda) d
      = 6 d^2 + 16 d
```

For arithmetic sanity only, at `d = 768`: **3,551,232**. **`d = 768` is not
locked** — D-B3B0-002 remains OPEN and `hidden_size` has no default.

**Affects.** `LETTER_TABLE_ROWS`, `PARAMETER_FORMULA`,
`AdapterConfig.parameter_count()`; proposal §4.7 and §8.2, corrected in v1.4. The
orthographic taxonomy is **unchanged**.

| | |
|---|---|
| **Proposal updated** | **YES** — §4.7's budget line becomes `5 × d` ≈ 4K and §8.2's sketch becomes `n_letter=5`, both narrow corrections of stale estimates. PDF stale: **YES** |

---

**The original ambiguity, preserved.**

**Cardinality.** §4.7 bills *"~10 × d"* and §8.2 defaults `n_letter=10`; §4.3
writes `{NONE, breve, circumflex, horn, stroke, circumflex+…}`. B1A determined the
applicable closed set is **5** (`NONE, BREVE, CIRCUMFLEX, HORN, STROKE`) plus
`NA`. The anticipated `circumflex+…` combinations do not arise: Vietnamese places
**at most one** letter-forming mark per character (`ă â ê ô ơ ư`, `đ` stroke).
`~10` is a budget estimate, not a lock — §5.1 says only "closed set". `n_λ`
follows the implemented inventory; the cost difference is `(10−5)·d ≈ 3.8K`
against ≈3.6M. The §4.7 table should be corrected when `n_λ` is finalised.

**The §8.2 sketch is superseded.** It returns `letter_ids: list[int]` — one id per
token — which is **incompatible** with embedding-space pooling: pooling needs the
character labels *and* `W_λ`, which lives inside the module. §4.4 step 4 and §5.1
are the locked specification; §8.2 is illustrative. Recorded so B4B does not
implement the sketch. Contributors are ragged; padded-dense, flat-segment and
sparse-matrix batchings are all equivalent by linearity of the mean, so the
choice is an implementation decision, not a scientific one.

**Proposal update summary for the B4A block.** D-B4A-006 and D-B4A-007 changed
`unmark-proposal.md` (v1.3 → **v1.4**): §4.6 now states the Stage-1 pooling, and
§4.7 / §8.2 correct the letter-table cardinality from the stale `~10` to `5`. The
remaining B4A entries are extractions and recorded decisions, not proposal
changes. **Compiled PDF stale: YES.**

---

## B4B — neural adapter implementation

### D-B4B-001 — the adapter is implemented exactly as B4A locked it

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation); real-model validation **PENDING** the Colab probe |
| **Owner** | B4B |

`unmark/modeling/adapter.py` implements proposal §4.5 with no deviation:

```
q_i = [ e_i ; t_i ; l_i ]
f_i = LN( W_f q_i + c_f )
g_i = sigmoid( W_g q_i + c_g )
z_i = g_i * f_i + (1 - g_i) * e_i
```

`W_f` and `W_g` are each `Linear(3d -> d, bias=True)`; the adapter LayerNorm has
dimension `d`; there is **no activation between `W_f` and the LayerNorm**, and
the combination is **convex, not residual**.

**Channel implementation.** The tone table has exactly 7 rows and the letter
table exactly 5. Both channels replace the out-of-table `NA` sentinel with row 0
*before* the lookup and then zero the result by the mask, so **the sentinel never
reaches `nn.Embedding`** and the substituted row cannot leak through. Unmasked
ids outside the table raise `ChannelContractViolation` rather than silently
indexing the wrong row. The letter mean clamps its denominator for vectorisation
and then multiplies by `(count > 0)`, which is what makes the zero-contributor
output **exactly** zero rather than zero by the accident that an empty sum is
zero.

**Initialisation.** `W_g = 0` and `c_g = logit(0.01)` per D-B4A-003. `W_f`, the
adapter LayerNorm and both embedding tables keep PyTorch's conventional module
defaults — the proposal locks no separate initialisation for them and inventing
one would be a new hyperparameter.

**Gate-zero.** `convex_combination(gate, fused, base)` is a free function, so the
forced `g := 0` wiring identity is testable **on the primitive**. No public
gate-zero flag exists on the module or config; a test asserts this across all
neural modules and the probe.

**Scope boundary.** The adapter consumes tensors only. It performs no
tokenization, no orthographic decomposition, no corruption and no eligibility
classification; `unmark/modeling/collate.py` is the documented seam between the
deterministic metadata and the tensors.

| | |
|---|---|
| **Proposal updated** | **NO** — this is the proposal's own equation, implemented. PDF stale: **YES** (unchanged) |

### D-B4B-002 — `position_ids` under `inputs_embeds`: decision rule pre-committed

| | |
|---|---|
| **Status** | **CLOSED** by the real PhoBERT run, 2026-08-20. The rule was pre-committed; the result is below. **Re-confirmed on the final 27/27 rerun** (`20260820T081554Z`), including the wrapper passing the authoritative ids and rejecting a wrong override. |
| **Owner** | B4B |
| **Evidence** | first real B4B run and the final rerun — [`docs/experiments/b4b-phobert-adapter-integration-result.md`](../experiments/b4b-phobert-adapter-integration-result.md) |

## RESOLUTION — explicit authoritative `position_ids` are REQUIRED

The pre-committed rule said: if **any** required case differs, the adapted
`inputs_embeds` path must pass explicit `position_ids` reproducing the
authoritative `input_ids` behaviour. Three of four cases differed.

| Case | Implicit `inputs_embeds` ids vs authoritative `input_ids` ids |
|---|---|
| 1 — one sentence, no padding | **identical** |
| 2 — right-padded batch | **DIFFERENT** |
| 3 — unequal-length batch | **DIFFERENT** |
| 4 — real special-token batch | **DIFFERENT** |

**Where the mismatch lives.** For a right-padded sequence the authoritative
`input_ids` path produced

```
2, 3, 4, 5, 6, 1, 1, 1        <- padding positions take the padding index
```

while the implicit `inputs_embeds` path produced

```
2, 3, 4, 5, 6, 7, 8, 9        <- numbering continues straight through padding
```

This is exactly the failure Audit 014 flagged and could not verify locally.
Case 1 matching is what makes it dangerous: a single unpadded sentence looks
correct, so the bug does not appear until batching.

**After supplying explicit authoritative `position_ids`**, the frozen-model
control was **exactly** equal — `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`,
`max_abs_diff_content = 0.0`, `max_abs_diff_padding = 0.0` — between
`model(input_ids=…)` and `model(inputs_embeds=Emb(input_ids), position_ids=…,
attention_mask=…)`. Real-model evidence, not a local assumption, and not a
tolerance.

**Implementation.** `UnmarkEncoder.forward` **derives and passes**
authoritative `position_ids` whenever the caller omits them, from the *same*
`input_ids` used for the frozen word-embedding lookup, so the two cannot drift.
Omission can no longer produce the sequential fallback; documenting "callers
should remember" would have left the wrong answer as the default.

`roberta_position_ids_from_input_ids` implements the verified rule; the padding
index is read from the model (`embeddings.padding_idx`, the word-embedding
table's, or `config.pad_token_id`) and **never hardcoded**. The attention mask is
deliberately not used as a substitute: it marks what to attend to, not how the
model numbers positions, and they disagree precisely where it matters.

**Caller-supplied ids are checked, not trusted.** A supplied `position_ids` must
equal the authoritative tensor **exactly** — same shape, exact integer equality,
no floating tolerance, because these are indices — or `PositionContractViolation`
is raised. Honouring an arbitrary tensor would reopen precisely the hole this
decision closes, and the resulting error is silent. A diagnostic that genuinely
needs arbitrary position ids (the probe's path C) calls the **frozen encoder
directly** rather than weakening the production wrapper.

**Backbone scope — checkpoint-specific, not family-wide.** An earlier form of
this entry cleared any model whose `model_type` was `roberta`. That was too
broad: the probe measured **`vinai/phobert-base`**, not every checkpoint sharing
a family, and weight-tying, resizing or a custom embedding subclass could change
the behaviour. Permission is now a `VerifiedPositionProfile` matching
**checkpoint, model type and model class together**:

```
VerifiedPositionProfile(
    checkpoint="vinai/phobert-base",
    model_type="roberta",
    model_class="RobertaModel",
    position_rule="roberta_input_ids_offset",
)
```

`roberta-base`, `xlm-roberta-base` and `vinai/phobert-large` are all **rejected**
until separately validated. A non-matching backbone raises
`UnsupportedPositionSemantics` **at wrapper construction**, not deep inside a
training run.

The distinction worth keeping straight: the *arithmetic* is ordinary
RoBERTa-style and nothing about it is unique to PhoBERT. What is
PhoBERT-specific is the **empirical permission** to rely on it.

**Profile identity and revision are separate obligations.** The profile
identifies the *checkpoint* (via `name_or_path`, which is the repo id);
[D-B4B-006](#d-b4b-006--modelname_or_path-is-not-revision-evidence) verifies the
exact *revision* from cache snapshot paths. Neither substitutes for the other,
and the real probe must prove **both**.

[D-B3B0-002](#d-b3b0-002) remains **OPEN**; a future backbone must validate its
own `inputs_embeds` position semantics and be registered with its own recorded
evidence. All checkpoint-specific logic lives in the encoder integration layer —
`OrthographyInputAdapter` stays backbone-independent and contains no position
logic at all, which a test enforces.

**Affected.** `unmark/modeling/adapter.py` (`UnmarkEncoder.forward`,
`authoritative_position_ids`, `roberta_position_ids_from_input_ids`,
`detect_padding_index`, `detect_model_family`, `UnsupportedPositionSemantics`);
`scripts/b4b_phobert_adapter_probe.py`; all future Stage-1 and Stage-2 forward
passes on the adapted path.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.5 already requires position embeddings to be supplied by the encoder exactly once; this is how that is achieved against a real model API. PDF stale: **YES** (unchanged) |

---

**The original framing, preserved.**

Audit 014 flagged that PhoBERT is RoBERTa-family, whose position ids are derived
from `input_ids` through a padding-aware offset, while passing only
`inputs_embeds` takes a different code path. **This was not verified and is not
verifiable locally** — the local environment has no torch and no weights.

**The rule is committed in advance so the result cannot be read selectively:**

| Observation | Consequence |
|---|---|
| Path B uses exactly the same effective position ids as path A in **every** required case | no explicit `position_ids` are needed |
| **Any** case differs | B4B must compute and pass explicit `position_ids` reproducing the authoritative `input_ids` path |

Required cases: one sentence with no padding; a batch with right padding; a batch
containing unequal sequence lengths; and real PhoBERT special tokens.

**Method.** `scripts/b4b_phobert_adapter_probe.py` registers a forward hook on
the real position-embedding module and records the **actual index tensors**.
Position ids are not inferred from library source.

**Reporting discipline.** The frozen-model control reports `max_abs_diff` and
`mean_abs_diff`, split into attended and padding positions. A mismatch confined
to padding is a materially different finding from one at content positions, and
the split says which it is. **A loose tolerance must not be used to hide a
mismatch**; if material differences remain at content positions, B4B is BLOCKED.

This is an **engineering integration decision derived from real-model
behaviour**, not a change to the adapter architecture.

| | |
|---|---|
| **Proposal updated** | **NO** — a model-API integration detail does not alter the scientific description. PDF stale: **YES** (unchanged) |

### D-B4B-003 — torch is kept out of `unmark.modeling.__init__`

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | B4B |

`unmark/modeling/__init__.py` re-exports the B4A pure-data contracts only. The
B4B modules (`adapter`, `pooling`, `collate`) are **not** re-exported, because
importing them pulls in torch and the local development environment is
deliberately ML-free — `import unmark.modeling` must keep working there.

`collate.py` goes further: only its final tensor-packing step imports torch, and
it does so lazily. Label-to-id mapping, the `NA` sentinel decision, and laying
content projections out against the model's own special tokens are therefore
pure Python and **genuinely tested locally** rather than only statically.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-B4B-004 — freezing weights and disabling dropout are different contracts

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation safety); **confirmed on real PhoBERT**, run `20260820T081554Z` — the encoder stayed in eval across construction, `train()`, `eval()` and `train()` again, with `requires_grad` false throughout |
| **Owner** | B4B |

**The gap.** `requires_grad = False` freezes *weights*. `eval()` disables
*stochastic training behaviour* such as dropout. Freezing does not imply eval,
and `nn.Module.train()` **recurses into registered children** -- so calling
`wrapper.train()` on a `UnmarkEncoder` would have flipped the pretrained encoder
into train mode and silently reactivated its dropout, while every encoder
parameter stayed correctly frozen. Calling `encoder.eval()` once at construction
is not enough.

**Why it matters for UNMARK specifically.** §4.6 aligns the adapted branch to a
reference branch produced by *the same frozen encoder*. If the encoder ran
dropout during adapter training, the two branches would see different dropout
draws of the same weights, injecting avoidable stochasticity straight into the
alignment objective. The failure is silent: training proceeds, the loss
decreases, and the alignment signal is noisier than the design intends.

**The contract.**

| Call | `wrapper.training` | `encoder.training` | `adapter.training` |
|---|---|---|---|
| after construction | — | **False** | — |
| `wrapper.train()` | True | **False** | True |
| `wrapper.eval()` | False | **False** | False |
| `wrapper.train()` again | True | **False** | True |

**Implementation.** `UnmarkEncoder.train(mode)` calls `super().train(mode)` for
normal `nn.Module` semantics -- the adapter follows `mode`, `self.training` is
set, and `self` is returned -- and then explicitly restores `self.encoder.eval()`.
`requires_grad` state is untouched: this is about module mode only.

**There is no flag to disable it.** A frozen representation encoder running
dropout is not a configuration anyone should be able to select by accident;
changing it requires a logged scientific decision. A test asserts `train()` takes
`(self, mode)` and nothing else.

**Affects.** `UnmarkEncoder.train`; the Colab probe, which now exercises the full
transition sequence and fails the run if the invariant breaks; future Stage-1
training.

| | |
|---|---|
| **Proposal updated** | **NO** — §5.1 already locks "encoder fully frozen"; this is how that is enforced against a PyTorch default, not a change of semantics. PDF stale: **YES** (unchanged) |

### D-B4B-005 — gradient routing is validated from the encoder output, not from `z`

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (probe design); **confirmed on real PhoBERT**, run `20260820T081554Z` — `encoder_output_requires_grad = true`, encoder gradient count 0, and all eight required adapter components carrying finite gradients |
| **Owner** | B4B |

**The gap.** The first B4B probe computed its diagnostic scalar as `z.sum()`,
where `z` is the adapter output. That validates only `phi -> z -> loss`. It would
have passed unchanged if the real integration path contained `z.detach()`, or ran
the encoder inside `torch.no_grad()` -- leaving Stage-1 **unable to train
`A_phi` through the encoder** while the probe reported success.

**The decision.** The gradient-routing diagnostic must be derived from the
**real encoder's final hidden state**, through the same wrapper future code will
use:

```
phi -> z = inputs_embeds -> frozen E_theta -> final hidden states -> scalar -> backward
```

The scalar is `masked_mean_non_special(hidden, attention_mask,
special_tokens_mask).sum()` -- a finite diagnostic over attended, non-special
positions. **It is not a scientific objective**; Stage-1's cosine loss belongs to
a later phase.

**Success conditions.** Encoder: `requires_grad == False` everywhere, and no
pretrained parameter carries a nonzero gradient. Adapter: gradients exist for
`W_g` weight and bias, `W_f` weight and bias, the adapter LayerNorm, and both
embedding tables; all observed gradients finite; at least one nonzero, so that a
connected-but-severed graph cannot pass. Embedding rows the batch does not touch
are **not** required to receive gradients, and `W_g` starting at zero does not
excuse a missing gradient *tensor*.

**Graph-break prohibition, scoped.** No `detach()` and no `no_grad` may sit on the
adapted path used for gradient validation or future Stage-1 training. The
separate `input_ids` vs `inputs_embeds` **equivalence control remains
inference-only and correctly uses `model.eval()` under `torch.no_grad()`** --
`no_grad` is not banned from the probe, only from the trainable path. Tests
enforce both halves.

**If the encoder-derived loss cannot backpropagate into `A_phi`, B4B is
INCOMPLETE.**

**Affects.** `scripts/b4b_phobert_adapter_probe.py`; the artifact records
`gradient_loss_source = "encoder_final_hidden_state"` and
`gradient_path_includes_encoder = true` so the run proves which path was tested.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-B4B-006 — `model.name_or_path` is not revision evidence

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (reproducibility engineering); **confirmed on real PhoBERT**, run `20260820T081554Z` — both tokenizer and model resolved to `01daacda68afe13d83023d16ec647239e344a1e6`, `revision_verified = true` |
| **Owner** | B4B |

**The defect.** The first real B4B run reported 21/22. The single failure was
`revision verified (tokenizer and model)`, and the model was **not** at fault:
the collector read `model.name_or_path` looking for a cache snapshot path, but
for a model that attribute is the repo id — `"vinai/phobert-base"` — and never a
path. It could not have verified anything. The tokenizer passed only because
tokenizers keep real resolved file paths on the instance.

An independent offline diagnostic on the same runtime confirmed the requested
revision **was** loaded: the exact snapshot directory existed, `config.json` and
`pytorch_model.bin` both sat under `snapshots/01daacda…`, transformers'
`cached_file` and `huggingface_hub.try_to_load_from_cache` independently returned
those paths, and `AutoConfig` reported `_commit_hash =
01daacda68afe13d83023d16ec647239e344a1e6`. Status:
`MODEL_REVISION_CACHE_PROVENANCE_CONFIRMED`.

**The repaired policy.**

| Evidence | Role |
|---|---|
| Cached **config** raw path under `snapshots/<revision>/` | **required** |
| Cached **weight artifact** raw path under `snapshots/<revision>/` | **required** |
| `model.config._commit_hash` | required to *agree* when present; absence is not failure |
| `model.name_or_path` | **not evidence** — recorded, flagged as such |
| `refs/main` | recorded as context, **never required** |

Config alone is insufficient: it would not show the *weights* came from that
revision. `pytorch_model.bin`, `model.safetensors` and both sharded index
layouts are accepted, so the verifier is not silently checkpoint-specific.

**Two traps, both recorded because both are easy to walk into.**

1. **Never resolve the symlink before extracting the revision.**
   `snapshots/<commit>/pytorch_model.bin` is a symlink into `blobs/<sha256>`,
   and the blob is *content-addressed* — it carries no revision at all. Calling
   `Path.resolve()` first destroys the only evidence being collected. The blob
   path is recorded separately as forensic information and explicitly flagged
   `weight_blob_is_not_revision_evidence`. A test asserts the extractor is never
   fed a resolved path.

2. **`HF_HOME` is not the hub cache root.** `HF_HOME=/x/.hf-cache` means the hub
   cache is `/x/.hf-cache/hub`. Passing `HF_HOME` itself makes transformers look
   for `HF_HOME/models--…` when the layout is `HF_HOME/hub/models--…`, and the
   lookup silently finds nothing. The verifier takes
   `huggingface_hub.constants.HF_HUB_CACHE` rather than reconstructing it.

**Why `refs/main` is not required.** The project pins an exact commit. Upstream
`main` may legitimately move later while the pinned snapshot stays correct and
reproducible; requiring a match would fail a valid pinned run for a reason that
has nothing to do with it.

**Affected.** `scripts/b4b_phobert_adapter_probe.py`
(`verify_model_revision`, `cached_artifact_path`, `hub_cache_root`,
`read_main_ref`); a new `provenance.json` artifact records the structured
evidence rather than collapsing it into one opaque boolean.

**Scope.** Reproducibility engineering, not a scientific architecture change.
The pinned revision is unchanged and remains a **probe** revision;
[D-B3B0-002](#d-b3b0-002) remains **OPEN**.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

---

## B4B phase closure

### D-B4B-007 — B4B is COMPLETE

| | |
|---|---|
| **Status** | **PHASE CLOSURE** |
| **Owner** | B4B |
| **Evidence** | run `20260820T081554Z`, HEAD `7f6e26c80c0acfa3cdf9168a9b0e2981e6ae1491`, return code 0, **27/27**, `B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE` — [`docs/experiments/b4b-phobert-adapter-integration-result.md`](../experiments/b4b-phobert-adapter-integration-result.md) |

**B4B NEURAL ADAPTER + REAL PHOBERT INTEGRATION: COMPLETE.**

This entry exists to state the phase boundary precisely, because "complete" is
easy to over-read. It restates no run detail that the experiment record already
carries.

**What COMPLETE means.**

* An actual PyTorch adapter exists (`unmark/modeling/adapter.py`).
* It implements the B4A-locked equation — convex combination, input-dependent
  sigmoid gate, LayerNorm after fusion — with no deviation.
* Real PhoBERT **weights were loaded**, at a **verified** revision for both
  tokenizer and model.
* Frozen-model `input_ids` vs `inputs_embeds` equivalence is **exact** (0.0)
  with authoritative position ids.
* `inputs_embeds` position semantics were **measured** and are **enforced** by
  the integration wrapper, which also rejects a wrong caller-supplied override.
* The parameter partition is verified: `6d² + 16d` trainable in the adapter,
  **zero** trainable in the encoder.
* Gradients route **through the frozen encoder** into `A_φ`.
* The frozen encoder's eval-mode invariant holds across every mode transition.
* The deterministic B3 → neural B4 interface was exercised with the real
  pipeline.
* **No scientific training has occurred.**

**What COMPLETE does NOT mean.**

* The **Stage-1 objective is not implemented** — §4.6's `L_align` / `L_clean`
  do not exist in code.
* **Nothing has been trained.** One diagnostic backward pass is not training.
* **[D-B3B0-002](#d-b3b0-002) is still OPEN** — the backbone is not selected.
  The pinned revision remains a *probe* revision.
* **No downstream task, baseline or evaluation has run.**
* Two probe sentences are **integration** evidence, not linguistic coverage.

**Stage-1 implementation may begin. Training may not**: it requires the
repository-wide PRE-TRAIN audit first.

| | |
|---|---|
| **Proposal updated** | **NO** — the run confirms the specification rather than changing it. PDF stale: **YES** (from the earlier v1.4 source changes) |

---

## Stage-1A — objective and data path

### D-S1A-001 — the clean reference branch tokenizes `canon(x)`, not the raw string

| | |
|---|---|
| **Status** | **CLARIFIED** — narrows a proposal statement that did not specify the form |
| **Owner** | Stage-1A |

**Proposal wording.** §4.6 defines the reference as `h(x)` for "clean original
text `x`", without saying whether `x` means the raw incoming string or its
canonical form.

**Implemented decision.** The reference branch tokenizes **`canon(x)`**.

**Reason.** §5.3 already locks corruption to operate on `canon(x)` "so that
placement variants and NFC/NFD forms are the same example and receive the same
noise". If the reference used the raw string, two inputs that the corruption
engine treats as **one example** would get **different** alignment targets — the
target would depend on incoming spelling variation. That variation is the
separate `VARIANT` evaluation axis (§6.3), not something the Stage-1 objective
should silently absorb.

**Mathematical consequence.** All three branches share one canonical identity per
`sample_id`. The reference target is a function of the example, not of how the
example happened to be typed.

**Affected.** `unmark/stage1/data.py::prepare_example`; every Stage-1 batch.

| | |
|---|---|
| **Proposal updated** | **NO** — this is the only reading consistent with §5.3. PDF stale: **YES** (unchanged) |

### D-S1A-002 — Stage-1 verifies base invariance rather than assuming it

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation safety) |
| **Owner** | Stage-1A |

**Prior assumption.** D-B3B2-001 established `b(C_c(x)) = b(x)` for the locked
conditions, and Stage-1 shares one base grid between the adapted clean and
adapted corrupted branches on the strength of it.

**Implemented decision.** `prepare_example` **proves** the invariant per example
before sharing: identical base strings, identical authoritative content token
ids, and equal projection counts. `padded_stage1_batch` re-checks the collated
base ids and special-token masks. A failure raises `BaseInvarianceViolation`.

**Reason.** Sharing tensors between two branches is exactly where a silent
divergence would be invisible — the batch would look well-formed and the two
adapted representations would describe different strings. Deriving the corrupted
branch independently and then asserting equality costs one extra decomposition
and turns that class of bug into a loud failure.

**Mathematical consequence.** None when the invariant holds; the objective is
unchanged. When it does not hold, Stage-1 stops instead of training on
mismatched pairs.

**Affected.** `unmark/stage1/data.py`; any future corruption condition whose
scope breaks base invariance would be caught here rather than downstream.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-S1A-003 — Stage-1 does not truncate; it refuses or skips

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (interface); the concrete `max_length` is **OPEN** |
| **Owner** | Stage-1A |

**Prior assumption.** Audit 017 recorded that the Stage-1 data path still needs a
`max_length` policy.

**Implemented decision.** `TruncationPolicy` requires **both** `max_length` and
`on_overflow` — it has **no constructible no-argument form**, and
`prepare_example` requires the policy with no default. **Truncation is not
offered at all**: `OverflowBehaviour` is `FAIL | SKIP | NOT_APPLICABLE`, with no
`TRUNCATE` member.

**No implicit experiment-facing default exists.** An omitted argument would have
selected "unbounded, FAIL" for an experiment without anyone choosing it — and
`SKIP` is especially scientific, since dropping long examples changes the Stage-1
corpus distribution. An **explicit** `max_length=None` is a legitimate caller
statement ("intentionally unbounded for this call") and is spelled
`TruncationPolicy.unbounded()`; that is different from an implicit default of
`None`. Inconsistent combinations — a bound with `NOT_APPLICABLE`, or no bound
with `FAIL`/`SKIP` — fail loud.

`visit` is required by `prepare_example` for the same reason: an implicit
`visit=0` would silently select "never redraw", and the redraw schedule is OPEN.

**Reason.** Trimming `input_ids` without the channel metadata would desynchronise
the B3 projection: tone and letter rows would describe positions the model no
longer sees, and the misalignment would be silent. Truncating both together is
possible in principle but requires deciding what happens to a syllable cut in
half — a question the proposal does not answer.

**Mathematical consequence.** No example is ever trained on with mismatched ids
and channels. Long examples are either refused loudly or excluded explicitly.

**Affected.** `unmark/stage1/contracts.py::TruncationPolicy`; the Stage-1 corpus
decision, which must now state a `max_length` and an overflow behaviour.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-S1A-004 — the corruption rate is a keyed digest, and `visit` is explicit

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (mechanism); the **redraw schedule is OPEN** |
| **Owner** | Stage-1A |

**Proposal wording.** §4.6 and the §5.1 lock: "`p ~ U(0,1)` per example,
continuous". The *distribution* is locked. Nothing states whether `p` is redrawn
per epoch.

**Implemented decision.** `CorruptionRatePolicy.rate_for(sample_id, visit)`
returns a BLAKE2b keyed digest over `(schema_version, seed, sample_id, visit)`,
mapped into `[0, 1)`. **No module-global RNG**; the draw is reproducible from its
key alone. `visit` is an explicit argument with **no default schedule attached** —
the caller states which draw it wants.

The continuous `p` is expressed through the existing B2 contract as a
`CorruptionCondition` with that probability, so the per-unit lottery remains B2's
keyed digest and nothing about corruption is reimplemented here.

**Reason.** `random.uniform` would make the same batch differ between processes
and could not be reproduced from a run record. Attaching a schedule to `visit`
would decide the redraw question by implementation default.

**Mathematical consequence.** `p` is uniform on `[0, 1)` and stable per
`(seed, sample_id, visit)`. Because B2 thresholds a per-unit score against `p`, a
syllable corrupted at some `p` is also corrupted at any larger `p` — the
corruption is monotone in the rate, which is the intended "fraction of syllables"
semantics.

**Affected.** `unmark/stage1/contracts.py`, `unmark/stage1/data.py`. The redraw
schedule and the optional letter-dropout rate remain OPEN.

| | |
|---|---|
| **Proposal updated** | **NO** — the distribution is implemented as written. PDF stale: **YES** (unchanged) |

### D-S1A-005 — Stage-1 scientific values that remain OPEN

| | |
|---|---|
| **Status** | **OPEN — RESEARCHER DECISION REQUIRED** before any training |
| **Owner** | Stage-1A |

**The rule applied throughout:** *an API default is a scientific decision if it
can reach an experiment.* Every value below is therefore a **required argument**
or an explicit `None`, never a convenient default.

| Value | Why it is open |
|---|---|
| `lambda_align`, `lambda_clean` | §4.6: "tuned on a development split". No value is locked. `ObjectiveWeights` requires both |
| Stage-1 corpus | §5 open-items table, §13 item 3 |
| `max_length`, overflow behaviour | no Stage-1 value specified (D-S1A-003) |
| Corruption redraw schedule | distribution locked, schedule not (D-S1A-004) |
| Optional letter-dropout rate | §4.6 calls it optional and gives no value |
| Stage-1 seed | required explicitly by `CorruptionRatePolicy` |
| Batch size, optimizer, learning rate, epochs/steps, warmup/scheduler, gradient accumulation, checkpoint selection | none specified; **not implemented in this phase** |
| Backbone finalisation | [D-B3B0-002](#d-b3b0-002) is OPEN |

`OPEN_STAGE1_VALUES` in `unmark/stage1/contracts.py` is the machine-readable
register, and `require_resolved(name)` raises for any of them. Tests assert the
lambdas and `max_length` cannot be defaulted.

**The existence of a config field does not mean a value is decided.**

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-S1A-006 — diagnostic values cannot become scientific configuration

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation safety) |
| **Owner** | Stage-1A |

**The risk.** The upcoming real-model dry run needs *some* numbers to exercise
the forward and backward path — plausibly `lambda_align = 1.0`,
`lambda_clean = 1.0`, an explicit `max_length`. Those are wiring values. Nothing
about writing them down resolves the scientific decisions, and the danger is that
a diagnostic number quietly becomes a training default because it was the value
sitting in the last config anyone wrote.

**Implemented decision.** `Stage1RunConfig` carries a required
`Stage1Purpose`, mirroring B2's `CorruptionPurpose`:

* `DIAGNOSTIC` — explicit values for exercising a path. Constructible today.
  `to_dict()` stamps `purpose`, `diagnostic_only: true` and
  `values_are_scientific: false` into the run artifact, so a diagnostic record
  cannot be mistaken for an experiment record.
* `SCIENTIFIC` — **cannot be constructed at all** until `resolved_values` names
  every entry of `SCIENTIFIC_REQUIRED_VALUES` (`lambda_align`, `lambda_clean`,
  `corpus`, `max_length`, `truncation_behaviour`,
  `corruption_redraw_schedule`, `stage1_seed`, `batch_size`). It raises
  `UnresolvedStage1Value` listing what is missing.

**Reason.** This is the strongest available guarantee that a diagnostic value
does not drift into a training run: the scientific configuration does not exist
as an object until the researcher has decided. The same pattern already guarded
B2 while GAP-2 was open.

**Mathematical consequence.** None — this is configuration hygiene. The
objective is unchanged.

**Affected.** `unmark/stage1/contracts.py`; the future real-model dry run, whose
artifact must record `purpose = DIAGNOSTIC`; the training runner, which cannot be
configured scientifically until §D-S1A-005 is resolved.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-S1A-007 — the PRE-TRAIN audit runs after the training runner exists

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (process ordering) |
| **Owner** | Stage-1A |

**Prior wording.** Audit 018 first listed the future order as: dry run → resolve
OPEN values → **PRE-TRAIN audit** → training runner.

**The problem.** That places the repository-wide audit *before* the code it is
supposed to inspect. The PRE-TRAIN audit's purpose is to check the **complete
training path** against the proposal before the first scientific optimizer step;
auditing a repository with no runner in it would inspect everything except the
part that does the training.

**Corrected order.**

1. **Real-model Stage-1 dry run** — real PhoBERT, all three branches, one
   diagnostic backward permitted, **no optimizer, no parameter update**,
   `purpose = DIAGNOSTIC`.
2. **Resolve the scientific values** needed to define the runner and its run
   configuration.
3. **Implement the Stage-1 training runner** — optimizer, scheduler and
   checkpointing may exist *in code*. **No scientific training is run**; no
   `optimizer.step()` is executed as an experiment.
4. **Run the mandatory repository-wide proposal-vs-code PRE-TRAIN audit**, which
   inspects the runner from step 3.
5. **Only if that audit PASSes:** the first scientific Stage-1 training run.

**What step 4 must be able to see.** Full proposal vs repository; the Stage-1
objective; corpus and split discipline; corruption sampling and redraw schedule;
stable sample identity and seeds; lambda values; `max_length` and overflow
policy; batch size; optimizer; learning rate; scheduler/warmup; epochs or steps;
gradient accumulation; mixed precision if used; the frozen/trainable partition;
the encoder eval invariant; checkpoint save/resume semantics and the selection
criterion; leakage risks; provenance; reproducibility; configs versus docs versus
tests; and the baseline/protocol commitments that must be fixed before any result
is seen.

**What the OPEN values and the PRE-TRAIN audit block.** They block **scientific
Stage-1 training**. They do **not** block the real-model integration dry run in
step 1, which uses explicit diagnostic-only values and runs no optimizer.

| | |
|---|---|
| **Proposal updated** | **NO** — §7's gate discipline is unchanged; this records where the PRE-TRAIN audit sits relative to code that did not exist when the gates were written. PDF stale: **YES** (unchanged) |

### D-S1A-008 — a scientific Stage-1 run must persist syllable-inventory provenance

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (requirement). **BLOCKING for scientific Stage-1 training and the PRE-TRAIN audit.** Non-blocking for the completed diagnostic. |
| **Owner** | Stage-1A |
| **Evidence** | run `20260820T093520Z` — [`docs/experiments/stage1-real-phobert-diagnostic-result.md`](../experiments/stage1-real-phobert-diagnostic-result.md) |

**Prior assumption.** B4B established a strong provenance standard for the
*model and tokenizer* (D-B4B-006), and the Stage-1 diagnostic inherited it
intact: `provenance.json` verifies both against raw cache snapshot paths.

**What inspecting the real artifact showed.** The run used the pinned Vietnamese
syllable inventory — a fresh Colab runtime fetched it through the repository's
checksum-verifying fetcher before the run — but `provenance.json` records only
model, tokenizer and position-profile provenance. **The inventory is absent from
the artifact.**

**Implemented decision.** A **scientific** Stage-1 run must additionally persist,
from the committed manifest `configs/linguistics/vietnamese_syllables.yaml`:

| Field | Value |
|---|---|
| `inventory_schema_version` | `vn-syllables-v1` |
| `source_name` | `all-vietnamese-syllables.txt` |
| `source_author` | `hieuthi` |
| `source_revision` | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| `sha256` | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| `size_bytes` | `116290` |
| `license_status` | `NO_EXPLICIT_LICENSE` (not vendored; fetched and checksum-verified) |

**Reason.** The inventory decides which spans are eligible, and therefore every
corruption denominator and every channel projection. The manifest itself states
that changing `source_revision` or `sha256` is a **scientific spec change**. A
training run whose artifact cannot name the inventory it used is not reproducible
in the sense the project has held itself to everywhere else.

**Why this does not fail the diagnostic.** The resource was fetched and
checksum-verified before the run by the repository's own fetcher, and the library
code is pinned exactly by a matching HEAD and a clean tree. The gap is in what the
*artifact records*, not in what the run *did*.

**Affected.** The future Stage-1 training runner and its run artifact; the
PRE-TRAIN audit checklist.

| | |
|---|---|
| **Proposal updated** | **NO** — §11 already requires reproducibility; this names one concrete obligation. PDF stale: **YES** (unchanged) |

### D-S1A-008a — no dedicated diagnostic driver was committed (NON-BLOCKING)

| | |
|---|---|
| **Status** | **RECORDED OBSERVATION — NON-BLOCKING.** Corrects an over-broad claim in the first draft of D-S1A-008. |
| **Owner** | Stage-1A |

**The observation.** At `6eb053f` the Stage-1 *library* (`unmark/stage1/*`) is
committed and the working tree was clean, but there is no `scripts/stage1_*`
driver for the diagnostic. Earlier phases each committed a probe script
(`b3b0`, `b3b1`, `b3b2`, `b4b`).

**Correction.** The first draft of D-S1A-008 treated this as **blocking for
scientific training**. That was wrong, and it was wrong in a specific way worth
naming: **no logged decision in this repository requires a dedicated driver
script per phase.** The pattern in `scripts/` is precedent, not a recorded
requirement, and generalising precedent into a blocker invents an obligation
nobody adopted. **Retroactively writing a script to reproduce a historical
dry-run is not required and is not a precondition for anything.**

**What the diagnostic's reproducibility actually rests on**, which is intact:
the library code is pinned by a matching HEAD and a clean tree; the artifact
records the run id, repository HEAD, exact checkpoint revision, sample ids,
canonical and corrupted texts, corruption rates, diagnostic seed and visit. The
inputs are recoverable; only the assembly step is not scripted.

**What *is* required before scientific training**, already covered and not
duplicated here: the future **Stage-1 training runner** must be committed and
reproducible, must fully encode the actual scientific data-assembly and training
path, and must persist the provenance named in D-S1A-008 — and the mandatory
PRE-TRAIN audit must inspect **that runner** before any scientific optimizer
step. That obligation is
[D-S1A-007](#d-s1a-007--the-pre-train-audit-runs-after-the-training-runner-exists)
and is unchanged.

**Affected.** Audit 019 §L wording. No code, no scientific value.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-S1A-009 — revised roadmap: downstream viability is measured before Stage-1 is tuned

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (process ordering). Extends [D-S1A-007](#d-s1a-007--the-pre-train-audit-runs-after-the-training-runner-exists); its core invariant is unchanged. |
| **Owner** | Stage-1A |

**Prior ordering.** D-S1A-007 set: dry run → resolve OPEN values → implement
runner → PRE-TRAIN audit → first scientific training run.

**Revised ordering.** The real-model dry run is now complete, and two steps are
inserted *before* Stage-1 values are resolved:

1. build a minimal downstream / Stage-2 evaluation harness;
2. run a **Vanilla vs Base-only** downstream diagnostic;
3. design and **precommit** the Stage-1 HPO / scientific configuration;
4. implement the Stage-1 training runner — **no scientific training run**;
5. regenerate and synchronise the compiled proposal PDF;
6. run the mandatory repository-wide proposal-vs-code **PRE-TRAIN audit**;
7. **only a PASS** allows scientific Stage-1 training.

**Reason.** This is the proposal's own gate discipline, applied earlier rather
than later. §4.5 states plainly that `g → 0` recovers the **base-only pathway**,
not the unmodified model, and that whether clean-input performance survives that
substitution "is not a structural guarantee at all — it is exactly hypothesis H1,
and exactly what G1 measures". §7's G1 is a *fail-fast* gate: if the frozen
encoder rejects the base-grid input distribution, the input-level design is in
trouble regardless of how Stage-1 is tuned. Measuring Vanilla vs Base-only first
answers that cheaply; tuning Stage-1 before knowing it would be spending effort
on a pathway that might not clear its own gate.

Step 5 is added because the PDF has been stale since the v1.4 source changes, and
a proposal-vs-code audit against a stale compiled artifact would compare code to
the wrong document.

**Unchanged invariant.** The PRE-TRAIN audit still runs **after** the training
runner exists and **before** the first scientific optimizer step, for the reason
D-S1A-007 gives: it must inspect the code that trains.

**Affected.** Phase sequencing; Audit 019 §M; the PRE-TRAIN checklist.

| | |
|---|---|
| **Proposal updated** | **NO** — §7's gates are unchanged; this records when they are exercised. PDF stale: **YES** (unchanged, and step 5 exists to fix it) |

---

## G1 — minimal downstream evaluation harness

### D-G1-001 — the pre-G1 Vanilla-vs-Base-only clean-path burden diagnostic

| | |
|---|---|
| **Status** | **CLARIFIED** — narrows what the harness measures |
| **Owner** | G1 |

**Proposal wording.** §7's G1: *"Attach the fusion layer, train briefly on a
small corpus, force the gate towards identity, evaluate on one classification
task with `FULL` input. **Pass:** within ≈1 point of the unmodified model."*

§4.5 says why that measurement matters: *"Since `e_i = Emb_θ(b_i)` is computed
from the **stripped** base stream, `g_i → 0` yields `E_θ(T(b(x)))`, not
`E_θ(T(x))`. … Whether clean-input performance survives that substitution is not
a structural guarantee at all — it is exactly hypothesis H1, and exactly what G1
measures."*

**Implemented decision.** The harness supports **Vanilla vs Base-only**: the same
frozen encoder on `canon(x)` versus on `b(x)`, with **no adapter**. This is
recorded as the **isolating lower bound of G1**, not as G1.

**Reason.** G1 as written bundles two things: the pathway substitution, and an
adapter trained briefly with its gate pushed towards identity. §4.5 identifies
the *substitution* as the H1 question. Measuring it alone is strictly cheaper and
strictly more diagnostic: if the frozen encoder cannot accept `T(b(x))` at all,
G1 cannot pass regardless of how the adapter is trained, and the failure would be
attributed to the adapter rather than to the pathway.

**Consequence, in both directions.** A Vanilla-vs-Base-only PASS does **not**
discharge G1 — the adapter-attached measurement is still required. A **FAIL does
not automatically equal a G1 FAIL either**, and an earlier draft of this entry
wrongly said it would be "decisive against the input-level design".

That was an over-claim. Base-only runs `b(x)` through the frozen encoder with
**no channels and no adapter**. The real UNMARK clean pathway is
`base + tone + letter -> trainable fusion -> frozen encoder`, and the channels
plus the adapter can recover information the bare base grid loses. So this
measurement quantifies the **burden** created by replacing the original clean
token grid with the stripped base grid — it does **not** establish a performance
ceiling for the trained adapter.

**Name it accordingly:** the *pre-G1 Vanilla-vs-Base-only clean-path burden
diagnostic*. §7's G1 — attach the fusion layer, train briefly, force the gate
towards identity, evaluate on `FULL` — is unchanged and still required.

**The G1 threshold does not transfer either.** §7's "within ≈1 point" is stated
for the fusion-attached measurement. The proposal defines **no** pass criterion
for a Base-only-only comparison, so none is borrowed: the burden diagnostic is
**descriptive** — report the clean score gap with uncertainty across the ≥3 seeds
§6.6 requires — and any gating threshold must be precommitted by the researcher
**before** the numbers are seen.

**Affected.** `unmark/evaluation/*`; the G1 pilot protocol; Audit 020.

| | |
|---|---|
| **Proposal updated** | **NO** — §7 and §4.5 are unchanged; this records which part of G1's measurement the harness isolates. PDF stale: **YES** (unchanged) |

### D-G1-002 — BASE_ONLY runs the frozen encoder directly, not the adapter at g = 0

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation), justified by real-model evidence |
| **Owner** | G1 |

**The question.** §4.5 defines the base-only pathway as the `g → 0` limit of the
UNMARK architecture. Should `BASE_ONLY` therefore exercise `UnmarkEncoder` with a
forced-zero gate, or run the frozen encoder directly on `T(b(x))`?

**Implemented decision.** Direct frozen-encoder execution on `T(b(x))`, with no
adapter constructed.

**Evidence, from the real B4B run** `20260820T081554Z` rather than from
reasoning: `model(input_ids=…)` versus
`model(inputs_embeds=Emb(input_ids), position_ids=authoritative, attention_mask=…)`
gave `max_abs_diff = 0.0` **exactly**, including padding positions; and the
forced `g := 0` wiring identity gives `z = g⊙f + (1−g)⊙e = e`. Composing the two:
the adapter at `g = 0` produces exactly `Emb_θ(T(b(x)))`, and feeding that is
exactly equivalent to feeding the ids. The simpler implementation is therefore
not an approximation of the architectural definition — it is numerically the same
thing.

**The caveat that must travel with it.** `g = 0` is **not attainable** by the
locked sigmoid gate (D-B4A-004); it is a limit. `BASE_ONLY` implements the
**architectural limit**, and is **not** the behaviour of any initialised or
trained adapter. It must never be reported as UNMARK.

**Affected.** `unmark/evaluation/pathways.py`; the future G1 pilot.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-G1-003 — GRR: the two formulations reconcile; no clamping, no epsilon

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — verified against the editable proposal |
| **Owner** | G1 |

**Proposal wording.** §6.5: `GRR = (S_system − S_FLOOR) / (S_UPPER − S_FLOOR)`.
§6.4: `UPPER` = "Clean input, unmodified model"; `FLOOR` = "Corrupted input,
unmodified model".

**Reconciliation.** Both anchors are the **VANILLA** pathway, so §6.5 is
identical to the per-condition form
`[S(system,c) − S(vanilla,c)] / [S(vanilla,FULL) − S(vanilla,c)]`. **There is no
discrepancy between them**, and the implementation follows §6.5 with the anchors
named explicitly so the identity cannot be lost.

**No clamping.** §6.5 prescribes none, and none is added. `GRR > 1` (the system
beat clean vanilla) and `GRR < 0` (it did worse than the corrupted unmodified
model) are both informative outcomes that clamping would erase.

**Degenerate denominator — OPEN.** When `S_UPPER == S_FLOOR`, corruption cost the
unmodified model nothing and "the fraction of the gap recovered" is not a
meaningful quantity. §6.5 defines **no** epsilon, clamp or fallback, so none is
invented: `gap_recovery_rate` raises `UndefinedGRR`, and
`grr_degenerate_denominator_policy` is registered OPEN.

Verified numerically: `84 / 60 / 72 → 0.5` exactly; floor → `0.0`; upper →
`1.0`; `90 → 1.25` unclamped; `48 → −0.5`.

**Affected.** `unmark/evaluation/metrics.py`; §6.5 reporting.

| | |
|---|---|
| **Proposal updated** | **NO** — implemented as written. PDF stale: **YES** (unchanged) |

### D-G1-004 — downstream values that remain OPEN before a real G1 run

| | |
|---|---|
| **Status** | **OPEN — RESEARCHER DECISION REQUIRED** before any real Vanilla-vs-Base-only measurement |
| **Owner** | G1 |

§5's open-items table names **"Classification head concrete values"** as blocking
**G1** specifically, and **"Dataset versions and splits"** as blocking G2 and the
full grid. §5.2 adds: *"The concrete values (hidden size, pooling, learning rate,
epochs, patience) are pinned during spec lock; 'identical' is not a specification
until the numbers are written down."*

Nothing in this task selects any of them. `HeadConfig` requires **every** field;
`EvaluationRunConfig(purpose=SCIENTIFIC, …)` **cannot be constructed** until
`resolved_values` covers `SCIENTIFIC_REQUIRED_VALUES`.

| Value | Why open |
|---|---|
| task / dataset, and G1's "one classification task" | §6.2 names four *categories*, no dataset; §5 table; §13 item 2 |
| head architecture, pooling, hidden size | §5.2 pinned during spec lock |
| head optimizer, learning rate, batch size, epochs, early stopping | §5.2 / not specified |
| seed list | §6.6 fixes a minimum of three; no list given |
| `max_length` | §5.3 pins one per task; no value given |
| checkpoint selection | not specified for the head |
| G1 pass-threshold precision | §7's "within ≈1 point" — the `≈` is not a decision rule, and the metric it applies to is unstated |
| GRR degenerate-denominator policy | §6.5 gives none (D-G1-003) |
| backbone finalisation | D-B3B0-002 |

**The G1 pass threshold deserves separate note, and a scope correction.**
"Within ≈1 point of the unmodified model" does not say whether the point is
accuracy or macro-F1, nor what "≈" tolerates across the ≥3 seeds §6.6 requires.
It must be pinned **before the full-G1 result is observed**, not after.

It is **not** a prerequisite for the pre-G1 Vanilla-vs-Base-only burden
diagnostic. §7 states that threshold for the **fusion-attached** measurement, and
the proposal defines **no** pass criterion for a Base-only-only comparison
(D-G1-001). That diagnostic is descriptive; borrowing the ≈1-point rule for it
would import a criterion the proposal never stated for it.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-G1-005 — Stage-2 head pooling stays OPEN; Stage-1's rule does not transfer

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (implementation safety). The pooling *value* remains **OPEN**. |
| **Owner** | G1 |

**Proposal wording, two different places.** §4.6, after the v1.4 clarification:
*"The pooling is attention-masked mean over non-special content tokens"* — that
sentence is in the **Stage-1 alignment-loss** section and is about the objective's
branches. §5.2, separately: the classification head's *"concrete values (hidden
size, **pooling**, learning rate, epochs, patience) are pinned during spec
lock"*, and §13 item 4 repeats *"Classification head: hidden size, pooling, …"*.

**The defect.** The first implementation of the G1 harness extracted
representations by calling `masked_mean_non_special` — the Stage-1 rule —
returning pooled `[N, d]`. `HeadConfig.pooling` was a required field that
**nothing read**. That silently promoted a Stage-1 decision into an OPEN Stage-2
one, and made the field that was supposed to carry the researcher's choice
decorative.

**Implemented decision.** Scientific extraction is
`encoder_hidden_states(...) -> HiddenStateSet`, returning **unpooled**
`[N, L, d]` together with the attention and special-token masks, so whichever
rule is eventually pinned can be applied correctly by the head.

Exactly one function in the harness pools, it is named
`TEST_ONLY_masked_mean_pool`, and it **raises** unless called with
`EvaluationPurpose.DIAGNOSTIC`. A `SCIENTIFIC` `EvaluationRunConfig` additionally
**refuses** any `HeadConfig.pooling` name prefixed `TEST_ONLY_`. So a scientific
path cannot reach an implicit masked mean while pooling is OPEN.

**Reason.** These are two decisions about two different things — how the Stage-1
objective compares two branches, and how a task head reduces a sequence to a
classification input. The first does not settle the second merely by existing
first. Inheriting it would have meant §5.2's spec-lock item was quietly answered
by an implementation convenience.

**No pooling option was invented.** The proposal has not chosen among CLS, mean,
max or attention pooling, so the harness does not enumerate them; `pooling` is a
required string carrying whatever the researcher pins.

**Stage-1 is untouched.** `STAGE1_POOLING` and `masked_mean_non_special` are
unchanged, and a test asserts it.

**Affected.** `unmark/evaluation/pathways.py`, `unmark/evaluation/contracts.py`,
`tests/test_evaluation_harness.py`; the future Stage-2 head; Audit 020.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.6 and §5.2 are unchanged; this stops one from being read as the other. PDF stale: **YES** (unchanged) |

---

## Pre-G1 burden diagnostic — precommitted protocol

Everything in this block is scoped to **one descriptive pre-G1 clean-path burden
diagnostic**. It is **not** full G1 (§7 attaches the fusion layer), **not** the
§6 multi-task protocol, and **not** a final Stage-2 decision for the paper.

### D-PREG1-001 — SA-VLSP2016 selected as the primary pre-G1 dataset

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — precommitted **before** any project Vanilla-vs-Base-only score exists |
| **Owner** | pre-G1 |

**Proposal wording.** §6.2 names four task *categories* — "emotion recognition,
hate-speech detection, sentiment analysis, and spam-review detection" — and no
dataset. §5's open-items table and §13 item 2 leave "Dataset versions and splits"
OPEN.

**Implemented decision.** **SA-VLSP2016**, three-class sentiment classification,
as the primary dataset for the pre-G1 burden diagnostic.

**Rationale, recorded before the measurement.**

* Three-class Vietnamese sentiment, matching one of §6.2's named categories.
* A published downstream protocol of **5,100 train / 1,050 test**, so the
  measurement is small enough to run cheaply and repeatedly.
* The original training set is **balanced** across positive / neutral /
  negative, which keeps macro-F1 interpretable without reweighting.
* Sources are TinhTe, VnExpress and Facebook, with published collected-pool
  counts of roughly **TinhTe 2,710 · VnExpress 7,998 · Facebook 1,488** — so
  Facebook is about **12.21%** of the pool.
* ViSoBERT's analysis describes the TinhTe/VnExpress portions as comparatively
  **proper-form Vietnamese**, which matters here: the diagnostic measures the
  burden of *removing* marks, so text that carries marks in the first place is
  what makes the comparison meaningful.
* Prior published PhoBERT diacritic-removal evidence shows substantial task
  sensitivity, so the task is **not trivially insensitive** to orthography — a
  dataset where stripping marks changed nothing would make the diagnostic
  uninformative.

**What is explicitly not claimed.** No published fine-tuned score is treated as
an expected UNMARK result, or as a target. Those numbers come from different
protocols, different splits and full fine-tuning; this diagnostic freezes the
encoder and trains a linear head.

**Comparators.** UIT-VSMEC, UIT-ViHSD and ViSpamReviews may be *profiled* to
validate the choice. They are **not** alternative primaries. Switching the
primary after this precommitment requires explicit evidence and its own logged
decision — **never** because another dataset produced a nicer downstream number.

**Affected.** `unmark/evaluation/preg1_protocol.py`;
`scripts/preg1_dataset_profile.py`; the future pre-G1 run.

| | |
|---|---|
| **Proposal updated** | **NO** — §6.2's categories are unchanged; this selects one dataset for one diagnostic, not the paper's task suite. PDF stale: **YES** (unchanged) |

### D-PREG1-002 — dataset provenance must distinguish OFFICIAL from MIRROR

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (reproducibility); authorised-copy acquisition remains **OPEN** |
| **Owner** | pre-G1 |

**The situation.** The official VLSP distribution requires a signed user
agreement. A public mirror may be inspected during profiling, but the two are not
interchangeable for a published result.

**Implemented decision.** `DatasetProvenance` requires an explicit
`DatasetSourceType` (`OFFICIAL` / `MIRROR`) and an explicit
`authorisation_established` boolean — **neither has a default**. It stamps the
source name, revision where applicable, per-file SHA-256, row counts, column
schema, label mapping and licence status. `usable_for_scientific_run` is true
only for an **authorised official** copy.

**Reason.** Silently presenting a mirror as the official dataset would make a
result unreproducible in the one way nobody would think to check. Following the
`vietnamese_syllables.yaml` precedent, the data itself is **not vendored into
git**; only provenance is recorded.

**Affected.** `unmark/evaluation/profiling.py`; every profile artifact; the
pre-G1 run, which requires an authorised copy or one proven equivalent.

| | |
|---|---|
| **Proposal updated** | **NO** — §11 already requires reproducibility. PDF stale: **YES** (unchanged) |

### D-PREG1-003 — the official TEST split is SEALED

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | pre-G1 |

**Proposal wording.** §5.4: test is for "one final evaluation, for the tables
that appear in the paper", and "the risk of drifting into decisions made on test
is high".

**Implemented decision.** The official test split must **not** be used for
dataset choice, head-protocol choice, LR selection, pooling choice, checkpoint
selection, the pre-G1 measurement, or HPO. The pre-G1 diagnostic is **not** that
one final evaluation, so it does not touch test at all. The profiler may read
test for *integrity checking only* — duplicates and cross-split leakage — and
selects `max_length` from **train alone**.

**Affected.** `scripts/preg1_dataset_profile.py`, where `max_length` is computed
from `records["train"]` only; tests assert it.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-004 — intended 70/15/15 internal division of the official TRAIN set

| | |
|---|---|
| **Status** | **RECORDED INTENT**; the splitter is **not materialised** pending duplicate inspection |
| **Owner** | pre-G1 |

Because test is sealed, the official **train** set is divided internally:

| Split | Fraction | Role |
|---|---|---|
| protocol-train | 70% | head training |
| protocol-dev | 15% | shared head-protocol and checkpoint selection |
| measurement-dev | 15% | final descriptive pre-G1 burden measurement |

Separating protocol-dev from measurement-dev is what stops the protocol from
being tuned on the same data the headline gap is reported on.

**The eventual splitter must be** label-stratified as closely as possible;
deterministic and stable under rerun; **group-aware by canonical text**, so a
canonical duplicate cannot cross splits; and independent of any downstream score.

**Precondition.** The splitter is **not built until duplicates have been
inspected**. If conflicting-label canonical duplicate groups exist, they are
**reported** and their handling left **OPEN** — silently dropping or relabelling
them would change the label distribution before anyone decided to.

**Affected.** `unmark/evaluation/preg1_protocol.py` (intent only);
`analyse_duplicates`; the future splitter.

| | |
|---|---|
| **Proposal updated** | **NO** — §5.4 is unchanged; this is an internal division of train. PDF stale: **YES** (unchanged) |

### D-PREG1-005 — `<s>` first-token pooling, for this pre-G1 protocol only

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, scoped to the pre-G1 diagnostic. The full §6 grid's head pooling remains **OPEN** (D-G1-005). |
| **Owner** | pre-G1 |

**Implemented decision.** Stage-2 pooling for this diagnostic is the **`<s>`
first-token (classifier-token) representation**.

**Reason, in order of weight.**

1. It is the native sentence-classification convention for the PhoBERT/RoBERTa
   family, so it is the least surprising choice for this backbone.
2. It avoids importing the **Stage-1 §4.6 masked-mean** rule into Stage-2, which
   D-G1-005 exists to prevent.
3. It avoids making a **length-dependent mean** an extra difference between
   Vanilla and Base-only. Their sequence lengths differ by construction, so a
   mean would vary with length in a way the first token does not — that would
   add a second confound to a measurement whose entire purpose is to isolate one.

**Scope, stated so it cannot creep.** This resolves pooling for **one
diagnostic on one dataset**. §5.2's classification-head pooling for the paper's
grid stays OPEN, and the Stage-1 rule is untouched — tests assert both.

**Affected.** `unmark/evaluation/preg1_protocol.py`; the future pre-G1 head.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-006 — linear head, shared primary protocol, secondary sensitivity

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (recorded); **nothing is trained** |
| **Owner** | pre-G1 |

**Head.** A single linear layer with bias, **no hidden layer, no dropout**,
shaped `d -> 3`. `d` comes from `model.config.hidden_size`; on the pinned probe
revision that is 768, so the head is `768 -> 3`. **`d` is not hardcoded** —
[D-B3B0-002](#d-b3b0-002) remains OPEN.

**Search budget, fixed in advance.** AdamW, weight decay `0.01`, batch size
`128` over cached frozen representations, at most `30` epochs, LR grid
`{1e-4, 3e-4, 1e-3, 3e-3}`, checkpoint on **protocol-dev Macro-F1**, ties broken
by higher Accuracy then earliest epoch.

**Primary protocol — shared.** One LR is selected on protocol-dev using
**Vanilla only**, *before* any measurement-dev result is observed, then
**frozen and reused unchanged for both pathways**. §6.4 requires all systems to
share the head architecture, hyperparameters, seeds and training budget; a shared
LR is what makes the two systems differ **only** in their input pathway.

**The caveat that must travel with it.** Tuning on Vanilla does **not** make
Vanilla an upper bound. It makes the protocol shared and the comparison
interpretable; it does not establish that Base-only could not do better under its
own tuning.

**Secondary sensitivity analysis.** Each pathway may later get the **same**
search space and **same** budget, selecting its own LR on protocol-dev. That
answers a different question and **must not replace** the primary shared-protocol
result.

**Affected.** `unmark/evaluation/preg1_protocol.py`. **Neither search is run in
this task**, and no head trainer exists.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-007 — precommitted, derivable seeds and paired reporting

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | pre-G1 |

**Seeds.** Two disjoint sets, from two different tags:

| Purpose | Tag | Seeds |
|---|---|---|
| Tuning | `UNMARK-PREG1-TUNE-v1` | 5509, 19422, 11800 |
| Final paired measurement | `UNMARK-PREG1-MEASURE-v1` | 53148, 59945, 42941, 720, 9428 |

**Derivation rule:** SHA-256 of the ASCII tag, read as successive 2-byte
big-endian words — `seed_i = int.from_bytes(sha256(tag)[2i:2i+2], "big")`.

**Reason.** Fully determined by the tag string, so anyone can recompute them and
verify they were not selected after seeing a result. A test recomputes both sets
from their tags. Using **different tags** for tuning and measurement keeps the
protocol from being selected on the same randomness the headline is reported on.

**Paired reporting.** The headline is the **paired per-seed gap**
`Delta_s = Score_vanilla_s - Score_baseonly_s`, for **Macro-F1** and
**Accuracy**. Report `mean(Delta)`, sample `std(Delta)`, and the raw per-seed
scores for both pathways. Pairing matters: the two pathways share a seed, so the
per-seed difference removes seed-to-seed variance that would otherwise swamp the
effect.

**No pre-G1 threshold.** None exists in the proposal and none is invented; §7's
"within ≈1 point" belongs to the fusion-attached full G1. The result is
**descriptive**, and must **not** be called an upper bound or ceiling on UNMARK.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-008 — `max_length` from train token coverage, never from a score

| | |
|---|---|
| **Status** | **RULE RESOLVED; VALUE UNRESOLVED** pending the real tokenizer profile |
| **Owner** | pre-G1 |

**Rule.** Choose the **smallest** value in `{64, 128, 256}` such that **at least
99%** of **TRAIN** examples fit in **both** the Vanilla and Base-only
tokenizations, including the evaluator's special tokens.

**Both pathways, jointly.** The binding constraint is the *worse* pathway, and
that is not a formality: Base-only tokenizations of stripped text can differ in
length from Vanilla, so a bound satisfying one may fail the other. A test covers
exactly that case.

**Failure is loud.** If no candidate satisfies the rule, `select_max_length`
raises and `max_length` stays **UNRESOLVED** for researcher review. Exceeding the
verified backbone limit would be a scientific change, not a fallback.

**Never from a downstream score.** The value is decided from token coverage
alone, before any measurement exists — a `max_length` tuned on a score would be
selecting a protocol on the result it is meant to produce.

**Current state: UNRESOLVED.** `Preg1Protocol().max_length is None`, because the
real tokenizer profile has not been run. The local environment is ML-free.

| | |
|---|---|
| **Proposal updated** | **NO** — §5.3 pins a maximum sequence length per task without giving one. PDF stale: **YES** (unchanged) |

---

## Pre-G1 protocol — supersession to UIT-VSFC v1.0

**Timing, stated first because it is what makes the change legitimate.** At the
moment of this supersession, **zero real Vanilla-vs-Base-only downstream scores
existed** — no dataset had been profiled, no head had been trained, and
`results/preg1` did not exist. Verified by inspection before editing. A dataset
change *after* seeing a result would be indefensible; before one, it is ordinary
specification work.

### D-PREG1-001b — UIT-VSFC v1.0 supersedes SA-VLSP2016 for the pre-G1 diagnostic

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, superseding [D-PREG1-001](#d-preg1-001--sa-vlsp2016-selected-as-the-primary-pre-g1-dataset) |
| **Owner** | pre-G1 |

**Original decision (preserved, not erased).** D-PREG1-001 selected
**SA-VLSP2016**, three-class sentiment, on the grounds of a balanced training
set, published source counts and comparatively proper-form Vietnamese in the
TinhTe/VnExpress portions.

**Superseding decision.** **UIT-VSFC version 1.0** — Vietnamese Students'
Feedback Corpus — **sentiment task only**, labels `0 negative / 1 neutral /
2 positive`.

| Split | Size | negative | neutral | positive |
|---|---|---|---|---|
| train | 11,426 | 5,325 | 458 | 5,643 |
| validation | 1,583 | 705 | 73 | 805 |
| test | 3,166 | 1,409 | 167 | 1,590 |

Each row sums exactly to its split size — checked, not assumed.

**Why changed.** The pre-G1 diagnostic wants the **cleanest identifiable
`x -> b(x)` manipulation**, not the most realistic noisy social-media benchmark.
The two goals pull in opposite directions, and the earlier choice optimised the
wrong one for this particular measurement. UIT-VSFC fits better because:

* the original paper describes an explicit **normalization phase** — sentence
  segmentation, abbreviation expansion, misspelling correction, personal-name
  anonymisation — producing >16,000 normalized sentences;
* it has an **official train/validation/test structure**, so the measurement set
  need not be carved out of train;
* its size makes a stable paired probe inexpensive;
* its **official validation split can stay untouched** by head-protocol tuning,
  which is what lets protocol selection and measurement use genuinely different
  data.

**What is not claimed.** This does **not** claim the corpus is perfectly
diacritized. A paper's normalization description is not evidence about
orthographic exposure; the profiler must measure that directly on the real data
(D-PREG1-009).

**Locked.** Profiling is an **integrity and characterisation gate**, not a
downstream-score-based selection contest. If profiling reveals a catastrophic
integrity problem, **STOP** and require a new explicit researcher decision — do
not automatically switch again.

**SA-VLSP2016 remains eligible** for the later full benchmark; it is superseded
for this diagnostic only.

**Affected.** `unmark/evaluation/preg1_protocol.py`,
`unmark/evaluation/profiling.py`, `scripts/preg1_dataset_profile.py`,
`tests/test_preg1_profiling.py`; the future pre-G1 run.

| | |
|---|---|
| **Proposal updated** | **NO** — §6.2's task categories are unchanged; this selects one dataset for one diagnostic. PDF stale: **YES** (unchanged) |

### D-PREG1-002b — access model: official public distribution is not the same as a license

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, repairing [D-PREG1-002](#d-preg1-002--dataset-provenance-must-distinguish-official-from-mirror) |
| **Owner** | pre-G1 |

**Original assumption.** D-PREG1-002 encoded an SA-VLSP-specific fact — that
official access requires a signed user agreement — as a boolean
`authorisation_established`, with `usable_for_scientific_run` true only for an
*authorised* official copy.

**The defect.** UIT's official NLP dataset page lists **UIT-VSFC (version 1.0)**
with a **direct public download**, and — unlike several other datasets on that
same page — presents no instruction to email the group and sign an agreement.
The boolean therefore misclassified an officially and publicly distributed
corpus as unusable, for a reason that does not apply to it.

**Repaired model.** `DatasetAccess` with four states:
`OFFICIAL_PUBLIC_DISTRIBUTION`, `OFFICIAL_AGREEMENT_AUTHORISED`, `MIRROR`,
`UNKNOWN`. `usable_for_scientific_run` is true for **either** official form and
false for a mirror or unknown provenance.

**License kept strictly separate.** `license_status` defaults to
`NOT_ESTABLISHED` and is **not** part of the usability test. *Official public
distribution* and *an explicitly identified license* are **different facts**; no
license is invented, and no legal claim is made beyond the evidence supplied. The
profiler's report prints a distinct warning for each.

Raw dataset files are still **never redistributed through git**; artifacts carry
provenance, hashes and counts, not corpus text.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-004b — split roles: official validation is measurement, train splits 80/20

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, superseding [D-PREG1-004](#d-preg1-004--intended-701515-internal-division-of-the-official-train-set) |
| **Owner** | pre-G1 |

**Original decision.** A 70/15/15 internal division of official train into
protocol-train / protocol-dev / measurement-dev, because SA-VLSP2016 offered no
separate development split.

**Superseding decision.** UIT-VSFC has an official validation split, so:

| Split | Role |
|---|---|
| official **test** | **SEALED** — integrity, hash and duplicate checks only; never protocol decisions, never scores |
| official **validation** | **measurement-dev** — never used to select dataset, pooling, LR, epoch, or any head hyperparameter |
| official **train** | split internally **80% protocol-train / 20% protocol-dev** |

**Why 80/20.** UIT-VSFC sentiment is strongly imbalanced — `neutral` is about
**4%** of train — so a 20% protocol-dev gives a more stable macro-F1 tuning
sample while still leaving over 9,000 training examples. Using the official
validation as measurement is what frees the internal division to be two-way.

**Split seed, precommitted.** Tag `UNMARK-PREG1-SPLIT-UITVSFC-v1`, SHA-256, first
2-byte big-endian word = **17486**. Recomputed and verified. A **third** tag,
distinct from the tuning and measurement tags, so split, tuning and measurement
randomness are independent.

**Splitter properties**, implemented as a generic mechanism
(`profiling.stratified_group_split`): deterministic and stable across reruns;
label-stratified as closely as grouping allows; **group-aware by canonical
text**, so a canonical duplicate cannot cross protocol-train/protocol-dev;
independent of any downstream score; and using a keyed digest rather than
`random`, so it is stable across processes.

**Not run on real data** in this phase: conflicting-label canonical groups must
be inspected first, and how to handle them is a researcher decision.

**Duplicate contract.** Canonical duplicates stay in one group; official
train↔validation overlap is detected and reported before head training;
conflicting-label groups are reported with ids and counts, **never** silently
relabelled or dropped; and if such groups affect split integrity, **STOP** for
researcher review before downstream training.

| | |
|---|---|
| **Proposal updated** | **NO** — §5.4 is unchanged. PDF stale: **YES** (unchanged) |

### D-PREG1-008b — `max_length` fixed at 256, not selected from data

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, superseding [D-PREG1-008](#d-preg1-008--max_length-from-train-token-coverage-never-from-a-score) |
| **Owner** | pre-G1 |

**Original rule.** The smallest of `{64, 128, 256}` covering ≥99% of train on
both pathways.

**Superseding decision.** **`max_length = 256`** for both pathways, with
`truncation = true` and `padding = "max_length"`.

**Why.** Pre-G1 aims to **minimise truncation**, not optimise inference
efficiency, and compute is not a constraint here. PhoBERT's pretrained positional
capacity is 256 for a task sequence, so this is the maximum supported length.
Fixing it removes an otherwise **data-dependent protocol decision** from the
measurement — the earlier rule made the protocol a function of the corpus, which
is one more thing that could differ between a rerun and the original.

**Statistics are still reported, and still matter.** Length distributions,
coverage at 64/128/256, the **overflow rate at 256**, and the Vanilla/Base-only
length delta. They now **characterise** the corpus and quantify truncation rather
than selecting the value. If records overflow 256, the exact aggregate rate is
reported and ordinary truncation applies; the verified backbone limit is never
exceeded automatically.

The selection machinery is **removed from the code**, not merely unused — a test
asserts `select_max_length`, `max_length_evidence`, `MaxLengthUnresolved` and the
old constants no longer exist, so a future caller cannot silently re-enable
data-driven selection.

| | |
|---|---|
| **Proposal updated** | **NO** — §5.3 pins a maximum sequence length per task without giving one. PDF stale: **YES** (unchanged) |

### D-PREG1-009 — the final pre-G1 probe protocol

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (recorded); **nothing is trained** |
| **Owner** | pre-G1 |

**Pathways, unchanged from Audit 020.** `VANILLA: canon(x) -> tokenizer ->
frozen encoder`; `BASE_ONLY: canon(x) -> b(x) -> the SAME tokenizer -> the SAME
frozen encoder`. **No word segmenter is introduced into either** — the
diagnostic must differ in the pathway transformation only, and adding
segmentation would introduce a second variable. Consistent with the locked
`RAW_BASE` contract (D-B3B1A-001). The known caveat is **preserved**: standard
PhoBERT usage expects word-segmented Vietnamese, and RAW_BASE is a deliberate
design choice and possible distribution shift — not something to silently "fix".

**Pooling.** `<s>` first token. No mean pooling. Scoped to **this diagnostic
only**: it does not change Stage-1's masked mean and does not lock pooling for
the full grid.

**Head.** `Linear(d, 3, bias=True)`, `d` from `model.config.hidden_size`. No
hidden layer, no dropout, no LayerNorm, no activation. **`768` is never
hardcoded** — D-B3B0-002 is OPEN.

**Loss.** Ordinary multiclass cross-entropy. **No class weights, no focal loss,
no label smoothing** — the imbalance is *exposed* through macro-F1 and per-class
F1 rather than compensated by a second modelling intervention that would itself
become a variable. Per-class F1 is reported as a diagnostic, `neutral` above all.

**Encoder and numerics.** Frozen, `eval()`, extraction under `torch.no_grad()`,
**FP32** throughout, cached representations in FP32, **no AMP, no BF16/FP16**.
Checkpoint `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` —
a pin for **this probe's reproducibility** that **does not close D-B3B0-002**.

**Optimisation.** AdamW, `betas=(0.9, 0.999)`, `eps=1e-8`,
`weight_decay=0.01`, `amsgrad=false`; **CONSTANT** schedule; warmup **0**; **no**
gradient clipping; batch size **128**; **30 complete epochs**; early stopping
**OFF**; shuffling **on** and deterministic under the run seed; no encoder
gradient or update.

**Primary LR search.** Grid `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`. For each LR and
each of the 3 tuning seeds: train all 30 epochs, evaluate every epoch on
protocol-dev, select that run's checkpoint by highest macro-F1, then higher
accuracy, then earliest epoch. Aggregate across seeds by highest mean macro-F1,
then highest mean accuracy, then **lowest sample SD of macro-F1**, then smaller
LR. **Vanilla only.** The winner is then **frozen and reused unchanged for both
pathways**. Official validation is never used here, and the grid is not altered
after viewing Base-only results.

**The caveat travels with it:** tuning on Vanilla does **not** make Vanilla an
upper bound; it makes the protocol shared and the comparison interpretable.

**Primary measurement.** For each of the 5 measurement seeds, train a Vanilla
head and a Base-only head sharing split, LR, optimizer, scheduler, loss, batch
size, epoch budget, seed, checkpoint criterion, architecture, `max_length` and
precision. **Each pathway trains its own head through its own clean pathway and
may select its own best epoch** under the same checkpoint rule — "same protocol"
does not require an identical epoch number, and demanding one would force a
pathway onto a checkpoint its own dev curve did not choose. Then freeze and
evaluate on the untouched official validation split.

Report `Delta_s = Score_vanilla_s - Score_baseonly_s` for **macro-F1** (primary)
and **accuracy** (secondary): all five raw paired scores, all five deltas,
`mean(Delta)`, sample `std(Delta)`, and raw per-pathway mean/std. **No p-value is
required for n = 5, and none is invented.**

**Secondary sensitivity**, precommitted but not run: each pathway may later
select its own LR under exactly the same grid, seeds, protocol-dev, budget and
checkpoint rule. It answers a different question and **must not replace** the
primary shared-LR result.

**No pre-G1 threshold.** The result is descriptive; full G1's "within
approximately 1 point" is not borrowed. **Neither result may be called an upper
bound or ceiling on UNMARK.**

**Stage-1 is unaffected** — no Stage-1 mathematics, objective or pooling changed.

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |

### D-PREG1-010 — paired initialisation and optimiser detail

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, completing [D-PREG1-009](#d-preg1-009--the-final-pre-g1-probe-protocol) |
| **Owner** | pre-G1 |

**The gap.** D-PREG1-009 said the two pathways share a run seed. Sharing a seed
**label** is not the same as starting from identical parameters: if Vanilla runs
first and Base-only then draws from the same advancing RNG stream, the second
head starts from **different** weights, and that difference lands in `Delta_s`
attributed to the pathway. For a paired measurement whose entire purpose is to
isolate one variable, that is the failure mode most likely to go unnoticed.

**Head initialisation, explicit.**

```
weight -> torch.nn.init.xavier_uniform_
bias   -> torch.nn.init.zeros_
```

`nn.Linear`'s implicit default is **not** relied upon: it is a Kaiming-uniform
variant whose exact form has changed across PyTorch versions, so depending on it
would make the comparison silently version-sensitive.

**Per-run sequence.** For **every** independent head-training run: reset the run
RNGs from the declared run seed **before** head construction; construct the head;
**explicitly** apply the initialisation above; construct the deterministic
shuffle generator from the same run seed.

**Paired guarantee.** For measurement seed `s`, `Vanilla(s)` and `BaseOnly(s)`
start from **bit-identical** classifier parameters, because each re-seeds from
`s` rather than inheriting RNG state from the other. The **data order is paired
too**: same example ids, same labels, same deterministic shuffle schedule. Only
the input pathway differs.

**Optimiser parameter groups.**

| Parameter | weight decay |
|---|---|
| head weight matrix | **0.01** |
| head bias | **0.0** |

The classifier intercept is **not** decayed. Shrinking it pulls the decision
boundary toward the origin, which on a corpus where `neutral` is ~4% of train
would penalise the minority class through a regularisation choice rather than
through the data — and macro-F1 is exactly the metric that would absorb it.

**Loss.** `CrossEntropyLoss(weight=None, label_smoothing=0.0, reduction="mean")`.

**Batching.** `gradient_accumulation_steps = 1`, `drop_last = false`.

**Checkpoint eligibility.** Evaluate and select after each **complete** epoch;
epochs are numbered **1..30**; **epoch 0 — the untrained head — is not
eligible.** An untrained linear head on a 4%-neutral corpus can post a
deceptively reasonable accuracy by favouring a majority class, and letting it
win would report the initialisation rather than the pathway.

**Unchanged.** `betas=(0.9, 0.999)`, `eps=1e-8`, `amsgrad=false`, constant LR,
warmup 0, no gradient clipping, batch 128, 30 epochs, no early stopping.

**Runtime options are not hyperparameters.** Implementation-level AdamW options
(`foreach`, `fused`, `capturable`) vary by PyTorch version. They must **not** be
tuned; the run artifact records the actual runtime version and options in force.

**Affected.** `unmark/evaluation/preg1_protocol.py`,
`tests/test_preg1_profiling.py`; the future pre-G1 head trainer.

| | |
|---|---|
| **Proposal updated** | **NO** — §6.4 already requires systems to share hyperparameters and seeds; this states what "same seed" has to mean in code. PDF stale: **YES** (unchanged) |

---

### D-PREG1-011 — conflicting canonical groups are excluded whole

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, discharging the STOP clause of `DUPLICATE_CONTRACT` |
| **Owner** | pre-G1 |
| **Trigger** | real UIT-VSFC TRAIN profile, run at HEAD `7654ce1bba3eb93d55d7821fbcf10b1fb6741bf9` |

**What the real data showed.** `DUPLICATE_CONTRACT` requires that
conflicting-label canonical groups be reported and that the pipeline **STOP for
researcher review** rather than silently repair them. Profiling the real corpus
raised exactly that stop. TRAIN contains **one** canonical group whose members
disagree on the gold label:

| | |
|---|---|
| **Canonical digest** | `a193a8ff49cc5ab43da189f9126aea19a0a0e9df1e16acc0a710cf7e880d0daa` |
| **Members** | `train:11293`, `train:11417` |
| **Labels in conflict** | one `negative`, one `positive` |

The official **validation** and **test** splits contain no such group. The raw
sentence is deliberately **not** recorded here or in any artifact: the profiler's
standing discipline is that committed evidence carries digests and counts, not
corpus text, and this decision is not an exception to it.

**The decision.** `EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP` — drop **every**
member of the group. Not majority vote, not keep-first, not relabel.

**Why the whole group, and not a repair.** Keeping one member means asserting
which annotation is correct. This diagnostic has no evidence for that and no
need of it.

The contradictory supervision is **avoidable annotation noise**. Pairing does
**not guarantee that its effect cancels**: Vanilla and Base-only use different
representations and may respond differently during optimization, checkpoint
selection or evaluation, so a shared noisy label can still enter `Delta_s`
asymmetrically. The earlier phrasing here — that such an error simply "does not
cancel" — was too categorical in the opposite direction; the honest statement is
that cancellation is **not guaranteed**, which is reason enough not to rely on
it. Whole-group exclusion removes the ambiguity **symmetrically**, without
asserting that either annotation is correct. Two examples out of 11 426 are not
worth carrying an unverifiable annotation into the measurement.

**Scope.** The exclusion applies to the **protocol-train pool only**. The
official validation split is the measurement set and the official test split is
SEALED ([D-PREG1-003](#d-preg1-003--the-official-test-split-is-sealed)); neither
is filtered, and neither contained such a group in any case.

**Derived TRAIN, after exclusion.**

| | Published | Derived |
|---|---|---|
| N | 11 426 | **11 424** |
| `negative` (0) | 5 325 | 5 324 |
| `neutral` (1) | 458 | **458 — unchanged** |
| `positive` (2) | 5 643 | 5 642 |

`neutral` is untouched, which matters: it is about 4% of TRAIN and is the reason
the reported metric is macro-F1 rather than accuracy
([D-PREG1-009](#d-preg1-009--the-final-pre-g1-probe-protocol)). The exclusion
does not move the class balance the metric choice was made against.

**Derived file.** The exclusion-applied TRAIN csv has SHA-256
`a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301`. The file is
**not** in this repository — no corpus file is. The digest is the reproducibility
handle: regenerating the exclusion from the official TRAIN must reproduce it.

**Evidence status.** The group digest, the member ids, the derived counts and the
derived file digest were **observed on Colab against the real corpus** and are
recorded here as external evidence. They were **not** produced by the local
ML-free suite, which has no access to the corpus and never will.

**Reproduction recipe** (Audit 023 Revision 1 — recovered, not decided). The
derived csv is deliberately not committed, so reproducing digest `a20c0f77…`
requires the exact serialisation, which was previously recorded nowhere and had
to be recovered by forensic inspection of the historical notebook. It is
recorded here so the next reconstruction needs no notebook:

*Adapter step* — read the official `sents.txt` + `sentiments.txt`; **no**
normalisation, **no** label transformation; write columns exactly
`id,text,label` with Python `csv.DictWriter` and **LF** terminators; sample ids
are `<split>:<zero-padded five-digit zero-based index>` (e.g. `train:00000`).
Adapter TRAIN is 11 426 rows / 1 067 637 bytes, SHA-256
`5bf8587343ef76231f14d57f1806387d387900c3cbc1635ecb24b97c248c9a9f` — the digest
of the **csv bytes**, not of any source file.

*Derived step* — exclude the entire conflicting group (`train:11293`,
`train:11417`), **no relabel**, retain all other rows in original order, write
with the same writer and terminators; copy dev/test byte-for-byte. This
reproduces TRAIN 11 424 rows / 1 067 331 bytes / `a20c0f77…`, DEV
`9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0`, TEST
`33b58c83a0783e45a12954f8aa761104d2ae0a59a81a641df066e356f6162910`.

Verified by rebuilding from a fresh official download whose ten raw-file digests
matched the Audit-022 records. **A first attempt that omitted the five-digit
zero-padding failed visibly and its output was discarded** — the digest lock was
never relaxed to accommodate it. **Nothing about this decision changed;** the
recipe makes an existing decision reproducible.

**Affected.** `unmark/evaluation/preg1_protocol.py`
(`CONFLICTING_GROUP_POLICY`, `OBSERVED_CONFLICTING_GROUPS`,
`DERIVED_TRAIN_SIZE`, `DERIVED_TRAIN_LABEL_COUNTS`,
`DERIVED_TRAIN_CSV_SHA256`; protocol version `preg1-protocol-v3` ->
`preg1-protocol-v4`), `tests/test_preg1_profiling.py`.

| | |
|---|---|
| **Proposal updated** | **NO** — the proposal does not legislate corpus-level annotation conflicts; this discharges a contract the repository already carried. PDF stale: **YES** (unchanged) |

---

### D-PREG1-012 — channel densities are measured per unit, at §4.3 granularity

| | |
|---|---|
| **Status** | **RESOLVED DECISION**, completing the Audit-021 profiler |
| **Owner** | pre-G1 |

**The gap.** The Audit-021 profiler counted **examples** that carry an observed
mark. That answers "how many sentences have any diacritic at all", which is a
much weaker question than the one the design rests on. A corpus where every
sentence carries one mark and a corpus where every sentence is fully marked are
indistinguishable under an example-level counter, yet they are completely
different inputs to a tone/letter channel.

**Granularity comes from §4.3, not from convenience.** The proposal fixes it:
tone is a **syllable** property — one syllable carries exactly one tone — while
letter diacritics are a **character** property, and one syllable may carry
several at once on different characters. The denominators follow:

| Channel | Denominator | Numerator |
|---|---|---|
| tone | syllables with `Eligibility.VIETNAMESE_CANDIDATE` | those whose `ObservedTone` is not `UNMARKED` |
| letter | character units whose `LetterDiacritic` is **not `NA`** | those whose `LetterDiacritic` is neither `NA` nor `NONE` |

**`NA` is not folded into `NONE`.** §4.3 keeps them distinct and so does the
profiler. `NONE` means a letter that *could* carry a Vietnamese letter diacritic
and does not; `NA` means the channel does not apply at all — digits,
punctuation, whitespace, symbols. Counting `NA` in the denominator would deflate
every letter density by the corpus's punctuation and digit rate, which on a
student-feedback corpus is not a small number.

**Unresolved eligibility reports `null`, never `0`.** The tone denominator needs
the B3A syllable inventory. Without it every syllable is `UNDECIDED`, so
`observed_tone_unit_density` returns `None` and serialises as JSON `null` —
following the same fail-visible rule B2 applies through
`EligibilityPolicy.UNRESOLVED`. A tone density of `0.0` is a *finding*; a
missing inventory is a *defect*, and the artifact must not let the second
impersonate the first. The letter density does not depend on the inventory and
stays defined.

**Aggregation.** A split density is `sum(numerators) / sum(denominators)`, not
the mean of per-example rates — the latter would weight a three-word sentence
equally with a forty-word one.

**Example-level counters are retained.** `with_observed_tone` and
`with_observed_letter` still mean what they meant. The unit densities are added
beside them, not substituted for them, so no earlier artifact is reinterpreted.

**Affected.** `unmark/evaluation/profiling.py` (schema `preg1-profile-v1` ->
`preg1-profile-v2`, `UNIT_DENSITY_SEMANTICS`, four counters and
`eligibility_resolved` on `OrthographyObservation` and `SplitProfile`, two
density properties, a `classifier` parameter on `observe_orthography` and
`profile_split`), `scripts/preg1_dataset_profile.py`,
`tests/test_preg1_profiling.py`.

**Schema-version clarification** (Audit 022 Revision 2, implementation only --
no decision changed). The bump above was applied to `PROFILE_SCHEMA_VERSION` in
`profiling.py`, but `scripts/preg1_dataset_profile.py` still hard-coded
`"preg1-profile-v1"` into `config.json`, so the first patched real run emitted
**v1 in `config.json`, v2 in `provenance.json`, and no top-level version in
`summary.json` at all.** The literal is now imported from the single
authoritative constant, and `summary.json` declares `schema_version` at the top
level. **Every profile-v2 artifact reports `preg1-profile-v2`** -- config,
summary, summary.provenance, provenance and the report heading -- verified by
tests that run the profiler rather than search its source. **No metric,
denominator or density semantic changed.**

**Empirical closure.** Measured on the real corpus by run
`uit-vsfc-v1.0-profile-v2-analysis-v1-5e707ecd-r2` (HEAD
`5e707ecdabc7df378f535bed80e8dd0adb99861e`), with the pinned B3A inventory
verified and `eligibility_resolved = true` on all three splits:

| Split | tone observed / eligible | tone density | letter observed / eligible | letter density |
|---|---|---|---|---|
| derived train | 104 731 / 142 467 | 0.7351246253518359 | 85 253 / 502 706 | 0.16958818872263312 |
| validation | 13 826 / 18 794 | 0.7356603171224859 | 11 130 / 66 451 | 0.16749183608975035 |
| test | 29 100 / 39 197 | 0.742403755389443 | 23 315 / 138 857 | 0.16790655134418864 |

Densities are **not** diacritization or missing-diacritic rates; the profiler
measures what is present. See
[Audit 022 §N](../audits/022-uit-vsfc-real-data-profile-integrity-closure.md).
The decision itself is unchanged.

| | |
|---|---|
| **Proposal updated** | **NO** — §4.3 already fixes the granularity; the profiler was measuring at the wrong one. PDF stale: **YES** (unchanged) |

---

### D-PREG1-013 — UNK counts are attributed to a pathway

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | pre-G1 |

**The defect.** The token profile emitted a single `unk_token_count`. The
accumulator sat inside a loop running over **both** the canonical and the
base-only surface, so the number was a Vanilla + Base-only **sum** with no
attribution. The real run reported `unk_token_count = 4`, which is consistent
with four unknown pieces in Vanilla and none in Base-only, none in Vanilla and
four in Base-only, or any split between them.

**Why that matters here specifically.** The question a two-pathway token profile
exists to answer is whether stripping marks pushes text *out* of the tokenizer's
vocabulary — whether `b(x)` is worse-covered than `x`. A summed counter cannot
answer it in either direction.

**The repair.** `vanilla_unk_token_count` and `base_only_unk_token_count` are
reported separately; `total_unk_token_count` is retained as an explicitly named
aggregate. **Tokenization itself is unchanged** — this is a reporting repair, and
the tests pin that the tokenizer still sees exactly the canonical surface
followed by the base surface and nothing else.

**Status of the pre-patch number.** The reported `4` was produced by the old
code and **cannot** be retroactively attributed. It is not reinterpreted; the
pathway split requires a rerun.

**Affected.** `scripts/preg1_dataset_profile.py`,
`tests/test_preg1_profiling.py`.

**Empirical closure.** Run
`uit-vsfc-v1.0-profile-v2-analysis-v1-5e707ecd-r2` reports
`vanilla_unk_token_count = 4`, `base_only_unk_token_count = 0`,
`total_unk_token_count = 4`, `unk_count_is_per_pathway = true`. The previously
unattributable `4` was **entirely Vanilla**.

**Narrow reading.** The RAW_BASE token-length burden in this corpus is **not**
explained by an increase in Base-only UNK tokens. It does **not** follow that
Base-only carries no tokenizer burden — Base-only sequences are measurably
longer. This remains a tokenization observation, not a downstream result. The
decision itself is unchanged.

| | |
|---|---|
| **Proposal updated** | **NO** — reporting defect. PDF stale: **YES** (unchanged) |

---

### D-PREG1-014 — internal split materialisation is fail-closed and mapping-order-independent

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — implementation contract narrowed |
| **Owner** | pre-G1 |
| **Trigger** | synthetic probes (S3P05) against the committed splitter, run at HEAD `819d09f2df95ca57444a63c86363e614b44ce458` |
| **Timing** | **before any real split membership was materialised or viewed** |

**The prior assumption.** [D-PREG1-004b](#d-preg1-004b--split-roles-official-validation-is-measurement-train-splits-8020)
and [D-PREG1-009](#d-preg1-009--the-final-pre-g1-probe-protocol) specify a
deterministic, group-aware, label-stratified 80/20 division of the derived train
pool under a precommitted seed. `DUPLICATE_CONTRACT` further requires that
conflicting-label canonical groups **STOP for researcher review** and never be
silently relabelled, dropped or assigned. The generic mechanism
(`profiling.stratified_group_split`) was taken to implement that contract.

**It did not.** Synthetic probes found three ways the implementation failed open.

| Id | Defect | Why it is scientific, not cosmetic |
|---|---|---|
| **S23-F1** | `names = list(fractions)` allocated parts in **dict insertion order**, so `{"protocol-train": 0.8, "protocol-dev": 0.2}` and the logically identical `{"protocol-dev": 0.2, "protocol-train": 0.8}` produced **different memberships** | The order in which a dict literal was typed became a scientific variable. Two readers writing the same mapping get different splits, and nothing in the artifact records which order was used |
| **S23-F2** | conflicting-label canonical groups were resolved by `Counter(...).most_common(1)`, a **majority/tie-break vote** | Directly contradicts `DUPLICATE_CONTRACT`. The vote manufactures a gold label no annotator assigned, and does it **precisely in the case a human was supposed to inspect**. A tie is decided by `Counter`'s internal ordering |
| **S23-F3** | duplicate `sample_id`s were accepted and **emitted twice** | A membership artifact is a list of ids. If two rows share one, the artifact cannot say which was assigned, and any downstream join silently doubles or drops a row |

**The decision.**

1. **Allocation order comes from the mapping's content, never its insertion
   order**: sort by descending fraction, then ascending name. For the locked
   mapping this is `protocol-train` (0.80) then `protocol-dev` (0.20) — **the
   order the locked mapping already had**, so **no membership changes**. Verified
   against the pre-hardening implementation on a synthetic pool with the real
   class totals: identical assignment, with and without canonical duplicates.
2. **Conflicting-label canonical groups raise `EvaluationContractViolation`.**
   No majority, no tie-break, no silent selection. The error reports the
   canonical digest, the labels and the sample ids — **never corpus text**.
3. **Duplicate sample ids raise.** Ids may appear in the error; text may not.
4. **Fractions are validated**: non-empty mapping, non-empty string names,
   finite, strictly positive, summing to 1.0 within the existing tolerance.
5. **"Majority label" is removed from the stratification contract.** After the
   fail-closed check every canonical group has exactly one distinct label, so
   groups are stratified by **that** label and no vote is ever taken.

**Why fail-closed rather than relying on the data.** The approved derived pool
has **zero** conflicting groups (Audit 022), so none of this changes the
imminent run. That is exactly why it had to be fixed now: a guarantee that holds
only because today's corpus happens to be clean is not a guarantee. The next
dataset, or a re-derivation, would silently take a majority vote.

**Expected real split, precommitted.** The derived pool has zero canonical
duplicate groups, so every group is a singleton and the per-class allocation is
fully determined by the class totals and the rule — computable **before any
membership is observed**:

| Part | negative | neutral | positive | total |
|---|---|---|---|---|
| `protocol-train` | 4 259 | 366 | 4 514 | **9 139** |
| `protocol-dev` | 1 065 | 92 | 1 128 | **2 285** |

These are **derived from committed aggregates**, not from a run. The materialiser
recomputes them and, when the input digest is the locked
`a20c0f77…`, refuses to write unless the membership matches exactly.

**No real membership influenced any of this.** No split has been materialised, no
downstream score exists, and the defects were found by synthetic probes.

**Affected.** `unmark/evaluation/profiling.py` (`stratified_group_split` and its
new validators; `SPLIT_ALLOCATION_ORDER_RULE`, `SPLIT_GROUPING_RULE`,
`SPLIT_STRATIFICATION_RULE`), `unmark/evaluation/preg1_split.py` (new),
`scripts/materialize_preg1_split.py` (new), `unmark/evaluation/__init__.py`,
`unmark/evaluation/preg1_protocol.py` (`SPLITTER_STATUS` no longer hard-codes
run state), `tests/test_preg1_split.py` (new), `tests/test_preg1_profiling.py`.
Cross-references [D-PREG1-011](#d-preg1-011--conflicting-canonical-groups-are-excluded-whole),
which supplied the derived pool this splitter consumes; D-PREG1-011, -012 and
-013 are otherwise unchanged.

**Empirical closure.** Executed on the real approved pool at HEAD
`66f4522fa86e5f02f583204ddcad560a62b013c0`, schema `preg1-split-v1`, seed
**17486**. The observed split matched the precommitted aggregates **exactly** —
`protocol-train` 9 139 (4 259 / 366 / 4 514), `protocol-dev` 2 285
(1 065 / 92 / 1 128) — with zero cross-part canonical leakage and zero
conflicting-label groups. All 11 424 canonical groups were confirmed singleton
on the real pool, which is the premise the per-class arithmetic rested on.

Assignment digest
`7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84`. A **second
independent materialisation** into a fresh directory produced **byte-identical**
artifacts — identical bytes, not merely identical counts — confirming both the
determinism contract and the runtime/deterministic artifact separation, since
run 2 omitted the runtime file and the scientific artifacts did not move.

**The decision itself is unchanged.** Running a precommitted measurement is not
a decision; no new decision id was created. See
[Audit 023](../audits/023-pre-g1-internal-split-materializer-and-fail-closed-contract.md)
§§M–P.

| | |
|---|---|
| **Proposal updated** | **NO** — the proposal specifies a deterministic group-aware stratified split; this makes the implementation actually satisfy that, and removes a vote the proposal never authorised. PDF stale: **YES** (unchanged) |


---

### D-PREG1-015 — the pre-G1 burden diagnostic is closed

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | pre-G1 |

**What ran.** The primary shared-LR paired measurement (HEAD `929f80e`) and the
precommitted **secondary own-LR sensitivity** (HEAD `3bb0edb`).

| | Macro-F1 | Accuracy |
|---|---|---|
| Vanilla (shared LR 0.01) | 0.745 601 988 142 145 4 | 0.901 326 595 072 646 9 |
| Base-only (shared LR 0.01) | 0.663 044 566 342 825 5 | 0.822 867 972 204 674 8 |
| **Burden (Vanilla − Base-only)** | **0.082 557 421 799 319 88** | **0.078 458 622 867 972 2** |

**The secondary result.** Base-only, tuning independently on the same
precommitted grid with the same three tuning seeds and the same protocol-dev,
**selected LR = 0.01** — the same value the shared protocol had frozen. The
secondary own-LR burden is therefore **numerically identical** to the primary
shared-LR burden.

**What this establishes.** One alternative explanation is removed: the gap is not
an artefact of Base-only being denied its own tuning budget.

**What it does NOT establish.** It is **not** a significance result, **not** an
upper or lower bound, and **not** a claim that the gap survives any other
protocol change. No p-value, threshold or hypothesis test was computed at any
point, by design ([D-PREG1-010](#d-preg1-010--paired-initialisation-and-optimiser-detail)).

**Consequences.** pre-G1 is **not rerun**, its LR grid is **not widened**, and
its numbers are inputs from here on. Official UIT-VSFC TEST remains **sealed**
and structurally unreachable (`Preg1Role` has no `OFFICIAL_TEST` member).

Cross-references [Audit 026](../audits/026-preg1-paired-measurement-runner.md),
[Audit 027](../audits/027-preg1-base-only-own-lr-sensitivity.md).

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |


### D-S1B-001 — UIT-VSFC may not select any Stage-1 value

| | |
|---|---|
| **Status** | **RESOLVED DECISION** |
| **Owner** | Stage-1B |

**The rule.** UIT-VSFC downstream scores **must not** be used to choose any
Stage-1 quantity. Named explicitly, and not limited to these: `lambda_align`,
`lambda_clean`, optimizer, learning rate, batch size, epoch/update budget,
warmup, gradient accumulation, gradient clipping, weight decay, seeds,
checkpoint selection, adapter capacity, corruption policy, corruption rate
sampling, redraw schedule, `pi_strip`, `max_length`, truncation behaviour, and
the Stage-1 corpus.

**Why.**

1. **Selection leakage into the headline number.** Official validation is the set
   the Stage-2 result is reported from. Selecting on it would make the reported
   number a report of the selection — the same failure
   [D-PREG1-009](#d-preg1-009--the-final-pre-g1-probe-protocol) prevents
   one level down.
2. **It would invert the scientific claim.** UNMARK claims that *self-supervised*
   Stage-1 alignment produces diacritic robustness. Tuning Stage-1 on a labelled
   downstream task would make Stage-1 a supervised search over that task, and the
   claim circular.

**What the pre-G1 diagnostic did establish** is downstream *burden*
([D-PREG1-015](#d-preg1-015--the-pre-g1-burden-diagnostic-is-closed)) — the cost
of losing diacritics under a frozen encoder. That is a motivating measurement,
not a selection signal.

**What replaces it.** Stage-1 model and configuration selection uses **Stage-1
held-out unlabeled signals only**: cosine distance between adapted and reference
pooled representations, measured at a fixed grid of corruption conditions on a
document-level held-out split. Procedure proposed in
[Audit 028 §G.4](../audits/028-stage1-scientific-config-review.md).

Official UIT-VSFC TEST remains **sealed**. Corpus-contamination screening does
**not** open it; see the contract in
[D-S1B-002](#d-s1b-002--stage-1-corpus-and-the-contamination-contract).

| | |
|---|---|
| **Proposal updated** | **NO** — the proposal already forbids adjusting the protocol while reading results (§5); this names the specific channel. PDF stale: **YES** (unchanged) |


### D-B3B0-007 — the main backbone is locked

| | |
|---|---|
| **Status** | **RESOLVED DECISION** — closes [D-B3B0-002](#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked) |
| **Owner** | B3B / spec lock |

**Locked.**

```
checkpoint : vinai/phobert-base
revision   : 01daacda68afe13d83023d16ec647239e344a1e6
hidden_size: 768
frozen     : true (proposal §5.1)
```

**Why now.** D-B3B0-002 was OPEN — EMPIRICAL PROBE REQUIRED. Every probe that
could have rejected this revision has now run **on this revision** and passed:

| Evidence | Audit |
|---|---|
| Input and tokenizer contract; tokenizer revision verified | 006, 010 |
| Manual alignment validation and channel projection | 013 |
| Adapter on the real model; position-id semantics repaired and verified | 016, 017 |
| Stage-1 three-branch graph, 31/31 checks; adapter `3,551,232` params, encoder `0` | 019 |
| Pre-G1 burden diagnostic, 30 real head runs across both pathways | 024–027 |

Leaving it open would leave the entire validated stack resting on a revision the
specification still calls provisional — the state D-B3B0-002 itself called "the
worst of the two states".

**A second backbone is not adopted.** §6.1 mentions ViSoBERT. That is a
**generalisation ablation** to run *after* the main result, with its own recorded
position-id evidence: `VERIFIED_POSITION_PROFILES` contains exactly one entry and
`resolve_position_profile` fails closed for anything else, so a second backbone
cannot be used silently. Adopting one now because GPU memory is available would
be selecting an experiment by resource availability rather than by question.

**Scope.** This locks the **main** Stage-1/Stage-2 backbone. It does not license
changing `d`, unfreezing the encoder, or altering the adapter architecture.

| | |
|---|---|
| **Proposal updated** | **NO** — §6.1 names PhoBERT-base but pins no revision; the researcher should add the revision to §5.1 or to the §5 open-items table. PDF stale: **YES** (unchanged) |


### D-S1B-002 — Stage-1 corpus, and the contamination contract

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (corpus choice + contract); **OPEN** for the revision pin |
| **Owner** | Stage-1B |

**Corpus.** The main Stage-1 corpus is **`undertheseanlp/UVW-2026`** (Vietnamese
Wikipedia). Researcher decision, 2026-08-22.

**Rationale.** Clean Vietnamese text is appropriate for learning orthographic
equivalence; document/article identity is available, which both the
document-level split and the corruption key (`sample_id`) require;
reproducibility is far simpler than a massive web mixture; and main Stage-1 is
self-supervised, so it does **not** need labelled downstream-domain matching —
domain-matching to a labelled task is the coupling
[D-S1B-001](#d-s1b-001--uit-vsfc-may-not-select-any-stage-1-value) forbids.

**Source shards — all three, and they are not a split.** Use the three root
Hugging Face parquet shards `train.parquet`, `validation.parquet`,
`test.parquet`. **The upstream UVW train/validation/test labels carry no
scientific split meaning for UNMARK**: they are source shards of one unlabeled
Wikipedia corpus. The shard named `test.parquet` is unrelated to UIT-VSFC's
sealed TEST, and honouring the upstream labels would import a partition never
designed for this study.

**Pipeline order, at the pinned revision — load, screen, split, then chunk:**

| # | Step |
|---|---|
| 1 | Load and **concatenate the three shards in the fixed order** `train → validation → test` |
| 2 | **Preserve article/document ids** through the concatenation |
| 3 | **Contamination screening** (below): exact/canonical duplicates against only legitimately opened UIT-VSFC material |
| 4 | Construct **UNMARK's own document-level train/dev partition** |
| 5 | **Only then**, deterministic chunking |
| 6 | **Every chunk inherits its parent document's partition** |

The **load order is part of the pin**: concatenation order determines document
enumeration, and `sample_id` keys the corruption draw, so a different order is a
different corruption stream at an identical revision.

**Required structural property.** It must be **structurally impossible** for
chunks of one Wikipedia article to appear in both Stage-1 train and Stage-1 dev.
Steps 4→5→6 give this by construction. Chunking before splitting would let two
chunks of one article land on opposite sides — near-duplicate leakage into the
very held-out signal that selects `r` and the learning rate.

**Still OPEN and blocking execution.** The exact Hugging Face **dataset
revision** and the **sha256 of all three parquet files** must be pinned before
any Stage-1 run, and verified at load. An unpinned corpus is as unreproducible as
an unpinned backbone. Public `main` was **observed** at review time as
`a0a79294e4568137e25828bb3f2a4cde8546e1fb`; this is recorded as an
**unverified observation only** — nothing was downloaded, `main` moves, and
execution must name an explicit full revision.

**Later ablation.** Broader corpora (for example CulturaX-vi) **may** be explored
as a corpus/domain ablation. Such an ablation **must not retroactively replace
the main result** — the same rule that governs a second backbone
([D-B3B0-007](#d-b3b0-007--the-main-backbone-is-locked)).

#### Contamination contract

An earlier draft of Audit 028 required zero overlap between the Stage-1 corpus
and UIT-VSFC validation **or test**, while also requiring TEST to stay sealed.
**Those requirements contradict each other**: verifying non-overlap with TEST
means reading TEST. Replaced by:

| # | Rule |
|---|---|
| 1 | Official UIT-VSFC **TEST remains SEALED** — not opened for contamination screening or anything else before final evaluation |
| 2 | Screening may compare **only** against UIT-VSFC material the pre-G1 protocol already legitimately opened: the derived TRAIN pool and official VALIDATION |
| 3 | That screen is an **exact/canonical duplicate** check — equality of `canon(x)` and its sha256 |
| 4 | Any **fuzzy/semantic** near-duplicate analysis is reported **separately**, never conflated with the exact check; it has a threshold, and thresholds are choices |
| 5 | **No claim of "zero TEST overlap" may be made before TEST is opened.** The honest statement is *"no exact overlap against the material legitimately available"* |
| 6 | **After** the full UNMARK configuration and model are frozen and TEST is unsealed for final evaluation, a contamination audit runs against TEST |
| 7 | That audit is **REPORT-ONLY**: it must not trigger retroactive corpus removal, retraining, model re-selection, configuration change, or any change to the reported result |

**Why rule 7.** Acting on a contamination finding after seeing TEST would make
TEST a selection signal — the failure the seal exists to prevent, arriving
through the back door. The honest response to post-hoc contamination is to
**report it as a measured limitation**, not to retrain until the number improves.

#### Chunking contract

Documents are **deterministically pre-chunked** before Stage-1 preparation.
Silent `SKIP` of long documents is **not** the policy: it biases the corpus
toward short examples invisibly. The contract: preserve text order; no
normalization beyond the already-specified `canon` pipeline; stable chunk ids
`"{document_id}#{chunk_index}"`; fit `max_length = 256` on **both** the reference
and base tokenizer paths; never split a syllable span; **run only after the
document-level partition exists**; and **inherit the parent document's
partition** rather than being assigned one. Runtime `on_overflow = FAIL` is then
a **guard** — any overflow means the chunking contract is wrong and the run
stops.

| | |
|---|---|
| **Proposal updated** | **NO** — §5's open-items table lists "Stage-1 corpus"; the researcher should record the pinned revision there. PDF stale: **YES** (unchanged) |


### D-S1B-003 — Stage-1 corruption covers STRIP-ALL

| | |
|---|---|
| **Status** | **RESOLVED DECISION** (mechanism + value); **NOT YET IMPLEMENTED** |
| **Owner** | Stage-1B |

**The defect.** [Audit 028 §F](../audits/028-stage1-scientific-config-review.md)
established, from the data path rather than from prose, that Stage-1's corrupt
branch used a **single** `CorruptionScope` for a whole run, defaulting to
`"TONE"`. Under that default the corrupted branch's **letter channel is
bit-identical to the clean branch's** — measured at **0 / 18** differing prepared
examples. `STRIP-ALL`, which proposal §6.3 says "should be reported as the
headline number", therefore had **exactly zero training support**.

**The mechanism.** The corruption **scope is drawn per example**, not fixed per
run:

```
with probability pi_strip      : scope = TONE_AND_LETTER,  p ~ U(0,1)
with probability 1 - pi_strip  : scope = TONE,             p ~ U(0,1)
```

The `TONE` component covers FULL→P100; the `TONE_AND_LETTER` component covers
FULL→STRIP-ALL. This is the *"optional second rate"* proposal §4.6 already
anticipates. **The corruption engine is unchanged** — both scopes are already
implemented and audited (Audits 003/004); only `CorruptionRatePolicy` gains a
`scope_for(sample_id, visit)`.

**The value. `pi_strip = 0.25`** — an **a-priori researcher decision**
(2026-08-22), fixed before any Stage-1 result exists. It is **never tuned**:
not on UIT-VSFC, not on any downstream score, and not on the Stage-1 held-out
signal either.

**Required independence property.** `rate_for` and `scope_for` must use
deterministic but **domain-separated** streams — both may derive from
`(seed, sample_id, visit)`, but under distinct namespace tags `"stage1-rate"` and
`"stage1-scope"`. It is **forbidden** to reuse one scalar draw for both, to
derive one from the other, or to make `scope` conditional on the sampled `p`.

```
P(p | scope = TONE)            = Uniform(0, 1)
P(p | scope = TONE_AND_LETTER) = Uniform(0, 1)
```

up to the deterministic finite sample. **Why:** conditioning scope on `p` would
confine the letter-degraded regime to part of the rate range, confounding
"letters missing" with corruption severity — any measured STRIP-ALL behaviour
could then not be attributed to letter information alone.

**Rejected alternative.** An independent per-syllable letter-dropout rate `q` —
the most literal reading of "second rate" — would require modifying the audited
`_apply()` and would create the state *"letter diacritic removed, tone kept"*,
which is in **no** evaluation condition and matches no real typing behaviour.
More machinery, less relevant support.

**Blocking.** `scope_for` **does not exist yet**. Until it is implemented, with
ML-free tests proving STRIP-ALL support exists, P100 support survives, the
streams are independent and `p` is uniform within each scope, **STRIP-ALL support
is still zero and Stage-1 training must not begin.**

| | |
|---|---|
| **Proposal updated** | **NO** — §4.6 already anticipates the optional second rate; this fixes its form and value. PDF stale: **YES** (unchanged) |


### D-S1B-004 — Stage-1 optimizer and training configuration

| | |
|---|---|
| **Status** | **RESOLVED DECISION** for the locked rows; **OPEN** for the two pilot values |
| **Owner** | Stage-1B |

Recorded so that engineering convenience is never later mistaken for evidence.
Full table and rationale in
[Audit 028 §H](../audits/028-stage1-scientific-config-review.md).

**LOCKED — researcher-approved.**

| Item | Value |
|---|---|
| Optimizer | AdamW, betas `(0.9, 0.999)`, eps `1e-8`, amsgrad `False` |
| LR schedule | constant; **no warmup** |
| Gradient accumulation | `1` |
| Gradient clipping | **none initially**; grad norm monitored. Revisit only on a non-finite loss or a grad norm >100× its running median — **never** on a downstream score |
| `max_length` | `256`; truncation not offered; `on_overflow = FAIL` |
| Corruption rate | `p ~ U(0,1)` per example (already locked by §4.6/§5.1) |
| Redraw schedule | **per visit** (`visit` = pass index) |
| Objective scale | `lambda_align + lambda_clean = 2`, so only `r = lambda_clean/lambda_align` varies |
| Selection data | Stage-1 held-out **unlabeled** only |
| Validation grid | fixed: `FULL, P50, P100, STRIP_ALL` |
| Checkpoint rule | lowest held-out worst-case condition distance → lower `d_clean` → **earliest** update |

**EMPIRICALLY SELECTED LATER — winners are not known and are not guessed.**

| Item | Precommitted grid | Protocol |
|---|---|---|
| Learning rate | `{1e-4, 3e-4, 1e-3}` | 3-run pilot at `r = 1`, scored by the checkpoint rule |
| `r` | `{0.25, 0.5, 1, 2, 4}` | **Phase 1**: one precommitted selection seed, 5 candidates; primary = lowest worst-case condition distance; tie-break = lower `d_clean`, then smaller `r`. **Phase 2**: rerun **only** the winner on the 3 precommitted Stage-1 seeds, report mean and sample SD **descriptively**. Phase 2 **may not reopen** LR or `r` selection |

A one-seed sweep has no sample SD, so no seed-variance term appears in the
Phase-1 tie-break.

**The complete main sequence — exactly 11 runs, and nothing follows them.**

| Stage | Runs | LR | `r` | Seed |
|---|---|---|---|---|
| LR pilot | **3** | swept `{1e-4, 3e-4, 1e-3}` | fixed `r = 1` | `selection` |
| `r` Phase 1 | **5** | frozen pilot winner | swept `{0.25, 0.5, 1, 2, 4}` | `selection` |
| **FINAL MAIN Stage-1** | **3** | selected | selected | `train\|0,1,2` |
| | **TOTAL 11** | | | |

The final three runs are **simultaneously** the Phase-2 descriptive
characterisation **and the FINAL MAIN Stage-1 trained adapters for the study**.
**There is no additional main Stage-1 training round after them.**

#### Update-budget rule — precommitted

| # | Rule |
|---|---|
| 1 | Train to update **20 000** |
| 2 | If the checkpoint chosen by the locked validation rule is **at update 20 000**, **continue the SAME run** from its last checkpoint/state to **40 000** |
| 3 | Preserve adapter state, optimizer state, corruption `visit`/pass state and every deterministic stream. **Do not restart from scratch** |
| 4 | Checkpoint selection then considers the **complete trajectory** (0 → 40 000) |
| 5 | If the selected checkpoint is **again the final update**, **STOP** and mark the run/config **BUDGET-LIMITED** |
| 6 | **No further extension** (60 k, 80 k, …) may be added after inspecting results |

This is a **precommitted stopping rule, not a downstream decision**: the trigger
reads only the Stage-1 held-out selection ("the best checkpoint is the last one
computed"), the ceiling is fixed before any run, and the outcome of hitting it is
a **reported limitation**, not a longer run. Rule 3 is load-bearing for
reproducibility: the corruption draw is keyed on `(seed, sample_id, visit)`, so a
continuation that reset `visit` would silently re-serve the same corruptions.

#### Seeds — derived, domain-separated, recorded before first use

Root tag **`UNMARK-STAGE1-v1`**, derived with the repository's established
`derive_seeds(tag, count)` convention (`sha256(tag)`, consecutive 2-byte
big-endian integers). Verified: `derive_seeds("UNMARK-PREG1-TUNE-v1", 3)`
reproduces the committed `TUNING_SEEDS = (5509, 19422, 11800)`.

| Role | Namespace tag | Seed |
|---|---|---|
| Pilot / Phase-1 selection | `UNMARK-STAGE1-v1\|selection` | **21230** |
| Final main Stage-1, run 0 | `UNMARK-STAGE1-v1\|train\|0` | **36930** |
| Final main Stage-1, run 1 | `UNMARK-STAGE1-v1\|train\|1` | **7309** |
| Final main Stage-1, run 2 | `UNMARK-STAGE1-v1\|train\|2` | **5993** |
| Corruption stream | `UNMARK-STAGE1-v1\|corruption` | **35422** |

All five are **distinct** (verified). Every integer is recomputable from its tag
string alone, so none can have been chosen to flatter a result. Domain separation
makes it structurally impossible for training, selection and corruption to share
an integer. This is **additional to** the `rate_for`/`scope_for` stream
separation of [D-S1B-003](#d-s1b-003--stage-1-corruption-covers-strip-all), which
separates two draws *within* the corruption seed.

**LOCKED — A-PRIORI ENGINEERING.** Researcher-approved, and deliberately kept in
its own tier so it can never later be re-read as something Stage-1 discovered:
batch size `128`; `initial_max_updates = 20 000` under the budget rule above;
eval cadence every `500` updates; dev split `5 000` documents; weight decay
`0.01` on the fusion/gate weight matrices and `0.0` on biases, LayerNorm, **tone
embeddings and letter embeddings**; best + last checkpoint persistence; optimizer
and corruption `visit` state persistence; the monitoring suite; and no gradient
clipping initially under the stated diagnostic trigger.

**No Stage-1 evidence supports any of these values** — they are engineering
choices fixed before any Stage-1 result existed. One exception worth naming: the
**embedding exclusion** from weight decay carries a genuine scientific argument,
since decaying the tone/letter tables shrinks channel information toward zero,
the opposite of Stage-1's purpose.

**No pilot value is locked by this entry.**

| | |
|---|---|
| **Proposal updated** | **NO**. PDF stale: **YES** (unchanged) |
