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
| **OPEN — EMPIRICAL PROBE REQUIRED** | D-B3B0-002 (backbone checkpoint not locked) |
| **RESOLVED DECISION** (cont.) | D-B3B1C-001 (manual alignment validated; tone ownership by candidate count) |
| **RESOLVED DECISION** (cont.) | D-B3B2-001 (deterministic B3B COMPLETE), D-B4A-001, D-B4A-007 |
| **RESOLVED DECISION** (cont.) | D-B4A-002 … D-B4A-007 — all six B4A items, resolved by researcher decision; **B4B unblocked** |
| **RESOLVED DECISION** (cont.) | D-B4B-001 (adapter implemented), D-B4B-003 (torch kept out of the package `__init__`), D-B4B-004 (frozen encoder stays in eval), D-B4B-005 (gradient validation via encoder output) |
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
| **Status** | **OPEN — SPEC LOCK ITEM** |
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
| **Status** | **CLOSED** by the real PhoBERT run, 2026-08-20. The rule was pre-committed; the result is below. |
| **Owner** | B4B |
| **Evidence** | first real B4B run — [`docs/experiments/b4b-phobert-adapter-integration-result.md`](../experiments/b4b-phobert-adapter-integration-result.md) |

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
| **Status** | **RESOLVED DECISION** (implementation safety) |
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
| **Status** | **RESOLVED DECISION** (probe design) |
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
| **Status** | **RESOLVED DECISION** (reproducibility engineering) |
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
