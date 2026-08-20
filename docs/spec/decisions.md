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
| **Status** | **OPEN — Colab probe required** |
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
