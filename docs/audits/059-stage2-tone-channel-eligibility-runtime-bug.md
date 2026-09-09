# Audit 059 - Stage-2 Tone Channel: Silent Eligibility Bug and Fail-Closed Repair

**Scope:** localise and minimally repair the defect that made Stage-2 degraded
representations bit-identical to `FULL`, and record its consequences.
**Date:** 2026-09-09
**Type:** **production bug repair + regression tests.** No scientific decision is
appended. No protocol constant, seed, split, checkpoint, hyperparameter or Drive
artifact changes.

---

## 1. Executive verdict

**Root cause found, reproduced synthetically, and repaired fail-closed.**

The tone channel was **silently dead** in every Stage-2 condition. With no
syllable-inventory classifier, every orthographic region carries
`Eligibility.UNDECIDED`, so `overlay_orthography` returns
`ToneOwnership.UNRESOLVED` for every piece and every tone label collapses to
`NA`. Tone-only corruption therefore could not change `tone_ids` at all.

**This was not a training problem, and not a corruption problem.** Corruption
worked, the base grid held, and the letter channel worked. The failure sat in one
place — eligibility resolution — and was invisible everywhere else.

| | |
|---|---|
| Root cause | eligibility `UNDECIDED` -> `ToneOwnership.UNRESOLVED` -> tone label `NA`, everywhere |
| Production files changed | **1** — `unmark/evaluation/stage2_dual_finalist.py` |
| Stage-1 affected | **NO** — Stage-1 always wires a real classifier |
| `CLEAN_FULL_PATH_AFFECTED` | **YES** — §7 |
| Ten Stage-2 heads | **cannot be reused** for corrected measurement; **not retrained here** |
| Stage-1 checkpoints | **no retraining required** |
| Official TEST | **UNREAD** |
| A/B decision | **none made** |

---

## 2. Authoritative runtime observation

Official validation: 1583 rows, corruption seed `19225`, official TEST unread.

| Observation | Value |
|---|---|
| `P100` corrupted text differs from `FULL` | **1576 / 1583** |
| `P100` base `input_ids` differ from `FULL` | **0 / 1583** |
| `P100` `tone_ids` differ from `FULL` | **0 / 1583** |
| `P100` `tone_mask` differs from `FULL` | **0 / 1583** |
| `P100` `letter_ids` differ from `FULL` | **0 / 1583** |
| `STRIP_ALL` `letter_ids` differ from `FULL` | **1556 / 1583** |
| Representations, both arms | `FULL == P25 == P50 == P75 == P100` bit-for-bit; `STRIP_ALL` differs |

---

## 3. Bug localisation

Traced from definitions, not assumptions:

```
prepare_stage2_unmark_input          unmark/evaluation/stage2_dual_finalist.py
  -> project_text                    unmark/stage1/data.py
       -> decompose                  unmark/orthography/decompose.py
       -> _regions                   unmark/stage1/data.py
       -> overlay_orthography        unmark/alignment/manual.py
       -> project_piece              unmark/alignment/channels.py
  -> build_example                   unmark/modeling/collate.py
  -> Stage2UnmarkInput.tone_ids
```

**Ruled out by reading source and by synthetic probe:**

* `prepare_stage2_unmark_input` correctly projects the **corrupted** text and
  passes `corrupt_projections` into `build_example` — it is not using the clean
  projection by mistake;
* `decompose` extracts `observed_tone` from the actual marks present —
  `decompose(canon("Tôi đã học"))` yields spans `Toi`/`UNMARKED`, `da`/`NGA`,
  `hoc`/`NANG`;
* the `tones` dict in `project_text` maps region index to `observed_tone`
  correctly;
* `project_piece` reads `region_tones` correctly.

**The defect is one branch in `overlay_orthography`:**

```python
undecided = [c for c in contributions if c.eligibility is Eligibility.UNDECIDED]
...
if undecided:
    ownership, region_index = ToneOwnership.UNRESOLVED, None
    detail = "a contributing region has UNDECIDED eligibility; resolve the inventory"
```

and its consequence in `project_piece`:

```python
if overlay.tone_ownership is ToneOwnership.SINGLE_CANDIDATE:
    ...
else:
    region_index, label = None, TokenToneLabel.NA
```

`decompose(text, eligibility_classifier=None)` leaves **every** span
`Eligibility.UNDECIDED`. So without a classifier every piece is `UNRESOLVED` and
every tone label is `NA`.

### 3.1 Answers to the four required questions

**1. Why can `project_text(canonical)` and `project_text(P100-corrupted)` produce
identical tone projections despite different visible marks?** Because neither
produces a tone projection at all. Both collapse to `NA` before the observed tone
is consulted. The tone values were never compared — they were never read.

**2. Is `FULL` tone extraction itself wrong, or only the degraded path?**
**`FULL` is equally wrong.** The channel is dead in *every* condition, `FULL`
included. Degraded conditions merely make it *visible*, because they are the only
ones whose text changes in a way that only the tone channel could record.

**3. Stage-2-specific, or shared with Stage-1?** **Stage-2-specific in practice.**
The projection code is shared, but Stage-1 always supplies a real classifier —
`make_classifier(try_load_inventory())` in `unmark/stage1/execute.py`,
`parallel.py` and `preparation.py`. `unmark/evaluation/` contains **zero**
occurrences of `make_classifier`, `load_inventory` or `try_load_inventory`: the
Stage-2 path never constructs one and `classifier=None` is a permitted default.

**4. Does correcting the bug change clean `FULL` representations?** **YES** —
§7.

### 3.2 Synthetic reproduction, at HEAD, before the repair

Calling `prepare_stage2_unmark_input` on `"Tôi đã học"` with seed `19225` and
`classifier=None`:

```
no exception raised      : True
FULL tone_ids            : (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
P100 tone_ids            : (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
corrupted text differs   : True
tone_ids IDENTICAL (bug) : True
```

With a classifier supplied, the same call at the same commit already behaved
correctly — `FULL (-1,5,5,5,3,3,4,4,4,-1)` versus
`P100 (-1,5,5,5,5,5,5,5,5,-1)`. **The classifier is the whole difference**, and
this reproduces the runtime signature exactly: text differs, ids identical, tone
identical, and `STRIP_ALL` still differing through the letter channel — which
needs no eligibility, because `character_letter_labels` reads the decomposition
directly.

---

## 4. The repair

**One production file: `unmark/evaluation/stage2_dual_finalist.py`.**

A new guard, `require_resolved_tone_channel(projections, *, sample_id, what)`,
raises `EvaluationContractViolation` when any projection carries
`ToneOwnership.UNRESOLVED`. It is called on **both** the clean and the corrupted
projections inside `prepare_stage2_unmark_input`, after the base-grid invariants.

**Why a guard rather than defaulting a classifier in.** Silently constructing an
inventory classifier inside the evaluation layer would make a scientific input
implicit, and would fail differently (or not at all) depending on whether the
git-ignored inventory cache happened to be present. Refusing is the repository's
existing idiom, and it makes the one previously invisible condition visible at
exactly one place.

**The repair honours every constraint:**

* it does **not** change what a tone state means;
* it adds **no** missing-tone oracle state;
* it does **not** touch seed `19225`, the dataset, splits, Stage-1 checkpoints,
  the PhoBERT revision or any Stage-2 hyperparameter;
* it does **not** special-case any condition — the guard's message is asserted by
  test to name no condition;
* it does **not** infer deleted tones from the clean string: it reads only
  `projection.tone.ownership` of the string actually being projected;
* it remains **deployable** — a test prepares the already-stripped text on its
  own and reproduces the `P100` tone ids exactly, with no access to the original.

**No false positives.** Text with no Vietnamese candidates yields
`ToneOwnership.NOT_APPLICABLE`, not `UNRESOLVED`; `"hello world"` still projects
to all-`NA` tones and is accepted.

---

## 5. Files changed

| File | Change |
|---|---|
| `unmark/evaluation/stage2_dual_finalist.py` | **production** — `ToneOwnership` import, `require_resolved_tone_channel`, two call sites |
| `tests/test_stage2_tone_channel_regression.py` | **new** — 19 tests |
| `docs/audits/059-stage2-tone-channel-eligibility-runtime-bug.md` | **new** — this file |

No other file was modified. No existing test was changed.

---

## 6. Regression tests

`tests/test_stage2_tone_channel_regression.py`, torch-free, 19 tests.

| Requirement | Tests |
|---|---|
| **The defect itself** | `..._unresolved_tone_channel_is_refused_rather_than_silently_dead`; `..._guard_names_the_remedy_and_does_not_special_case_a_condition`; `..._guard_reads_only_projection_ownership` |
| **A.** `FULL` vs `P100` | `..._changes_tone_ids_and_nothing_else` — text differs, base text/ids/special mask identical, at least one tone id differs, `letter_ids` identical; `..._every_tone_removed_by_p100_becomes_the_observable_unmarked_state`; `..._p100_introduces_no_new_tone_state` |
| **B.** partial | `..._partial_conditions_are_deterministic_and_keep_the_base_grid` (`P25/P50/P75`); `..._that_removes_a_tone_shows_it_in_tone_ids`; `..._at_least_one_partial_condition_actually_removes_a_tone` |
| **C.** `STRIP_ALL` | `..._changes_both_channels_and_keeps_the_base_grid` |
| **D.** genuine ngang | `..._genuine_unmarked_syllable_and_a_stripped_one_share_one_observable_state`; `..._stripped_text_alone_determines_the_tone_state` |
| **E.** non-Vietnamese / special | `..._non_vietnamese_text_keeps_na_tones_and_is_not_refused`; `..._special_token_positions_stay_na_and_masked_out` |
| **F.** direct Stage-2 regression | `..._stage2_full_and_p100_are_not_identical_for_an_eligible_marked_syllable` |

**Which tests would have failed before the repair.** Stated precisely rather than
loosely: with a classifier supplied, `FULL` and `P100` already differed at HEAD,
so the A–F content tests pass on both sides and serve as locks, not as the
detector. **The tests that would have failed are the three guard tests** — before
the repair, the classifier-less call returned an all-`NA` `Stage2UnmarkInput`
instead of raising (§3.2). That is the defect, and it is what the new tests
catch.

### Test results

| Suite | Result |
|---|---|
| `tests/test_stage2_tone_channel_regression.py` | **18 passed, 1 skipped, 0 failed** |
| Stage-1 orthography / projection / alignment / corruption / eligibility (9 files) | **1669 passed, 0 failed** |
| Stage-2 dual-finalist + head-campaign + campaign (5 files) | **170 passed, 62 skipped, 0 failed** |
| Whole repository | **4361 passed, 171 skipped, 0 failed, 0 errors** in 160.23s |

The whole-repository delta against the pre-repair baseline — `4343 passed`,
`170 skipped` — is exactly **+18 passed, +1 skipped** — the 19 new tests and nothing
else. **No existing test changed behaviour**, which is the evidence that the
guard fires only on the dead-channel configuration.

The 62 Stage-2 skips are torch-gated and torch is not installed here; it was not
installed to remove them.

One parametrised case skips when a partial condition removes no tone from the
fixture under seed `19225`;
`..._at_least_one_partial_condition_actually_removes_a_tone` exists so that skip
can never hide a universally dead channel.

---

## 7. `CLEAN_FULL_PATH_AFFECTED=YES`

**Two claims, kept apart.**

**The repair itself changes no arithmetic.** It adds a guard; the projection
computation is untouched. Any input that previously produced a resolved tone
channel projects bit-identically after the repair.

**But the `FULL` pathway output differs between the two configurations**, and
that difference is exactly what the ten heads were trained on. Under
`classifier=None` the `FULL` tone channel is uniformly `-1`; with the classifier
it carries real tone states. The runtime evidence of §2 establishes that the
Stage-2 downstream path was executing **classifier-less** — a resolved tone
channel could not have produced `0 / 1583` tone differences under `P100`. The
clean caches were produced by that same production path.

**Therefore `CLEAN_FULL_PATH_AFFECTED=YES`**: a corrected run yields different
`FULL` representations from the historical ones.

**Consequence for the ten heads.** The historical Stage-2 clean representation
caches and the ten trained heads **cannot be reused for corrected scientific
measurement** — their inputs carried a dead tone channel, so they were trained on
representations a corrected pipeline will not reproduce. **They are not retrained
in this task**, and nothing about them is modified or deleted.

**Not claimed:** the existing real Drive caches were **not** revalidated locally.
This environment has no Drive access, and §7's conclusion rests on the runtime
signature plus source, not on re-reading those files. Confirming which
configuration produced the clean caches requires A100 revalidation (§9).

---

## 8. Measurement artifact policy

The existing measurement-v2 artifacts are **historical evidence of this bug**.
They are **not edited, deleted or overwritten**, and none was touched here.

Their degraded scores — `P25`, `P50`, `P75`, `P100`, `STRIP_ALL` — are
**scientifically invalid for reporting**: they were computed over representations
that could not encode tone-only corruption. A corrected future run must use a
**new execution commit identity**, a **new measurement namespace**, **newly
generated degraded caches**, and must **reuse no v2 degraded score**.

**Official TEST remains unread. No A/B decision has been made** — the author's
recorded instruction to review both arms' validation metrics before deciding
(Audit 058 §6a) is unaffected by this repair, and no metric produced so far is
valid input to it.

---

## 9. Runtime revalidation still required

| Item | Status |
|---|---|
| Which configuration produced the historical clean caches | **A100 revalidation required** — not determinable locally |
| Corrected Stage-2 measurement, all 12 caches and 60 scores | required, under a new execution identity |
| Whether the ten heads must be retrained under a corrected clean pathway | **follows from §7 as YES**; the retraining itself is a separate authorised task |
| Torch-gated Stage-2 suites | 62 skipped locally; re-run where torch exists |

---

## 10. Final state

```
STAGE2_TONE_CHANNEL_BUG=FOUND_AND_REPAIRED
ROOT_CAUSE=UNDECIDED_ELIGIBILITY_COLLAPSES_TONE_OWNERSHIP_TO_UNRESOLVED
PRODUCTION_FILES_CHANGED=1
REPAIR_KIND=FAIL_CLOSED_GUARD

CLEAN_FULL_PATH_AFFECTED=YES
TEN_STAGE2_HEADS_REUSABLE=NO
TEN_STAGE2_HEADS_RETRAINED_HERE=NO
STAGE1_CHECKPOINTS_RETRAIN_REQUIRED=NO

MEASUREMENT_V2_ARTIFACTS=PRESERVED
MEASUREMENT_V2_DEGRADED_SCORES=SCIENTIFICALLY_INVALID
CORRECTED_RUN_REQUIRES_NEW_EXECUTION_IDENTITY=YES

STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
OFFICIAL_TEST_READ=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE
AUTHOR_AB_DECISION_MADE=NO

READY_FOR_REAL_MEASUREMENT_EXECUTION=NO
```

`READY_FOR_REAL_MEASUREMENT_EXECUTION=NO`: the pipeline no longer fails silently,
but a corrected measurement needs a clean pathway whose `FULL` representations
are valid — which, per §7, the historical ones are not.
