# Audit 027 — pre-G1 Base-only own-LR sensitivity (SECONDARY)

| | |
|---|---|
| **Audit id** | 027 |
| **Created (UTC)** | 2026-08-22 |
| **Baseline HEAD** | `929f80e6071a6c9839010423fced353a6264a990` |
| **Scope** | Wire the **precommitted** SECONDARY own-LR sensitivity as a `sensitivity` subcommand. **Run nothing.** |
| **Predecessor** | [026](026-preg1-paired-measurement-runner.md) — the primary paired measurement |
| **Type** | Wiring + tests. No protocol change, no model, no data, no training, no score |
| **Runtime Review** | **2026-08-22** — real Colab execution attempted at `b751c3a`; **stopped before training** with `ImportError`. Packaging defect, not a scientific one. Repaired; see §M. |

---

## A. VERDICT

**IMPLEMENTATION PASS — SECONDARY SENSITIVITY WIRED; NOT RUN**

At creation: **2 383 local tests passed, 91 skipped** (2 353 / 91 before — **+30**).

After the Runtime Review (§M): **2 395 pass, 91 skip, and 2 deliberately FAIL**
— the new committed-tree regression test, which stays red until the omitted
file is committed. That red is the repair signal, not a defect in the wiring.

No scientific constant changed. **No new scientific decision.** The analysis
wired here was precommitted in `preg1_protocol` *before any primary result
existed*; this audit adds the executable path, not the intent.

**D-B3B0-002 OPEN**, final Stage-2 pooling **OPEN**, official TEST **SEALED**
and structurally unreachable, PDF **STALE**, Stage-1 untouched.

**The primary result is unchanged, unrecomputed and unreplaced.**

---

## B. THE PRECOMMITMENT PREDATES THE RESULT

This is the load-bearing claim of the whole audit, so it is evidenced rather
than asserted. `unmark/evaluation/preg1_protocol.py` has carried, since the
pre-G1 protocol precommit (Audit 021) and unchanged since:

```
SECONDARY_SENSITIVITY = "A SECONDARY sensitivity analysis may later let each
pathway select its own LR using exactly the same grid, the same 3 tuning seeds,
the same protocol-dev, the same 30-epoch budget and the same checkpoint rule.
It answers 'best achievable head fit under equal tuning budget' and MUST NOT
replace the headline primary shared-LR result."

PRIMARY_LR_CAVEAT = "Tuning the LR on Vanilla does NOT make Vanilla an upper
bound. It makes the protocol shared and the comparison interpretable; it does
not establish that Base-only could not do better under its own tuning."
```

Both strings entered the repository in commit `7654ce1` ("lock pre-G1 UIT-VSFC
profiling protocol", **2026-08-21 11:57 +0700**) — the Audit 021 precommit — and
`git log -S` shows **no commit has modified them since**. `7654ce1` is a git
ancestor of `929f80e`, the commit at which the primary measurement ran
(**2026-08-22 17:49 +0700**). The precommitment therefore predates the primary
result by construction, not by recollection.

The primary consequently never claimed to bound Base-only, and this analysis is
not a reaction to the primary numbers. `test_the_secondary_analysis_was_precommitted_not_invented_after_the_fact`
pins both strings so the claim cannot quietly decay.

The five terms the precommitment fixes — same grid, same 3 tuning seeds, same
protocol-dev, same 30-epoch budget, same checkpoint rule — are each enforced in
code, not honoured by convention. §F and §G give the enforcement point for each.

---

## C. THE PRIMARY, FROZEN

Completed at HEAD `929f80e6071a6c9839010423fced353a6264a990`:

| | Macro-F1 | Accuracy |
|---|---|---|
| Vanilla (shared LR 0.01) | 0.745 601 988 142 145 4 | 0.901 326 595 072 646 9 |
| Base-only (shared LR 0.01) | 0.663 044 566 342 825 5 | 0.822 867 972 204 674 8 |
| **Delta (Vanilla − Base-only)** | **0.082 557 421 799 319 88** | **0.078 458 622 867 972 2** |

These numbers are **inputs** to this audit. `sensitivity` reads them from the
persisted primary artifact and copies them into its own report under
`primary_reference_burden`; it never recomputes them and cannot alter them.

Before use, the primary is **verified, not trusted** (`load_primary_measurement`,
`scripts/preg1_head_diagnostic.py`):

| Property verified | Refusal if absent |
|---|---|
| LR selected on `VANILLA` | `selected on VANILLA` |
| LR is in the precommitted grid | `not in the precommitted grid` |
| Artifact is the PRIMARY, not another sensitivity | `PRIMARY shared-LR` |
| No separate Base-only LR recorded | `not the primary` |
| Measured on `official-validation` | `reports on …` |
| `official_test_used = false` | explicit |
| `encoder_trained = false` | explicit |
| All five precommitted seeds present, each with numeric Vanilla F1 **and** accuracy | `needs all of …` |

The two primary artifacts are additionally **cross-bound**: the LR read from the
measurement is passed into the committed `load_tuning_artifact`, which requires
the tuning sweep to have selected that same LR, on Vanilla, with
`official_validation_used = false`. Nothing supplies an LR on the command line —
`test_the_two_primary_artifacts_are_cross_checked_against_each_other` asserts
`args.frozen_lr` is absent from the secondary path entirely.

---

## D. WHAT WAS WIRED

`scripts/preg1_head_diagnostic.py` gains a third subcommand. `tune` is
**untouched**. `measure` receives exactly **one additive line** — the
`.require_primary(…)` assertion described in §E — and is otherwise unchanged;
both subcommands have already executed for real, and their bodies are pinned by
existing tests. Neither was refactored to share code with `sensitivity`:
factoring the sweep out of `run_tune` would have altered a function that has
already produced a committed scientific result, for no scientific gain.

**Phase A — Base-only LR selection.** 15 runs = the same 5 learning rates × the
same 3 tuning seeds, `BASE_ONLY` only, trained on protocol-train, selected on
protocol-dev, 30 epochs, committed checkpoint rule. The winner goes through the
committed `select_learning_rate` → `freeze_learning_rate`.

**Phase B — the paired secondary measurement.** 5 runs (not 10): one `BASE_ONLY`
run per measurement seed at the Base-only LR, scored on official validation via
the committed `score_measurement`, paired seed-for-seed against the primary
Vanilla results. **Vanilla is neither retuned nor rerun.**

Ordering is load-bearing and enforced structurally: official validation is not
loaded, encoded or cached until **after** `freeze_learning_rate` has returned.
`test_official_validation_is_opened_only_after_the_base_only_lr_is_frozen`
compares AST line numbers, so the file cannot be touched during selection even
by an ordering accident.

Reused wholesale, not reimplemented: `train_head`, `select_learning_rate`,
`freeze_learning_rate`, `LrCandidate`, `PairedSeedResult`, `PairedDiagnostic`,
`score_measurement`, `build_head`, `extract_or_load`, `pathway_text`,
`load_membership`, `load_derived_pool`, `tuning_schedule`. **No second trainer,
selector, metric, checkpoint rule or protocol exists.**

CLI flags are runtime paths only: `--split-dir --derived-train --text-column
--label-column --id-column --cache-root --output-dir --official-validation
--primary-tuning --primary-measurement --revision --repository-head`. There is
no flag that can move a precommitted scientific value.

---

## E. LOAD-BEARING CHANGE — FLAGGED FOR REVIEW

The secondary analysis needs a Base-only-selected LR to be *representable*. Two
committed guards refused that outright, so both were relaxed **in an opt-in
direction, with the default behaviour unchanged**:

| Object | Before | After |
|---|---|---|
| `LrCandidate` | every run must be `VANILLA` | new field `pathway`, **default `VANILLA`**; runs must match it |
| `FrozenLearningRate` | `selected_on` must be `VANILLA` | `selected_on` may be `VANILLA` *or* `BASE_ONLY`; nothing else |
| `freeze_learning_rate` | always froze as `VANILLA` | carries the winning candidate's pathway |

**This is a genuine weakening if left there**, so the guarantee was moved rather
than deleted. `FrozenLearningRate.require_primary(context)` refuses a Base-only
LR wherever the primary shared LR is meant, and every primary consumer now calls
it — **four sites**: `run_measure`, the `measure` pre-flight in `main`,
`PairedDiagnostic.__post_init__` (which requires the Vanilla arm to be a primary
selection in **both** report shapes), and `run_sensitivity` itself, where the
comparator read from the primary artifact must prove it is the shared LR before
it is paired against anything. One existing test asserted the old
constructor guard; it was **replaced, not loosened**, by
`test_a_base_only_lr_is_representable_but_never_passes_as_the_primary`, which
pins the new contract in both directions.

Net effect: a Base-only LR can now exist and be reported, and still cannot reach
the primary measurement or masquerade as the shared LR.

`PairedDiagnostic` gained one optional field, `base_only_learning_rate`. When it
is `None` the emitted report is **byte-identical to what the completed primary
run emitted** — asserted by
`test_the_primary_report_shape_is_unchanged_by_the_secondary_feature`. When it
is supplied, the report labels itself and carries both LRs. The delta and
aggregation arithmetic is one implementation serving both shapes
(`test_both_shapes_share_one_delta_implementation`).

---

## F. THE GRID IS NOT EXPANDED

`sensitivity` calls the **same** `tuning_schedule()` the primary used, so the
15 runs are derived from `LR_GRID` × `TUNING_SEEDS` rather than typed. Three
independent checks:

1. `require_full_grid=False` and `expected_seeds=None` appear nowhere in the
   runner — `select_learning_rate` therefore refuses any candidate set that is
   not exactly the precommitted grid on exactly the three tuning seeds.
2. No float literal outside `LR_GRID` occurs anywhere in `run_sensitivity`
   (AST-asserted).
3. `PairedDiagnostic` refuses a secondary whose `grid` or `tuning_seeds` differ
   from the primary's, so a widened grid cannot even be *reported*.

---

## G. BOUNDARIES

| Boundary | Status | Enforcement |
|---|---|---|
| Official TEST | **SEALED, unreachable** | `Preg1Role` has no `OFFICIAL_TEST`; AST-asserted that `run_sensitivity` names only the three legal roles |
| Official validation during Phase A | **never read** | opened after `freeze_learning_rate`; line-order asserted |
| Official validation in Phase B | measurement only | `score_measurement` refuses any other role |
| Vanilla arm | **not retuned, not rerun** | `SystemPathway.VANILLA` absent from `run_sensitivity`; comparator read from the artifact |
| Encoder | frozen, eval, pinned revision | unchanged from the primary |
| Raw text | never persisted | artifact carries ids, counts and digests only |
| Significance machinery | **absent** | no p-value, threshold or pass/fail — AST-asserted against a banned-token list |

---

## H. TESTS

**30 new test cases** (25 functions, one parametrized over 6 refusal cases).
All ML-free; the module still imports without torch.

Because prose-matching tests have been a recurring defect in this project,
every structural assertion here was **mutation-verified**: the property was
deliberately broken and the test confirmed to fail.

| Injected violation | Caught by |
|---|---|
| Stray LR literal in Phase A | `test_the_grid_is_not_expanded_post_hoc` |
| `require_full_grid=False` | `test_the_grid_is_not_expanded_post_hoc` |
| Phase B loops over pathways (10 runs) | `test_phase_b_is_five_runs_not_ten` |
| A genuine third `train_head` call site | `test_phase_b_is_five_runs_not_ten` |
| Vanilla text encoded in Phase A | `test_vanilla_is_neither_retuned_nor_rerun` |
| Vanilla representations built at all | `test_vanilla_is_neither_retuned_nor_rerun` |
| Official validation read during Phase A | `test_official_validation_is_opened_only_after_…` |
| `base_only_learning_rate` dropped from the report | `test_the_sensitivity_artifact_names_itself_secondary` |
| The two LRs swapped | `test_the_sensitivity_artifact_names_itself_secondary` |
| Official TEST role reached | `test_official_test_stays_unreachable_from_the_sensitivity_path` |
| A `--epochs` override flag added | `test_sensitivity_exposes_no_scientific_overrides` |
| LR taken from the caller | `test_the_two_primary_artifacts_are_cross_checked_…` |
| A `p_value` key added to the artifact | `test_the_sensitivity_artifact_reports_no_significance_machinery` |

Two mutations initially reported "caught" against the wrong function
(`run_measure` precedes `run_sensitivity`, so a first-occurrence replace landed
there) and one was a non-call reference that proved nothing. All three were
re-run scoped to `run_sensitivity` with corrected anchors; the results above are
from the corrected runs.

One end-to-end check ran locally: `sensitivity` refuses a SECONDARY artifact
offered as the primary comparator, and refuses it **before** importing torch or
transformers.

---

## I. WHAT DID **NOT** RUN

**The secondary sensitivity has never executed.** No encoder pass, no Phase A
sweep, no Phase B measurement, no artifact written. There is **no Base-only
own-LR result in this repository**, and this audit reports none. Every check
above is structural or local-fixture based.

One execution was **attempted** on Colab at `b751c3a` and stopped at import time,
before the encoder loaded and before any head was trained (§M). An attempt that
fails during module import produces no science; the statement above is unchanged
by it.

The local `.venv` remains ML-free. Nothing was downloaded, trained or scored.

---

## J. SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 027 created and persisted | **yes** |
| 2 | Analysis precommitted before the primary result | **yes** — §B, verbatim, test-pinned |
| 3 | Primary result unchanged, unrecomputed, unreplaced | **yes** — read-only input |
| 4 | Primary provenance verified, not trusted | **yes** — 8 properties + cross-binding |
| 5 | Phase A is exactly 5 × 3 = 15 BASE_ONLY runs | **yes** — derived from the protocol |
| 6 | Phase B is 5 runs, not 10 | **yes** — Vanilla is not rerun |
| 7 | Vanilla not retuned; LR unchanged | **yes** — AST-asserted |
| 8 | Grid not expanded post-hoc | **yes** — three independent checks (§F) |
| 9 | Official validation unused during LR selection | **yes** — line-order asserted |
| 10 | Official TEST unreachable | **yes** — AST, not grep |
| 11 | Labelled SECONDARY OWN-LR SENSITIVITY | **yes** — in the artifact, single-sourced constant |
| 12 | Never called an upper/lower bound, significance or corrected result | **yes** |
| 13 | No second trainer, selector, metric, checkpoint rule or protocol | **yes** |
| 14 | No CLI override of a locked scientific value | **yes** |
| 15 | No p-value, threshold or pass/fail | **yes** |
| 16 | No raw text persisted | **yes** |
| 17 | No scientific constant changed | **yes** — re-verified from the modules |
| 18 | No new scientific decision | **yes** — `decisions.md` untouched |
| 19 | Guard relaxation disclosed and compensated | **yes** — §E; `require_primary` + replaced test |
| 20 | Guards mutation-verified | **yes** — 13 injected violations, all caught |
| 21 | No model, data, training or score executed | **yes** |
| 22 | D-B3B0-002 OPEN; Stage-2 pooling OPEN; PDF STALE; Stage-1 untouched | **yes** |
| 23 | Tests | **2 383 passed, 91 skipped** (2 353 / 91 before) |
| 24 | `git diff --check` clean; unstaged; no prohibited git operation | **yes** |

---

## K. LIMITATIONS

1. **Nothing has executed.** No Base-only own-LR number exists. Every claim
   about behaviour is structural.
2. **The guard relaxation in §E is real.** The compensating `require_primary`
   is called at four sites verified today; a future consumer that forgets it
   would not be caught by these tests.
3. **The secondary reuses the primary's cached representations by key.** Correct
   by construction — the cache key pins role, pathway, source and order — but the
   Base-only official-validation cache has itself never been exercised by this
   subcommand.
4. **Phase A cost is not free.** 15 further 30-epoch head runs on GPU, on top of
   the primary's 25.
5. **The comparison remains descriptive.** Whatever Phase B produces, it
   describes a burden under two tuning regimes. It tests nothing, bounds
   nothing, and does not evaluate UNMARK.

---

## L. NEXT ACTION

**First: commit `unmark/evaluation/preg1_protocol.py`** — the committed tree is
unimportable without it (§M), and the new contract test stays red until it lands.

**Then: Colab execution of the secondary sensitivity**, at that reviewed commit,
with the persisted primary `tuning.json` and `measurement.json` as inputs. The
result is reported *alongside* the primary shared-LR burden, never in place of
it.

---

## M. RUNTIME REVIEW — 2026-08-22

### M.1 What was attempted

Real Colab execution of the `sensitivity` subcommand at HEAD
`b751c3a368f6ed3ff320bb7918866d4b2ccb45c8`.

**It stopped during module import.** The encoder was never loaded, no
representation was extracted, no head was trained, no checkpoint selected, no
score computed, no artifact written. Execution never reached Phase A.

```
ImportError: cannot import name 'SECONDARY_ANALYSIS_LABEL'
from 'unmark.evaluation.preg1_protocol'
```

### M.2 Root cause — a packaging defect, not a scientific one

Commit `b751c3a` contains **five** of the six paths this work touched:

| Path | In `b751c3a`? |
|---|---|
| `scripts/preg1_head_diagnostic.py` | yes |
| `unmark/evaluation/preg1_head.py` | yes |
| `tests/test_preg1_head.py` | yes |
| `tests/test_preg1_runner.py` | yes |
| `docs/audits/027-…md` | yes |
| **`unmark/evaluation/preg1_protocol.py`** | **NO — still unstaged** |

The two importers were committed; the file **defining** what they import was
not. Reading the committed tree directly confirms the exact breakage:

| Committed importer | Unresolvable protocol symbols |
|---|---|
| `unmark/evaluation/preg1_head.py` | `SECONDARY_ANALYSIS_LABEL` |
| `scripts/preg1_head_diagnostic.py` | `PRIMARY_ANALYSIS_LABEL`, `SECONDARY_ANALYSIS_LABEL` |

**Why the local suite passed.** The suite imports the **working tree**, where
the constants have existed and been correct throughout. Colab does not run the
working tree — it runs `git clone`, i.e. the **committed** tree. Nothing in the
suite compared the two, so a file omitted from a commit was invisible locally
and fatal remotely. That gap, not the sensitivity logic, is the defect.

### M.3 The repair

Of the two candidate fixes, **option 1 is correct and was already implemented**:
the label is single-sourced in `preg1_protocol`. Option 2 was checked and is not
available — no existing constant carries the short human-readable label.
`SECONDARY_SENSITIVITY` is the long precommitment prose and `PRIMARY_LR_CAVEAT`
is the upper-bound disclaimer; reusing either, or inlining the string at each
call site, would duplicate or dilute the semantics the constant exists to fix.

So **no source change was required**. The uncommitted diff is exactly:

```
+SECONDARY_ANALYSIS_LABEL = "SECONDARY OWN-LR SENSITIVITY"
+PRIMARY_ANALYSIS_LABEL   = "PRIMARY SHARED-LR"
```

plus their docstrings — **11 added lines, 0 removed, 0 scientific values
touched**. All 13 locked constants (`LR_GRID`, `TUNING_SEEDS`,
`MEASUREMENT_SEEDS`, `EPOCHS`, `BATCH_SIZE`, `MAX_LENGTH`, `PADDING`,
`TRUNCATION`, `ENCODER_CHECKPOINT`, `ENCODER_REVISION`, the two derived-CSV
digests, `PREG1_PROTOCOL_VERSION`) were compared between `HEAD` and the working
tree and are **byte-identical**.

**The repair is therefore an act of committing, not of coding:**
`unmark/evaluation/preg1_protocol.py` must be included in the commit. Simulating
that commit — HEAD's importers against the working tree's protocol — resolves
all six importers, so that one file is the entire fix. Per this project's
standing rule the file is left **unstaged** for the researcher.

### M.4 Regression test

`tests/test_preg1_import_contract.py` (new, ML-free, 14 cases).

The test that matters is `test_committed_tree_resolves_every_protocol_symbol`:
it reads the importers **and** the protocol from the *same* commit and asserts
every imported symbol is defined there. Because both sides come from one commit,
it checks that commit's internal consistency and never compares HEAD against the
working tree — so the project's standing "leave everything unstaged" workflow
does not trip it. It fails only when a commit is itself unimportable.

Run against `b751c3a` it **fails now**, naming exactly the two importers and
three symbols above — it reproduces the Colab `ImportError` locally, in
milliseconds, without torch. **These two failures are expected and are the
regression proof.** They turn green the moment the omitted file is committed;
they must not be silenced.

Supporting cases: the working-tree equivalent, an importer-list completeness
check so a new importer cannot slip past the contract, and a single-source check
that the label is defined exactly once repository-wide and hard-coded nowhere.

### M.5 Scientific status — unchanged

**The secondary own-LR sensitivity remains NOT RUN.** No Phase A sweep, no
Phase B measurement, **no Base-only own-LR result exists** and none is reported
here. The primary shared-LR result is untouched: not recomputed, not adjusted,
not replaced. No scientific constant, seed, split, role, trainer, checkpoint
rule, metric or protocol changed in this review.

### M.6 Self-audit for this review

| # | Check | Result |
|---|---|---|
| 1 | Root cause identified from the repository, not guessed | **yes** — committed blobs read directly |
| 2 | Smallest correct change | **yes** — no source edit; the constants were already right |
| 3 | Option 1 vs option 2 actually evaluated | **yes** — §M.3 |
| 4 | No duplicated constant with the same meaning | **yes** — one definition, asserted by test |
| 5 | Label text unchanged | **yes** — `SECONDARY OWN-LR SENSITIVITY` |
| 6 | Primary result untouched | **yes** |
| 7 | LR grid, tuning seeds, measurement seeds unchanged | **yes** — byte-compared vs HEAD |
| 8 | Datasets, roles, trainer, checkpoint rule, metrics unchanged | **yes** |
| 9 | Regression test added and proven to catch the exact bug | **yes** — currently red at `b751c3a` |
| 10 | Test is ML-free | **yes** — `ast` + `git show`; no torch |
| 11 | Focused and lightweight suites run | **yes** — 2 395 pass, 91 skip, 2 intended failures |
| 12 | No Audit 028 created | **yes** — 027 revised in place |
| 13 | No Base-only own-LR result claimed | **yes** |
| 14 | Everything unstaged; no prohibited git operation | **yes** |

---

**STATUS: IMPLEMENTATION PASS — SECONDARY OWN-LR SENSITIVITY WIRED; NOT RUN**
**RUNTIME REVIEW: COLAB RUN AT `b751c3a` STOPPED AT IMPORT, BEFORE TRAINING**
**ROOT CAUSE: `preg1_protocol.py` OMITTED FROM THE COMMIT — PACKAGING, NOT SCIENCE**
**REPAIR: COMMIT THAT FILE; NO SOURCE CHANGE WAS REQUIRED**
**SECONDARY SCIENTIFIC EXECUTION: NOT RUN — NO BASE-ONLY OWN-LR RESULT EXISTS**
**PRIMARY SHARED-LR RESULT: UNCHANGED**
**NO NEW SCIENTIFIC DECISION**
**ALL CHANGES UNSTAGED**
