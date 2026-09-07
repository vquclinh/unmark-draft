# Audit 051 - Stage-2 Head Campaign: Cache, Runner and Artifact Implementation

**Scope:** implement the Stage-2 runner infrastructure that must exist *before*
head training — representation cache, cached-only head runner, checkpoint
selection, head artifacts, run store, campaign plan and measurement/reporting
path.
**Date:** 2026-09-07
**Type:** **implementation audit, not a protocol amendment.** No scientific
decision is appended.

---

## 1. Executive verdict

**PASS WITH REAL-TORCH ACCEPTANCE PENDING.**

The static implementation is complete and its identity, role and selection-safety
contracts are fully exercised. **Runtime acceptance of the runner is not claimed
here**: all 39 torch-level tests were skipped locally, and they carry the
runtime-critical guarantees — matched A/B initialisation, optimiser isolation,
30-epoch execution, cache tensor round trip, artifact and run-store lifecycle,
deterministic rerun, campaign execution, and the Stage-2/pre-G1 anti-drift
equivalence. Until those run on a machine with torch, the runner has a verified
*contract* and an unverified *runtime*.

| | |
|---|---|
| Audit 049 remains protocol authority | **YES** — nothing here changes it |
| Audit 050 accepted the frozen UNMARK runtime path | **YES** — reused, never duplicated |
| Runner code implemented now | **YES** |
| Static implementation acceptance | **PASS** |
| **Authoritative runner torch acceptance** | **PENDING** — §12 |
| Real Stage-2 campaign run | **NO** |
| Downstream metric observed | **NO** — no UIT-VSFC row was read |
| Official TEST | **SEALED** — structurally unnameable |
| A/B selection | **NONE** — no winner rule, no ranking, no "best arm" |
| New scientific decision appended | **NO** |
| One unresolved scientific value surfaced | **YES** — §10, measurement corruption seed |

---

## 2. Starting state

```
branch : main
HEAD   : 6ca3d0f90215ca0e49ad14d5f9dc43c2e23418d4
status : clean (git status --porcelain produced no output)
```

Reconstructed from the repository, not from the task description:

* **Audit 048** froze exactly two Stage-1 finalists, both verified PASS, digests
  bound, `freeze_complete = true`, neither selected.
* **Audit 049 / D-S2-001** adopted **option (c)**: no downstream A-vs-B
  selection; both finalists carried as separately reported arms; D-S1B-001
  intact with no exception.
* **Audit 050** implemented and accepted the frozen UNMARK runtime path in
  `unmark/evaluation/stage2_dual_finalist.py` (998 lines) with
  `tests/test_stage2_dual_finalist_infra.py` (917 lines). It declares
  `STAGE2_HEAD_TRAINING_IMPLEMENTED = False` — the gap this audit fills.
* `docs/spec/stage2-dual-finalist-protocol.json` is `stage2-dual-finalist-protocol-v1`.

Nothing in the repository contradicted the stated scientific state.

---

## 3. What was reused rather than rebuilt

The single largest correctness risk in this task was re-implementing something
the project has already locked. Reused verbatim:

| From | Reused |
|---|---|
| `stage2_dual_finalist` (Audit 050) | `extract_stage2_unmark_representations` — the accepted forward pass; `Stage2UnmarkArm`, `finalist_for_arm`, `require_dual_finalist_state`, `require_stage2_condition`, the frozen condition grid and pooling/dtype constants |
| `preg1_head` (closed pre-G1) | `build_head`, `build_optimizer`, `deterministic_batches`, `EpochScore`, `select_checkpoint`, `require_full_schedule`, `score_predictions`, `require_protocol_settings`, `ordered_id_digest`, and **`Preg1Role`** |
| `preg1_protocol` | `EPOCHS`, `BATCH_SIZE`, `MEASUREMENT_SEEDS`, `MAX_LENGTH`, `TRUNCATION`, `PADDING`, dataset identity, `PRIMARY_NUM_LABELS` |
| `metrics` | `per_class_scores`, and `macro_f1`/`accuracy` through `score_predictions` |
| `stage1.checkpoint` | `atomic_write_bytes` (temp → fsync → replace → re-verify) |

**`Preg1Role` is reused deliberately.** It has no `OFFICIAL_TEST` member, so
official TEST cannot be *named* by the Stage-2 runner either — the seal is
structural rather than a check that could be forgotten.

Verified mechanically: the new module contains **zero** occurrences of
`base_word_embeddings`, `authoritative_position_ids`, `inputs_embeds` or
`last_hidden_state`, and a test asserts by AST that it *calls*
`extract_stage2_unmark_representations`. The frozen forward pass exists once.

---

## 4. Cache design

`Stage2RepresentationKey` is modelled on `preg1_head.RepresentationKey` and is
deliberately **wider**, because a Stage-2 tensor depends on more things. It binds
all 22 fields:

```
repository_head            arm                        finalist_checkpoint_sha256
backbone_checkpoint        backbone_revision          protocol_version
dataset                    dataset_version            task
role                       condition                  corruption_seed
pooling                    max_length                 truncation
padding                    ordered_id_digest          label_digest
dtype                      hidden_size                count
schema_version
```

* **Comparison is exact.** `require_compatible` diffs every field and refuses on
  any difference — a mismatch is an error, never a recomputation.
* **One arm can never read the other's cache.** `arm` *and*
  `finalist_checkpoint_sha256` are both bound: reusing A's vectors as B would
  make the two arms look identical, which is a plausible-looking non-result.
* **Commit and protocol are bound**, so a cache from another commit or a
  re-versioned protocol is refused rather than silently reused.
* **No raw text.** Sample identity travels as an ordered-id digest; labels as an
  order-sensitive `label_digest`, so a re-ordered label vector is caught.
* **Construction-time invariants**: `FULL` must carry *no* corruption seed; every
  degraded condition *must* carry one; pooling must be `FIRST_TOKEN`; dtype must
  be `torch.float32`; `hidden_size` must be 768.
* **Writes are atomic.** Metadata goes through `atomic_write_bytes`; the tensor is
  written to a temp path and renamed, so a crash mid-write leaves the previous
  cache intact rather than a truncated file a later run would load.
* **Immutable in practice**: re-saving the same key is idempotent; saving a
  *different* key into the same directory is refused.
* **Nothing is committed to git** — `*.pt` is already git-ignored.

`Stage2BoundRepresentations` carries the tensor *and* its key, so role, arm and
condition are **properties of the tensor**, not claims about it. There is no
argument anywhere by which a caller can declare a role.

---

## 5. Extraction plan

| Plan | Items |
|---|---|
| `stage2_training_extraction_plan()` | 4 — clean `FULL` × {protocol-train, protocol-dev} × {A, B} |
| `stage2_measurement_extraction_plan(corruption_seed=...)` | 12 — measurement-dev × 6 conditions × {A, B} |

`extract_and_cache_stage2_representations` is a thin driver: it checks the
pathway's bound arm and checkpoint against the cache key, delegates each batch to
the Audit-050 forward, concatenates in order, checks the row count against the
key, and stores. Its identity guards run **before** `import torch`, so a
cross-arm write is refused even on a machine with no torch.

**No measurement-dev row was materialised or inspected.** The code path is
exercised with synthetic fixtures only.

---

## 6. Head runner and optimizer isolation

`train_stage2_head` takes **tensors, never a pathway** — verified by a test that
inspects the signature for `pathway`, `encoder`, `adapter`, `model`, `tokenizer`.
Once representations are cached, head training cannot re-enter PhoBERT.

It enforces: role `protocol-train` for training and `protocol-dev` for scoring
(both read from provenance); clean `FULL` for both; same arm for both; labels
matching the cached digest; `learning_rate == 0.01`; `epochs == 30`; and a seed
drawn from the frozen five. All 30 epochs run — `require_full_schedule` checks the
evidence, not a flag — and selection happens afterwards.

`require_head_only_optimizer` compares parameter **object identity** between the
optimiser's param groups and the head. A frozen encoder or adapter tensor that
reached a param group would train silently while every downstream number still
looked reasonable, so name- or count-based checks were not sufficient. It refuses
both a foreign parameter and a missing head parameter.

---

## 7. Checkpoint selection

Per arm and per seed, independently: after each of the 30 epochs the head is
scored on **clean `FULL` protocol-dev**, and selection uses the existing locked
total order — highest macro-F1, then highest accuracy, then **earliest** epoch —
via the pre-G1 `select_checkpoint`, not a re-implementation.

A corrupted protocol-dev score cannot participate: `require_clean_condition`
refuses any non-`FULL` condition on the selection path, because selecting on
degraded scores would tune the head for robustness and break the "trained on
clean, then frozen" contract.

Persisted: selected epoch, the full 30-epoch history, the history digest, the
selected head state and its SHA-256, the initial-head fingerprint, seed, arm and
both cache identities.

---

## 8. Artifact and resume contract

`build_stage2_head_artifact` / `validate_stage2_head_artifact` bind a closed
23-field schema. Refused: an A artifact loading as B; a seed mismatch; a
checkpoint sha that is not that arm's; a drifted LR, epoch budget or stopping
rule; a selection role or condition that is not clean protocol-dev; a cache key
from another arm or another commit; **partial binding** (any missing field); and
any unknown field. `measurement_used_for_selection` and `ab_selection_performed`
must both be `false`.

**Resume contract: `atomic_rerun_no_midrun_resume`.** A head run is atomic —
complete, or absent.

> One run is 30 epochs of a linear head over cached `[N, 768]` FP32 vectors —
> minutes, not hours — and it is fully deterministic: head init is reseeded from
> `seed`, batch order is a pure function of `(seed, epoch)`, and no encoder or
> adapter runs. A restart reproduces the run bit-for-bit, which a torch test
> asserts directly. Persisting optimiser state would therefore add a resume
> surface that could reattach a partial run to a drifted identity while buying no
> scientific value.

`Stage2HeadRunStore` implements it: a completed artifact is **immutable** and
refuses overwrite; an in-progress marker records `(arm, seed)` and a partial run
whose identity differs is refused rather than adopted. This is the smallest
defensible fail-closed alternative, chosen after inspecting the repository's
existing conventions, and it is **reported for review rather than presented as a
new scientific choice**.

---

## 9. Campaign, measurement and the absence of selection

`stage2_campaign_plan()` returns exactly **10** runs — 2 arms × 5 frozen seeds —
and `require_paired_campaign_plan` refuses a third arm, an extra or unknown seed,
a duplicate, or a seed that is not paired across both arms.

`measure_stage2_head` scores a **frozen** selected head on measurement-dev; the
role is read from the tensor's key, so a protocol-dev tensor cannot be scored as
measurement and a measurement tensor cannot reach selection. It holds no
reference to any selection state and a test asserts it does not mutate the head.

### 9a. Campaign orchestration and registry

Review found that although each run bound its own commit, protocol and caches,
**nothing asserted that the ten runs together shared one identity**. A campaign
assembled from runs produced at two commits would look complete and be
meaningless, and no per-run check can see it. `unmark/evaluation/stage2_campaign.py`
closes that.

`Stage2CampaignManifest` binds one campaign identity:

| Field | Content |
|---|---|
| `repository_head` | the one commit every run and cache must share |
| `protocol_version` | `stage2-dual-finalist-protocol-v1` |
| `arms` | exactly `("UNMARK-A", "UNMARK-B")` |
| `seeds` | exactly `(53148, 59945, 42941, 720, 9428)` |
| `cache_keys` | four slots — `{A,B} x {protocol-train, protocol-dev}`, each a full `Stage2RepresentationKey` |
| `expected_runs` | exactly the ten `(arm, seed)` pairs |
| `schema_version` | `stage2-campaign-manifest-v1` |
| `digest` | SHA-256 over the serialised manifest; re-entry compares it |

`validate_stage2_campaign_manifest` fails closed on: a missing arm; a third arm; a
missing seed; an extra seed; a duplicate or unpaired run; a short run list; a
missing or unknown cache slot; a cache holding **the wrong arm** or the wrong
role; a corrupted (non-`FULL`) cache on the training path; a cache from a
**different commit** — "a campaign may not mix commits"; a cache from a
**different protocol**; a manifest protocol or schema that is not the frozen one;
and both arms binding the **same finalist checkpoint sha256**, which would make
the arms identical and any comparison meaningless.

`Stage2CampaignRegistry` persists the manifest with its digest and tracks
per-run completion. Re-entry is permitted **only under a byte-identical
manifest**: a drifted one is refused rather than adopted, so a partial campaign
cannot absorb run state produced under different caches, commit or protocol. A
completed run is skipped, never re-executed, and `require_writable` refuses to
overwrite it. `require_campaign_complete` refuses to report a partial campaign —
both arms and all five seeds, or nothing.

`campaign_status` is deliberately a **completion report, not a comparison**: it
carries counts and pending runs, and no score, delta or ranking. A test asserts
that — after excluding the two safety declarations whose purpose is to record the
absence — the serialised status contains no `macro_f1`, `accuracy`, `best`,
`rank`, `score` or `delta`.

`run_stage2_campaign` is the entry point. It **reads no dataset**: it takes the
four cache directories and the caller's label vectors, which the caches'
`label_digest` then verifies. Every step delegates —
`Stage2RepresentationCache.load`, `train_stage2_head`,
`build_stage2_head_artifact`, `validate_stage2_head_artifact`,
`Stage2HeadRunStore.commit`. Two AST tests enforce that: one asserts those calls
are present, the other that the module contains no `CrossEntropyLoss`,
`backward()`, `optimizer.step`, `macro_f1(`, `deterministic_batches(` or
`build_head(` — i.e. that training and metrics were not re-implemented. A third
asserts it references no `read_csv`, `.csv`, `uit-vsfc` or official-test name.

`aggregate_stage2_campaign` returns per-arm, per-condition mean and sd over the
five seeds, per-class F1, descriptive per-seed paired `B − A` deltas, and the two
frozen robustness summaries (`STRIP_ALL` macro-F1; equal-weight mean over
`P25, P50, P75, P100, STRIP_ALL`). It **requires both arms** and refuses a report
missing either, or one whose conditions do not cover all five seeds — a dropped
arm or seed is exactly what Audit 049 forbids.

Safeguards against A/B selection, all tested:

* `winner` is `None` and `ab_selection_performed` is `false` in the report;
* module constants `STAGE2_AB_WINNER_RULE = None`, `STAGE2_AB_TIE_BREAK = None`,
  `STAGE2_AB_SELECTION_IMPLEMENTED = False`;
* an **AST** test asserts no function or class in the module is named anything
  containing `winner`, `best_arm`, `rank_arms`, `select_arm`, `choose_arm`,
  `pick_finalist`, `drop_arm` or `promote_arm`;
* a test asserts the serialised report contains no `best_arm` / `winner_arm` /
  `selected_arm` / `ranking` key.

---

## 10. Unresolved: the measurement corruption seed

Implementation surfaced one value the frozen protocol does **not** resolve.

`docs/spec/stage2-dual-finalist-protocol.json` pins the corruption *mechanism*
— `unmark.corruption.corrupt`, `purpose=SCIENTIFIC`, keyed by `sample_id` and
never by row order — but pins **no seed value**. `apply_stage2_corruption` takes
`seed` from its caller.

That seed changes which corruption realisation each sample receives, and
therefore the absolute reported robustness numbers. It applies **identically to
both arms**, so it cannot bias the A-vs-B contrast — but it is still a scientific
value, and inventing one here would be an unrecorded choice.

**It was not invented.** `stage2_measurement_extraction_plan(corruption_seed=...)`
has **no default**, rejects a non-integer, and binds the value into every cache
key; `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED = False` records the fact. Two
tests assert the absent default. This follows the repository's own
`require_resolved` idiom: a value the project has not decided must not acquire a
default.

**Scope of the block, stated explicitly.** `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED`
stays `False`. This blocks **degraded measurement extraction** — the
`P25, P50, P75, P100, STRIP_ALL` caches on measurement-dev, which cannot be keyed
without it. It does **not** block clean `protocol-train` / `protocol-dev`
extraction, and it does **not** block head training: both are clean-only and use
no corruption seed at all. `FULL` measurement extraction is also unaffected,
since the clean condition binds no seed by construction.

It requires an author decision before the degraded measurement phase.

---

## 11. Files changed

| File | State | Purpose |
|---|---|---|
| `unmark/evaluation/stage2_head_campaign.py` | **new** | cache, extraction driver, head runner, selection, artifact, run store, campaign plan, measurement, aggregation |
| `unmark/evaluation/stage2_campaign.py` | **new** | campaign manifest, registry and entry point (§9a) |
| `tests/test_stage2_head_campaign.py` | **new** | torch-free contract tests (74) |
| `tests/test_stage2_head_campaign_torch.py` | **new** | tensor-level tests (29), skipped locally |
| `tests/test_stage2_campaign.py` | **new** | torch-free orchestration tests (31) |
| `tests/test_stage2_campaign_torch.py` | **new** | campaign-execution tests over synthetic caches (10), skipped locally |
| `tests/test_preg1_import_contract.py` | modified, 1 line | `stage2_head_campaign` is a protocol importer; the pinned `IMPORTERS` list must name it or the contract silently narrows |

`stage2_campaign.py` deliberately imports its protocol constants **through**
`stage2_head_campaign` rather than from `preg1_protocol`, so it is not a protocol
importer and the pinned list needs no second entry — verified by the contract
test, which discovers importers rather than trusting the list.

The torch/torch-free split follows the warning in
`tests/test_stage1_device_contract.py`: a module-level `importorskip` would skip
the structural checks too. The torch file uses the per-test `skipif` marker of
`test_stage2_dual_finalist_infra.py`, so its tests stay **collected** and their
skips are visible rather than the whole file vanishing.

---

## 12. Tests

```
.venv/bin/python -m pytest -q \
  tests/test_stage2_head_campaign.py tests/test_stage2_head_campaign_torch.py

74 passed, 29 skipped
```

Orchestration (§9a):

```
.venv/bin/python -m pytest -q \
  tests/test_stage2_campaign.py tests/test_stage2_campaign_torch.py

31 passed, 10 skipped
```

All four Stage-2 files together: **105 passed, 39 skipped**.

The 10 orchestration skips are the campaign-execution tests, which build four
synthetic caches and run all ten head runs end to end; they need torch. They
cover: the full ten-run campaign; re-entry executing nothing; a partial campaign
resuming only its pending runs; a completed run never re-executed or overwritten
(asserted on the artifact bytes); matched seeds producing identical initial heads
but distinct results; a caller head disagreeing with the campaign; a missing
cache directory or label vector; a cache directory holding the other arm; a run
outside the campaign; and every committed artifact validating against its own
identity and refusing the other arm's.

All 29 head-campaign skips are the torch module: **torch is not installed in this environment**
(`ModuleNotFoundError`), and per standing instruction it was not installed. Those
29 tests were **collected**, not silently absent, and are **not reported as
passing**. They cover: A/B bit-identical pairing at the same seed and its survival
of intervening RNG consumption; different seeds diverging; zero bias; no shared
weights; optimiser isolation including a foreign and a missing parameter;
weight-decay split; the full 30-epoch schedule; selection equal to the locked
total order; independent per-arm training; bit-identical rerun; the frozen budget
and seed guards; role and condition refusals; label-digest mismatch; cache round
trip, cross-arm refusal, overwrite refusal and tensor/key contradiction; artifact
and store lifecycle; measurement scoring and non-mutation; and an **anti-drift
test asserting the Stage-2 epoch loop reproduces the closed pre-G1 `train_head`
scores exactly on identical inputs**.

Related suites:

```
tests/test_stage2_dual_finalist_infra.py tests/test_preg1_head.py
tests/test_preg1_runner.py tests/test_preg1_split.py tests/test_preg1_profiling.py
tests/test_preg1_import_contract.py tests/test_evaluation_harness.py
tests/test_stage1_finalist_freeze.py tests/test_stage1_finalist_checkpoint_torch.py

570 passed, 63 skipped
```

Corruption and metrics selection: `596 passed, 8 skipped`.

Whole repository:

```
.venv/bin/python -m pytest -q

4293 passed, 169 skipped in 154.57s (0:02:34)
```

Zero failures and zero errors. The 169 skips are the repository's pre-existing
torch/CUDA-gated tests plus this audit's 39.

```
git diff --check
```

produced no output.

---

## 13. Manual diff inspection

| Risk | Finding |
|---|---|
| Scientific constant drift | none — every frozen value is *imported*; a check compared `epochs`, `batch_size`, seeds, LR and arm count against their source constants |
| TEST exposure | none — `Preg1Role` has exactly `protocol-train`, `protocol-dev`, `official-validation`; an AST test asserts no identifier or non-docstring literal reaches an `OFFICIAL_TEST` role |
| Hidden A/B-selection code | none — AST scan of every definition found nothing matching the banned name set |
| Duplicated Stage-2 pathway logic | none — 0 forward-pass symbols in the new module; it calls the Audit-050 function |
| Hard-coded Drive paths | none — no `/content/`, no `MyDrive` |
| Accidental downstream artifacts | none — no `.csv`, no dataset file in the tree |
| Unrelated changes | none — 6 new implementation/test files, 1 new audit, and a 1-line pinned-list extension |

---

## 14. Final state

```
git status --short
 M tests/test_preg1_import_contract.py
?? docs/audits/051-stage2-head-campaign-runner-implementation.md
?? tests/test_stage2_campaign.py
?? tests/test_stage2_campaign_torch.py
?? tests/test_stage2_head_campaign.py
?? tests/test_stage2_head_campaign_torch.py
?? unmark/evaluation/stage2_campaign.py
?? unmark/evaluation/stage2_head_campaign.py
```

HEAD unchanged at `6ca3d0f90215ca0e49ad14d5f9dc43c2e23418d4`; nothing staged,
nothing committed, nothing pushed, no git history mutated.

---

## 15. Limitations

1. **The torch half was not executed here.** §12. It is collected and skipped;
   the tensor-level guarantees (pairing, optimiser isolation, 30-epoch schedule,
   cache round trip) are unverified in this environment and must run where torch
   exists.
2. **The measurement corruption seed is unresolved.** §10. Required before
   measurement; does not block head training.
3. **The epoch loop is written once in this module** rather than shared with
   `preg1_head.train_head`, because the Stage-2 role types are arm-bound and
   cannot be expressed by the pre-G1 `RepresentationKey`. Mitigated by an
   anti-drift test asserting identical scores on identical inputs — but that test
   needs torch, so the mitigation is currently unexecuted here too.
4. **[RESOLVED]** An orchestrator now exists (§9a): `run_stage2_campaign` walks
   the pending runs over already-materialised caches and reads no dataset. Its
   execution path is torch-gated and unexecuted here.
5. **[RESOLVED]** `Stage2CampaignRegistry` supplies the cross-run registry: one
   manifest binds the commit, protocol, both arms, the five seeds, the four cache
   identities and the ten expected runs, and re-entry requires a byte-identical
   manifest digest.
6. **The campaign entry point assumes caches already exist.** It validates their
   keys but does not extract them; extraction remains the separate driver in
   `stage2_head_campaign`. That split is deliberate — the campaign runner should
   not be able to trigger a forward pass — but it means an operator must run
   extraction first and the two steps are not yet chained by a script.
7. **No script under `scripts/` exposes any of this on a CLI.** Everything is
   importable API. Adding a CLI invites running it, so it is left for the task
   that actually authorises the campaign.

---

## 16. Exact next allowed step

**Not** "run Stage 2." In order:

1. **Independently review this audit** and the seven files, and commit them.
2. **Execute the torch suite** on a machine with torch and confirm all **39**
   skipped tests pass — including the anti-drift equivalence test and the ten
   campaign-execution tests. This is the *authoritative runner acceptance* this
   audit does not claim, and it is the blocker for Stage-2 training.
3. **Decide the measurement corruption seed** (§10) and record it — a spec
   amendment plus a decision record. It blocks **degraded** measurement
   extraction only, not head training.
4. Only then may representation extraction run — clean `protocol-train` and
   `protocol-dev` first, which need no corruption seed at all.

---

```
STAGE2_PROTOCOL_FROZEN=YES
AUDIT049_PROTOCOL_AUTHORITY=UNCHANGED
AUDIT050_RUNTIME_PATH=PASS
STAGE2_RUNNER_IMPLEMENTED=YES
STAGE2_CACHE_IMPLEMENTED=YES
STAGE2_HEAD_RUNNER_IMPLEMENTED=YES
STAGE2_MEASUREMENT_RUNNER_IMPLEMENTED=YES
STAGE2_CAMPAIGN_ORCHESTRATION_IMPLEMENTED=YES
STAGE2_CAMPAIGN_REGISTRY_IMPLEMENTED=YES
STAGE2_RUNNER_STATIC_IMPLEMENTATION=PASS
AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PENDING
STAGE2_AB_SELECTION_IMPLEMENTED=NO
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_TRAINING_STARTED=NO
READY_FOR_STAGE2_RUNTIME_ACCEPTANCE=YES
READY_FOR_STAGE2_TRAINING=NO
```

`READY_FOR_STAGE2_RUNTIME_ACCEPTANCE=YES`: the implementation and its 39
torch-gated tests are complete and collected, so the acceptance run can proceed
on a machine with torch. `READY_FOR_STAGE2_TRAINING=NO` until that run passes.
