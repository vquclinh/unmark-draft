# Audit 051 - Stage-2 Head Campaign: Cache, Runner and Artifact Implementation

**Scope:** implement the Stage-2 runner infrastructure that must exist *before*
head training — representation cache, cached-only head runner, checkpoint
selection, head artifacts, run store, campaign plan and measurement/reporting
path.
**Date:** 2026-09-07
**Type:** **implementation audit, not a protocol amendment.** No scientific
decision is appended.
**Runtime closeout:** 2026-09-08. §14 and §15 were appended after the
authoritative A100 acceptance (2026-09-07) of the committed implementation at
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7`. That closeout is documentation-only:
no implementation file, test, protocol constant or scientific decision was
changed by it, and the pre-acceptance record in §12 is preserved rather than
rewritten.

---

## 1. Executive verdict

**PASS.**

The static implementation is complete and its identity, role and selection-safety
contracts are fully exercised. **Runtime acceptance is now proved as well.** When
§12 was written the 39 torch-level tests were collected and skipped, and this
audit declined to claim their guarantees; they have since been executed on an
authoritative NVIDIA A100 runtime at the exact tested commit
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7`, where the four Stage-2 files reported
`144 passed, 0 failed, 0 errors, 0 skipped` — §15. The runtime-critical
guarantees are therefore evidenced rather than contracted: matched A/B
initialisation, optimiser isolation, 30-epoch execution, cache tensor round trip,
artifact and run-store lifecycle, deterministic rerun, campaign execution, and
the Stage-2/pre-G1 anti-drift equivalence.

That acceptance was executed against **synthetic fixtures only**. It proves the
runner runs; it does not begin Stage 2. No UIT-VSFC row was read, no real
representation was extracted, and no real head was trained — §15.

| | |
|---|---|
| Audit 049 remains protocol authority | **YES** — nothing here changes it |
| Audit 050 accepted the frozen UNMARK runtime path | **YES** — reused, never duplicated |
| Runner code implemented now | **YES** |
| Static implementation acceptance | **PASS** |
| **Authoritative runner torch acceptance** | **PASS** — §15, A100, `144 passed, 0 skipped` |
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
| `tests/test_stage2_head_campaign_torch.py` | **new** | tensor-level tests (29), skipped locally, executed and passed in §15 |
| `tests/test_stage2_campaign.py` | **new** | torch-free orchestration tests (31) |
| `tests/test_stage2_campaign_torch.py` | **new** | campaign-execution tests over synthetic caches (10), skipped locally, executed and passed in §15 |
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

Those 39 were subsequently executed and passed on the authoritative A100 runtime;
this local section is preserved as the static-acceptance record, and §15 is the
runtime one.

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

## 14. Authoritative runtime — preflight prerequisite on the first invocation

Commit tested:

```
6693e728ccaebc987e4786bd5cd5e0f5c16143f7
```

The **first** invocation of the authoritative runtime cell stopped inside the
**Stage-1 finalist verifier**, before any Stage-2 module was reached: the fresh
clone did not carry the pinned Vietnamese syllable inventory.

**This is an environment/preflight prerequisite, not a Stage-2 runner defect.**
The raw inventory is deliberately not committed. The manifest
`configs/linguistics/vietnamese_syllables.yaml` records that the upstream gist
carries no license statement, so redistribution permission is not established,
and the fetch cache `.resources-cache/` is git-ignored. A fresh clone therefore
*never* has it. `resolve_inventory()` in `unmark/stage1/finalists.py` fails
closed on exactly this condition and names the remedy itself:

> the pinned Vietnamese syllable inventory is not available, so a finalist's
> inventory identity cannot be checked. Provision it with
> `scripts/fetch_vietnamese_syllable_inventory.py` and re-run.

The guard exists because D-S1A-008 makes the `inventory` half of the provenance
comparison load-bearing; degrading instead of stopping would silently skip that
half. The stop is the repository's own designed behaviour, in Stage-1 provenance
code that this audit does not touch.

**Provisioning changed no pin.** The inventory was obtained only through the
repository's authoritative fetch-and-verify script, at the already-pinned
identity:

| | |
|---|---|
| Provisioned by | `scripts/fetch_vietnamese_syllable_inventory.py` |
| sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| size | `116290` bytes |
| Pin changed | **NO** — byte-identical to `configs/linguistics/vietnamese_syllables.yaml`, to `unmark/stage1/finalists.INVENTORY_SHA256` / `INVENTORY_SIZE_BYTES`, and to the Audit-048 freeze |

The script never advances the pin: a changed upstream fails its checksum and it
refuses, because changing the inventory revision would change which spans are
eligible and therefore every corruption denominator. The complete runtime
acceptance cell was then **rerun unchanged** and passed — §15.

---

## 15. Authoritative runner runtime acceptance — PASS

Exact tested repository SHA:

```
6693e728ccaebc987e4786bd5cd5e0f5c16143f7
```

Authoritative runtime:

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |

### Audit-051 suite

```
tests/test_stage2_head_campaign.py  tests/test_stage2_head_campaign_torch.py
tests/test_stage2_campaign.py       tests/test_stage2_campaign_torch.py

144 passed, 0 failed, 0 errors, 0 skipped
```

`144 = 105 + 39`: the tests that passed locally plus **every one of the 39 that
§12 could only collect and skip**. Zero skips is the load-bearing figure — it is
what distinguishes "the torch tests ran" from "the torch file was absent". The
runtime acceptance §1 originally declined to claim is now claimed on evidence.

### Critical regression suites

| Suite | Result | Meaning |
|---|---|---|
| `tests/test_stage2_dual_finalist_infra.py` | **37 passed, 0 skipped** | the accepted Audit-050 runtime path is unbroken by this audit's additions — the same 37 Audit 050 recorded |
| `tests/test_preg1_import_contract.py` | **18 passed, 0 skipped** | the pre-G1 protocol-importer contract holds with `stage2_head_campaign` in the pinned list |
| `tests/test_stage1_finalist_checkpoint_torch.py` | **40 passed, 0 skipped** | the Stage-1 finalist verifier is unchanged — the same 40 Audit 050 recorded |

The importer contract moved from Audit 050's **16** to **18**. That delta is
exactly this audit's change and not drift: `IMPORTERS` grew from seven entries to
eight (§11), and two tests are parametrised over it, so one new importer adds two
cases.

### Runtime acceptance proved

| Guarantee | Result |
|---|---|
| Full synthetic 2-arm × 5-seed = 10-run campaign | **PASS** |
| Matched A/B same-seed head initialisation | **PASS** |
| Head-only optimiser isolation | **PASS** |
| Exact full 30-epoch execution contract | **PASS** |
| Checkpoint-selection runtime | **PASS** |
| Representation-cache tensor runtime | **PASS** |
| Artifact / run-store lifecycle | **PASS** |
| Deterministic rerun | **PASS** |
| PREG1 anti-drift equivalence | **PASS** |
| Audit-050 infrastructure regression | **PASS** |
| A/B-selection implementation | **NONE** |

The anti-drift equivalence is the one §15 result that Limitation 3 was written
against: the Stage-2 epoch loop reproduces the closed pre-G1 `train_head` scores
exactly on identical inputs. That mitigation is now executed rather than assumed.

### Campaign plan observed exactly

```
UNMARK-A / 53148
UNMARK-B / 53148
UNMARK-A / 59945
UNMARK-B / 59945
UNMARK-A / 42941
UNMARK-B / 42941
UNMARK-A / 720
UNMARK-B / 720
UNMARK-A / 9428
UNMARK-B / 9428
```

Ten runs: exactly the two frozen arms across exactly the five frozen measurement
seeds `(53148, 59945, 42941, 720, 9428)`, paired at every seed, in plan order. No
eleventh run, no unpaired seed, no third arm.

### Durable evidence

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-runner-runtime-acceptance/6693e728ccae/20260907T103607Z/051-authoritative-runner-runtime-final.json
```

### Negative evidence preserved

* UIT-VSFC rows read: **NO**;
* real `protocol-train` read: **NO**;
* real `protocol-dev` read: **NO**;
* `measurement-dev` read: **NO**;
* official TEST read: **NO**;
* real representation extraction: **NO**;
* real Stage-2 head training: **NO**;
* real Stage-2 campaign run: **NO**;
* training performed: **synthetic test fixtures only**.

Every campaign, head-training and cache execution above ran on tensors and label
vectors constructed inside the tests. §15 proves the runner *executes* the frozen
protocol correctly. It does not start Stage 2, and no downstream number exists.

---

## 16. Final state

Two states, deliberately kept apart: the **committed implementation** that the
A100 accepted, and the **uncommitted documentation closeout** that records the
acceptance. Conflating them would let a reader read §15's evidence as covering
this file's own text.

### A. Committed implementation state — the state tested on A100

```
6693e728ccaebc987e4786bd5cd5e0f5c16143f7  Implement Stage-2 head campaign runner
```

The Audit-051 change set of §11 was reviewed and committed by the author as that
commit, on `main`, whose parent is `6ca3d0f90215ca0e49ad14d5f9dc43c2e23418d4` —
the HEAD §2 recorded as this audit's starting state. That is the exact
implementation commit the authoritative runtime tested (§14, §15), and the
working tree was **clean** when it was tested: the acceptance ran against
committed code and nothing else.

### B. Current documentation-closeout working state — NOT clean

```
git status --short
 M docs/audits/051-stage2-head-campaign-runner-implementation.md
```

`git diff --check` produced no output. `HEAD` is still
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7`.

This closeout is **uncommitted and awaiting author review**. It modifies exactly
one path — this audit file — and nothing else: no implementation file, no test,
no spec, no protocol constant, no decision record. Nothing has been staged,
committed, pushed or history-mutated by the closeout.

State B therefore does **not** carry the §15 acceptance: the evidence in §14 and
§15 is about state A. When the author commits this file, state B becomes a
documentation commit on top of the tested commit, and the tested implementation
bytes are unchanged by it.

---

## 17. Limitations

1. **[RESOLVED]** The torch half was unexecuted when §12 was written. It has since
   been executed on the authoritative A100 runtime — §15, which reported
   `144 passed, 0 failed, 0 errors, 0 skipped`. The tensor-level guarantees
   (pairing, optimiser isolation, 30-epoch schedule, cache round trip) are
   verified.
2. **The measurement corruption seed is unresolved.** §10.
   `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED` remains `False`. It blocks
   **degraded measurement-dev extraction** only — the
   `P25, P50, P75, P100, STRIP_ALL` caches, which cannot be keyed without it. It
   does **not** block clean `protocol-train` extraction, clean `protocol-dev`
   extraction, head training, or `FULL` measurement extraction. It requires an
   author decision before the degraded measurement phase, and it was not
   invented here.
3. **The epoch loop is written once in this module** rather than shared with
   `preg1_head.train_head`, because the Stage-2 role types are arm-bound and
   cannot be expressed by the pre-G1 `RepresentationKey`. The mitigation — an
   anti-drift test asserting identical scores on identical inputs — **executed and
   passed** in §15, so the duplication is now empirically bounded rather than
   argued.
4. **[RESOLVED]** An orchestrator exists (§9a): `run_stage2_campaign` walks the
   pending runs over already-materialised caches and reads no dataset. Its
   execution path passed in §15.
5. **[RESOLVED]** `Stage2CampaignRegistry` supplies the cross-run registry: one
   manifest binds the commit, protocol, both arms, the five seeds, the four cache
   identities and the ten expected runs, and re-entry requires a byte-identical
   manifest digest.
6. **The campaign entry point assumes caches already exist.** It validates their
   keys but does not extract them; extraction remains the separate driver in
   `stage2_head_campaign`. That split is deliberate — the campaign runner should
   not be able to trigger a forward pass — but it means an operator must run
   extraction first and the two steps are not yet chained by a script. §18 step 2
   is that extraction.
7. **No script under `scripts/` exposes any of this on a CLI.** Everything is
   importable API. Adding a CLI invites running it, so it is left for the task
   that actually authorises the campaign.
8. **Runtime acceptance is synthetic-only.** §15 proves execution, not science.
   No UIT-VSFC row has been read, no real representation cache exists, no real
   head has been trained, and no downstream result has been observed. The frozen
   UIT-VSFC protocol split has not been materialised yet — that is §18 step 1.
9. **A fresh clone needs the inventory preflight.** §14. Any future authoritative
   runtime on a new machine must provision the pinned inventory through
   `scripts/fetch_vietnamese_syllable_inventory.py` before the Stage-1 finalist
   verifier will run. This is a documented prerequisite of the environment, not a
   defect of any runner.

---

## 18. Exact next allowed step

**Not** "run Stage 2." In order, after the author commits this closeout:

1. **Materialise and verify the frozen UIT-VSFC protocol split.** The deterministic,
   label-stratified, group-aware split already pinned by the protocol: seed
   `17486`, tag `UNMARK-PREG1-SPLIT-UITVSFC-v1`, `protocol-train` 0.8 /
   `protocol-dev` 0.2, drawn from the **official train split only**. Verify its
   digests before anything consumes it.
2. **Extract exactly four clean representation caches** — precisely
   `stage2_training_extraction_plan()`, no more:
   * `UNMARK-A` · `protocol-train` · `FULL`
   * `UNMARK-A` · `protocol-dev` · `FULL`
   * `UNMARK-B` · `protocol-train` · `FULL`
   * `UNMARK-B` · `protocol-dev` · `FULL`
3. **Verify cache identities and digests** — every `Stage2RepresentationKey` field,
   both the `ordered_id_digest` and the order-sensitive `label_digest`, and the
   four cache slots of the campaign manifest, before any head is built.

**Only after step 3 passes** may `READY_FOR_STAGE2_TRAINING` transition from `NO`
to `YES`. That transition is the authorisation boundary of this audit: steps 1–3
are execution on frozen inputs and produce no trained parameter, while step 4 is
real Stage-2 head training. The gate is what keeps a verified-cache failure from
silently becoming a training run.

4. **Only then launch the frozen 10-head campaign** — `run_stage2_campaign` over
   those four verified caches: 2 arms × 5 seeds, exactly the plan in §15. Not
   before `READY_FOR_STAGE2_TRAINING=YES`.

All four caches are clean `FULL`, which binds **no** corruption seed by
construction. Steps 1–3 are therefore *not* blocked by the unresolved value in
§10; they are blocked only by their own verification.

Explicitly not permitted yet:

* **no measurement-dev degraded conditions** — `P25`, `P50`, `P75`, `P100` and
  `STRIP_ALL` stay blocked until the corruption seed is decided and recorded
  (§10); do not touch them at this step;
* **no official TEST** — SEALED, and structurally unnameable because `Preg1Role`
  has no `OFFICIAL_TEST` member;
* **no A-vs-B selection**, at any point, under Audit 049 / D-S2-001 option (c).

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
AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PASS
STAGE2_AB_SELECTION_IMPLEMENTED=NO
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_TRAINING_STARTED=NO
READY_FOR_CLEAN_REPRESENTATION_EXTRACTION=YES
READY_FOR_STAGE2_TRAINING=NO
```

`AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PASS` is the state change this closeout
records, on the §15 evidence. `STAGE2_TRAINING_STARTED=NO` is unchanged and
remains true: the §15 campaign was synthetic.

The two readiness flags are a deliberate pair, and they are not the same gate.

`READY_FOR_CLEAN_REPRESENTATION_EXTRACTION=YES` is scoped to §18 **steps 1–3**:
materialise and verify the frozen split, extract the four clean `FULL` caches,
verify their identities and digests. Runner runtime acceptance is complete, so
that execution may begin. It does **not** authorise degraded measurement-dev
extraction, which stays blocked while
`STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False`, and it does not touch TEST.

`READY_FOR_STAGE2_TRAINING=NO` still holds, and step 4 — the frozen 10-head
campaign — is not authorised. Real head training becomes permissible only once
the frozen split is materialised and the four clean `FULL` caches are extracted
**and verified**; at that point, and not before, this flag transitions to `YES`.
An accepted runner is a runner that executes correctly, not a licence to train on
inputs whose identity nobody has checked.

`READY_FOR_STAGE2_RUNTIME_ACCEPTANCE` is deliberately absent: that gate is spent,
discharged by §15, and reintroducing it would imply an acceptance still owed.
