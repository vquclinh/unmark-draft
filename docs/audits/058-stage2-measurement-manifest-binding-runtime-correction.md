# Audit 058 - Stage-2 Measurement Manifest Binding: Runtime Correction

**Scope:** correct one incorrect manifest-identity binding in Audit 057, using
authoritative runtime evidence and committed source. Supersedes **only** that
binding.
**Date:** 2026-09-09
**Type:** **DOCUMENTATION-ONLY RUNTIME CORRECTION. NO SCIENTIFIC DECISION.** No
decision record is appended. No implementation, test, protocol, spec, config,
scientific constant, cache, head artifact or registry object changes.

---

## 1. Classification

Audit 057 §6.1a bound the raw SHA-256
`4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` to
`Stage2CampaignRegistry(...).manifest_path`. **Those are two different
artifacts**, and the runtime proved it.

```
AUDIT057_MANIFEST_RAW_SHA_BINDING=DOCUMENTATION_BUG
NOTEBOOK_DRAFT_V1_MANIFEST_PREFLIGHT=BUG
HISTORICAL_CAMPAIGN_CORRUPTION=NO
SCIENTIFIC_IMPLEMENTATION_FAILURE=NO
RETRAIN_REQUIRED=NO
```

**The bug is exactly the manifest artifact identity / raw-SHA binding** in
Audit 057 and notebook draft v1 — nothing wider. No production code was wrong,
no historical evidence was corrupted, and nothing needs retraining.

**Committed source shows this was diagnosable without the runtime.**
`Stage2CampaignRegistry` writes its manifest file as

```python
payload = {"manifest_digest": manifest.digest, "manifest": manifest.to_dict()}
```

— a **two-key wrapper**, under `MANIFEST_NAME = "stage2-campaign-manifest.json"`.
The Audit-052 pre-training artifact is a different file with a different shape.
Audit 057 asserted one file's byte digest of the other. The preflight it
specified could never have passed, which is why notebook v1 stopped where it did.

---

## 2. Authoritative runtime evidence

Execution HEAD `57b3d31e1cf8145835117efee43a3f5172075ab8`, worktree clean.
Historical campaign root:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1
```

### 2.1 Pre-training verified manifest artifact

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-clean-cache-preparation/6693e728ccae/20260908T014348Z/real-clean-cache-extraction/stage2-campaign-manifest-verified.json
```

| | |
|---|---|
| Bytes | `5019` |
| Raw SHA-256 | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` |
| Top-level keys (7) | `arms`, `cache_keys`, `expected_runs`, `protocol_version`, `repository_head`, `schema_version`, `seeds` |

**This is the artifact `4a940d87…` gates — and only this artifact.**

### 2.2 Historical campaign registry wrapper

`registry.manifest_path`:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1/stage2-campaign-manifest.json
```

| | |
|---|---|
| Bytes | `5844` |
| Raw SHA-256, runtime-observed | `9020c81bc486e892e4f5d96f66750a6770c16cbd67c69c6c72269fc85daf7899` |
| Top-level keys (exactly 2) | `manifest`, `manifest_digest` |
| `manifest_digest` | `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` |

The two-key shape is exactly what source writes (§1).

**`9020c81b…` must NOT be promoted into a frozen protocol or scientific
constant.** It is **runtime-observed raw wrapper-byte evidence only**. It records
what the wrapper's bytes were when observed; it is not an identity the project
has frozen, and no future gate should require it.

### 2.3 Nested manifest vs pre-training artifact — not object-equivalent

Nested manifest keys (10): `arms`, `cache_keys`, `expected_runs`,
`head_campaign_schema_version`, `protocol_version`, `ranks_arms`,
`repository_head`, `schema_version`, `seeds`, `winner`.

```
DIRECT_OBJECT_EQUALITY=False
ONLY_PRETRAIN_KEYS=[]
ONLY_REGISTRY_KEYS=["head_campaign_schema_version", "ranks_arms", "winner"]
DIFFERING_COMMON_KEYS=["cache_keys"]
```

**Committed source accounts for the three extra keys exactly.**
`Stage2CampaignManifest.to_dict()` emits precisely ten keys —
`schema_version`, `head_campaign_schema_version`, `repository_head`,
`protocol_version`, `arms`, `seeds`, `cache_keys`, `expected_runs`, `winner`,
`ranks_arms` — which is the observed nested set. So the registry's nested
manifest **is** `manifest.to_dict()`, repository-produced, while the
pre-training file is a different, seven-key serialisation produced by the
Audit-052 notebook. `ONLY_REGISTRY_KEYS` is therefore explained by source, not
by conjecture.

**`DIFFERING_COMMON_KEYS=["cache_keys"]` is recorded and left unexplained.**
Source shows `to_dict()` renders `cache_keys` as
`{slot: k.to_dict() for slot, k in sorted(self.cache_keys.items())}`.

**Slot ordering cannot be the explanation.** The `sorted(...)` there affects
insertion order during serialisation only; Python dictionary equality **ignores
key order entirely**, so two parsed dicts differing only in ordering compare
equal. The observed inequality must therefore lie in the **slot set** or in some
**value** beneath it — a differing per-key field, a differing type, or a
missing/extra slot or field.

**Which of those it is remains unresolved**, and is not guessed here: the
available source and supplied evidence do not establish it, and determining it
requires comparing the two payloads, which this documentation task did not do.
**No cause is asserted without committed source or evidence proving it.** Gate D
step 5 does not depend on the answer — it compares the stored tree against the
rebuilt serialisation directly.

Diagnostic-only canonical JSON hashes:

| Payload | Canonical hash |
|---|---|
| pre-training | `bd4059ac13071be1c2661a5e6a5ca71d02e6ad6173ec8703c3175abcfea13f5d` |
| registry nested manifest | `d538a83bcb24cacd39b3af80986bf8f81e8fa517c50b0f1d4d5458e806e938b7` |

```
CANONICAL_DIAGNOSTIC_EQUAL=False
```

**These two hashes are diagnostic only.** They are **not** protocol values, not
frozen scientific identities, not registry semantic digests, and not replacement
manifest gates. **They must not be used in future scientific execution.**

---

## 3. The historical clean campaign is intact

Final evidence:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1/stage2-clean-10-head-campaign-final.json
```

| | |
|---|---|
| Raw SHA-256 | `69dc1447f1c340c18285f363d1ebfaa7d5d93f954e487e2fd99c4f556e9faf59` |

Fields observed:

| Field | Value |
|---|---|
| `accepted_pretraining_manifest_file_sha256` | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` |
| `manifest_digest` | `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` |
| `campaign_status.complete` | `true` |
| `campaign_status.completed` | `10` |
| `campaign_status.pending` | `[]` |
| `campaign_status.ranks_arms` | `false` |
| `campaign_status.winner` | `null` |
| `state_after_campaign.stage2_training_complete` | `true` |
| `state_after_campaign.ranks_arms` | `false` |
| `state_after_campaign.winner` | `null` |

Note that this file **names the pre-training manifest SHA under a field that says
so** — `accepted_pretraining_manifest_file_sha256` — beside a separate
`manifest_digest`. The distinction Audit 057 collapsed was already recorded here.

```
STAGE2_CLEAN_HEAD_CAMPAIGN=PASS
STAGE2_HEAD_RUNS_COMPLETED=10_OF_10
UNMARK_A_HEADS_COMPLETED=5_OF_5
UNMARK_B_HEADS_COMPLETED=5_OF_5
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE
```

**No historical head, cache, artifact or registry object is to be rewritten. No
retraining is permitted or required.**

### 3.1 What is historically accepted, and what is still pending

These are different claims and this audit keeps them apart.

**Historically accepted, unchanged by this correction:** Stage-2 head training
completed **10 of 10**; the final training evidence file matched its
authoritative raw SHA-256 `69dc1447…`; and the wrong-file SHA comparison of §1
**does not justify retraining** — it compared one file's digest against a
different file, which says nothing about either file's contents.

**Still pending, and not claimed here:**

* the diagnostics **did not** newly verify all ten selected-head state files —
  they examined manifest artifacts, not `stage2-selected-head.pt` bytes;
* **full live manifest reconstruction and the complete stored-versus-rebuilt
  comparison** of gate D step 5 remain to be run;
* **ten-head state revalidation** — existence, load, `_state_digest` versus
  `selected_head_state_sha256` — remains to be run.

Both remaining items must complete in notebook v2 **before `measurement-dev` is
read**. **The current preflight is not described as fully passed**, because it is
not: it failed at its first gate, and the gates after it never ran.

---

## 4. Reconciling Audit 053 and Audit 057

**Audit 053 was correct.** Its §9 distinguishes two identities of the manifest —
the accepted pre-training **file** SHA-256 `4a940d87…` and the campaign manifest
**semantic** digest `571100c3…` — and says in terms that they are "different
values of different things".

**Audit 057 made an incorrect inference.** It took the pre-training artifact's
file SHA and applied it to `registry.manifest_path`, and described the two
payloads as one identical manifest artifact. That inference is what this audit
supersedes.

**Audit 058 supersedes ONLY** that incorrect binding and the prose treating the
two payloads as one artifact. **Audit 053 is not edited and none of its
historical values were wrong.** Audit 057 is not edited either — audit history is
append-only, and the record of what was believed is part of the evidence.

---

## 5. The corrected preflight chain

Four gates, in order. Each is independently anchored; none certifies itself.

### A. Pre-training verified manifest

Require `stage2-campaign-manifest-verified.json` at the authoritative Audit-052
preparation path (§2.1), with raw SHA-256 exactly:

```
4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962
```

**This SHA applies only to this file. It must not be compared against
`registry.manifest_path`.**

### B. Final clean-campaign evidence

Require `stage2-clean-10-head-campaign-final.json` with raw SHA-256 exactly:

```
69dc1447f1c340c18285f363d1ebfaa7d5d93f954e487e2fd99c4f556e9faf59
```

Then require its recorded facts:
`accepted_pretraining_manifest_file_sha256 == 4a940d87…`;
`manifest_digest == 571100c3…`; `complete == true`; `completed == 10`;
`pending == []`; `ranks_arms == false`; `winner == null`;
`stage2_training_complete == true`.

This yields an **independently raw-hash-gated closeout anchor**: a file whose own
bytes are gated, which then vouches for both manifest identities.

### C. Live historical registry wrapper

Instantiate `Stage2CampaignRegistry` at the known campaign root. Require
`registry.manifest_path` exists — **filename from the property, never invented**
(source: `MANIFEST_NAME = "stage2-campaign-manifest.json"`). Parse in the
repository-compatible format and require the wrapper structure source defines —
the two keys `manifest_digest` and `manifest`. Then require:

```
wrapper["manifest_digest"] == 571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6
```

* **Do NOT require the wrapper's raw SHA to equal `4a940d87…`** — that was the bug.
* **Do NOT freeze `9020c81b…`** merely because it was observed at runtime.

### D. Rebuild the semantic identity from repository APIs

Using only existing APIs — no second serializer:

```
Stage2RepresentationKey.from_dict(...)
build_stage2_campaign_manifest(repository_head=..., cache_keys=...)
validate_stage2_campaign_manifest(...)
```

#### D.1 Rebuilding alone does not prove the stored manifest is intact

**Reconstruction normalises.** `build_stage2_campaign_manifest` accepts only
`repository_head` and `cache_keys`; it draws `protocol_version`, `arms`, `seeds`
and `expected_runs` from frozen module constants, and `to_dict()` emits
`schema_version`, `head_campaign_schema_version`, `winner` and `ranks_arms` from
constants too. **A drifted stored value in any of those eight fields is discarded
and replaced during reconstruction**, so the rebuilt digest is unchanged and the
drift is invisible.

Two further limits, from source:

* `Stage2CampaignRegistry.open` compares `recorded.get("manifest_digest")` — a
  **stored string** — against the rebuilt object's digest. It never recomputes a
  digest over the stored nested `manifest` body, so a body that disagrees with
  its own recorded digest is not detected.
* `Stage2RepresentationKey.from_dict` reads all 22 key fields, so a *missing* or
  *changed* cache-key field does propagate into the rebuild and does change the
  digest. But an **unknown or extra** field in a stored cache-key dict is never
  read, and is silently dropped.

So rebuilding catches cache-key **value** drift and nothing else. The earlier
claim that it would catch "a drifted nested manifest or cache-key set" was too
broad and is withdrawn.

#### D.2 The complete required sequence

1. Require the registry wrapper to have **exactly** the supported keys and types
   — `manifest_digest` (string) and `manifest` (object), no others.
2. Reconstruct the expected manifest with repository APIs only —
   `Stage2RepresentationKey.from_dict(...)` over the wrapper's nested
   `cache_keys`, then
   `build_stage2_campaign_manifest(repository_head="6693e728ccaebc987e4786bd5cd5e0f5c16143f7", cache_keys=...)`.
3. `validate_stage2_campaign_manifest(expected_manifest)`.
4. Require its repository-defined semantic digest to equal the independently
   anchored historical value exactly:

   ```
   571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6
   ```

5. **BEFORE `registry.open`**, require the **complete stored nested manifest** to
   match `expected_manifest.to_dict()` — not merely the constructor inputs.

**The comparison must cover:**

* every top-level field — `repository_head`, `protocol_version`, `arms`, `seeds`,
  `expected_runs`, `schema_version`, `head_campaign_schema_version`, `winner`,
  `ranks_arms`;
* the exact cache **slot set** — no missing slot, no extra slot;
* every serialised cache-key field, including derived metadata emitted by
  `Stage2RepresentationKey.to_dict()`;
* missing fields, unknown fields, and derived values inconsistent with the rest.

**Comparison semantics — type-aware JSON-tree equality.** Dictionary key order is
irrelevant; **list order matters** (`expected_runs`, `arms`, `seeds` are
sequences) and **scalar types matter**. `True` must not compare equal to `1`, nor
`1` to `1.0`, so Python's loose `==` on bare values is insufficient — compare
`type(...)` alongside value at each leaf. **No new scientific digest and no
replacement serializer is introduced**: the comparison is between the stored tree
and `to_dict()`'s own output.

**Do not assume `from_dict` validates fields its implementation does not read.**
Its `KeyError`/`ValueError` guard catches malformed and missing fields; it is not
an unknown-field check.

**Why this is not self-certification.** The expected digest is **frozen
historical authority**, not recomputed from the object under test, and the
independently raw-hash-gated evidence file of gate B **binds that same digest** —
two independent sources agreeing on one value the rebuilt object must reproduce.
Step 5 then closes what the digest cannot see, by comparing the stored bytes'
parsed tree against the rebuilt object's own serialisation field by field.

**Only after step 5 passes** may the notebook re-open the existing registry and
verify completion and the ten head artifacts.

> **These are requirements for notebook v2.** This complete revised gate has
> **not** been run against the author's Drive files. Nothing in this audit
> reports its outcome.

### Only then

`registry.open(expected_manifest)` → `registry.require_campaign_complete(...)` →
exactly ten expected `(arm, seed)` runs → every run store complete → ten artifact
JSONs → **ten exact historical selected epochs** → ten selected state files →
ten `selected_head_state_sha256` checks.

---

## 6. What Audit 057 keeps

Superseded: **only** the §6.1a raw-SHA binding to `registry.manifest_path` and
prose treating the two payloads as one artifact.

Preserved unchanged: the documentation-HEAD history;
`MEASUREMENT_ACCEPTED_EXECUTION_HEAD = 57b3d31e…`;
`CLEAN_HEAD_CAMPAIGN_EXECUTION_HEAD = 6693e728…`; measurement corruption seed
`19225`, pinned `True`; arms `UNMARK-A`/`UNMARK-B`; conditions
`FULL, P25, P50, P75, P100, STRIP_ALL`; **12** representation caches; **10**
frozen heads; **60** head-condition scores; `FULL` cache identity
`corruption_seed=None` and degraded `corruption_seed=19225`; the `FULL` API-only
placeholder `0` as **precedent only, never a protocol value**; both Stage-1
finalist checkpoint SHA gates; the ten exact selected-epoch gates; the ten
selected-head state digest gates before any data read; **no automatic** A/B
selection, winner or ranking; no p-values; GRR deferred; official TEST
structurally sealed; the hard stop before CELL 6; CELL 6 as the first authorised
`measurement-dev` read; and MyDrive-backup-only persistent storage.

The A/B item is scoped deliberately: what is preserved is the prohibition on the
**pipeline** selecting, ranking or dropping an arm. It is **not** a blanket bar on
the author making a choice after reviewing metrics — see §6a.

---

## 6a. Subsequent author instruction on A/B

**This is a later instruction from the author, recorded separately from the
historical no-selection facts of §3.** Verbatim:

> "Run both UNMARK-A and UNMARK-B on validation, give me the metrics, and I will
> decide which version to continue with afterwards."

**The historical facts are unchanged and remain true of the completed campaign
and the diagnostics:** `A_B_SELECTION=NO`, `winner=None`, `ranks_arms=False`.
Nothing about the training campaign is revised by an instruction issued after it.

**For the future notebook, this means:**

* evaluate **both arms fully** — all five frozen heads and all six conditions per
  arm, the same 12 caches and 60 score units;
* produce the **complete comparison** and **retain both arms'** results and
  artifacts;
* **do not automatically choose, drop or rewrite either arm** — the run produces
  metrics, not a decision;
* **the author may make an A/B choice later**, after reviewing the validation
  metrics;
* **no winner, threshold or head-seed choice has been decided yet**;
* **official TEST is not part of this run.**

```
AUTOMATIC_AB_SELECTION=NO
AUTHOR_AB_DECISION_AFTER_VALIDATION=REQUESTED
AUTHOR_AB_DECISION_MADE=NO
```

**No new decision ID, automatic selection rule or numerical margin is invented
here**, and **no protocol or spec file has been amended** — this task changes only
Audit 058. Whether the eventual choice needs a recorded decision is the author's
call, not this audit's.

**Notebook-v2 execution metadata must record this intended use**, so a later
reader of the evidence can see the run was performed to inform an author decision
rather than to execute an automatic one. §6's preserved contracts stand except
for this scope: **the earlier blanket prohibition is not carried forward as a bar
on the author's own later A/B choice.**

---

## 7. Runtime negative evidence

At the failed notebook-v1 preflight and both diagnostics:

```
MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
OFFICIAL_TEST_READ=NO
```

No scientific Stage-2 measurement occurred. **No real measurement representation
cache exists from this attempt. No measurement score exists from this attempt.
No head was retrained. No historical artifact was modified.** The repository
execution tree remained `57b3d31e1cf8145835117efee43a3f5172075ab8`, clean.

**What actually stopped execution was the erroneous raw-SHA gate**, which failed
at the first manifest check — before any pathway was built, before any tensor
existed, and before the planned `measurement-dev` read. That is the accurate
account: **the deliberate CELL-5 hard-stop mechanism was never reached and
therefore was not tested by this failure.** A bug halting a run early is not
evidence that the designed boundary works; it is only evidence that this
particular run did not get past a broken check.

---

## 8. Notebook v1 status, and v2 requirements

```
NOTEBOOK_DRAFT_V1=STOPPED_BEFORE_MEASUREMENT_DEV
NOTEBOOK_DRAFT_V1_MUST_NOT_BE_RESUMED
```

v1 carries the incorrect manifest binding in its preflight. It must not be
resumed or patched in place; the next draft is **v2**, which must include all of:

1. the corrected historical manifest chain of §5;
2. all ten historical head integrity gates before any data read;
3. exact detached execution HEAD `57b3d31e…`, asserted before and after;
4. proper measurement resume (§8.1);
5. W&B telemetry (§8.2);
6. Drive-only persistent storage (§8.3);
7. the hard stop before the first `measurement-dev` read.

### 8.1 Measurement resume

**Measurement is not head training**, so the resume unit is different. Resume
operates over **measurement units**, never over epochs or optimiser state.

For the **12 representation caches**:

| State | Action |
|---|---|
| exact compatible completed cache | validate, load, **skip extraction** |
| absent | extract, **atomically persist** |
| partial | **FAIL CLOSED** |
| incompatible | **FAIL CLOSED** |

For the **60 score units**:

| State | Action |
|---|---|
| exact completed immutable score unit | validate, **skip scoring** |
| absent | score once, **atomically persist** |
| partial or incompatible | **FAIL CLOSED** |

**Nothing is silently overwritten.** The aggregation and report are produced only
when **all** expected score identities exist — a partial report is not a report.

**Must not introduce:** optimiser state, head retraining, checkpoint selection,
or mid-head resume. Those belong to training, which is complete and closed.

### 8.2 W&B

**W&B is monitoring and observability only. It is NOT scientific authority** —
scientific authority remains the immutable Drive artifacts and evidence. This
matches D-S1B-018 and Audit 053 §10, where the scientific process cannot even
import `wandb`.

Persistent W&B local state must live under
`/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/`, with explicit MyDrive-backed
namespaces set for it. **W&B resume must never be used as the scientific resume
source** — §8.1's cache and score units are the only resume authority.

### 8.3 Persistent storage

All newly generated persistent UNMARK artifacts, caches and logs must live under:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/
```

Including measurement representation caches, per-score artifacts, reports, final
evidence, W&B local state, HF/model/tokenizer cache, the Vietnamese syllable
inventory's persistent bytes, and runtime scientific manifests.

**No persistent UNMARK state under `/content`, `/tmp`, `/root/.cache` or
`.shortcut-targets-by-id`.**

**One deliberate exception:** the disposable git clone may live under `/content`.
It is ephemeral execution state reconstructible from a pinned commit, not a
persistent scientific artifact — and pinning it under Drive would persist a copy
of something the checkout already guarantees.

---

## 9. Files changed

| File | Change |
|---|---|
| `docs/audits/058-stage2-measurement-manifest-binding-runtime-correction.md` | **new** — this file |

One path. Audits 053 and 057 are **not** edited. No implementation, test,
protocol, spec, decision, config, scientific constant, cache, head artifact or
Stage-1 artifact was modified.

**What was executed:** read-only source inspection of
`unmark/evaluation/stage2_campaign.py` (the wrapper payload shape, `MANIFEST_NAME`,
`to_dict()`, `digest`, `build_stage2_campaign_manifest`) plus `git status --short`
and `git diff --check`. **No pytest.** No dataset read, no `measurement-dev`
read, no Drive access from this environment, no model execution, no cache
creation, no scoring, no training.

---

## 10. Remaining blockers

| Item | Status |
|---|---|
| `DIFFERING_COMMON_KEYS=["cache_keys"]` unexplained | **open, non-blocking** — recorded without conjecture (§2.3); slot ordering is excluded as a cause. Gate D step 5 compares the stored tree against the rebuilt serialisation directly, so it does not depend on the answer. |
| Full stored-versus-rebuilt manifest comparison | **not yet run** — gate D step 5 is a v2 requirement, unexecuted against the author's Drive files |
| Ten selected-head state revalidation | **not yet run** — the diagnostics examined manifest artifacts, not `stage2-selected-head.pt` bytes (§3.1) |
| Notebook v2 not yet written | expected — this audit specifies it |
| Audit 057's three standing gaps | unchanged: no repository function reproduces the canonical artifact-payload serialisation; `Stage2HeadRunStore` has no public state loader; measurement-dev ordered-id and label digests exist only after the first authorised read |

**Nothing blocks writing notebook v2.** The two "not yet run" rows are v2's job,
not prerequisites for drafting it.

**Implementation prerequisite for v2.** The type-aware JSON-tree comparison of
gate D step 5 has **no repository implementation** — `require_compatible` compares
`Stage2RepresentationKey` objects, not stored manifest trees, and nothing in
`stage2_campaign.py` compares a stored nested manifest against `to_dict()`. v2
must supply that comparator itself, using `to_dict()` as the sole serialisation
authority and adding no new digest.

---

## 11. Final state

```
git status --short
?? docs/audits/058-stage2-measurement-manifest-binding-runtime-correction.md
```

`git diff --check` produced no output. `HEAD` is
`19b14c8d2d5dcb776762997159b057d650989fe3`, unchanged. Uncommitted and awaiting
author review; nothing staged, committed, pushed or history-mutated.

---

```
AUDIT057_MANIFEST_RAW_SHA_BINDING=SUPERSEDED_BY_AUDIT058
AUDIT057_OTHER_PREPARATION_CONTRACTS=PRESERVED

NOTEBOOK_DRAFT_V1=STOPPED_BEFORE_MEASUREMENT_DEV
NOTEBOOK_DRAFT_V1_MUST_NOT_BE_RESUMED

PRETRAIN_VERIFIED_MANIFEST_RAW_SHA256=4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962
REGISTRY_WRAPPER_RAW_SHA256_RUNTIME_OBSERVED=9020c81bc486e892e4f5d96f66750a6770c16cbd67c69c6c72269fc85daf7899
CLEAN_CAMPAIGN_MANIFEST_SEMANTIC_DIGEST=571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6
FINAL_CLEAN_CAMPAIGN_EVIDENCE_RAW_SHA256=69dc1447f1c340c18285f363d1ebfaa7d5d93f954e487e2fd99c4f556e9faf59

PRETRAIN_VS_REGISTRY_NESTED_OBJECT_EQUALITY=False
PRETRAIN_VS_REGISTRY_NESTED_CACHE_KEYS_EQUAL=False

HISTORICAL_STAGE2_CAMPAIGN=PASS
STAGE2_HEAD_RUNS_COMPLETED=10_OF_10
RETRAIN_REQUIRED=NO

MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
OFFICIAL_TEST_READ=NO

AUTOMATIC_AB_SELECTION=NO
AUTHOR_AB_DECISION_AFTER_VALIDATION=REQUESTED
AUTHOR_AB_DECISION_MADE=NO

MEASUREMENT_NOTEBOOK_V2_REQUIRED=YES
MEASUREMENT_WANDB_REQUIRED=YES
MEASUREMENT_RESUME_REQUIRED=YES
PERSISTENT_STORAGE=MYDRIVE_BACKUP_ONLY

READY_FOR_REAL_MEASUREMENT_EXECUTION=NO
```

`READY_FOR_REAL_MEASUREMENT_EXECUTION=NO` is unchanged and unmoved. A corrected
preflight chain is not an execution authorisation: **it is `NO` because notebook
v2 is not yet implemented and accepted** — not because the author's A/B
instruction is unconfirmed. That instruction is recorded in §6a and needs no
further confirmation.
