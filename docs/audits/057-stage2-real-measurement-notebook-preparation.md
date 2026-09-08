# Audit 057 - Stage-2 Real Measurement Notebook: Preparation

**Scope:** design the real Stage-2 measurement execution notebook against the
repository's actual APIs, and record every unresolved runtime input — **without
reading `measurement-dev`, without executing the scientific measurement
notebook, the model/data pipeline or any scientific run, and without writing to
Drive.**
**Date:** 2026-09-08
**Type:** **PREPARE-ONLY / NO-DATA-READ execution plan. NOT a scientific
decision.** No decision record is appended. No implementation, test, protocol,
spec, config, scientific constant, cache schema, head artifact or Stage-1
artifact changes.

---

## 1. Starting state

```
branch : main
HEAD   : 76f3ba8e7e1a054cf4ec56876547e6f43ef20867
status : clean (git status --short produced no output)
```

Both preconditions were verified before any edit.

`76f3ba8e…` adds exactly one path — Audit 056 — so its implementation tree is
**byte-identical** to `57b3d31e…`: the same diff restricted to `unmark/`,
`tests/`, `docs/spec/`, `configs/` and `scripts/` produces no output. The
notebook that checks out `57b3d31e…` therefore runs the same code this
preparation inspected.

Every API, signature, constant and on-disk name below was read from source at
this HEAD. Nothing is guessed.

---

## 2. Three distinct identities

| Role | SHA | Meaning |
|---|---|---|
| **Documentation HEAD at preparation start** | `76f3ba8e7e1a054cf4ec56876547e6f43ef20867` | documentation provenance only; never a scientific execution identity |
| **Accepted measurement execution HEAD** | `57b3d31e1cf8145835117efee43a3f5172075ab8` | the notebook checks this out detached and binds it as `repository_head` in **new** measurement cache keys |
| **Closed clean-cache / 10-head execution HEAD** | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` | immutable provenance of the four clean `FULL` caches, the ten trained heads, their selected epochs and artifacts |

**The old artifacts are never rewritten to `57b3d31e…` or `76f3ba8e…`.** The ten
head artifacts bind `repository_head = 6693e728…` inside their own payload, and
`validate_stage2_head_artifact` cross-checks that their two cache keys carry the
same commit. Rewriting it would break that internal consistency, not repair it.

**Consequence, and it is deliberate:** the notebook executes at `57b3d31e…` while
loading artifacts stamped `6693e728…`. §6 explains why that is correct and how to
validate them without touching their identity.

---

## 3. Frozen measurement protocol, confirmed from repository authority

| Quantity | Value | Source |
|---|---|---|
| Measurement corruption seed | `19225` | `STAGE2_MEASUREMENT_CORRUPTION_SEED` |
| Pinned | `True` | `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED` |
| Arms | `('UNMARK-A', 'UNMARK-B')` | `Stage2UnmarkArm` |
| Conditions, frozen order | `('FULL', 'P25', 'P50', 'P75', 'P100', 'STRIP_ALL')` | `STAGE2_UNMARK_CONDITIONS` |
| Degraded subset | `('P25', 'P50', 'P75', 'P100', 'STRIP_ALL')` | `STAGE2_DEGRADED_CONDITIONS` |
| Measurement role | `official-validation` (`Preg1Role.OFFICIAL_VALIDATION`) | `STAGE2_MEASUREMENT_ROLE` |
| Head seeds | `(53148, 59945, 42941, 720, 9428)` | `STAGE2_CAMPAIGN_SEEDS` |
| Campaign runs | `10` | `stage2_campaign_plan()` returns 10 |
| Official TEST | structurally unreachable | `Preg1Role` has no `OFFICIAL_TEST` member |
| A/B selection / winner / ranking | none | `STAGE2_AB_SELECTION_IMPLEMENTED = False`, `STAGE2_AB_WINNER_RULE = None`, `STAGE2_AB_TIE_BREAK = None` |

### 3.1 Twelve caches, sixty scores — and why they are different numbers

**Representation extraction is head-independent.** A cache is a function of
(arm, role, condition, dataset rows) — it does not know which head will consume
it. `stage2_measurement_extraction_plan(corruption_seed=19225)` returns exactly
**12** requests: `2 arms x 6 conditions`.

Scoring is then head-dependent. Each arm's five frozen heads are scored against
that arm's six caches:

```
2 arms x 5 heads x 6 conditions = 60 condition-score evaluations
```

**These 60 are scores, not extractions.** Describing them as 60 representation
extractions would imply 60 forward passes over the encoder, which would be five
times the work and would invent five redundant copies of every cache. The
correct statement is **12 extractions, 60 scores**, and the notebook must be
written that way.

---

## 4. Cache identity contract

`Stage2RepresentationKey` is a frozen dataclass binding **22 fields**, all
required, all compared exactly. Read from source:

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

For every measurement cache the notebook must bind:

| Field | Value |
|---|---|
| `repository_head` | `57b3d31e1cf8145835117efee43a3f5172075ab8` |
| `arm` | `UNMARK-A` / `UNMARK-B` |
| `finalist_checkpoint_sha256` | that arm's, from `finalist_for_arm(arm).checkpoint_sha256` |
| `backbone_checkpoint` / `backbone_revision` | `ENCODER_CHECKPOINT` / `ENCODER_REVISION` (pinned PhoBERT) |
| `protocol_version` | `stage2-dual-finalist-protocol-v1` |
| `dataset` / `dataset_version` / `task` | `PRIMARY_DATASET` / version / task |
| `role` | `official-validation` |
| `condition` | one of the six |
| `corruption_seed` | `None` for `FULL`; `19225` for each degraded condition |
| `pooling` | `FIRST_TOKEN` |
| `max_length` / `truncation` / `padding` | `MAX_LENGTH` / `TRUNCATION` / `PADDING` |
| `ordered_id_digest` | over measurement-dev ids **in order** — §7 |
| `label_digest` | over measurement-dev integer labels **in order** — §7 |
| `dtype` / `hidden_size` | `torch.float32` / `768` |
| `count` | measurement-dev row count |
| `schema_version` | `stage2-head-campaign-v1` |

The notebook should build these through `stage2_representation_key_for(...)` —
which takes `request`, `repository_head`, `ordered_ids` and `labels` — rather
than by hand, so the frozen fields come from the module.

**Same degraded realisation for both arms** follows structurally: the plan binds
one `corruption_seed` for every degraded request, and a measurement degraded key
that is not `19225` cannot be constructed at all.

### 4.1 Batch provenance is checked before forward and before write

`require_batch_provenance(batches, key)` runs inside
`extract_and_cache_stage2_representations` **before `import torch` and before any
cache write** — so a mislabelled extraction is refused on any machine and leaves
no artifact. `collate_stage2_unmark_batch` supplies the provenance as
`batch["corruption_seeds"]`, taken verbatim from each input's
`corruption_metadata["corruption_seed"]`; it is non-tensor and the forward mover
uses an allowlist (`STAGE2_FORWARD_TENSOR_KEYS`), so it never reaches the encoder.

### 4.2 Exact existing fail-closed behaviours

| Situation | Where | Behaviour |
|---|---|---|
| batch condition != key condition | `require_batch_provenance` | "was prepared for condition `X` but the cache key declares `Y`" |
| mixed conditions in one batch | same | "mixes conditions […]; one cache holds exactly one condition" |
| missing `corruption_seeds` on a degraded batch | same | "carries no `corruption_seeds` provenance" |
| a `None` seed among degraded rows | same | "no recorded corruption seed; missing provenance fails closed" |
| seed list shorter than rows | same | "provenance must cover every row" |
| mixed seeds | same | "mixes corruption seeds […]" |
| wrong seed vs key | same | "…the cache key declares `Y`. Saving it would store a tensor whose key falsely claims a realisation it was not produced under." |
| wrong arm / pathway | `extract_and_cache_…` | "pathway carries arm `X` but the cache key is `Y`; one arm may never write the other's cache" |
| pathway checkpoint mismatch | same | "pathway checkpoint sha256 does not match the cache key" |
| wrong role | `Stage2RepresentationKey.__post_init__` via `Preg1Role(role)`; `require_role` at use sites | unknown role raises; measurement tensors cannot reach selection |
| degraded measurement key not `19225` | `__post_init__` | "must bind the frozen seed 19225 (D-S2-002)" |
| `FULL` key carrying any seed | `__post_init__` | "FULL is the clean condition and must carry no corruption seed" |
| degraded key carrying no seed | `__post_init__` | "must bind the corruption seed that produced it" |
| row-count contradiction | `extract_and_cache_…` | "extracted N rows but the key declares M" |
| dtype / hidden_size contradiction | `__post_init__` | dtype must be `torch.float32`; `hidden_size` must be 768 |
| tensor/key contradiction on load | `Stage2RepresentationCache.load` | key compared field-by-field |
| existing incompatible cache | `require_compatible` | "incompatible on N field(s) … Refusing to reuse it" — a mismatch is an error, never a recomputation |
| re-saving a different key into the same directory | `Stage2RepresentationCache.save` | refused; same key is idempotent |

---

## 5. The `FULL` API-only placeholder — conclusion

**The repository does not freeze any placeholder value.** Verified: no
placeholder constant exists in `unmark/evaluation/`; the only mention is the
explanatory docstring in `require_batch_provenance`.

| Question | Answer, from source |
|---|---|
| Must the notebook supply a placeholder for `FULL`? | **Yes.** `prepare_stage2_unmark_input(..., corruption_seed: int)` is keyword-only with **no default**, so a call mechanically requires an integer even for the clean condition. |
| What does existing precedent use? | **`0`** — Audit 052 §8 records that the clean-cache execution passed `0` as an API-only placeholder. This is *precedent*, not a frozen value. |
| What proves `FULL` is a no-op? | Structurally: `FULL = CorruptionCondition("FULL", CorruptionScope.NONE, 0.0, …)` — scope `NONE`, probability `0.0`, so no seed can change it. Empirically: Audit 052 §8 verified `canonical_text == corrupted_text` and `condition_probability == 0.0` for all 11 424 clean rows. |
| What proves the placeholder never enters cache identity? | Two independent guards. `Stage2RepresentationKey.__post_init__` **refuses** a `FULL` key carrying any seed, so a clean key binding a placeholder cannot be constructed. And `require_batch_provenance` checks only the *condition* for `FULL`, never the seed, so a clean batch carrying `0` is accepted while the key stays `None`. |

**`STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED = False` → `True` refers to the
degraded seed only.** The `FULL` placeholder is an implementation artefact of a
required parameter and **must not be recorded as a scientific protocol value.**
The notebook should pass `0` for consistency with precedent and assert
`key.corruption_seed is None` immediately afterwards.

---

## 6. The ten real head artifacts — loading and verification contract

### 6.1 On-disk layout, from source

`Stage2CampaignRegistry(directory)`:

| Member | Value |
|---|---|
| `manifest_path` | `<directory>/` + the registry's manifest file |
| `run_directory(arm, seed)` | `<directory>/<arm>/seed-<seed>` |
| `store_for(arm, seed)` | `Stage2HeadRunStore(run_directory(arm, seed))` |

`Stage2HeadRunStore(directory)`:

| Class constant | Value |
|---|---|
| `ARTIFACT_NAME` | `stage2-head-artifact.json` |
| `STATE_NAME` | `stage2-selected-head.pt` |
| `IN_PROGRESS_NAME` | `IN_PROGRESS` |

**No Drive filenames are invented here** — these are the committed names.

### 6.1a The campaign registry root is known, not operator-invented

Audit 053 §14 records the clean-campaign durable evidence file
`stage2-clean-10-head-campaign-final.json`, under
`.../stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1/`.
The registry root is therefore that directory:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1
```

**This is audit authority, not an operator guess**, and it is not inferred from
W&B — the W&B group of Audit 053 §10 is telemetry and is explicitly not
scientific authority.

#### The manifest must be anchored to independent authority

`registry.open(expected_manifest)` compares the stored manifest against an
expectation. **That expectation may not be the registry's own manifest.** Reading
the manifest from disk and then handing the same unanchored object back as the
thing it must match proves only that a file equals itself — the check would pass
against any manifest, including a drifted one.

Audit 053 records **two** independent identities of the accepted manifest, and
they are **different checks of different things**:

| Identity | Value | Gates |
|---|---|---|
| `MANIFEST_FILE_SHA256` | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` | the **on-disk bytes** of the manifest file |
| `MANIFEST_SEMANTIC_DIGEST` | `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` | the manifest's **scientific identity** — `Stage2CampaignManifest.digest`, the value the registry itself compares on re-entry |

**Both are historical authority, inherited from Audit 053 — neither is chosen
here, and neither may be derived from the live manifest.** Recomputing an
"expected" digest from the file under test is the same circularity the anchoring
exists to prevent. They must not be conflated: a byte-identical file with a
drifted semantic identity, or a re-serialised file with the right identity, are
different failures and only two separate checks distinguish them.

At that root the notebook must fail closed unless **all** of, in order:

1. `Stage2CampaignRegistry` is instantiated at the known campaign root;
2. `registry.manifest_path` exists — **the filename is taken from the property,
   never invented**;
3. the SHA-256 of the **raw manifest file bytes** equals
   `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` exactly;
4. the `Stage2CampaignManifest` is loaded or constructed through the
   repository-supported representation;
5. it validates under repository authority
   (`validate_stage2_campaign_manifest`);
6. its **semantic** digest equals
   `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` exactly;
7. **only then** `registry.open(expected_manifest)` — now anchored, because the
   expectation passed steps 3 and 6 before being trusted;
8. `registry.require_campaign_complete(manifest)` succeeds;
9. exactly the ten frozen `(arm, seed)` runs exist;
10. every run directory's `Stage2HeadRunStore.is_complete()` is true.

The historical campaign's `repository_head = 6693e728…` is **not** altered by any
of this; these are read-only gates over evidence that already exists.

### 6.1b Stage-1 finalist checkpoint inputs

The two frozen pathways cannot be constructed without `checkpoint_path`. Both are
**existing source artifacts, not newly generated outputs**:

| Arm | Checkpoint path | Expected SHA-256 |
|---|---|---|
| `UNMARK-A` | `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage1-training/7773c77b1df9/final-main/run-seed36930/_checkpoint/training-checkpoint-best.pt` | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| `UNMARK-B` | `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage1-training/bca24ade2082/lr-pilot/run-lr0.0001/_checkpoint/training-checkpoint-best.pt` | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |

Both digests are the ones D-S2-001 binds and `finalist_for_arm(arm)` returns.

**A preflight runs in CELL 5, before the hard stop**: both files exist; the raw
checkpoint SHA-256 equals the bound finalist digest; `finalist_for_arm` agrees;
**neither checkpoint is mutated**. The actual `load_frozen_unmark_pathway` stays
in CELL 7 — the point of the early preflight is to discover a missing or wrong
checkpoint **before** the first measurement-dev read, not to move the load.

### 6.2 Methods the notebook should call

| Call | Purpose |
|---|---|
| `registry.exists()` | manifest present |
| `registry.open(manifest)` | re-open **only** under a byte-identical manifest digest; a drifted one is refused, never adopted |
| `registry.completed_runs(manifest)` / `pending_runs(manifest)` | run accounting |
| `registry.campaign_status(manifest)` | completion report — counts and pending only, no score, delta or ranking |
| `registry.require_campaign_complete(manifest)` | refuses a partial campaign: both arms and all five seeds, or nothing |
| `registry.store_for(arm, seed).is_complete()` | artifact present |
| `registry.store_for(arm, seed).read_artifact(expected_arm=…, expected_seed=…)` | reads **and validates** in one call |

`Stage2CampaignRun` is the run identity `(arm, seed)`; `stage2_campaign_plan()`
returns the ten, and `require_paired_campaign_plan` refuses a third arm, an extra
or unknown seed, a duplicate, or a seed unpaired across arms.

### 6.3 What `validate_stage2_head_artifact` enforces

A closed **24**-field schema (`STAGE2_HEAD_ARTIFACT_FIELDS`) — **partial binding
and unknown fields are both refused**. The count was resolved mechanically, not
read from prose: `len(STAGE2_HEAD_ARTIFACT_FIELDS) == 24` at this tree, and
`build_stage2_head_artifact` emits exactly that key set (set-equal, none missing,
none extra). `validate_stage2_head_artifact` closes it in both directions — a
missing field is refused as partial binding, an unknown field as schema drift.

**Reconciling the 23-vs-24 discrepancy.** Audit 051 §8 and Audit 053 §6 both call
this a "closed 23-field schema". That wording is **stale documentation, not
schema drift**: the tuple's source text at `6693e728…`, where it was created, is
**byte-identical** to its text at this HEAD, and `git log -L` over the tuple
reports exactly one commit — its creation. It was 24 fields when both audits were
written, so no field was added afterwards and none is missing from them; the
prose simply miscounted. **The 24 fields, in source order, are:**

```
schema_version              arm                          finalist_checkpoint_sha256
repository_head             protocol_version             seed
initial_head_fingerprint    head_architecture            optimizer
learning_rate               epochs                       early_stopping
selected_epoch              selected_macro_f1            selected_accuracy
selected_head_state_sha256  train_cache_key              protocol_dev_cache_key
history_digest              precision                    selection_role
selection_condition         measurement_used_for_selection
ab_selection_performed
```

**Audits 051 and 053 are not edited** — audit history is append-only, and the
reconciliation belongs here. The notebook must treat **24** as authoritative,
because that is what the validator enforces.

**What it checks**: schema version; arm identity (an A artifact may never load as
B); seed; that the checkpoint sha is that arm's; LR
`0.01`; `epochs == 30` and `early_stopping is False`; `selection_role` is
`protocol-dev`; `selection_condition` is `FULL`; both
`measurement_used_for_selection` and `ab_selection_performed` are `false`; and
that each of `train_cache_key` and `protocol_dev_cache_key` has the right role,
the clean condition, the right arm, and **the same `repository_head` as the
artifact**.

That last check is why §2 matters: the artifacts are internally consistent at
`6693e728…`, and validation compares them against *themselves*, not against the
running HEAD. **The notebook validates them without rewriting their identity —
`read_artifact` takes only `expected_arm` and `expected_seed`.**

### 6.4 Digests available, and one that is not

| Digest | In the artifact? | Recomputable by repository code? |
|---|---|---|
| `selected_head_state_sha256` | **yes** | via `_state_digest(state)` — order-independent, exact float `repr` |
| `history_digest` | **yes** | via the public `history_digest(scores)` |
| **canonical artifact-payload SHA-256** | **no** | **NO — see below** |

**Finding.** The ten values below are *canonical artifact-payload SHA-256*
digests, computed by the Audit-053 closeout notebook as
`canonical_json_sha256(artifact)`. **That function does not exist in this
repository** — `canonical_json_sha256` appears only in Audit 053's prose. The
repository stores **no** separate raw file digest of the artifact either. So:

* these ten values are correctly called **canonical artifact-payload SHA-256**,
  and must **not** be relabelled raw file SHA-256;
* the notebook **cannot recompute them with repository code**. To re-verify them
  it must reproduce the closeout notebook's canonical JSON serialisation, which
  is not defined in repository authority.

**`OPERATOR_RUNTIME_INPUT_TO_BE_VERIFIED_AT_EXECUTION`:** the exact canonical
JSON serialisation used for those digests. Until it is stated, artifact
verification should rely on `validate_stage2_head_artifact` plus
`selected_head_state_sha256`, and the payload digests should be recorded rather
than treated as a gate.

| Arm | Seed | Canonical artifact-payload SHA-256 | Selected epoch |
|---|---|---|---|
| `UNMARK-A` | `53148` | `75ed0db7158e8f4912689d19923fb43f2bb6887d8592f942cb3adca3ddf0208d` | `30` |
| `UNMARK-B` | `53148` | `c0379fff439efcf611599b97af18117f907794e48eb3cc945b604b2e55c0ab19` | `11` |
| `UNMARK-A` | `59945` | `a4aa17f4453138fec90c8e1c6681753bb1b13ccf8e252f1654fbda6ab9609932` | `27` |
| `UNMARK-B` | `59945` | `4befb7ccf887bec51f5b5086a70a269da11ad10df154ba8c2146529ae9c55827` | `26` |
| `UNMARK-A` | `42941` | `e8e0108938e0385e65879c33c64f35a342ca69f78025d44224f5ee468cf3ec43` | `9` |
| `UNMARK-B` | `42941` | `afe12275d4fa9bb6bdf9f4d337807ed9895d8820fc947fc3de742ad0d7f4113c` | `19` |
| `UNMARK-A` | `720` | `7c8c2996be41543bfa223d0d697dd73dae82aafc4c5a6cba0f4c2ecdd9f40a48` | `26` |
| `UNMARK-B` | `720` | `4defc22b40c59858e778891f057749fc79e086cdc93f7bc94b8a7fa333e88021` | `22` |
| `UNMARK-A` | `9428` | `20caea0a7bbb4e94c004e9f90717aa806a01ea213e1f56646d59c53bf6020091` | `25` |
| `UNMARK-B` | `9428` | `b5f5417343846e678d87c0be5c024c540864b1a46f0fd6c73320fed251a2e4ee` | `25` |

**On the selected epochs:**

> Có selection, nhưng chỉ là chọn epoch tốt nhất của từng head; không phải chọn
> A hay B.

They are within-head checkpoint selections under the frozen total order — highest
macro-F1, then highest accuracy, then earliest epoch — computed per arm and per
seed independently. **No A/B ranking may ever be derived from them**, and
comparing the two columns is not a permitted operation.

### 6.5 A second gap: loading the head state

`Stage2HeadRunStore` has `commit(artifact, selected_head_state)` but **no
corresponding load method**, and `_state_digest` is private and absent from
`__all__`.

**The notebook must supply the load itself**, and it must do so **before the hard
stop**, not at CELL 9. For each of the ten completed runs, in CELL 4:

1. `store.read_artifact(expected_arm=…, expected_seed=…)`;
2. require `artifact["selected_epoch"]` to equal **exactly** the historical value
   for that `(arm, seed)` in the §6.4 table — see the note below;
3. locate `store.directory / Stage2HeadRunStore.STATE_NAME`;
4. require that state file to exist;
5. `torch.load(..., map_location="cpu")`;
6. require it to be the expected state mapping;
7. compute `_state_digest(state)`;
8. compare **exactly** with `artifact["selected_head_state_sha256"]`;
9. do not alter tensors, do not train;
10. optionally retain only verified, immutable state snapshots or references for
    later use.

**On the selected-epoch gate (step 2).** This is an **integrity check of ten
already-selected real heads** — it confirms the artifacts on disk are the ones
Audit 053 accepted. It is **not** checkpoint selection, not A/B selection, not
A/B ranking, and not a comparison between arms; the epochs are compared
individually against their own historical values, never against each other.

> Có selection, nhưng chỉ là chọn epoch tốt nhất của từng head; không phải chọn
> A hay B.

The notebook may **record** the epochs only *after* exact equality has been
checked — recording first and comparing later would let a drifted value into the
evidence before anything had refused it.

**No gate is added on the canonical artifact-payload SHA-256 values.** The
repository still does not define the canonical JSON serialisation needed to
recompute them (§6.4), so they remain recorded rather than enforced. Gating on a
digest the repository cannot reproduce would be a check that either always passes
or fails for the wrong reason.

**Using the private `_state_digest` is deliberate and recorded.** There is no
public equivalent, and re-implementing the digest would create a second
definition of the same quantity — the failure mode where two implementations
drift and the mismatch is discovered by a number silently disagreeing rather than
by an error. One definition, used from a private name, is the safer trade.

**Why before the hard stop.** A missing, truncated or digest-mismatched
`stage2-selected-head.pt` is discoverable with zero dataset access. Deferring it
to CELL 9 would mean discovering it *after* measurement-dev had already been
read — spending the project's first authorised read of the official validation
split to find a storage problem. CELL 9 therefore instantiates heads from states
**already verified in CELL 4**, re-checking the digest if it re-reads the files,
and is no longer the first point at which corruption is detected.

This preserves all four invariants: no head training, no checkpoint selection,
selected epochs unchanged, and the ten Audit-053 artifacts immutable.

---

## 7. Measurement-dev input boundary

**Nothing was opened, hashed, counted, parsed or materialised for this audit.**
The properties below come from repository authority only.

### 7.1 Already frozen in the repository

| Property | Value | Source |
|---|---|---|
| Official validation row count | **`1583`** | `PUBLISHED_SPLIT_SIZES["validation"]`, asserted in two tests |
| Published label counts | `negative 705`, `neutral 73`, `positive 805` — sums to `1583` | `PUBLISHED_LABEL_COUNTS["validation"]` |
| Derived official-validation CSV SHA-256 | `9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0` | `DERIVED_VALIDATION_CSV_SHA256` |
| Raw `dev/sents.txt` SHA-256 | `fb7c3cc3173e1383edc03779883d91bb4d6110c8dd881612572a256878aa23b4` | `docs/experiments/preg1-uit-vsfc-real-profile-result.md` |
| Raw `dev/sentiments.txt` SHA-256 | `a9584a22c926a54c6042236380c9a65ab8c41467477f7a5d794fb2505c96a9c3` | same |
| Role rule | measurement-only; read **after** the LR is frozen, never for selection (D-PREG1-004b) | `DERIVED_VALIDATION_CSV_SHA256` docstring |
| Label mapping | `negative 0`, `neutral 1`, `positive 2` | protocol artifact |

`1583` is stated here **because repository authority confirms it**, not because
the brief asserted it.

### 7.2 Unresolved — `OPERATOR_RUNTIME_INPUT_TO_BE_VERIFIED_AT_EXECUTION`

| Input | Status |
|---|---|
| Operator runtime path to the derived official-validation CSV | **not in repository authority** — supplied at execution, gated on `DERIVED_VALIDATION_CSV_SHA256` rather than on its filename |
| measurement-dev **ordered-id digest** | **not pinned anywhere** — computed at first read via `ordered_id_digest(ids)` |
| measurement-dev **label digest** | **not pinned anywhere** — computed at first read via `label_digest(labels)` |
| Actual label representation (`str` vs `int`) as returned by the loader | **unverified for validation** — Audit 052 §7 found the loader returned `str` for the train split; the same may hold here, and normalisation must go through the frozen mapping |
| Canonical JSON serialisation behind the ten payload digests | **not in repository authority** — §6.4 |

**No value above is guessed.** The first authorised read of measurement-dev is
CELL 6, and it does not happen in this task.

---

## 8. Drive namespace contract

**Every newly generated persistent artifact _and cache_ must live under:**

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/
```

This is deliberately wider than "scientific output". It covers:

* measurement representation caches;
* score, report and evidence files;
* the Hugging Face / `transformers` model cache;
* tokenizer and model download caches;
* the pinned Vietnamese syllable inventory's persistent bytes;
* any other persistent runtime cache.

**No persistent cache may be rooted under `/content`, `/tmp`, `/root/.cache` or
`.shortcut-targets-by-id`.** A Colab runtime is disposable; anything cached
outside MyDrive is lost with it, and a re-download is a second chance to fetch
different bytes than the pinned ones.

**Existing source inputs are exempt.** Files that already exist — the two Stage-1
finalist checkpoints (§6.1b), the campaign registry (§6.1a), the
measurement-dev CSV — may be *read* from their existing approved locations. They
are inputs, not newly generated outputs, and this contract does not relocate
them.

Nothing was written to Drive by this task.

### 8.1 Lexical namespace validation

**Validate lexically, never by resolution.** `Path.resolve()` follows symlinks —
and Drive shortcuts resolve to `.shortcut-targets-by-id/...`, so resolving would
turn a correct declared path into one that fails the check, or mask an escape.
Use instead:

```
root = os.path.abspath("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")
out  = os.path.abspath(declared_output_root)
if os.path.commonpath([root, out]) != root:
    raise SystemExit("output namespace escapes the MyDrive backup root")
```

Fail closed on any escape — and apply the same check to **cache** roots, not only
to the output root.

### 8.2 Model and tokenizer cache namespace

CELL 2 and CELL 7 must pass an explicit MyDrive-backed `cache_dir`, or set the
equivalent HF cache environment, so the pinned PhoBERT and its tokenizer do not
land in the default local cache. A suitable namespace is:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/runtime-input-cache/
```

**The exact basename is an operator choice, not a scientific protocol choice.**
Any existing MyDrive-backup cache namespace is equally acceptable; what is
binding is that it lies under the backup root and is validated lexically.

### 8.3 Inventory persistence

CELL 3 must preserve the accepted pattern:

* persistent inventory bytes under the MyDrive backup — for example
  `runtime-input-cache/vietnamese-syllables/all-vietnamese-syllables.txt`;
* verify sha256
  `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2`;
* verify `116290` bytes;
* if the repository verifier requires `.resources-cache/vietnamese-syllables/`,
  satisfy it with a **repo-local ephemeral symlink or view** pointing at the
  MyDrive-backed bytes;
* **do not persist a second inventory copy under `/content`.**

One set of persistent bytes, verified once, viewed where the verifier expects
it — two copies is two things that can diverge.

### 8.4 Proposed output hierarchy — operator namespace, not protocol

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement/
    57b3d31e1cf8/                      # accepted execution SHA, 12-char prefix
        <YYYYMMDDTHHMMSSZ>/            # runtime timestamp
            caches/<arm>/<condition>/  # 12 representation caches
            scores/                    # 60 condition scores
            report/                    # descriptive aggregate
            evidence/                  # final JSON + logs
```

**This layout is a notebook/operator output namespace, not a scientific protocol
choice.** It carries no frozen meaning and may be changed by the operator without
a decision record; only the MyDrive-root constraint is binding.

---

## 9. Notebook design — CELL 0 to 12, with a hard stop

**Everything before the hard stop must be runnable without touching
`measurement-dev`.**

### Phase A — no data read

| Cell | Purpose | Must not |
|---|---|---|
| **CELL 0** | mount Drive; declare constants and **MyDrive namespaces** only — execution SHA, seed `19225`, arms, conditions, seeds, output root, cache root | read measurement-dev |
| **CELL 1** | clone/fetch; **checkout detached exactly `57b3d31e1cf8145835117efee43a3f5172075ab8`**; assert `git rev-parse HEAD` equals it; assert clean tree | proceed on any mismatch |
| **CELL 2** | install requirements; import real torch; verify CUDA/GPU; **point every model/tokenizer cache at the MyDrive-backed namespace** (§8.2) | read measurement-dev; leave HF caches at their default local root |
| **CELL 3** | verify the pinned Vietnamese syllable inventory — sha256 `78eeb840…`, `116290` bytes — from **persistent MyDrive-backed bytes** (§8.3) | advance the pin; persist a second copy under `/content` |
| **CELL 4** | open the registry at the **known root** (§6.1a); `require_campaign_complete`; exactly ten `(arm, seed)` runs, each store complete; anchor the manifest by **file SHA-256 `4a940d87…` and semantic digest `571100c3…`** before `registry.open` (§6.1a); for all ten: `read_artifact(expected_arm=…, expected_seed=…)`, **require `selected_epoch` to equal its exact historical value**, load `stage2-selected-head.pt` with `map_location="cpu"`, compute `_state_digest`, compare exactly with `selected_head_state_sha256`; record the ten payload digests and selected epochs | train anything; alter a state; rewrite `repository_head`; read measurement-dev |
| **CELL 5** | **Stage-1 finalist checkpoint preflight** — both files exist, raw SHA-256 equals the bound digest, `finalist_for_arm` agrees, neither mutated (§6.1b); `stage2_measurement_extraction_plan(corruption_seed=19225)` → assert exactly **12**; assert `FULL` → `None` and each degraded → `19225`; assert both arms; validate every output **and cache** namespace lexically (§8); assert no cache directory collision and that no existing cache would be overwritten | read measurement-dev |

> ### ⛔ HARD STOP — REVIEW BOUNDARY
>
> Everything above runs with **zero** measurement-dev access. The notebook must
> stop here for review. Nothing below may execute until that review passes.

### Phase B — specified, NOT executed

| Cell | Purpose | Must not |
|---|---|---|
| **CELL 6** | **FIRST AUTHORISED measurement-dev READ.** Operator path; gate on `DERIVED_VALIDATION_CSV_SHA256 = 9c475c89…`; read **exactly once** into one canonical in-memory structure; verify schema, `count == 1583`, label counts `705/73/805`; normalise labels through the frozen mapping; compute `ordered_id_digest` and `label_digest` | accept any TEST path or role |
| **CELL 7** | `load_frozen_unmark_pathway(arm, checkpoint_path, …)` for both arms; verify finalist checkpoint sha256; `require_frozen_unmark_pathway` — frozen, eval, FP32, coherent device | train; unfreeze |
| **CELL 8** | prepare + collate + `extract_and_cache_stage2_representations` for the **12** requests; `FULL` key seed `None` (placeholder `0` passed to the API only, §5); degraded keys `19225`; same realisation both arms; caches immutable and fail-closed | write 60 caches; reuse an incompatible cache |
| **CELL 9** | instantiate the ten frozen heads from the states **already verified in CELL 4** — `build_head(HIDDEN_SIZE, seed)`, `load_state_dict`; re-check the digest if the files are re-read | train; alter a selected state; treat this as the first integrity check |
| **CELL 10** | `measure_stage2_head(head, measurement, labels, seed=…)` for all **60** arm x head x condition combinations | perform any checkpoint selection |
| **CELL 11** | `aggregate_stage2_campaign(scores)` — descriptive only | add p-values, significance tests, winner or ranking |
| **CELL 12** | final evidence; re-assert HEAD still `57b3d31e…`; tree clean; TEST unread; heads unchanged; caches immutable | overwrite prior evidence |

---

## 10. Result and aggregation contract, from source

`measure_stage2_head(head, measurement, measurement_labels, *, seed)` returns a
`Stage2ConditionScore` with exactly:

| Field | Type |
|---|---|
| `arm` | `str` |
| `seed` | `int` |
| `condition` | `str` |
| `macro_f1` | `float` |
| `accuracy` | `float` |
| `per_class_f1` | `tuple[float, ...]` — 3 classes |

**No field is invented.** The role is read from the tensor's key, so a
protocol-dev tensor cannot be scored as measurement.

`aggregate_stage2_campaign(scores)` returns, per arm and per condition:
`macro_f1_mean`, `macro_f1_std`, `accuracy_mean`, `accuracy_std`,
`per_class_f1_mean`; plus `robustness_summaries` with
`strip_all_macro_f1_mean` and `degraded_equal_weight_macro_f1_mean`; plus
`paired_deltas_b_minus_a` (per-seed `macro_f1_delta_b_minus_a` and
`accuracy_delta_b_minus_a`, descriptive); plus `ab_selection_performed: False`
and `winner: None`.

**Frozen reporting policy, enforced in code:** both arms are **required** — a
single-arm report raises, because "Audit 049 forbids dropping an arm"; each
arm x condition must cover the frozen five seeds or it raises, because "a seed may
never be dropped because a result looks bad". Headline robustness stays
`STRIP_ALL` macro-F1 and the equal-weight mean over the five degraded conditions.

**GRR remains `DEFERRED`** in the protocol artifact — "compute only later, once
frozen UPPER/FLOOR anchors exist" — so it is not produced here. `significance_test`
and `p_value` are both `false`; **no statistical test may be silently added.**

---

## 11. Official TEST seal — structural

`Preg1Role` has exactly `PROTOCOL_TRAIN`, `PROTOCOL_DEV` and
`OFFICIAL_VALIDATION`. **There is no `OFFICIAL_TEST` member**, so official TEST
cannot be *named* by any Stage-2 code path and no argument can carry it — the
seal is structural, not a check that could be forgotten.
`Preg1Role.OFFICIAL_VALIDATION.may_select` is `False`.

The execution notebook must therefore accept **no** official TEST path or role,
and must **not** offer a generic dataset-path option that could point at TEST.
Its only downstream input is the explicit official-validation / measurement-dev
file of CELL 6.

---

## 12. A/B contract

| | |
|---|---|
| `A_B_SELECTION` | **NO** |
| `A_B_WINNER` | **NONE** |
| `A_B_RANKING` | **NONE** |

D-S2-001 option (c) stands, D-S1B-001 intact with no exception. No winner rule,
tie-break, no-decision margin, significance test or p-value exists, and none may
be added. Both arms are mandatory and neither may be dropped after results are
seen. The §6.4 selected epochs are within-head only.

---

## 13. Negative evidence for this task

| | |
|---|---|
| `MEASUREMENT_DEV_READ` | **NO** |
| `DEGRADED_MEASUREMENT_CACHE_CREATED` | **NO** |
| `DEGRADED_MEASUREMENT_RESULTS_SEEN` | **NO** |
| `OFFICIAL_TEST_READ` | **NO** |

No dataset file was opened, hashed, counted, parsed or materialised. No Drive
access. No model run, no training, no scoring, no cache created. **No fabricated
measurement results appear anywhere in this audit, because none exist.**

---

## 14. Contradictions and blockers found

**One documentation contradiction, resolved; three gaps, each recorded rather
than papered over.**

### 14.0 The 23-vs-24 artifact-field discrepancy — resolved

Audits 051 §8 and 053 §6 call the head-artifact schema "23-field"; the source
tuple has **24** entries. Resolved mechanically in §6.3: the tuple's text at its
creation commit `6693e728…` is byte-identical to its text at this HEAD, and
`git log -L` over it reports only that one commit — so it was already 24 when
both audits were written. **The discrepancy is a prose miscount, not schema
drift**, no field was added later, and the validator enforces 24. Those audits
are not edited; this is the reconciliation, and Audit 057 does not claim "no
contradiction" while leaving it open.

1. **No canonical-payload digest function in the repository** (§6.4).
   `canonical_json_sha256` exists only in Audit 053's prose, and no raw file
   digest of the artifact is stored either. The ten payload digests keep their
   correct name and are recorded, not used as a gate, until the serialisation is
   specified.
2. **No head-state loader on `Stage2HeadRunStore`** (§6.5). `commit` exists;
   there is no `load`. The notebook must `torch.load` the committed `STATE_NAME`
   and call the private `_state_digest` to verify `selected_head_state_sha256` —
   now in **CELL 4, before the hard stop**, so a storage problem is never
   discovered after measurement-dev has been read.
3. **measurement-dev ordered-id and label digests are not pinned** (§7.2). They
   are computed at first read. Row count, label counts and the CSV SHA-256 *are*
   pinned, so identity is still gated on content rather than filename.

None blocks preparation. Items 1 and 2 are notebook-side obligations; item 3 is
inherent — those digests can only exist once the file is read, which is CELL 6.

**All five preparation repairs are internally resolved**: the field count is
source-derived (§6.3, §14.0); the campaign registry root is audit authority
(§6.1a); both finalist checkpoints are recorded with a pre-hard-stop preflight
(§6.1b, CELL 5); all ten selected-head states are digest-verified before any data
read (§6.5, CELL 4); and the MyDrive contract covers every persistent cache
(§8).

---

## 15. Files changed

| File | Change |
|---|---|
| `docs/audits/057-stage2-real-measurement-notebook-preparation.md` | **new** — this file |

One path. No implementation, test, protocol, spec, decision, config, scientific
constant, cache schema, head artifact or Stage-1 artifact was modified.

**What was executed, precisely.** No pytest run was required and none was
performed. Read-only source inspection and git hygiene commands **were** used —
reading modules under `unmark/evaluation/`, enumerating
`STAGE2_HEAD_ARTIFACT_FIELDS` and comparing its source text across commits,
`git diff --name-status`, `git diff --check`, `git status --short`. Those are
compatible with PREPARE-ONLY and are how the §6.3 field count and the §14.0
reconciliation were established rather than assumed. What did **not** run: the
measurement notebook, any model or data pipeline, any scientific run, any
training, any scoring, any cache creation, any Drive access, and any read of
`measurement-dev`.

---

## 16. Exact next allowed step

**Instantiate this reviewed plan into exact Colab cells — and do not execute
them.**

The instantiated cells must still be **separately reviewed before any
execution**, and the hard stop before CELL 6 must survive instantiation. No
measurement-dev read, no cache, no score, no Drive write is authorised by this
audit or by instantiation.

---

## 17. Final state

```
git status --short
?? docs/audits/057-stage2-real-measurement-notebook-preparation.md
```

`git diff --check` produced no output. `HEAD` is
`76f3ba8e7e1a054cf4ec56876547e6f43ef20867`, unchanged. Uncommitted and awaiting
author review; nothing staged, committed, pushed or history-mutated.

---

```
AUDIT056_MEASUREMENT_GUARD_RUNTIME_ACCEPTANCE=PASS

DOCUMENTATION_HEAD_AT_PREPARATION_START=76f3ba8e7e1a054cf4ec56876547e6f43ef20867
MEASUREMENT_ACCEPTED_EXECUTION_HEAD=57b3d31e1cf8145835117efee43a3f5172075ab8
CLEAN_HEAD_CAMPAIGN_EXECUTION_HEAD=6693e728ccaebc987e4786bd5cd5e0f5c16143f7

STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=True

MEASUREMENT_ARMS=2
MEASUREMENT_CONDITIONS_PER_ARM=6
MEASUREMENT_REPRESENTATION_CACHES_PLANNED=12
FROZEN_REAL_HEADS=10
MEASUREMENT_HEAD_CONDITION_SCORES_PLANNED=60

A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE

MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
OFFICIAL_TEST_READ=NO

STAGE2_HEAD_ARTIFACT_FIELDS_SOURCE_COUNT=24
CAMPAIGN_REGISTRY_ROOT=KNOWN
CLEAN_CAMPAIGN_MANIFEST_FILE_SHA256=4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962
CLEAN_CAMPAIGN_MANIFEST_SEMANTIC_DIGEST=571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6
TEN_SELECTED_EPOCHS_EXACT_GATE=SPECIFIED_BEFORE_DATA_READ
FINALIST_CHECKPOINT_PREFLIGHT=SPECIFIED_BEFORE_HARD_STOP
TEN_SELECTED_HEAD_STATES_VERIFIED_BEFORE_DATA_READ=SPECIFIED
PERSISTENT_CACHE_NAMESPACE=MYDRIVE_BACKUP_ONLY

REAL_MEASUREMENT_NOTEBOOK_PLAN=PREPARED
REAL_MEASUREMENT_NOTEBOOK_INSTANTIATED=NO
READY_FOR_REAL_MEASUREMENT_NOTEBOOK_INSTANTIATION=YES
READY_FOR_REAL_MEASUREMENT_EXECUTION=NO
```

`MEASUREMENT_REPRESENTATION_CACHES_PLANNED=12` and
`MEASUREMENT_HEAD_CONDITION_SCORES_PLANNED=60` are deliberately separate numbers
(§3.1): extraction is head-independent, scoring is not. The broad
`DOWNSTREAM_RESULTS_SEEN` flag is absent, for the reason Audit 053 §11 gives.
