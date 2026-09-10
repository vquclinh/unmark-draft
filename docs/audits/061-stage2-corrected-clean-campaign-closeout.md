# Audit 061 - Stage-2 Corrected Clean Campaign Closeout

**Scope:** close out the completed corrected Stage-2 clean campaign that follows
the Audit-059 tone-channel repair and Audit-060 NVIDIA L4 runtime acceptance.
**Date:** 2026-09-10
**Type:** **documentation-only provenance closeout.** No production code, tests,
protocol constants, scientific hyperparameters, cache schemas, head artifact
schemas, historical audits, dataset artifacts or Drive artifacts are modified.

---

## 1. Executive verdict

**CORRECTED_STAGE2_CLEAN_10_HEAD_CAMPAIGN=PASS.**

The corrected Stage-2 clean campaign completed successfully on **NVIDIA L4** at
the repaired scientific execution implementation HEAD:

```text
cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

The campaign rebuilt the four clean `FULL` representation caches and trained
exactly ten new Stage-2 heads:

```text
2 arms x 5 frozen seeds = 10 heads
```

This closes the clean-head prerequisite for corrected Stage-2 measurement. It
does **not** read official validation, read TEST, score degraded measurement
conditions, rank UNMARK-A versus UNMARK-B, select a best seed, or create a
deployable winner.

| | |
|---|---|
| Documentation HEAD before this audit | `b3949a2cb07ca1ef8e055d6e809a8164e27b25d2` |
| Scientific execution implementation HEAD | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` |
| Corrected clean campaign runtime GPU | NVIDIA L4 |
| Corrected clean caches created | **4 / 4** |
| Corrected Stage-2 heads trained | **10 / 10** |
| Campaign complete | **YES** |
| Campaign winner | `null` |
| Campaign ranks arms | `false` |
| Official validation read | **NO** |
| Measurement-dev read | **NO** |
| Official TEST read | **NO** |
| A/B selection | **NONE** |
| Stage-1 retraining required | **NO** |
| Historical clean caches reusable | **NO** |
| Historical ten heads reusable | **NO** |

---

## 2. Repository state for this closeout

Verified before this audit file was written:

```text
branch : main
HEAD   : b3949a2cb07ca1ef8e055d6e809a8164e27b25d2
status : clean (git status --short --branch produced only ## main...origin/main)
```

This documentation checkout is not the scientific execution checkout. The
scientific cache and head artifacts closed here bind:

```text
repository_head=cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

The documentation commit produced after this audit must not be used as the
Stage-2 representation cache `repository_head`. Corrected measurement caches
continue to bind the execution implementation HEAD
`cb78114eb0fac3167fbc1f618d7e8267ab24b114` unless a later scientific code change
is separately reviewed and accepted.

No Drive path was opened, no UIT-VSFC material was read, no training was run,
and no production file was edited during this closeout.

---

## 3. Files inspected

The closeout inspected the required audit, protocol, implementation and test
surfaces:

| Area | Files / commands |
|---|---|
| Current HEAD and worktree | `git rev-parse HEAD`; `git status --short --branch` |
| Prior repair and runtime closeout | `docs/audits/059-stage2-tone-channel-eligibility-runtime-bug.md`; `docs/audits/060-stage2-tone-channel-l4-runtime-acceptance-closeout.md` |
| Stage-2 protocol / decisions | `docs/spec/decisions.md`; `docs/spec/stage2-dual-finalist-protocol.json` |
| Corrected tone path | `unmark/evaluation/stage2_dual_finalist.py` |
| Clean cache and head runner | `unmark/evaluation/stage2_head_campaign.py` |
| Campaign manifest / registry | `unmark/evaluation/stage2_campaign.py` |
| Inherited protocol constants | `unmark/evaluation/preg1_protocol.py`; `unmark/evaluation/preg1_head.py` |
| Relevant committed tests | `tests/test_stage2_tone_channel_regression.py`; `tests/test_stage2_dual_finalist_infra.py`; `tests/test_stage2_head_campaign.py`; `tests/test_stage2_campaign.py`; `tests/test_preg1_import_contract.py` |

No current authoritative spec required a change for this closeout. The existing
Stage-2 protocol and runner already encode the clean-cache roles, the 2-arm x
5-seed campaign, within-head checkpoint selection, no A/B selection, and the
official-TEST seal.

---

## 4. Provenance states kept separate

These states have different meanings and must not be merged:

| State | Identity | Meaning |
|---|---|---|
| Historical dead-tone clean campaign | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` | Produced the historical clean caches and ten heads through the classifier-less dead-tone `FULL` pathway. Preserved as history; not reusable for corrected reporting. |
| Audit-059 repaired implementation | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` | The fail-closed tone-channel repair implementation. |
| Audit-060 L4 runtime acceptance closeout | documentation HEAD `b3949a2cb07ca1ef8e055d6e809a8164e27b25d2`, anchored to runtime evidence SHA `9d80506f856a42e5411f0dfbbda13a10a4b475a35255ceb2eaae41827a9f5e19` | Established that the repaired tone path receives observable tone-state changes on NVIDIA L4. |
| Corrected clean campaign closed here | execution HEAD `cb78114eb0fac3167fbc1f618d7e8267ab24b114`, NVIDIA L4 | Rebuilt the four clean `FULL` caches and trained the ten corrected clean heads. |
| Future corrected measurement | must bind `cb78114eb0fac3167fbc1f618d7e8267ab24b114` unless a future reviewed implementation change supersedes it | Will score the corrected heads on official validation across six conditions. |

No cross-GPU bitwise equivalence between A100 and L4 is claimed. The corrected
campaign is accepted on the author-selected L4 runtime; that is sufficient for
this closeout.

---

## 5. Evidence anchors

Audit-059 L4 runtime acceptance anchor, already accepted in Audit 060:

```text
9d80506f856a42e5411f0dfbbda13a10a4b475a35255ceb2eaae41827a9f5e19
```

Final durable corrected clean-campaign evidence path:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/cb78114eb0fa/audit060-b3949a2c/corrected-clean-10-head-campaign-v1/stage2-corrected-clean-10-head-campaign-final.json
```

Final durable evidence raw SHA-256:

```text
7924a7c5b576e3b0011f485dd4564edb46bf57fd3c2411f232cb85fb71ffe625
```

Additional evidence raw SHA-256 values:

| Evidence | Raw SHA-256 |
|---|---|
| Corrected clean-cache evidence | `31236738cf667ddc680730b388042c4a5d5f9946854edcd0fe9d3c2c42b41e74` |
| Corrected pretraining evidence | `0710a45b61414391755ce229b155dfd7b2463ce1ecc19673b655f0e2d3547564` |

The repository did **not** open, hash or verify those Drive paths during this
closeout. They are recorded from the author's authoritative evidence statement;
no artifact hash is invented here.

Corrected campaign semantic manifest digest:

```text
d3af6f1360389280ecb090ba46d2b3041e35c32b7ee268f324958e56daa4b8e0
```

Final campaign state:

```text
complete=true
completed=10
expected=10
pending=[]
ranks_arms=false
winner=null
```

---

## 6. Corrected clean caches

Corrected clean cache root:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-clean-cache-preparation/cb78114eb0fa/audit060-b3949a2c/corrected-real-clean-cache-extraction-v1
```

Exactly four corrected clean `FULL` caches were created. All four are FP32
`[N, 768]` clean `FULL` representations bound to execution HEAD
`cb78114eb0fac3167fbc1f618d7e8267ab24b114`.

| Cache | Shape | Tensor raw SHA-256 |
|---|---:|---|
| UNMARK-A protocol-train | `(9139, 768)` | `e4032c0539d3658eef18a9db5e0c1bc6307bc26f6f0035f33eecd0cb30f91ac7` |
| UNMARK-A protocol-dev | `(2285, 768)` | `bcc3de1ef802bd3054cdff6f4e961590c15a9cdb8821d1af4d23df0b4c29b270` |
| UNMARK-B protocol-train | `(9139, 768)` | `bc0e63b4f2605b163c9a3a76c8746c179eba0694e551b5e73ee5ec3c7123987d` |
| UNMARK-B protocol-dev | `(2285, 768)` | `55000147858dbd56cfaa0d54148afd6d9e8d3625fdf83a3f22d9a4445a3a71a3` |

These hashes are the corrected clean-cache identities for downstream corrected
measurement. They do not license reuse of any historical `6693e728...` clean
cache.

---

## 7. Data and role boundaries

The completed clean campaign used only the already-authorized UIT-VSFC derived
TRAIN material needed for `protocol-train` and `protocol-dev`. This closeout did
not read that material.

Derived UIT-VSFC TRAIN source verified in the runtime evidence:

| | |
|---|---|
| Raw SHA-256 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| Rows | `11424` |

Frozen role membership remained:

| Role | Count | ordered-id digest | label digest |
|---|---:|---|---|
| protocol-train | `9139` | `2cad022dd4aabbc875e388030e453ba2479f1046bea884309398b79f7d878cbd` | `1c1377b2cd8c8016765229079fb469c2ead5c6ec85229d986f4344fa817f6128` |
| protocol-dev | `2285` | `63192edf811f8249404597103dc5ba4f48bb7843d242a592d13764013c095860` | `546817741453edcb968c4d5de3530e435c5de9374995c1665260e92aea07ca1c` |

Campaign data boundaries:

```text
HEAD_TRAINING_INPUT=clean protocol-train only
WITHIN_HEAD_EPOCH_SELECTION_INPUT=clean protocol-dev only
OFFICIAL_VALIDATION_READ=NO
MEASUREMENT_DEV_READ=NO
OFFICIAL_TEST_READ=NO
DEGRADED_MEASUREMENT_REPRESENTATION_CACHE_CREATED=NO
```

---

## 8. Historical dead-tone non-reuse proof

The corrected `FULL` representations were compared against the historical
classifier-less dead-tone clean caches produced at:

```text
6693e728ccaebc987e4786bd5cd5e0f5c16143f7
```

Sample ordering and labels were held fixed. The runtime evidence established
that every row changed:

| Cache | Rows changed | max abs delta | mean abs delta |
|---|---:|---:|---:|
| UNMARK-A protocol-train | `9139 / 9139` | `2.743312597` | `0.2112392336` |
| UNMARK-A protocol-dev | `2285 / 2285` | `1.944702625` | `0.2114745528` |
| UNMARK-B protocol-train | `9139 / 9139` | `2.715682030` | `0.2711474001` |
| UNMARK-B protocol-dev | `2285 / 2285` | `2.513433933` | `0.2709122598` |

Therefore the non-reuse conclusion is now empirical real-data evidence, not
only an inference from the synthetic Audit-059 reproduction:

```text
CLEAN_FULL_PATH_AFFECTED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
```

Delta magnitude is not a quality metric. It is implementation/provenance
evidence that the corrected clean pathway differs from the historical dead-tone
pathway.

---

## 9. Stage-1 state

Stage-1 was not retrained.

| Finalist | Checkpoint SHA-256 |
|---|---|
| A | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| B | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |

Therefore:

```text
STAGE1_RETRAIN_REQUIRED=NO
```

Stage-1 is not implicated because Stage-1 already supplied the real inventory
classifier; the corrected Stage-2 clean campaign reused the same frozen
Stage-1 finalist checkpoints.

---

## 10. Corrected ten heads

Exactly ten new Stage-2 heads were trained under the frozen protocol:

```text
2 arms x 5 frozen seeds
```

All ten use corrected clean `FULL` representations bound to execution HEAD
`cb78114eb0fac3167fbc1f618d7e8267ab24b114`.

Frozen seeds:

```text
53148
59945
42941
720
9428
```

Selected epochs and repository semantic selected-state digests:

| Arm | Seed | Selected epoch | selected_head_state_sha256 |
|---|---:|---:|---|
| UNMARK-A | `53148` | `16` | `03086d4af939ff70b4932259d3cf98b42033c44d959fc09f8e8a9f9b82d08b9d` |
| UNMARK-B | `53148` | `16` | `96bfc4778bd7393d4b8c858da6aa0777cea4f567841c98b71862b4640bbd1b15` |
| UNMARK-A | `59945` | `27` | `9112a7c4db26818e2d5cfcb29d3d828a989308932898f40d11da360823465d72` |
| UNMARK-B | `59945` | `8` | `a1e4807940b53b90c4af62211f30a583a181d22ce3cfa616990e8c4688d89eeb` |
| UNMARK-A | `42941` | `5` | `9f753edd39bc4e91891e0195ff8ce7c9f86f3520a0a7ce4d46bda8a276a288c5` |
| UNMARK-B | `42941` | `19` | `d7025bf90c0eedc15648933aa0f17d846eb7f96f3bcd9f106aa66732769f81c0` |
| UNMARK-A | `720` | `19` | `04946c013f1915256a1f81d8038854e5a766646bd4d3db8f32ff3c7d28f4aea5` |
| UNMARK-B | `720` | `26` | `bd43aa5f8e4fd59538c9b9d86b4682e625f25db6a2b2ed5bfafdd4103e89e88e` |
| UNMARK-A | `9428` | `21` | `94e8c040d9dfb88130c723dfc1b797612f6a2a019754bc01646ec638bb6e883c` |
| UNMARK-B | `9428` | `29` | `c1a26c8e152e34e138a6d4e900014fc272e7ce7c239169b1baf7e7f615e4bd5c` |

Interpretation:

```text
Có selection, nhưng chỉ là chọn epoch tốt nhất của từng head; không phải chọn A hay B.
```

The selected epochs are within-head checkpoint selections on clean
`protocol-dev` under the already frozen ordering:

```text
Macro-F1 -> accuracy -> earliest epoch
```

They do **not** select a best seed, select A or B, rank A and B, or create a
deployable winner. Five seeds per arm remain scientific replicates.

---

## 11. Frozen clean campaign protocol preserved

The completed campaign preserved the existing Stage-2 linear-head protocol:

| Field | Value |
|---|---|
| Encoder | frozen |
| Adapter | frozen |
| Head | `Linear(768, 3, bias=True)` |
| Training input | clean `protocol-train` only |
| Epoch selection input | clean `protocol-dev` only |
| Epochs | `30` complete epochs |
| Batch size | `128` |
| LR | `0.01` |
| Optimizer | AdamW |
| Epoch selection ordering | `Macro-F1 -> accuracy -> earliest epoch` |
| A/B selection | **NONE** |
| Best seed selection | **NONE** |

No downstream A/B performance result is recorded in this audit. The selected
epochs and selected-state digests are head-artifact provenance, not downstream
measurement evidence.

---

## 12. Notebook-driver recovery

The first attempt at the corrected clean campaign stopped before representation
extraction because the notebook wrapper called:

```python
cache_slot(request.arm, request.role)
```

`Stage2ExtractionRequest.role` is a string, while the repository-owned
`cache_slot` API expects a `Preg1Role`-like enum carrying `.value`.

The same-runtime continuation converted the request role string back through
`Preg1Role` before calling `cache_slot`. The campaign then completed
successfully.

This was notebook glue type mismatch before representation extraction, not a
production-model failure, not a scientific failure, and not a reason to modify
production code. No production code is changed to accommodate the wrapper error.

---

## 13. Next allowed step

After this documentation closeout is reviewed and committed by the author, the
next scientific step is **corrected Stage-2 measurement on official
validation**.

That measurement must:

1. use corrected execution implementation HEAD
   `cb78114eb0fac3167fbc1f618d7e8267ab24b114`;
2. use the ten corrected heads from this campaign, not the historical `6693...`
   heads;
3. use the same official validation rows for both A and B;
4. evaluate exactly `FULL`, `P25`, `P50`, `P75`, `P100`, `STRIP_ALL`;
5. use the pinned degraded corruption seed `19225`;
6. bind `FULL` to `corruption_seed=None` in representation identity;
7. explicitly provision the real pinned Vietnamese syllable inventory
   classifier;
8. verify that degraded text and the corrected tone/letter channels behave as
   intended before scores are accepted;
9. score all `2 arms x 5 heads x 6 conditions = 60` score units;
10. report per-head/per-condition Macro-F1, accuracy and per-class F1;
11. aggregate per arm as mean plus sample standard deviation over the five
    seeds;
12. report descriptive paired `B - A` differences;
13. do **not** automatically choose or rank A versus B;
14. present both arms' measurement results to the author;
15. leave the post-measurement continuation decision to the author;
16. keep official TEST sealed.

Do not silently reuse any artifact from the invalid measurement-v2 run. That run
remains preserved as historical bug evidence only. Do not overwrite historical
namespaces. The corrected measurement must use a new persistent namespace under:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/
```

---

## 14. Local verification for this closeout

This audit is documentation-only. The completed campaign evidence is the
external L4 runtime evidence recorded above. Local verification is limited to
repository static/contract checks that do not train, do not read Drive, do not
read UIT-VSFC, and do not read TEST.

The checks for this closeout are:

```text
python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.formatted
python -m pytest -q tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_preg1_import_contract.py
git diff --check
git diff --no-index --check /dev/null docs/audits/061-stage2-corrected-clean-campaign-closeout.md
```

Results:

| Check | Result |
|---|---|
| `python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.formatted` | **PASS** |
| `python -m pytest -q tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_preg1_import_contract.py` | **206 passed, 24 skipped** |
| `git diff --check` | **PASS** |
| `git diff --no-index --check /dev/null docs/audits/061-stage2-corrected-clean-campaign-closeout.md` | **PASS** -- no whitespace errors printed. Exit status is nonzero because `/dev/null` differs from the added file. |

---

## 15. Final state

```text
AUDIT061_DOCUMENTATION_CLOSEOUT=PASS
CORRECTED_STAGE2_CLEAN_10_HEAD_CAMPAIGN=PASS

DOCUMENTATION_HEAD_BEFORE_AUDIT061=b3949a2cb07ca1ef8e055d6e809a8164e27b25d2
SCIENTIFIC_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
RUNTIME_GPU=NVIDIA_L4

AUDIT059_L4_ACCEPTANCE_RAW_SHA256=9d80506f856a42e5411f0dfbbda13a10a4b475a35255ceb2eaae41827a9f5e19
CORRECTED_CAMPAIGN_FINAL_EVIDENCE_RAW_SHA256=7924a7c5b576e3b0011f485dd4564edb46bf57fd3c2411f232cb85fb71ffe625
CORRECTED_CAMPAIGN_SEMANTIC_MANIFEST_DIGEST=d3af6f1360389280ecb090ba46d2b3041e35c32b7ee268f324958e56daa4b8e0

CORRECTED_CLEAN_CACHES_CREATED=4
CORRECTED_STAGE2_HEADS_TRAINED=10
CAMPAIGN_COMPLETE=true
CAMPAIGN_COMPLETED=10
CAMPAIGN_EXPECTED=10
CAMPAIGN_PENDING=[]
CAMPAIGN_RANKS_ARMS=false
CAMPAIGN_WINNER=null

CLEAN_FULL_PATH_AFFECTED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
STAGE1_RETRAIN_REQUIRED=NO

OFFICIAL_VALIDATION_READ=NO
MEASUREMENT_DEV_READ=NO
OFFICIAL_TEST_READ=NO
DEGRADED_MEASUREMENT_REPRESENTATION_CACHE_CREATED=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE

NEXT_ALLOWED_STEP=CORRECTED_STAGE2_MEASUREMENT_ON_OFFICIAL_VALIDATION
NEXT_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
NEXT_DEGRADED_CORRUPTION_SEED=19225
NEXT_SCORE_UNITS=60
NEXT_AUTOMATIC_AB_SELECTION=NO
NEXT_OFFICIAL_TEST_READ=NO
```

**CORRECTED STAGE-2 CLEAN CAMPAIGN CLOSED. READY FOR THE SEPARATE CORRECTED
STAGE-2 MEASUREMENT EXECUTION, WITH OFFICIAL TEST STILL SEALED.**
