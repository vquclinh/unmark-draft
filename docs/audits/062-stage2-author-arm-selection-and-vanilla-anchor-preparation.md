# Audit 062 - Stage-2 Author Arm Selection and Vanilla Anchor Preparation

**Scope:** close out the corrected Stage-2 official-validation measurement v3,
record the author's post-measurement UNMARK arm decision, and freeze the next
smallest comparison step: vanilla PhoBERT `UPPER` / `FLOOR` anchors.
**Date:** 2026-09-10
**Type:** **documentation-only provenance closeout and next-step preparation.**
No production code, tests, protocol constants, scientific hyperparameters,
historical audits, dataset artifacts or Drive artifacts are modified.

---

## 1. Executive verdict

**CORRECTED_MEASUREMENT_V3=CLOSED.**

The corrected Stage-2 official-validation measurement v3 completed on
**NVIDIA L4** using the corrected UNMARK scientific execution implementation:

```text
cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

After reviewing the complete corrected official-validation measurement results,
the **AUTHOR** explicitly selected:

```text
UNMARK-A
```

UNMARK-A is the author-selected continuation arm after review of the complete
corrected official-validation measurement. This is a post-artifact author
decision, not an automatic winner and not a notebook-selected result. UNMARK-B
remains preserved as a measured scientific arm and must not be deleted,
overwritten, or retroactively relabelled.

The next authorized comparison is the cheapest remaining main-system baseline:
one frozen vanilla PhoBERT pathway producing `UPPER` / `FLOOR` anchors. `RESTORE`
and `ALIGN` remain later, larger experiments and are not authorized by this
audit.

| | |
|---|---|
| Documentation HEAD before this audit | `8fa991e5cb9623033aaecb5ca755503b665ea22c` |
| Corrected UNMARK scientific execution HEAD | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` |
| Corrected measurement v3 runtime GPU | NVIDIA L4 |
| Corrected measurement caches | **12 / 12** |
| Corrected measurement score units | **60 / 60** |
| Historical invalid measurement-v2 reused | **NO** |
| Historical `6693...` heads reused | **NO** |
| Best seed selection | **NO** |
| Automatic A/B selection | **NO** |
| Author A/B decision after review | **YES** |
| Author-selected continuation arm | **UNMARK-A** |
| Official TEST read | **NO** |
| Next baseline | `UPPER_FLOOR` |
| Next pathway | frozen vanilla PhoBERT |
| Next vanilla heads | `5` |
| Next vanilla measurement caches | `6` |
| Next vanilla score units | `30` |

---

## 2. Repository state for this closeout

Verified before this audit file was written:

```text
branch : main
HEAD   : 8fa991e5cb9623033aaecb5ca755503b665ea22c
status : clean (git status --short --branch produced only ## main...origin/main)
```

This documentation checkout is not the scientific execution checkout. The
corrected UNMARK caches, heads and measurement v3 evidence closed here bind:

```text
repository_head=cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

The documentation commit produced after this audit must not be used as a
Stage-2 representation cache `repository_head`. Corrected UNMARK artifacts
continue to bind the execution implementation HEAD
`cb78114eb0fac3167fbc1f618d7e8267ab24b114` unless a later scientific code change
is separately reviewed and accepted.

No Drive path was opened, no UIT-VSFC material was read, no official validation
file was read by this documentation closeout, no TEST file was read, no training
was run, and no production file was edited.

---

## 3. Files inspected

The closeout inspected the required audit, proposal, protocol, implementation
and test surfaces:

| Area | Files / commands |
|---|---|
| Current HEAD and worktree | `git rev-parse HEAD`; `git status --short --branch`; existence check for this Audit-062 path |
| Prior audits | `docs/audits/059-stage2-tone-channel-eligibility-runtime-bug.md`; `docs/audits/060-stage2-tone-channel-l4-runtime-acceptance-closeout.md`; `docs/audits/061-stage2-corrected-clean-campaign-closeout.md` |
| Proposal and decisions | `unmark-proposal.md`; `docs/spec/decisions.md`; `docs/spec/stage2-dual-finalist-protocol.json` |
| PREG1 / vanilla protocol | `unmark/evaluation/contracts.py`; `unmark/evaluation/preg1_protocol.py`; `unmark/evaluation/preg1_split.py`; `unmark/evaluation/preg1_head.py`; `unmark/evaluation/pathways.py`; `scripts/preg1_head_diagnostic.py` |
| Corrected UNMARK Stage-2 APIs | `unmark/evaluation/stage2_head_campaign.py`; `unmark/evaluation/stage2_campaign.py`; `unmark/evaluation/stage2_dual_finalist.py` |
| Corruption implementation | `unmark/corruption/deterministic.py`; `unmark/corruption/conditions.py`; package searches over `unmark/corruption` |
| RESTORE authority | `configs/restore/nrl_vit5_base.yaml`; `scripts/g_minus1_restore_smoke.py`; RESTORE searches over `README.md`, `docs/spec`, `scripts`, `unmark` and `tests` |
| ALIGN authority | ALIGN searches over `unmark-proposal.md`, `docs/spec`, `README.md`, `scripts`, `unmark` and `tests` |
| Relevant committed tests | `tests/test_evaluation_harness.py`; `tests/test_preg1_runner.py`; `tests/test_preg1_head.py`; `tests/test_corruption.py`; `tests/test_stage2_head_campaign.py`; `tests/test_stage2_campaign.py`; `tests/test_stage2_tone_channel_regression.py`; `tests/test_stage2_dual_finalist_infra.py`; `tests/test_preg1_import_contract.py` |

The inspection found that existing PREG1 vanilla infrastructure is sufficient at
the committed primitive/API level for the next `UPPER` / `FLOOR` execution. A
small execution wrapper or notebook should orchestrate the exact anchor flow in
a new persistent namespace, but no production implementation change is required
before the next compute.

---

## 4. Provenance states kept separate

These states have different meanings and must not be conflated:

| State | Identity | Meaning |
|---|---|---|
| Historical dead-tone Stage-2 campaign | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` | Produced the historical clean caches, heads and invalid measurement line through the classifier-less dead-tone pathway. Preserved as history only. |
| Audit-059 repaired implementation | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` | The accepted fail-closed tone-channel repair implementation. |
| Audit-060 L4 runtime acceptance closeout | L4 runtime evidence SHA `9d80506f856a42e5411f0dfbbda13a10a4b475a35255ceb2eaae41827a9f5e19` | Established that the repaired Stage-2 path receives observable tone-state changes on NVIDIA L4. |
| Audit-061 corrected clean campaign | execution HEAD `cb78114eb0fac3167fbc1f618d7e8267ab24b114`, NVIDIA L4 | Rebuilt four corrected clean `FULL` caches and trained ten corrected clean heads. |
| Corrected measurement v3 closed here | execution HEAD `cb78114eb0fac3167fbc1f618d7e8267ab24b114`, NVIDIA L4 | Scored the corrected heads on official validation across six conditions, with no automatic A/B selection and TEST still sealed. |
| Author decision recorded here | post-measurement documentation state | The author selected UNMARK-A after reviewing the completed corrected official-validation measurement. |
| Future vanilla anchor execution | must use a new persistent namespace under `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/` | Will train and measure one frozen vanilla PhoBERT pathway for `UPPER` / `FLOOR` anchors. |

No cross-GPU bitwise equivalence between A100 and L4 is claimed. The accepted
corrected UNMARK executions were run on the author-selected NVIDIA L4 runtime.

---

## 5. Corrected measurement v3 evidence

Final durable corrected measurement v3 evidence path:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement/cb78114eb0fa/audit061-8fa991e5/corrected-official-validation-v3/stage2-corrected-measurement-v3-final.json
```

Raw SHA-256:

```text
8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2
```

The repository did **not** open or verify this Drive path during this
documentation closeout. The path and hash are recorded from the author's
authoritative evidence statement; no artifact hash is invented here.

Accepted corrected measurement v3 state:

```text
CORRECTED_MEASUREMENT_CACHES=12_OF_12
MEASUREMENT_SCORE_UNITS=60_OF_60
HISTORICAL_INVALID_MEASUREMENT_V2_REUSED=NO
HISTORICAL_6693_HEADS_REUSED=NO
BEST_SEED_SELECTION=NO
AUTOMATIC_AB_SELECTION=NO
OFFICIAL_TEST_READ=NO
```

The v3 evidence artifact was written before the author's arm decision. Its
recorded state:

```text
AUTHOR_AB_DECISION_MADE=NO
```

is preserved as a correct fact about the instant the artifact was created. This
audit appends the later author-review state; it does not rewrite the evidence
artifact and does not call it wrong.

---

## 6. Official-validation identity observed by v3

The authorized corrected measurement v3 execution read official validation for
measurement. This documentation closeout did not read it.

Official validation identity observed at the authorized first read:

| Field | Value |
|---|---:|
| Rows | `1583` |
| Negative / `0` | `705` |
| Neutral / `1` | `73` |
| Positive / `2` | `805` |

| Digest | SHA-256 |
|---|---|
| Raw CSV | `9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0` |
| ordered-id digest | `825714672bcf27543825b85ac72c3fea2875647f55a117c41cc42ecc8e305d1a` |
| label digest | `70d5ed20e31dea2cacddc7cdb737ef3d0d1f980699b7e3c6d6aff0806711852b` |

Official TEST was not read and was not used for selection.

---

## 7. Corrected UNMARK results reviewed by the author

Aggregate official-validation results from the corrected v3 measurement:

### UNMARK-A

| Condition | Macro-F1 | Accuracy |
|---|---:|---:|
| `FULL` | `0.667074 +/- 0.017841` | `0.812508 +/- 0.005312` |
| `P25` | `0.644674 +/- 0.011589` | `0.797599 +/- 0.001874` |
| `P50` | `0.641626 +/- 0.007792` | `0.788882 +/- 0.009005` |
| `P75` | `0.624020 +/- 0.006570` | `0.763108 +/- 0.014461` |
| `P100` | `0.618434 +/- 0.013859` | `0.756665 +/- 0.015656` |
| `STRIP_ALL` | `0.610777 +/- 0.010004` | `0.751358 +/- 0.013330` |

Degraded equal-weight Macro-F1 mean:

```text
0.627906
```

### UNMARK-B

| Condition | Macro-F1 | Accuracy |
|---|---:|---:|
| `FULL` | `0.605885 +/- 0.008652` | `0.767783 +/- 0.003114` |
| `P25` | `0.585125 +/- 0.003424` | `0.731017 +/- 0.006196` |
| `P50` | `0.531266 +/- 0.008597` | `0.666835 +/- 0.012733` |
| `P75` | `0.495523 +/- 0.009996` | `0.627037 +/- 0.013743` |
| `P100` | `0.449115 +/- 0.015366` | `0.583828 +/- 0.016674` |
| `STRIP_ALL` | `0.442081 +/- 0.022048` | `0.572457 +/- 0.023978` |

Degraded equal-weight Macro-F1 mean:

```text
0.500622
```

Mean `B - A` Macro-F1:

| Condition | Mean `B - A` Macro-F1 |
|---|---:|
| `FULL` | `-0.061189` |
| `P25` | `-0.059549` |
| `P50` | `-0.110360` |
| `P75` | `-0.128497` |
| `P100` | `-0.169319` |
| `STRIP_ALL` | `-0.168696` |

Every one of the 30 paired arm/seed/condition Macro-F1 differences was
`B - A < 0`. Every one of the 30 paired accuracy differences was also
`B - A < 0`.

These are descriptive paired observations only. No significance test is
introduced. The selected epochs from Audit 061 remain within-head checkpoint
selection evidence, not downstream quality claims, and no best seed is selected.

---

## 8. Author decision recorded by this audit

After reviewing the complete corrected measurement v3 results, the author
selected:

```text
AUTHOR_AB_DECISION_MADE=YES
AUTHOR_SELECTED_UNMARK_ARM=UNMARK-A
AUTOMATIC_AB_SELECTION=NO
SELECTION_SOURCE=AUTHOR_REVIEW_OF_CORRECTED_OFFICIAL_VALIDATION_RESULTS
OFFICIAL_TEST_USED_FOR_SELECTION=NO
OFFICIAL_TEST_READ=NO
```

This audit does not call UNMARK-A an automatic winner. It records a human
continuation decision made after the corrected official-validation measurement
was complete. The notebook and repository did not automatically rank, select or
drop an A/B arm.

UNMARK-B remains part of the preserved scientific record. Its corrected caches,
heads and measurement results must remain durable and must not be overwritten,
deleted, or relabelled as failed.

---

## 9. Historical artifact non-reuse

The historical dead-tone artifacts remain preserved as evidence of the
Audit-059 bug and remain scientifically invalid for corrected reporting.

The corrected clean-campaign evidence in Audit 061 empirically established on
real data:

```text
CLEAN_FULL_PATH_AFFECTED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
```

The corrected measurement v3 evidence further established:

```text
HISTORICAL_INVALID_MEASUREMENT_V2_REUSED=NO
HISTORICAL_6693_HEADS_REUSED=NO
```

No artifact from the invalid measurement-v2 run may be silently reused. That run
remains historical bug evidence only.

Stage-1 remains unchanged:

```text
STAGE1_RETRAIN_REQUIRED=NO
```

The preserved Stage-1 finalist checkpoint identities remain:

| Finalist | Checkpoint SHA-256 |
|---|---|
| UNMARK-A | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| UNMARK-B | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |

---

## 10. Next experimental order

The author explicitly requests that remaining comparisons proceed from the
smallest, easiest and fastest experiment to larger experiments. Therefore the
next baseline is:

```text
UPPER / FLOOR
```

before `RESTORE` or `ALIGN`.

The proposal definitions remain:

| System | Meaning |
|---|---|
| `UPPER` | clean input, unmodified model |
| `FLOOR` | corrupted input, unmodified model |
| `RESTORE` | off-the-shelf diacritic restorer, then encode |
| `ALIGN` | contrastive alignment adapter on encoder output |
| `UNMARK` | input-side proposed method |

Gap Recovery Rate remains:

```text
GRR = (S_system - S_FLOOR) / (S_UPPER - S_FLOOR)
```

GRR is undefined when the denominator is zero. GRR for UNMARK-A remains
deferred until the `UPPER` / `FLOOR` anchors actually exist:

```text
UNMARK_A_GRR=DEFERRED_UNTIL_UPPER_FLOOR_ANCHORS_EXIST
```

---

## 11. UPPER / FLOOR pathway frozen here

The next experiment is **one vanilla PhoBERT pathway**, not two independently
trained neural systems:

```text
NEXT_BASELINE=UPPER_FLOOR
NEXT_BASELINE_PATHWAY=FROZEN_VANILLA_PHOBERT
```

Training:

1. extract one clean vanilla protocol-train representation cache;
2. extract one clean vanilla protocol-dev representation cache;
3. train five independent linear heads, one per frozen Stage-2 seed;
4. select the best epoch within each head on clean protocol-dev;
5. freeze each selected head.

Measurement:

1. `UPPER` is the same vanilla pathway and corresponding frozen heads on
   official-validation `FULL`;
2. `FLOOR` is the same vanilla pathway and corresponding frozen heads on
   corrupted official-validation `P25`, `P50`, `P75`, `P100`, `STRIP_ALL`;
3. degraded strings must use the same deterministic corruption implementation
   and corruption seed `19225` used by corrected UNMARK measurement;
4. the same official-validation rows, labels, order and corruption realization
   must be used for all conditions.

There are no separate `FLOOR` heads. Corrupted labelled examples are not exposed
during head training. Official validation is not used for epoch selection.

Selection language:

```text
Có selection, nhưng chỉ là chọn epoch tốt nhất của từng head.
```

There is no cross-seed selection. Five seeds are replicates.

---

## 12. Frozen fairness contract for UPPER / FLOOR

The vanilla anchors must use the same downstream head protocol as corrected
UNMARK:

| Component | Required value |
|---|---|
| Encoder | `vinai/phobert-base` |
| Encoder revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| Encoder state | frozen |
| Encoder mode | eval |
| Precision | FP32 |
| max_length | `256` |
| truncation | `true` |
| padding | `max_length` |
| Head | `Linear(768, 3, bias=True)` |
| Weight init | Xavier-uniform |
| Bias init | zero |
| Loss | cross-entropy mean |
| Optimizer | AdamW |
| AdamW betas | `(0.9, 0.999)` |
| AdamW eps | `1e-8` |
| Weight decay, weight | `0.01` |
| Weight decay, bias | `0` |
| Learning rate | `0.01` |
| Schedule | constant |
| Warmup | `0` |
| Batch size | `128` |
| Epochs | `30` complete epochs |
| Early stopping | none |
| Seeds | `53148`, `59945`, `42941`, `720`, `9428` |
| Epoch selection | highest Macro-F1, then highest accuracy, then earliest epoch |
| Selection split | clean protocol-dev only |
| Training split | clean protocol-train only |

The five frozen heads are scientific replicates, not candidates for
cherry-picking.

---

## 13. Data contract for UPPER / FLOOR

Clean head training must use the exact already-frozen UIT-VSFC derived TRAIN
membership:

| Field | Value |
|---|---:|
| Derived TRAIN raw SHA-256 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| Derived TRAIN rows | `11424` |
| protocol-train count | `9139` |
| protocol-dev count | `2285` |

| Role | ordered-id digest | label digest |
|---|---|---|
| protocol-train | `2cad022dd4aabbc875e388030e453ba2479f1046bea884309398b79f7d878cbd` | `1c1377b2cd8c8016765229079fb469c2ead5c6ec85229d986f4344fa817f6128` |
| protocol-dev | `63192edf811f8249404597103dc5ba4f48bb7843d242a592d13764013c095860` | `546817741453edcb968c4d5de3530e435c5de9374995c1665260e92aea07ca1c` |

Official validation may be read only after all five vanilla selected heads have
been frozen and verified. Official TEST remains sealed.

---

## 14. Expected next compute

The vanilla `UPPER` / `FLOOR` anchor experiment is the cheapest remaining
main-system comparison because it requires:

| Step | Count |
|---|---:|
| Clean training extraction | `1` vanilla protocol-train cache |
| Clean selection extraction | `1` vanilla protocol-dev cache |
| Head training | `5` small linear heads |
| Measurement extraction | `1` `FULL` official-validation cache plus `5` corrupted official-validation caches |
| Total vanilla measurement caches | `6` |
| Total measurement score units | `5 seeds x 6 conditions = 30` |

There is no Stage-1 adapter training, no UNMARK adapter forward, no restoration
model and no ALIGN training.

After the anchors exist, compute the descriptive comparison against the
already-author-selected UNMARK-A and compute condition-wise GRR for UNMARK-A.
Do not compute GRR before the anchors exist.

The new corrected measurement must use a new persistent namespace under:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/
```

Do not overwrite historical namespaces and do not reuse historical vanilla head
artifacts unless they are proven to match the now-required five seeds, learning
rate, complete schedule, split identities and scientific protocol exactly. In
the absence of that exact proof, create new anchor caches and heads.

---

## 15. Existing vanilla infrastructure reuse

The committed PREG1 infrastructure provides the required primitives for the
next vanilla anchor execution:

| Requirement | Reusable committed surface |
|---|---|
| Vanilla pathway identity | `SystemPathway.VANILLA` in `unmark/evaluation/contracts.py` |
| UPPER/FLOOR meaning | `LOCKED_EVALUATION_VALUES["grr_anchors"]` and `pathway_text(..., VANILLA)` |
| Frozen PhoBERT extraction | `load_frozen_encoder`, `encode_texts`, `extract_representations`, `extract_or_load` pattern |
| Provenance-bound caches | `RepresentationKey`, `RepresentationCache`, `BoundRepresentations` |
| Role separation | `Preg1Role.PROTOCOL_TRAIN`, `Preg1Role.PROTOCOL_DEV`, `Preg1Role.OFFICIAL_VALIDATION`; no `OFFICIAL_TEST` role |
| Clean train/dev membership guards | `load_derived_pool`, `load_membership`, `SplitMembership.require_partitions` |
| Head protocol | `build_head`, `build_optimizer`, `train_head`, `select_checkpoint`, `require_full_schedule` |
| Measurement scoring | `score_measurement`, `score_predictions`, Macro-F1 / accuracy / per-class metrics |
| Corruption | `unmark.corruption` deterministic conditions `FULL`, `P25`, `P50`, `P75`, `P100`, `STRIP_ALL` |

The current `scripts/preg1_head_diagnostic.py` runner is not itself the exact
Stage-2 `UPPER` / `FLOOR` runner: it was built for a VANILLA versus BASE_ONLY
diagnostic. The next execution should use its committed primitives and patterns
to run one VANILLA pathway across six official-validation conditions.

Conclusion:

```text
EXISTING_PREG1_VANILLA_INFRASTRUCTURE_SUFFICIENT=YES
NEW_PRODUCTION_IMPLEMENTATION_REQUIRED_BEFORE_UPPER_FLOOR=NO
```

A lightweight Colab/notebook orchestration layer is sufficient if it:

1. writes a new namespace;
2. verifies the derived TRAIN split identities before extraction;
3. builds clean protocol-train and protocol-dev VANILLA caches;
4. trains exactly five heads at LR `0.01` and the frozen seeds;
5. freezes each selected checkpoint by the within-head epoch rule;
6. only then verifies and reads official validation;
7. materializes the six official-validation VANILLA condition caches;
8. scores all `30` units;
9. reports descriptive aggregates and UNMARK-A GRR after anchors exist;
10. keeps official TEST sealed.

---

## 16. RESTORE and ALIGN status

`RESTORE` is **not yet authorized** for execution.

The repository has a locked G-1 smoke-test candidate configuration at
`configs/restore/nrl_vit5_base.yaml` for:

```text
model_id=nrl-ai/vn-diacritic-vit5-base
revision=30ea5a9e4a0b9436e18915fd4dbb5876eaee7325
generation.do_sample=false
generation.num_beams=1
generation.max_new_tokens=256
tokenizer.max_input_tokens=256
tokenizer.truncation=true
expected_dtype=float32
```

That locks an engineering smoke candidate. It does not, by itself, freeze the
full Stage-2 RESTORE baseline protocol. The current repository/spec authority
does not yet settle every RESTORE scientific detail needed for downstream
comparison, including how `FULL` input passes through the restorer in the main
measurement pathway, output normalization for scored inputs, and the exact
RESTORE cache/reporting identity for Stage-2.

`ALIGN` is **not yet authorized** for execution and remains scientifically open.
The proposal still names the ALIGN architecture and budget as open. The
evaluation contracts deliberately omit `RESTORE` and `ALIGN` system pathways.
The current repository/spec authority does not yet freeze the ALIGN baseline's
exact architecture, parameter budget, objective, Stage-1 training corpus,
corruption support, optimizer, schedule, seed/budget, or clean Stage-2 pathway.

Therefore:

```text
RESTORE_SCIENTIFIC_DEFINITION=STILL_OPEN_FOR_STAGE2_BASELINE
ALIGN_SCIENTIFIC_DEFINITION=STILL_OPEN
RESTORE_EXECUTION=NOT_YET_AUTHORISED
ALIGN_EXECUTION=NOT_YET_AUTHORISED
```

Do not invent a restorer policy or ALIGN design to move faster. They are later,
larger experiments.

---

## 17. Local verification for this closeout

This audit is documentation-only. The completed corrected measurement v3
evidence is the external L4 runtime evidence recorded above. Local verification
is limited to repository static/contract checks that do not train scientifically,
do not read Drive, do not read UIT-VSFC, do not read official validation, and do
not read TEST.

The checks for this closeout are:

```text
python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.audit062.formatted
python -m pytest -q tests/test_evaluation_harness.py tests/test_preg1_runner.py tests/test_preg1_head.py tests/test_corruption.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_preg1_import_contract.py
git diff --check
git diff --no-index --check /dev/null docs/audits/062-stage2-author-arm-selection-and-vanilla-anchor-preparation.md
```

Results:

| Check | Result |
|---|---|
| `python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.audit062.formatted` | **PASS** |
| `python -m pytest -q tests/test_evaluation_harness.py tests/test_preg1_runner.py tests/test_preg1_head.py tests/test_corruption.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_preg1_import_contract.py` | **967 passed, 64 skipped** |
| `git diff --check` | **PASS** |
| `git diff --no-index --check /dev/null docs/audits/062-stage2-author-arm-selection-and-vanilla-anchor-preparation.md` | **PASS** - no whitespace errors printed. Exit status is nonzero because `/dev/null` differs from the added file. |

---

## 18. Final state

```text
AUDIT062_DOCUMENTATION_CLOSEOUT=PASS

CORRECTED_MEASUREMENT_V3=CLOSED
CORRECTED_MEASUREMENT_V3_EVIDENCE_PATH=/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement/cb78114eb0fa/audit061-8fa991e5/corrected-official-validation-v3/stage2-corrected-measurement-v3-final.json
CORRECTED_MEASUREMENT_V3_EVIDENCE_SHA256=8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2

DOCUMENTATION_HEAD_BEFORE_AUDIT062=8fa991e5cb9623033aaecb5ca755503b665ea22c
CORRECTED_UNMARK_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
CORRECTED_MEASUREMENT_RUNTIME_GPU=NVIDIA_L4
CORRECTED_MEASUREMENT_CORRUPTION_SEED=19225

CORRECTED_MEASUREMENT_CACHES=12_OF_12
MEASUREMENT_SCORE_UNITS=60_OF_60
HISTORICAL_INVALID_MEASUREMENT_V2_REUSED=NO
HISTORICAL_6693_HEADS_REUSED=NO
BEST_SEED_SELECTION=NO

AUTHOR_AB_DECISION_MADE=YES
AUTHOR_SELECTED_UNMARK_ARM=UNMARK-A
AUTOMATIC_AB_SELECTION=NO
SELECTION_SOURCE=AUTHOR_REVIEW_OF_CORRECTED_OFFICIAL_VALIDATION_RESULTS
OFFICIAL_TEST_USED_FOR_SELECTION=NO

UNMARK_B_PRESERVED=YES
CLEAN_FULL_PATH_AFFECTED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
STAGE1_RETRAIN_REQUIRED=NO

NEXT_BASELINE=UPPER_FLOOR
NEXT_BASELINE_PATHWAY=FROZEN_VANILLA_PHOBERT
NEXT_VANILLA_HEAD_COUNT=5
NEXT_VANILLA_MEASUREMENT_CACHE_COUNT=6
NEXT_VANILLA_SCORE_UNIT_COUNT=30
NEXT_CORRUPTION_SEED=19225
UPPER_FLOOR_OFFICIAL_VALIDATION_READ_ALLOWED_ONLY_AFTER_HEADS_FROZEN=YES
UPPER_FLOOR_FLOOR_HEADS_SEPARATE=NO

EXISTING_PREG1_VANILLA_INFRASTRUCTURE_SUFFICIENT=YES
NEW_PRODUCTION_IMPLEMENTATION_REQUIRED_BEFORE_UPPER_FLOOR=NO

UNMARK_A_GRR=DEFERRED_UNTIL_UPPER_FLOOR_ANCHORS_EXIST

RESTORE_SCIENTIFIC_DEFINITION=STILL_OPEN_FOR_STAGE2_BASELINE
ALIGN_SCIENTIFIC_DEFINITION=STILL_OPEN
RESTORE_EXECUTION=NOT_YET_AUTHORISED
ALIGN_EXECUTION=NOT_YET_AUTHORISED

OFFICIAL_VALIDATION_READ_BY_THIS_DOC_CLOSEOUT=NO
OFFICIAL_TEST_READ=NO
```

**CORRECTED MEASUREMENT V3 CLOSED. AUTHOR SELECTED UNMARK-A FOR CONTINUATION.
NEXT AUTHORIZED COMPUTE IS VANILLA PHOBERT UPPER/FLOOR ANCHORING, WITH OFFICIAL
TEST STILL SEALED.**
