# Audit 060 - Stage-2 Tone Channel: L4 Runtime Acceptance Closeout and Corrected Clean Campaign Authorization

**Scope:** close out the accepted Audit-059 tone-channel repair under the
author-selected NVIDIA L4 runtime, record the provenance of the L4 evidence, and
state the exact next allowed corrected Stage-2 clean-head campaign.
**Date:** 2026-09-10
**Type:** **documentation-only provenance closeout.** No implementation, test,
protocol constant, spec schema, scientific hyperparameter, cache schema, head
artifact schema, dataset artifact or historical audit is modified.

---

## 1. Executive verdict

**AUDIT059_CORRECTED_TONE_PATH_L4_RUNTIME_ACCEPTANCE=PASS.**

The Stage-2 tone-channel repair accepted in Audit 059 now has authoritative
runtime evidence on the author-selected continuation GPU, **NVIDIA L4**, at the
exact execution HEAD:

```text
cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

This closes the runtime eligibility of the repair. It proves that the corrected
Stage-2 path receives observable tone-state changes. It does **not** establish
downstream sentiment performance, rank UNMARK-A versus UNMARK-B, choose a seed,
select an arm, justify reusing the historical ten Stage-2 heads, read official
validation, or read TEST.

| | |
|---|---|
| Audit-059 repair implementation HEAD | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` |
| Accepted runtime GPU for the corrected continuation | **NVIDIA L4** |
| L4 runtime acceptance | **PASS** |
| Historical A100 statements | **preserved as historical facts** |
| Live A100-only requirement found | **NO** |
| Cross-GPU bitwise equivalence claimed | **NO** |
| Stage-1 checkpoints retrain required | **NO** |
| Corrected clean caches need rebuild | **YES** |
| Historical ten Stage-2 heads reusable | **NO** |
| Corrected ten Stage-2 heads retrain required | **YES** |
| Stage-2 retraining started here | **NO** |
| Official validation read here | **NO** |
| Official TEST read here | **NO** |
| A/B selection | **NONE** |

The next scientific execution is therefore authorised only as a **corrected
Stage-2 clean campaign** under the existing frozen Stage-2 protocol and the
integrity gates in `unmark/evaluation/stage2_head_campaign.py` and
`unmark/evaluation/stage2_campaign.py`.

---

## 2. Starting repository state

Verified before this audit file was written:

```text
branch : main
HEAD   : cb78114eb0fac3167fbc1f618d7e8267ab24b114
status : clean (git status --short --branch produced only ## main...origin/main)
```

No Drive path was opened, no UIT-VSFC material was read, no model was trained,
and no production file was edited during this closeout.

---

## 3. Files inspected

The closeout inspected the files and surfaces relevant to the runtime,
provenance and next-step boundary:

| Area | Files / commands |
|---|---|
| Current HEAD and worktree | `git rev-parse HEAD`; `git status --short --branch` |
| Prior audit | `docs/audits/059-stage2-tone-channel-eligibility-runtime-bug.md` |
| Historical Stage-2 runtime records | `docs/audits/050-stage2-dual-finalist-infrastructure-implementation.md`; `docs/audits/051-stage2-head-campaign-runner-implementation.md`; `docs/audits/052-stage2-real-clean-cache-materialization-pre-training-acceptance.md`; `docs/audits/053-stage2-clean-10-head-campaign-runtime-closeout.md`; `docs/audits/054-stage2-measurement-corruption-seed-freeze.md`; `docs/audits/055-stage2-measurement-guard-runtime-test-fixture-repair.md`; `docs/audits/056-stage2-measurement-guard-runtime-acceptance-closeout.md`; `docs/audits/057-stage2-real-measurement-notebook-preparation.md`; `docs/audits/058-stage2-measurement-manifest-binding-runtime-correction.md` |
| Live protocol / decision docs | `docs/spec/decisions.md`; `docs/spec/stage2-dual-finalist-protocol.json`; `docs/spec/neural-adapter.md`; `README.md`; `unmark-proposal.md` |
| Stage-2 APIs | `unmark/evaluation/stage2_dual_finalist.py`; `unmark/evaluation/stage2_head_campaign.py`; `unmark/evaluation/stage2_campaign.py`; `unmark/evaluation/preg1_protocol.py` |
| Helper resolution | `unmark/stage1/execute.py`; `unmark/linguistics/__init__.py`; `unmark/linguistics/inventory.py`; `unmark/linguistics/classify.py` |
| Test contract surface | `tests/test_stage2_tone_channel_regression.py`; `tests/test_stage2_dual_finalist_infra.py`; `tests/test_stage2_head_campaign.py`; `tests/test_stage2_campaign.py`; `tests/test_preg1_import_contract.py`; `tests/test_stage1_finalist_checkpoint_torch.py` |
| Dependency declarations | `requirements/base.txt`; `requirements/experiment.txt`; `requirements/monitoring.txt`; `requirements/dev.txt`; root `requirements.txt` absence checked |

Searches for live A100-only language were run over `README.md`, `docs/spec`,
`docs/colab`, `scripts`, `unmark`, `tests`, and `unmark-proposal.md`, excluding
historical audit text where appropriate.

---

## 4. A100 versus L4: states kept separate

There are four distinct states. They must not be collapsed into one provenance
claim.

| State | Runtime / identity | Meaning |
|---|---|---|
| Historical A100 acceptances | A100 records in Audits 050, 051, 052, 055 and 056 | Historical runtime facts for the then-current implementations and guards. They remain true historical records and are not rewritten. |
| Historical clean Stage-2 campaign | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` | Produced the four historical clean caches and ten historical heads. Audit 059 established these were produced through the dead-tone `FULL` pathway and are scientifically invalid for corrected reporting. |
| Current Audit-059 repair implementation | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` | The accepted fail-closed tone-channel repair implementation. |
| Current L4 runtime acceptance | `cb78114eb0fac3167fbc1f618d7e8267ab24b114` on NVIDIA L4 | Proves the repaired path receives tone-state changes on the author-selected continuation runtime. |
| Future corrected clean campaign | Must execute at `cb78114eb0fac3167fbc1f618d7e8267ab24b114` unless the author explicitly changes implementation HEAD before execution | Regenerate corrected clean `FULL` caches and retrain exactly ten clean heads in a new namespace. |

The author explicitly changed the corrected continuation runtime from A100 to
**NVIDIA L4**. This audit records that current-state operational choice. It does
not rewrite historical A100 statements, and it does not assert that A100 and L4
are bitwise equivalent. Cross-GPU bitwise equivalence was not tested and is not
required for this closeout.

The live protocol does not require A100. The frozen Stage-2 protocol binds the
dataset, arms, checkpoint hashes, roles, conditions, seed `19225`, pooling,
optimizer, head architecture, epoch rule, reporting policy and TEST seal. It
does not bind a GPU SKU.

---

## 5. Accepted L4 evidence

Final evidence artifact:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-tone-channel-l4-runtime-acceptance/cb78114eb0fa/20260910T005756Z/059-corrected-tone-path-l4-runtime-acceptance-final.json
```

Raw SHA-256:

```text
9d80506f856a42e5411f0dfbbda13a10a4b475a35255ceb2eaae41827a9f5e19
```

The repository did **not** open or verify that Drive path during this closeout.
The path and hash above are recorded from the author's authoritative runtime
evidence statement; no artifact hash is invented here.

Accepted facts from the L4 evidence:

```text
AUDIT059_CORRECTED_TONE_PATH_L4_RUNTIME_ACCEPTANCE=PASS
EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
GPU=NVIDIA_L4
REAL_INVENTORY_CLASSIFIER=PASS
CLASSIFIER_NONE_FAILS_CLOSED=YES
FULL_P100_BASE_GRID_INVARIANT=YES
FULL_P100_TONE_IDS_DIFFER=YES
UNMARK_A_FULL_P100_REPRESENTATIONS_DIFFER=YES
UNMARK_B_FULL_P100_REPRESENTATIONS_DIFFER=YES
STAGE1_RETRAIN_REQUIRED=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
CORRECTED_CLEAN_CACHE_REBUILD_REQUIRED=YES
CORRECTED_10_STAGE2_HEADS_RETRAIN_REQUIRED=YES
STAGE2_RETRAINING_STARTED=NO
UIT_VSFC_PROTOCOL_TRAIN_READ=NO
UIT_VSFC_PROTOCOL_DEV_READ=NO
MEASUREMENT_DEV_READ=NO
OFFICIAL_TEST_READ=NO
AUTOMATIC_AB_SELECTION=NO
AUTHOR_AB_DECISION_MADE=NO
```

Runtime details:

| | |
|---|---|
| Python | `3.13.15` |
| torch | `2.11.0+cu128` |
| CUDA runtime | `12.8` |
| GPU | NVIDIA L4 |
| Compute capability | `8.9` |
| Deterministic algorithms | enabled |
| TF32 matmul | disabled |
| TF32 cuDNN | disabled |

Pinned Vietnamese syllable inventory:

| | |
|---|---|
| SHA-256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Size | `116290` bytes |

Stage-1 finalist checkpoint identities remained byte-identical and passed the
authoritative verifier:

| Arm | Checkpoint SHA-256 |
|---|---|
| UNMARK-A | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| UNMARK-B | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |

---

## 6. Helper-source correction

The first notebook wrapper guessed that `try_load_inventory` and
`make_classifier` were exported by `unmark.stage1.execute`. That import failed
before any scientific operation.

Source-driven continuation resolved the actual public helpers:

```text
try_load_inventory -> unmark.linguistics.try_load_inventory
make_classifier    -> unmark.linguistics.make_classifier
```

The relevant source confirms this resolution:

| Helper | Source |
|---|---|
| public exports | `unmark/linguistics/__init__.py` |
| inventory loader | `unmark/linguistics/inventory.py` |
| classifier factory | `unmark/linguistics/classify.py` |
| Stage-1 internal use | `unmark/stage1/execute.py` imports from `unmark.linguistics` inside the execution path |

This is a notebook-driver import error, not a scientific model failure, not a
checkpoint failure, and not evidence against the repaired Stage-2 implementation.
The failure happened before dataset access, forward computation, cache writing,
training, measurement or TEST access.

---

## 7. Dependency-wrapper correction

Earlier runtime wrapper attempts assumed a root `requirements.txt`. The
repository does not have one. This closeout checked that root
`requirements.txt` is absent.

The declared requirement files are:

```text
requirements/base.txt
requirements/experiment.txt
requirements/monitoring.txt
requirements/dev.txt
```

For the corrected Stage-2 Colab continuation, the relevant experiment dependency
declaration remains `requirements/experiment.txt`, with Colab's CUDA PyTorch
provided by the runtime as documented in `README.md`. The failed root-file
attempts were dependency-wrapper mistakes before scientific execution, not
scientific failures.

---

## 8. Corrected tone-path proof, interpreted narrowly

Synthetic semantic proof from the L4 acceptance:

```text
clean:
"Tôi đã học"

P100:
"Tôi đa hoc"

FULL/P100 base input IDs:
identical

FULL tone_ids:
(-1, 5, 5, 3, 4, -1)

P100 tone_ids:
(-1, 5, 5, 5, 5, -1)

tone-difference positions:
[3, 4]

P100 letter channel:
unchanged relative to FULL

STRIP_ALL letter channel:
changed
```

Real frozen representation proof from the L4 acceptance:

| Arm | Shape | FULL/P100 exact equal | max abs delta | mean abs delta | cosine |
|---|---:|---|---:|---:|---:|
| UNMARK-A | `(1, 768)` | `False` | `0.5205746889` | `0.1421983391` | `0.882626176` |
| UNMARK-B | `(1, 768)` | `False` | `2.036383390` | `0.3542406559` | `0.233094439` |

These values are implementation evidence only: the repaired path propagates
tone-state changes into frozen representations. They are **not** downstream
sentiment metrics, not an A/B ranking, and not evidence that either arm is
better.

---

## 9. Consequence for historical clean caches and heads

Audit 059 established the dead-tone failure was present in `FULL`, not merely in
degraded conditions. The L4 acceptance confirms that the corrected `FULL` pathway
receives real tone states.

Therefore:

```text
CLEAN_FULL_PATH_AFFECTED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
CORRECTED_CLEAN_CACHE_REBUILD_REQUIRED=YES
CORRECTED_10_STAGE2_HEADS_RETRAIN_REQUIRED=YES
STAGE1_CHECKPOINTS_RETRAIN_REQUIRED=NO
```

The historical measurement-v2 artifacts remain preserved as evidence of the
discovered bug and remain scientifically invalid for degraded reporting. They
must not be overwritten, mutated, relabelled as corrected evidence, or used as
inputs to corrected reporting.

Stage-1 is not implicated because Stage-1 already constructed the real inventory
classifier. The two frozen Stage-1 finalist checkpoints remain the inputs to the
corrected Stage-2 continuation.

---

## 10. Live A100-only requirement check

This closeout searched for active A100 requirements in the current
non-historical surfaces:

```text
README.md
docs/spec
docs/colab
scripts
unmark
tests
unmark-proposal.md
```

Result:

```text
LIVE_A100_ONLY_REQUIREMENT_FOUND=NO
```

Only two non-audit A100 mentions were found:

| File | Meaning |
|---|---|
| `tests/test_stage2_dual_finalist_infra.py` | skip reason for an older authoritative A100 smoke context |
| `tests/test_stage1_finalist_checkpoint_torch.py` | fixture payload asserting that verifier preserves a recorded GPU name |

Neither is a live requirement that the corrected Stage-2 continuation must use
A100. Historical audit A100 statements remain untouched and historical.

Because no live A100-only requirement exists, this audit makes no protocol
amendment, no decision-log amendment, and no test-expectation amendment.

---

## 11. Next allowed execution: corrected Stage-2 clean campaign

The next authorised scientific execution after this documentation closeout is
the corrected clean Stage-2 campaign, and only that campaign.

Required execution identity and runtime:

```text
EXECUTION_IMPLEMENTATION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
RUNTIME_GPU=NVIDIA_L4
```

Allowed data access in that execution:

| Material | Status |
|---|---|
| UIT-VSFC derived protocol-train | **allowed**, only as needed for corrected clean-cache extraction and head training |
| UIT-VSFC derived protocol-dev | **allowed**, only as needed for corrected clean-cache extraction and within-head epoch selection |
| official validation / measurement-dev | **must remain unread** |
| official TEST | **SEALED** |

The execution must:

1. verify existing split identities and digests before extraction;
2. construct the real pinned inventory classifier explicitly from
   `unmark.linguistics`;
3. regenerate exactly four corrected clean `FULL` representation caches:
   `UNMARK-A protocol-train`, `UNMARK-A protocol-dev`,
   `UNMARK-B protocol-train`, `UNMARK-B protocol-dev`;
4. use a **new MyDrive namespace** and never overwrite or mutate historical
   `6693e728...` clean caches or heads;
5. retrain exactly `2 arms x 5 frozen measurement seeds = 10` independent linear
   heads;
6. preserve the existing frozen Stage-2 protocol:
   encoder frozen, adapter frozen, `Linear(768, 3, bias=True)`, clean
   protocol-train only for training, 30 complete epochs, LR `0.01`, AdamW, batch
   size `128`, within-head epoch selection only on clean protocol-dev, and
   selection ordering `Macro-F1 -> accuracy -> earliest epoch`.

Use this exact terminology for the selection boundary:

```text
Có selection, nhưng chỉ là chọn epoch tốt nhất của từng head; không phải chọn A hay B.
```

Five seeds per arm remain scientific replicates, not candidates for
cherry-picking. Do not select the best seed.

---

## 12. Later measurement remains separate

After all ten corrected clean heads are complete, measurement is a separate
later execution. It must not be folded into the clean-head campaign.

The later measurement, when separately authorised, is:

```text
A/B x FULL,P25,P50,P75,P100,STRIP_ALL
same official validation rows
same degradation corruption seed 19225
```

Those metrics are then presented to the author. The author decides which
arm/version to continue. No notebook or code automatically selects or drops A/B.

---

## 13. Local verification for this closeout

This audit is documentation-only. The committed runtime acceptance evidence is
the L4 artifact recorded in §5. Local verification is limited to repository
static/contract checks that do not train the Stage-2 campaign, do not read
Drive, do not read UIT-VSFC, and do not read TEST.

The checks for this closeout are:

```text
python -m json.tool docs/spec/stage2-dual-finalist-protocol.json
pytest -q tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_preg1_import_contract.py
git diff --check
git diff --no-index --check /dev/null docs/audits/060-stage2-tone-channel-l4-runtime-acceptance-closeout.md
```

Results:

| Check | Result |
|---|---|
| `python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.formatted` | **PASS** |
| `python -m pytest -q tests/test_stage2_tone_channel_regression.py tests/test_stage2_dual_finalist_infra.py tests/test_stage2_head_campaign.py tests/test_stage2_campaign.py tests/test_preg1_import_contract.py` | **206 passed, 24 skipped** |
| `git diff --check` | **PASS** |
| `git diff --no-index --check /dev/null docs/audits/060-stage2-tone-channel-l4-runtime-acceptance-closeout.md` | **PASS** -- no whitespace errors printed. Exit status is nonzero because `/dev/null` differs from the added file. |

---

## 14. Final state

```text
AUDIT060_DOCUMENTATION_CLOSEOUT=PASS
AUDIT059_CORRECTED_TONE_PATH_L4_RUNTIME_ACCEPTANCE=PASS

EXECUTION_HEAD_FOR_ACCEPTED_REPAIR=cb78114eb0fac3167fbc1f618d7e8267ab24b114
AUTHOR_SELECTED_CORRECTED_CONTINUATION_GPU=NVIDIA_L4
CROSS_GPU_BITWISE_EQUIVALENCE_CLAIMED=NO
LIVE_A100_ONLY_REQUIREMENT_FOUND=NO

CLEAN_FULL_PATH_AFFECTED=YES
CORRECTED_CLEAN_CACHE_REBUILD_REQUIRED=YES
HISTORICAL_CLEAN_CACHES_REUSABLE=NO
HISTORICAL_10_STAGE2_HEADS_REUSABLE=NO
CORRECTED_10_STAGE2_HEADS_RETRAIN_REQUIRED=YES
STAGE1_CHECKPOINTS_RETRAIN_REQUIRED=NO

STAGE2_RETRAINING_STARTED=NO
UIT_VSFC_PROTOCOL_TRAIN_READ=NO
UIT_VSFC_PROTOCOL_DEV_READ=NO
MEASUREMENT_DEV_READ=NO
OFFICIAL_VALIDATION_READ=NO
OFFICIAL_TEST_READ=NO

AUTOMATIC_AB_SELECTION=NO
AUTHOR_AB_DECISION_MADE=NO
A_B_WINNER=NONE

NEXT_ALLOWED_STEP=CORRECTED_STAGE2_CLEAN_CAMPAIGN_ONLY
NEXT_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
NEXT_RUNTIME_GPU=NVIDIA_L4
NEXT_OFFICIAL_VALIDATION_READ=NO
NEXT_OFFICIAL_TEST_READ=NO
```

**CORRECTED TONE-PATH L4 RUNTIME ACCEPTANCE CLOSED. READY FOR THE SEPARATE
CORRECTED STAGE-2 CLEAN CAMPAIGN UNDER THE FROZEN PROTOCOL.**
