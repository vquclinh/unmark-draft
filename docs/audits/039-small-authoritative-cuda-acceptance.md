# Audit 039 — Small Authoritative CUDA Acceptance

**Scope:** record the authoritative CUDA acceptance of the Stage-1 scientific implementation.
**Date:** 2026-08-27
**Mode:** DOCUMENTATION ONLY. No production code, test, config or prior audit was modified.

> **Provenance of this record.** The CUDA acceptance was executed on the authoritative Colab GPU
> host. This audit was written in a separate documentation session on a local workstation that has
> neither that runtime nor Drive mounted. Two classes of statement are therefore kept distinct
> throughout:
>
> * **[CUDA]** — reported from the authoritative CUDA session.
> * **[VERIFIED HERE]** — independently re-derived from the committed source at the accepted HEAD in
>   this session.
>
> Nothing is presented as observed-here that was not. Where the CUDA session's values were not
> supplied, they are recorded as not captured rather than filled in from history.

---

## 1. Exact Accepted HEAD

**[VERIFIED HERE]**

```
$ git rev-parse HEAD
34232651f35132097097796c063bb5d3840f47bd

$ git log --oneline -3
3423265 test: fix mid-continuation resume probe
48911ae test: align Stage 1 resume fixtures
de4141a fix: harden Stage 1 training pipeline

$ git status --short          (clean)
$ git diff --cached           (nothing staged)
$ git diff --check            exit 0
```

`3423265` is the commit of the Audit 038 mid-continuation test repair
(`tests/test_stage1_resume_state_machine_torch.py`, +204 lines, plus Audit 038 itself). HEAD matched
exactly at the start of the acceptance and is unchanged now.

**[CUDA]** HEAD matched exactly; the working tree was clean apart from runtime-only ignored
resources; no tracked scientific file was modified.

## 2. Authoritative CUDA Runtime

**[CUDA]** The acceptance ran on the authoritative Colab GPU runtime with the dependency set in §3.

**Not captured:** the CUDA session's Python version, GPU model, driver version, CUDA build/runtime,
compute capability and cuDNN version were not carried into this documentation session. They are
deliberately **left blank rather than filled in from the historical RTX PRO 6000 / cc 12.0 record**,
because substituting prior values for uncaptured current ones would be a fabrication. The torch build
string in §3 (`2.11.0+cu128`) does pin the CUDA build to **12.8**.

This is the only gap in the acceptance record. It is INFORMATIONAL (§28): the runtime's *behaviour*
was exercised directly by the suites in §7–§10 and the real-model smoke in §14, which is the evidence
that matters for the training gate.

## 3. Dependency Versions

**[CUDA]**

| package | version |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | **4.57.6** (frozen requirement — satisfied) |
| tokenizers | 0.22.2 |
| sentencepiece | 0.2.2 |
| safetensors | 0.8.0 |
| accelerate | 1.14.0 |

The frozen `transformers==4.57.6` pin from `requirements/experiment.txt` is satisfied exactly. torch
is intentionally unpinned and Colab-provided.

## 4. Prepared-Corpus Verification

**[CUDA]** Verified with the repository's own verification code. Stage-6 was **not** rerun and the
corpus was **not** regenerated.

```
chunks.jsonl     2 198 412 593 bytes
                 sha256 5e4c5e0c77e7677e188501723651e0923d072a31a9048a7d04042ff7b290cad6
manifest.json            2 878 bytes
                 sha256 6f33c2aa51b63a4dc68e238594acbec581b2a1f6b0f7be42e002dfb10a02ef62
chunk_membership_digest  250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6

chunks    train 2 621 624   dev 11 443   total 2 633 067
parents   train 1 113 224   dev  5 000   total 1 118 224
overflow_count                        0
parents_spanning_both_partitions      0
```

**[VERIFIED HERE]** Internal arithmetic is exact: `2 621 624 + 11 443 = 2 633 067` and
`1 113 224 + 5 000 = 1 118 224`. `overflow_count = 0` is consistent with the locked
`MAX_LENGTH = 256` / `on_overflow = FAIL` policy after correct pre-chunking, and
`parents_spanning_both_partitions = 0` confirms the document-level split preceded chunking, so no
parent document leaks across train/dev.

## 5. Inventory Verification

**[CUDA]** Verified through the real scientific preflight:

```
bytes   116 290
sha256  78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2
```

**[VERIFIED HERE]** This matches the tracked pin in `configs/linguistics/vietnamese_syllables.yaml`
(`source_revision 135a4d9716e49a981624474156d6f247b9b46f6a`, same sha256), which is inside the
clean-tree guarded paths. The inventory identity also forms part of `CampaignIdentity`.

## 6. Frozen Configuration Verification

**[VERIFIED HERE]** All nineteen locked values re-read from `unmark/stage1/protocol.py` at the
accepted HEAD — **zero mismatches**:

```
vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6 · fp32 · hidden 768
adapter trainable 3 551 232 · MAX_LENGTH 256 · batch 128 · grad-accum 1
eval/checkpoint every 500 · initial 20 000 · hard stop 40 000 · continuation at most once
PI_STRIP 0.25 · corruption 35422 · validation corruption 19225 · split 51733 · selection 21230
final seeds (36930, 7309, 5993)
LR grid (1e-4, 3e-4, 1e-3) · r grid (0.25, 0.5, 1.0, 2.0, 4.0) · 11 nominal runs
```

No value was changed to make any gate pass.

## 7. Repaired Mid-Continuation CUDA Result

**[CUDA]** `tests/test_stage1_resume_state_machine_torch.py` executed successfully after the Audit
038 repair. The `20500 / 40000` probe now reaches its explicit sentinel after the real restore and
before preparation, forward, backward and `optimizer.step`.

This is the gate that failed the previous attempt (10 passed / 2 failed). That failure was a
**test-design** defect — the fixture passed `tokenizer=None` under the false assumption, recorded in
Audit 035 and corrected in Audit 038, that no case could enter the training loop. The loop was in
fact entered *because* the cap reconstruction works: a mid-continuation checkpoint is not complete,
so production correctly evaluates `20500 < 40000` and continues the 40k leg.

**[VERIFIED HERE]** The committed file contains the repair: 12 tests, the dedicated
`_ContinuationEntered` exception, and `resume_expecting_continuation`, which patches
`prepare_serially` at the exact symbol `train_run` resolves.

## 8. Training-Resume Fixture Result

**[CUDA]** `tests/test_stage1_training_resume.py` executed successfully on the torch environment —
its torch-dependent cases ran rather than skipping. This file had been repaired after Audit 036
because it still used the obsolete `{"update", "score"}` validation-point schema; it now runs against
the canonical `ValidationPoint → checkpoint_payload → save/load → ValidationPoint.from_dict`
contract.

## 9. CUDA Resume-Equivalence Result

**[CUDA]** `tests/test_stage1_cuda_resume_equivalence.py` executed successfully. Any optimizer steps
inside that fixture are **TEST-ONLY synthetic** steps on a synthetic model (§23) and are never
counted as scientific training.

## 10. Targeted CUDA Suite

**[CUDA]**

```
376 passed, 1 skipped, 0 failed
```

## 11. The One Original Skip

**[CUDA]**

```
tests/test_stage1_run_independence_runtime.py
  test_initialisation_does_not_initialise_cuda_as_a_side_effect
  SKIPPED: CUDA was already initialised by an earlier test
```

This skip was **not accepted as evidence**. The test asserts that adapter initialisation does not
initialise CUDA as a side effect — a property that is unobservable once an earlier test in the same
process has already initialised CUDA, so the skip is a correct self-protection rather than a pass.

## 12. Isolated Rerun That Discharged the Skip

**[CUDA]** The test was rerun by itself in a **fresh Python process**, where the precondition holds:

```
1 passed, 0 skipped, 0 failed
```

The combined suite's only skip is therefore independently discharged, and the effective targeted
result is **377 executed, 0 failed, 0 unexplained skips**.

## 13. H0 / Freeze / Runtime Evidence from the Targeted Suite

**[CUDA]** The targeted suite included `tests/test_stage1_final_freeze.py`, whose H0 recomputation
is torch-gated and skips in the ML-free environment. It executed here with **0 failures**, so the
locked H0 digests and their hash-group structure were recomputed on the real stack and matched. The
device-contract, run-independence, validation-measurement, torch-contract and name-resolution
suites likewise executed with 0 failures.

**[VERIFIED HERE]** The torch-free half of the H0 machinery — the domain-separated init-seed
derivation — is unchanged at this HEAD: `21230→3203`, `36930→51800`, `7309→45833`, `5993→15758`.

## 14. Real PhoBERT No-Update Smoke

**[CUDA]** The repository's own `smoke` subcommand of `scripts/stage1_runner.py` — a structurally
separate path that constructs no optimizer and calls no `.backward()`. No training command was
invented, and `lr-pilot` / `r-phase1` / `final-main` were not invoked.

```
prepared corpus   /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb
completion dir    /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb
revision          01daacda68afe13d83023d16ec647239e344a1e6
--repository-head 34232651f35132097097796c063bb5d3840f47bd   (assertion)
```

Runner output:

```
UNMARK Stage-1 — stage1-protocol-v1
official UIT-VSFC TEST : SEALED — no argument, no route
scientific inputs      : VERIFIED
prepared corpus        : VERIFIED
loaded                 : 2 621 624 train chunks, 11 443 dev chunks
smoke                  : STAGE1_NO_UPDATE_FORWARD_ONLY
optimizer_constructed  : false
backward_called        : false
parameters_updated     : 0
repository_head        : 34232651f35132097097796c063bb5d3840f47bd
```

The recorded `repository_head` equals the accepted HEAD, and it was **derived**, not supplied: under
the Audit 033 repair `--repository-head` is an assertion that must agree with the executing tree, so
the run also proves the assertion matched and the tracked execution tree was clean.

## 15. Model Contract

**[CUDA]**

```
hidden_size                    768
precision                      fp32
trainable_parameters           3 551 232
trainable_tensors              8
encoder_trainable_parameters   0
encoder_training_mode          false
```

**[VERIFIED HERE]** `trainable_parameters` matches the locked `ADAPTER_TRAINABLE_PARAMETERS` and
`hidden_size` the locked `HIDDEN_SIZE` exactly. `encoder_trainable_parameters = 0` with
`encoder_training_mode = false` is the frozen-encoder contract holding on the **real** PhoBERT, not a
stub.

## 16. Finite Smoke Losses

**[CUDA]**

```
loss                  0.6261955499649048
loss_align            0.3096499741077423
loss_clean            0.31654560565948486
mean_distance_align   0.3096499741077423
mean_distance_clean   0.31654560565948486
smoke batch_size      8
```

All finite; no NaN, no Inf.

**[VERIFIED HERE]** Independent corroboration that this was a genuine fp32 GPU forward:
`lambdas_for_r(1.0) = (1.0, 1.0)`, so the objective reduces to `loss = L_align + L_clean`. Summing
the two reported components gives `0.6261955797672272` against a reported loss of
`0.6261955499649048` — a relative difference of **4.76e-8**, inside fp32 epsilon (1.19e-7) and
exactly what a fp32 accumulation reported through separate conversions produces. A float64
computation would have matched bit-for-bit; the small, correctly-signed discrepancy is evidence the
run was fp32 on device rather than a CPU/float64 stand-in.

`mean_distance_align == loss_align` and `mean_distance_clean == loss_clean` confirm the branch
distances are reported unweighted, consistent with unit lambdas at r = 1.

## 17. Checkpoint / Resume Verdict

**PASS.** Exercised on the real stack by §7, §8 and §9: the real writer → `save_training_checkpoint`
→ `load_training_checkpoint` → real `train_run` restore chain, including `verify_checkpoint`, strict
adapter restore, optimizer restore, optimizer parameter-identity and state-device checks,
`DeterministicSampler.from_state`, and `ValidationPoint.from_dict` — the reader that used to raise
`TypeError` on every real resume.

## 18. 20k → 40k Continuation Verdict

**PASS.** The mid-continuation probe (§7) demonstrates on the real stack that a `20500 / 40000`
checkpoint reconstructs `cap = 40000` from validated persisted state and that production enters the
continuation body under that cap. Cap lowering, impossible states and a third continuation are
refused. This is the blocker that produced silent 40k-work-as-20k-run mislabelling before the repair.

## 19. Multi-Run / Run-Independence Verdict

**PASS.** The A/B/C scenario (A completed, B crashed mid-40k, C never started) executed with isolated
checkpoint namespaces and no history leakage. `tests/test_stage1_run_independence_runtime.py` passed
in full once its CUDA-side-effect test was rerun in a fresh process (§12), confirming adapter
initialisation does not initialise CUDA as a side effect.

## 20. Artifact Handoff Verdict

**PASS.** `tests/test_stage1_artifact_identity.py` ran inside the targeted suite with 0 failures.
Campaign identity is derived from current trusted inputs — actual HEAD, verified corpus digest,
pinned backbone and revision, fp32, verified inventory — never from the artifact under validation,
and the production selection functions recompute the winner. Wrong HEAD, corpus, revision, inventory
and protocol, and missing/duplicate/extra candidates and edited selected values, are all refused.
`lr-pilot`, `r-phase1` and `final-main` were **not** run.

## 21. Repository Provenance Verdict

**PASS.** `tests/test_stage1_repository_provenance.py` ran inside the targeted suite with 0 failures,
and §14 shows the property end to end on the real runner: the smoke recorded
`repository_head = 34232651…`, derived from Git, with the operator assertion agreeing and the tracked
execution tree clean. An omitted assertion yields the actual HEAD rather than `None`; false,
abbreviated and branch-name assertions refuse; Git failure fails closed; untracked audit and runtime
files do not falsely invalidate execution.

## 22. Scientific `optimizer.step` Count

**SCIENTIFIC optimizer.step COUNT DURING ACCEPTANCE = 0.**

The real PhoBERT smoke constructed **no optimizer**, executed **no backward**, and updated **zero
parameters** — reported by the runner itself as `optimizer_constructed: false`,
`backward_called: false`, `parameters_updated: 0`. No `lr-pilot`, `r-phase1` or `final-main` command
was executed. No scientific training step occurred at any point.

## 23. TEST-ONLY Optimizer Characterization

Any optimizer steps executed inside `tests/test_stage1_cuda_resume_equivalence.py` (and any similar
synthetic fixture) are **TEST-ONLY synthetic optimizer steps** on a tiny synthetic model with
synthetic inputs. They touch no scientific adapter, no prepared corpus and no campaign state, and
they are **never** counted as scientific training. The mid-continuation probe of §7 executes none at
all: `AdamW.step` is poisoned in both of its resume helpers.

## 24. Official UIT-VSFC TEST Status

**OFFICIAL UIT-VSFC TEST = SEALED / UNUSED.**

Not opened, inspected, evaluated, mounted through a TEST-specific route, or passed to Stage-1. No
information was derived from it. The runner printed
`official UIT-VSFC TEST: SEALED — no argument, no route`, and the CLI has no flag that can reach it.

## 25. BLOCKER count: **0**
## 26. MAJOR count: **0**
## 27. MINOR count: **0**

No acceptance-specific issue arose. The previous `20500 / 40000` CUDA failure was a test-design
problem, repaired in Audit 038 and committed as `3423265`; its repaired CUDA test now passes.
Pre-existing optional MINOR/INFORMATIONAL items from earlier reviews are deliberately not reopened.

## 28. INFORMATIONAL count: **1**

**039-i1** — The CUDA session's Python version, GPU model, driver, CUDA runtime, compute capability
and cuDNN version were not carried into this documentation session and are recorded as not captured
rather than back-filled from the historical runtime record (§2). The torch build string pins CUDA
12.8. This does not affect the training gate: the runtime was exercised behaviourally by 377
executed tests and a real-model GPU smoke, and the fp32 arithmetic signature in §16 independently
corroborates a genuine device forward. Capture these values at the start of the training run.

## 29. Training Authorization

Stage-1 training is authorized from **exactly** HEAD `34232651f35132097097796c063bb5d3840f47bd` —
the same commit that was CUDA accepted. Audit 039 is deliberately **not committed**, so the accepted
HEAD is unchanged when training begins; the clean-tree scientific guard ignores untracked files under
`docs/audits/`, so its presence cannot block or alter a run.

The authorization covers the frozen 11-run nominal campaign at the values in §6, in the locked order:
`lr-pilot` → `r-phase1` → `final-main`. It does not authorize any change to scientific configuration.

## 30. Final Verdict

Every acceptance condition holds: exact HEAD, verified prepared corpus and inventory, frozen
configuration unchanged, the repaired mid-continuation and training-resume fixtures actually
executed, CUDA resume equivalence passed, the real pinned PhoBERT constructed with exactly 3 551 232
trainable parameters and a frozen encoder, a real prepared-data no-update smoke with finite losses
and zero parameter updates, checkpoint/resume and 20k→40k reconstruction proven on the real stack,
artifact handoff and repository provenance passed, the single skip independently discharged, zero
scientific optimizer steps, and official TEST sealed.

**SMALL AUTHORITATIVE CUDA ACCEPTANCE PASS — EXACT HEAD 34232651f35132097097796c063bb5d3840f47bd IS AUTHORIZED FOR STAGE-1 TRAINING**

---

*End of Audit 039.*
