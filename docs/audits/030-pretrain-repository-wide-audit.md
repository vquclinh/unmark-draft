# Audit 030 — PRE-TRAIN repository-wide audit

| | |
|---|---|
| **Audit id** | 030 |
| **Created (UTC)** | 2026-08-23 |
| **Baseline HEAD** | `aa49785eadcbd67b64be28a5f67d725c79b41bbb` — clean tree, nothing staged |
| **Predecessor** | [029](029-stage1-runner-implementation.md) §AB — the first successful complete real Stage-6 prepare |
| **Type** | **Phase-boundary gate.** Repository-wide review of everything between a finished corpus and a first training run |
| **Executed here** | **Nothing.** No encoder load, no forward pass, no optimizer, no update, no weight download |
| **Revision 1** | 2026-08-23 — the audit as first written (§A-§Q). Findings F1-F6 |
| **Revision 2 — hardening** | 2026-08-23 — **F1, F2 and F3 repaired**, plus three further defects found while wiring F3. See **§R**. F4/F5/F6 unchanged |
| **Revision 3 — addendum** | 2026-08-23 — F3 re-read against the log: it was a **wiring defect**, the cadence and **best + last** semantics were already locked by **D-S1B-004**. The invented D-S1B-014 is **withdrawn**; `best` is now persisted too. Validation-cost concern **measured, not acted on**. See **§S** |
| **Revision 4 — final consistency check** | 2026-08-23 — **BLOCKER found in §S's own tool**: `--validation` loads the real encoder but runs **no forward pass**, so it does not measure validation wall-clock. Tool and audit corrected to say so; the measurement remains outstanding. Token sampling verified **deterministic**. See **§T** |
| **Revision 5 — measurement repair** | 2026-08-23 — the §T blocker is **repaired**: `--validation` now runs the authoritative `validation.evaluate` with real forwards, CUDA-synchronised timing, real GPU peak, clean-reference attribution and parameter-hash proof of zero updates. Profile sampling is partition-aware (**all 11 443 dev**). See **§U** |
| **Revision 6 — first real smoke** | 2026-08-23 — the **first real Colab pre-train smoke** was run at `8f07842`. It passed every corpus gate (byte-exact restore, manifest, `COMPLETE.json`) and stopped fail-closed at `tests/test_stage1_training_resume.py`: **6 failed, 8 passed**, all six dying in setup on `RunProvenance(**mine.to_dict())`. Traced to a **test-construction bug** — `to_dict()` is artifact serialization, not a constructor round trip — plus one real gap: the derived weights were serialized and never validated on read. **No model was loaded.** See **§V** |
| **Revision 7 — second real smoke** | 2026-08-23 — the **second real Colab no-update smoke** ran at `b84b4da`. Corpus gates, and the 29 + 17 + 22 provenance/resume/measurement tests, all **passed on real hardware** — confirming §V. It then failed closed in condition preparation with `EligibilityUnresolved`: the pinned syllable inventory is **deliberately not committed** and a fresh runtime had not provisioned it. Classified **A + C** — the artifact was already locked by D-B3A-001, but the check ran *after* model load and the **blocking D-S1A-008 was unimplemented**. **No model forward, no optimizer, no update.** See **§W** |
| **Revision 8 — third real smoke** | 2026-08-23 — the **third real Colab no-update smoke** ran at `ebbe553`. §W's inventory repair **held completely** (fetch, SHA, shape, `--verify-only`, Drive persistence, preflight before model load), and 31 + 30 + 17 + 22 tests passed on real hardware. Validation then failed closed on `prepare_condition_batch(..., truncation=None, ...)` — `'NoneType' object has no attribute 'check'`. **Measurement-only** (A + D): every production caller passes the authoritative `TRUNCATION`. **PhoBERT was resident but no forward, no optimizer, no update.** See **§X** |
| **Revision 9 — fourth real smoke** | 2026-08-23 — the **fourth real Colab no-update smoke** ran at `6bd6452`. All corpus, inventory and preparation gates passed (14 + 31 + 30 + 17). It stopped at the **test gate**: the §X real-seam tests reached `evaluate` and hit `Expected all tensors to be on the same device`. A **true positive** — the tool moved the encoder to CUDA and left every batch on the CPU, so real PhoBERT would have failed identically. Repaired in the shared layer, not the fixture. **The real validation command was never run.** See **§Y** |
| **Revision 10 — fifth smoke probe** | 2026-08-23 — the fifth smoke reached the **device runtime gate** at `9651610` and **the device contract passed on real CUDA: 7 passed, 0 failed, 0 skipped**. The run stopped only because the orchestrator expected the string `"8 passed"`, a **prose miscount in §Y.6** — the file has always held **7** tests, and all nine claimed semantic assertions are present across them. **No code or test was at fault; none was changed.** See **§Z** |
| **Revision 11 — first full real validation** | 2026-08-23 — at `a5da538` the **full four-condition real validation PASSED**: 11 443 dev chunks, **810 forward passes**, CUDA, fp32, adapter 3 551 232 frozen-encoder split, **0 backward / 0 steps / 0 updates**, parameters hash-identical before and after. The subsequent `stage1_runner.py smoke` then failed at **0.09 s** with `AttributeError: 'Namespace' object has no attribute 'completion_dir'` — a **parser omission** (`smoke` never declared an option its handler reads). Repaired; the validation surface is byte-unchanged. See **§AA** |
| **Revision 12 — no-update smoke CLOSED** | 2026-08-23 — the §AA CLI repair was committed and the **corrected real runner smoke PASSED** at `2363e33`: return code 0, 24.36 s, real encoder frozen (`encoder_trainable_parameters 0`, adapter 3 551 232), prepared corpus verified against its Drive `COMPLETE.json`, real forward `loss 0.6354566812515259`, and `backward_called false` / `optimizer_constructed false` / `parameters_updated 0`. With §AA.1 this closes the **PRE-TRAIN no-update smoke gate**. AA.1 was reused, not rerun. See **§AB** |
| **Revision 13 — final device audit** | 2026-08-23 — the dedicated phase-boundary **training-device audit** at `0588b72`. Confirms training would run **silently end-to-end on CPU** (no placement anywhere in `execute_stage`/`train_run`, recorded in no artifact) and finds a **second, scientific blocker**: the adapter is initialised from an **unseeded** global RNG — `run_seed` drives only data order, so a published seed cannot reproduce its run. **Two decision entries required, neither written.** Audit only. See **§AC** |
| **Revision 14 — cross-candidate leakage** | 2026-08-23 — the positive nominal-run-independence gate found a **scientific defect**: `build_objective` is called **once, outside** the run loop, so every nominal run in a stage shares one `UnmarkEncoder` and candidates 2..N inherit the previous candidate's **trained** adapter. `lr-pilot` would be one trajectory with two LR changes; the three `final-main` seeds would not be independent replicates. **Implementation stopped; nothing implemented.** No campaign has run, so no result is contaminated. See **§AD** |
| **Revision 15 — device/init/independence repair** | 2026-08-23 — **D-S1B-015, D-S1B-016 and D-S1B-017 persisted and implemented** at `3a5368c`. Every nominal run now builds a **fresh adapter** from a domain-separated, CPU-initialised seed (21230→3203, 36930→51800, 7309→45833, 5993→15758); the frozen encoder is the only shared model state. CUDA required and fail-closed under an enforced *and re-asserted* deterministic true-fp32 policy. Checkpoints are adapter-only and strict (schema **v2**). **3 572 passed, 101 skipped.** CUDA half pending a GPU. See **§AE** |
| **Revision 15a — pre-commit correction** | 2026-08-23 — a targeted review of the §AE draft found a **real defect in it**: `torch.manual_seed` seeds *all* devices, so pairing it with `fork_rng(devices=[])` perturbed CUDA RNG without restoring it. Repaired to `torch.default_generator.manual_seed`. Also corrected: **four** init-hash groups `[8,1,1,1]`, not two; **GPU model is now resume-blocking**. And the torch runtime tests are recorded as **IMPLEMENTED, NOT EXECUTED** — no torch exists on this machine. See **§AE** |
| **Revision 16 — fresh-CUDA runtime verification** | 2026-08-24 — the torch/CUDA contracts §AE could only *implement* are now **executed** on a fresh Colab runtime (torch 2.11.0+cu128, CUDA 12.8, transformers 4.57.6, RTX PRO 6000 Blackwell, cc 12.0, cuDNN 91900), bound to implementation commit `3c3489b9`. **Stage-1 1 344 passed / 1 skipped; full 3 754 passed / 1 skipped; 0 failed, 0 errors.** Four H0 hashes and the `[8,1,1,1]` grouping recorded in full. **Two claims reserved** — CUDA resume byte-identity and optimizer-state placement *on CUDA*. See **§AE.11** |
| **Revision 17 — real acceptance + performance blocker** | 2026-08-24 — at `ac20cfb7`, **every real zero-update acceptance gate PASSED** on the real corpus and real PhoBERT: Probe 1, Probe 2, CUDA interrupted-vs-uninterrupted exact equivalence, populated optimizer-state CUDA placement, and the full four-condition validation. `optimizer.step` count still **ZERO**. Those runs produced the first real training-path timing and revealed a **material performance blocker**: the path is **preparation-bound** (79.05 % prepare, 12.60 % GPU), projecting **≈403 h / 16.8 days** for the 11 nominal runs. See **§AF** |
| **Revision 18 — parallel preparation** | 2026-08-24 — the first performance repair §AF called for: **deterministic 8-worker parallel preparation, and nothing else**. Real benchmark: serial 4.605 s/batch → 8-worker 0.666 s (**6.912×**), **all prepared output exactly equal**. Production uses **`spawn`**, never the benchmark's `fork`, because the parent holds CUDA. Tokenizer reuse **rejected** (~5.01 % of preparation); classifier cache **deferred** (1.059× real). 17 new tests executed locally. See **§AG** |
| **Decides** | Whether the *next* step — a bounded real **no-update model smoke** — may proceed. **It does not authorise training** |

---

## A. VERDICT

**IMPLEMENTED — POST-IMPLEMENTATION CUDA/SPAWN PERFORMANCE VERIFICATION PENDING**

> **§AG is the current verdict.** §AF's performance blocker has its first repair:
> **deterministic 8-worker parallel preparation, and deliberately nothing else.**
> The real benchmark measured serial **4.605 s/batch → 8-worker 0.666 s
> (6.912×)** with **every prepared output exactly equal**, and two candidates were
> closed out by evidence rather than taste — tokenizer reuse **rejected** (the real
> tokenizer is only ~5.01 % of preparation, not worth weakening an independently
> computed base-invariance check) and the classifier cache **deferred** (1.059× on
> a real batch).
>
> **Production uses `spawn`, never the benchmark's `fork`** — the parent holds a
> CUDA context by then. So the benchmark establishes parallelisability, order and
> exact equality, **not production throughput**, and no speedup is claimed for the
> implementation. §AF.4's caveat is closed: a persistent CUDA resume-equivalence
> test now exists.
>
> **A post-implementation CUDA/spawn benchmark on the authoritative GPU is
> mandatory**, along with the configuration freeze, the final review and human
> approval. §T–§AF are preserved verbatim, with §AF.6's over-strong wording
> corrected in place. **Training is not authorised.**

~~**PRE-TRAIN RUNTIME ACCEPTANCE PASS — PERFORMANCE BLOCKER UNDER REVIEW**~~ **— superseded by §AG**

> **§AF was the previous verdict.** At `ac20cfb786ca770a7296339d48263ff8e09acf66`
> every real zero-update acceptance gate **passed** against the real prepared
> corpus and real PhoBERT — Probe 1, Probe 2, **CUDA interrupted-vs-uninterrupted
> exact equivalence**, populated optimizer-state CUDA placement, and the full
> four-condition validation (810 forwards, parameter hashes identical). The
> scientific **`optimizer.step` count is still ZERO** and official UIT-VSFC TEST
> remains sealed. §AE.11.8's two reserved claims are **closed** — though as a
> one-off fixture, **not** persistent regression coverage (§AF.4).
>
> **A new, material blocker replaces them.** The training path is
> **preparation-bound**: 79.05 % of each step is `prepare_example`, 87.40 % is
> CPU-side, and only 12.60 % is GPU. That projects a **lower bound of ≈36.67 h per
> 20k run and ≈403 h / 16.8 days for the 11 nominal runs**, before
> `optimizer.step`, checkpoint I/O or any continuation. Root cause: eligibility
> classification recurses into a **full `decompose` per syllable span** — ≈301
> `decompose` and ≈602 `canon` calls per example — and 49.7 % of tokenize calls
> recompute a result the code already proves identical.
>
> **Nothing was optimised in this task**, and no speedup of the real path is
> claimed. §AF also **corrects §AC.15 item 3**: Adam's scalar `step` is
> legitimately on CPU and production must not be changed to move it.
>
> §T–§AE are preserved verbatim. **Training is not authorised.**

~~**FRESH-CUDA RUNTIME VERIFICATION PASS — READY FOR HUMAN REVIEW**~~ **— superseded by §AF**

> **§AE.11 was the previous verdict.** The torch and CUDA contracts that §AE could
> only *implement* have now **executed on real hardware** — a fresh Colab runtime
> at torch 2.11.0+cu128 / CUDA 12.8 / transformers 4.57.6 on an RTX PRO 6000
> Blackwell — bound to implementation commit `3c3489b9`. Stage-1 **1 344 passed /
> 1 skipped**, full suite **3 754 passed / 1 skipped**, **0 failed, 0 errors**; the
> single skip is order-sensitive by design and passed alone in a fresh subprocess.
> The four H0 hashes, the `[8, 1, 1, 1]` grouping, the cuBLAS-before-CUDA ordering
> and the 13-field resume-blocking set are all recorded in full.
>
> **This closes the implementation/runtime-verification gate ONLY. It is not
> "Stage-1 training ready."** Two claims are deliberately **reserved** — CUDA
> interrupted-vs-uninterrupted byte identity, and optimizer-state placement *on
> CUDA* — because no committed test evidences them (AE.11.8). And nothing here
> touched the real prepared corpus: the two zero-update acceptance probes, the
> performance measurement, the **FINAL CONFIGURATION FREEZE**, the final
> repository-wide review and **human approval** all remain.
>
> §T–§AD and §AE.1–§AE.10 are preserved verbatim. **Training remains
> unauthorised.**

~~**PRE-TRAIN IMPLEMENTATION RUNTIME VERIFICATION INCOMPLETE — DO NOT COMMIT YET**~~ **— the runtime gate is closed by §AE.11**

> **The superseded verdict, after a pre-commit correction pass that found a
> real defect in §AE's own first draft:** `torch.manual_seed` seeds *all* devices,
> so pairing it with `fork_rng(devices=[])` perturbed CUDA RNG and never restored
> it — the opposite of what D-S1B-016 requires. Repaired to
> `torch.default_generator.manual_seed`. Two further corrections: there are
> **four** init-hash groups with multiplicities **[8, 1, 1, 1]**, not two; and
> **GPU model name is now resume-blocking**, conservatively.
>
> **The blocker on committing is evidence, not code.** No torch exists on this
> machine, so the torch runtime tests are **IMPLEMENTED, NOT EXECUTED** — AE.9.1
> lists every contract that status covers. They must run under pinned torch before
> commit; the CUDA-gated half must run on the GPU.
>
> §T–§AD are preserved verbatim. **Training remains unauthorised.**

~~**PRE-TRAIN IMPLEMENTATION PASS — FRESH CUDA ZERO-UPDATE PROBES REQUIRED**~~ **— superseded by the correction pass**

> **The superseded §AE verdict claimed:** the three decisions §AC and §AD proposed are
> **persisted and implemented**: scientific training now requires CUDA and fails
> closed under a deterministic, true-fp32 policy that is enforced *and*
> re-asserted; every nominal run builds a **fresh adapter** initialised on CPU
> from a domain-separated seed, with the frozen encoder the only shared model
> state; checkpoints are adapter-only and strictly restored.
>
> **Two things are deliberately not claimed.** The CUDA-gated tests — including
> **CUDA resume byte-identity** — could not run in this ML-free venv, so that
> claim stays **scoped to CPU** until a GPU executes them. And no fresh-runtime
> probe has run.
>
> §T–§AD are preserved verbatim. **Training remains unauthorised**: both
> zero-update probes, the performance measurement, the FINAL CONFIGURATION
> FREEZE, the final repository-wide review and human approval are all still
> outstanding.

~~**PRE-TRAIN CROSS-CANDIDATE LEAKAGE RECORDED — THREE DECISIONS AWAIT IMPLEMENTATION**~~ **— superseded by §AE**

> **§AD was the previous verdict, and it records the most serious finding in this
> audit.** `build_objective` is called **once, before** the nominal-run loop, and
> `Stage1Objective` stores the encoder **by reference**, so all runs in a stage
> command share one adapter and the optimizer mutates it in place. Candidates
> 2..N therefore begin from the previous candidate's **trained** weights:
> `lr-pilot` would be a single trajectory with two LR changes, and the three
> `final-main` seeds would not be independent replicates at all.
>
> This is **distinct from §AC.7** — runs 2..N get **no** fresh initialisation,
> not merely an unseeded one. **No scientific campaign has ever run, so no result
> is contaminated**, and the single-run validation and smoke passes (§AA, §AB)
> stand: they could not have contradicted this.
>
> **Three decisions now await implementation — D-S1B-015 (CUDA execution and
> numerics), D-S1B-016 (deterministic CPU-first adapter init), and D-S1B-017
> (nominal-run independence) — and none is written.** §T–§AC are preserved
> verbatim. **Training remains forbidden.**

~~**PRE-TRAIN DEVICE CONTRACT AUDIT FOUND ADDITIONAL BLOCKERS — DO NOT IMPLEMENT YET**~~ **— superseded by §AD**

> **§AC was the previous verdict.** The no-update smoke gate closed in §AB **stands**
> — nothing here reopens it. The dedicated device audit then confirmed the known
> blocker (training would run **silently on CPU**, recorded in no artifact) and
> found a **second, scientific one**: the adapter is initialised from an
> **unseeded** global RNG, so `run_seed` identifies a run it does not determine
> (§11 requires publishable per-seed reproducibility).
>
> **Two decision entries are required — D-S1B-015 (device) and D-S1B-016
> (initialisation seed) — and this audit wrote neither**, because the second is a
> genuine scientific choice. Five further findings are classified in §AC.17, and
> two zero-update fresh-runtime probes are specified in §AC.15.
>
> §T–§AB are preserved verbatim. **Training remains unauthorised.**

~~**PRE-TRAIN NO-UPDATE SMOKE GATE PASS — TRAINING DEVICE CONTRACT REMAINS**~~ **— superseded by §AC**

> **§AB was the previous verdict. The PRE-TRAIN no-update smoke gate is CLOSED.**
> The corrected real runner smoke passed at `2363e335` — real encoder frozen, real
> prepared corpus verified against its Drive `COMPLETE.json`, a real forward, and
> `backward_called false` / `optimizer_constructed false` / `parameters_updated 0`
> — which together with §AA.1's full four-condition validation (810 forwards on
> CUDA, parameters hash-identical before and after) is everything §N asked for.
>
> **This does NOT authorise training.** One known implementation blocker remains,
> and it is the only one Audit 030 has identified: **`execute_stage` performs no
> accelerator placement, so the training-device operational contract is unchosen**
> (§Y.3). A dedicated review must close it. Note that the runner smoke reports no
> device of its own — the CUDA evidence is §AA.1 and §Z, not §AB.
>
> §T–§AA are preserved verbatim.

~~**PRE-TRAIN RUNNER-SMOKE CLI REPAIR PASS — READY TO COMMIT AND RERUN SMOKE ONLY**~~ **— superseded by §AB**

> **§AA was the previous verdict, and it carries this audit's largest positive
> result: the first full four-condition real validation PASSED** — 11 443 dev
> chunks, 810 forward passes on real PhoBERT under CUDA/fp32, with the no-update
> boundary held by parameter-hash equality before and after (0 backward, 0
> optimizer steps, 0 updates). Every repair from §U onward is confirmed against
> the real encoder and the real corpus in that one run.
>
> Afterwards, `stage1_runner.py smoke` failed at 0.09 s: the `smoke` subparser
> never declared `--completion-dir`, which `run_smoke` reads. A **parser
> omission** — the handler and every other consumer were already correct. Repaired
> by declaring the pair once for training and smoke alike; `getattr` was refused
> because it would have weakened the F1 COMPLETE gate. **The validation's
> executable surface is byte-unchanged, so no re-run is required.**
>
> **The runner smoke has still never completed**, and training-device placement
> remains a separate open item (§Y.3). §T–§Z are preserved unchanged.

~~**PRE-TRAIN DEVICE-CONTRACT REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**~~ **— superseded by §AA**

> **§Y was the previous verdict.** The fourth real smoke passed every corpus,
> inventory and preparation gate and then stopped at the validation-measurement
> **test gate** with a cross-device error. That failure was a **true positive**:
> the injected fixture mirrors the production objective exactly — neither moves
> its inputs — so the measurement tool, which put the encoder on CUDA and left
> every batch on the CPU, would have failed identically against real PhoBERT.
> Repaired in the **shared layer**: `evaluate`, `train_run` and `smoke_check` now
> all place the batch on the model's device, derived from the module. No fixture
> was changed, the CPU path is a no-op, and `objective.py` is byte-unchanged.
> §T–§X are preserved unchanged.
>
> **One open operational item** is recorded, not decided: `execute_stage` performs
> no device placement at all, so the *training* device remains unchosen.
>
> **§Z (Revision 10) corrects one number in §Y.6** — the runtime device file holds
> **7** tests, not 8 — and records that the fifth smoke ran those seven on **real
> CUDA hardware: 7 passed, 0 failed, 0 skipped**. The device repair below is
> therefore confirmed in production conditions. No code or test changed.
>
> **The real no-update smoke has NOT passed.** Neither the fourth nor the fifth
> smoke ran `--validation --require-cuda` to completion: no forward, no optimizer,
> no update has occurred.

~~**PRE-TRAIN TRUNCATION-WIRING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**~~ **— superseded by §Y**

> **§X was the previous verdict.** The third real smoke confirmed §W's inventory
> repair end to end on real hardware, then failed closed in condition preparation:
> the measurement tool passed `truncation=None`, which is not a valid
> `TruncationPolicy`. **Production training, validation and the smoke were never
> affected** — `truncation=None` existed at exactly one site in the repository,
> introduced by the §U measurement repair. The locked contract
> (`MAX_LENGTH = 256`, no truncation, `ON_OVERFLOW = FAIL`) is unchanged and was
> merely restored at that call site.
>
> **The more important finding is why 22 measurement tests passed anyway**: nothing
> ever executed `validation_timing` or `prepare_condition_batch`, and the runtime
> fixture replaced the whole preparation stage with hand-built integers. The real
> seam is now exercised end to end. §T–§W are preserved unchanged.
>
> **The real no-update smoke has NOT passed.** Validation never completed: no
> forward, no optimizer, no update has yet occurred.

~~**PRE-TRAIN INVENTORY PROVISIONING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**~~ **— superseded by §X**

> **§W was the previous verdict.** The second real smoke confirmed §V on real
> hardware and then failed closed before completing validation: the pinned
> Vietnamese syllable inventory is deliberately **not committed** (no upstream
> license statement) and a fresh runtime had not provisioned it. That failure was
> the **designed** contract working. The real defects it exposed were that the
> check ran **after** the encoder was loaded, and that **D-S1A-008** — a decision
> whose status line reads "BLOCKING for scientific Stage-1 training and the
> PRE-TRAIN audit" — had **never been implemented**, so no run artifact could name
> the inventory it used. Both are repaired; §T, §U and §V are preserved unchanged.
>
> **§W also corrects this audit's own earlier "No UNRESOLVED MISMATCH"**, which
> was true of the code and too strong as a pre-train readiness claim.
>
> **The real no-update smoke has NOT passed.** Validation never completed: no
> PhoBERT forward, no optimizer, no update has yet occurred.

~~**PRE-TRAIN REAL-SMOKE BLOCKER REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**~~ **— superseded by §W**

> **§V was the previous verdict.** The first real Colab smoke passed every corpus
> gate and then stopped fail-closed at the training-resume gate, before any
> model was loaded. The cause was a **test-construction bug** — six foreign-run
> cases died in setup and had never demonstrated rejection — together with one
> real gap it exposed: the derived `lambda_align`/`lambda_clean` were written
> into every checkpoint and never validated on read. Both are repaired; §T and
> §U are preserved above unchanged.
>
> **The real no-update smoke has NOT passed.** No PhoBERT load, no forward, no
> validation, no optimizer, no update has yet occurred.

~~**PRE-TRAIN HARDENING FINAL PASS — READY TO COMMIT AND RUN REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §V**

> **§U was the previous verdict.** §T's blocker — `--validation` loading the real
> encoder and running **no forward pass** — is repaired: the tool now
> instruments the authoritative `validation.evaluate`, and a report cannot claim
> validation without real forwards, four conditions, and byte-identical
> parameters. §T is preserved above as the record of the defect. F1/F2/F3 are
> unaffected and still hold.
>
> **A PASS does not authorise training.**

> **§A-§Q are the audit as first written and are preserved unrevised.** F1, F2
> and F3 have since been repaired; **§R** records exactly what changed and what
> evidence now exists. The original verdict below stands as the audit's finding
> at the time.

**REVISION 1 (original verdict): PRE-TRAIN REPOSITORY-WIDE AUDIT PASS — READY FOR REAL NO-UPDATE MODEL SMOKE**

The corpus is complete and verified, the scientific configuration is single-sourced
and unchanged, official TEST is unreachable by any route I could construct, and
the corruption/validation/selection mathematics are deterministic and
seed-separated.

**Four findings.** None is a scientific mismatch, a leakage path, or a
corpus-integrity defect. Two must be closed before the *first* training run, two
before *long* runs. All four are stated below with their classification, and none
was silently fixed.

**A PASS here does not authorise training.** The next step is the bounded
no-update smoke in §N. Training remains unauthorised until that smoke is run and
reviewed.

---

## B. FINDINGS, CLASSIFIED

| # | Finding | Classification |
|---|---|---|
| **F1** | The training consumer never verifies `chunks.jsonl` against the manifest digest or `COMPLETE.json`; it records a digest it did not check | **MUST FIX BEFORE FIRST TRAIN** |
| **F2** | `PI_STRIP` is two independent literals: `contracts.PI_STRIP` governs corruption, `protocol.PI_STRIP` is what the manifest records | **MUST FIX BEFORE FIRST TRAIN** |
| **F3** | Training checkpointing is implemented but **never invoked**; `execute_stage` hard-codes `resume=None` and nothing calls `checkpoint_payload` | **SHOULD FIX BEFORE LONG RUNS** |
| **F4** | `load_prepared_chunks` materialises the whole corpus (~2.73 GB of Python objects) and discards every field except `chunk_id` and `text` | **MUST VERIFY IN REAL NO-UPDATE SMOKE** |
| **F5** | Stage-6 transform counters are structurally blind under `--prepare-workers > 1` | **NON-BLOCKING (observability)** |
| **F6** | Compiled proposal PDF is stale | **DOCUMENTATION ONLY** |

**No BLOCKER was found.** Each finding below is traced, not asserted.

---

## C. F5 first — the zero counters were NOT a bypassed length contract

Task 4 required this to be resolved before anything else, because if the
authoritative length contract had been bypassed the audit stops. **It was not.**

The full run printed:

```
length queries 0, BPE run evaluations 0, run-cache hits 0,
evictions 0, incremental appends 0, full fallbacks 0,
authoritative verifications 2
```

Reproduced locally: a **freshly constructed, never used** pair of length
functions reports **exactly** that signature.

| Counter | Fresh pair | Real run |
|---|---|---|
| `authoritative_queries` | **2** | **2** |
| `length_queries`, `bpe_run_evaluations`, `run_cache_hits`, `incremental_appends`, `full_fallbacks` | **0** | **0** |

`authoritative_queries` is 2 because `build_length_functions` creates **two**
`RunLengthComposer`s (reference and base) and each derives its special-token
count with one `authoritative_length("")` call in `__init__`.

**Why the main process's counters are blind.** `scripts/stage1_runner.py:191`
builds `transforms` in the main process and prints its counters at line 286. It
passes them at line 255 as `serial_length_functions`, which
`parallel.ordered_document_chunks` uses **only when `workers <= 1`**
(`parallel.py:146`). At `--prepare-workers 16`, every worker builds **its own**
length functions in `_initialise_worker` (`parallel.py:69`) — including its own
fail-closed 256-query authoritative verifier.

**So the contract ran 16 times over, in 16 processes, and the main process could
not see any of it.** Independent confirmation that it held: the run completed
with **no `Stage1ContractViolation`**, and the manifest reports
`overflow_count 0` — every one of 2 633 067 chunks was independently
length-checked inside `chunk_document` itself, which does not depend on these
counters at all.

**Classification: observability issue, science correct.** Pinned by
`test_a_freshly_built_length_pair_reports_exactly_two_authoritative_queries` and
`test_each_worker_builds_its_own_length_functions`, so this reading can be
re-derived and a refactor that shares main's functions with workers is caught.

---

## D. Task 1 — proposal → implementation traceability

Proposal reread end to end (758 lines), against `docs/spec/decisions.md`
(4 496 lines) and Audits 028/029.

| Requirement (proposal) | Implementation | Class |
|---|---|---|
| §4.2 `dec(x) = (b(x), τ(x), λ(x))`; NFD, enumerated Vietnamese marks, **recomposition of the base** | `orthography/decompose.py` | **EXACT MATCH** (restored by D-S1B-013; §AA) |
| §4.2 `rec(dec(x)) = canon(x)` | `recompose`, tested across 7 script families | **EXACT MATCH** |
| §4.2 nucleus-based tone placement (`hòa` → `hoà`) | `orthography/placement.py`, D-001 | **EXACT MATCH** |
| §4.3 candidate = alphabetic span matching the inventory after stripping; pure function of the stripped form | `linguistics/classify.py` | **EXACT MATCH** |
| §5.x split **before** chunking; no truncation; preserve text | `stage1/chunking.py`; manifest asserts `split_before_chunk`, `truncation=false` | **EXACT MATCH** |
| §5.x protected unit must not be split | `source_letter_runs` + `protects_a_vietnamese_candidate` | **IMPLEMENTATION CLARIFICATION** — narrowed from "any Unicode alphabetic" to Latin-script (D-S1B-013). Recorded, not silent |
| §4.4 `T(b(x))` defines the grid; no post-strip segmenter | `lengths.build_length_functions` | **EXACT MATCH** (D-B3B1A-001) |
| §4.4 every non-Vietnamese subword carries `N/A` in both channels | `Eligibility.NOT_APPLICABLE` | **EXACT MATCH** |
| §6 corruption `p ~ U(0,1)`, redraw per visit | `contracts.CorruptionRatePolicy` | **EXACT MATCH** |
| §6 `pi_strip` scope mixture | `contracts.scope_for`, D-S1B-003 | **EXACT MATCH** — but see **F2** |
| §5 objective: cosine, pooled, attention-masked mean over non-special | `stage1/objective.py` | **EXACT MATCH** |
| §5 frozen encoder, adapter only | `verify_model_contract`, `build_optimizer` | **EXACT MATCH — NOT YET EXECUTED** |
| §6 11-run plan, budget 20k → 40k | `stage1/selection.py` | **EXACT MATCH — NOT YET EXECUTED** |
| Official TEST sealed | `OFFICIAL_TEST_ACCESSIBLE = False`, no flag, screen refuses | **EXACT MATCH** |
| Chunking safe-cut **coordinate system** | source coordinates (D-S1B-012) | **IMPLEMENTATION CLARIFICATION** — proposal is silent on coordinates, correctly |

**No UNRESOLVED MISMATCH.** Everything in the training half is **NOT YET
EXECUTED**, which is precisely why the no-update smoke exists.

**Drift check.** The two clarifications (D-S1B-012, D-S1B-013) both moved the
*implementation* toward the specification, not away from it: §4.2 already said
"recomposition of the base", and the implementation had drifted from it. The
editable proposal needs no change.

**F6 — stale compiled PDF.** The editable `unmark-proposal.md` is current; only
the compiled PDF is stale. It is a **release artifact, not a scientific input**:
no code reads it, and no run binds it. **It must be regenerated before external
release, not before training.** Not regenerated here — doing so cannot affect
science, and doing so unnecessarily would add an unreviewed binary diff to a
phase-boundary commit.

---

## E. Task 2 — scientific constants, re-derived from the repository

Enumerated by importing `unmark.stage1.protocol`, not from memory:

| Constant | Value |
|---|---|
| corpus / revision | `undertheseanlp/UVW-2026` / `a0a79294e4568137e25828bb3f2a4cde8546e1fb` |
| shard order | `('train.parquet', 'validation.parquet', 'test.parquet')` |
| `MAX_LENGTH` | **256** |
| `SPLIT_SEED` / `DEV_DOCUMENTS` | **51733** / **5000** |
| encoder / revision | `vinai/phobert-base` / `01daacda68afe13d83023d16ec647239e344a1e6` |
| `PI_STRIP` | **0.25** |
| `CORRUPTION_SEED` | **35422** |
| rate distribution / redraw | `uniform_0_1_per_example` / `per_visit` |
| `VALIDATION_CONDITIONS` | `('FULL', 'P50', 'P100', 'STRIP_ALL')` |
| `VALIDATION_CORRUPTION_SEED` | **19225** |
| `BATCH_SIZE` / `EVAL_EVERY_UPDATES` | **128** / **500** |
| budget / continuation | **20 000** → one continuation → **40 000** → `BUDGET_LIMITED` |
| `LR_PILOT_GRID` | `(1e-4, 3e-4, 1e-3)` |
| `R_PHASE1_GRID` | `(0.25, 0.5, 1.0, 2.0, 4.0)` |
| `TOTAL_NOMINAL_RUNS` | **11** |
| adapter trainable parameters | **3 551 232** |
| `PRECISION` / `ON_OVERFLOW` | `fp32` / `FAIL` |
| selection score | `max over VALIDATION_CONDITIONS of mean cosine distance to h(x)` |
| `RAW_BASE_POLICY` | `RAW_BASE` |
| `OFFICIAL_TEST_ACCESSIBLE` | **False** |

**No constant was changed by this audit.** All match Audit 028's lock.

### E.1 F2 — `PI_STRIP` is two independent literals

Searched for competing definitions of every constant. One real duplicate inside
Stage-1:

| Site | Role |
|---|---|
| `unmark/stage1/contracts.py:375` | **governs the science** — `CorruptionRatePolicy.pi_strip` defaults to it (`contracts.py:427`) and `is_locked_mixture` compares against it |
| `unmark/stage1/protocol.py:89` | **recorded in the manifest** (`protocol.py:270`) and printed by the runner |

`contracts.py` does **not** import `protocol.py`; these are two separately typed
`0.25` literals. They agree today, so the science is correct and the manifest is
honest. The risk is a future single-sided edit: the corruption engine would draw
against one value while every artifact recorded the other, and **nothing in the
repository would notice**.

**Classification: MUST FIX BEFORE FIRST TRAIN.** The fix is one line — make one
import the other — but it is a change to a locked scientific constant's
definition site, so it belongs to the researcher, not to an audit. Pinned
meanwhile by `test_pi_strip_has_one_value_even_though_it_has_two_definitions`,
which fails the moment they diverge.

**Other apparent duplicates are separate namespaces, not conflicts.**
`MAX_LENGTH`, `BATCH_SIZE` and `ENCODER_REVISION` also appear under
`unmark/evaluation/preg1_*`, which is the completed pre-G1 diagnostic phase with
its own protocol. Values agree; the namespaces are deliberately distinct.

---

## F. Task 3 — leakage

**Official UIT-VSFC TEST is unreachable.** Adversarially checked:

| Route | Result |
|---|---|
| CLI flags | **12 total** in `stage1_runner.py`; none contains "test". The only UIT-VSFC flags are `--uitvsfc-derived-train` and `--uitvsfc-official-validation`, both already-opened sources, and both reachable **only** from `prepare-corpus` |
| contamination screen | `screen_contamination` raises `CorpusContractViolation` on any key outside `CONTAMINATION_SCREEN_INPUTS`; asserted by test with `uitvsfc_official_test` |
| filesystem discovery | The **only** glob in `unmark/stage1/` is `directory.glob("*.tmp")` for orphan-temp cleanup in `checkpoint.py:199`. No `rglob`, `iterdir`, `listdir` or `os.walk` anywhere in Stage-1 |
| manifest | `require_compatible` refuses any manifest not recording `official_test_used=false` |
| smoke / debug | `smoke_check` takes `--prepared-corpus` only; `stage1_blocker_probe.py` takes an explicit shard and row |
| every textual occurrence | All 20 matches for "official test" across `unmark/`, `scripts/`, `configs/` are **seal assertions or refusals**, never a read path |

**Stage-1 train/dev isolation.** The split is document-level and happens
**before** chunking: `chunk_document` *takes* a partition and copies it onto
every chunk, and no code path assigns one. `chunk_corpus` refuses a document
without a partition, and `verify_no_parent_spans_partitions` asserts the
invariant independently. The real run confirms it end to end:
**`parents_spanning_both_partitions = 0`** over 1 118 224 parents.

**Contamination semantics** match the proposal: exact/canonical only,
`sha256(canon(x))`, no fuzzy matching, decided only at tier 3, with the tier-1/2
prefilters proven necessary conditions (Audit 029 §Q). Real result: **0 excluded
of 1 118 224**, 0 candidates.

**Selection leakage.** Stage-1 selection reads only the held-out **unlabeled** dev
signal. No downstream score enters it; `validation.py` binds
`VALIDATION_CORRUPTION_SEED` and records
`training_seed_affects_validation_corruption: False`.

**Assessment: no leakage path found.**

---

## G. Task 4 — prepared-corpus integrity

Traced source → split → slice → chunk → both pathways → serialisation →
manifest/checkpoint/COMPLETE → loader.

| Property | Evidence |
|---|---|
| no normalisation into chunks | `chunking.py` AST-asserted to call no `canon`/`normalize`/`replace`/`sub`; every chunk is `content[start:end]` |
| no truncation | `TRUNCATION_OFFERED = False`, `ON_OVERFLOW = FAIL`, manifest asserts `truncation=false`; real `overflow_count = 0` |
| no dropped documents | `parent_documents_total = 1 118 224` |
| exact source tiling | `verify_tiles_source`; `PreparedChunk.__post_init__` refuses any chunk whose text length disagrees with its range |
| stable ids | `{document_id}#{chunk_index}`, re-derived and asserted |
| deterministic ordering | JSONL line order is document order; the ordered collector emits strictly by index; byte-identical across 1/2/4/8 workers |
| both pathways ≤ 256 | `overflow_count = 0` over 2 633 067 chunks |
| partition membership | train 2 621 624 + dev 11 443 = 2 633 067 ✓; parents 1 113 224 + 5 000 = 1 118 224 ✓ |
| digest | `chunk_membership_digest` sorts its keys, so it is order-independent; external merge sort proven to produce the identical digest |
| COMPLETE | written **last**, binds artifact hashes; `ALREADY_COMPLETE` re-verified every artifact on a **cold second process** |

**Assessment: the prepared corpus is internally consistent and durably verified.**

---

## H. Task 5 — orthography / RAW_BASE

Re-audited after both repairs. NFC, NFD, tone marks, letter-forming marks, `đ`,
non-canonical placement, combining sequences, non-Vietnamese scripts,
Latin-but-not-Vietnamese, and OOV-valid Vietnamese are covered by 93 dedicated
tests plus 1 098 orthography tests.

### H.1 The `señor` → `senor` collision — classified, not fixed

`U+0303` (tilde) **is** the Vietnamese *ngã* tone mark; `U+0301` is *sắc*;
`U+0300` is *huyền*. So `señor` → `senor`, `café` → `cafe`, `règle` → `regle`.

**Is this consistent enough with the proposal to permit training? Yes, and the
proposal already decided it.** §4.3 states the candidate rule is "orthographic
and structural, never semantic", and `linguistics/classify.py` forbids language
identification, corpus frequency, context and dictionaries **by design**, with
the accepted error mode written out: "an English word whose letters happen to
form a valid stripped Vietnamese syllable is classified as Vietnamese."

The collision is the *same* accepted trade-off one layer down: without language
ID, a combining mark cannot be attributed to a language. Removing it would
require exactly the mechanism the proposal forbids.

**Crucially, it does not threaten the experiment's validity**, because the
property the design rests on is **invariance, not accuracy**: `b(x) = b(x̃)` for
every corruption rate. A Spanish `ñ` is stripped identically in the clean and
corrupted branches, so the base grid stays invariant and corruption remains a
purely channel-level phenomenon. The measured
`base_invariance_violations = 0` over the real corpus is the direct evidence.

**Classification: NON-BLOCKING, documented limitation.** It is now pinned by test
so a future change to it is deliberate. It is **not** a scientific ambiguity that
could materially alter the experiment, so PRE-TRAIN is not blocked on it.

---

## I. Tasks 6, 13 — the training input loader

This is where the audit found the most.

### I.1 F1 — the consumer never verifies the corpus it loads

`load_manifest` → `require_compatible` **is** called by all three training
commands and validates schema, protocol version, `official_test_used=false`,
corpus revision, `max_length`, `split_before_chunk`,
`chunks_inherit_parent_partition`, `truncation=false`, tokenizer revision and
split seed. That is a real gate and it works.

**But it validates the manifest's *declarations*, not the data.**
`execute.load_prepared_chunks` opens `chunks.jsonl` and reads it whole. It does
**not** recompute `chunk_membership_digest`, and it **never consults
`COMPLETE.json`** — the very artifact Stage 6 built to bind file hashes, and
which `ALREADY_COMPLETE` proved works.

`execute_stage:127` then reads `manifest["counts"]["chunk_membership_digest"]`
and writes it into `RunProvenance`. So a run **records a digest it never
checked**. If `chunks.jsonl` were truncated, swapped, or copied from a different
prepare, training would proceed and every artifact would carry provenance
describing data it was not trained on.

**Classification: MUST FIX BEFORE FIRST TRAIN.** Provenance that can silently
misdescribe its own training data is exactly the defect class this gate exists
to catch. The fix is small — have the consumer call `read_completion` (already
written and proven) and/or recompute the membership digest over what it loaded —
but it is a substantive change to the training path and belongs in a reviewed
commit, not folded silently into an audit.

### I.2 F4 — the loader materialises the whole corpus

`load_prepared_chunks` builds two `dict[chunk_id, text]` over all 2 633 067
chunks. Measured cost: **~1 038 B/entry → ~2.73 GB of Python objects**, on top of
parse churn, for a 2.15 GiB payload.

This is the same non-streaming pattern Audit 029 §U removed from the *writer*
(where it would have been ~29.9 GB). It was never applied to the *reader*. On the
observed 176.88 GiB runtime it is survivable, which is why this is not a blocker.

It also **discards** every field except `chunk_id` and `text` — including the
recorded `reference_length` / `base_length`, `source_start/end`, `document_id`
and `source_shard`. Nothing currently needs them (corruption keys on `chunk_id`),
but it means the loader cannot cross-check the lengths the prepare recorded.

**Classification: MUST VERIFY IN REAL NO-UPDATE SMOKE** — the smoke opens the
real 2.2 GB corpus and is the natural, cheap place to observe both peak RSS and
load time before an 11-run plan depends on them.

---

## J. Tasks 7, 8 — corruption and validation

**Corruption is exemplary and needs nothing.** `_unit_draw` is a pure
`blake2b(namespace | schema_version | seed | sample_id | visit)` digest scaled to
`[0, 1)`. Consequences, all structural rather than tested-into-existence:

* **no global RNG** — nothing to seed, nothing to leak between train and dev;
* **no worker-count dependence** — the draw is a function of the example, not of
  scheduling;
* **no ordering dependence** — same reason;
* **resume-deterministic by construction** — replaying a `(sample_id, visit)`
  reproduces the same `p` with no state to restore;
* `p ~ U(0,1)`, redraw per `visit`, `pi_strip` scope mixture, and **rate and
  scope are domain-separated** by distinct namespaces so neither is derivable
  from the other.

**Validation** uses `VALIDATION_CORRUPTION_SEED = 19225` at a fixed visit, on the
fixed grid `FULL, P50, P100, STRIP_ALL`. The held-out realisation is built
**once** and reused by every candidate, so candidates differ only by the model.
`prepare_condition_batch` mentions no run seed. Selection score is `max` over the
grid; checkpoint tie-break `d_clean` then earliest update; `r` tie-break `d_clean`
then smaller `r`; update 0 is required.

**Assessment: deterministic, seed-separated, no train/dev RNG contamination.**

---

## K. Tasks 9, 10 — model, adapter, run plan (static only)

Inspected statically; **no weights loaded**.

* `verify_model_contract` requires no encoder parameter with `requires_grad`,
  the encoder in `eval`, and exactly **3 551 232** trainable adapter parameters
  at `d = 768`;
* `build_optimizer` **refuses** any parameter that does not require grad —
  silently filtering the encoder out would hide a wiring error;
* weight decay `0.01` on `fusion.weight`/`gate.weight` only; `0.0` on biases,
  LayerNorm and both embedding tables;
* `PRECISION = fp32`, and `autocast`/`GradScaler`/`half`/`bfloat16` are
  AST-asserted absent;
* schedule `LR_SCHEDULE = CONSTANT`, `WARMUP = None`,
  `GRADIENT_ACCUMULATION_STEPS = 1`, `GRADIENT_CLIPPING = None`;
* run plan 3 + 5 + 3 = **11**, budget 20 000 → one continuation → 40 000 →
  `BUDGET_LIMITED`; `selection.py` verifies the LR grid it was given against
  `LR_PILOT_GRID`;
* `smoke_check` calls no `backward`, `step`, `zero_grad`, `build_optimizer` or
  `AdamW` — **structurally incapable of an update**.

**Diagnostic containment.** The pre-G1 frozen-head and burden diagnostics live in
`unmark/evaluation/preg1_*` with their own protocol namespace and their own
`SECONDARY_ANALYSIS_LABEL`. They cannot become Stage-1 scientific results:
Stage-1 artifacts are produced only by `execute_stage`, bind
`STAGE1_PROTOCOL_VERSION`, and `require_compatible` refuses to pool artifacts
across protocol versions.

**All of this is NOT YET EXECUTED**, which the smoke exists to change.

---

## L. Task 11 — F3, training resume

The Stage-6 prepare checkpoint system is proven on real data. **Training
checkpointing is a different system, and it is not wired.**

* `trainer.checkpoint_payload` and `trainer.verify_checkpoint` exist and are
  unit-tested; `train_run` accepts `resume=` and, given one, restores adapter
  state, optimizer, `visit` and the in-pass cursor;
* **but `checkpoint_payload` is called from nowhere** in `unmark/` or `scripts/`,
  and `execute_stage` passes `resume=None` unconditionally.

So a training run that dies mid-stage cannot be resumed: there is nothing to
resume from, and `--output-dir` refuses to reuse its directory.

**Does it block the first training run? No.** A restarted run is deterministic
and scientifically identical — the corruption draws are stateless (§J), the
sampler is a pure function of `(chunk_ids, seed, visit)`, and no partial artifact
is left behind. The cost is wall-clock, not validity.

**Does it matter? Yes, before long runs.** Stage-6 needed several real Colab
runtime deaths to complete, and the 11-run plan is 20 000–40 000 updates per run
on the same infrastructure. Losing a run to a runtime death is likely, not
hypothetical.

**Classification: SHOULD FIX BEFORE LONG RUNS.** Pinned by
`test_training_checkpoint_payload_is_currently_never_persisted`, which documents
the gap as a decision and **fails the moment persistence is wired** — at which
point resume-equivalence needs its own evidence, exactly as the prepare
checkpoint did.

---

## M. Tasks 12, 14, 15 — provenance, blind spots, code quality

**Provenance.** `RunProvenance` binds run seed, corruption seed, learning rate,
`r`, corpus manifest digest and repository HEAD; `verify_checkpoint` refuses to
resume into a different experiment. Output directories must not already exist, so
a run cannot overwrite another. **The gap is F1**: the manifest digest is
recorded but unverified.

**Blind spots, searched for by the patterns that produced the previous real
bugs:**

| Pattern | Status |
|---|---|
| tests constructing `Namespace` manually | Closed in Audit 029 §W; `test_stage1_prepare_cli.py` goes through the real parser |
| tokenizer doubles hiding wrapper semantics | Closed in §V; composition is gated on the tokenizer's own added tokens, direct BPE removed |
| synthetic Unicode assumptions | Closed in §AA; the Hangul case is now explicit, 7 script families tested |
| **multiprocessing instrumentation differences** | **F5, found here** — and now pinned |
| **paths never entered end-to-end** | **The entire training path.** This is the largest remaining blind spot and it is exactly what the no-update smoke is for |
| mocks bypassing real behaviour | The training tests use toy tensors and injected length functions; no test has ever opened the real prepared corpus |

**Highest remaining risk, stated plainly: no code in this repository has ever
read the real 2.2 GB prepared corpus.** Every consumer test uses fixtures. That
is why §N is bounded but real.

**Code quality.** No exception swallowing on scientific paths (violations are
re-raised unchanged); no mutable global RNG; no nondeterministic iteration in
output paths (the membership digest sorts); writes are temp → fsync → replace;
JSON schemas are validated on load. Direct BPE remains absent.

**Audit evidence accuracy** was treated as correctness: Audit 029's header,
verdict and §AB were checked against the real numbers, historical failures were
left intact and labelled, and the collector-wait figure is explained rather than
reported as a defect.

---

## N. Task 17 — the exact real no-update model smoke

**Do not run this until this audit is reviewed.** It is the next step, and it is
the only thing that may load the model.

**Allowed:** model load, a small number of forward passes.
**Forbidden:** `optimizer.step()`, any parameter update, any training loop,
`loss.backward()`.

```bash
# Colab, GPU runtime. Exact HEAD, exact prepared corpus, exact pins.
cd /content/unmark-draft
git rev-parse HEAD            # must print aa49785eadcbd67b64be28a5f67d725c79b41bbb
git status --porcelain        # must print nothing

python - <<'PY'
import json, pathlib
d = pathlib.Path("/content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb")
c = json.loads((d / "COMPLETE.json").read_text())
print(json.dumps(c["counts"], indent=2, sort_keys=True))
assert c["counts"]["chunk_membership_digest"] == \
    "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6"
PY

# The prepared payload and the COMPLETE marker live under different roots, so
# the completion directory is explicit. The smoke verifies the corpus BEFORE the
# model is loaded (F1) and reports load time and peak RSS (F4).
python scripts/stage1_runner.py smoke \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
    --revision 01daacda68afe13d83023d16ec647239e344a1e6 \
    --repository-head "$(git rev-parse HEAD)"
```

**Note (§R):** the HEAD printed by `git rev-parse` will be the *hardening*
commit, not `aa49785`. That is correct and expected — the prepared corpus is
bound by `COMPLETE.json` and its artifact hashes, not by the commit that reads
it. The hardening changed no scientific constant and did not touch Stage 6.

It must report, and a reviewer must check:

| # | Must verify | Why |
|---|---|---|
| 1 | `COMPLETE.json` digest equals `250859a5…78413d6` | binds the smoke to *this* corpus |
| 2 | the consumer opens the real corpus; **peak RSS and load time recorded** | **F4** |
| 3 | manifest counts match §AB.5 exactly | consumer/manifest compatibility |
| 4 | all **5 000** dev parents and **11 443** dev chunks present | dev set is what selection will use |
| 5 | a few train chunks tokenize on **both** pathways, every length ≤ 256 | reference and RAW_BASE agree with the recorded lengths |
| 6 | collator shapes, attention masks, padding, special tokens correct | never exercised on real data |
| 7 | encoder is frozen, adapter has exactly **3 551 232** trainable parameters | model contract |
| 8 | loss is **finite** on a forward pass | numerics |
| 9 | **zero parameter updates** — parameter hashes identical before and after | the smoke's own boundary |
| 10 | no optimizer constructed | structural |

Report `PASS`/`FAIL` with the counts. **Training remains unauthorised until that
report is reviewed.**

---

## O. Tests

Focused suites and the full lightweight repository suite were run. No test was
weakened.

| Suite | Result |
|---|---|
| `tests/test_stage1_pretrain_audit.py` | **8 passed** (new, this audit) |
| `tests/test_stage1_non_vietnamese_orthography.py` | 93 passed |
| `tests/test_stage1_source_coordinates.py` | 26 passed |
| `tests/test_stage1_chunking.py` | 35 passed |
| `tests/test_stage1_lengths.py` | 325 passed |
| `tests/test_stage1_checkpoint.py` | 41 passed |
| `tests/test_stage1_parallel.py` | 18 passed |
| `tests/test_stage1_prepare_cli.py` | 17 passed |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| orthography / decompose / alignment | 1 098 passed |
| **Full repository** | **3 354 passed, 97 skipped, 0 failed** |

The 97 skips are torch-gated; **no model weights were downloaded**, and those
tests are delegated to the Colab smoke.

---

## P. Decision log

**No new decision-log entry was created, and that is deliberate.** This audit
made no specification decision, no clarification, no narrowing and no deviation:
it *verified* D-S1B-001 … D-S1B-013 against code and real artifacts. Manufacturing
an entry because an audit occurred would dilute the log's meaning.

Two findings **may** require entries **when they are fixed**, by whoever fixes
them: F1 (whether the consumer verifies `COMPLETE.json`, the membership digest,
or both) and F3 (training checkpoint cadence and resume-equivalence evidence).
Neither is decided here.

---

## Q. Self-audit

| # | Question | Answer |
|---|---|---|
| 1 | Proposal reread completely? | **yes** — 758 lines, end to end |
| 2 | Traceability matrix complete? | **yes** — §D, 15 requirements classified |
| 3-4 | Constants re-derived; none changed? | **yes** — §E, imported from the repository |
| 5-6 | Official TEST unreachable; leakage review complete? | **yes** — §F, six route classes checked adversarially |
| 7 | Parent partition isolation verified? | **yes** — structural + `parents_spanning_both_partitions = 0` |
| 8-9 | Completion evidence recorded; COMPLETE/ALREADY_COMPLETE reviewed? | **yes** — Audit 029 §AB |
| 10 | Real blocker history preserved? | **yes** — §P–§AA untouched |
| 11-12 | Orthography matches proposal; `señor` classified? | **yes** — §H, NON-BLOCKING with reasoning |
| 13-14 | Training loader and sampler reviewed? | **yes** — §I; **F1**, **F4** |
| 15-17 | Corruption, resume and validation RNG reviewed? | **yes** — §J, stateless by construction |
| 18-20 | Selection metric, LR plan, `r` plan reviewed? | **yes** — §J, §K |
| 21-24 | Adapter, trainable parameters, optimizer groups, scheduler reviewed? | **yes** — §K, statically |
| 25-27 | Run namespaces, training checkpoint identity, provenance reviewed? | **yes** — §L, §M; **F3** |
| 28 | Prepared-corpus consumer compatibility reviewed? | **yes** — §I, delegated to the smoke |
| 29-30 | Multiprocessing and real-tokenizer blind spots reviewed? | **yes** — §C, §M |
| 31-33 | No direct BPE; no normalisation/truncation/drop; nothing silently overwritten? | **yes** |
| 34-35 | No model update occurred; TEST still sealed? | **yes** |
| 36-38 | Full suite passes; `git diff --check` clean; nothing staged? | **yes** — 3 354 / 97 / 0 |
| 39-40 | Audit 029 updated in place; Audit 030 created? | **yes** |
| 41 | Decision log updated only if needed? | **yes** — deliberately **not** updated (§P) |
| 42-43 | Smoke specified; all findings classified? | **yes** — §N, §B |
| 44 | Same substantive audit returned in chat? | **yes** |

---

## R. PRE-TRAIN HARDENING — F1, F2, F3 REPAIRED

**Date:** 2026-08-23 **Baseline:** `aa49785eadcbd67b64be28a5f67d725c79b41bbb`

Sections A-Q above are the audit **as first written** and are not revised: F1-F6
were real, and the record of what was found matters as much as the fix. This
section records what was repaired and what evidence now exists.

| Finding | Original class | Now |
|---|---|---|
| **F1** consumer never verifies the corpus it loads | MUST FIX BEFORE FIRST TRAIN | **RESOLVED** (§R.1) |
| **F2** `PI_STRIP` is two independent literals | MUST FIX BEFORE FIRST TRAIN | **RESOLVED** (§R.2) |
| **F3** training checkpointing never invoked | SHOULD FIX BEFORE LONG RUNS | **RESOLVED, and three further defects found while wiring it** (§R.3) |
| **F4** loader materialises ~2.73 GB | MUST VERIFY IN SMOKE | **UNCHANGED — still delegated to the real smoke**, which now measures it (§R.4) |
| **F5** Stage-6 counters blind under workers>1 | NON-BLOCKING (observability) | **UNCHANGED — deliberately untouched** (§R.5) |
| **F6** stale compiled PDF | DOCUMENTATION ONLY | **UNCHANGED — release work** (§R.6) |

**No scientific constant changed.** 32 re-compared against `aa49785`: none moved.
`unmark/stage1/protocol.py` is **byte-unchanged**.

### R.1 F1 — the consumer now verifies the corpus it loads

`checkpoint.verify_prepared_corpus(prepared_dir, completion_dir) -> VerifiedCorpus`.
**One authoritative path**: it reuses the same `COMPLETE.json`
`write_completion_marker` produces, the same `verify_file`, and the same
relative-name artifact binding the `ALREADY_COMPLETE` path proved on real data.
No second hash verifier was written.

It fails closed unless, in order: the marker exists and parses; `complete` is
true; the schema matches; the identity is the **locked** protocol
(`require_locked_protocol`, new — `require_match` compares two identities, but a
consumer has only one and the repository's own constants to compare it to); every
bound artifact re-hashes correctly **from disk**; both `chunks.jsonl` and
`manifest.json` are actually bound; `load_manifest`/`require_compatible` accepts
the manifest; and the manifest's counts equal the marker's counts.

**On the portable-path question §R asked about**: `COMPLETE.json` binds artifacts
by **relative name plus size and sha256**, never by absolute path, so a prepared
corpus restored to a different root verifies identically — asserted by
`test_a_relocated_prepared_corpus_still_verifies`. No `/content/`,
`/content/drive/` or `aa49785eadcb` appears anywhere in library code. Because the
real run persists the payload and the checkpoint under different roots, the
completion directory is an **explicit** CLI argument (`--completion-dir`,
defaulting to `<prepared-corpus>/_checkpoint` for the co-located case) rather
than inferred.

**Provenance now records a verified value.** `execute_stage` reads
`verified.chunk_membership_digest`, which can only exist once the bytes have been
checked, instead of `manifest["counts"]["chunk_membership_digest"]`. All three
training commands call `_verified_corpus(args)` **before anything else**, and so
does `smoke_check` — before the model is loaded.

**Tests** (`tests/test_stage1_corpus_verification.py`, **28**): acceptance of a
valid synthetic COMPLETE-bound corpus and of a relocated one; and rejection of a
missing marker, a malformed marker, an incomplete prepare, **one flipped bit** in
`chunks.jsonl`, a truncated payload, an appended payload, either bound file
missing, a marker that fails to bind either artifact, a foreign manifest, a
marker whose counts disagree with the manifest, a marker with no membership
digest, and **ten** identity fields off the locked protocol. Plus two structural
tests that provenance uses the verified digest and that every training command
verifies first.

### R.2 F2 — one authoritative `PI_STRIP`

`contracts.py` now does `from unmark.stage1.protocol import PI_STRIP`. Exactly
one numeric definition remains in the repository:
`unmark/stage1/protocol.py:89`.

**Direction, and why.** `protocol.py` is authoritative because it declares itself
"the locked Stage-1 protocol. One source of truth" and already *derives* rather
than types — its seven role seeds come from `derive_seeds`. `contracts.py` is a
mechanism module (policies, config objects), so it consumes the locked value.
The direction is **acyclic**: `protocol` imports only
`unmark.evaluation.profiling`, which reaches `evaluation.contracts` and
`orthography` and never returns to `stage1.contracts`.

**Value unchanged: `PI_STRIP == 0.25`.** `protocol.py` is byte-unchanged; only
`contracts.py`'s duplicate literal was replaced by an import. Editing the
authoritative definition now necessarily changes **both** corruption behaviour
and recorded manifest identity, which is the property F2 required.

### R.3 F3 — training checkpoints, and three defects found while wiring them

**No new cadence was chosen, and no new decision entry was created.** Re-reading
the decision log settles it: **D-S1B-004 already locks** "eval cadence every
`500` updates ... **best + last checkpoint persistence**; optimizer and
corruption `visit` state persistence". F3 was therefore a pure
**implementation-wiring defect** — the machinery existed and `execute_stage`
never invoked it — not an undecided operational question.

The implementation now follows that locked contract:

| D-S1B-004 clause | Implementation |
|---|---|
| eval cadence every 500 updates | `CHECKPOINT_EVERY_UPDATES = EVAL_EVERY_UPDATES`, so the two cannot drift |
| **best + last** persistence | `training-checkpoint-last.pt` (the resume target) **and** `training-checkpoint-best.pt` |
| optimizer state persisted | `optimizer_state` in the payload |
| corruption `visit` state persisted | `sampler_state` carries `seed`, `visit`, `position`, corpus digest |
| deterministic streams and experiment identity preserved | `verify_checkpoint` refuses on ten identity fields |

**"Best" is the already-locked rule, not a new comparison.** The loop asks
`selection.select_checkpoint(result.points)` — lowest score, then lower
`d_clean`, then earliest update — and writes `best` only when that rule names the
current update. There is exactly one definition of "best" in the repository.

**Contents.** Schema version, provenance, adapter state, optimizer state,
`global_update`, sampler state (`seed`, `visit`, `position`, corpus digest),
`cap`, `budget_limited` and **`points`**.

**Atomicity.** temp → flush → `fsync` → `os.replace` → directory fsync. A crash
mid-write leaves the previous checkpoint intact; no `.tmp` survives publication.

**Identity binding.** `verify_checkpoint` refuses on run seed, corruption seed,
LR, `r`, corpus manifest digest, backbone checkpoint and revision, protocol
version, precision — and now **repository HEAD**.

**CLI.** `--resume` on `lr-pilot`, `r-phase1`, `final-main`. Without it an
existing `--output-dir` is still refused, so a fresh run cannot overwrite
evidence and the operator is never asked to delete it. With it the directory must
exist and every checkpoint is identity-verified before use. **Nothing
auto-resumes.** Per-run namespace `run-<label>/_checkpoint`.

**Three further defects, found only because the wiring forced the state to be
examined:**

1. **`points` were read on resume but never written.** `train_run` restored
   `resume.get("points", [])`; `checkpoint_payload` never stored them. A resumed
   run would have silently lost every validation measurement before the
   interruption — and **selection consumes exactly those**. Now persisted and in
   `REQUIRED_CHECKPOINT_KEYS`.
2. **`repository_head` was recorded but never compared.** `require_match`
   checked nine fields and omitted it, so a checkpoint from one commit could
   resume under another whose trainer or corruption code had changed. Now
   compared.
3. **The 20 000 → 40 000 continuation did not continue.** It called `train_run`
   again with `resume=None`, rebuilding the optimizer and restarting the sampler
   at `visit 0`, while the locked budget rule requires preserving "adapter,
   optimizer, `visit`, cursor and streams". It now resumes from the checkpoint
   the first leg writes at exactly `cap`. **This restored an already-locked rule
   rather than changing one** — had it required changing the rule, this audit
   would have stopped and said so.

**Equivalence evidence.** `tests/test_stage1_training_resume_state.py` (**23**,
torch-free so it runs in the ML-free venv) proves interrupted + resumed equals
uninterrupted by **exact** equality of the numeric state, the full
`(chunk_id, visit)` sequence consumed, the update count, `visit`, the sampler
cursor and the validation history — at **five** interruption points: on a
validation boundary, after a pass boundary is crossed, several passes in, one
boundary before the end, and at the final update. The chunk count is deliberately
not a multiple of the batch size, so a pass boundary falls *inside* a batch. A
mutation test proves the comparison actually catches a reset cursor. The payload
round-trips through JSON, so serialisation is exercised.
`tests/test_stage1_training_resume.py` (**torch-gated**) adds exact adapter-tensor
and optimizer-state equality with a real `AdamW`; it skips locally and runs in
Colab, where it needs no model download.

### R.4 F4 — still delegated, and now measured

`load_prepared_chunks` is unchanged: **correctness first**, and the observed
runtime has ~176.88 GiB. `smoke_check` now reports load time, the train/dev chunk
counts and process RSS after the load, so the smoke produces the measurement that
would justify — or not — any later optimisation.

### R.5 F5 — deliberately untouched

Stage-6 behaviour and logging are unchanged. `chunking.py`, `corpus.py`,
`lengths.py`, `parallel.py` and `decompose.py` are **byte-unchanged**.

### R.6 F6 — unchanged

The compiled PDF is still stale and remains release work, not a training
blocker. Not regenerated.

### R.7 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_corpus_verification.py` | **28 passed** (new) |
| `tests/test_stage1_training_resume_state.py` | **23 passed** (new) |
| `tests/test_stage1_training_resume.py` | **skipped locally** (torch-gated; runs in Colab) |
| `tests/test_stage1_pretrain_audit.py` | 9 passed (2 F3 gap-markers **replaced**, §R.8) |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| `tests/test_stage1_schedule.py`, `test_stage1.py` | 90 passed, 14 skipped |
| **Full repository** | **3 406 passed, 98 skipped, 0 failed** (was 3 354 / 97) |

### R.8 Two tests replaced, not weakened

`test_training_checkpoint_payload_is_currently_never_persisted` and
`test_execute_stage_does_not_yet_resume` **asserted the F3 gap**, and the first
said in as many words that it "will fail the moment persistence is wired, at
which point resume equivalence needs its own evidence". It duly failed. Both are
replaced by the post-hardening contract — persistence is wired, resume is
explicit, the continuation genuinely continues — so the gap cannot silently
reopen. A third was added for the continuation defect.

### R.9 Boundaries

**No model weight downloaded, no encoder loaded, no forward pass, no backward,
no optimizer step, no training.** The equivalence tests use a tiny synthetic
module and a pure-Python accumulator. Official UIT-VSFC TEST remains sealed and
unreachable. The prepared corpus was not modified and Stage 6 was not rerun.

### R.10 Hardening self-audit

| # | Question | Answer |
|---|---|---|
| 1-2 | F1 repaired; does training verify the payload it consumes? | **yes** — before model load, in all three commands and the smoke |
| 3 | COMPLETE.json used by the authoritative existing verifier? | **yes** — same marker, same `verify_file`, no second verifier |
| 4 | Can modified/truncated/swapped data train? | **no** — 1-bit, truncation, append, substitution and absence all refused |
| 5 | Provenance records a verified identity? | **yes** — `verified.chunk_membership_digest` |
| 6-9 | F2 repaired; one authoritative value; still 0.25; shared source? | **yes / yes / yes / yes** |
| 10-13 | F3 repaired; persisted; crash-safe; explicit fail-closed resume? | **yes** — implementing **D-S1B-004**'s locked best+last contract; no new decision |
| 14-18 | Model, optimizer, sampler/cursor/visit, update count, validation schedule restored? | **yes** — all asserted |
| 19-20 | Run identity and prepared-corpus identity verified on resume? | **yes** — including repository HEAD, newly enforced |
| 21-22 | Interrupted ≡ uninterrupted; crosses validation/pass boundaries? | **yes** — exact equality at 5 points, pass boundary inside a batch |
| 23 | Stage-6 code untouched? | **yes** — byte-unchanged |
| 24-25 | F4 left for the smoke; F5 non-blocking? | **yes / yes** |
| 26-27 | No scientific constant changed; TEST inaccessible? | **yes** — 32 compared, none moved |
| 28-32 | No encoder, forward, backward, optimizer step, training? | **yes** |
| 33-35 | Full tests pass; `git diff --check` clean; nothing staged? | **yes** — 3 406 / 98 / 0 |
| 36-37 | Audit 030 updated in place; decision log updated only where needed? | **yes** — §R only. **No decision entry created**: D-S1B-004 already locked the contract, and duplicating it would have been worse than leaving it |
| 38 | Real no-update smoke still required? | **YES — it is the next step, and it is unchanged (§N)** |

---

## S. ADDENDUM — F3 WAS A WIRING DEFECT, AND THE VALIDATION-COST CONCERN IS UNMEASURED

**Date:** 2026-08-23. Two corrections and one preparation.

### S.1 Correction: no new cadence decision was needed, and the one I made is withdrawn

§R first recorded a new decision entry, **D-S1B-014**, choosing a checkpoint
cadence. **That was wrong and it has been removed.** Re-reading the decision log
settles the question: **D-S1B-004** already locks, in its
`LOCKED — A-PRIORI ENGINEERING` tier,

> "eval cadence every `500` updates; ... **best + last checkpoint persistence**;
> optimizer and corruption `visit` state persistence"

So F3 was never an undecided operational question. It was a pure
**implementation-wiring defect**: the machinery existed, `execute_stage` never
invoked it. Creating a second entry would have duplicated a locked decision and,
worse, made it look as though the cadence had been chosen here.

`docs/spec/decisions.md` is therefore **unchanged by this hardening**, and every
reference in the code and in §R now cites **D-S1B-004**.

### S.2 Correction: the implementation was incomplete against that locked contract

Writing only a single overwriting checkpoint satisfied "last" and **not
"best"**. Now both are persisted:

| D-S1B-004 clause | Implementation |
|---|---|
| eval cadence every 500 updates | `CHECKPOINT_EVERY_UPDATES is EVAL_EVERY_UPDATES` — asserted identical, so they cannot drift |
| **best + last** persistence | `training-checkpoint-last.pt` (the resume target) **and** `training-checkpoint-best.pt` |
| optimizer state persisted | `optimizer_state` |
| corruption `visit` state persisted | `sampler_state` — `seed`, `visit`, `position`, corpus digest |
| deterministic streams, experiment identity | `verify_checkpoint`, ten fields incl. repository HEAD |

**"Best" is the already-locked rule.** The loop calls
`selection.select_checkpoint(result.points)` — lowest score, then lower
`d_clean`, then earliest update — and writes `best` only when that rule names the
current update. A second comparison written inside the trainer is how a
selection rule silently forks, so an AST test requires `select_checkpoint` to
appear in `train_run`'s call graph.

### S.3 Nothing was changed in response to the validation-cost concern

The external concern is **plausible but unmeasured**, so this commit changes
**none** of: the 500-update eval cadence, the full held-out dev set, the four
validation conditions, or the selection metric. No validation subset, no
1 000-update cadence, no reference-embedding cache, no retrieval-based exclusion
rule.

**Any future change to dev subset, eval cadence, or a hard collapse-exclusion
threshold must be an explicit pre-training scientific decision recorded in the
log — never a silent performance optimisation.**

### S.4 A finding the measurement request surfaced: the recorded lengths are not recorded

Requirement 1 asks for a profile "using **RECORDED** lengths". **They are not in
the artifact.** `chunk_record` persists `chunk_id`, `document_id`, `partition`,
`chunk_index`, `text`, `source_start`, `source_end`, `source_shard` — and
nothing else. Both `reference_length` and `base_length` were computed and
*enforced* during Stage 6 (that is how `overflow_count = 0` was established) but
never written down.

Consequences, stated plainly rather than worked around:

* a token-length profile can only be **recomputed** with the pinned tokenizer,
  which is a full re-tokenization pass over 2 633 067 chunks;
* adding the lengths to the payload would change the artifact bytes and require
  **re-running Stage 6**, which is forbidden and not worth it;
* so the profiler reports **two** things and labels them differently:
  **exact and free** (character lengths, chunks-per-parent — from the payload,
  every chunk) and **recomputed on a sample**, marked
  `recomputed_not_recorded`.

Classification: **DOCUMENTATION / FUTURE-ARTIFACT**, non-blocking. If a future
prepare is ever re-run for another reason, persisting the two lengths would make
this free.

### S.5 The measurement tool

`scripts/stage1_pretrain_measurements.py` (new). It **verifies the corpus first**
(F1), **streams** the payload rather than materialising it (so the profile does
not reintroduce F4), constructs **no optimizer**, and calls no `backward`,
`step`, `zero_grad`, `AdamW`, `build_optimizer`, `train_run` or `execute_stage` —
AST-asserted. `--profile` needs no model at all.

| Requirement | How |
|---|---|
| 1 profile, train/dev separately | `--profile`: p25/50/75/90/95/99/max, fractions ≤32/≤64/≤128, chunks-per-parent, per partition. Character lengths exact over all 2 633 067 chunks; token lengths via `--tokens` on a sample, labelled recomputed |
| 2 full validation wall-clock | ~~`--validation`~~ **NOT DELIVERED — see §T.1.** The mode loads the real encoder but times only condition *preparation*; it runs no forward pass. The measurement remains outstanding |
| 3 clean-reference h(x) timing | **NOT DELIVERED — see §T.1.** No forward pass runs, so no h(x) timing exists |
| 4 collapse diagnostics | the locked per-batch SD of `h'` is retained unchanged in `MonitorWindow`; self-retrieval metrics are **not** added in this commit (§S.6) |

### S.6 Reference-embedding caching — semantics assessed, implementation deferred

**Asked for: an assessment, not an implementation.** Here it is.

**Is it semantically valid?** For the clean reference `h(x)` — **yes, in
principle**. The encoder is frozen and in `eval`, the held-out realisation is
built **once** and reused by every candidate, and the reference pathway depends
only on `canon(x)` and the pinned tokenizer. None of that varies across
candidates or across updates, so `h(x)` for a given dev chunk is a constant of
the experiment, not of the run.

**What a cache identity would have to bind**, or it silently serves the wrong
vectors:

| # | Field | Why |
|---|---|---|
| 1 | `chunk_membership_digest` (verified) | ties the cache to *this* prepared corpus |
| 2 | chunk id **and** a digest of the chunk text | a chunk whose text changed must miss |
| 3 | `ENCODER_CHECKPOINT` + `ENCODER_REVISION` | different weights, different `h(x)` |
| 4 | Transformers version | tokenizer or model-code changes move the vectors |
| 5 | `MAX_LENGTH`, `RAW_BASE_POLICY` | change the input the encoder sees |
| 6 | pooling identity (`attention_masked_mean_non_special`) | changes what `h` *is* |
| 7 | `PRECISION` (`fp32`) | fp16/bf16 would give different numbers |
| 8 | condition (`FULL`/`P50`/`P100`/`STRIP_ALL`) + `VALIDATION_CORRUPTION_SEED` + visit | only `FULL`'s clean reference is candidate-invariant; the corrupted conditions depend on the corruption draw |
| 9 | repository HEAD | orthography or pooling code can change under it |

**Deferred deliberately.** The benefit is unquantified: §S.5 measurement 3 is
exactly what would quantify it. A cache that is wrong is far worse than a
validation pass that is slow, and nine identity fields is a real surface. **Wait
for the timing.**

### S.7 Collapse diagnostics — retained, not extended

The already-locked per-batch SD of `h'` in `MonitorWindow` is **unchanged**.
Self-retrieval Recall@1 / MRR / positive-vs-hardest-negative margin are *not*
added here: they were offered as optional, they are diagnostic-only, and adding
a metric to a monitoring surface immediately before a phase boundary invites it
being read as a gate. **No automatic exclusion threshold exists, and none is
proposed.** If such a rule is ever wanted it must be an explicit recorded
decision.

### S.8 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_pretrain_measurements.py` | **6 passed** (new) |
| `tests/test_stage1_training_resume_state.py` | **26 passed** (+3 for best/last) |
| `tests/test_stage1_corpus_verification.py` | 28 passed |
| `tests/test_stage1_pretrain_audit.py` | 9 passed |
| **Full repository** | **3 415 passed, 98 skipped, 0 failed** |

### S.9 The smoke, restated with the measurements

Run **after** this commit is reviewed, in the order below. **Step 1 requires no
model. Steps 2 and 3 are real model execution — no-update, but the encoder is
loaded.** Step 2 is **incomplete** (§T.1) and must not be read as validation
cost.

```bash
# 1. Descriptive profile -- no model, streams the payload.
python scripts/stage1_pretrain_measurements.py --profile --tokens \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb

# 2. INCOMPLETE (§T.1): loads the real encoder but times condition preparation
#    only -- no forward pass. NOT validation wall-clock.
python scripts/stage1_pretrain_measurements.py --validation \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb

# 3. The no-update model smoke itself (§N), unchanged.
python scripts/stage1_runner.py smoke \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
    --revision 01daacda68afe13d83023d16ec647239e344a1e6 \
    --repository-head "$(git rev-parse HEAD)"
```

**The validation timing decides nothing by itself.** It produces the number the
cost concern has so far lacked. If it turns out to be material, the response is
an explicit recorded decision — not a quiet change to cadence, dev subset,
conditions or metric.

---

## T. FINAL CONSISTENCY CHECK — A BLOCKER IN §S's OWN MEASUREMENT TOOL

**Date:** 2026-08-23. Two points were checked before commit. The first found a
real defect, in work this audit itself added.

### T.1 BLOCKER — `--validation` does not measure validation

§S.9 said "Steps 1-2 need no model" and §S.5 said `--validation` measures
"full four-condition validation wall-clock ... `torch.no_grad` ... peak GPU
memory ... clean-reference h(x) timing". **Both were false.** Traced through the
implementation and confirmed on the call graph:

| Claim | Reality |
|---|---|
| "needs no model" | **False.** It calls `build_objective(revision)` → `AutoModel.from_pretrained` — it loads the real pinned encoder |
| "validation wall-clock" | **False.** It times only `prepare_condition_batch` — tokenisation and corruption preparation |
| forward passes | **Zero.** `evaluate` is imported and **never called**; the objective is never invoked |
| `torch.no_grad` | **Absent** |
| peak GPU memory | **Meaningless** — nothing is placed on the GPU, so the figure would be ~0 |
| clean-reference h(x) timing | **Does not exist** |
| `--reference`, `--collapse` | **Advertised in the module docstring but never implemented** — no such flags exist |

This is precisely the substitution the measurement request forbade:
*"Do not replace real validation timing with tokenisation-only or synthetic
timing."* §S.5's table asserted a capability that the code does not have.

**Classification: BLOCKER.** Not because anything scientific is wrong — no
protocol value moved, no training path is affected, and the F1/F2/F3 repairs in
§R are untouched and still hold — but because an audit that advertises a
measurement it cannot take is worse than one that takes none. The cost concern
would have been "answered" with a number that measures tokenisation.

**Deliberately NOT rewritten here.** Making `--validation` real means batching
the dev set through the objective, device placement, `torch.no_grad`, and CUDA
synchronisation for honest wall-clock. That is a design change requiring review,
not something to slip into a consistency check minutes before a commit. What was
done instead is to **make the tool and this audit tell the truth**:

* the mode is renamed in its output to
  `condition_preparation_only_NOT_validation_wall_clock`, with
  `incomplete: true`, `forward_passes: 0` and an explicit warning field;
* its docstring states plainly that it must not be read as validation cost;
* the module docstring no longer advertises `--reference` or `--collapse`.

**Consequence: the validation-cost concern remains unmeasured**, exactly as it
was before §S. Measuring it is now a named, separate piece of work.

### T.2 The recomputed token profile is reproducible

Verified against the implementation, not the prose:

| Property | Value |
|---|---|
| sample size | `--sample`, default **20 000 per partition** (a cap) |
| selection method | every **97th** payload line, in document order, until each partition's cap fills |
| deterministic? | **Yes** — fixed stride over a fixed line order; no RNG, no seed, no dependence on process, environment or iteration order |
| seed | **None**, and none needed |
| chosen from observed statistics? | **No** — a constant stride fixed in advance |
| train / dev separately | **Yes**, but see below |
| tokenizer | `vinai/phobert-base` @ `--revision` (default the locked `01daacda…`) |
| Transformers version | now **recorded in the output** |
| both pathways | **Yes** — `reference_length` and `base_length` |

**Deterministic, so it was not redesigned** — per instruction. What was added is
the *identity*, so the numbers are reproducible from their own report: stride,
requested and **obtained** counts per partition, `deterministic: true`,
`seed: null`, tokenizer checkpoint and revision, whether that revision is the
locked one, Transformers version, and both pathway names.

**One honest caveat, reported rather than fixed.** The counter is shared across
partitions, and dev is ~0.43 % of the payload, so on the real corpus the stride
yields roughly **117 of 11 443** dev chunks while train fills its 20 000 cap.
Reproducible, but the **dev token profile is thin**. The `obtained` field now
surfaces this in the output rather than leaving it to be inferred. Changing it
would be a redesign, which the instruction excluded; if a denser dev sample is
wanted it is a small, explicit change.

### T.3 Everything else re-verified

| Check | Result |
|---|---|
| `EVAL_EVERY_UPDATES` | **500** |
| dev | **5 000** parents / **11 443** chunks |
| `VALIDATION_CONDITIONS` | `('FULL', 'P50', 'P100', 'STRIP_ALL')` |
| selection metric | unchanged |
| `PI_STRIP` | **0.25**, single-sourced (`protocol.py:89` only) |
| best + last persistence | follows **D-S1B-004**; `select_checkpoint` decides "best" |
| new decision-log entry | **none** |
| `docs/spec/decisions.md` diff vs `aa49785` | **empty** |
| Stage-6 implementation | `chunking.py`, `corpus.py`, `lengths.py`, `parallel.py`, `decompose.py` **byte-unchanged** |
| official TEST | unreachable |
| model load / forward / backward / optimizer / training in this check | **none** |
| full suite | **3 415 passed, 98 skipped, 0 failed** |
| `git diff --check` / staged | clean / **0** |

### T.4 Verdict of this check

**A code defect was found, not merely a wording slip**, so this is not a
documentation-only pass. §R's F1/F2/F3 repairs are unaffected and remain sound;
the defect is confined to the §S measurement tool, which now reports its own
incompleteness instead of overstating it.

---

## U. REAL VALIDATION MEASUREMENT PATH REPAIRED

**Date:** 2026-08-23. **§T is preserved above as the record of the defect** —
it was real, it was found before commit, and deleting it would hide that the
tool this audit added had to be repaired.

### U.1 The defect, and the repair

| §T found | Now |
|---|---|
| `evaluate` imported, **never called** | `validation.evaluate` is **called** — the same function `execute_stage` hands to `train_run` as `evaluate_fn` |
| zero forward passes | real encoder **and** adapter forwards, counted |
| no `torch.no_grad` | the authoritative `evaluate` supplies it; the proxy **records** whether grad was enabled and fails closed if it ever was |
| no device placement | model moved to CUDA when available; `--require-cuda` refuses to report a CPU number as a GPU one |
| meaningless GPU peak | `reset_peak_memory_stats` immediately before the recurring region; peak **allocated** and **reserved** reported; process RSS never labelled GPU |
| no clean-reference timing | `reference_representation` timed separately from `adapted_representation` |
| `--reference` / `--collapse` advertised, absent | removed from the docstring; only implemented modes are documented |
| placeholder mode name | deleted; a test forbids its return |

### U.2 One authoritative evaluator — instrumented, not reimplemented

`InstrumentedObjective` is a **transparent proxy**. Every call delegates to the
real objective, so `evaluate` runs its own `objective.eval()`, its own
`torch.no_grad()`, its own batching, distance, pooling and aggregation. The tool
observes; it does not compute.

Asserted structurally: `representation_distance`, `collate_stage1_batch`,
`masked_mean` and `cosine_similarity` are **not** called anywhere in the
measurement script — they come from the evaluator or nowhere.

**One-time vs recurring, the distinction §T's predecessor got wrong.**
`execute_stage` builds `prepared_by_condition` **once** before the run loop and
reuses it for every evaluation. So condition preparation is *setup*; only
`evaluate` recurs, every `EVAL_EVERY_UPDATES = 500` updates. They are timed and
reported separately, and the projection multiplies **only** the recurring
figure — the old tool would have charged a one-time cost 41 times.

Reported: `corpus_verification`, `prepared_corpus_load`,
`one_time_condition_setup`, `recurring_validation_total`, per-condition
estimate, batches per condition, and observed forward-call counts.

### U.3 Clean-reference h(x): honest, because the evaluator already separates it

`evaluate` calls `reference_representation` once per batch **per condition** —
so the clean reference is recomputed four times over, and that is exactly the
candidate-invariant work a frozen cache would remove. Measuring it needed no
rewriting: the proxy times that call separately.

Reported as `reference_forward_seconds` / `adapted_forward_seconds` with call
counts, so the cache's ceiling is a measured fraction rather than a guess.
**No cache is implemented** (§S.6 stands).

### U.4 The no-update boundary, enforced not asserted

Every parameter is hashed **before and after** the full measurement, split into
**trainable** and **frozen encoder** digests, and the report carries
`parameter_hash_before`, `parameter_hash_after` and `parameters_identical`.
`validation_failures` refuses to report PASS unless:

* `forward_passes > 0`;
* all four locked conditions executed;
* grad was never enabled during a forward, and no output required grad;
* **parameters are identical**;
* no optimizer was constructed, no backward, no step;
* CUDA timing was synchronised when on CUDA.

The CLI exits non-zero on FAIL. Structurally, `backward`, `step`, `zero_grad`,
`AdamW`, `build_optimizer`, `train_run`, `execute_stage` and
`save_training_checkpoint` are **unreachable** from the measurement script —
AST-asserted.

### U.5 Profile sampling repaired — `partition-local-stride-v2`

§T.2 recorded that v1 was reproducible but starved dev: one counter shared
across partitions gave dev roughly **117 of 11 443**.

v2 counts **within each partition**, with a stride derived from that partition's
own population: `stride = max(1, population // requested)`. On the real corpus:

| Partition | Population | Requested | Stride | Obtained |
|---|---|---|---|---|
| dev | 11 443 | 20 000 | **1** | **11 443 — the whole population** |
| train | 2 621 624 | 20 000 | **131** | **20 000**, spread across the full order |

Still deterministic and **data-independent**: selection depends only on
partition-local position — never on token length, text, corruption outcome,
observed statistics or labels. No RNG, no seed, and **no `hash()`** (salted per
process; asserted absent from the call graph, not from the prose — the
docstring legitimately mentions it).

The output records `sampling_scheme_version`, `selection_method`,
`deterministic`, `seed: null`, `data_independent`, and per partition the
`population_count`, `requested_count`, `obtained_count`, `stride` and
`complete_population`. Token lengths remain labelled **`recomputed_not_recorded`**
with tokenizer checkpoint, revision, whether that revision is the locked one,
Transformers version and both pathway names — because the prepared artifact
never persisted them (§S.4).

**This profiling is descriptive only.** It affects no training, validation,
selection or scientific decision.

### U.6 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_measurement_contract.py` | **31 passed** (new, **torch-free** — runs in the ML-free venv) |
| `tests/test_stage1_validation_measurement.py` | **skipped locally**, torch-gated; tiny injected model, no PhoBERT download |
| **Full repository** | **3 446 passed, 99 skipped, 0 failed** |

The fail-closed matrix runs **locally, every run**: zero forwards, each of the
four conditions missing in turn, grad enabled, an output requiring grad,
parameter mutation, optimizer constructed, a backward, a step, and
unsynchronised CUDA timing — each must fail; a CPU report must not be forced to
synchronise. Sampling: dev-below-cap taken whole, train spread over its
population, identical across repeated runs, partition-aware, independent of the
data, full identity recorded.

Three of my own test bugs were caught and fixed while writing these, one of them
the recurring prose-matching defect: `"hash(" not in SOURCE` matched the
docstring that *forbids* `hash()`. It now checks the call graph.

### U.7 What did NOT change

F1, F2 and F3 are untouched and re-verified: prepared-corpus verification before
model load, provenance on the **verified** digest, one `PI_STRIP` (still 0.25,
single-sourced), D-S1B-004's best+last with cadence `is` `EVAL_EVERY_UPDATES`,
explicit fail-closed resume, `points` persisted, `repository_head` verified, and
the 20k→40k continuation genuinely continuing.

**No decision-log entry**, and `git diff aa49785 -- docs/spec/decisions.md` is
**empty**. Stage-6 implementation byte-unchanged. Prepared corpus untouched.
No reference cache, no validation subset, no cadence change, no retrieval
exclusion rule. Every scientific constant re-verified unchanged.

### U.8 Self-audit

| # | Question | Answer |
|---|---|---|
| 1-2 | Real forwards; same authoritative evaluator as training? | **yes / yes** — `validation.evaluate`, instrumented by proxy |
| 3-6 | All 11 443 dev chunks; four conditions; batch 128; seed 19225? | **yes** — all read from the locked constants, none redefined |
| 7-8 | Recurring wall-clock honest; one-time setup separated? | **yes** — projection multiplies the recurring figure only |
| 9-10 | CUDA synchronised; GPU peak real? | **yes** — sync before/after; allocated **and** reserved; RSS never labelled GPU |
| 11 | Clean-reference timing honest? | **yes** — the evaluator already separates the call; timed, not fabricated |
| 12-15 | No optimizer; no backward; zero steps; parameter hashes identical? | **yes** — enforced, fails closed |
| 16 | Can a zero-forward run masquerade as validation? | **NO** — fails closed, tested |
| 17-19 | Dev profile is all 11 443; train deterministic; still `recomputed_not_recorded`? | **yes / yes / yes** |
| 20-25 | F1/F2/F3 intact; D-S1B-004 authoritative; decisions.md unchanged; constants unchanged; Stage 6 byte-unchanged; TEST unreachable? | **yes** to all |
| 26-27 | No model download/load; no local forward/backward/optimizer/training? | **yes** — torch is absent locally; the torch half is Colab-gated |
| 28-30 | Full suite passes; `git diff --check` clean; nothing staged? | **3 446 / 99 / 0**; clean; **0** |
| 31-33 | §T preserved; §U added; no Audit 031? | **yes / yes / yes** |
| 34 | Real Colab no-update smoke still required after commit? | **YES** |

---

**STATUS: PRE-TRAIN HARDENING FINAL PASS — READY TO COMMIT AND RUN REAL NO-UPDATE MODEL SMOKE**
**§T's BLOCKER REPAIRED: `--validation` NOW RUNS THE AUTHORITATIVE `validation.evaluate`**
**REAL ENCODER + ADAPTER FORWARDS, COUNTED; ALL FOUR CONDITIONS; BATCH 128; SEED 19225**
**ONE-TIME CONDITION SETUP SEPARATED FROM RECURRING WORK — THE PROJECTION USES RECURRING ONLY**
**CUDA SYNCHRONISED AROUND TIMING; PEAK ALLOCATED AND RESERVED; RSS NEVER LABELLED GPU**
**CLEAN-REFERENCE h(x) TIMED VIA THE EVALUATOR'S OWN CALL — NO CACHE IMPLEMENTED**
**EVERY PARAMETER HASHED BEFORE AND AFTER; ZERO FORWARDS CANNOT MASQUERADE AS VALIDATION**
**PROFILE SAMPLING IS PARTITION-AWARE: ALL 11 443 DEV, 20 000 TRAIN AT STRIDE 131**
**F1/F2/F3 INTACT; decisions.md STILL UNCHANGED; STAGE 6 BYTE-UNCHANGED; NO CONSTANT MOVED**
**A PASS DOES NOT AUTHORISE TRAINING — THE REAL NO-UPDATE SMOKE IS STILL REQUIRED**

~~**STATUS: PRE-TRAIN HARDENING FINAL CONSISTENCY FAIL — DO NOT COMMIT AS-IS (§T)**~~ **— resolved by §U**
**BLOCKER: §S's `--validation` LOADS THE REAL ENCODER BUT RUNS NO FORWARD PASS**
**IT TIMES `prepare_condition_batch` ONLY — TOKENISATION STANDING IN FOR VALIDATION COST**
**`--reference` AND `--collapse` WERE ADVERTISED IN THE DOCSTRING AND NEVER IMPLEMENTED**
**TOOL AND AUDIT NOW STATE THIS; THE VALIDATION-COST MEASUREMENT REMAINS OUTSTANDING**
**THE F1/F2/F3 REPAIRS IN §R ARE UNAFFECTED AND STILL HOLD**
**TOKEN SAMPLING IS DETERMINISTIC (STRIDE 97, NO SEED); IDENTITY NOW RECORDED IN OUTPUT**
**CAVEAT REPORTED, NOT REDESIGNED: DEV YIELDS ~117 OF 11 443 UNDER THE SHARED COUNTER**
**NO PROTOCOL VALUE MOVED; decisions.md STILL HAS NO HARDENING DIFF; STAGE 6 BYTE-UNCHANGED**

~~**STATUS: PRE-TRAIN HARDENING PASS — READY TO COMMIT, THEN RUN THE REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §T**
**F3 IS A WIRING DEFECT: D-S1B-004 ALREADY LOCKED CADENCE 500 AND best+last PERSISTENCE**
**THE INVENTED D-S1B-014 IS WITHDRAWN; `decisions.md` IS UNCHANGED BY THIS HARDENING**
**`best` IS NOW PERSISTED TOO, CHOSEN BY THE ALREADY-LOCKED `select_checkpoint` RULE**
**NO CHANGE TO EVAL CADENCE, DEV SET, THE FOUR CONDITIONS, OR THE SELECTION METRIC**
**VALIDATION-COST CONCERN IS PLAUSIBLE BUT UNMEASURED — THE SMOKE WILL TIME IT (§S)**
**REFERENCE-EMBEDDING CACHE ASSESSED AS SEMANTICALLY VALID BUT DEFERRED PENDING TIMING**
**FINDING: reference_length/base_length ARE NOT PERSISTED, SO A PROFILE MUST RECOMPUTE THEM**
**F1 RESOLVED: THE TRAINING CONSUMER NOW VERIFIES THE PAYLOAD IT LOADS, BEFORE MODEL LOAD**
**F2 RESOLVED: ONE AUTHORITATIVE `PI_STRIP`, STILL EXACTLY 0.25, `protocol.py` BYTE-UNCHANGED**
**F3 RESOLVED AS A WIRING DEFECT: D-S1B-004's LOCKED best+last CONTRACT IS NOW IMPLEMENTED**
**NO NEW DECISION ENTRY — THE CADENCE AND PERSISTENCE SEMANTICS WERE ALREADY LOCKED**
**THREE FURTHER DEFECTS FOUND WHILE WIRING F3: `points` NEVER PERSISTED, `repository_head`**
**NEVER COMPARED, AND THE 20k->40k CONTINUATION SILENTLY RESTARTING INSTEAD OF CONTINUING**
**INTERRUPTED == UNINTERRUPTED BY EXACT EQUALITY AT FIVE INTERRUPTION POINTS**
**F4 STILL DELEGATED TO THE SMOKE, WHICH NOW MEASURES LOAD TIME AND PEAK RSS**
**32 SCIENTIFIC CONSTANTS RE-COMPARED AGAINST `aa49785` — NONE CHANGED**
**NO ENCODER, NO FORWARD, NO BACKWARD, NO OPTIMIZER STEP, NO TRAINING; TEST STILL SEALED**
**A PASS HERE DOES NOT AUTHORISE TRAINING — THE BOUNDED NO-UPDATE SMOKE (§N) IS NEXT**

~~**STATUS: PRE-TRAIN REPOSITORY-WIDE AUDIT PASS — READY FOR REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §R**
**NO BLOCKER FOUND; 2 MUST-FIX-BEFORE-FIRST-TRAIN, 1 SHOULD-FIX-BEFORE-LONG-RUNS, 1 SMOKE-VERIFY**
**F1 THE TRAINING CONSUMER RECORDS A CORPUS DIGEST IT NEVER VERIFIES — MUST FIX**
**F2 `PI_STRIP` IS TWO INDEPENDENT LITERALS; CORRUPTION AND THE MANIFEST READ DIFFERENT ONES — MUST FIX**
**F3 TRAINING CHECKPOINTING IS IMPLEMENTED BUT NEVER INVOKED — BEFORE LONG RUNS**
**F4 THE LOADER MATERIALISES ~2.73 GB AND HAS NEVER OPENED THE REAL CORPUS — VERIFY IN SMOKE**
**F5 STAGE-6 ZERO COUNTERS ARE MAIN-PROCESS BLINDNESS, NOT A BYPASSED LENGTH CONTRACT**
**OFFICIAL UIT-VSFC TEST UNREACHABLE BY ANY ROUTE CONSTRUCTED IN THIS AUDIT**
**CORRUPTION IS STATELESS AND SEED-SEPARATED; VALIDATION CANNOT SEE A TRAINING SEED**
**NO ENCODER LOADED, NO FORWARD PASS, NO OPTIMIZER, NO UPDATE, NO WEIGHTS DOWNLOADED**
**A PASS HERE DOES NOT AUTHORISE TRAINING — THE BOUNDED NO-UPDATE SMOKE (§N) IS NEXT**

---

## V. FIRST REAL NO-UPDATE SMOKE — PROVENANCE ROUND-TRIP FAILURE

**Revision 6.** The first real Colab pre-train smoke was run against the
committed hardening HEAD. It is the first time any of this code has executed on
the real prepared corpus in a GPU runtime. It **did not reach the model**, and
this section records that failure rather than replacing it with the repair.

### V.1 What the real smoke established before it stopped

| Gate | Result |
|---|---|
| Hardening HEAD | `8f07842a1434b40ec4f4ffa2a2681da499fd1fc6` |
| Exact-HEAD verification | **PASS** |
| Clean-repository verification | **PASS** |
| Prepared artifact | `aa49785eadcb` |
| Byte-exact prepared restore | **PASS, byte exact** |
| Persistence manifest verification | **PASS** |
| `COMPLETE.json` / member digest | **PASS** |
| `pytest -q tests/test_stage1_training_resume.py` | **6 failed, 8 passed** |
| Model validation | **NOT RUN** |
| PhoBERT real forward | **NOT RUN** |
| Optimizer | **NOT RUN** |
| Training | **NOT RUN** |

The corpus half of the smoke is therefore **real evidence and it passed**: the
prepared payload restores byte-exactly in a fresh runtime and verifies against
its own manifest and completion marker. The failure is downstream of that, in
the training-resume gate, and it stopped the smoke fail-closed **before** any
model was loaded — which is the behaviour the gate ordering was designed for.

### V.2 The failure

All six failures were the same construction error inside
`test_a_foreign_run_cannot_resume`:

```
mine   = provenance()
theirs = RunProvenance(**{**mine.to_dict(), field: value})
TypeError: RunProvenance.__init__() got an unexpected keyword argument 'lambda_align'
```

The six intended mutations — `run_seed`, `corruption_seed`, `learning_rate`,
`r`, `corpus_manifest_digest`, `repository_head` — therefore **never reached
`verify_checkpoint`**. The test that exists to prove foreign runs are rejected
had never once demonstrated it in a torch-enabled environment.

### V.3 The contract, re-derived

`RunProvenance` (`unmark/stage1/trainer.py`) is a `@dataclass(frozen=True)`.

| | |
|---|---|
| **Constructor fields (10)** | `run_seed`, `corruption_seed`, `learning_rate`, `r`, `corpus_manifest_digest`, `repository_head`, `backbone_checkpoint`, `backbone_revision`, `protocol_version`, `precision` |
| **`to_dict()` keys (12)** | the 10 above **plus** `lambda_align`, `lambda_clean` |
| **The extra two** | **Derived, not stored.** `weights` is a `@property` computing `lambdas_for_r(self.r)`; `to_dict()` splices in `self.weights.to_dict()` |

`lambda_align` is therefore **not an alias and not a compatibility field**. It is
a *derived scientific quantity*, recorded so an artifact states the weights its
objective actually used. Its authoritative definition is `protocol.py:121`:

```
lambdas_for_r(r) = ( S/(1+r),  S·r/(1+r) )     with S = LAMBDA_SCALE_SUM = 2.0
```

so `lambda_align + lambda_clean == S` and `lambda_clean / lambda_align == r`.
`r` and `lambda_align` do have a required mathematical relationship, and it is
one-directional: `r` is the identity, the lambdas follow from it and cannot be
set independently.

**`to_dict()` is artifact serialization, not a constructor round trip.** That is
now stated in its own docstring, and `DERIVED_KEYS` is declared on the class.

### V.4 Is the production path affected? — traced, and no

The decisive question was whether a real persisted checkpoint can ever be
reconstructed with `RunProvenance(**serialized_dict)`. It cannot. The full
lifecycle:

| Step | Code | Direction |
|---|---|---|
| construct | `execute.py:150` — the **only** production construction site | explicit keyword arguments from the *plan*, never `**dict` |
| serialize | `checkpoint_payload` → `"provenance": provenance.to_dict()` | object → dict |
| persist | `save_training_checkpoint` → `_publish` → `torch.save` | dict → disk |
| load | `load_training_checkpoint` → `torch.load` | disk → **raw dict** |
| verify | `verify_checkpoint` → `provenance.require_match(payload["provenance"])` | **dict compared against a freshly constructed identity** |

There is **no `from_dict` anywhere in the repository**, and nothing rebuilds a
`RunProvenance` from an artifact. `scripts/stage1_runner.py:478`
(`_load_selection`) reads result artifacts as plain dicts and touches only
scalars. This is not an omission — it is a **stronger** design than round
tripping: a run's identity comes from its plan and its environment, so a
corrupted or foreign checkpoint can never define which experiment it belongs to,
only fail to match one.

**Classification: CASE A — test-construction bug.** No production path had the
defect. The audit did not assume this from the traceback; it was established by
tracing every construction, serialization and read site.

### V.5 The one real gap the trace did expose

`require_match` compared all **10** constructor fields — no identity field was
unguarded — but the **2 derived keys were serialized into every checkpoint and
never read back by anything**. Under honest production they cannot disagree with
`r` (same pure function, and `repository_head` is compared, so `lambdas_for_r`
itself cannot have changed). A corrupted, truncated or hand-edited artifact
could: it could claim `r = 1.0` while carrying `lambda_align = 99.0`, and no gate
in the repository would have looked. They were the only scientific quantity a
checkpoint carried that nothing ever checked.

`require_match` now validates them, after the identity comparison so that a
foreign `r` is still reported as an `r` mismatch:

* a derived key **missing** → refused;
* a derived key **inconsistent with the artifact's own `r`** → refused as
  *"internally inconsistent"*.

**No scientific value changed.** The relationship being enforced is the one
already locked in `protocol.py`; enforcing a locked formula is not a new
decision, so **no decision-log entry was created** and `docs/spec/decisions.md`
is **byte-unchanged**.

### V.6 Repair

| Layer | Change |
|---|---|
| `trainer.py` | `DERIVED_KEYS` declared; `to_dict()` docstring states the non-round-trippable contract; `require_match` validates the derived keys fail-closed |
| `test_stage1_training_resume.py` | `dataclasses.replace` — the authoritative constructor-level derivation — replaces `RunProvenance(**to_dict())` |
| both resume test files | `pytest.raises(Exception)` → `pytest.raises(TrainerContractViolation)` **plus the mutated field asserted in the message** |
| `test_stage1_provenance_contract.py` | **new, torch-free**: the contract and the lifecycle round trip, running in the ML-free venv on every run |

The `pytest.raises(Exception)` weakness is the reason this survived: it accepts
*any* failure, so a setup error is indistinguishable from the rejection under
test. The torch-free twin in `test_stage1_training_resume_state.py` had the same
weakness and has been tightened even though its construction was already
correct.

### V.7 Re-verification evidence (appended, not replacing V.2)

| Check | Result |
|---|---|
| All six foreign identities reach `verify_checkpoint` | **6/6**, each rejected on its own field |
| Every **10** constructor fields gated | **10/10** rejected with the field named |
| `RunProvenance(**to_dict())` raises `TypeError` | pinned as the **contract** |
| No production `RunProvenance(**…)` | AST-verified across `unmark/` and `scripts/` |
| Lifecycle round trip, dict + JSON | every identity field preserved exactly |
| Lifecycle round trip, real `torch.save`/`torch.load` | added, Colab-gated |
| Derived-key tamper / omission | refused |
| Full lightweight suite | **3 475 passed, 99 skipped, 0 failed** |

Because the six repaired cases live in a torch-gated module, they were also
executed **locally** by running the repaired function body outside that module
against the real `verify_checkpoint` — they touch no torch API — giving 6/6 here
in the ML-free venv. The authoritative torch evidence still comes from the next
Colab rerun.

### V.8 What is still NOT established

The real no-update smoke has **not** passed. It stopped at the resume gate, so
everything downstream remains unevidenced: **no PhoBERT load, no real forward,
no validation, no optimizer, no update, no training**. Stage 6 is
byte-unchanged, the prepared corpus was not touched, and official UIT-VSFC TEST
remains unreachable.

**STATUS: PRE-TRAIN REAL-SMOKE BLOCKER REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**FIRST REAL SMOKE PASSED EVERY CORPUS GATE AND STOPPED FAIL-CLOSED AT THE RESUME GATE**
**CASE A — TEST-CONSTRUCTION BUG; NO PRODUCTION PATH RECONSTRUCTS PROVENANCE FROM AN ARTIFACT**
**`to_dict()` IS ARTIFACT SERIALIZATION, NOT A CONSTRUCTOR ROUND TRIP — NOW STATED AND PINNED**
**ONE REAL GAP CLOSED: THE DERIVED WEIGHTS WERE SERIALIZED AND NEVER VALIDATED ON READ**
**ALL SIX FOREIGN IDENTITIES NOW REACH `verify_checkpoint`; ALL TEN FIELDS ARE GATED**
**NO SCIENTIFIC CONSTANT CHANGED; `decisions.md` BYTE-UNCHANGED; NO NEW DECISION REQUIRED**
**NO MODEL LOAD, NO FORWARD, NO OPTIMIZER, NO UPDATE, NO TRAINING; TEST STILL SEALED**
**THE REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM IT PASSED**

---

## W. SECOND REAL NO-UPDATE SMOKE — SCIENTIFIC ELIGIBILITY INVENTORY BLOCKER

**Revision 7.** The second real Colab no-update smoke ran at the committed §V
HEAD. It got further than the first: every corpus gate and every repaired test
passed. It then failed closed **before completing validation**, because the
pinned Vietnamese syllable inventory was not present in the fresh runtime.

### W.1 What the real rerun established

| Gate | Result |
|---|---|
| HEAD | `b84b4daac0f2be31266e171d3f56a71611a421e0` |
| Runtime | Python 3.13.15, Torch 2.11.0+cu128, Transformers 4.57.6, RTX PRO 6000 Blackwell, 176.88 GiB RAM |
| Exact HEAD / clean repository | **PASS** |
| Prepared corpus byte-exact restore | **PASS** |
| `COMPLETE.json` / membership digest | **PASS** |
| `tests/test_stage1_provenance_contract.py` | **29 passed** |
| `tests/test_stage1_training_resume.py` | **17 passed** |
| `tests/test_stage1_validation_measurement.py` | **22 passed** |
| Real descriptive corpus profile | **PASS** |
| **Real validation** | **FAIL CLOSED** |
| Exception | `unmark.corruption.eligibility.EligibilityUnresolved` |
| Policy requested / available | `SCIENTIFIC` / **`UNRESOLVED`** |
| Real validation forward | **NOT COMPLETED** |
| Optimizer | **NONE** |
| Updates | **ZERO** |
| Training | **NOT STARTED** |

**§V is confirmed by real evidence.** The 29 + 17 provenance and resume tests
that §V repaired all passed in the torch-enabled runtime — the six foreign-run
cases that previously died in setup now really execute.

### W.2 Classification: **A + C**

**Not B.** The inventory was **already scientifically locked**, exactly and
completely, by **D-B3A-001** (2026-08-19), which closed GAP-2 and D-B2-003:

| | |
|---|---|
| source | `all-vietnamese-syllables.txt` by `hieuthi` |
| gist | `0f5adb7d3f79e7fb67e0e499004bf558` |
| **immutable revision** | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| **sha256** | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| bytes | 116 290 |
| shape | 17 974 raw → 17 954 unique canonical → **2 489 unique stripped forms** (15 465 collisions) |
| license | **NO_EXPLICIT_LICENSE** |

**A — provisioning.** The raw list is deliberately **not committed**: "A public
gist is not a licence, so the raw list is **not committed**. It is fetched into
the git-ignored `.resources-cache/` … and everything scientific fails loudly
without it." A fresh Colab runtime therefore starts without it **by design**.
`active_eligibility_policy()` is *computed* from whether the inventory loads and
verifies, precisely so a missing cache re-arms the guard. **The exception the
smoke hit was the designed contract working, not a defect.**

**C — implementation defects around that locked artifact.** Three, all real:

| # | Defect | Severity |
|---|---|---|
| **W-1** | The check ran **after** the encoder was downloaded and resident. `execute_stage`, `smoke_check` and `validation_timing` all called `build_objective` before anything touched the inventory | operational |
| **W-2** | **D-S1A-008 was never implemented.** Its status line reads "**BLOCKING** for scientific Stage-1 training and the PRE-TRAIN audit", and it requires a scientific run to persist seven named inventory fields. `RunProvenance` recorded none of them | **MUST-FIX, scientific provenance** |
| **W-3** | The manifest declares three counts and a byte size under "*Expected shape (verified on load)*", but `load_manifest` parsed only `expected_entry_count`. The other two counts and `size_bytes` were **checked by nothing** | fail-closed gap |

### W.3 Was a wrong inventory able to reach training? — No

Traced before repairing. `load_inventory` hashes the cached bytes against the
pinned SHA-256 on every load, and `InventoryChecksumMismatch` **subclasses**
`InventoryUnavailable`, so `try_load_inventory` degrades a corrupted cache to
`None` → policy `UNRESOLVED` → `corrupt()` raises. The guard also fires **before**
`get_condition`, so every condition including `FULL` is gated, and no Stage-1
module ever passes `SELF_CHECK`. A wrong or absent inventory **cannot** silently
produce scientific corruption.

What it *could* do was report a corrupted cache as a *missing* one. The preflight
now distinguishes the two.

### W.4 The fetcher, audited statically (not executed for the audit)

| | |
|---|---|
| URL | `https://gist.githubusercontent.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558/raw/135a4d97…/all-vietnamese-syllables.txt` |
| Mutable "latest"? | **No** — the revision SHA is *in the path*; no `/raw/main/`, no `HEAD` |
| Hash verified before publication? | **Yes** — the write is unreachable unless the digest matches |
| Advances the pin? | **Never** — mismatch is a hard refusal with an explicit "this is a scientific spec change" message |
| Size guard | 8 MiB ceiling, 60 s timeout |
| Output | `.resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt` (git-ignored) |
| Parsing | `canon()` → `strip_to_base()` → `casefold()` → set membership |

**Reproducible and pinned, not dependent on mutable network state.** One
weakness found and repaired: publication used `path.write_bytes`, so a lost
runtime mid-write could leave a truncated file. It now uses the repository's
temp → fsync → replace → dir-fsync discipline, after the digest check.

### W.5 License boundary

Repository evidence (manifest `license_notes`) states no LICENSE file, no license
field, and no statement in the gist description were found. The architecture
separates the three cases deliberately:

| | |
|---|---|
| **Fetch as an external research input** | done, by the operator, into a git-ignored cache |
| **Redistribute inside this repository** | **avoided** — this is why only provenance is committed |
| **Redistribute inside released artifacts** | not done; the run artifact records *identity*, never content |

This audit gives no legal conclusion beyond what that evidence supports. From an
engineering standpoint the conclusion is unambiguous: the repository intentionally
avoids vendoring, so the reproducibility contract must be an **external-artifact
contract**, and the inventory was **not** vendored by this repair.

### W.6 Repair — all restoration of already-locked decisions

| Layer | Change |
|---|---|
| `unmark/stage1/preflight.py` **(new)** | `verify_scientific_inputs()` — fail-closed, **before** model load. Calls the authoritative `load_inventory`; adds no second verifier. Distinguishes missing / wrong-hash / wrong-shape. Refuses `SELF_CHECK`. Refuses an unresolved policy |
| `unmark/linguistics/inventory.py` | parses and verifies `size_bytes` and the two other declared counts — **W-3**, restoring the manifest's own claim |
| `unmark/stage1/trainer.py` | `RunProvenance.inventory` — **D-S1A-008's seven fields**, compared by `require_match`, so resume rejects inventory drift |
| `unmark/stage1/execute.py` | preflight before `build_objective` in `execute_stage` **and** `smoke_check`; provenance binds the *verified* identity |
| `scripts/stage1_pretrain_measurements.py` | preflight before `build_objective`; the verified identity is recorded in the report |
| `scripts/fetch_vietnamese_syllable_inventory.py` | atomic publication |
| `unmark/stage1/parallel.py`, `unmark/corruption/eligibility.py` | stale comments corrected — see W.7 and W.9 |

**`InventoryIdentity` is exactly D-S1A-008's seven fields and no more.**

An earlier revision of this section said the counts were "not compared against
any constant — because no decision locks one". **That was wrong for the counts**,
and it contradicted W-3 in this same section. D-B3A-001 *does* lock them
(`17,974 raw → 17,954 unique canonical → 2,489 unique stripped forms`), the
manifest declares them, and the repair makes `load_inventory` verify them
fail-closed. The correct statement separates four distinct roles:

| Role | Quantities | Where enforced |
|---|---|---|
| **1. Identity fields** — persisted in `RunProvenance`, compared by `require_match`, so resume rejects drift | the **seven** D-S1A-008 fields: `inventory_schema_version`, `source_name`, `source_author`, `source_revision`, `sha256`, `size_bytes`, `license_status` | `RunProvenance.inventory` |
| **2. Raw artifact identity** — binds the exact bytes | `source_revision` + `sha256` | `load_inventory` hashes the cached bytes on every load |
| **3. Locked shape assertions** — verified fail-closed on load, **not** duplicated into `RunProvenance` because D-S1A-008 does not require them | `size_bytes` (**also** an identity field), `expected_entry_count` **17 974**, `expected_unique_canonical_entry_count` **17 954**, `expected_unique_stripped_form_count` **2 489** | `load_inventory` — all four, since W-3 |
| **4. Report-only derived evidence** — recorded so two runs can be compared; **not** an identity field and **not** compared against any constant, because no decision locks one | `parsed_membership_digest` | preflight `report` only |

`collisions_after_stripping` (**15 465**, also stated in D-B3A-001) belongs to
none of these categories on its own: it is `unique_canonical − unique_stripped`,
so once **3** verifies 17 954 and 2 489 it is *entailed* and cannot differ. It is
reported for readability, not separately checked.

`size_bytes` deliberately appears in both **1** and **3**: D-S1A-008 names it as a
field a run must persist, and the manifest declares it as shape to verify. The
three counts appear **only** in **3** — locked and checked, but not promoted into
the run identity, because `sha256` already makes any different byte sequence a
different identity and D-S1A-008 does not list them.

The only genuinely unlocked quantity is the **parsed-membership digest**. It stays
report-only: promoting it would invent a scientific constant the decision log never
locked, which this audit must not do.

**Inspection confirms the implementation already matches this distinction** —
`InventoryIdentity` has exactly seven fields, `load_inventory` raises on any of the
four shape values, and `parsed_membership_digest` is produced into the report and
compared to nothing. No code behaviour was changed by this correction.

### W.7 Stage 6 does **not** depend on the inventory — verified, no rerun

`unmark/stage1/parallel.py` builds a classifier in each worker, and its comment
claimed *"dropping it would change chunk boundaries"*. **That comment was wrong**,
and it is the kind of wrong that would have implied the prepared corpus is
inventory-dependent. Verified two ways:

* **structurally** — `classifier` is a parameter of `safe_cut_offsets` and is
  **never read in its body** (AST);
* **empirically** — over 2 000+ adversarial strings, the cut set is identical for
  `None`, for a classifier claiming *everything* is Vietnamese, and for one
  claiming *nothing* is.

This matches **D-S1B-013**, which requires the cut predicate to be lexicon-free:
membership would permit a cut **inside a genuine Vietnamese word**. Where a cut
may land is orthographic (unit boundaries, maximal letter runs), not lexical.

> **NO Stage-6 rerun is required. The prepared corpus remains authoritative and
> was not touched.** The inventory is needed for SCIENTIFIC corruption and
> classification, never for the already-produced chunk bytes.

The misleading comment has been corrected.

### W.8 Reconciling Audit 030's earlier "No UNRESOLVED MISMATCH"

**The earlier statement was too strong, and is corrected here rather than
explained away.**

What §A–§Q checked was **implementation semantics**: proposal §4.3's rule against
`classify.py`. That comparison was and remains **correct** — the membership rule
is an exact match, and D-B3A-001 closed GAP-2 in code.

What it did **not** check was whether the mandatory **external artifact** that
rule consumes would be **present** in a fresh runtime, nor whether **D-S1A-008** —
a decision whose own status line says "**BLOCKING for scientific Stage-1 training
and the PRE-TRAIN audit**" — had been implemented. It had not. A repository-wide
pre-train gate should have caught an unimplemented blocking decision; that is the
miss, and it is squarely within the scope §A–§Q claimed.

The precise correction:

> *Implementation semantics are complete and match the proposal. The required
> external scientific artifact was **not provisioned** in a fresh runtime, and the
> blocking decision requiring its identity to be persisted (D-S1A-008) was **not
> implemented**. "No UNRESOLVED MISMATCH" was true of the code and false of the
> pre-train readiness gate.*

§A–§Q's wording is preserved unchanged above.

### W.9 Other stale claims corrected

`unmark/corruption/eligibility.py` still described GAP-2 as open — "the Vietnamese
syllable inventory … does not exist in this repository", and
`VIETNAMESE_SYLLABLE_INVENTORY` was documented "**Not implemented.**" Both have
been false since D-B3A-001. Corrected to describe the real state: implemented, and
`UNRESOLVED` is now a *deployment* condition, not an open scientific gap.

### W.10 The real corpus profile

Recorded as **descriptive evidence only** in
[`docs/experiments/stage1-prepared-corpus-profile-result.md`](../experiments/stage1-prepared-corpus-profile-result.md).
Chunks: train **2 621 624**, dev **11 443**. Chunks per parent: dev mean 2.29
(p50 1, p99 20, max 481); train mean 2.35 (p50 1, p99 22, max 2 479).

Token lengths, recomputed under the locked tokenizer over **all 11 443 dev** and a
deterministic **20 000** train sample:

| Stream | mean | p50 | p75 | p99 | max |
|---|---|---|---|---|---|
| dev reference | 164.20 | 211 | 233 | 256 | 256 |
| dev RAW_BASE | 181.06 | 253 | 256 | 256 | 256 |
| train reference | 164.81 | 211 | 230 | 256 | 256 |
| train RAW_BASE | 182.71 | 254 | 256 | 256 | 256 |

**`over_max_length`: 0.**

This **falsifies** the informal inference that 2.35 chunks/document implies most
training chunks are short. The median document yields one chunk, but that chunk is
commonly **at or near the 256-token ceiling** — median RAW_BASE is 253–254, and
p75 onward is exactly 256. **No chunking or `MAX_LENGTH` change was made.**

### W.11 The Colab provisioning contract — for review, not executed

Claude did **not** access Drive and did **not** run the fetcher for this audit;
the local cache was already provisioned and was only *verified*
(`--verify-only` → present and matching, no network). The cell below is the
proposed contract to review **before** the next rerun:

```bash
# 1. Provision the pinned inventory (network; ~116 KB). Idempotent: if a valid
#    cache is already restored from Drive this verifies and exits 0 without
#    downloading. Refuses on any checksum mismatch and never advances the pin.
python scripts/fetch_vietnamese_syllable_inventory.py

# 2. Verify without network, and fail the cell if it does not match the pin.
python scripts/fetch_vietnamese_syllable_inventory.py --verify-only

# 3. Persist to Drive so later fresh runtimes need no network.
#    NOTE: .resources-cache/ is git-ignored; this copies ONLY the verified file.
mkdir -p "$DRIVE/unmark-resources/vietnamese-syllables"
cp .resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt \
   "$DRIVE/unmark-resources/vietnamese-syllables/"

# 4. On a later runtime, restore BEFORE step 2 and let --verify-only be the gate:
mkdir -p .resources-cache/vietnamese-syllables
cp "$DRIVE/unmark-resources/vietnamese-syllables/all-vietnamese-syllables.txt" \
   .resources-cache/vietnamese-syllables/
```

The restored copy is never trusted on provenance: step 2 re-hashes it against the
pinned SHA-256, and `verify_scientific_inputs` re-verifies again before model
load. Redistribution boundary unchanged — the file goes to the operator's own
Drive, not into git and not into a released artifact.

### W.12 What is still NOT established

The real no-update smoke has **not** passed. Validation never completed, so there
is still **no real forward evidence**, no optimizer, no update, no training. This
section does not claim otherwise.

**STATUS: PRE-TRAIN INVENTORY PROVISIONING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**CLASSIFICATION A + C — THE ARTIFACT WAS ALREADY LOCKED BY D-B3A-001; PROVISIONING AND BINDING WERE NOT**
**THE `EligibilityUnresolved` FAILURE WAS THE DESIGNED CONTRACT WORKING, NOT A DEFECT**
**W-2 IS A MUST-FIX: D-S1A-008 WAS BLOCKING AND UNIMPLEMENTED; PROVENANCE NOW BINDS THE INVENTORY**
**AUDIT 030's EARLIER "NO UNRESOLVED MISMATCH" WAS TOO STRONG AND IS CORRECTED, NOT RATIONALISED**
**STAGE 6 IS LEXICON-FREE — VERIFIED STRUCTURALLY AND EMPIRICALLY; NO RERUN; CORPUS UNTOUCHED**
**NO NEW SCIENTIFIC DECISION — NO INVENTORY WAS CHOSEN, NO CONSTANT CHANGED, NOTHING VENDORED**
**NO MODEL LOAD, NO FORWARD, NO OPTIMIZER, NO UPDATE, NO TRAINING; TEST STILL SEALED**
**A THIRD REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM IT PASSED**

---

## X. THIRD REAL NO-UPDATE SMOKE — TRUNCATION-CONTRACT WIRING FAILURE

**Revision 8.** The third real Colab no-update smoke ran at the committed §W
HEAD. §W's inventory repair **held completely** — provisioning, verification and
Drive persistence all passed, and the preflight ran before model load. The run
then failed closed in condition preparation on an invalid argument that the
measurement tool had been passing since §U.

### X.1 What the real rerun established

| Gate | Result |
|---|---|
| HEAD | `ebbe5534a7cb8dd759642d2e9e6f6165aafde21d` |
| Exact HEAD / clean repository | **PASS** |
| Pinned inventory fetch | **PASS** |
| Inventory immutable revision | **PASS** |
| Inventory SHA-256 | **PASS** |
| Inventory size | **PASS** |
| Inventory shape counts | **PASS** |
| `--verify-only` | **PASS** |
| Verified inventory persisted to Drive | **PASS** |
| Prepared corpus byte-exact restore | **PASS** |
| `COMPLETE.json` / membership digest | **PASS** |
| `tests/test_stage1_inventory_preflight.py` | **31 passed** |
| `tests/test_stage1_provenance_contract.py` | **30 passed** |
| `tests/test_stage1_training_resume.py` | **17 passed** |
| `tests/test_stage1_validation_measurement.py` | **22 passed** |
| **Real validation** | **FAIL CLOSED** |
| Exception | `AttributeError: 'NoneType' object has no attribute 'check'` |
| Offending wiring | `prepare_condition_batch(..., truncation=None, ...)` |
| Real validation completed | **NO** |
| Parameter update | **ZERO** |
| Training | **NOT STARTED** |

### X.2 The contract, re-derived

| Question | Answer |
|---|---|
| What is the `truncation` argument? | a **`TruncationPolicy`** (`unmark/stage1/contracts.py`) |
| Is `None` ever valid? | **No.** The parameter is typed `TruncationPolicy` with **no default** on both `prepare_condition_batch` and `prepare_with_condition`. "Intentionally unbounded" is an explicit *object*, `TruncationPolicy.unbounded()` — the class exists precisely so an implicit `None` cannot select a policy nobody chose |
| What does `.check(length, what)` do? | returns `True` to keep the example; returns `False` under `SKIP`; **raises `Stage1ContractViolation`** under `FAIL` |
| How does the locked rule map onto it? | `TruncationPolicy(max_length=MAX_LENGTH, on_overflow=OverflowBehaviour.FAIL)` — 256 / FAIL, and truncation is **not offered as a behaviour at all** |
| Authoritative instance | **`unmark/stage1/execute.py:47`** — `TRUNCATION`, one definition, repository-wide |
| Exactly one implementation? | **Yes.** One `TruncationPolicy` class, one Stage-1 instance, one `MAX_LENGTH` |

**The locked science is unchanged: `MAX_LENGTH = 256`, truncation forbidden,
overflow = FAIL.** Nothing here required a scientific decision — see §X.7.

### X.3 Every caller, traced

| Caller | Supplied | Real data? | Kind |
|---|---|---|---|
| `execute.py:140` — validation prep in `execute_stage` | **`TRUNCATION`** | yes | scientific |
| `execute.py:183`, `:213` — `train_run` (initial + continuation) | **`TRUNCATION`** | yes | scientific |
| `execute.py:325` — `smoke_check` | **`TRUNCATION`** | yes | scientific |
| `validation.py:87`, `trainer.py:534`, `data.py:274` | threaded through | yes | scientific |
| **`scripts/stage1_pretrain_measurements.py:415`** | **`None`** | **yes** | **measurement** |

> **Production training, production validation and the smoke were all correct.**
> `truncation=None` existed at exactly **one** site in the entire repository, and
> it was the measurement tool.

**Classification: A + D.** A measurement-tool-only wiring defect (**A**), plus the
API-contract/test-boundary ambiguity that let it survive (**D**). **Not B** — no
production path could reach the failure. The defect was introduced by the §U
measurement repair.

### X.4 Why 22 measurement tests passed while the real call died

The most important finding in this section. Three seams, each individually
reasonable, which together left the defect completely unobserved:

| # | Boundary | Effect |
|---|---|---|
| 1 | **`validation_timing` was never executed by any test.** Its three appearances in the suite are `inspect.getsource` / AST reads | the wiring was *read*, never *run* |
| 2 | **`prepare_condition_batch` was never executed by any test.** `test_stage1_measurement_contract.py` asserts only that its *name* appears in `validation_timing`'s call graph | proved the call exists, not that its arguments are valid |
| 3 | **The runtime fixture substituted the whole preparation stage.** `evaluated` builds `prepared = {c: list(range(10))}` — plain integers — and starts at `evaluate`, **downstream** of the defect | every "real forward" test ran on hand-built input |

So the suite verified that `validation_timing` *mentions* `prepare_condition_batch`,
and separately that `evaluate` works on synthetic input. **The seam between them —
the actual arguments — was never exercised by anything.** A structural assertion
that a function is called is not evidence that calling it works.

This is the same defect class §V recorded (`pytest.raises(Exception)` accepting any
failure) in a different disguise: a test that cannot fail for the reason it exists.

### X.5 Repair

`validation_timing` now imports and passes **`unmark.stage1.execute.TRUNCATION`** —
the same object `execute_stage` passes. No measurement-specific policy, no
`truncation=None` fallback, no second `MAX_LENGTH`, no truncation, no silent skip.
AST-verified that the tool constructs no `TruncationPolicy` and assigns no
`MAX_LENGTH`/`TRUNCATION` of its own.

**New `tests/test_stage1_validation_preparation.py` (torch-free, 14 tests)** runs
the **real** `prepare_condition_batch` and the **real** `prepare_with_condition`
with the **real** `TRUNCATION` and the **real** resolved classifier, proving:
all four conditions prepare; `≤ 256` succeeds with lengths **unclipped**; `> 256`
raises `Stage1ContractViolation`; both `truncation.check` calls exist and guard
`reference sequence` **and** `base sequence` (the RAW_BASE grid); `SKIP` surfaces
as a hard `ValidationContractViolation` rather than a shorter batch; and
`truncation=None` reproduces the exact third-smoke `AttributeError`.

**New end-to-end tests** in `tests/test_stage1_validation_measurement.py` execute
the **real `validation_timing`** with an injected tiny model, so the real
preparation seam runs — including a **runtime spy** that wraps (rather than
replaces) `prepare_condition_batch` and asserts the captured argument **is**
`execute.TRUNCATION`. With `truncation=None` these fail with the real
`AttributeError`. The regression test that pins the argument uses **AST**, not a
source-text search — a text match would trip over the tool's own comment
explaining the defect.

### X.6 Model-load order at the moment of failure

Traced in `validation_timing`, by line:

| Line | Step |
|---|---|
| 389 | **inventory preflight** — §W's repair, before any model |
| 407 | **PhoBERT tokenizer + encoder weights downloaded and instantiated** |
| 408 | `verify_model_contract` |
| 409 | **moved to CUDA** |
| 411–414 | classifier built, parameters hashed |
| **419** | **condition preparation — the third smoke died here** |
| 438 | `evaluate()` — the first real **forward**, **never reached** |

So the encoder **was** downloaded, instantiated and resident on the GPU before the
failure. **No forward pass, no optimizer, no backward, no parameter update.**

**Classified as harmless operational ordering, and deliberately not reordered.**
Preparation needs the **tokenizer**, which `build_objective` returns together with
the encoder; the weight load is incidental to that call. Splitting it would be a
redesign with no existing contract requiring it. The ordering that *does* matter —
**inventory preflight before model load** — is intact and re-verified.

### X.7 No decision required

This restores the already-locked contract (`MAX_LENGTH = 256`, no truncation,
`ON_OVERFLOW = FAIL`) at a call site that violated it. The meaning of the argument
was never underspecified: it is a required, non-optional, typed parameter with a
named constructor for the unbounded case. **No decision-log entry was created and
`docs/spec/decisions.md` is byte-unchanged.**

### X.8 Preserved

§W re-verified intact: revision `135a4d97…`, sha256 `78eeb840…`, 116 290 bytes,
17 974 / 17 954 / 2 489 counts, `InventoryIdentity` **exactly seven** D-S1A-008
fields, `parsed_membership_digest` report-only, preflight before model load,
`SELF_CHECK` rejected, nothing vendored, no decision changed.

The third smoke again confirmed the prepared corpus: **2 633 067** chunks
(train **2 621 624**, dev **11 443**), **1 118 224** parents, 5 000 dev parents,
`overflow_count 0`, `base_invariance_violations 0`,
`parents_spanning_both_partitions 0`, byte-exact restore. **No Stage-6 rerun. The
prepared corpus remains authoritative and was not touched.**

### X.9 What is still NOT established

Validation still has **not** completed on real data. There is **no real forward
evidence**, no optimizer, no update, no training. A **fourth** real no-update
smoke is required.

**STATUS: PRE-TRAIN TRUNCATION-WIRING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**CLASSIFICATION A + D — MEASUREMENT-ONLY WIRING DEFECT PLUS THE TEST BOUNDARY THAT HID IT**
**PRODUCTION TRAINING, VALIDATION AND SMOKE ALL PASSED `TRUNCATION` CORRECTLY — NOT AFFECTED**
**`truncation=None` EXISTED AT EXACTLY ONE SITE IN THE REPOSITORY, INTRODUCED BY THE §U REPAIR**
**22 TESTS PASSED BECAUSE NOTHING EVER EXECUTED `validation_timing` OR `prepare_condition_batch`**
**THE REAL PREPARATION SEAM IS NOW EXERCISED END TO END, INCLUDING A RUNTIME ARGUMENT SPY**
**MAX_LENGTH 256 / NO TRUNCATION / ON_OVERFLOW=FAIL UNCHANGED; NO NEW DECISION**
**PhoBERT WAS RESIDENT ON GPU BUT NO FORWARD, NO OPTIMIZER, NO UPDATE OCCURRED**
**§W INVENTORY REPAIR INTACT; STAGE 6 UNCHANGED; PREPARED CORPUS AUTHORITATIVE; TEST SEALED**
**A FOURTH REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM ONE PASSED**

---

## Y. FOURTH REAL NO-UPDATE SMOKE — DEVICE-CONTRACT FAILURE

**Revision 9.** The fourth real smoke ran at the committed §X HEAD. Every corpus,
inventory and preparation gate passed. It stopped at the **validation-measurement
test gate**, where the §X real-seam tests — which now genuinely execute the
preparation and evaluation path — surfaced a cross-device error.

**The test failure was a TRUE POSITIVE.** It is the reason this section is not
about a broken fixture.

### Y.1 What the real rerun established

| Gate | Result |
|---|---|
| HEAD | `6bd6452dabebb266e0c4bb7561dad7687d64ffd2` |
| Exact HEAD / clean repository | **PASS** |
| Pinned inventory restore from Drive | **PASS** |
| Inventory `--verify-only` | **PASS** |
| Prepared corpus byte-exact restore | **PASS** |
| `COMPLETE.json` / membership digest | **PASS** |
| `tests/test_stage1_validation_preparation.py` | **14 passed** |
| `tests/test_stage1_inventory_preflight.py` | **31 passed** |
| `tests/test_stage1_provenance_contract.py` | **30 passed** |
| `tests/test_stage1_training_resume.py` | **17 passed** |
| `tests/test_stage1_validation_measurement.py` | **2 failed, 23 passed** |
| Failing tests | `test_validation_timing_runs_the_real_preparation_end_to_end`, `test_the_tool_passes_the_production_truncation_object_at_runtime` |
| Exact error | `RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0` |
| Real validation command | **NOT RUN** |
| Real validation forward | **NOT RUN IN FOURTH SMOKE** |
| Optimizer | **NONE** |
| Parameter update | **ZERO** |
| Training | **NOT STARTED** |

Both failures occurred **after** passing through the §X preparation seam, inside
`evaluate`. §X's repair therefore held; the new test reached further than any
previous run and found the next real defect.

### Y.2 The device contract, re-derived

| Question | Answer |
|---|---|
| Where are parameters moved to CUDA? | **`scripts/stage1_pretrain_measurements.py:409`** — `unmark_encoder.to(device)`. This was the **only** `.to(device)` in the entire Stage-1 stack |
| Where are input ids moved? | **Nowhere.** `collate_stage1_batch` builds CPU tensors with no device argument |
| Where are attention masks moved? | **Nowhere**, same call |
| Who owns transfer? | **The batch assembler** — by implication, never by statement |
| Explicit or incidental? | **Incidental.** No decision, spec or docstring named an owner |
| Do the objective methods move tensors? | **No.** `reference_representation` passes `input_ids` straight to the encoder; `adapted_representation` passes them to `UnmarkEncoder`. AST-verified: no `.to(...)` in either |
| Does the model wrapper move them? | **No.** `unmark/modeling/adapter.py:621`'s `.to(derived.device)` is a comparison convenience, not input transfer |

So the implemented contract is: **the objective never moves its inputs; whoever
assembles the batch must place it on the model's device.** Nothing did.

### Y.3 Classification: **B + D**, not A

**Not A (fixture-only).** The injected `_TinyObjective` mirrors the production
interface *exactly* — it does not move its inputs, because `Stage1Objective`
does not either. That fidelity is precisely why it reproduced a real failure.
"Fixing" the fixture with `ids.to("cuda")` would have made the test green and left
the real `--validation --require-cuda` run failing **identically** on real PhoBERT,
because `evaluate` would still have handed CPU tensors to a CUDA encoder.

**B — a real measurement/evaluator device-wiring defect.** The tool moved the
model to CUDA and left every batch on the CPU. On any CUDA host this fails, with
the tiny model or with real PhoBERT alike.

**D — a contract that existed only by accident.** Three production sites assemble
batches (`evaluate`, `train_run`, `smoke_check`) and none placed them; nothing
documented who should.

**Could real production have hit this?** Not as a crash: `execute_stage` never
moves the model to an accelerator either, so model and batch were consistently on
the CPU. The cross-device error was reachable only where something *did* move the
model — the measurement tool.

> **Recorded as an open operational item, not repaired here.** `execute_stage`
> performs **no device placement at all**, so Stage-1 training as wired today
> would run on the CPU. That is not a crash and not a scientific question, but
> selecting a training device is an **operational contract nobody has chosen**
> (which device, from a flag, auto-detected?). This audit does not choose it. It
> must be decided before a real training run; it does not block the no-update
> smoke, which owns its own placement.

### Y.4 Why this survived local review

The §X tests were de-risked locally **without torch**: the preparation path was
executed for real, but everything downstream of `collate_stage1_batch` was
unreachable in an ML-free venv. The defect lives strictly in the interaction
between a **CUDA-resident model** and a **CPU-resident batch**, and neither
condition can exist on a CPU-only host without torch.

**This is a hardware-gated test-fixture blind spot.** The honest statement is that
the new fixture's CUDA execution had never been exercised before commit, and the
audit said so at the time. Recorded so the pattern is visible: CPU reasoning about
a CUDA-only failure mode proves nothing, and the mitigation is a contract test that
runs wherever the hardware exists, not a claim that the code looks right.

### Y.5 Repair — the shared layer, not the fixture

**No fixture was changed.** The repair is in the shared authoritative layer, and
the two failing tests pass because production is now correct.

`unmark/stage1/data.py` gains the boundary, beside the collator that creates the
tensors:

* `module_device(module)` — the device of the module's parameters, **derived**,
  with a CPU fallback for a parameterless module;
* `batch_to_device(batch, device)` — moves every tensor, passes non-tensors
  (`sample_ids`, `corruption_rates`, `corruption_scopes`) through unchanged, and
  is a no-op when the batch is already there.

All **three** production batch assemblers now apply it:

| Site | Change |
|---|---|
| `validation.evaluate` | `batch_to_device(collate_stage1_batch(...), module_device(objective))` |
| `trainer.train_run` | same |
| `execute.smoke_check` | same |

No hard-coded `cuda`, no environment variable, no global default device, no
test-only branch in production, and **no scientific behaviour change** — on CPU
every call is a no-op, so the existing CPU path is bit-identical.

`unmark/stage1/objective.py` and `unmark/modeling/` are **byte-unchanged**: the
objective's "I do not move my inputs" half of the contract is deliberately
preserved, and is now pinned by a test so a future change to it has to be
deliberate.

### Y.6 Tests

`tests/test_stage1_device_contract.py` — **torch-free, 6 tests, runs in the
ML-free venv**: all three assemblers route through `batch_to_device` **and**
`module_device`; no hard-coded device anywhere in `unmark/stage1`; the objective
still does not move its own inputs; the measurement tool places the model and does
not assemble batches.

`tests/test_stage1_device_contract_runtime.py` — **torch-gated, 7 tests**
(~~8~~ — a miscount in the original wording of this section, corrected in §Z; the
file has always contained seven): a `RecordingCore` encoder captures the device of
every tensor it receives, through the **real** `evaluate` over **real** prepared
examples. It proves the helpers behave, that CPU+CPU works, and that **every**
tensor the encoder sees — `input_ids` *and* `attention_mask`, on both the reference
*and* adapted paths — is on the model's device. Two tests are **CUDA-gated**: one
asserts a CUDA objective receives CUDA tensors from a CPU-prepared batch (the
fourth-smoke failure inverted into a passing contract), and one proves that test is
not vacuous by showing the raw cross-device call still raises.

**Seven tests, nine semantic claims.** Test count and assertion count are not the
same number, and the mapping is explicit so the difference is not mistaken for a
gap:

| # | Semantic claim | Covered by |
|---|---|---|
| 1 | `module_device` derives the device from parameters | `test_module_device_reads_the_parameters` |
| 2 | parameterless module falls back to CPU | `test_module_device_falls_back_to_cpu_without_parameters` |
| 3 | `batch_to_device` moves tensors and preserves non-tensors (and dtype) | `test_batch_to_device_moves_tensors_and_passes_non_tensors_through` |
| 4 | CPU objective + CPU batch works, and the encoder really ran | `test_cpu_objective_with_cpu_batch_works` |
| 5 | `input_ids` on the model's device | `test_every_tensor_the_encoder_sees_is_on_the_models_device` |
| 6 | `attention_mask` on the model's device | *(same test)* |
| 7 | **reference** path obeys the contract | *(same test)* — `evaluate` calls `reference_representation` and `RecordingObjective` routes it into the recorder |
| 8 | **adapted** path obeys the contract | *(same test)* — `evaluate` calls `adapted_representation` twice per batch, also routed into the recorder |
| 9 | CUDA objective receives CUDA tensors from a CPU-prepared batch, and the check is non-vacuous | `test_a_cuda_objective_receives_cuda_tensors_from_a_cpu_prepared_batch` + `test_removing_the_transfer_really_would_fail` |

Claims 5–8 are one test because they are one fact: the recorder captures **every**
call the encoder receives, from **both** representation paths, and asserts each
tensor's device. No eighth test was added, because none of the nine claims was
missing.

Split into two files deliberately: a module-level `importorskip` would have skipped
the structural half too, which is how an earlier repair lost its torch-free
coverage (§V).

### Y.7 §X's seam is intact

The §X arrangement is preserved exactly — **real** `validation_timing` → **real**
`prepare_condition_batch` → **real** `prepare_with_condition` → **real** `evaluate`.
Preparation is **not** back to `{condition: list(range(...))}`, and the runtime spy
proving `truncation is execute.TRUNCATION` remains. Those tests now pass on CPU and
CUDA alike, unchanged.

### Y.8 No decision required

Device placement changes no scientific value: `MAX_LENGTH`, the conditions, the
seeds, the grids, the precision and the architecture are untouched, and the CPU
path is a no-op. The ownership question was *undocumented*, not *contested* — one
implemented behaviour existed (the objective does not move inputs) and the repair
states it and completes it. **No decision-log entry was created; `docs/spec/decisions.md`
is byte-unchanged.** The one genuinely unchosen item — which device *training*
should use — is recorded in Y.3 as an open operational item rather than decided here.

### Y.9 Preserved

**§V** — `DERIVED_KEYS` intact, derived-lambda consistency, foreign-run rejection,
checkpoint identity. **§W** — revision `135a4d97…`, sha256 `78eeb840…`,
116 290 bytes, 17 974 / 17 954 / 2 489, seven-field `InventoryIdentity`,
parsed-membership digest report-only, preflight before model load, `SELF_CHECK`
rejected, nothing vendored. **§X** — `execute.TRUNCATION`, `MAX_LENGTH 256`,
`ON_OVERFLOW FAIL`, no truncation, real preparation integration intact, no
`truncation=None` on the measurement path.

Stage 6, `protocol.py`, `contracts.py`, `objective.py`, `unmark/modeling/`, the
prepared corpus and the inventory manifest are all **byte-unchanged**. Official
UIT-VSFC TEST remains sealed.

### Y.10 What is still NOT established

The fourth smoke **never ran** `scripts/stage1_pretrain_measurements.py --validation
--require-cuda`; it stopped at the test gate. There is still **no real validation
forward evidence**, no optimizer, no update, no training. A **fifth** real no-update
smoke is required.

**STATUS: PRE-TRAIN DEVICE-CONTRACT REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**CLASSIFICATION B + D — A REAL DEVICE-WIRING DEFECT, NOT A BROKEN FIXTURE**
**THE FAILING TEST WAS A TRUE POSITIVE: THE FIXTURE FAITHFULLY MIRRORS THE PRODUCTION INTERFACE**
**"FIXING" IT WITH `ids.to("cuda")` WOULD HAVE LEFT REAL PhoBERT VALIDATION FAILING IDENTICALLY**
**ONE SHARED BOUNDARY: `evaluate`, `train_run` AND `smoke_check` NOW ALL PLACE THE BATCH**
**DEVICE IS DERIVED FROM THE MODULE — NO HARD-CODED cuda, NO ENV VAR, NO GLOBAL DEFAULT**
**CPU PATH IS A NO-OP; `objective.py` AND `unmark/modeling/` ARE BYTE-UNCHANGED**
**OPEN OPERATIONAL ITEM: `execute_stage` PLACES NO MODEL AT ALL — TRAINING DEVICE UNCHOSEN**
**NO SCIENTIFIC CONSTANT CHANGED; NO NEW DECISION; §V/§W/§X ALL INTACT**
**NO FORWARD, NO OPTIMIZER, NO UPDATE, NO TRAINING; PREPARED CORPUS AND STAGE 6 UNTOUCHED**
**A FIFTH REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM ONE PASSED**

---

## Z. FIFTH REAL SMOKE — DEVICE RUNTIME GATE PROBE, AND A COUNT CORRECTION

**Revision 10.** The fifth smoke reached the device runtime gate on a fresh CUDA
runtime and **the device contract passed on real hardware**. The run stopped only
because the orchestrator had been given an expected summary string that did not
match reality: §Y.6 said the runtime device file held **8** tests. It holds **7**.

**This is not a fifth-smoke scientific failure, and not a code failure.**

### Z.1 The probe

| | |
|---|---|
| HEAD | `9651610708dcf038fdc14a81a8338d839ad77ea3` |
| Fifth smoke reached | the **device runtime gate** |
| AST top-level tests in `tests/test_stage1_device_contract_runtime.py` | **7** |
| `pytest --collect-only` | **7 tests collected** |
| **Real CUDA execution** | **7 passed, 0 failed, 0 skipped** |
| Orchestrator | **FAIL-CLOSED only because it expected the incorrect string `"8 passed"`** |
| Production / device code | **NOT FAILED** |
| Full real validation | **NOT RUN** |
| Optimizer | **NONE** |
| Parameter update | **ZERO** |
| Training | **NOT STARTED** |

`0 skipped` is itself evidence: both CUDA-gated tests and both inventory-gated
tests actually executed, so the cross-device assertions in §Y ran for real rather
than being skipped into a false green.

### Z.2 Classification: **A — a counting error in prose, not a coverage gap**

Verified rather than assumed. The seven functions carry **no** `parametrize`
decorator, so seven definitions collect as exactly seven tests — matching the
probe. Each of the nine semantic claims §Y makes was mapped to the test that
carries it; the mapping table is now in **§Y.6**. Claims 5–8 (`input_ids`,
`attention_mask`, reference path, adapted path) are one test because they are one
fact: `evaluate` calls `reference_representation` once and `adapted_representation`
twice per batch, `RecordingObjective` routes **both** into the recording encoder,
and the test asserts the device of **every** captured tensor.

**Every claimed semantic requirement is present. No eighth test was added**, and no
test was altered to manufacture one — that would have been redundant coverage
existing only to make a wrong number true.

### Z.3 What was corrected

* **§Y.6**: "torch-gated, **8** tests" → "torch-gated, **7** tests", with the
  original figure struck rather than erased, plus the nine-claim mapping table.
* No other count in this audit derived from the mistaken number. The companion
  claim "torch-free, **6** tests" was re-checked and is **correct** — four
  functions, one parametrized three ways, six collected. Every other "8" in this
  document is unrelated historical evidence (§V's `6 failed, 8 passed`,
  `test_stage1_pretrain_audit.py`'s 8) and is preserved untouched.
* The fifth-smoke orchestrator's expected string must be updated to `7 passed`
  before the next run.

### Z.4 Boundaries

**No production code changed.** No test changed. §V, §W, §X and §Y's findings and
evidence are untouched — the fourth-smoke failure record in §Y stands exactly as
written, and its device repair is unaffected and now confirmed on real CUDA
hardware.

Local full suite at this HEAD: **3 527 passed, 100 skipped, 0 failed** — measured,
not adjusted. The seven runtime device tests are among the local skips (no torch in
the ML-free venv) and are the ones the CUDA probe executed.

`docs/spec/decisions.md` byte-unchanged. No scientific constant touched. No
Audit 031.

### Z.5 Still outstanding

The fifth smoke did **not** run `--validation --require-cuda` to completion. There
is still **no real validation forward evidence**, no optimizer, no update, no
training. The **open operational item from §Y.3 stands**: `execute_stage` performs
no device placement, so the *training* device remains unchosen.

**STATUS: PRE-TRAIN DEVICE TEST-COUNT CONSISTENCY PASS — SEVEN TESTS ARE COMPLETE COVERAGE**
**THE DEVICE CONTRACT PASSED ON REAL CUDA HARDWARE: 7 PASSED, 0 FAILED, 0 SKIPPED**
**THE ONLY DEFECT WAS A PROSE MISCOUNT IN §Y.6; NO CODE AND NO TEST WAS AT FAULT**
**NINE SEMANTIC CLAIMS MAP ONTO SEVEN TESTS — NO EIGHTH TEST WAS ADDED OR NEEDED**
**§Y's FOURTH-SMOKE FAILURE RECORD PRESERVED EXACTLY; §V/§W/§X UNTOUCHED**
**FULL REAL VALIDATION STILL NOT RUN; NO OPTIMIZER, NO UPDATE, NO TRAINING**
**A FIFTH-SMOKE RERUN IS STILL REQUIRED AFTER THE ORCHESTRATOR EXPECTS `7 passed`**

---

## AA. FIRST FULL REAL VALIDATION PASS, AND A RUNNER-SMOKE CLI CONTRACT FAILURE

**Revision 11.** Two events at the same HEAD, recorded separately because they
stand differently. The first is the **largest piece of positive evidence this
audit has produced**. The second is a real repository bug, found afterwards, in a
different command.

### AA.1 THE FULL FOUR-CONDITION REAL VALIDATION **PASSED**

At `a5da53805498a12ed64ffa28a6a13232dc8e4b1b`, on a fresh CUDA runtime:

```
scripts/stage1_pretrain_measurements.py --validation --require-cuda \
  --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
  --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
  --revision        01daacda68afe13d83023d16ec647239e344a1e6
```

**completed successfully — status PASS.**

| | |
|---|---|
| Encoder | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` |
| Device / GPU | **CUDA** — NVIDIA RTX PRO 6000 Blackwell Server Edition |
| torch / CUDA / transformers | 2.11.0+cu128 / 12.8 / 4.57.6 |
| Precision | **fp32** |
| Dev chunks | **11 443** |
| Conditions | **FULL / P50 / P100 / STRIP_ALL** |
| Batch size / batches per condition | **128** / **90** |
| Validation corruption seed | **19225** |
| **Forward passes** | **810** (reference 360, adapted 450) |
| Encoder trainable parameters | **0** |
| Adapter trainable parameters | **3 551 232** in **8** tensors |
| `grad_enabled_during_forward` | **false** |
| `outputs_requiring_grad` | **0** |
| `backward_calls` / `optimizer_steps` / `parameter_updates` | **0 / 0 / 0** |
| `optimizer_constructed` | **false** |
| `parameters_identical` | **true** |
| Frozen encoder SHA-256, before **and** after | `6d9f6a5796818bb36ea2f9b31e1441623a7d1b086ec5f8ad1f782bfa049ddc7a` |
| Trainable SHA-256, before **and** after | `3627e9953324b1d9dc8f9fbb0a9ee2eacd04538f1ce490f49f58ffbc3f54c21b` |

Timing: prepared-corpus load **13.21 s**; one-time condition setup **1 585.65 s**;
recurring validation total **302.245 s**; whole command **1 918.02 s**.

**What this establishes.** Every repair from §U onward is now confirmed against
the real encoder and the real dev corpus, in one run: §U's instrumented evaluator
really measured (810 forwards, not zero); §W's inventory preflight resolved
eligibility; §X's `TRUNCATION` carried all 11 443 dev chunks with no overflow;
§Y's device boundary delivered CUDA tensors to a CUDA encoder. And the no-update
boundary held by **hash equality before and after**, not by assertion.

**The projection is descriptive operational evidence only.** The tool emitted
**41** evaluations per 20 000-update run and a projected **12 392.0 s** of
validation per run. That is recorded as a measurement of the locked protocol, not
as an argument about it. **The cadence stays at 500. No validation subset, no
reference cache, no protocol change follows from this number**, per the standing
instruction that any such change must be an explicit scientific decision.

### AA.2 The runner smoke — both attempts, preserved

**Attempt 1.** The orchestrator invoked `stage1_runner.py smoke` with
`--completion-dir`. **argparse rejected it**: the smoke subcommand did not declare
that option. Runner logic was never entered.

**Attempt 2.** Corrected to the documented CLI, exactly as `smoke -h` advertised:

```
python -u scripts/stage1_runner.py smoke \
  --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
  --revision        01daacda68afe13d83023d16ec647239e344a1e6 \
  --repository-head a5da53805498a12ed64ffa28a6a13232dc8e4b1b
```

**argparse PASS. `run_smoke` entered. Immediate failure**, 0.09 s in:

```
File "scripts/stage1_runner.py", line 529, in run_smoke
    completion_dir=Path(args.completion_dir) if args.completion_dir else None,
AttributeError: 'Namespace' object has no attribute 'completion_dir'
```

| | |
|---|---|
| `smoke_check` executed | **NO** |
| Runner model forward | **NOT ESTABLISHED** |
| Optimizer | **NONE** |
| Parameter update | **ZERO** |
| Training | **NOT STARTED** |

Attempt 1 is preserved deliberately: it was not an operator error in *kind*. Its
two roots — payload on local disk, `COMPLETE.json` on Drive — are exactly the real
deployment, and exactly the case `--completion-dir` exists to serve.

### AA.3 The `completion_dir` contract, traced

| Question | Answer |
|---|---|
| What does it identify? | the directory holding **`COMPLETE.json`** for a prepared corpus |
| Required to verify `COMPLETE.json`? | **Yes** — `verify_prepared_corpus(prepared_dir, completion_dir)` reads `completion_dir / COMPLETE_NAME` |
| Part of prepared-corpus provenance verification? | **Yes** — it is the F1 gate's input |
| Optional at the `smoke_check` API? | **Yes**, with an explicit documented fallback: `<prepared-corpus>/_checkpoint` |
| What is lost when `None`? | **No verification strength.** Verification still runs and still fails closed; only the ability to name a *different root* is lost |
| Do the training subcommands pass it? | **Yes** — `_corpus_consumer` declares it, `_verified_corpus` consumes it with the identical fallback |
| Does the measurement tool use it? | **Yes** — and the successful run in AA.1 supplied it |
| Was `run_smoke` written assuming the parser provides it? | **Yes**, consistently with every other consumer |
| Which side is inconsistent? | **The parser.** The handler, the `smoke_check` API, the training subcommands and the measurement tool all agree |

**Classification: A — parser omission.** Not B: the option is neither irrelevant
nor optional-by-design at the CLI; the real deployment needs it. Not C: the
downstream contract is consistent everywhere else. Not D.

### AA.4 Repair

`--prepared-corpus` and `--completion-dir` are now declared **once**, in
`_prepared_corpus_inputs(parser)`, used by both `_corpus_consumer` (training) and
the `smoke` subparser. One CLI contract, one downstream contract, one help text.

**`getattr(args, "completion_dir", None)` was deliberately NOT used.** For this
field that is not a defensive default but an integrity choice: it would silently
substitute an inferred marker for an explicitly named one, and it would let the
parser and handler drift apart again. The F1 gate is untouched — `smoke_check`
still calls `verify_prepared_corpus` before any model load, whether or not the
option is supplied.

The whole repair is **`scripts/stage1_runner.py`, +15 / −2**.

### AA.5 Tests

New `tests/test_stage1_runner_cli_contract.py` (19 tests) closes the **defect
class**, not this instance: for **every** subcommand it parses that command's own
minimal valid argv and asserts that every `args.<attr>` the handler
reads — following helpers that are handed `args`, such as `_verified_corpus` and
`_execute` — exists in the resulting `Namespace`. It also forbids `getattr` on
`args` in any handler.

Verified against the committed HEAD `a5da5380`: the smoke `Namespace` there holds
`prepared_corpus, repository_head, revision`, `run_smoke` reads `completion_dir`
as well, and executing that handler reproduces the exact production message —
`'Namespace' object has no attribute 'completion_dir'`. **The new test fails at
HEAD and passes after the repair.**

The suite also executes the parser → handler boundary with `smoke_check` stubbed
(no model, no torch), proving `run_smoke` now reaches it and forwards `None` or an
explicit path correctly; asserts the help text matches the accepted options
exactly; and re-proves the no-update guarantee **from the call graph** —
`smoke_check` reaches no `backward`, `step`, `zero_grad`, optimizer construction,
`train_run` or checkpoint write, runs its forward under `no_grad`, and still calls
`build_objective`, `verify_model_contract`, `prepare_example`,
`collate_stage1_batch` and `verify_scientific_inputs`.

### AA.6 The AA.1 validation evidence is untouched

Every file the successful validation executes is **byte-unchanged** versus
`a5da5380`: `scripts/stage1_pretrain_measurements.py`, `validation.py`, `data.py`,
`objective.py`, `unmark/modeling/`, `preflight.py`, `protocol.py`, `contracts.py`,
`checkpoint.py`, `chunking.py`, `orthography/`, `linguistics/`, `configs/` and
`docs/spec/decisions.md`. The diff is confined to the runner CLI.

**No re-run of the 32-minute validation is required or requested.**

### AA.7 Preserved, and still open

§T–§Z unchanged. §V provenance, §W inventory (`135a4d97…` / `78eeb840…` /
116 290 B / 17 974 / 17 954 / 2 489, seven-field identity, digest report-only),
§X truncation and §Y device boundary all intact. Stage 6, the prepared corpus and
official TEST are untouched.

**Still separately open, and deliberately not touched here: `execute_stage`
performs no accelerator placement, so the *training* device remains unchosen
(§Y.3).** That is a training-authorisation question and has nothing to do with the
smoke's `completion_dir`.

### AA.8 What is still NOT established

The runner smoke has **never completed**. There is no runner-side model-forward
evidence, no optimizer, no update, no training. **One corrected real runner smoke
is still required after commit** — the full validation of AA.1 does not stand in
for it.

**STATUS: PRE-TRAIN RUNNER-SMOKE CLI REPAIR PASS — READY TO COMMIT AND RERUN SMOKE ONLY**
**THE FIRST FULL FOUR-CONDITION REAL VALIDATION PASSED: 810 FORWARDS, 11 443 DEV CHUNKS, CUDA, fp32**
**NO-UPDATE BOUNDARY HELD BY HASH EQUALITY BEFORE AND AFTER — 0 BACKWARD, 0 STEPS, 0 UPDATES**
**THE 12 392 s PROJECTION IS DESCRIPTIVE ONLY — CADENCE STAYS 500, NO SUBSET, NO CACHE**
**CLASSIFICATION A — PARSER OMISSION; THE HANDLER AND EVERY OTHER CONSUMER WERE ALREADY RIGHT**
**ONE SHARED DECLARATION: `_prepared_corpus_inputs` SERVES TRAINING AND SMOKE ALIKE**
**`getattr(args, ...)` REFUSED — IT WOULD HAVE WEAKENED THE F1 COMPLETE GATE AND HIDDEN THE DRIFT**
**NEW TEST FAILS AT `a5da5380` AND REPRODUCES THE EXACT AttributeError; PASSES AFTER REPAIR**
**FULL-VALIDATION EXECUTABLE SURFACE BYTE-UNCHANGED — NO RE-RUN NEEDED**
**TRAINING DEVICE PLACEMENT REMAINS A SEPARATE UNRESOLVED OPERATIONAL ITEM**
**ONE CORRECTED REAL RUNNER SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM ONE PASSED**

---

## AB. FIFTH REAL NO-UPDATE SMOKE CLOSED — CORRECTED RUNNER SMOKE PASS

**Revision 12.** The §AA repair was committed and the corrected runner smoke was
run against the real model and the real prepared corpus. **It passed.** With
§AA.1's full validation already established, the **PRE-TRAIN no-update smoke gate
is closed**.

This section is documentation only: no production code, test or constant changed.

### AB.1 Continuation boundary — why AA.1 was reused, not rerun

A still-live runtime that had completed the successful full validation at
`a5da53805498a12ed64ffa28a6a13232dc8e4b1b` was switched to exact committed HEAD
`2363e33588c4ee70402ea0074f26baf93cdc88d7`, clean repository.

The diff between the two consisted **exactly** of:

```
docs/audits/030-pretrain-repository-wide-audit.md
scripts/stage1_runner.py
tests/test_stage1_runner_cli_contract.py
```

The full-validation **executable surface was independently checked byte-identical**
to `a5da5380` — the measurement tool, validation, data, objective, modeling,
preflight, protocol, contracts, checkpoint, chunking, orthography, linguistics,
`configs/` and `docs/spec/decisions.md`. One audit file, one CLI file, one new
test file: nothing the validation runs.

> **AA.1's evidence therefore remains valid and was NOT rerun.** The 1 918 s
> validation stands on its own HEAD, and this continuation neither repeats nor
> weakens it.

Retained runtime inputs were re-verified before the smoke:

| | |
|---|---|
| Inventory bytes | **116 290** |
| Inventory sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Prepared membership digest | `250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6` |
| `tests/test_stage1_runner_cli_contract.py` | **19 passed** |

The public smoke CLI now exposes exactly `--prepared-corpus`, `--completion-dir`,
`--revision`, `--repository-head` — the contract §AA repaired.

### AB.2 The corrected real runner smoke

```
python -u scripts/stage1_runner.py smoke \
  --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
  --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
  --revision        01daacda68afe13d83023d16ec647239e344a1e6 \
  --repository-head 2363e33588c4ee70402ea0074f26baf93cdc88d7
```

**Completed successfully. Return code 0, wall time 24.36 s.**

This is the invocation shape §AA.2 Attempt 1 tried and argparse rejected: two
roots, payload on local disk and `COMPLETE.json` on Drive. It now works, which is
the point of the repair.

**Gates, in the order they ran:**

| Gate | Evidence |
|---|---|
| Scientific inputs preflight | **VERIFIED** — eligibility `VIETNAMESE_SYLLABLE_INVENTORY` |
| Inventory identity | sha256 `78eeb840…` |
| Prepared corpus | **VERIFIED against** `/content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb/COMPLETE.json` |
| Chunk membership digest | `250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6` |
| Loaded | train **2 621 624**, dev **11 443** chunks in **12.9 s** |
| Process RSS after load | **3.78 GB** |

Both preflights ran **before** the model, as §W and F1 require, and the
`--completion-dir` the smoke was given is the marker it actually verified against.

**Real forward result** (`batch_size` 8, `lambda_align` 1.0, `lambda_clean` 1.0):

| | |
|---|---|
| `loss` | **0.6354566812515259** |
| `loss_align` / `mean_distance_align` | **0.31667956709861755** |
| `loss_clean` / `mean_distance_clean` | **0.3187771439552307** |

**Model contract:**

| | |
|---|---|
| `encoder_trainable_parameters` | **0** |
| `encoder_training_mode` | **false** |
| `hidden_size` | **768** |
| `precision` | **fp32** |
| `trainable_parameters` | **3 551 232** |
| `trainable_tensors` | **8** |

**No-update evidence:**

| | |
|---|---|
| `smoke` | **`STAGE1_NO_UPDATE_FORWARD_ONLY`** |
| `backward_called` | **false** |
| `optimizer_constructed` | **false** |
| `parameters_updated` | **0** |
| `repository_head` | `2363e33588c4ee70402ea0074f26baf93cdc88d7` |

### AB.3 The runner smoke's device is NOT claimed

**The runner-smoke report contains no device field** — its keys are `smoke`,
`repository_head`, `model_contract`, `losses`, `optimizer_constructed`,
`backward_called`, `parameters_updated`. **This audit therefore does not describe
it as a CUDA runner smoke.** Nothing about the accelerator is asserted from it.

The repository's CUDA evidence comes from elsewhere and is unaffected:

* **§AA.1** — the full four-condition real validation, explicitly on CUDA (RTX PRO
  6000 Blackwell, torch 2.11.0+cu128, CUDA 12.8), 810 forward passes;
* **§Z** — the device-contract runtime tests, **7 passed, 0 failed, 0 skipped** on
  a real CUDA host.

That distinction matters because of what remains open in AB.5.

### AB.4 PRE-TRAIN NO-UPDATE SMOKE GATE: **PASS**

Taken with §AA.1, the no-update gate this audit has been driving toward since §N
is closed:

* the **real encoder** loaded at the pinned revision, frozen, `encoder_trainable_parameters 0`;
* the **real prepared corpus** verified against its own `COMPLETE.json` before model load;
* the **real inventory** verified, eligibility resolved on the scientific path;
* a **real forward** producing real losses on the real objective;
* and **zero** optimizers, **zero** backward calls, **zero** parameter updates —
  in AA.1 additionally proved by parameter-hash equality before and after.

**This is NOT a statement that training is authorised.**

### AB.5 The one remaining known blocker

> **`execute_stage` performs no accelerator placement.** The training path never
> moves the model to a device, so Stage-1 training as wired would run on the CPU.
> Which device training uses, and how it is selected, is an **operational contract
> nobody has chosen** — first recorded in **§Y.3** and deliberately left unresolved
> in §AA and here.

This is now the **only known pre-training implementation blocker identified by
Audit 030**, subject to the forthcoming dedicated training-device review. It is
**not resolved in this task**, and the note in AB.3 is why: the runner smoke's
silence about its device is precisely the gap that review must close.

### AB.6 Preserved

§T through §AA are preserved verbatim. Unchanged and re-verified at this HEAD:
`vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`, transformers
4.57.6, fp32, `MAX_LENGTH` 256 with `ON_OVERFLOW FAIL` and truncation not offered,
batch 128, eval every 500, checkpoint cadence 500, validation seed 19225,
corruption seed 35422, `PI_STRIP` 0.25, dev documents 5000, 20 000 → 40 000
continuation, best + last, adapter 3 551 232 trainable parameters, the LR grid and
the r grid, the inventory identity, the prepared corpus, Stage 6, and the official
UIT-VSFC TEST seal.

`docs/spec/decisions.md` is unchanged. No Audit 031. No executable or test diff.

**STATUS: PRE-TRAIN NO-UPDATE SMOKE GATE PASS — TRAINING DEVICE CONTRACT REMAINS**
**THE CORRECTED REAL RUNNER SMOKE COMPLETED: RETURN CODE 0, 24.36 s, REAL MODEL, REAL CORPUS**
**`STAGE1_NO_UPDATE_FORWARD_ONLY` — backward false, optimizer false, parameters_updated 0**
**PREPARED CORPUS VERIFIED AGAINST ITS DRIVE `COMPLETE.json`; MEMBERSHIP DIGEST `250859a5…78413d6`**
**§AA's `--completion-dir` REPAIR CONFIRMED BY THE VERY INVOCATION ARGPARSE ONCE REJECTED**
**AA.1's FULL VALIDATION REUSED, NOT RERUN — EXECUTABLE SURFACE BYTE-IDENTICAL TO `a5da5380`**
**THE RUNNER SMOKE REPORTS NO DEVICE; IT IS NOT CLAIMED AS A CUDA RUN (CUDA EVIDENCE: §AA.1, §Z)**
**DOCUMENTATION ONLY — NO PRODUCTION CODE, NO TEST, NO CONSTANT, NO DECISION CHANGED**
**ONE KNOWN BLOCKER REMAINS: THE TRAINING-DEVICE OPERATIONAL CONTRACT (§Y.3)**
**THIS DOES NOT AUTHORISE TRAINING**

---

## AC. FINAL PRE-TRAIN TRAINING-DEVICE CONTRACT AUDIT

**Revision 13.** The dedicated phase-boundary review §AB deferred, at HEAD
`0588b722c7c5e34bf6bda8f5703cfda80f7939be`. **Audit only — no production code,
test, constant or decision changed.**

It did not stop at the known device issue. **It found a second blocker.**

### AC.1 What the proposal and specs actually say

| Constraint | Source | Kind |
|---|---|---|
| **FP32** | proposal §5.1.1 optimizer row — "AdamW … accumulation 1, no clipping initially, **FP32**" | **SCIENTIFIC PROTOCOL** |
| **"One GPU."** Stage 1 is "a few hours"; "Colab or Kaggle is sufficient" | proposal **§8.4 Compute** | **OPERATIONAL** — a resourcing statement |
| G0 is "Day 1, **no GPU**" | proposal §7 | implies later stages *do* use a GPU |
| Deterministic corruption, keyed by `(schema, seed, sample_id, text identity, unit index)` | proposal §6 / D-B2-001… | **SCIENTIFIC** |
| "Public repository with code, configs, **and seeds**"; "raw **per-seed** numbers, not only aggregates" | proposal **§11** | **SCIENTIFIC** (reproducibility) |
| "**Seeds.** At least three per configuration; report mean and standard deviation. If the improvement is within seed variance, there is no result" | proposal §7 | **SCIENTIFIC** |
| 20k → one continuation → 40k, batch 128, eval every 500, best+last | §5.1.1 / D-S1B-004 | **SCIENTIFIC** |

> **The proposal states "One GPU" as a compute expectation and never specifies a
> device-selection mechanism, a fail-closed rule, a device index, or a
> CPU-fallback policy. Those are unstated.** It does *not* say Stage-1 may run on
> CPU either; "a few hours" is only true on an accelerator.

`docs/spec/decisions.md` contains **no** device entry (verified: zero matches for
device/cuda/GPU).

### AC.2 The complete Stage-1 training call graph

```
stage1_runner.py  {lr-pilot | r-phase1 | final-main}
  └─ _verified_corpus(args)          verify_prepared_corpus  ← F1, before model
  └─ _execute(args, schedule, stage, verified)
      └─ execute_stage(...)                                  unmark/stage1/execute.py:97
          ├─ verify_scientific_inputs()                      ← §W preflight, before model
          ├─ load_prepared_chunks(prepared_corpus)           train/dev text  (CPU, python)
          ├─ build_objective(revision)                       execute.py:71
          │    ├─ AutoTokenizer.from_pretrained
          │    ├─ AutoModel.from_pretrained  ─────────────►  ★ CPU  (no device_map)
          │    ├─ requires_grad_(False) on every encoder param;  encoder.eval()
          │    ├─ OrthographyInputAdapter(...)  ───────────►  ★ CPU, RANDOM INIT (see AC.7)
          │    └─ UnmarkEncoder(encoder, adapter)
          ├─ ★ NO .to(device) ANYWHERE
          ├─ classifier = make_classifier(try_load_inventory())
          ├─ prepared_by_condition = {c: prepare_condition_batch(...)}   ← built ONCE
          ├─ RunProvenance(... inventory=inputs.inventory)               ← §V/§W identity
          └─ train_run(...)                                  unmark/stage1/trainer.py:463
              ├─ verify_model_contract(objective.unmark_encoder)
              ├─ build_optimizer(trainable named_parameters, lr)  ★ optimizer over CPU params
              ├─ DeterministicSampler(sorted(chunks), seed=provenance.run_seed)
              ├─ if resume:  verify_checkpoint → load_state_dict(adapter_state, strict=False)
              │              → optimizer.load_state_dict → sampler.from_state → global_update
              ├─ evaluate_fn(0)                              ← update-0 point before any step
              ├─ objective.train(True)                       ← UnmarkEncoder.train override
              └─ while global_update < cap:
                   ├─ sampler.next_batch(BATCH_SIZE)
                   ├─ prepare_example(...) × 128             ← CPU
                   ├─ collate_stage1_batch(...)  ──────────►  ★ CPU tensors
                   ├─ batch_to_device(batch, module_device(objective))   ← §Y boundary
                   ├─ objective(batch) → Stage1LossResult
                   ├─ optimizer.zero_grad → loss.backward() → optimizer.step()
                   ├─ every 500: evaluate_fn(u) → validation.evaluate → objective.eval()
                   │             then objective.train(True)   ← trainer.py:577, restored
                   └─ every 500: save_training_checkpoint(best + last)
```

### AC.3 Device-operation inventory (complete, whole repository)

| Location | Operation | Classification |
|---|---|---|
| `stage1/data.py:499` | `torch.device("cpu")` — `module_device` fallback | **production, training + validation** |
| `stage1/data.py:510` | `batch_to_device` — `value.to(device)` | **production, training + validation + smoke** |
| `stage1/trainer.py:411` | `torch.load(..., map_location="cpu", weights_only=False)` | **production, resume** |
| `modeling/adapter.py:621` | `supplied.to(derived.device)` | production, *comparison convenience only* |
| `modeling/pooling.py:81-82`, `adapter.py:168/186/190` | `.to(dtype)` | **dtype, not device** |
| `scripts/stage1_pretrain_measurements.py:394,401,409,435,456,506-509` | device select, `unmark_encoder.to(device)`, CUDA sync/memory | **measurement-only** |
| `scripts/g_minus1_restore_smoke.py`, `b4b_phobert_adapter_probe.py`, `evaluation/preg1_head.py` | device select / `.to(device)` / `manual_seed` | **other phases, unrelated to Stage-1 training** |

> **`unmark/stage1/` contains no `torch.cuda`, no `is_available`, no model
> `.to(device)`, no `autocast`, no `GradScaler`, no AMP, no `set_default_device`,
> no TF32 flag, and no `manual_seed`.** The only device code on the training path
> is the batch-following boundary added in §Y.

### AC.4 Fresh-run device contract at this HEAD — proven

| # | Question | Answer (from code) |
|---|---|---|
| 1 | Where is the model instantiated? | `build_objective`, `execute.py:86` — `AutoModel.from_pretrained` |
| 2 | Device immediately after construction? | **CPU** — no `device_map`, no `.to()` |
| 3 | Is it explicitly moved? | **No.** `execute.py` has no device op outside `smoke_check` |
| 4 | Optimizer constructed when? | `trainer.py:508`, before any (non-existent) movement |
| 5 | Prepared batches initially on? | **CPU** — `collate_stage1_batch` → `torch.tensor(...)` |
| 6 | Where are batches moved? | `trainer.py:550`, `batch_to_device(..., module_device(objective))` |
| 7 | Does that use the real module device? | **Yes** — `next(module.parameters()).device` |
| 8 | Does training run end-to-end on CPU? | **YES.** Every tensor and parameter stays on CPU |
| 9 | Could one part be CUDA and another CPU? | **No** — the batch follows the model, so it self-corrects |
| 10 | Crash, silent CPU, or environment-dependent? | **Silent CPU, identically on every machine.** No warning, no record |

**The run would not fail. It would quietly take a scale of time §8.4 never
contemplated, and no artifact would say so** — the runner smoke report has **no
device field** (§AB.3), and `RunProvenance` records no device either.

### AC.5 Validation during training — device-safe

Traced from `execute_stage`'s `evaluate_fn` closure:

| Question | Finding |
|---|---|
| Same model object being trained? | **Yes** — the closure binds `_obj=objective`, the identical instance |
| Device inferred from that module? | **Yes** — `evaluate` computes `module_device(objective)` (§Y) |
| All tensors moved consistently? | **Yes** — one `batch_to_device` per batch, covering reference, base, and every channel |
| eval/train mode correct? | `evaluate` calls `objective.eval()`; **`trainer.py:577` restores `objective.train(True)` immediately after every `evaluate_fn`** |
| Frozen encoder ever in train mode? | **No.** `UnmarkEncoder.train()` is overridden to force `self.encoder.eval()` after `super().train(mode)` — so the encoder's dropout can never reactivate, in either mode |
| Gradients disabled? | `evaluate` wraps everything in `torch.no_grad()`; `reference_representation` adds its own |
| Could validation alter training behaviour? | **No** — it moves batches, never parameters; the model is not moved |
| Reference path same device? | **Yes** — same batch, same transfer |
| Hidden CPU-only tensors/buffers in the objective? | **None found.** `ObjectiveWeights` are Python floats; position ids are derived from `input_ids` on its device; no registered buffer is created outside the modules `.to()` moves |

**VERIFIED / NO ISSUE.** The standalone validation success in §AA.1 is *not* what
proves this — the call graph is.

### AC.6 Optimizer construction order

Current order in `train_run`: `verify_model_contract` → **`build_optimizer`** →
sampler → (resume: `load_state_dict` → `optimizer.load_state_dict`).

There is no model movement anywhere, so ordering is inert **today** and becomes
load-bearing the moment placement is added. The correct invariant for this
implementation is:

```
construct model → PLACE MODEL → construct optimizer → load model state → load optimizer state
```

Two points, both checked against how this repository actually restores state:

* **Placement after optimizer construction would not corrupt parameter
  references.** `nn.Module.to()` mutates `param.data` in place and keeps the same
  `Parameter` objects, so the optimizer's `param_groups` stay valid. This is a
  real PyTorch guarantee, not luck — but it is *incidental* to the repository,
  which documents no dependency on it.
* **Optimizer state placement is entirely dependent on parameter device at load
  time.** `Optimizer.load_state_dict` casts loaded state to each parameter's
  current device/dtype. Because `load_training_checkpoint` uses
  `map_location="cpu"`, **the checkpoint always arrives on CPU**, and the state
  lands wherever the parameters already are. **If the model were placed *after*
  `optimizer.load_state_dict`, Adam's exponential-moving-average state would stay
  on CPU while the parameters moved to CUDA** — a cross-device optimizer step.

**SHOULD FIX (documentation + test):** the ordering requirement is currently
unwritten and untested.

### AC.7 ★ NEW BLOCKER — adapter initialisation is not seeded

**Evidence.** `run_seed` is consumed at exactly one place:
`trainer.py:512`, `DeterministicSampler(..., seed=provenance.run_seed)` — **data
order only**. There is **no `torch.manual_seed` anywhere in `unmark/stage1/` or
`scripts/stage1_runner.py`**. The adapter's trainable parameters —
`nn.Embedding` ×2 (tone, letter), `nn.Linear` ×2 (fusion, gate), `nn.LayerNorm` —
are created in `OrthographyInputAdapter.__init__` and initialised by PyTorch's
**global default RNG, unseeded**.

**Consequences.**

1. **Re-running the same run is not reproducible.** `final-main seed=36930`
   executed twice yields different initial adapter weights and therefore
   different results.
2. **`RunProvenance.run_seed` is recorded as run identity but does not determine
   the run.** This is exactly the defect class §V closed for `lambda_align` and
   §W closed for the inventory: an identity that does not identify.
3. **Proposal §11 requires publishing "code, configs, **and seeds**" and "raw
   per-seed numbers".** A published seed that cannot reproduce its number does not
   satisfy that.
4. Resume equivalence is **unaffected** — `adapter_state` is restored from the
   checkpoint — so §V's and F3's evidence stands. The gap is *fresh-run* identity.

**Why it never surfaced.** Nothing before this point ever ran two training runs;
every gate so far was single-shot or state-restoring.

**Classification: MUST DECIDE BEFORE TRAINING — scientific (reproducibility).**
The audit does **not** choose the mechanism. The open question is what seeds
initialisation: `run_seed` itself, or a separately derived
`SEED_ROOT_TAG|init|i` stream domain-separated from data order in the same style
as `CORRUPTION_SEED_TAG` / `SPLIT_SEED_TAG`. Both are defensible; the repository's
established pattern favours the latter, and either needs a decision entry.

### AC.8 Checkpoint / resume / continuation device semantics

| # | Question | Finding |
|---|---|---|
| 1 | Device of saved checkpoint tensors? | Whatever the parameters are on — CUDA tensors are saved as CUDA |
| 2 | Device on load? | **CPU, always** — `map_location="cpu"` |
| 3 | Is `map_location` explicit? | **Yes**, `trainer.py:411`. Good practice, already correct |
| 4 | CUDA checkpoint → CPU host? | **Yes**, safe, because of (3) |
| 5 | CPU checkpoint → CUDA run? | **Yes** — `Module.load_state_dict` copies **into** existing params in place, preserving their device |
| 6 | Optimizer state device after `load_state_dict`? | The **parameters'** device at that moment (PyTorch casts state to the param) |
| 7 | Explicit or incidental? | **Incidental** — correct, documented PyTorch behaviour, but nothing in this repository states or tests the dependency |
| 8 | Safe on torch 2.11.0? | Yes, this is stable documented behaviour — but see the probe in AC.12 |
| 9 | Relying on framework behaviour? | **Yes**, for (5) and (6). Worth pinning with a test |
| 10 | Adam state on CPU while params CUDA? | **Only** if placement happened after `optimizer.load_state_dict` — see AC.6 |
| 11 | Continuation silently switching backend? | **Yes, possible today.** Nothing records or verifies the device, so a 20k leg on CUDA could continue to 40k on CPU (or the reverse) and *every* provenance check would pass |
| 12 | best vs last restore differ? | Resume always uses **last** (`load_training_checkpoint(run_checkpoints)`); best is written for reporting. Correct per D-S1B-004 |
| 13 | Same-run continuation identity preserved? | **Yes** — `resume=carried` from the same `run-{label}/_checkpoint`, verified by `verify_checkpoint` against the full provenance including `repository_head` and `inventory` |

**(11) is a real gap:** device is absent from `RunProvenance`, so backend drift
across a continuation is undetectable.

### AC.9 ★ NEW FINDING — every checkpoint stores the frozen encoder

`trainer.py:594` saves `adapter_state=objective.unmark_encoder.state_dict()` —
the **whole wrapper**, so every checkpoint contains the complete frozen PhoBERT
encoder alongside the 3 551 232 trainable parameters.

* Estimated ≈ **540 MB** per file at fp32 for a RoBERTa-base-sized encoder, versus
  ≈ 14 MB for adapter + Adam state alone. *(Estimate — the exact figure needs the
  runtime probe in AC.12.)*
* Written to **best and last**, at **every 500 updates**: ~41 publications per
  20k leg, each an atomic temp → fsync → replace of ~540 MB, on Google Drive.
* `load_state_dict(..., strict=False)` means a key mismatch would **silently
  restore nothing** and continue from random weights without error.

**Correctness is not affected** (frozen weights are re-restored identically), but
this is a material operational risk for an 11-run campaign on Drive, and the
`strict=False` silence is a fail-open in an otherwise fail-closed codebase.

**Classification: SHOULD FIX BEFORE TRAINING — operational.** Persisting only
trainable parameters, and asserting the restored key set, would remove both.

### AC.10 Reproducibility, RNG and backend

| Stream | State | Checkpointed? |
|---|---|---|
| Corruption | **Stateless** — keyed `blake2b(schema, seed, sample_id, text identity, unit index)`. No RNG object | N/A — reproducible by construction |
| Sampler / data order | `DeterministicSampler(seed=run_seed)`, explicit `state_dict` | **Yes** (`sampler_state`) |
| Validation corruption | fixed seed 19225, fixed visit | N/A |
| **torch global RNG** | **never seeded** (AC.7) | **No** |
| Python `random` / NumPy | not used on the Stage-1 path | N/A |
| CUDA RNG | never seeded; **no consumer** — encoder dropout is forced off by `UnmarkEncoder.train()`, and the adapter has no stochastic layer | N/A |

**What Stage-1 currently promises:** *deterministic data order and deterministic
corruption*, plus *exact interrupted-vs-uninterrupted continuation* (§V, F3
evidence). It does **not** promise exact fresh-run numerical reproducibility, and
because of AC.7 it cannot.

Because there is **no stochastic layer in the training forward**, CUDA introduces
no *sampling* nondeterminism. It can introduce **floating-point reduction-order
nondeterminism** (atomics, cuBLAS algorithm selection). Classification of the
optional hardening flags, per the mission's own scheme:

| Flag | Classification |
|---|---|
| `torch.manual_seed` at run start | **REQUIRED** — see AC.7 |
| `torch.cuda.manual_seed_all` | **OPTIONAL HARDENING** — no CUDA RNG consumer exists today; cheap insurance |
| `torch.use_deterministic_algorithms(True)` | **OPTIONAL HARDENING.** The proposal never asks for bitwise cross-run equality, and it can cost throughput or raise on unsupported ops |
| cuDNN deterministic flags | **IRRELEVANT** — no convolutions in a transformer |
| TF32 flags | see AC.11 |

### AC.11 fp32 must remain fp32

**AMP is absent, verifiably.** Zero occurrences of `autocast`, `GradScaler`,
`torch.amp`, `fp16`, `bf16` or `half()` anywhere in `unmark/` or the Stage-1
scripts. `PRECISION = "fp32"` is recorded in the protocol and in every artifact.
Moving to CUDA changes none of that: `.to(device)` alters device, never dtype.

**TF32 is the one real question.** On Ampere-and-later hardware — which the
RTX PRO 6000 Blackwell is — cuBLAS may execute `float32` matmuls with TF32
internal precision (10-bit mantissa) depending on backend flags. Every
`nn.Linear` in the encoder and the adapter is such a matmul.

* Current PyTorch defaults set `torch.backends.cuda.matmul.allow_tf32 = False`
  (changed in 1.12), while `torch.backends.cudnn.allow_tf32` defaults to `True`
  and governs **convolutions**, of which there are none here.
* So the expected behaviour is **true fp32 matmul**, and the §AA.1 validation ran
  under exactly these defaults.
* **But the repository asserts nothing.** A future torch release, a global config,
  or `torch.set_float32_matmul_precision("high")` set elsewhere would change the
  arithmetic while every artifact still recorded `precision: fp32`.

**Classification: SHOULD FIX BEFORE TRAINING — operational, protecting a
scientific constant.** The fix is to *record and assert* the TF32/matmul-precision
state in provenance rather than to change any numerics. The exact runtime probe is
in AC.12; I cannot read torch's defaults from this ML-free venv and will not guess
them as established fact.

### AC.12 Single-GPU / multi-GPU / CPU behaviour

| Environment | Behaviour at this HEAD |
|---|---|
| No CUDA | training runs on CPU, silently |
| Exactly one GPU | training runs on CPU, silently |
| Multiple GPUs | training runs on CPU, silently |
| `CUDA_VISIBLE_DEVICES` set | no effect — nothing consults CUDA |
| Default index ≠ physical GPU 0 | no effect |

Nothing in `unmark/stage1/` hardcodes `cuda:0` or calls `current_device()`. The
only index literal in the repository is
`stage1_pretrain_measurements.py:456`, `torch.cuda.get_device_name(0)` —
**reporting only**, in the measurement tool, and consistent with that tool's
`torch.device("cuda")` logical default. It names the first *visible* device, which
is what `CUDA_VISIBLE_DEVICES` is for. Not a defect; `torch.cuda.current_device()`
would be marginally more precise.

### AC.13 The candidate operational contract, clause by clause

| Clause | Classification |
|---|---|
| "Stage-1 scientific training requires CUDA" | **B — operational clarification consistent with the proposal.** §8.4 says "One GPU… a few hours"; it is the natural reading, but the proposal never states it as a requirement, so it must be *recorded*, not assumed |
| "The entry point fails closed if CUDA is unavailable" | **B.** Consistent with this repository's fail-closed discipline everywhere else (F1, §W, §X). Not stated anywhere yet |
| "Selects the logical default visible CUDA device without hardcoding a physical GPU ID" | **B, and correct.** `torch.device("cuda")` honours `CUDA_VISIBLE_DEVICES`; hardcoding an index would not |
| "The complete model/objective is placed on that device **before optimizer construction**" | **B, and necessary** — see AC.6. Should be strengthened to *before optimizer **state** loading* too |
| "Training and validation batches are moved to the model's actual device" | **A — already required and already implemented** (§Y, `batch_to_device`/`module_device`) |
| "Checkpoint resume explicitly restores/migrates model and optimizer state to the selected device" | **E — partially correct but incomplete.** Model state already migrates correctly via in-place copy; optimizer state migrates *incidentally* via `Optimizer.load_state_dict`. Making it explicit is right; the clause should say **assert**, not re-implement |
| "A continuation may not silently change the execution backend" | **B, and it closes AC.8(11)** — but it is unenforceable until the device is recorded in provenance. **Add the device to the run artifact**; whether it joins `RunProvenance.require_match` is the decision to make, since a legitimate crash-resume onto a different GPU model is plausible |
| "fp32 remains mandatory" | **A — already required** (proposal §5.1.1). Should be *extended* to assert TF32/matmul precision (AC.11) |
| "No automatic CPU fallback for scientific Stage-1 training" | **B.** Note this must **not** leak into `smoke`, the measurement tool's CPU path, or the test suite, all of which legitimately run on CPU |

**A better contract, precisely.** Adopt the candidate with three amendments:

1. **Place before optimizer *state* load**, not merely before construction:
   `build → place → construct optimizer → load model state → load optimizer state`.
2. **Record the resolved device, GPU name, torch/CUDA versions and the effective
   matmul/TF32 precision in the run artifact**, so a continuation's backend is
   *evidence* rather than an assumption — and decide separately whether device
   identity is a resume-blocking field.
3. **Scope the CUDA requirement to the scientific training entry point only**
   (`lr-pilot`, `r-phase1`, `final-main`), explicitly exempting `smoke`,
   measurement and tests.

### AC.14 Decision-log consequence

**Yes — a decision entry is required**, and this audit did not write one. Two
distinct decisions are needed. Draft content for the next implementation task:

**D-S1B-015 — Stage-1 scientific training requires an explicitly selected CUDA device**

* *Original proposal wording/assumption*: §8.4 "One GPU. Stage 1 is a few hours…
  Colab or Kaggle is sufficient." No selection mechanism, no fallback policy, no
  device index is specified anywhere; `docs/spec/decisions.md` has no device entry.
* *Implemented operational decision*: the scientific training entry points require
  CUDA and fail closed when it is unavailable; the logical default visible device
  is selected (`torch.device("cuda")`, honouring `CUDA_VISIBLE_DEVICES`, never a
  hardcoded index); the objective is placed on it **before optimizer construction
  and before optimizer-state loading**; batches follow the model's device (already
  implemented, §Y); the resolved device, GPU name, torch/CUDA versions and
  effective fp32 matmul precision are recorded in the run artifact; `smoke`,
  the measurement tool and tests are exempt and may run on CPU.
* *Reason*: at HEAD, training silently executes end-to-end on CPU (AC.4), which no
  artifact records; and optimizer state placement depends on ordering (AC.6).
* *Affected*: `unmark/stage1/execute.py`, `unmark/stage1/trainer.py`,
  `scripts/stage1_runner.py`, run artifacts, the Stage-1 experiments.
* *Editable proposal source*: **not required** — §8.4 already assumes one GPU;
  this names the mechanism. PDF remains stale regardless.

**D-S1B-016 — Stage-1 adapter initialisation must be seeded**

* *Original proposal wording/assumption*: §7 "Seeds. At least three per
  configuration"; §11 "code, configs, and seeds", "raw per-seed numbers".
  Reproducibility is assumed; the initialisation RNG is never named.
* *Implemented decision*: **to be chosen** — `run_seed` directly, or a derived
  `SEED_ROOT_TAG|init|i` stream domain-separated from data order. **This audit
  does not choose.**
* *Reason*: `run_seed` currently controls only data order, so a published seed
  cannot reproduce its run, and `RunProvenance.run_seed` does not determine the
  artifact it identifies (AC.7).
* *Affected*: `unmark/stage1/execute.py` or `trainer.py`, `protocol.py` if a new
  seed tag is derived, every Stage-1 run.
* *Editable proposal source*: **not required**.

### AC.15 Required fresh-runtime ZERO-UPDATE probes

The prior runtime was deleted, so both probes below must run on a fresh CUDA
runtime **after** the repair and **before** any training. Neither may call
`backward()` or `optimizer.step()`.

**Probe 1 — placement acceptance (zero update).** Must report, and fail closed on
any mismatch:

1. exact repaired HEAD; clean repository;
2. prepared corpus restored and verified against its `COMPLETE.json`; membership
   digest `250859a5…78413d6`;
3. inventory restored, 116 290 bytes, sha256 `78eeb840…`, counts 17 974 / 17 954 / 2 489;
4. CUDA **required** and selected; report `torch.cuda.get_device_name`,
   `torch.version.cuda`, torch version;
5. **every** parameter's device — trainable **and** frozen — enumerated, plus every
   registered buffer; assert one device, equal to the selected one;
6. `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32` and
   `torch.get_float32_matmul_precision()` recorded (AC.11);
7. a real prepared training batch built through `prepare_example` +
   `collate_stage1_batch`, asserted **CPU** on creation, then `batch_to_device`,
   asserted on the model device;
8. **one real training-entry forward** — `objective(batch)` — under `no_grad`;
9. loss finite; `loss_align`, `loss_clean` reported;
10. `backward` **not** called; `optimizer.step` **not** called;
11. parameter SHA-256, trainable and frozen, **identical before and after**;
12. exact per-file checkpoint size that *would* be written (settles AC.9).

**Probe 2 — resume placement (zero update).** Optimizer **construction** is
permitted here; optimizer **execution** is not. The audit states that distinction
explicitly: constructing an `AdamW` and calling `load_state_dict` performs no
update, changes no parameter, and is the only way to observe state placement.

1. write a checkpoint from Probe 1's placed model (no training);
2. fresh process: build → **place** → construct optimizer → `load_state_dict`
   (model) → `load_state_dict` (optimizer);
3. assert every optimizer state tensor (`exp_avg`, `exp_avg_sq`, `step`) is on the
   selected device — this is the AC.6/AC.8(6) claim, verified rather than assumed;
4. assert `map_location="cpu"` still loads a CUDA-written checkpoint;
5. assert sampler `visit`/`position`, `global_update`, `points` and full run
   provenance restore and `verify_checkpoint` passes;
6. **no `optimizer.step()`, no `backward()`, zero parameter change** — parameter
   hashes identical before and after the whole probe.

### AC.16 Other pre-train risks reviewed

| Item | Status |
|---|---|
| LR/r run mapping | **VERIFIED** — `LR_PILOT_GRID (1e-4, 3e-4, 1e-3)`, `R_PHASE1_GRID (0.25, 0.5, 1.0, 2.0, 4.0)`, both asserted against the locked grids in `selection.py` |
| Seed selection | `TRAIN_SEEDS (36930, 7309, 5993)`, domain-separated, collision-checked at import — **but see AC.7** |
| Continuation counted as a new candidate | **VERIFIED / NO ISSUE** — `candidates.append` sits outside the `if result.continued:` block, so one candidate per planned run |
| Checkpoint overwrite / collision | **VERIFIED** — one namespace per run, `run-{label}/_checkpoint`; artifacts `run-{label}.json` |
| Resume from wrong run | **VERIFIED** — `verify_checkpoint` compares all identity fields incl. `repository_head` and `inventory` (§V, §W) |
| best vs last semantics | **VERIFIED** — resume uses last; best chosen by the locked `select_checkpoint` |
| Budget 20k → 40k | **VERIFIED** — one continuation, then `BUDGET_LIMITED` |
| Accidental unfreezing / train-eval mistakes | **VERIFIED** — `freeze_encoder` + `UnmarkEncoder.train()` override + `verify_model_contract` (0 encoder trainable, 3 551 232 adapter) |
| Gradient accumulation / scheduler | **VERIFIED** — accumulation 1, constant LR, no warmup, no scheduler state to persist |
| Accidental truncation | **VERIFIED** — §X, `TRUNCATION` 256/FAIL on all paths |
| Corruption redraw semantics | **VERIFIED** — per-visit redraw, locked mixture asserted in `train_run` (`is_locked_mixture`) |
| Stale override flags | **VERIFIED** — no scientific override flag on any subcommand (§AA tests) |
| Official TEST access | **VERIFIED** — sealed; no CLI route; screen raises on any unlisted source |
| Validation / candidate leakage | **VERIFIED** — held-out realisation built once from dev only, fixed seed 19225, no downstream score reachable |
| Memory-loading risk | **VERIFIED** — 3.78 GB RSS measured after real load (§AB) |
| Mixed scientific/diagnostic config | **VERIFIED** — `Stage1Purpose.SCIENTIFIC` cannot be constructed with unresolved values |
| **Frozen encoder in every checkpoint** | **SHOULD FIX** — AC.9 |
| **`strict=False` on adapter restore** | **SHOULD FIX** — AC.9, fail-open in a fail-closed codebase |

### AC.17 Final blocker table

| # | Finding | Class | Kind |
|---|---|---|---|
| 1 | Training silently runs **end-to-end on CPU**; no placement, no record | **MUST FIX BEFORE TRAINING** | operational |
| 2 | Device contract unstated — CUDA requirement, selection, fallback | **MUST DECIDE BEFORE TRAINING** (D-S1B-015) | operational |
| 3 | **Adapter initialisation unseeded** — `run_seed` controls data order only | **MUST DECIDE BEFORE TRAINING** (D-S1B-016) | **scientific** (reproducibility) |
| 4 | Backend drift across a 20k→40k continuation is undetectable | **MUST FIX BEFORE TRAINING** (with #1) | operational |
| 5 | Model placement / optimizer-state ordering unwritten and untested | **SHOULD FIX BEFORE TRAINING** | operational |
| 6 | Every checkpoint stores the frozen encoder (~540 MB est.); `strict=False` restore | **SHOULD FIX BEFORE TRAINING** | operational |
| 7 | TF32 / matmul precision unasserted while artifacts claim `fp32` | **SHOULD FIX BEFORE TRAINING** | operational, guards a scientific constant |
| 8 | `get_device_name(0)` in the measurement tool | **DOCUMENTATION ONLY** | reporting |
| 9 | Validation-during-training device safety, mode restoration, no-dropout guarantee | **VERIFIED / NO ISSUE** | — |
| 10 | Everything in AC.16 marked VERIFIED | **VERIFIED / NO ISSUE** | — |

**STATUS: PRE-TRAIN DEVICE CONTRACT AUDIT FOUND ADDITIONAL BLOCKERS — DO NOT IMPLEMENT YET**
**THE KNOWN DEVICE BLOCKER IS CONFIRMED: TRAINING WOULD RUN SILENTLY ON CPU, RECORDED NOWHERE**
**A SECOND, SCIENTIFIC BLOCKER WAS FOUND: ADAPTER INITIALISATION IS NOT SEEDED (AC.7)**
**`run_seed` DRIVES ONLY DATA ORDER, SO A PUBLISHED SEED CANNOT REPRODUCE ITS RUN (§11)**
**TWO DECISION ENTRIES ARE REQUIRED — D-S1B-015 (DEVICE) AND D-S1B-016 (INIT SEED); NEITHER WRITTEN HERE**
**PROPOSAL §8.4 SAYS "ONE GPU" BUT SPECIFIES NO SELECTION, NO FALLBACK, NO INDEX — IT IS UNSTATED**
**FP32 IS SCIENTIFIC AND INTACT; AMP IS ABSENT; TF32 IS UNASSERTED AND SHOULD BE RECORDED**
**VALIDATION-DURING-TRAINING IS DEVICE-SAFE AND MODE-SAFE — VERIFIED FROM THE CALL GRAPH**
**TWO ZERO-UPDATE FRESH-RUNTIME PROBES ARE SPECIFIED IN AC.15 AND MUST RUN BEFORE TRAINING**
**AUDIT ONLY — NO PRODUCTION CODE, NO TEST, NO CONSTANT, NO DECISION CHANGED**
**TRAINING REMAINS UNAUTHORISED**

---

## AD. CROSS-CANDIDATE TRAINABLE-STATE LEAKAGE FOUND

**Revision 14.** The positive nominal-run-independence gate that §AC's successor
task opened found a **real scientific defect**, at HEAD
`0588b722c7c5e34bf6bda8f5703cfda80f7939be`. Implementation was stopped
immediately; **nothing was implemented**, and this section records the finding
only.

**No scientific Stage-1 campaign has ever been run, so no scientific result is
contaminated.**

### AD.1 The defect, from code

`unmark/stage1/execute.py`:

| Line | Statement | Position |
|---|---|---|
| **130** | `tokenizer, unmark_encoder, objective_cls = build_objective(revision)` | **once, BEFORE the loop** |
| **154** | `for planned in schedule:` | the nominal-run loop |
| **168** | `objective = objective_cls(unmark_encoder, provenance.weights)` | **inside the loop** |

`Stage1Objective.__init__` stores the argument **by reference** —
`self.unmark_encoder = unmark_encoder` — with no copy and no `deepcopy`.
AST-verified: `unmark_encoder` is **never rebound inside the loop**;
`build_objective` appears exactly twice in the file, at line 130 and at line 321
in the unrelated `smoke_check`; and there is no `reset_parameters`, no
`apply(...)` and no re-initialisation anywhere in `execute.py`.

`train_run` then builds its optimizer over
`objective.unmark_encoder.named_parameters()` (`trainer.py:508`) and calls
`optimizer.step()` (`trainer.py:571`), which mutates those **shared**
`nn.Parameter` objects **in place**.

> **Every nominal run within one stage command therefore shares one
> `UnmarkEncoder` instance, and hence one set of trainable adapter parameters.**

### AD.2 Consequences per stage command

| Command | Nominal runs | Effect |
|---|---|---|
| `lr-pilot` | 3 — `lr=0.0001`, `lr=0.0003`, `lr=0.001` | candidate 2 begins from candidate 1's **trained** adapter; candidate 3 from candidate 2's |
| `r-phase1` | 5 — `r=0.25 … r=4` | the five candidates form a sequential trained-state chain |
| `final-main` | 3 — `seed=36930`, `seed=7309`, `seed=5993` | seed 7309 begins from the trained 36930 adapter; seed 5993 from the trained 7309 adapter |

The LR pilot is therefore **not three fresh-start LR candidates**. It is
functionally **one sequential adapter trajectory with the learning rate changed
twice**, and the r sweep is the same. Worst of all, the three `final-main` runs
would be one continuously-trained adapter, which **destroys their intended
interpretation as independent seed replicates** — the very evidence proposal §7
requires ("If the improvement is within seed variance, there is no result").

### AD.3 This is NOT §AC.7

| | §AC.7 | AD (this finding) |
|---|---|---|
| Claim | fresh adapter initialisation **exists** but is uncontrolled by a declared seed | nominal runs 2..N receive **no fresh initialisation at all** — they inherit the previous candidate's **trained** adapter |
| Severity | reproducibility of a run's starting point | independence of the runs themselves |

**Both blockers are real.** The cross-candidate leakage is the more fundamental
of the two: while it stands, a fresh-init hash contract cannot be satisfied at
all, because runs 2..N have no fresh-init state to compare against.

### AD.4 Why the previous real gates did not detect it

**The prior PASS results stand and are not reopened.** The full four-condition
real validation (§AA.1) and the corrected real runner smoke (§AB) were
**single-run, no-update** gates: neither executed two nominal candidates through
one `execute_stage` invocation. They therefore provide **no contradictory
evidence**, and nothing here weakens them.

Recorded honestly about this audit's own earlier work: **§AC's call graph was
technically accurate** — it correctly showed `build_objective` occurring before
`train_run`. §AC was interrogating **device placement**, and it did not ask where
`build_objective` sat *relative to the nominal-run loop*. The omission was a
question never posed, not a fact misread. The new **positive** independence gate
— which demanded a hash-based contract rather than the absence of a suspicious
call — is what exposed it.

### AD.5 Proposed D-S1B-017 — nominal Stage-1 runs are independent fresh-start experiments

**Not written to `docs/spec/decisions.md` in this audit-only task.** Proposed
authoritative contract:

1. The pinned frozen PhoBERT encoder **MAY** be loaded once per stage command and
   reused between nominal runs — it is immutable scientific backbone state.
2. Reuse is legal **only if**: encoder trainable parameters remain **zero**; the
   encoder remains **eval**; it receives **no gradient**; and its **full
   `state_dict` hash is unchanged before and after every nominal run, covering
   parameters and registered buffers** (see AD.6(A)).
3. **Every** nominal run **MUST** construct a **new** adapter instance.
4. **No** trainable `Parameter` object or tensor storage may be shared between
   nominal runs.
5. **No** optimizer state and **no** sampler state may be shared between distinct
   nominal runs.
6. The adapter is initialised **on CPU** under D-S1B-016, before CUDA placement.
7. Fresh initialisation is keyed by the **scientific seed**, never by LR, `r`, or
   candidate execution order.
8. At **update 0** of every fresh nominal run: *actual adapter hash* **==**
   *expected fresh-init hash for that run_seed/init_seed*.
9. The **only** legal trained-state inheritance is the **same** nominal run's
   20k → 40k continuation.
10. A continuation **MUST** restore checkpoint adapter state and **MUST NOT**
    replace it with fresh-init state.

#### AD.5.1 The seed nuance — verified against the real schedule

Inspected at this HEAD:

| Command | `run_seed` carried by each `PlannedRun` |
|---|---|
| `lr-pilot` | **21230** for all three (`SELECTION_SEED`) |
| `r-phase1` | **21230** for all five (`SELECTION_SEED`) |
| `final-main` | **36930**, **7309**, **5993** (`TRAIN_SEEDS`) |

So the required invariant is **not** eleven distinct hashes. It is:

```
same run_seed  →  same deterministic init_seed
               →  same expected fresh-init hash
               →  INDEPENDENT adapter object / storage

different init_seed → different expected fresh-init hash
```

Concretely: **all 8 hyperparameter-selection candidates share one expected
fresh-init hash**, and the 3 `final-main` runs have 3 distinct ones — **two hash
groups, not eleven**. LR, `r` and candidate label/order **MUST NOT** enter
init-seed derivation.

#### AD.5.2 Why LR and r must not touch the init seed — the scientific reason

Because all 8 selection candidates share one `run_seed`, they share one
data-order realization and one initialisation. That makes the LR and `r` sweeps
**paired comparisons**: changing LR changes *only* LR.

If LR or `r` entered the init-seed derivation, the conclusion *"LR A beats LR B"*
would be confounded with *"initialisation A happened to be better than
initialisation B"*, and the sweep would measure a mixture of the two. The
conceptual contract is therefore:

```
scientific run_seed
    ├── domain-separated train-order stream
    └── domain-separated adapter-init stream
```

This matters most for the three `final-main` seeds, which remain the **only**
source of seed-variance evidence.

### AD.6 Requirements added for the future implementation

Recorded here as obligations on the implementation task; **none implemented**.

**(A) Frozen-encoder reuse needs structural *and* runtime guards.**
The structural guarantee **already exists and is correct**: `UnmarkEncoder.train()`
is overridden to force `self.encoder.eval()` after `super().train(mode)`, so the
frozen encoder's dropout cannot reactivate through wrapper mode transitions.
**This is not a newly discovered defect and must not be reported as one.** Reuse
across nominal runs must add, as defence in depth:

| Layer | Guarantee |
|---|---|
| existing `train()` override | **structural** |
| `encoder.training is False` asserted immediately before **every** encoder forward | **runtime** — protects against a future direct `encoder.train()` bypassing wrapper discipline |
| **full `state_dict` hash** unchanged before/after every nominal run | **mutation** — the *full* state dict, not parameters alone, so registered buffers are covered |

Plus: encoder trainable parameter count **== 0**, and encoder gradients remain
**None**.

**(B) Hash equality does not prove storage independence.** Because the 8
selection candidates *intentionally* share an init seed, equal hashes are
**expected** and prove nothing about object identity. A **positive storage
independence** test is required: build adapters A and B from the **same**
init_seed, then prove `hash(A) == hash(B)`, `A is not B`, every authoritative
trainable `Parameter` object in A `is not` its counterpart in B, and no tensor
storage is shared — then **mutate one trainable tensor of A in place** and prove
`hash(A)` changes while `hash(B)` is **exactly** unchanged. That mutation is
**test-only and is not scientific training**. The contract therefore has two
halves: **value reproducibility** *and* **storage independence**.

**(C) Optimizer parameter identity must be verified on resume.** It is not enough
that optimizer state values and devices are right. After restore, assert by
**Python object identity** (`is`, not tensor-value equality) that every
`Parameter` in `optimizer.param_groups` **is** one of the current authoritative
adapter parameters, and that every authoritative trainable adapter parameter
appears **exactly once**. No stale parameter, none from a prior nominal run, none
missing, none duplicated, and no frozen-encoder parameter. This prevents the
classic failure where a restored adapter is used for the forward while the
optimizer still points at an older adapter's parameters.

**(D) Continuation hash contract.** For a fresh nominal run let `H0` be the
expected deterministic fresh-init adapter hash, and `Hc` the checkpoint adapter
hash at its continuation point. The 20k → 40k continuation of the **same** run
must restore **`Hc`**, and must **not** reset to `H0`. A fixture that
deliberately mutates adapter state so that `Hc != H0` gives positive proof that
continuation restores trained state rather than fresh initialisation — **with no
scientific optimizer step**. That same fixture also establishes (C).

### AD.7 Intended architecture — refined

Re-instantiating PhoBERT per candidate is **not** required. The scientific
isolation boundary is the **trainable adapter state**; the frozen encoder is a
pinned immutable shared dependency.

```
per stage command:
    load pinned tokenizer once
    load pinned frozen PhoBERT encoder once; verify freeze / eval / identity
    resolve the scientific CUDA execution contract (D-S1B-015)
    the immutable encoder MAY REMAIN RESIDENT on the selected device
        across nominal runs — no CPU↔CUDA shuttling of ~135M parameters

for EACH fresh nominal run:
    derive scientific run identity
    derive domain-separated init_seed from run_seed          (D-S1B-016)
    explicitly seed CPU initialisation
    construct a NEW OrthographyInputAdapter on CPU
    establish the expected fresh-init adapter hash on CPU
    move ONLY the new adapter to the already-selected encoder device
    construct a NEW wrapper/objective:  SAME immutable encoder + NEW adapter
    verify:  encoder.training is False
             adapter fresh hash correct
             adapter storage independent
    construct a NEW optimizer bound to THIS adapter
    execute only this nominal candidate
```

This preserves **hardware-independent adapter initialisation** while avoiding
needless transfer or re-instantiation of the ~135M-parameter frozen backbone.
**The frozen immutable encoder is the only cross-nominal shared model object.
Trainable adapter state, optimizer state and sampler state are NEVER shared
between distinct nominal runs.**

### AD.8 Methodological limitation to carry forward

Confirmed by inspection (AD.5.1): all eight hyperparameter-selection candidates
run under **one** precommitted selection seed, `21230`.

> Hyperparameter candidates use a **single paired development realization**, so
> that LR and `r` comparisons hold initialisation and data order fixed.
> Consequently hyperparameter selection itself is optimised on one stochastic
> realization; seed variability is evaluated separately by the three precommitted
> `final-main` seeds.

**Classification: DOCUMENTATION / LIMITATION. Not a bug. Not a reason to add
runs. Not a reason to change the 3 + 5 + 3 protocol.** Run counts and selection
seeds are unchanged. The repository has no dedicated persistent-limitations
document, so **the FINAL CONFIGURATION FREEZE and the final repository-wide
review must carry this limitation forward** into the eventual write-up.

### AD.9 Final blocker table

| # | Finding | Class | Kind |
|---|---|---|---|
| 1 | **Cross-candidate trainable-state leakage** — nominal runs share one adapter; candidates 2..N inherit trained weights | **MUST FIX BEFORE TRAINING** | **scientific** |
| 2 | D-S1B-015 — CUDA execution selection, fail-closed, and deterministic numerical policy | **MUST DECIDE BEFORE TRAINING** | operational |
| 3 | D-S1B-016 — domain-separated, CPU-first deterministic adapter initialisation | **MUST DECIDE BEFORE TRAINING** | **scientific** (reproducibility) |
| 4 | **D-S1B-017 — nominal-run independence** (this section) | **MUST DECIDE BEFORE TRAINING** | **scientific** |
| 5 | Training silently runs end-to-end on CPU; backend recorded nowhere (§AC.4) | **MUST FIX BEFORE TRAINING** | operational |
| 6 | Backend drift across a 20k→40k continuation undetectable (§AC.8) | **MUST FIX BEFORE TRAINING** | operational |
| 7 | Placement / optimizer-state ordering unwritten and untested (§AC.6) | **SHOULD FIX BEFORE TRAINING** | operational |
| 8 | Frozen encoder in every checkpoint (~540 MB est.); `strict=False` restore (§AC.9) | **SHOULD FIX BEFORE TRAINING** | operational |
| 9 | TF32 / matmul precision unasserted while artifacts claim `fp32` (§AC.11) | **SHOULD FIX BEFORE TRAINING** | operational, guards a scientific constant |
| 10 | Training-time validation device/mode safety; `UnmarkEncoder.train()` override | **VERIFIED / NO ISSUE** | — |
| 11 | Everything in §AC.16 marked VERIFIED | **VERIFIED / NO ISSUE** | — |

### AD.10 State of this task

**No implementation occurred.** No production code, no test, and no
`docs/spec/decisions.md` change. **D-S1B-015, D-S1B-016 and D-S1B-017 are all
proposed and none is written.** The two fresh-CUDA zero-update probes (§AC.15),
the separate performance measurement, the **FINAL CONFIGURATION FREEZE**, the
final repository-wide review, and **human approval** all remain outstanding.

**STATUS: PRE-TRAIN CROSS-CANDIDATE LEAKAGE RECORDED — THREE DECISIONS AWAIT IMPLEMENTATION**
**NOMINAL RUNS SHARE ONE `UnmarkEncoder`: `build_objective` IS CALLED ONCE, OUTSIDE THE RUN LOOP**
**`lr-pilot` IS ONE ADAPTER TRAJECTORY WITH TWO LR CHANGES, NOT THREE FRESH-START CANDIDATES**
**THE THREE `final-main` SEEDS WOULD NOT BE INDEPENDENT REPLICATES — SEED-VARIANCE EVIDENCE IS LOST**
**DISTINCT FROM §AC.7: RUNS 2..N GET NO FRESH INITIALISATION AT ALL, NOT MERELY AN UNSEEDED ONE**
**PRIOR VALIDATION AND SMOKE PASSES STAND — THEY WERE SINGLE-RUN GATES AND CANNOT CONTRADICT THIS**
**§AC's CALL GRAPH WAS ACCURATE; IT ASKED ABOUT DEVICE PLACEMENT, NOT LOOP POSITION**
**EXPECTED INVARIANT IS TWO HASH GROUPS — 8 SELECTION CANDIDATES SHARE ONE, 3 SEEDS DIFFER**
**LR AND r MUST NOT ENTER INIT-SEED DERIVATION, OR THE PAIRED COMPARISON IS CONFOUNDED**
**THE FROZEN ENCODER MAY BE SHARED AND MAY STAY RESIDENT; TRAINABLE STATE NEVER MAY**
**NO SCIENTIFIC TRAINING HAS OCCURRED — NO RESULT IS CONTAMINATED**
**AUDIT ONLY — NO CODE, NO TESTS, NO decisions.md; TRAINING REMAINS FORBIDDEN**

---

## AE. DEVICE / INITIALISATION / NOMINAL-INDEPENDENCE REPAIR

**Revision 15.** Implementation of the three decisions §AC and §AD proposed,
starting from HEAD `3a5368c4b7951c9ba370611ff5e32e7d9c64e4ae`, **plus a pre-commit
correction pass that found a real defect in this section's own first draft**.

**No scientific Stage-1 campaign was run. Training is not authorised.**

> **Three corrections were made to this section before commit, and they are stated
> up front rather than buried:**
>
> 1. **The CPU-only initialisation claim was WRONG.** The first draft used
>    `torch.manual_seed` inside `fork_rng(devices=[])` and claimed it did not touch
>    CUDA RNG. It does — see AE.4. Repaired to
>    `torch.default_generator.manual_seed`.
> 2. **"Two hash groups" was WRONG.** There are **four** initialisation groups
>    across eleven runs, multiplicities **[8, 1, 1, 1]** — see AE.2. Two is the
>    number of *methodological categories*, not of groups. (§AD carries the same
>    slip and is preserved verbatim as history; this section supersedes it.)
> 3. **GPU model is now resume-blocking**, conservatively — see AE.7.
>
> **And the evidence status is separated honestly: the torch runtime tests are
> IMPLEMENTED, not EXECUTED.** See AE.9.

### AE.1 Decisions persisted

`docs/spec/decisions.md` now carries **D-S1B-015** (CUDA execution + deterministic
numerical policy), **D-S1B-016** (deterministic, domain-separated, CPU-first
adapter initialisation) and **D-S1B-017** (nominal-run independence), each with
its original proposal wording, the implemented decision, the reason, affected
files and experiments, and an explicit "proposal source updated: NO".

### AE.2 The init-seed derivation, and proof the sampler is untouched

New pinned tag `UNMARK-STAGE1-v1|adapter-init`, used with the **existing**
`derive_seeds` primitive — no second hash scheme was invented:

```
adapter_init_seed(run_seed) = derive_seeds(f"UNMARK-STAGE1-v1|adapter-init|{run_seed}", 1)[0]
```

| `run_seed` | used by | derived `init_seed` |
|---|---|---|
| **21230** (`SELECTION_SEED`) | all **8** selection candidates | **3203** |
| **36930** | `final-main` seed 1 | **51800** |
| **7309** | `final-main` seed 2 | **45833** |
| **5993** | `final-main` seed 3 | **15758** |

All four are distinct, none collides with any existing role seed (import-time
guard added), and each differs from the `run_seed` it derives from.

**The sampler's semantics are unchanged, and that is asserted rather than
asserted-to.** A test reads the real `DeterministicSampler(...)` call in
`train_run` from the AST and requires its `seed=` argument to be literally
`provenance.run_seed`. A second test proves the trainer never *calls*
`adapter_init_seed` — checked on the call graph, because the trainer's docstrings
legitimately mention it.

**Paired-selection rationale.** `adapter_init_seed` takes `run_seed` and nothing
else — enforced by its signature, and tested. So the eight selection candidates
share one initialisation and the LR/`r` sweeps vary only their target.

**The grouping is FOUR, not two.** Across the eleven nominal runs there are
**four distinct expected fresh-init hashes**, with multiplicities **[8, 1, 1, 1]**:

| Group | Runs | `init_seed` |
|---|---|---|
| 1 | the **8** selection candidates | 3203 |
| 2 | `final-main` 36930 | 51800 |
| 3 | `final-main` 7309 | 45833 |
| 4 | `final-main` 5993 | 15758 |

There are **two methodological categories** — paired selection, and seed-varied
final-main — and an earlier draft of this section (and §AD) wrote "two hash
groups", conflating the category count with the group count. A torch-free test now
asserts `Counter(...) == {3203: 8, 51800: 1, 45833: 1, 15758: 1}` so the number
cannot drift again.

### AE.3 Construction refactor

`build_objective` is retained unchanged for the CPU-capable `smoke` and
measurement paths. `execute_stage` no longer uses it:

| Scope | What |
|---|---|
| **stage** | `build_backbone(revision)` → tokenizer + frozen encoder + hidden size; device resolved; encoder placed **once**; `E0` = full `state_dict` hash |
| **per nominal run** | `adapter_init_seed` → `fresh_adapter` on **CPU** → `H0` → `.to(device)` → **new** `UnmarkEncoder(shared encoder, new adapter)` → **new** `Stage1Objective` → **new** optimizer → **new** sampler |
| **after each run** | `require_frozen_backbone_unchanged`: full `state_dict` hash `== E0`, zero trainable parameters, no gradients |

The frozen backbone is **never moved inside the loop** — asserted structurally, so
~135M parameters are not shuttled per candidate.

### AE.4 Initialisation isolation — CORRECTED

**The first draft of this repair was wrong, and the claim it made was unsound.**
It used:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(init_seed)          # WRONG
```

and asserted that no CUDA RNG was touched. `torch.manual_seed`'s documented
contract is to seed **all devices**: it calls `torch.cuda.manual_seed_all(seed)`
before seeding the CPU generator. `fork_rng(devices=[])` snapshots and restores
**only** the CPU generator. So on a CUDA process the pairing perturbs every CUDA
generator and never restores it — and when CUDA is *not* yet initialised,
`manual_seed_all` defers through `_lazy_call`, queueing a seed that fires later at
CUDA init. Either way an operation that is supposed to be a pure CPU construction
silently alters accelerator RNG, which is exactly what D-S1B-016 forbids.

**Repaired to the CPU default generator specifically:**

```python
with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(int(init_seed))
    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=hidden_size))
```

`torch.default_generator` **is** the CPU generator that every CPU
`reset_parameters` draws from, so seeding it is exactly sufficient and strictly
confined: no CUDA generator is read, written or initialised, and the fork restores
ambient CPU state on exit. A call-graph test proves `fresh_adapter` calls
`torch.default_generator.manual_seed` and never `torch.manual_seed`,
`manual_seed_all`, `.cuda()`, `.to()` or `torch.device`.

The runtime file adds a CUDA-gated proof — `torch.cuda.get_rng_state_all()`
compared **byte for byte** across `fresh_adapter` — a byte-exact CPU
`get_rng_state()` restoration check, and a check that `fresh_adapter` does not
initialise CUDA as a side effect.

### AE.5 Hash contract

One canonical mechanism, `trainable_state_hash`: sorted key names, each
contributing name, dtype, shape and raw bytes, with tensors moved to CPU and made
contiguous first — so a hash computed on CPU compares equal against a model
already placed on an accelerator. `module_state_hash` applies it to a **full**
`state_dict` for the encoder's immutability check, covering buffers.

`execute_stage` asserts, per run, that the fresh adapter's hash equals
`expected_fresh_init_hash(hidden, run_seed)` **and** that placement did not change
it.

### AE.6 Storage independence, proven positively

Because the eight selection candidates deliberately share values, hash equality is
**expected** and proves nothing. The runtime tests build two adapters from one
seed and assert equal hashes, `a is not b`, pairwise `pa is not pb`, distinct
`data_ptr()`, and then **mutate one in place** (TEST-ONLY) and prove the other's
hash is byte-identical. A further test mutates a "trained" candidate and shows the
next two candidates still start from the expected hash — §AD's leakage, inverted.

### AE.7 Device, numerics, fp32/TF32, fingerprint

`unmark/stage1/device.py` is the single resolver. CUDA required, fail closed, no
CPU fallback, logical `torch.device("cuda")` honouring `CUDA_VISIBLE_DEVICES`, no
physical index anywhere.

`enforce_numerical_policy` sets deterministic algorithms, cuDNN
deterministic/benchmark, `float32_matmul_precision="highest"` and TF32 off;
`verify_numerical_policy` re-asserts all of it afterwards, so a global setting
changed elsewhere in the process cannot reach a run whose artifact claims `fp32`.
**AMP remains absent** — no `autocast`, `GradScaler`, `fp16` or `bf16` anywhere.

**cuBLAS timing is handled, not guessed.** `CUBLAS_WORKSPACE_CONFIG` is read when
the cuBLAS handle is created, so `require_deterministic_cublas_workspace` sets it
only while `torch.cuda.is_initialized()` is False and **refuses** otherwise,
rather than setting it too late and claiming a determinism the run does not have.
It is called first in `execute_stage`, before anything touches CUDA.

The `ExecutionFingerprint` records backend, device, GPU name, compute capability,
torch/CUDA/cuDNN versions, deterministic and TF32 states, cuBLAS workspace and
matmul precision. **13 fields are resume-blocking.**

**GPU model name now blocks — corrected, and deliberately conservative.** The
first draft excluded it on the reasoning that compute capability is the
numerically relevant property. That is what one would *expect*, but nothing in
this repository has demonstrated that two different GPU models sharing a
capability produce byte-identical interrupted-vs-uninterrupted training, and this
project's reproducibility claim is too strong to rest on an untested expectation.
A continuation therefore stays on the same model until a CUDA experiment proves
cross-model identity; relaxing it is a later explicit decision.

**Still not blocking:** the logical `device` index, which `CUDA_VISIBLE_DEVICES`
renumbers freely, and the physical GPU UUID, which is not recorded as identity at
all. Neither changes the arithmetic, so neither may block a legitimate
crash-resume onto the same model of card.

**cuBLAS ordering was re-traced after the RNG repair.** Inside `execute_stage`
only `verify_scientific_inputs` precedes `require_deterministic_cublas_workspace`,
and the preflight touches no torch at all; `fresh_adapter` has **zero** CUDA
references in its call graph and runs far later, inside the run loop. Nothing can
initialise CUDA before the workspace configuration is settled.

### AE.8 Checkpoint, optimizer, continuation

Persisted model state is now `adapter.state_dict()` — proven to be exactly the
trainable parameter set, with **no** `encoder.` keys — restored with
`strict=True`. `CHECKPOINT_SCHEMA_VERSION` is `stage1-checkpoint-v2`; v1 fails
closed and no migration is offered, because no scientific checkpoint exists.
`execution` joins the payload and `REQUIRED_CHECKPOINT_KEYS`.

`require_optimizer_parameter_identity` asserts by **object identity** that
optimizer parameters are exactly the current adapter's, each once — no stale,
duplicate, missing, foreign or frozen parameter — and runs both after fresh
construction and after restore. `require_optimizer_state_device` walks optimizer
state **recursively** and asserts placement, treating Adam's scalar `step`
correctly; it asserts the postcondition of supported PyTorch behaviour rather than
re-implementing migration.

**Continuation** restores `Hc`, never `H0`, proven by a fixture that deliberately
mutates state so `Hc != H0`, rebuilds deterministically, restores strictly, and
then re-checks optimizer identity — **with no scientific optimizer step**.

**Checkpoint-size evidence.** Old model state was the whole wrapper — frozen
encoder included; new state is the adapter alone, **3 551 232** parameters
(≈14 MB at fp32 versus an estimated ≈540 MB). The exact real-model file size is
deferred to the fresh-runtime probe.

### AE.9 Test results

| Suite | Result |
|---|---|
| **Full lightweight suite** | **3 575 passed, 101 skipped, 0 failed** |
| Torch-free structural (`run_independence` + `device_contract`) | **33 passed** |
| **Torch runtime CPU tests** | **0 passed, 0 failed — the whole module SKIPPED (no torch on this machine)** |
| **CUDA-only tests** | **0 passed, 0 failed — not reached; the module skipped before gating** |

New: `tests/test_stage1_run_independence.py` — **25 tests, torch-free**, running in
the ML-free venv: seed derivation and its exclusions, the locked seed table, the
untouched sampler contract, the construction boundary, device-contract ordering,
adapter-only checkpoints, strict restore, and both optimizer contracts.

New: `tests/test_stage1_run_independence_runtime.py` — **torch-gated**:
determinism, CPU-only initialisation, RNG isolation and restoration, hash
contract, storage independence and mutation isolation, cross-candidate isolation,
optimizer identity and state-device assertions, strict-restore failure modes,
the `H0`/`Hc` continuation fixture, encoder immutability including buffers, the
runtime eval guard, and the device/numerics/fingerprint contracts.

Updated: the §Y device test now forbids a **physical** GPU index rather than the
string `"cuda"` — `device.py` legitimately names the logical device — plus two new
assertions that only `device.py` names CUDA and that the scientific CLI exposes no
device/determinism/init override. The §W preflight-ordering test now targets
`build_backbone`.

#### AE.9.1 IMPLEMENTED is not EXECUTED

> **Superseded by AE.11 (2026-08-24).** Every row below was accurate on the
> ML-free development machine. The fresh-CUDA runtime has since executed them;
> see **AE.11.3** for the counts and **AE.11.8** for the two claims that remain
> reserved. The table is kept unedited as the record of what was pending.

**There is no torch anywhere on this machine** — the development venv is
deliberately ML-free and no pinned-torch environment was available before commit,
so `tests/test_stage1_run_independence_runtime.py` **skipped in its entirety**.
The first draft of this section described several of its contracts as though they
were established. They are not. The status of every claim whose evidence lives in
that file:

| Contract | Status |
|---|---|
| Deterministic init: same seed → same bytes, different seed → different | **IMPLEMENTED, NOT EXECUTED** |
| CPU-only initialisation; ambient CPU RNG restored byte-for-byte | **IMPLEMENTED, NOT EXECUTED** |
| Storage independence + in-place mutation isolation | **IMPLEMENTED, NOT EXECUTED** |
| Cross-candidate isolation (a mutated candidate cannot move the next) | **IMPLEMENTED, NOT EXECUTED** |
| Fresh-init hash reproduction; hash survives a device move | **IMPLEMENTED, NOT EXECUTED** |
| Locked 3 551 232 parameter count at `HIDDEN_SIZE` | **IMPLEMENTED, NOT EXECUTED** |
| Optimizer parameter object-identity contract, and its failure modes | **IMPLEMENTED, NOT EXECUTED** |
| Optimizer state-device recursive assertion | **IMPLEMENTED, NOT EXECUTED** |
| Adapter-only state; strict-restore rejects missing/unexpected/wrong-shape | **IMPLEMENTED, NOT EXECUTED** |
| `H0` vs `Hc` continuation fixture | **IMPLEMENTED, NOT EXECUTED** |
| Encoder full-`state_dict` immutability, buffers included | **IMPLEMENTED, NOT EXECUTED** |
| Runtime encoder-eval guard | **IMPLEMENTED, NOT EXECUTED** |
| Device fail-closed, perturbed-precision, late-cuBLAS, fingerprint blocking | **IMPLEMENTED, NOT EXECUTED** |
| **CUDA generators byte-identical across `fresh_adapter`** | **IMPLEMENTED — needs a GPU** |
| **CUDA resume byte-identity** | **IMPLEMENTED — needs a GPU** |

**What IS executed evidence:** the **25** torch-free structural tests in
`tests/test_stage1_run_independence.py`, and the full lightweight suite. Those
cover the seed derivation and its exclusions, the four-group table, the untouched
sampler contract, the construction boundary, device-contract ordering,
adapter-only checkpoints, strict restore, both optimizer contracts, and — by AST —
that `fresh_adapter` seeds only the CPU default generator.

Everything else above is **written and reviewed, not run**. The next environment
must execute the CPU-capable half under pinned torch **before** the two acceptance
probes, and the CUDA-gated half on the GPU.

**The repository's byte-identical resume claim remains scoped to CPU.** This
section does not claim CUDA byte-identity, and does not claim runtime proof of
the contracts marked above.

### AE.10 What remains

| Requirement | Status |
|---|---|
| **Execute the CPU-capable torch runtime tests under pinned torch** (AE.9.1) | **REQUIRED, BEFORE the probes** |
| Fresh-runtime Probe 1 — real training-entry placement, **zero update** | **REQUIRED** |
| Fresh-runtime Probe 2 — real checkpoint/resume, **zero update** (optimizer construction allowed, execution not) | **REQUIRED** |
| CUDA-gated tiny resume byte-identity test executed on a GPU | **REQUIRED** |
| Non-scientific performance measurement — one full four-condition validation, representative training-path cost | **REQUIRED, separate** |
| **FINAL STAGE-1 CONFIGURATION FREEZE**, machine-readable and mechanically compared against code | **REQUIRED** |
| Final proposal-aware repository-wide review | **REQUIRED** |
| Human approval | **REQUIRED** |

### AE.11 FRESH-CUDA RUNTIME VERIFICATION — 2026-08-24

**Everything above this subsection is preserved as written.** AE.9.1's
"IMPLEMENTED, NOT EXECUTED" table is history: it was accurate on the ML-free
development machine, and this subsection is what closes it. The evidence below
binds specifically to implementation commit
**`3c3489b9701fb45fc92e0737c1b765f1b0d2aebd`**, on a **fresh** Colab CUDA runtime
created after the previous runtime had been deleted.

#### AE.11.1 Environment

| | |
|---|---|
| Python | **3.13.15** |
| torch | **2.11.0+cu128** |
| torch CUDA build | **12.8** |
| transformers | **4.57.6** |
| GPU | **NVIDIA RTX PRO 6000 Blackwell Server Edition** |
| Compute capability | **12.0** |
| cuDNN | **91900** |
| At the gate | `torch.cuda.is_available() == True`, `torch.cuda.is_initialized() == False` |

> **The image shipped with transformers 5.15.0.** It was corrected to the locked
> **4.57.6** and the Python session **restarted** before authoritative
> verification. **Nothing observed under 5.15.0 is used as evidence here.**

#### AE.11.2 Pinned inventory, provisioned by the repository's own fetcher

| | |
|---|---|
| Source | `all-vietnamese-syllables.txt` by `hieuthi` |
| Gist revision | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| SHA-256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Bytes | **116 290** |
| Raw / unique canonical / unique stripped | **17 974 / 17 954 / 2 489** |
| Collisions after strip | **15 465** |
| License | `NO_EXPLICIT_LICENSE` |

The fetcher reported **"Verified and cached"**; no alternate or moving upstream
revision was accepted, and the raw list remains outside Git.

#### AE.11.3 Regression evidence

Before provisioning, failures were **fail-closed `InventoryUnavailable` /
`EligibilityUnresolved`** — the §W contract working, **not** implementation
failures. After provisioning the exact pinned inventory:

| Suite | Result |
|---|---|
| **Stage-1 suite** | **1 344 passed, 1 skipped, 0 failed, 0 errors** — 91.65 s |
| **Full suite** | **3 754 passed, 1 skipped, 0 failed, 0 errors** — 100.44 s |

**The single skip, stated plainly rather than hidden.**
`tests/test_stage1_run_independence_runtime.py:126` —
`test_initialisation_does_not_initialise_cuda_as_a_side_effect` — skipped with
*"CUDA was already initialised by an earlier test"*. That is a **test-order-sensitive
skip by design**: the contract is only meaningful in a virgin process, and the
test declines rather than asserting something vacuous. It was then executed
**alone in a fresh Python subprocess**: **1 passed in 0.76 s**.

So the combined suite is **3 754 passed, 1 skipped** — **not** "3 755 passed" —
and the skipped contract is separately evidenced under the condition it requires.

#### AE.11.4 Fresh-initialisation hashes — the four H0 values

Executed with the real Stage-1 initialisation helpers at `hidden_size=768`:

| `run_seed` | `init_seed` | H0 |
|---|---|---|
| **21230** | **3203** | `4ef7ee1a2357d0b9819225aca6708d2ad04614ceda8191dc5a46176a47fa3d25` |
| **36930** | **51800** | `cf1a28cf640f6697075d086486cc5395e9a9ff0ef994ca0a06576325f7de68ad` |
| **7309** | **45833** | `d90186c0f77e434d73f173426a401a99aa93e6c0d90a5bb38efd732bba607516` |
| **5993** | **15758** | `660db230168dd5cf6fc9758aa9e868e70fc5f44a60a8cb0b6422160af1be2842` |

Runtime assertions: **distinct H0 groups = 4**, schedule multiplicities
**[8, 1, 1, 1]**. The eight selection candidates intentionally share `run_seed`
21230 and therefore share H0 — that is the paired design of D-S1B-016, not a
collision. **Four groups, not two.**

#### AE.11.5 cuBLAS-before-CUDA ordering, measured in a fresh subprocess

| Step | Observation |
|---|---|
| Process start | `CUDA available: True`, `CUDA initialized: False` |
| `require_deterministic_cublas_workspace()` | returned **`:4096:8`**, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, **`CUDA initialized == False`** |
| Numerical policy established | deterministic algorithms **True**; `cudnn.deterministic` **True**; `cudnn.benchmark` **False**; float32 matmul **highest**; `cuda.matmul.allow_tf32` **False**; `cudnn.allow_tf32` **False**; **`CUDA initialized == False`** |
| **Then** device resolution | `device = cuda`, `CUDA initialized == True` |

This is the measurement §AC.6 and AE.7 asked for: the deterministic cuBLAS
workspace was in place **before** anything initialised CUDA, so the policy is
effective rather than merely declared.

#### AE.11.6 Authoritative execution fingerprint

```
backend                   cuda
device                    cuda
gpu_name                  NVIDIA RTX PRO 6000 Blackwell Server Edition
compute_capability        12.0
torch_version             2.11.0+cu128
cuda_version              12.8
cudnn_version             91900
deterministic_algorithms  true
cudnn_deterministic       true
cudnn_benchmark           false
cublas_workspace_config   :4096:8
float32_matmul_precision  highest
cuda_matmul_allow_tf32    false
cudnn_allow_tf32          false
```

`AUTHORITATIVE EXECUTION FINGERPRINT: PASS`. **AMP remains absent** — no
`autocast`, `GradScaler`, fp16 or bf16 anywhere.

#### AE.11.7 Resume-blocking set — 13 fields, confirmed at runtime

`backend`, `gpu_name`, `compute_capability`, `torch_version`, `cuda_version`,
`cudnn_version`, `deterministic_algorithms`, `cudnn_deterministic`,
`cudnn_benchmark`, `cublas_workspace_config`, `float32_matmul_precision`,
`cuda_matmul_allow_tf32`, `cudnn_allow_tf32`.

**GPU model and compute capability both block. Logical CUDA index does not, and
physical GPU UUID is not recorded as identity at all.** The conservative contract
stands: **no cross-GPU-model byte identity has been established or claimed.**

#### AE.11.8 ★ Two claims deliberately RESERVED

Traced against the committed tests at `3c3489b9`, because a passing suite total is
not a substitute for a named contract:

| Claim | Status | Why |
|---|---|---|
| **CUDA interrupted-vs-uninterrupted byte identity** | **NOT ESTABLISHED** | No CUDA resume-equivalence test exists in the tree. `test_stage1_training_resume.py` and `test_stage1_training_resume_state.py` contain **zero** CUDA references, so on this runtime they still exercised **CPU** tensors. The claim therefore **remains scoped to CPU**, exactly as before |
| **Optimizer-state placement on CUDA** | **MECHANISM PROVEN, CUDA PLACEMENT NOT** | `test_optimizer_state_device_is_asserted_recursively` executed, but asserts `torch.device("cpu")` and carries no `needs_cuda` gate. The recursive traversal and its failure mode are evidenced; landing on a **CUDA** device is not |

**What did execute on the real GPU** — four CUDA-gated tests, none of which
skipped: `test_initialisation_leaves_every_cuda_generator_byte_identical`,
`test_initialisation_does_not_depend_on_cuda_rng`, and from
`test_stage1_device_contract_runtime.py`
`test_a_cuda_objective_receives_cuda_tensors_from_a_cpu_prepared_batch` and
`test_removing_the_transfer_really_would_fail`. So §Y's device boundary and
§AE.4's CUDA-RNG isolation are now **real-hardware evidence**.

#### AE.11.9 What this subsection establishes, and what it does not

**Established:** pinned environment identity; exact inventory identity; runtime
execution of the Stage-1 hardening; Stage-1 and full regression suites with zero
failures and zero errors; the fresh-process CUDA-initialisation side-effect
contract; the exact init-seed table; the four H0 hashes; **[8, 1, 1, 1]**
multiplicities; the CUDA numerical policy; cuBLAS-before-CUDA ordering; the
authoritative execution fingerprint; the 13-field resume-blocking policy.

**Not established:** the two reserved claims in AE.11.8, and **every real
prepared-corpus contract** — no acceptance probe, no validation, no performance
measurement was run here. This runtime verified the *implementation*, not the
science.

#### AE.11.10 Remaining gates, in order

1. **Human review of this runtime evidence**
2. Audit-only commit/push by the human
3. **Delete this verification runtime**
4. New fresh CUDA runtime at the exact committed HEAD
5. Real scientific-entry **Probe 1**, zero optimizer updates
6. Real checkpoint/resume **Probe 2**, zero optimizer updates
7. Separate populated-optimizer-state placement fixture (see the nuance below)
8. CUDA tiny resume-equivalence confirmation — **still owed**, per AE.11.8
9. Non-scientific performance measurement
10. **FINAL STAGE-1 CONFIGURATION FREEZE**
11. Final proposal-aware repository-wide review
12. Explicit human approval
13. **Only then** may the first scientific `optimizer.step` occur

> **Nuance for Probes 1 and 2.** A virgin `AdamW` has **empty** state before its
> first `step`, so `exp_avg` / `exp_avg_sq` / `step` do not exist to be placed. A
> zero-update probe **must not** manufacture a scientific optimizer step merely to
> populate them. Populated optimizer-state device migration belongs in a separate
> **synthetic, non-scientific** fixture — which is also what would close the second
> reserved claim in AE.11.8.

**AE.11 STATUS: FRESH-CUDA RUNTIME VERIFICATION PASS — READY FOR HUMAN REVIEW**
**BOUND TO IMPLEMENTATION COMMIT `3c3489b9701fb45fc92e0737c1b765f1b0d2aebd`**
**THIS CLOSES THE IMPLEMENTATION/RUNTIME-VERIFICATION GATE ONLY — IT IS NOT "STAGE-1 TRAINING READY"**
**STAGE-1 1 344 PASSED / 1 SKIPPED / 0 FAILED; FULL 3 754 PASSED / 1 SKIPPED / 0 FAILED**
**THE ONE SKIP IS ORDER-SENSITIVE BY DESIGN AND PASSED ALONE IN A FRESH SUBPROCESS**
**FOUR DISTINCT H0 HASHES, MULTIPLICITIES [8, 1, 1, 1] — RECORDED IN FULL**
**CUBLAS `:4096:8` WAS SET WHILE CUDA WAS STILL UNINITIALISED — MEASURED, NOT ASSUMED**
**13 RESUME-BLOCKING FIELDS; GPU MODEL BLOCKS; NO CROSS-MODEL IDENTITY IS CLAIMED**
**RESERVED: CUDA RESUME BYTE-IDENTITY, AND OPTIMIZER-STATE PLACEMENT *ON CUDA*, ARE NOT ESTABLISHED**
**NO PROBE, NO VALIDATION, NO PERFORMANCE RUN, NO SCIENTIFIC TRAINING OCCURRED**

---

#### AE.12 HISTORICAL SNAPSHOT — the pre-runtime status block, retained verbatim

> ### ⚠ HISTORICAL ONLY — NOT THE CURRENT STATUS
>
> **The status block below is a snapshot of the ML-free development machine, taken
> before any torch or CUDA runtime existed.** It is reproduced **verbatim and
> unedited** as audit history, and nothing in it should be read as describing the
> repository today.
>
> Several of its lines are **now out of date** — in particular *"THE TORCH RUNTIME
> TESTS ARE IMPLEMENTED, NOT EXECUTED"*, *"NO TORCH EXISTS ON THIS MACHINE"* and
> *"THE FULL LIGHTWEIGHT SUITE PASSES, BUT THE TORCH RUNTIME FILE SKIPPED
> ENTIRELY"*. Those tests have since been executed on real hardware.
>
> **For current runtime status, read §AE.11**, which supersedes this block in its
> entirety.
>
> **Two claims are NOT superseded and remain open** (§AE.11.8): CUDA
> interrupted-vs-uninterrupted byte identity is **NOT ESTABLISHED**, and
> optimizer-state placement **on CUDA** is **NOT ESTABLISHED**. §AE.11 does not
> claim Stage-1 training readiness.

**STATUS: PRE-TRAIN IMPLEMENTATION RUNTIME VERIFICATION INCOMPLETE — DO NOT COMMIT YET**
**THE TORCH RUNTIME TESTS ARE IMPLEMENTED, NOT EXECUTED: NO TORCH EXISTS ON THIS MACHINE**
**THREE PRE-COMMIT CORRECTIONS: CPU-ONLY RNG, FOUR HASH GROUPS, GPU MODEL BLOCKS RESUME**
**`torch.manual_seed` SEEDS ALL DEVICES — REPLACED BY `torch.default_generator.manual_seed`**
**FOUR INIT-HASH GROUPS ACROSS ELEVEN RUNS, MULTIPLICITIES [8, 1, 1, 1] — NOT TWO**
**D-S1B-015, D-S1B-016 AND D-S1B-017 PERSISTED; THE §AD LEAKAGE IS REPAIRED**
**EVERY NOMINAL RUN NOW BUILDS A FRESH ADAPTER; THE FROZEN ENCODER IS THE ONLY SHARED MODEL STATE**
**INIT SEEDS: 21230→3203 (ALL 8 SELECTION CANDIDATES), 36930→51800, 7309→45833, 5993→15758**
**LR AND r CANNOT ENTER THE DERIVATION — THE SELECTION SWEEPS STAY PAIRED**
**THE SAMPLER STILL RECEIVES `run_seed` ITSELF — DATA ORDER IS UNCHANGED, AND AST-ASSERTED**
**INITIALISATION IS CPU-ONLY INSIDE `fork_rng`, SO IT IS HARDWARE-INDEPENDENT AND RESTORES AMBIENT RNG**
**CUDA REQUIRED AND FAIL-CLOSED; DETERMINISTIC POLICY ENFORCED *AND* RE-ASSERTED; TRUE fp32 MATMUL**
**CHECKPOINTS ARE ADAPTER-ONLY AND STRICT; SCHEMA v2 FAILS CLOSED ON v1; NO MIGRATION NEEDED**
**THE FULL LIGHTWEIGHT SUITE PASSES, BUT THE TORCH RUNTIME FILE SKIPPED ENTIRELY**
**CUDA BYTE-IDENTICAL RESUME IS NOT YET ESTABLISHED; THE CLAIM STAYS SCOPED TO CPU**
**NO SCIENTIFIC TRAINING OCCURRED — TRAINING REMAINS UNAUTHORISED**

---

## AF. REAL ZERO-UPDATE ACCEPTANCE, AND A NEWLY QUANTIFIED PERFORMANCE BLOCKER

**Revision 17.** Fresh CUDA runtime at HEAD
`ac20cfb786ca770a7296339d48263ff8e09acf66`. Every acceptance gate §AE.11.10 left
open has now run against the **real** prepared corpus and the **real** PhoBERT —
and all of them passed. In doing so they produced the first real timing of the
training path, which turns the long-deferred performance concern into a
**material, quantified blocker**.

**Scientific `optimizer.step` count remains ZERO. Training is not authorised.**

### AF.1 Environment and corpus

Same locked stack as §AE.11.1 — torch **2.11.0+cu128**, CUDA **12.8**, cuDNN
**91900**, transformers **4.57.6**, **NVIDIA RTX PRO 6000 Blackwell Server
Edition**, compute capability **12.0** — under the locked numerical policy:
deterministic algorithms **true**, `cudnn.deterministic` **true**,
`cudnn.benchmark` **false**, `CUBLAS_WORKSPACE_CONFIG` **`:4096:8`**, float32
matmul **highest**, CUDA matmul TF32 **false**, cuDNN TF32 **false**.

| Prepared corpus | |
|---|---|
| Total / train / dev chunks | **2 633 067 / 2 621 624 / 11 443** |
| Membership digest | `250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6` |
| Authoritative verification | **PASS** |

### AF.2 Acceptance results — all PASS

| Gate | Result |
|---|---|
| Prepared-corpus verification | **PASS** |
| **Probe 1 — real training-entry placement, ZERO update** | **PASS** |
| **Probe 2 — real checkpoint/resume, ZERO update** | **PASS** |
| **CUDA interrupted-vs-uninterrupted exact equivalence** | **PASS** |
| **Populated optimizer-state CUDA placement** | **PASS** |
| Full four-condition real validation | **PASS** |
| Representative backward / no-step training path | **PASS** |
| Repository safety | **PASS** |
| Scientific `optimizer.step` count | **ZERO** |
| Official UIT-VSFC TEST | **SEALED / UNTOUCHED** |

**Full four-condition validation.** All **11 443** dev chunks, FULL / P100 / P50 /
STRIP_ALL, batch 128, validation corruption seed **19225**, **810 real forwards**;
optimizer constructed **false**, backward calls **0**, optimizer steps **0**,
parameter updates **0**, parameter hashes **identical** before and after.
Recurring validation **304.526 s**; **41** evaluations per 20k run; projected
recurring validation **12 485.6 s** per run; one-time condition setup **1 584.79 s**.

**Representative training path** — explicitly **NON-SCIENTIFIC** acceptance
measurement: real PhoBERT, real Stage-1 adapter, real prepared examples, batch
128, one warm-up plus three measured batches. **Backward executed;
`optimizer.step` NEVER executed.** Adapter hashes identical before/after, encoder
hashes identical before/after, optimizer state remained **empty**, all **8**
adapter gradient tensors finite and non-zero, encoder gradient tensors **0**.

### AF.3 ★ CORRECTION — Adam's scalar `step` is legitimately on CPU

**§AC.15's Probe-2 item 3 is SUPERSEDED.** It read:

> *"assert every optimizer state tensor (`exp_avg`, `exp_avg_sq`, `step`) is on the
> selected device"*

That was **wrong about `step`**. The runtime evidence, and PyTorch's own AdamW
semantics, are:

* `exp_avg` and `exp_avg_sq` are on **`cuda:0`** after restore — correct, and
  asserted;
* Adam's scalar `step` tensors are on **CPU** — **legitimate PyTorch AdamW
  behaviour**, not a defect.

The production implementation was already right: `require_optimizer_state_device`
in `unmark/stage1/trainer.py` explicitly exempts zero-dimensional scalar `step`
(`if value.dim() == 0 and path.endswith("step"): return`). The authoritative
production optimizer-state postcondition therefore **PASSED**.

**No production change is required, and none should be made to force Adam's
scalar `step` onto CUDA.** §AC.15 is preserved verbatim as history; this entry
supersedes its item 3.

### AF.4 ★ The two §AE.11.8 reserved claims are now CLOSED — with one caveat

A tiny **non-scientific** CUDA fixture, driving the **production**
checkpoint/resume APIs, established:

| Property | Result |
|---|---|
| Uninterrupted 16-update execution | **PASS** |
| Interrupted at update 8 → checkpointed → process rebuilt → resumed to 16 | **PASS** |
| Final adapter tensors | **exactly equal** |
| Final optimizer state | **exactly equal** |
| Sampler state | **exactly equal** |
| Update count | **exactly equal** |
| Validation-point history | **exactly equal** |

So **CUDA interrupted-vs-uninterrupted byte identity is ESTABLISHED**, and
**populated optimizer-state CUDA placement is ESTABLISHED** (under the correct
Adam semantics of AF.3). Both reserved claims in §AE.11.8 are discharged.

> **Caveat, stated rather than glossed.** This was a **one-off fixture run**, not
> persistent regression coverage. **The repository still contains no CUDA
> resume-equivalence test**; `test_stage1_training_resume.py` and
> `test_stage1_training_resume_state.py` remain CPU-only. Nothing here should be
> read as "CUDA regression coverage exists".
>
> **Stale docstring to fix (documentation only, NOT changed in this task):**
> `tests/test_stage1_run_independence_runtime.py:7` states *"Synthetic optimizer
> steps appear ONLY in the CUDA resume-equivalence test"* — **no such test exists
> in that file.** Proposed replacement for lines 7–9:
>
> ```
> No synthetic optimizer step is taken anywhere in this file. (An earlier draft of
> this docstring referred to a CUDA resume-equivalence test that was never added;
> CUDA resume equivalence was established once by an out-of-tree fixture — see
> Audit 030 §AF.4 — and still has no persistent regression test.)
> ```

### AF.5 ★ NEW BLOCKER — the training path is preparation-bound

**Measured means, batch 128:**

| Stage | Seconds | Share of pre-step |
|---|---|---|
| `prepare_example` ×128 | **4.723522706** | **79.05 %** |
| collate | **0.498691270** | 8.35 % |
| H2D | **0.000928833** | 0.02 % |
| forward | **0.446289907** | 7.47 % |
| backward | **0.306093282** | 5.12 % |
| **pre-step total** | **5.975525999** | 100 % |

CPU-side (**prepare + collate**) is **87.40 %**; all GPU work (H2D + forward +
backward) is **12.60 %**. Peak GPU allocated **27 566 880 768 B** (≈25.7 GiB),
reserved **28 022 145 024 B** — comfortable on a ~97.9 GiB card, so **memory is
not the constraint**.

**Campaign projection — lower bound, before `optimizer.step`, checkpoint I/O or
any continuation:**

| Quantity | Value |
|---|---|
| Pre-step path, one 20k run | 20 000 × 5.975526 s = **119 510.5 s ≈ 33.20 h** |
| Recurring validation, one 20k run | **12 485.6 s ≈ 3.47 h** |
| **One nominal 20k run** | **≈ 131 996 s ≈ 36.67 h** |
| **All 11 nominal runs** | **≈ 403.3 h ≈ 16.8 days** single-GPU |

A 40k continuation on any run adds roughly another 36.7 h to that run.

**Amdahl ceilings — theoretical, not achieved.** With preparation at 79.05 % of
the pre-step path, holding everything else fixed:

| Target | Required preparation speedup |
|---|---|
| 2× total training-path | **2.72×** |
| 4× total training-path | **19.5×** |
| Ceiling, preparation → 0 | **4.77×** total (floor 1.252 s/step) |
| Ceiling, preparation **and** collate → 0 | **7.93×** total (floor 0.753 s/step) |

> **Consequence worth stating plainly: prefetch/overlap alone does not solve
> this.** Overlapping CPU preparation with GPU compute replaces `CPU + GPU` by
> `max(CPU, GPU)`. Here CPU is **7×** GPU, so perfect overlap buys only ≈12.6 %.
> Overlap becomes valuable *after* preparation is made much cheaper, not instead
> of it.

### AF.6 Root cause — traced, and it is NOT tokenisation

> **Wording corrected by §AG (Revision 18).** "It is NOT tokenisation" overstated
> what was then known — the local profile used a *stub* tokenizer, so it could not
> measure the real one. The real-runtime measurement has since put the pinned
> tokenizer at **~5.01 %** of preparation wall time. The better-established
> statement is: **the real tokenizer accounts for only ~5.01 % of measured
> preparation wall time; the dominant cost is the deterministic
> orthography/alignment path.** The conclusion is unchanged and now measured; the
> original heading is left as written.


`prepare_example` → `prepare_with_condition` runs three streams: PATH R
(reference), PATH C (clean base + channels), PATH K (corrupt base + channels).
Profiling the deterministic phase on a representative Vietnamese chunk (~99
whitespace chunks) gives, **per single example**:

| Observation | Count |
|---|---|
| `decompose()` invocations | **≈ 301** |
| `canon()` invocations | **≈ 602** |
| `tokenizer.tokenize()` invocations | **199** |

**Why 301 decomposes for one example.** Eligibility classification recurses into a
full decomposition:

```
decompose(text, eligibility_classifier=classify)
  └─ _segment_syllables → flush → _build_span → classify(span)
       └─ classify_candidate(span, inventory)
            └─ membership_form(span)  =  strip_to_base(canon(span)).casefold()
                 └─ strip_to_base → decompose(span)   ← a FULL nested decompose
                      └─ canon(span)                  ← and another canon
```

So **every candidate syllable span triggers its own `canon` + full `decompose`**,
and that happens once per span per stream. In the profile, `canon` /
`apply_modern_placement` accounted for **≈44 %** of deterministic-phase cumulative
time and the `classify` chain for **≈32 %**. `canon` is additionally called
redundantly on already-canonical text: `prepare_with_condition` canonicalises,
`project_text` canonicalises again, and `decompose` canonicalises a third time.

**Second finding — half the tokenizer work is provably redundant.** PATH C and
PATH K tokenise **byte-identical inputs**: 99 calls each, and instrumentation
confirms the input lists are equal. The code *already asserts* this —
`if list(clean_content_ids) != list(corrupt_content_ids): raise
BaseInvarianceViolation`. Because base invariance holds, `base_text` is identical
between the clean and corrupted streams, hence identical whitespace chunks,
tokens, ids and alignment; **only the channel overlays differ**. Measured:
**99 / 199 = 49.7 %** of tokenizer calls per example are recomputation of a result
the code has already proven identical (≈30 % of tokenized characters).

### AF.7 Ranked SAFE optimisation candidates

**None implemented in this task.** Each is stated with the exact equivalence proof
it would need.

| # | Candidate | Evidence | Equivalence proof required |
|---|---|---|---|
| **1** | **Memoise the eligibility classifier** (`classify_candidate` / `membership_form`), keyed on the exact span string | **MEASURED: 2.66× on the deterministic phase**, with all prepared fields verified byte-identical; 5 513 hits / 31 misses | Cache key **is** the complete input; `membership_form` is pure given a fixed inventory. Must prove: keyed by exact string (no normalisation of the key); cache scoped to one verified inventory identity (§W) and invalidated if it changes; unbounded-growth bound argued from the ~2 489 stripped-form vocabulary |
| **2** | **Reuse tokenisation + alignment across PATH C and PATH K** | **MEASURED: 49.7 % of tokenize calls are on identical inputs** | `clean_base == corrupt_base` (already asserted) ⇒ identical chunks ⇒ identical tokens/ids/alignment, tokenisation being a pure function of the string. **Caveat: this converts a computed check into an inferred one** — the existing `clean_content_ids == corrupt_content_ids` assertion would no longer be independently computed. Keep the string equality check, and decide explicitly whether losing the id-level check is acceptable |
| **3** | **Remove redundant `canon()` on already-canonical text** | ≈602 `canon` calls/example; `canon` is idempotent | Prove `canon(canon(x)) == canon(x)` exhaustively over the corpus alphabet, then pass a canonical-form marker rather than re-canonicalising. Zero output change by idempotence |
| **4** | **Deterministic prefetch / producer-consumer overlap** | H2D already 0.001 s | Order-preserving only: batches must be reassembled in **exact sampler order** before collation; no change to `(chunk_id, visit)` pairing. **Ceiling ≈12.6 % — do this last, not first** |
| **5** | **Deterministic multi-process CPU preparation** | Stage-6 already proved this pattern | Main process owns order, membership and collation; workers compute pure functions only. Must reproduce byte-identical batches for 1/2/4/8 workers, exactly as Stage-6's `ordered_document_chunks` was proven |
| **6** | Collate-path allocation/copy reduction | collate = 8.35 % | Byte-identical padded tensors, identical dtypes, identical padding |

### AF.8 REJECTED — unsafe at any speed

| Rejected | Why |
|---|---|
| Batched/vectorised tokenizer calls | The pinned tokenizer is the **slow Python** PhoBERT tokenizer; batching changes the call pattern and cannot be assumed id-identical. Only admissible with an exhaustive corpus-wide byte-equality proof, which is more expensive than candidate 2 |
| Fast (Rust) tokenizer | Different implementation, unproven id-identity, and `use_fast=False` is part of the locked contract |
| Caching prepared examples across visits | Corruption is redrawn **per visit** — a cache keyed on `chunk_id` alone would freeze the draw and destroy the experiment |
| Any asynchronous preparation that can reorder | Would change which example pairs with which visit |
| Reducing dev set, eval cadence, batch size, or precision | Locked scientific constants — forbidden by construction |
| Skipping PATH K decomposition entirely | The **channel overlays genuinely differ** between clean and corrupt; only the *tokenisation* is redundant, not the decomposition |
| Moving Adam's scalar `step` to CUDA | AF.3 — the current behaviour is correct |

### AF.9 Is another micro-profile needed before implementing? — YES

**The one number that decides the plan has not been measured**: how the real
**4.7235 s** splits between the **real PhoBERT slow tokenizer** and the
**deterministic phase**. The local profile above used a stub tokenizer, so it
measured the deterministic phase only; the 2.66× is a measured speedup **of that
phase**, not of the whole preparation. Candidate 1 attacks the deterministic
phase and candidate 2 attacks the tokenizer, and their combined effect cannot be
projected without the split.

**Required Colab micro-profile — read-only, zero update, no `optimizer.step`:**

1. `cProfile` over `prepare_example` for one real batch of 128 real prepared
   chunks, at the locked tokenizer and inventory;
2. report cumulative seconds for `tokenizer.tokenize`, `convert_tokens_to_ids`,
   `canon`, `decompose`, `classify_candidate`, `align_chunk`, `project_piece`;
3. report per-example counts of `decompose`, `canon` and `tokenize` on **real**
   corpus text (the local counts used synthetic prose);
4. re-run the same batch with an `lru_cache` wrapped **around the classifier
   argument only** — no source change — and report both the wall-clock delta and
   a byte-equality check of every prepared field;
5. report distinct-span cache cardinality over ≥10 000 real chunks, to bound
   memory.

Nothing in that list writes an artifact, constructs an optimizer, or steps one.

### AF.10 Files a future implementation would touch, and the decision question

**Files:** `unmark/linguistics/classify.py` and/or `unmark/linguistics/inventory.py`
(candidate 1); `unmark/stage1/data.py` — `prepare_with_condition`, `project_text`
(candidates 2, 3); `unmark/orthography/decompose.py` (candidate 3);
`unmark/stage1/trainer.py` (candidate 4/5 only); plus new tests asserting
byte-identical prepared output, and a persistent CUDA resume-equivalence test to
close AF.4's caveat.

**Decision-log consequence.** Measuring a bottleneck needs **no** decision entry,
and none is added here. But **candidate 2 would require one**: it replaces an
independently *computed* base-invariance check with one *inferred* from string
equality, which narrows a fail-closed guarantee. That is a specification-level
change and must be recorded in `docs/spec/decisions.md` **before** implementation,
not alongside it. Candidates 1, 3, 4, 5 and 6 are pure implementation changes with
byte-identical outputs and need no decision entry — provided their equivalence
proofs in AF.7 are actually produced.

### AF.11 What this section does and does not establish

**Established:** every acceptance gate in AF.2; the two reserved claims of
§AE.11.8, subject to AF.4's caveat; the AF.3 correction; and the quantified
performance blocker with its root cause.

**Not established:** any speedup of the *real* preparation path — the 2.66× is a
measured result for the deterministic phase under a stub tokenizer, and the 49.7 %
is a call count, not a time. No optimisation has been implemented. The
**FINAL STAGE-1 CONFIGURATION FREEZE**, the final proposal-aware review and
**explicit human approval** all remain outstanding.

**STATUS: PRE-TRAIN RUNTIME ACCEPTANCE PASS — PERFORMANCE BLOCKER UNDER REVIEW**
**ALL REAL ZERO-UPDATE ACCEPTANCE GATES PASSED AT `ac20cfb7` ON THE REAL CORPUS AND REAL PhoBERT**
**SCIENTIFIC `optimizer.step` COUNT IS STILL ZERO; OFFICIAL UIT-VSFC TEST REMAINS SEALED**
**§AE.11.8's TWO RESERVED CLAIMS ARE CLOSED — BUT AS A ONE-OFF FIXTURE, NOT REGRESSION COVERAGE**
**CORRECTION: ADAM'S SCALAR `step` IS LEGITIMATELY ON CPU; §AC.15 ITEM 3 IS SUPERSEDED**
**NO PRODUCTION CHANGE MAY BE MADE TO FORCE SCALAR `step` ONTO CUDA**
**THE TRAINING PATH IS PREPARATION-BOUND: 79.05 % PREPARE, 87.40 % CPU-SIDE, 12.60 % GPU**
**LOWER BOUND ≈36.67 h PER 20k RUN AND ≈403 h / 16.8 DAYS FOR THE 11 NOMINAL RUNS**
**ROOT CAUSE: ELIGIBILITY CLASSIFICATION RECURSES INTO A FULL `decompose` PER SYLLABLE SPAN**
**≈301 `decompose` AND ≈602 `canon` CALLS PER EXAMPLE; 49.7 % OF TOKENIZE CALLS ARE REDUNDANT**
**MEASURED: CLASSIFIER MEMOISATION GIVES 2.66x ON THE DETERMINISTIC PHASE, OUTPUT BYTE-IDENTICAL**
**PREFETCH ALONE CANNOT FIX THIS — ITS CEILING IS 12.6 % WHILE CPU EXCEEDS GPU SEVENFOLD**
**A COLAB MICRO-PROFILE OF THE TOKENIZER/DETERMINISTIC SPLIT IS REQUIRED BEFORE IMPLEMENTING**
**CANDIDATE 2 WOULD NARROW A FAIL-CLOSED CHECK AND WOULD REQUIRE A DECISION ENTRY FIRST**
**NO OPTIMISATION IMPLEMENTED; NO SCIENTIFIC CODE, TEST, CONFIG OR CONSTANT CHANGED**
**TRAINING IS NOT AUTHORISED**

---

## AG. DETERMINISTIC PARALLEL PREPARATION — FIRST PERFORMANCE REPAIR

**Revision 18.** Implementation of the smallest safe repair §AF identified, from
base HEAD `ac20cfb786ca770a7296339d48263ff8e09acf66`.

**Scientific `optimizer.step` count remains ZERO. Official UIT-VSFC TEST remains
sealed and unused. Training is not authorised.**

### AG.1 New real measurements that decided the plan

Real prepared corpus, real pinned PhoBERT **slow** tokenizer, batch 128, on a
24-physical / 48-logical-core host:

| Configuration | Seconds / batch | Speedup | Prepared output |
|---|---|---|---|
| serial | **4.605099308** | 1.000× | — |
| 2 workers | 2.328961237 | **1.977319×** | **exactly equal** |
| 4 workers | 1.210541180 | **3.804166×** | **exactly equal** |
| **8 workers** | **0.666244782** | **6.912023×** | **exactly equal** |
| 8 workers + per-worker classifier cache | 0.602913878 | 7.638071× | **exactly equal** |

And the measurements that **narrowed** the plan:

| Finding | Value | Consequence |
|---|---|---|
| Real tokenizer share of preparation | **~5.01 %** | **Tokenizer reuse is not worth it.** §AF candidate 2 would have traded an independently *computed* base-invariance check for an *inferred* one to chase ~5 %. **Rejected** — and with it the decision-log entry §AF.10 said it would need |
| Classifier memoisation, real batch | only **1.059×** | Far below the 2.66× measured on the stub-tokenizer deterministic phase. **Deferred**, not implemented |
| Exact-span cache cardinality, 10 000 real chunks | **50 092** | Larger than the ~2 489 stripped-form vocabulary suggested; a real memory question |
| Exact-span cache hit fraction | **97.29 %** | High, but 1.059× makes the complexity unjustified for a first repair |

> **§AF.6's wording is corrected accordingly** (see the note now in §AF.6): the
> tokenizer is **~5.01 %** of preparation, not negligible-by-assertion, and the
> dominant cost remains the deterministic orthography/alignment path.

**Decision for this repair: deterministic 8-worker parallel preparation ONLY.**
No classifier cache, no tokenizer reuse, no `canon` de-duplication, no prefetch.

### AG.2 ★ Production uses `spawn`, and the benchmark's `fork` must not be copied

The benchmark above used **`fork`**, legitimately: it was a CPU-only process in
which CUDA had never been initialised. **Production is the opposite.** By the time
`train_run` reaches its first batch the parent holds a CUDA context, the placed
model and the optimizer. Forking a CUDA-initialised parent copies state that is
invalid in the child; it is documented as unsupported and tends to deadlock rather
than fail cleanly.

`MULTIPROCESSING_START_METHOD = "spawn"`, and a source-level test asserts
`get_context(...)` is called with that constant and that no `"fork"` literal can
reappear. That also explains a design consequence: a spawned worker inherits
nothing, so it must **rebuild** its tokenizer and inventory from identity.

**Therefore the benchmark establishes parallelisability, order and exact output
equality — NOT production spawn throughput.** No speedup is claimed for the
implementation; see AG.7.

### AG.3 Architecture

New `unmark/stage1/preparation.py`:

| Component | Contract |
|---|---|
| `PREPARATION_WORKERS = 8` | **Operational, not scientific** — see AG.6 |
| `PreparationPool` | Persistent for the whole stage command, not per batch: under `spawn` each worker reloads the pinned tokenizer and re-verifies the inventory. Context-managed, so it shuts down on normal completion, on exception and on fail-closed abort alike |
| `_initialise_worker` | Once per process: pinned tokenizer via `pinned_tokenizer` (refuses any revision but the locked one, **before** importing transformers), `load_inventory()` — the strict loader, so a missing or altered cache raises rather than degrading eligibility — and the classifier built from it |
| `_prepare_one` | Calls the **authoritative** `prepare_example`. Scientific preparation logic is not reimplemented in the worker |
| `worker_config` | Deliberately tiny and picklable. **The 2.6M-entry corpus dictionary never appears in it**; only the text of already-selected examples travels, one batch at a time |
| `prepare_serially` | The serial reference — **tests and diagnostics only**, never reachable as a fallback |

`train_run` changed in exactly one place. The main process still does:

```
pairs = sampler.next_batch(BATCH_SIZE)          ← main process, once per step
tasks = [(chunk_id, visit, train_chunks[chunk_id]) for chunk_id, visit in pairs]
prepared = preparation_pool.prepare(tasks)      ← the ONLY thing that moved
batch = batch_to_device(collate_stage1_batch(prepared, pad_token_id), ...)
objective(batch) → zero_grad → backward → gradient_report → step → global_update += 1
```

> **Loop order re-verified from source at pre-commit, not carried over from this
> prose** (§AH). The AST-extracted order inside the real `while` loop is
> `next_batch` (636) → `prepare` (644) → `batch_to_device`/`collate` (662–663) →
> `objective` (665) → `zero_grad` (667) → `backward` (668) → `gradient_report`
> (669) → `step` (677) → `global_update += 1` (678). **The forward runs before
> `zero_grad`** — that is the pre-existing production order, it is unchanged by
> this repair, and it is equivalent because nothing accumulates into `.grad`
> between the forward and `zero_grad`. `gradient_report` is added to the summary
> above for precision; it reads `.grad` between `backward` and `step`, which is
> where it must be.

**Workers never** own or advance a sampler, choose a sample or a visit, reorder
anything, construct an optimizer, touch CUDA, write a checkpoint, increment
`global_update`, run validation, or load model weights. They receive
`(sample_id, visit, text)` triples the main process already chose. Asserted on the
call graph, and `preparation.py` imports no torch at all.

### AG.4 Order, and the absence of prefetch

Results are reconstructed with `Executor.map`, which yields in **input** order
regardless of completion order — no unordered API is used. A test submits a
deliberately slow task **first** so it finishes last, and requires the returned
sequence to match sampler order exactly.

**No prefetch, deliberately.** Exactly one `next_batch` per training step,
asserted by AST on the real loop. Checkpointing commits sampler state together
with the completed update, so consuming future batches would create a
resume-state problem this change is not scoped to solve. `PREFETCH_ENABLED` is
`False` and the module uses no `submit`, `as_completed`, `Queue` or `Thread`.

### AG.5 Failure is loud; there is no silent serial fallback

`prepare` re-raises any worker exception as `PreparationContractViolation`, so a
partial batch can never reach training. Degrading silently to serial would hide a
broken pool behind a merely slow run. Tested both ways: a worker whose tokenizer
identity cannot be established, and a task that raises inside preparation.

### AG.6 Worker count is operational provenance, not scientific identity

Recorded in the stage artifact:

```
preparation_backend  multiprocessing_spawn
preparation_workers  8
order_preserving     true
prefetch             false
```

**It is deliberately NOT in `RunProvenance` and NOT resume-blocking.** The
reasoning is the equivalence proof: prepared output is byte-identical across
worker counts, so worker count changes wall-clock and nothing else. A run
interrupted with 8 workers may legitimately resume with 4 and remain the same
experiment; making it resume-blocking would reject a scientifically identical
continuation — a cost with no corresponding guarantee.

**Exactly how far that proof currently reaches, stated precisely:**

| Evidence | Status |
|---|---|
| Real-tokenizer CPU benchmark, serial vs 2/4/8, real inputs | **established** — but it used **`fork`** and was **not the production spawn implementation** |
| Persistent local tests: serial vs **1 / 2 / 4** spawn workers, exact equality | **established**, executed locally — but with a **tiny injected tokenizer** |
| **Production `spawn` + the REAL pinned PhoBERT tokenizer** | **POST-IMPLEMENTATION AUTHORITATIVE VERIFICATION PENDING** |

So the non-resume-blocking classification is **by design and consistent with all
current evidence**, but the final production-real equivalence has **not** yet been
executed. It must be established on Colab before the configuration freeze. Worker
count is **not** being made resume-blocking to paper over that gap.

### AG.7 Tests

New `tests/test_stage1_parallel_preparation.py` — **17 tests, all executed
locally**, starting real spawn worker processes with a tiny injected tokenizer
(no PhoBERT downloaded):

| Requirement | Evidence |
|---|---|
| **A** serial vs parallel exact equality | every field of `PreparedStage1Example`, no tolerance, over fixtures spanning multiple sample ids, **three visits**, marked/unmarked syllables, punctuation, mixed Vietnamese/non-Vietnamese, and lengths from one character to near-`MAX_LENGTH` |
| **B** order preservation | slow task submitted first; returned order matches sampler order and every example matches serial |
| **C** worker-count invariance | 1 / 2 / 4 workers all byte-identical to serial; the production constant 8 asserted structurally |
| **D** failure propagation | worker-init failure and in-task failure both abort; no partial batch; no fallback |
| **E** sampler ownership | `preparation.py` calls no `next_batch`, `DeterministicSampler`, optimizer, checkpoint, `evaluate`, `backward`, `step`, `AutoModel`, collate or device helper; imports no torch; no CUDA call |
| **F** no look-ahead | exactly one `next_batch` per step; `global_update` incremented exactly once; no async primitives |
| **G** CUDA safety | `spawn` asserted; `get_context` must use the locked constant; no `"fork"` literal can reappear; `execute_stage` must take the pinned factory and never inject one |
| **H** scientific output | the full suite is unchanged — see AG.9 |

New `tests/test_stage1_cuda_resume_equivalence.py` — **CUDA-gated, closes §AF.4's
caveat.** It uses the production `checkpoint_payload` / `save_training_checkpoint`
/ `load_training_checkpoint` / `verify_checkpoint` /
`require_optimizer_parameter_identity` / `require_optimizer_state_device` helpers
on a tiny synthetic model, and proves uninterrupted 16-update execution equals
interrupt-at-8 → checkpoint → **fresh reconstruction** → resume → finish, with
exact equality of adapter tensors, populated optimizer state, sampler state,
update count and validation-point history. Respecting real AdamW semantics
(§AF.3): `exp_avg` / `exp_avg_sq` asserted on **concrete CUDA**, the
zero-dimensional scalar `step` left on CPU and **not** forced across.

**The stale docstring §AF.4 reported is fixed** —
`tests/test_stage1_run_independence_runtime.py:7` no longer claims a CUDA
resume-equivalence test that did not exist, and now points at the file above.

### AG.8 Two issues found while testing — one production, one fixture

Both were found by tests that failed for the right reason, and both were repaired.
**They are not the same kind of thing**, and an earlier draft of this heading
called both "production defects", which was wrong about the second:

1. **`pinned_tokenizer` imported transformers *before* checking the revision** —
   an **implementation defect, found and repaired**. A worker handed a foreign
   revision would have failed on the way to fetching one rather than refusing it.
   The revision check now precedes the import.
2. **The order-preservation test fixture exceeded `MAX_LENGTH`** — a
   **test-fixture defect. Production behaviour was CORRECT**: the fail-closed
   overflow guard fired exactly as specified (`reference sequence length 802
   exceeds max_length 256`). Nothing in production was changed; the fixture was
   resized to stay under the ceiling while still finishing last.

### AG.9 Test results

All figures below are the **actual** output of the commands named, re-run at
pre-commit (§AH). Nothing here is recorded as "PASS" that was not observed.

| Command | Result |
|---|---|
| `pytest -q tests/test_stage1_parallel_preparation.py` | **19 passed, 0 skipped, 0 failed** — 15.85 s |
| `pytest -q` *(full lightweight suite)* | **3 594 passed, 102 skipped, 0 failed, 0 errors** — 133.03 s |
| `pytest -q` over preparation + Stage-1 data/corruption + checkpoint/resume + run-independence + provenance | **738 passed, 17 skipped, 0 failed** — 22.40 s |
| `pytest -q tests/test_stage1_cuda_resume_equivalence.py` | **0 passed, 1 skipped** — `needs torch` |

**Skips are not passes.** The two Stage-1 modules that skipped in the targeted
run did so for stated reasons: `test_stage1_cuda_resume_equivalence.py:24`
(*needs torch*) and `test_stage1_run_independence_runtime.py:22` (*the runtime
half needs torch*).

**Could not execute locally, and therefore NOT claimed:** everything requiring
torch, CUDA or transformers — the new CUDA resume-equivalence file, and the
production pool driving the **real** PhoBERT tokenizer. The local pool tests use
a tiny injected tokenizer, which is precisely what makes them runnable here and
also what keeps them short of authoritative.

### AG.10 Decision log and proposal — unchanged

**No decision-log entry was added, and none is required.** §AF classified
deterministic parallel execution as a pure engineering change *if* output
equivalence is proven; it is proven in AG.7. Nothing scientific or fail-closed was
weakened — in particular the base-invariance check remains **independently
computed**, because tokenizer reuse was rejected in AG.1. `docs/spec/decisions.md`
and the proposal source are **untouched**.

### AG.11 What remains

| Requirement | Status |
|---|---|
| **Post-implementation CUDA/spawn benchmark on the authoritative GPU** | **MANDATORY** — no production speedup is claimed |
| Execute `test_stage1_cuda_resume_equivalence.py` on the GPU | **REQUIRED** |
| Execute the pool tests against the real pinned tokenizer | **REQUIRED** |
| Classifier memoisation (1.059× real, 50 092 distinct spans) | **DEFERRED** |
| Tokenizer reuse | **REJECTED** — ~5.01 % gain, would weaken a computed check |
| Redundant `canon` removal, prefetch | **DEFERRED** |
| **FINAL STAGE-1 CONFIGURATION FREEZE**, final review, human approval | **REQUIRED** |

**STATUS: IMPLEMENTED — POST-IMPLEMENTATION CUDA/SPAWN PERFORMANCE VERIFICATION PENDING**
**DETERMINISTIC 8-WORKER PARALLEL PREPARATION ONLY; NO CACHE, NO TOKENIZER REUSE, NO PREFETCH**
**PRODUCTION USES `spawn`, NEVER `fork` — THE BENCHMARK'S FORK MUST NOT REACH A CUDA PARENT**
**BENCHMARK ESTABLISHES PARALLELISABILITY AND EXACT EQUALITY, NOT PRODUCTION SPAWN THROUGHPUT**
**SERIAL 4.605099308 s → 8-WORKER 0.666244782 s (6.912023x), ALL OUTPUTS EXACTLY EQUAL**
**TOKENIZER IS ONLY ~5.01 % OF PREPARATION — TOKENIZER REUSE REJECTED, NOT MERELY DEFERRED**
**CLASSIFIER CACHE GAVE ONLY 1.059x ON A REAL BATCH; 50 092 DISTINCT SPANS; DEFERRED**
**MAIN PROCESS STILL OWNS SAMPLER, VISITS, ORDER, UPDATE, CHECKPOINT, OPTIMIZER AND CUDA**
**EXACTLY ONE SAMPLER BATCH PER STEP — NO PREFETCH, NO LOOK-AHEAD, AST-ASSERTED**
**ORDER RECONSTRUCTED BY `Executor.map`; COMPLETION ORDER CANNOT REACH SCIENTIFIC ORDER**
**FAILURE ABORTS LOUDLY; THERE IS NO SILENT SERIAL FALLBACK ON THE SCIENTIFIC PATH**
**WORKER COUNT IS OPERATIONAL PROVENANCE, NOT RUN IDENTITY AND NOT RESUME-BLOCKING**
**§AF.4's CAVEAT CLOSED: A PERSISTENT CUDA RESUME-EQUIVALENCE TEST NOW EXISTS**
**NO DECISION ENTRY REQUIRED; decisions.md AND THE PROPOSAL ARE UNTOUCHED**
**SCIENTIFIC `optimizer.step` COUNT IS ZERO; OFFICIAL UIT-VSFC TEST IS SEALED AND UNUSED**
**TRAINING IS NOT AUTHORISED**

---

## AH. PRE-COMMIT VERIFICATION OF THE PARALLEL-PREPARATION REPAIR

**Revision 19.** Verification only — re-derived **from source**, not from §AG's
prose. Base HEAD `ac20cfb786ca770a7296339d48263ff8e09acf66`, working tree
carrying the §AF/§AG changes.

### AH.1 Architecture, re-derived from the AST

| Question | Verified answer |
|---|---|
| Where is `PreparationPool` constructed? | `execute.py:274`, **once** |
| Persistent, or per batch? | **Persistent** — one construction, and the schedule loop (line 275) is nested inside its `with` |
| When is it shut down? | `__exit__` → `shutdown(wait=True)`; a context manager, so on normal completion, on exception and on fail-closed abort alike |
| Start method | `multiprocessing.get_context(MULTIPROCESSING_START_METHOD)` = **`"spawn"`**; **no `"fork"` literal exists in the module** |
| Worker initializer | `_initialise_worker` — pinned tokenizer via `pinned_tokenizer`, strict `load_inventory()`, classifier |
| Worker task payload | `[(chunk_id, visit, train_chunks[chunk_id]) for chunk_id, visit in pairs]` |
| Can the 2.6M-entry dict reach a worker? | **No.** `worker_config` holds 8 small scalars; `train_chunks` appears nowhere in it. Only the selected texts travel, one batch at a time |
| Workers touching sampler / CUDA / model / optimizer / checkpoints? | **None.** No such call appears in `preparation.py`, and **it imports no torch** |
| Order preserved? | `Executor.map` only; **no** `submit`, `as_completed` or unordered API |
| Serial fallback? | See AH.4 |
| Prefetch / look-ahead? | **None.** Exactly one `next_batch` per step |

### AH.2 ★ The training-loop order — AG.3 was CORRECT, and a proposed "fix" was not

The pre-commit brief suggested AG.3 might need correcting to
`zero_grad → objective → backward → step`. **It does not: that order is not what
the code does**, and applying it would have made the audit *less* accurate. The
AST-extracted order inside the real `while` loop is:

```
636  next_batch          644  prepare             662  batch_to_device
663  collate             665  objective           667  zero_grad
668  backward            669  gradient_report     677  step
678  global_update += 1  681  evaluate_fn         696  save_training_checkpoint
```

**The forward runs before `zero_grad`.** That is pre-existing production
behaviour, untouched by this repair, and equivalent — nothing accumulates into
`.grad` between the forward and `zero_grad`. AG.3 has been left as it was and
extended only to name `gradient_report`, which reads `.grad` between `backward`
and `step`. **No training semantics were changed by this verification.**

### AH.3 Spawn-specific hazards — reviewed, all clear

| Hazard | Finding |
|---|---|
| Picklability of initializer / worker callable / factory | `_initialise_worker`, `_prepare_one`, `pinned_tokenizer` are module-level; round-trip verified `is`-identical |
| Importability under spawn | all live in `unmark.stage1.preparation`, an ordinary package module |
| `__main__` recursion | `scripts/stage1_runner.py` ends in `if __name__ == "__main__":`, so a spawned child re-importing `__main__` does **not** re-enter `main()` |
| CUDA state in workers | none — no CUDA call, no torch import |
| Model-weight loading in workers | none — `AutoTokenizer` only, never `AutoModel` |
| HF revision advancement | `pinned_tokenizer` refuses any revision but the locked one, **before** importing transformers |
| Inventory fallback | strict `load_inventory()`, never `try_load_inventory` |
| Pool startup failure / worker crash / partial results | re-raised as `PreparationContractViolation`; tested both for initializer failure and in-task failure |
| Shutdown on exception | context manager |
| Ordering under unequal completion | tested with a deliberately slow task submitted **first** |
| Repeated process construction per batch | none — one pool per stage command |

**One concrete confirmation of the `spawn` decision.** CUDA is resolved at
`execute.py:211` and the backbone placed at `:230` — both **before** the pool is
created at `:274`. So at pool-creation time the parent *does* hold a CUDA context.
Under `fork` that is the documented-unsupported case; under `spawn` it is
irrelevant. The design choice is load-bearing, not precautionary.

### AH.4 The serial branch — examined, and classified

`train_run` retains `if preparation_pool is not None: ... else: prepare_serially(...)`.
Assessed rather than waved through:

* It is **not a failure fallback.** A pool failure raises out of `prepare`; there
  is no `try/except` that degrades. A test now asserts no exception handler
  contains both `preparation_pool` and `prepare_serially`.
* `execute_stage` passes the pool at **both** `train_run` call sites, so the
  scientific path never reaches the serial branch.
* Residual risk is **silent 7× degradation**, not incorrectness — serial output is
  byte-identical. But it is the same drift class this audit has been bitten by
  twice (§AA parser/handler, §AD construction-outside-the-loop).

**Closed structurally rather than by convention:** a new test asserts every
`train_run` call inside `execute_stage` passes `preparation_pool=preparation_pool`.
**No production behaviour changed.**

### AH.5 Wording corrections made to §AG

| Location | Correction |
|---|---|
| **AG.3** | Extended to name `gradient_report`; loop order re-verified from source and recorded. The suggested reordering was **rejected as inaccurate** (AH.2) |
| **AG.6** | Now states exactly how far the equivalence proof reaches: the real-tokenizer benchmark used **fork** and was not the production implementation; the local spawn tests used a **tiny injected tokenizer**; **production spawn + real pinned PhoBERT is POST-IMPLEMENTATION AUTHORITATIVE VERIFICATION PENDING**. Worker count remains non-resume-blocking **by design**, not to avoid that wording |
| **AG.8** | Retitled *"Two issues found while testing — one production, one fixture"*. Item 1 is an **implementation defect, repaired**; item 2 is a **test-fixture defect** where **production fail-closed behaviour was correct** and nothing in production changed. The earlier heading called both production defects, which was wrong |
| **AG.9** | Dangling *"see the summary below AG.11"* removed; replaced with the **actual** commands and counts, and an explicit statement that skips are not passes |

### AH.6 Result

**No implementation defect was discovered by this verification.** The one
production defect in this repair (`pinned_tokenizer` import ordering) was already
found and fixed during implementation and is recorded in AG.8. The only change
made here is **two additional guard tests**; production source is otherwise
untouched by this task.

`docs/spec/decisions.md` and the proposal remain **unchanged**.

**STATUS: PARALLEL PREPARATION IMPLEMENTED — AUTHORITATIVE CUDA/SPAWN VERIFICATION PENDING**
**ARCHITECTURE RE-DERIVED FROM SOURCE: PERSISTENT POOL, SPAWN, ORDERED map, NO PREFETCH**
**AG.3's LOOP ORDER WAS ALREADY CORRECT — THE PROPOSED REORDERING WOULD HAVE BEEN WRONG**
**FORWARD RUNS BEFORE `zero_grad`; THAT IS PRE-EXISTING AND UNCHANGED BY THIS REPAIR**
**CUDA IS LIVE IN THE PARENT BEFORE THE POOL IS BUILT — `spawn` IS LOAD-BEARING, NOT PRECAUTIONARY**
**THE 2.6M-ENTRY CORPUS DICT CANNOT REACH A WORKER; ONLY SELECTED TEXTS TRAVEL PER BATCH**
**SERIAL BRANCH IS NOT A FAILURE FALLBACK; A NEW TEST PINS THE POOL TO EVERY `train_run` CALL**
**AG.8 RECLASSIFIED: ONE IMPLEMENTATION DEFECT, ONE TEST-FIXTURE DEFECT WITH CORRECT PRODUCTION BEHAVIOUR**
**AG.9's DANGLING REFERENCE REPLACED WITH MEASURED COUNTS: FULL SUITE 3 594 PASSED / 102 SKIPPED / 0 FAILED**
**REAL-TOKENIZER SPAWN EQUIVALENCE IS EXPLICITLY MARKED PENDING, NOT ASSUMED**
**NO NEW OPTIMISATION; NO PRODUCTION SEMANTIC CHANGE; decisions.md AND PROPOSAL UNTOUCHED**
**TRAINING IS NOT AUTHORISED**
