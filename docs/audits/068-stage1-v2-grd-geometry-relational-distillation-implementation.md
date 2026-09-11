# Audit 068 - Stage-1 V2-GRD (Geometry / Relational Distillation) Implementation and Audit

**Scope:** implement UNMARK-v2 Candidate **C2 / V2-GRD** side by side with the
already-implemented C1 / V2-SCF and C3 / V2-GC, then audit C2 and regression-check
C1 and C3.
**Date:** 2026-09-11
**Type:** implementation plus scientific audit. No commit was created, no model
was trained, no UIT-VSFC data, official validation or official TEST was accessed,
and no ML package was installed in the ML-free local environment.

Audits 065, 066 and 067 are byte-unchanged (`61672fb15d1a22342848cf4e5d18b23d`,
`35076e34f9ffd8360da4fb53c15cbc00`, `e8038e6b12d442226f3904bdab3bf13c`).

---

## 0. Executive verdict

**PASS_WITH_COLAB_GATE.**

C2 is implemented as an isolated, training-objective-only candidate: an
off-diagonal MSE between the pairwise cosine Gram matrices of the adapted
branches and the CLEAN NATIVE teacher, in FIRST_TOKEN space. It adds **zero
trainable parameters**, keeps the **historical fusion**, runs the **same three
encoder forwards**, and is hard-capped at **20 000 updates** through the generic
budget abstraction. C1 and C3 are untouched.

All three candidates now coexist in one HEAD with distinct identities, distinct
budgets, distinct W&B projects and a fully fail-closed cross-candidate
compatibility matrix.

The remaining gate is environmental: 68 torch-gated tests (17 C1 + 22 C2 + 29 C3)
cannot execute locally and must run on Colab/GPU before the screening runs.

---

## 1. Scientific hypothesis (restated, locked)

Post-hoc diagnostic D4 found that historical UNMARK-A does not preserve the
native PhoBERT decision geometry that clean Vanilla heads use. On protocol-dev,
even FULL UNMARK-A had approximately:

```
native-vs-UNMARK FIRST_TOKEN cosine distance ~= 0.303
same-Vanilla-head prediction agreement       ~= 0.485
centered-logit cosine                        ~= 0.23
```

The historical objective constrains per-example MASKED-MEAN pooled
representations and says nothing about how examples sit *relative to one another*
in the FIRST_TOKEN space Stage-2 reads.

C2 tests exactly one hypothesis: **preserving CLEAN NATIVE relational geometry in
FIRST_TOKEN space improves downstream geometry while retaining robustness.**

**The teacher is ALWAYS native PhoBERT on the CLEAN ORIGINAL input.** Never
PhoBERT on P25/P50/P75/P100/STRIP. There is no batch field on this path that
could supply a corrupted teacher, and distilling from one would pull the severe
conditions toward Vanilla's degraded behaviour -- the opposite of UNMARK's
purpose.

---

## 2. Exact implementation

`unmark/stage1/objective_relational.py`, verbatim from the current tree:

```python
first_token(hidden)      ->  hidden[:, 0, :]                                   # line 180

cosine_gram(x, eps):
    normalised = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)               # line 202
    return normalised @ normalised.transpose(0, 1)                             # line 203

offdiagonal_mask(B)      ->  ~torch.eye(B, dtype=torch.bool, device=device)    # line 215

relational_distance(student, teacher_target, eps):
    target = teacher_target.detach()                                           # line 262
    mask = offdiagonal_mask(student.shape[0], device=student.device)
    difference = cosine_gram(student, eps) - cosine_gram(target, eps)
    return (difference[mask] ** 2).mean()                                      # line 266
```

and the composition:

```python
loss_grd = (self.spec.relational_clean_weight   * loss_rel_clean               # 0.5
          + self.spec.relational_corrupt_weight * loss_rel_corrupt)            # 0.5
loss     = (self.weights.lambda_align * loss_align                             # 1.0
          + self.weights.lambda_clean * loss_clean                             # 1.0
          + self.spec.lambda_grd      * loss_grd)                              # 1.0
```

| Locked requirement | Status |
| --- | --- |
| `X_hat_i = X_i / clamp(||X_i||, 1e-8)`, `G = X_hat @ X_hat.T`, `[B, B]` | **PASS** |
| epsilon exactly `1e-8` (`protocol.RELATION_EPSILON`) | **PASS** |
| `R(S,T)` = mean over `i != j` of `(G(S)-G(T))^2` | **PASS** |
| diagonal EXCLUDED by construction (`~torch.eye` mask) | **PASS** |
| no L1, no KL, no covariance, no centering, no temperature, no top-k, no mining, no subsampling | **PASS** (asserted by a banned-token scan of code-only source) |
| relations BETWEEN EXAMPLES, never tokens (`cosine_gram` refuses a 3-D tensor) | **PASS** |
| `B >= 2` fails closed, never a silent `0.0` | **PASS** (`MINIMUM_BATCH = 2`) |
| all representations finite; nothing sanitised | **PASS** (two `_require_finite` calls; no `nan_to_num`) |
| teacher detached unconditionally | **PASS** |
| students never detached | **PASS** |
| `L_grd = 0.5*L_rel_clean + 0.5*L_rel_corrupt` | **PASS** |
| `lambda_grd = 1.0`, not a CLI flag | **PASS** |

**Verified against a spec-derived reference implementation** written from the
prompt, not from the code. For `T = [[1,0],[0,1],[1,1]]`, `S = [[1,0],[0,1],[1,0]]`:
`G(T)` off-diagonals are `0, 0.70710678, 0.70710678`; `R(S,T) = 0.19526214587563495`;
`R(T,T) = 0` exactly; a zero row stays finite; and **including the diagonal would
scale the loss by exactly `(B-1)/B = 0.6667`**, confirming the exclusion is
load-bearing rather than cosmetic.

### Why this is not DP, not GC, not SCF

* **Not DP.** `representation_distance` -- the pointwise cosine -- is called
  exactly twice in C2's forward, and only on the two POOLED historical terms. An
  AST test asserts no `representation_distance(student_*)` or
  `representation_distance(teacher)` call exists. FIRST_TOKEN is used *only* to
  build inter-example geometry.
* **Not GC.** The module contains no `token_grid_distance`, no `loss_grid`, no
  token-level relation over `[B, L, D]`; `C2.objective.lambda_grid is None`.
* **Not SCF.** `C2.fusion.fusion_id == "historical-fusion-v1"`; the module never
  mentions `fusion_id` or `scale_calibrated`.

---

## 3. Identity, registry and provenance

C2 joins through the generic candidate register introduced for C3 and extended
for C1. The register now holds six stages and three post-hoc candidates:

```
lr_pilot / r_phase1 / final_main
    (align-clean-pooled-v1,                  historical-fusion-v1)       precommitted | unmark-stage1
v2_scf (C1)
    (align-clean-pooled-v1,                  scale-calibrated-fusion-v1) hard cap 20k | UNMARK-v2-C1-SCF-Stage1
v2_grd (C2)
    (geometry-relational-distillation-v1,    historical-fusion-v1)       hard cap 20k | UNMARK-v2-C2-GRD-Stage1
v2_gc  (C3)
    (grid-consistency-v1,                    historical-fusion-v1)       hard cap 20k | UNMARK-v2-C3-GC-Stage1
```

**The objective identity carries the full relational specification.** A new typed
`RelationalSpec` -- not a loose dict -- is derived from the objective id and
serialised into every artifact, so a reader recovers HOW the geometry was
measured without opening any code:

```json
"objective": {
  "objective_id": "geometry-relational-distillation-v1",
  "lambda_grid": null,
  "lambda_grd": 1.0,
  "relational": {
    "lambda_grd": 1.0,
    "relation_space": "FIRST_TOKEN",
    "relation_metric": "pairwise-cosine-gram",
    "relation_loss": "off-diagonal-mse",
    "relational_clean_weight": 0.5,
    "relational_corrupt_weight": 0.5,
    "epsilon": 1e-08
  }
}
```

`RelationalSpec.__post_init__` refuses weights that do not sum to exactly 1.0.
Every field is derived, never supplied, so an artifact cannot record an identity
and a specification that disagree. The historical, C1 and C3 objectives report
`lambda_grd: null` and `relational: null` -- honest absence, never zero.

**Executed 5x4 cross-candidate compatibility matrix (20/20 correct):**

| Payload \ Environment | historical | C1 | C2 | C3 |
| --- | --- | --- | --- | --- |
| legacy (no objective/fusion) | **PASS** | REFUSE | REFUSE | REFUSE |
| historical | **PASS** | REFUSE | REFUSE | REFUSE |
| C1 | REFUSE | **PASS** | REFUSE | REFUSE |
| C2 | REFUSE | REFUSE | **PASS** | REFUSE |
| C3 | REFUSE | REFUSE | REFUSE | **PASS** |

A missing objective block is read as, and only as, the historical objective --
so it can never mean C2. No historical checkpoint was rewritten.

`docs/spec/stage1-adapter-finalists.json` is **unchanged by this task** -- still
exactly the two leaves under `/evidence/provenance_fields_verified/` that C1
required. `adjudication.finalists` is byte-identical to the baseline: no hash,
seed, selected update, validation score, robust score or scientific result was
touched.

---

## 4. Forward-pass accounting

```
historical  3 encoder forwards
C1          3 encoder forwards
C2          3 encoder forwards
C3          3 encoder forwards
```

C2's forward calls `reference_branch` once and `adapted_branch` twice, and never
calls `reference_representation` or `adapted_representation` (which would be a
second forward of the same branch). `first_token`, `cosine_gram`,
`offdiagonal_mask`, `offdiagonal_mean` and `relational_distance` call no encoder
at all -- asserted for each by AST.

**One refactor was required.** The teacher's hidden states had to come from the
forward Stage-1 already runs, so `reference_branch` was added to
`Stage1Objective`, returning `(final hidden states, pooled)` from one call;
`reference_representation` now delegates to it. Same forward, same locked
pooling, same operation order, same value. This mirrors the `adapted_branch`
refactor C3 required.

The teacher-geometry diagnostic (`teacher_offdiag_cos_mean`) is computed from the
teacher Gram this forward already needed, detached, and read by telemetry only --
no fourth forward.

---

## 5. Architecture and parameters

C2 uses `historical-fusion-v1` and changes no inference architecture. The module
defines no `nn.Parameter`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`register_parameter`, `register_buffer`, projection head, distillation head,
learned temperature or learned relation scale -- asserted by a word-boundary scan
of code-only source.

```
ADAPTER_PARAMETER_DELTA     = 0
EXPECTED_ADAPTER_PARAMETERS = 3,551,232   (historical == C1 == C2 == C3)
```

`verify_model_contract` still enforces 3 551 232 at runtime for every run.

---

## 6. Budget

C2 uses the **same generic `FIRST_SCREEN_HARD_CAP`** object as C1 and C3 -- no
second bespoke implementation. An import guard now checks all three first-screen
ceilings agree, so a future divergence fails at import rather than silently
producing an unmatched comparison.

Verified for C2: a 40 000 cap is refused fresh and on resume; resumes at
0/500/19 500/20 000 are allowed; carried updates of 20 001/25 000/40 000 are
**refused, never clamped**; at a 20 000 boundary the run stops with `cap = 20000`,
`continued = False`, `hard_capped = True`. Historical continuation to 40 000 is
unchanged. No CLI flag can relax it.

Locked C2 values, all inherited and none retuned: LR 1e-4, `r` 1.0,
`lambda_align` 1.0, `lambda_clean` 1.0, `lambda_grd` 1.0, `run_seed` 36930,
`init_seed` 51800, `corruption_seed` 35422, batch 128, `max_length` 256, fp32,
eval every 500, hard cap 20 000.

---

## 7. Validation and checkpoint selection - unchanged

`unmark/stage1/validation.py` remains **entirely untouched**. `selection.py` is
purely additive (`v2_grd_schedule` beside the C1 and C3 schedules); the selection
rule is re-executed in the C2 suite and still returns the same checkpoint.

No relational quantity reaches either module: a scan for `loss_grd`,
`relational_distance`, `cosine_gram`, `objective_relational`,
`RelationalDistillation`, `first_token` and `teacher_offdiag` returns nothing in
both. No sentiment labels, no UIT-VSFC Macro-F1, no official validation, no TEST,
no Stage-2 predictions, no candidate ranking, no best-seed selection.

`GRD_METRIC_USED_FOR_SELECTION = NO.`

---

## 8. W&B

Three candidate-separated projects, resolved by the same register-driven
mechanism, with the historical stages unchanged:

```
C1 -> UNMARK-v2-C1-SCF-Stage1
C2 -> UNMARK-v2-C2-GRD-Stage1
C3 -> UNMARK-v2-C3-GC-Stage1
historical -> unmark-stage1
```

The run key is namespaced by project, and a test proves that C1/C2/C3 -- which
share a label (`seed=36930`) and a seed and differ only by stage -- produce three
distinct keys, so no candidate can resume into another's dashboard.

`SAFE_CONFIG_KEYS` gained `lambda_grd` and `relational` (a real gap this audit
found and closed: without them the GRD specification could not have reached the
run config). The config carries `candidate_id`, `objective_id`, `fusion_id`,
`lambda_grid`, `lambda_grd`, `relational`, `hard_max_updates`, `wandb_project`
alongside the seeds, LR, batch size, backbone pin, corpus digest and inventory
identity.

Telemetry is strictly candidate-separated, verified by executing
`loss_telemetry` on three result shapes:

```
historical  ['loss', 'loss_align', 'loss_clean']
C3          [... , 'loss_grid', 'mean_distance_grid']
C2          [... , 'loss_grd', 'loss_rel_clean', 'loss_rel_corrupt',
                   'grd_teacher_offdiag_cos_mean']
```

The first three keys and their order are identical in every case, so the
historical path is byte-unchanged and **no candidate ever sees another's keys**.
C1's `train/scf/*` series and C3's `train/loss_grid` / `train/mean_distance_grid`
are both asserted still present in the monitor.

W&B is observational only: no module that computes anything mentions it, the
scientific package never imports it, the project name is absent from provenance,
and failure degrades to console-only. **No network call was made here.**

---

## 9. Smoke dispatch

`smoke --candidate {lr_pilot,r_phase1,final_main,v2_gc,v2_grd,v2_scf}`. For
`v2_grd` the shared `candidate_for_stage` -> `build_candidate_objective` path
instantiates `RelationalDistillationObjective` over a historical-fusion adapter,
and **not** `GridConsistencyObjective` -- dispatch is on the objective identity,
never a stage name.

The smoke requires finite `loss`, `loss_align`, `loss_clean` plus, for C2,
`loss_grd`, `loss_rel_clean` and `loss_rel_corrupt`; the grid terms are required
only where `lambda_grid is not None`, so C2 is never asked for an `L_grid` it does
not have and C3 keeps its own requirement. It **refuses a batch of fewer than 2
examples** for any candidate with a relational term, before the forward, because
relational geometry is undefined there.

Still no optimizer, no `.backward()`, no update, no selection, no official
validation, no TEST, no downstream label.

---

## 10. Future Stage-2 loadability

C2 changes the loss, not the model, so reconstruction correctly selects
`historical-fusion-v1` and loads the adapter state strictly -- while the
scientific identity remains C2. The round-trip test proves all four steps,
including the one that matters: `require_loadable_as(payload,
"historical-fusion-v1")` **passes** (it really is the historical architecture),
but `verify_checkpoint(payload, HISTORICAL_PROVENANCE)` **refuses**, so a
historical environment cannot claim a C2 checkpoint as UNMARK-A.

Stage-2 itself is not implemented, planned, run or scored.

---

## 11. Local test results

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| `python -m pytest tests/test_stage1_v2_grd.py -q` | **81 passed, 22 skipped** |
| `python -m pytest tests/test_stage1_v2_scf.py -q` | **67 passed, 17 skipped** |
| `python -m pytest tests/test_stage1_v2_gc.py -q` | **52 passed, 28 skipped** |
| `python -m pytest tests/test_stage1_v2_gc_budget.py -q` | **42 passed, 1 skipped** |
| `python -m pytest tests/ -q` | **4696 passed, 241 skipped, 0 failed** (161s) |

Movement against the Audit 067 state (4614 passed / 218 skipped): **+82 passed**
(81 new C2 torch-free tests, 1 new C3 test added while updating for the third
objective) and **+23 skipped** (22 new C2 torch-gated tests, plus one
name-resolution parametrisation that globs `unmark/stage1/*.py` and picked up
`objective_relational.py`).

**C1's 17 and C3's 29 torch-gated tests are intact** -- verified by count. None
were deleted, weakened, xfailed or bypassed.

Four tests were updated to follow real changes, each preserving its guarantee:
`test_only_the_reference_branch_uses_no_grad` now checks `reference_branch` (where
the `no_grad` moved) **and** additionally asserts the public accessor is a real
delegation rather than a second forward; three of my own C3 tests assumed only
two objectives existed and now assert the property that matters -- exactly one
registered stage trains the grid objective, and every other objective reports
`lambda_grid is None`.

---

## 12. Regression checks

**C1 (V2-SCF):** SCF formula unchanged at `adapter.py:130-133`
(`e_norm`/`f_norm`/`clamp(min=FUSION_SCALE_EPSILON)`/`scale * fused`); objective
still the historical one; fusion still `scale-calibrated-fusion-v1`; 3 forwards;
hard cap 20 000; W&B project `UNMARK-v2-C1-SCF-Stage1`.

**C3 (V2-GC):** grid formula unchanged at `objective_grid.py:214-235`
(teacher detach, `eps=COSINE_EPS`, per-example `sum(dim=1)/counts`);
`lambda_grid` still 1.0; fusion still historical; 3 forwards; hard cap 20 000;
W&B project `UNMARK-v2-C3-GC-Stage1`.

---

FINAL AUDIT — V2-GRD

BASELINE_ANCESTRY:
c3e4cc40117ac0b3647991901971d08177117ca3

CURRENT_HEAD:
c3e4cc40117ac0b3647991901971d08177117ca3 (no commit created)

WORKTREE_DIRTY:
YES — 20 modified, 11 untracked, 0 staged, 0 commits made

CANDIDATE_ID:
v2_grd

OBJECTIVE_ID:
geometry-relational-distillation-v1

FUSION_ID:
historical-fusion-v1

TEACHER_REPRESENTATION:
native clean FIRST_TOKEN — hidden[:, 0, :] of the frozen encoder's reference branch on the CLEAN ORIGINAL input, never a corrupted condition

STUDENT_CLEAN_REPRESENTATION:
adapted clean FIRST_TOKEN — hidden[:, 0, :] of the adapted CLEAN branch

STUDENT_CORRUPT_REPRESENTATION:
adapted corrupt FIRST_TOKEN — hidden[:, 0, :] of the adapted CORRUPTED branch

GRAM_DEFINITION:
X_hat_i = X_i / clamp(||X_i||_2, min=1e-8) over dim=-1; G(X) = X_hat @ X_hat.T, shape [B, B]

RELATION_LOSS:
R(S,T) = mean over ordered pairs i != j of ( G(S)[i,j] - G(T)[i,j] )^2 — squared error only, no L1, no KL, no covariance, no centering, no temperature, no top-k, no mining, no subsampling

DIAGONAL_INCLUDED:
NO — excluded structurally by a ~torch.eye mask; including it would scale the loss by exactly (B-1)/B, confirmed numerically

RELATION_EPSILON:
1e-8 exactly (protocol.RELATION_EPSILON)

TEACHER_RELATION_TARGET_DETACHED:
YES — teacher_target.detach() inside relational_distance, unconditionally

CLEAN_RELATION_GRADIENT:
YES — the adapted clean branch is never detached

CORRUPT_RELATION_GRADIENT:
YES — the adapted corrupt branch is never detached

RELATIONAL_CLEAN_WEIGHT:
0.5

RELATIONAL_CORRUPT_WEIGHT:
0.5

LAMBDA_GRD:
1.0

OBJECTIVE_FORMULA:
L_total = 1.0*L_align + 1.0*L_clean + 1.0*L_grd, with L_grd = 0.5*L_rel_clean + 0.5*L_rel_corrupt

GRID_LOSS_PRESENT:
NO — C2.objective.lambda_grid is None; the module contains no token_grid_distance and no loss_grid

SCALE_CALIBRATION_PRESENT:
NO — C2 uses historical-fusion-v1; the module never mentions fusion_id or scale calibration

ENCODER_FORWARDS:
3

ADAPTER_PARAMETER_DELTA:
0 (STATICALLY_VERIFIED; RUNTIME_NOT_EXECUTED — torch absent locally)

EXPECTED_ADAPTER_PARAMETERS:
3551232

CROSS_CANDIDATE_PROVENANCE_FAIL_CLOSED:
YES — 20/20 cells of the 5x4 matrix correct (legacy/historical/C1/C2/C3 against historical/C1/C2/C3)

C2_HARD_MAX_UPDATES:
20000

C2_CAN_EXECUTE_UPDATE_20001:
NO — resolve_run_cap refuses any cap above 20000 as the first statement of train_run, before the update loop exists

HISTORICAL_CONTINUATION_PRESERVED:
YES — the historical identity keeps PRECOMMITTED_CONTINUATION and still promotes 20k -> 40k at the boundary

STAGE1_SELECTION_CHANGED:
NO — validation.py untouched; selection.py purely additive; the rule re-executed

GRD_METRIC_USED_FOR_SELECTION:
NO — no relational quantity appears in validation.py or selection.py; the terms reach only the telemetry sink

WANDB_PROJECT_C1:
UNMARK-v2-C1-SCF-Stage1

WANDB_PROJECT_C2:
UNMARK-v2-C2-GRD-Stage1

WANDB_PROJECT_C3:
UNMARK-v2-C3-GC-Stage1

WANDB_OBSERVATIONAL_ONLY:
YES — no computing module mentions wandb, the scientific package never imports it, the project name is absent from provenance, run keys are project-namespaced so candidates cannot cross-resume, and failure degrades to console-only; no network call was made

C2_REAL_MODEL_SMOKE_DISPATCH:
IMPLEMENTED — `smoke --candidate v2_grd` resolves via candidate_for_stage and builds RelationalDistillationObjective over a historical-fusion adapter through the shared constructor; asserts L_align, L_clean, L_rel_clean, L_rel_corrupt, L_grd and the total finite; refuses a batch of fewer than 2 examples

C2_CHECKPOINT_RECONSTRUCTABLE_FOR_FUTURE_STAGE2:
YES — reconstruction selects historical-fusion-v1 and loads strictly, while the recorded identity stays C2 and a historical environment still refuses the checkpoint

C1_REGRESSION:
PASS — SCF formula, historical objective, 3 forwards, 20k cap and W&B project all unchanged; 17 torch-gated tests intact

C3_REGRESSION:
PASS — grid formula, historical fusion, lambda_grid 1.0, 3 forwards, 20k cap and W&B project all unchanged; 29 torch-gated tests intact

LOCAL_TESTS_PASSED:
4696

LOCAL_TESTS_FAILED:
0

LOCAL_TESTS_SKIPPED:
241

C1_TORCH_TESTS_PENDING:
17

C2_TORCH_TESTS_PENDING:
22

C3_TORCH_TESTS_PENDING:
29

COLAB_GPU_TESTS_REQUIRED_BEFORE_TRAINING:
- the 22 C2 torch-gated tests: hand-computed Gram, hand-computed off-diagonal MSE, genuine diagonal exclusion via the (B-1)/B identity, exact zero on identical geometry, scale invariance, feature-dimension-only normalisation, the zero-row path, B<2 and shape mismatch fail-closed, NaN/Inf refusal, teacher-gradient absence, FIRST_TOKEN indexing, the builder returning the relational objective, the 3-forward counter, exact 0.5/0.5 and lambda_grd composition, pooled-term equality with the historical objective, both students receiving relational gradient, frozen-encoder gradient absence, zero parameter delta, the full identity in the result dict, teacher-vs-corrupt distinctness, and end-to-end B=1 refusal
- the 17 C1 and 29 C3 torch-gated tests, still pending from Audits 066/067
- `stage1_runner.py smoke --candidate v2_grd`, `--candidate v2_scf` and `--candidate v2_gc` against real PhoBERT and the real prepared corpus

BLOCKERS_BEFORE_20K_TRAINING:
- NONE in the repository. C1, C2 and C3 coexist in one HEAD, each isolated, each hard-capped on the real execution path, each provenance-separated in every direction, each with its own W&B project, and each reconstructable.
- REMAINING GATE (environmental, not a defect): 68 torch-gated tests (17 C1 + 22 C2 + 29 C3) and the three real-model smokes have never executed, because torch is absent from the local ML-free venv by design and was not installed. The relational numerics, gradient routing, forward counts and parameter deltas are STATICALLY_VERIFIED only. Run them on Colab/GPU first.
- OPEN (non-blocking, unchanged): the adjudication protocol for comparing C1, C2, C3 and UNMARK-A/B is undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_COLAB_GATE
