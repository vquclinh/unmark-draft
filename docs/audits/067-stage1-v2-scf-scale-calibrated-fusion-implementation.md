# Audit 067 - Stage-1 V2-SCF (Scale-Calibrated Fusion) Implementation and Audit

**Scope:** implement UNMARK-v2 Candidate **C1 / V2-SCF** side by side with the
already-implemented C3 / V2-GC, then audit C1 and re-check that C3 still has the
semantics Audits 065 and 066 established.
**Date:** 2026-09-11
**Type:** implementation plus scientific audit. No commit was created, no model
was trained, no UIT-VSFC data, official validation or official TEST was accessed,
and no ML package was installed in the ML-free local environment.

Audits 065 and 066 are byte-unchanged (`md5 61672fb15d1a22342848cf4e5d18b23d`
and `35076e34f9ffd8360da4fb53c15cbc00`).

---

## 0. Executive verdict

**PASS_WITH_COLAB_GATE.**

C1 is implemented as an isolated candidate: a per-token scale calibration of `f`
to `||e||` inserted between the existing LayerNorm and the existing gated
mixture. It adds **zero trainable parameters**, keeps the **historical Stage-1
objective**, runs the **same three encoder forwards**, and is hard-capped at
**20 000 updates** through the generic budget abstraction introduced for C3.
Historical UNMARK and C3 are both untouched.

The remaining gate is environmental: 17 new C1 torch-gated tests (and the 29 C3
ones) cannot execute locally and must run on Colab/GPU before the screening run.

---

## 1. Scientific motivation (restated, locked)

Post-hoc diagnostics D3/D4 on protocol-dev measured, for the historical adapter:

```
mean ||e||          ~= 1.41
mean ||f||          ~= 27.9
mean ||f|| / ||e||  ~= 20.5
gate mean (FULL)    ~= 0.03
||z-e|| / ||e||     ~= 1.42
```

A numerically small gate does not imply a small intervention, because the two
branches of `z = g*f + (1-g)*e` live on very different scales. C1 tests exactly
one mechanism: **calibrate the scale of `f` to `e` before the existing gated
mixture**, and change nothing else.

C1 does not attempt grid consistency, relational distillation, severity routing
or anything downstream.

---

## 2. Exact implementation

`unmark/modeling/adapter.py :: scale_calibrated_fusion`, verbatim:

```python
e_norm = base.norm(dim=-1, keepdim=True)
f_norm = fused.norm(dim=-1, keepdim=True)
scale = e_norm / f_norm.clamp(min=FUSION_SCALE_EPSILON)
return scale * fused
```

and in `OrthographyInputAdapter.forward`:

```python
f = self.layer_norm(self.fusion(q))
if self.config.is_scale_calibrated:          # C1 ONLY
    f = scale_calibrated_fusion(f, e)
if self.gate is None:
    return f
g = torch.sigmoid(self.gate(q))
z = convex_combination(g, f, e)
```

| Locked requirement | Status |
| --- | --- |
| `scale = ||e|| / clamp(||f||, 1e-8)`, `f_cal = scale * f` | **PASS** |
| epsilon exactly `1e-8` (`FUSION_SCALE_EPSILON`) | **PASS** |
| per token, over the final hidden dimension only (`dim=-1, keepdim=True`) | **PASS** |
| `scale` shaped `[B, L, 1]` | **PASS** |
| fully differentiable through `f` and through `scale`; nothing detached | **PASS** |
| `g` unchanged, gate dimensionality unchanged, `g` not clipped | **PASS** |
| no learned scale parameter | **PASS** |
| no severity features | **PASS** |
| tone / letter channels, `q`, `W_f`, LayerNorm unchanged | **PASS** |
| vector gate retained (not replaced by a scalar) | **PASS** |
| NOT batch-level, NOT sequence-level | **PASS** |
| `e` itself is never normalised | **PASS** |
| calibration sits strictly between LayerNorm and the gated mixture | **PASS** (asserted by ordering test) |

The formula was verified against a reference implementation written from the
specification rather than from the code: `||e||=5, ||f||=27.9` gives
`scale = 0.17921147` and `||f_cal|| = 5.00000000` exactly, and a degenerate
`||f|| = 0` yields a finite, deterministic `f_cal = 0`.

The same reference shows the mechanism doing what C1 claims: at `g = 0.03` and a
20x scale gap, `||z-e||/||e||` falls from `0.1445` (historical) to `0.0190`
(calibrated).

---

## 3. Candidate isolation

`AdapterConfig.fusion_id` defaults to `historical-fusion-v1`, so **every existing
construction site keeps its exact behaviour** and the branch above is not entered
on the historical path -- not one tensor op is added to it.

The candidate register (`unmark/stage1/candidates.py`) now binds four things per
stage, and the budget key moved from the objective alone to the **pair**:

```
lr_pilot / r_phase1 / final_main
    objective align-clean-pooled-v1 | fusion historical-fusion-v1
    budget precommitted_initial_then_one_continuation | wandb unmark-stage1

v2_scf  (C1)
    objective align-clean-pooled-v1 | fusion scale-calibrated-fusion-v1
    budget first_screen_hard_cap (20 000) | wandb UNMARK-v2-C1-SCF-Stage1

v2_gc   (C3)
    objective grid-consistency-v1  | fusion historical-fusion-v1
    budget first_screen_hard_cap (20 000) | wandb UNMARK-v2-C3-GC-Stage1
```

**Why the key had to become a pair.** C1 trains the *historical* objective. An
objective-keyed register -- which is what Audit 066 left -- would have handed C1
`PRECOMMITTED_CONTINUATION` and let it run to 40 000 updates: the exact class of
defect Audit 065 raised against C3, reintroduced through a different door. This
is now `budget_for_identity(objective_id, fusion_id)`, and there is a test for
the pair that was never registered.

C1's Stage-1 objective is the historical one: `L = 1.0*L_align + 1.0*L_clean`.
**No `L_grid`, no geometry/relational loss, no downstream loss.** C1 differs from
historical UNMARK by its fusion rule and nothing else.

C2 (V2-GRD) is added by appending one `StageCandidate`; nothing in `trainer` or
`execute` learns a candidate's name, asserted by test.

---

## 4. Provenance

`RunProvenance` gained `fusion: FusionIdentity`, defaulting to the historical
fusion, compared by `require_match`, and covered by the finalist verifier.

**Executed compatibility matrix (9/9 correct):**

| Payload | Environment | Required | Observed |
| --- | --- | --- | --- |
| legacy (no objective/fusion) | historical | PASS | **PASS** |
| legacy | C1 | REFUSE | **REFUSE** |
| C1 | historical | REFUSE | **REFUSE** |
| C1 | C1 | PASS | **PASS** |
| C1 | C3 | REFUSE | **REFUSE** |
| C3 | C1 | REFUSE | **REFUSE** |
| historical | C1 | REFUSE | **REFUSE** |
| historical | historical | PASS | **PASS** |
| C3 | C3 | PASS | **PASS** |

A provenance without a `fusion` block is read as, and only as,
`historical-fusion-v1` -- which keeps the frozen UNMARK-A/B checkpoints verifying
unchanged. No historical checkpoint was rewritten.

`docs/spec/stage1-adapter-finalists.json` changed in exactly **two leaves**, both
under `/evidence/provenance_fields_verified/` (the field list gained `fusion`,
and its note). A structural comparison against the baseline confirms
`adjudication.finalists` is **identical**: no hash, seed, selected update,
validation score, robust score or scientific result was touched.

---

## 5. Model construction and future Stage-2 loadability

One authoritative path: `AdapterConfig(fusion_id=...)` ->
`OrthographyInputAdapter`, reached through `initialisation.fresh_adapter(...,
fusion_id)` and `execute.build_candidate_objective`. `execute_stage` derives the
fusion from the candidate register, and `build_candidate_objective` **fails
closed** if the adapter it is handed does not implement the fusion the candidate
declares.

`unmark/stage1/reconstruct.py` is the new candidate-checkpoint construction API:

```
payload -> recorded_fusion_id -> adapter_for_provenance -> load_adapter_state(strict=True)
```

This exists because a Stage-1 adapter's `state_dict` has the **same eight keys
and the same shapes for every fusion** -- a C1 checkpoint loads into the
historical adapter with `strict=True` and no error, producing a model that is
numerically wrong and structurally plausible. `require_loadable_as` is the guard
a single-architecture loader calls, and the round-trip test proves both
directions: a C1 payload reconstructs into the calibrated fusion and is refused
by a historical loader; a legacy payload reconstructs into the historical fusion.

**Stage-2 itself is not implemented, planned, run or scored here.**

---

## 6. Training protocol and the 20 000 hard cap

Every C1 value is inherited from the closed campaign and none is retuned:

```
learning_rate 1e-4 | r 1.0 | lambda_align 1.0 | lambda_clean 1.0
run_seed 36930 | init_seed 51800 | corruption_seed 35422
batch_size 128 | max_length 256 | fp32 | eval every 500
hard max_updates 20 000
```

`run_seed`, `init_seed` and the parameter set are identical to UNMARK-A's and to
C3's, and a fusion consumes no RNG, so **C1 starts from bit-identical weights**
and diverges only through the mixture rule. That is what makes the three-way
comparison paired, and it is asserted by test
(`test_runtime_the_two_adapters_start_from_identical_weights`).

The cap is enforced by the **same generic abstraction built for C3** -- no second
bespoke implementation:

| Refusal point | What it refuses |
| --- | --- |
| `trainer.resolve_run_cap` (first statement in `train_run`) | any cap or resumed `global_update`/`cap` above 20 000 |
| `trainer.resolve_budget` | raising `result.cap` for a hard-capped candidate |
| `execute.continuation_permitted` | entering the second `train_run` leg at all |

Verified for C1: a 40 000 cap is refused fresh and on resume; resumes at
0/500/19 500/20 000 are allowed; carried updates of 20 001/25 000/40 000 are
**refused, never clamped**; at a 20 000 boundary the run stops with
`cap = 20000`, `continued = False`, `hard_capped = True`. Historical continuation
to 40 000 still works, unchanged.

No CLI flag can relax it: the runner declares no `--max-updates`, `--cap`,
`--budget`, `--fusion`, `--scale`, `--epsilon` or `--objective`.

---

## 7. Validation and checkpoint selection - unchanged

`unmark/stage1/validation.py` is **entirely untouched**. `selection.py` is purely
additive (`v2_scf_schedule` beside `v2_gc_schedule`); `ValidationPoint`,
`.score`, `select_checkpoint`, `select_learning_rate`, `select_r` and
`budget_decision` are unmodified, and the selection rule is re-executed in the
C1 suite.

No scale quantity reaches either module: a grep for `scale_calibrated`,
`fusion_id`, `scale_diagnostics`, `intervention_ratio`, `scale_factor`,
`postcal`, `scale_telemetry` and `AdapterConfig` returns nothing in both. No
downstream label, no UIT-VSFC Macro-F1, no official validation, no TEST, no
candidate ranking and no best-seed selection.

`SCALE_TELEMETRY_USED_FOR_SELECTION = NO.`

---

## 8. W&B - candidate-separated, observational only

`project_for_event` resolves the project from the candidate register:
explicit `--project` > register > `DEFAULT_PROJECT`. `--project` now defaults to
`None` (resolve per candidate); the historical stages resolve to
`unmark-stage1`, so the existing campaign's dashboard is unchanged.

The W&B run id key is namespaced by project, so C1 and C3 -- which share a label
and a seed and differ only by stage -- cannot resume into each other's run.

`run_start` emits `candidate_id`, `objective_id`, `fusion_id`,
`hard_max_updates`, `wandb_project` alongside the existing seeds, LR, batch size,
backbone pin, corpus digest and inventory identity, and `SAFE_CONFIG_KEYS`
whitelists exactly those. The whitelist filters; production emits.

**Passive C1 telemetry** is implemented without an extra encoder forward: the
adapter already computes `e_norm` and `f_norm` for the scale, so
`_record_scale_diagnostics` stores four **detached 0-dim tensors** under
`torch.no_grad()` and `scale_telemetry` converts them to floats only at emit
time, off the training path. It is inert for every non-calibrated adapter -- the
historical and C3 paths execute nothing there, asserted structurally. The values
surface as `train/scf/scale_factor`, `train/scf/postcal_f_over_e`,
`train/scf/intervention_ratio`, `train/scf/gate_mean`.

`_scale_diagnostics` is a plain attribute, never a registered buffer, so it
cannot enter `state_dict()`, a persisted payload or the parameter count.

Observational only, asserted by test: no module under `unmark/` imports or
mentions `wandb`; the project name is absent from provenance; a W&B failure
degrades to console-only; nothing it touches reaches a loss, seed, budget or
selection. **No network call was made in this environment.**

---

## 9. Smoke dispatch

`smoke_check(stage=...)` resolves the candidate through `candidate_for_stage`
**before** building the model, then builds the candidate's own adapter via
`build_objective(revision, fusion_id=...)` and its objective via the shared
`build_candidate_objective`. The CLI offers every registered candidate:
`--candidate {lr_pilot,r_phase1,final_main,v2_gc,v2_scf}`.

For `--candidate v2_scf` this instantiates the historical objective over the
scale-calibrated adapter, and **not** `GridConsistencyObjective` -- the grid
terms are required only when `candidate.objective.lambda_grid is not None`, so C1
is never asked for an `L_grid` it does not have. The smoke asserts `L_align`,
`L_clean` and the total finite, and additionally asserts every scale diagnostic
finite. It still constructs no optimizer, calls no `.backward()`, selects no
checkpoint, and touches no official validation, TEST or downstream label.

---

## 10. Local test results

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| `python -m pytest tests/test_stage1_v2_scf.py -q` | **67 passed, 17 skipped** |
| `python -m pytest tests/test_stage1_v2_gc.py -q` | **51 passed, 28 skipped** |
| `python -m pytest tests/test_stage1_v2_gc_budget.py -q` | **42 passed, 1 skipped** |
| `python -m pytest tests/ -q` | **4614 passed, 218 skipped, 0 failed** (157s) |

Movement against the Audit 066 state (4545 passed / 201 skipped): **+69 passed**
(67 new C1 torch-free tests, 1 new budget-pair test, 1 new parametrisation of the
name-resolution sweep, which globs `unmark/stage1/*.py` and picked up
`reconstruct.py`) and **+17 skipped** (the new C1 torch-gated tests).

**The 29 C3 torch-gated tests are intact** -- 28 in `test_stage1_v2_gc.py` and 1
in `test_stage1_v2_gc_budget.py`, verified by name. None were deleted, weakened,
xfailed or converted into vacuous static tests. The only skip marker in any
candidate test file is the `torch is None` gate.

Four historical/C3 tests were updated to follow real changes, each preserving its
guarantee: the C3 budget tests now call `budget_for_identity` (the key became a
pair), two of my own C3 tests follow the new provenance field and the new
fail-closed guard in the shared constructor, and one adapter prose string was
reworded because it contained the word "checkpoint", which a historical guard
greps for as backbone-specific position logic. That historical test itself was
**not** modified.

---

## 11. C3 regression re-check

| Property | Status |
| --- | --- |
| grid-consistency formula unchanged | **PASS** -- `objective_grid.py` untouched by C1; detach, `eps=COSINE_EPS`, per-example reduction and the three-term composition verified in place |
| C3 still uses historical fusion | **PASS** -- `candidate_for_stage("v2_gc").fusion == historical-fusion-v1` |
| `lambda_grid` still 1.0 | **PASS** |
| 3 encoder forwards | **PASS** |
| hard cap still 20 000 | **PASS** |
| C3 W&B project | **PASS** -- `UNMARK-v2-C3-GC-Stage1` |

---

FINAL AUDIT — V2-SCF

BASELINE_ANCESTRY:
c3e4cc40117ac0b3647991901971d08177117ca3

CURRENT_HEAD:
c3e4cc40117ac0b3647991901971d08177117ca3 (no commit created)

WORKTREE_DIRTY:
YES — 19 modified, 8 untracked, 0 staged, 0 commits made

CANDIDATE_ID:
v2_scf

OBJECTIVE_ID:
align-clean-pooled-v1

FUSION_ID:
scale-calibrated-fusion-v1

SCF_FORMULA:
e_norm = ||e||_2 ; f_norm = ||f||_2 ; scale = e_norm / clamp(f_norm, min=1e-8) ; f_cal = scale * f ; z = g * f_cal + (1 - g) * e

SCALE_NORM_SCOPE:
per token, over the final hidden dimension only (dim=-1, keepdim=True -> scale shape [B, L, 1]). Not batch-level, not sequence-level.

SCALE_EPSILON:
1e-8 exactly (modeling.contracts.FUSION_SCALE_EPSILON, imported never retyped)

SCALE_GRADIENT_DETACHED:
NO — nothing is detached; gradient flows through f and through scale (hence into e via ||e||)

HISTORICAL_FUSION_CHANGED:
NO — fusion_id defaults to historical-fusion-v1 and the calibration branch is not entered; the historical equation is reproduced independently from the same weights and compared bit-exactly

C3_FUSION_CHANGED:
NO — C3 keeps historical-fusion-v1

OBJECTIVE_FORMULA:
L_total = 1.0*L_align + 1.0*L_clean

GRID_LOSS_PRESENT:
NO — C1 has no L_grid, no geometry/relational loss and no downstream loss

ENCODER_FORWARDS:
3

ADAPTER_PARAMETER_DELTA:
0 (STATICALLY_VERIFIED from the symbolic parameter count, per-term identical; RUNTIME_NOT_EXECUTED — torch absent locally)

EXPECTED_ADAPTER_PARAMETERS:
3551232

FUSION_PROVENANCE_DURABLE:
YES — RunProvenance.fusion is recorded in every checkpoint and artifact, compared by require_match, and covered by VERIFIED_PROVENANCE_FIELDS

LEGACY_CHECKPOINT_DEFAULT_FUSION:
historical-fusion-v1, and only that — absence is read as the historical claim, never as "unknown"

CROSS_CANDIDATE_RESUME_FAIL_CLOSED:
YES — 9/9 matrix cases correct (legacy/historical/C1/C3 in both directions)

C1_HARD_MAX_UPDATES:
20000

C1_CAN_EXECUTE_UPDATE_20001:
NO — resolve_run_cap refuses any cap above 20000 as the first statement of train_run, before the update loop exists

HISTORICAL_CONTINUATION_PRESERVED:
YES — the historical (objective, fusion) pair keeps PRECOMMITTED_CONTINUATION and still promotes 20k -> 40k at the boundary

STAGE1_SELECTION_CHANGED:
NO — validation.py untouched; selection.py purely additive; the selection rule re-executed

SCALE_TELEMETRY_USED_FOR_SELECTION:
NO — no scale quantity appears in validation.py or selection.py; the diagnostics reach only the telemetry sink

WANDB_PROJECT_C1:
UNMARK-v2-C1-SCF-Stage1

WANDB_PROJECT_C3:
UNMARK-v2-C3-GC-Stage1

WANDB_OBSERVATIONAL_ONLY:
YES — no module under unmark/ imports or mentions wandb; the project name is absent from provenance; failure degrades to console-only; no RNG, loss, budget or selection is touched; no network call was made in this environment

C1_REAL_MODEL_SMOKE_DISPATCH:
IMPLEMENTED — `smoke --candidate v2_scf` resolves via candidate_for_stage, builds the scale-calibrated adapter and the historical objective through the shared constructor, never GridConsistencyObjective, asserts L_align/L_clean/total and every scale diagnostic finite, and requires grid terms only where the candidate has one

C1_CHECKPOINT_RECONSTRUCTABLE_FOR_FUTURE_STAGE2:
YES — unmark/stage1/reconstruct.py builds the adapter from the recorded fusion and loads strictly; require_loadable_as refuses a C1 payload offered to a historical loader, which is necessary because the state_dict is shape-compatible across fusions

LOCAL_TESTS_PASSED:
4614

LOCAL_TESTS_FAILED:
0

LOCAL_TESTS_SKIPPED:
218

C1_TORCH_GATED_TESTS_PENDING:
17

C3_TORCH_GATED_TESTS_STILL_PENDING:
29 (28 in test_stage1_v2_gc.py + 1 in test_stage1_v2_gc_budget.py; intact and unmodified)

COLAB_GPU_TESTS_REQUIRED_BEFORE_TRAINING:
- the 17 C1 torch-gated tests: scale formula vs a hand-computed value, per-token calibration, not-batch/not-sequence scope, the [B,L,1] scale shape, the zero-norm path, gradient flow through f and scale, z reconstruction, the intervention-ratio reduction at equal gate, gate identity across fusions, identical start weights, zero parameter delta, bit-unchanged historical adapter output, diagnostics not changing z, no diagnostics on a historical adapter, C1's 3 encoder forwards with no L_grid, and both checkpoint reconstruction directions
- the 29 C3 torch-gated tests, still pending from Audit 066
- `stage1_runner.py smoke --candidate v2_scf` and `--candidate v2_gc` against real PhoBERT and the real prepared corpus

BLOCKERS_BEFORE_20K_TRAINING:
- NONE in the repository. C1 is isolated, hard-capped on the real execution path, provenance-separated in both directions, and its checkpoints are reconstructable.
- REMAINING GATE (environmental, not a defect): 46 torch-gated tests (17 C1 + 29 C3) and both real-model smokes have never executed, because torch is absent from the local ML-free venv by design and was not installed. The scale numerics, gradient routing, zero-parameter delta, bit-unchanged historical output and the reconstruction round trip are STATICALLY_VERIFIED only. Run them on Colab/GPU first.
- OPEN (non-blocking, unchanged): the adjudication protocol for comparing C1, C3 and UNMARK-A/B is undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_COLAB_GATE
