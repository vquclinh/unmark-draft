# Audit 066 - Stage-1 V2-GC Hard-Cap and Smoke-Dispatch Repair

**Scope:** repair the two blockers Audit 065 raised against the V2-GC (Grid
Consistency) candidate, then re-audit the complete C3 implementation.
**Date:** 2026-09-11
**Type:** repair plus fresh scientific audit. No commit was created, no model was
trained, no UIT-VSFC data, official validation or official TEST was accessed, and
no ML package was installed in the ML-free local environment.

Audit 065 is unchanged (`md5 61672fb15d1a22342848cf4e5d18b23d`). This audit does
not restate it; it records what was repaired and re-verifies C3 end to end.

---

## 0. Executive verdict

**PASS_WITH_COLAB_GATE.**

Both Audit 065 blockers are closed:

* **BLOCKER 1 (hard cap)** -- V2-GC is now hard-capped at 20 000 updates by the
  execution path itself. The 20k -> 40k promotion is refused at three
  independent points, and `V2_GC_MAX_UPDATES` is the value those points enforce
  rather than a decorative constant.
* **BLOCKER 4 (smoke dispatch)** -- the real-model smoke now resolves and builds
  the requested candidate through the *same* register and the *same* constructor
  a real run uses, so `smoke --candidate v2_gc` exercises a genuine
  `GridConsistencyObjective` forward.

The audited V2-GC mathematics is untouched. The remaining gate is environmental:
29 torch-gated tests cannot execute locally and must run on Colab/GPU before the
20 000-update screening run.

---

## 1. What was repaired

### 1.1 The hard cap (Audit 065 BLOCKER 1)

Audit 065 proved the promotion path:

```
execute.py  leg_cap = INITIAL_MAX_UPDATES (20 000)
trainer.resolve_budget        -> budget_decision says "continue" at the boundary
                              -> result.cap = EXTENDED_MAX_UPDATES (40 000)
execute.py  if leg_cap == INITIAL and result.cap == EXTENDED:  -> train_run(cap=40000)
```

The defect was that **no component owned a candidate's screening budget**. The
repair gives it one.

**New module `unmark/stage1/candidates.py` (torch-free).** A `ScreeningBudget`
declares a candidate's ceiling and whether the locked precommitted continuation
may be applied to it; a `StageCandidate` binds a stage name to an objective
identity and a budget; `CANDIDATES` is the closed register.

```
lr_pilot / r_phase1 / final_main  -> align-clean-pooled-v1  + PRECOMMITTED_CONTINUATION
v2_gc                             -> grid-consistency-v1    + FIRST_SCREEN_HARD_CAP (20 000)
```

`FIRST_SCREEN_HARD_CAP.hard_max_updates` **is** `V2_GC_MAX_UPDATES`. The constant
Audit 065 called decorative is now the enforced value, and
`test_v2_gc_max_updates_is_load_bearing_not_decorative` proves it by asserting
the enforcement path refuses `V2_GC_MAX_UPDATES + 1`.

`ScreeningBudget` refuses; it never clamps. `require_cap` rejects a cap above the
ceiling, and `require_update_within` rejects a carried update above it, because
clamping a payload that records 30 000 executed updates would silently
reinterpret real work as absent.

A self-contradicting budget (a ceiling *and* permission to continue to 40 000)
cannot be constructed at all -- it raises in `__post_init__`.

**Three independent refusals now stand between V2-GC and update 20 001:**

| # | Where | What it refuses |
| --- | --- | --- |
| 1 | `trainer.resolve_run_cap` (`unmark/stage1/trainer.py`) | any cap, or any resumed `global_update`/`cap`, above the candidate's ceiling |
| 2 | `trainer.resolve_budget` | raising `result.cap` for a candidate whose budget forbids the continuation |
| 3 | `execute.continuation_permitted` | entering the second `train_run` leg at all |

Refusal 1 is the structural one. `train_run`'s loop is `while global_update <
cap`, so proving `cap <= 20 000` before the loop proves update 20 001 is
unreachable. The gate was deliberately placed **first in `train_run`**, ahead of
the adapter lookup, the corruption check, `verify_model_contract` and the
optimizer: a run that is not allowed to happen should not get as far as building
something.

**Truthful metadata (Audit 065 requirement C).** `RunResult` gained
`hard_capped`, and `to_dict()` now emits `budget_policy` and
`stopped_at_hard_cap`. A V2-GC run whose best checkpoint is exactly 20 000
records `cap = 20000`, `continued_past_initial_budget = false`,
`stopped_at_hard_cap = true`, `budget_limited = true`. It never claims a 40 000
cap that no execution was authorised to reach.

**Historical behaviour is byte-identical.** `PRECOMMITTED_CONTINUATION` has
`hard_max_updates = None`, so `require_cap` and `require_update_within` are
no-ops for it and `resolve_budget` takes exactly its former branch. CASE 3 below
passes both before and after the repair.

### 1.2 The real-model smoke dispatch (Audit 065 BLOCKER 4)

`smoke_check` built `Stage1Objective` unconditionally, so no real-model check of
a candidate objective was possible.

The objective construction moved into one shared module-level function,
`execute.build_candidate_objective(unmark_encoder, candidate, weights)`, used by
**both** `execute_stage` and `smoke_check`. Dispatch is on the candidate's
objective identity -- never on a stage string compared at the call site -- so
adding C1/C2 adds a register entry, not a branch here.

`smoke_check` gained a `stage` parameter (default: the historical objective, so
prior behaviour is unchanged) resolved by the same `candidate_for_stage`. The CLI
exposes it as `smoke --candidate {lr_pilot,r_phase1,final_main,v2_gc}`, a
dispatch selector over the closed register. It is not a scientific override:
smoke takes no optimizer step, so nothing it selects can change a locked value or
relax a budget.

The smoke now also **asserts every loss term finite before reporting**, requiring
`loss_grid` and `mean_distance_grid` exactly when the candidate's objective has a
grid term -- which doubles as proof that the dispatch built what the register
promised. It still constructs no optimizer, calls no `.backward()`, selects no
checkpoint, and touches no official validation, TEST or downstream label.

### 1.3 Historical tests updated (declared, not hidden)

Four historical tests pinned the objective construction to its old site and had
to follow it. Each was updated to preserve its guarantee, and three were
strengthened:

| Test | Change |
| --- | --- |
| `test_stage1_name_resolution.py::test_the_original_defect_is_detected_by_this_test` | mutation literal retargeted to the moved construction line; mutation check identical in kind |
| `test_stage1_name_resolution.py::test_the_objective_is_constructed_from_a_locally_bound_class` | now checks `build_candidate_objective` binds **all three** lazily imported names (was: one name in `execute_stage`), and still asserts `execute_stage` has no `objective_cls` |
| `test_stage1_run_independence.py::test_a_fresh_adapter_is_constructed_inside_the_loop` | requires `build_candidate_objective` per nominal run; **new** companion test asserts that helper really constructs both objective classes |
| `test_stage1_runner_cli_contract.py::test_smoke_help_reflects_the_accepted_contract` | expected smoke option set gained `--candidate`; the declared-equals-documented guarantee is unchanged |

No test was deleted, weakened, xfailed, or converted into a vacuous static check.

---

## 2. Mutation check - the new tests detect the Audit 065 defect

Audit 065 required that the repair's tests fail against the pre-repair code. This
was demonstrated by restoring the exact pre-repair `resolve_budget` **in memory**
(no file was modified) and re-running the boundary cases:

```
test_case2_v2_gc_at_the_boundary_hard_stops                    FAILED
  AssertionError: cap was raised to the 40k leg: this is the
  Audit 065 BLOCKER 1 regression
  assert 40000 == 20000
test_case2_the_result_artifact_is_honest_about_the_hard_cap    FAILED
  assert 40000 == 20000
test_case3_historical_at_the_boundary_still_continues_to_40000  PASSED
```

The CASE 2 tests detect the exact defect; the CASE 3 historical test passes both
before and after, which is the evidence that historical behaviour was preserved
rather than merely claimed. `resolve_run_cap` and `continuation_permitted` did
not exist pre-repair, so every test touching them fails against that code too.

---

## 3. Required cases (Audit 065 section 3)

| Case | Requirement | Result |
| --- | --- | --- |
| 1 | V2-GC best < 20k -> stop normally, no 40k leg | **PASS** (4 parametrisations) |
| 2 | V2-GC best == 20k -> hard stop, no second `train_run(cap=40000)` | **PASS** (3 tests) |
| 3 | historical best == 20k -> continuation semantics unchanged | **PASS** (4 tests) |
| 4 | V2-GC resume from a legal <= 20k checkpoint cannot exceed 20k | **PASS** (7 parametrisations) |
| 5 | V2-GC resume artifact > 20k -> fail closed, never clamped | **PASS** (5 tests) |
| 6 | no CLI/environment switch can relax the hard maximum | **PASS** (4 tests) |

Two further tests tie the pure decisions to the real control flow, so a future
edit cannot leave the decisions correct while the loop stops consulting them:

* `test_train_run_resolves_the_cap_before_its_update_loop` -- AST proof that
  `resolve_run_cap` is called, is handed `provenance`/`cap`/`resume`, and appears
  **before** the `while` loop;
* `test_the_second_train_run_is_gated_on_the_candidate_budget` -- AST proof that
  `execute_stage` contains exactly two `train_run` calls and that the second is
  inside the single `if continuation_permitted(candidate, ...)`.

---

## 4. Fresh audit of the complete C3 implementation

### 4.1 Objective mathematics - unchanged and re-verified

`unmark/stage1/objective_grid.py` was not touched by this repair. Re-verified in
the current tree:

```
objective_grid.py:214   target = hidden_clean_target.detach()          # stop-gradient
objective_grid.py:216   mask = content_mask(attention_mask, special_tokens_mask)
objective_grid.py:227   ... dim=-1, eps=COSINE_EPS                      # locked epsilon
objective_grid.py:229   distance = 1.0 - similarity
objective_grid.py:235   (distance * weights).sum(dim=1) / counts        # per-example mean
objective_grid.py:335   loss_grid = distance_grid.mean()               # then batch mean
objective_grid.py:337-339
        self.weights.lambda_align * loss_align
      + self.weights.lambda_clean * loss_clean
      + self.weights.lambda_grid  * loss_grid
```

* **valid mask** -- the locked `unmark.modeling.pooling.content_mask`
  (`attention_mask == 1 AND special_tokens_mask == 0`); no second definition.
* **clean target detached** -- unconditionally, inside the primitive; callers
  cannot opt out.
* **corrupted gradient path** -- `hidden_corrupt` is never detached.
* **L_clean gradient path** -- the detach is scoped to `token_grid_distance`; the
  clean adapted branch's own graph is untouched.
* **reduction** -- per-example masked token mean, then batch mean. Not a global
  token-weighted mean.
* **lambda_grid** -- locked at 1.0; `GridConsistencyWeights` refuses any other
  value; no CLI flag and no environment variable reaches it.
* **fail closed** -- four shape gates plus a zero-valid-token gate; nothing is
  padded, trimmed or reshaped.

### 4.2 Encoder forwards - re-counted

```
HISTORICAL encoder forwards = 3
V2-GC      encoder forwards = 3
token_grid_distance encoder calls = 0
```

One reference forward and two adapted forwards through `adapted_branch`, which
returns the final hidden states and the pooled representation from a single call.
`L_grid` reuses them and runs nothing. **No forward was added for the smoke
path**: `build_candidate_objective` only constructs, and the smoke performs one
ordinary forward under `no_grad`.

### 4.3 Architecture and parameters

`unmark/modeling/` remains untouched. `unmark/stage1/protocol.py` has **zero
deleted lines** -- purely additive -- so `ADAPTER_TRAINABLE_PARAMETERS =
3_551_232`, `HIDDEN_SIZE`, the encoder pin and `LAMBDA_SCALE_SUM` are unchanged.
`objective_grid.py` and `candidates.py` create no parameters or buffers.
`verify_model_contract` still enforces 3 551 232 at runtime for every run.

```
ADAPTER_PARAMETER_DELTA     = 0     (STATICALLY_VERIFIED; RUNTIME_NOT_EXECUTED)
EXPECTED_ADAPTER_PARAMETERS = 3,551,232
```

### 4.4 Historical compatibility - re-executed

```
legacy (no objective key) -> historical env   want PASS   got PASS
legacy (no objective key) -> V2-GC env        want REFUSE got REFUSE
grid-consistency-v1       -> historical env   want REFUSE got REFUSE
grid-consistency-v1       -> V2-GC env        want PASS   got PASS
historical                -> V2-GC env        want REFUSE got REFUSE
```

`Stage1Objective.forward` still has zero diff lines against the baseline. The
freeze artifact `docs/spec/stage1-adapter-finalists.json` is unchanged by this
repair (still the Audit 065 two-leaf verifier-contract synchronization).

### 4.5 Provenance identity

`RunProvenance.objective` carries `{"objective_id": "grid-consistency-v1",
"lambda_grid": 1.0}`. Recoverable from a V2-GC artifact: repository HEAD,
objective identity, `lambda_grid`, run/init/corruption seeds, learning rate, `r`,
both historical lambdas, corpus digest, backbone pin, protocol version,
precision, inventory -- and now the **enforced budget policy**, via
`RunResult.to_dict()["budget_policy"]` and the stage artifact's `v2_gc.budget`
block. The budget is re-derivable from the checkpoint alone, because
`budget_for_objective` is keyed on the objective id the provenance records.

### 4.6 Checkpoint selection - unchanged

`unmark/stage1/validation.py` remains entirely untouched. `selection.py` is still
purely additive (the `v2_gc_schedule` function only); `ValidationPoint`, `.score`,
`select_checkpoint`, `select_learning_rate`, `select_r` and `budget_decision` are
unmodified. A grep for `loss_grid`, `token_grid_distance` and
`GridConsistencyObjective` in both modules returns **0 hits**.

The decisive structural fact is unchanged: `validation.evaluate` never calls
`objective(batch)`, so `GridConsistencyObjective.forward` is never invoked during
validation and `L_grid` is never computed on the selection path. No downstream
label, no UIT-VSFC Macro-F1, no official validation, no candidate-vs-candidate or
best-seed selection.

`GRID_METRIC_USED_FOR_SELECTION = NO.`

### 4.7 Candidate isolation for C1/C2

`candidate_for_stage`, `objective_for_stage` and `budget_for_objective` are
asserted by test to contain no reference to `GRID_CONSISTENCY` or `V2_GC` -- they
read the register. `trainer.py` is asserted to contain no `"v2_gc"` string
literal: the budget reaches it through `provenance.objective`, never a stage
name. Adding C1 (V2-SCF) or C2 (V2-GRD) is a `StageCandidate` entry plus its
objective; no call site learns a candidate's name.

---

## 5. Local test results

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** (no output) |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| `python -m pytest tests/test_stage1_v2_gc.py -q` | **51 passed, 28 skipped** |
| `python -m pytest tests/test_stage1_v2_gc_budget.py -q` | **41 passed, 1 skipped** |
| `python -m pytest tests/ -q` (full local suite) | **4545 passed, 201 skipped, 0 failed** (158s) |

Movement against the Audit 065 state (4495 passed / 198 skipped):

* **+50 passed** -- 41 new hard-cap tests, 7 new smoke-dispatch tests, 1 new
  run-independence companion test, 1 new parametrisation of the
  name-resolution sweep (it globs `unmark/stage1/*.py` and picked up
  `candidates.py`);
* **+3 skipped** -- 1 torch-gated end-to-end `train_run` refusal, 2 torch-gated
  smoke-dispatch tests.

No pre-existing test regressed. No ML package was installed.

**Torch-gated tests pending on Colab: 29.**
The original 26 from Audit 065 are intact and unmodified (verified by name);
3 were added by this repair. None were deleted, weakened or xfailed -- the only
skip marker in either file is the `torch is None` gate.

---

## 6. Remaining Colab-only work

Everything below needs a GPU/torch environment and nothing else blocks it.

1. The 26 original V2-GC numeric/gradient tests -- cosine correctness, padding and
   special-token exclusion, per-example reduction, the detach, both gradient
   directions, frozen-encoder gradient absence, exact composition, zero added
   parameters, shape/grid fail-closed, and the 3-forward counter.
2. The 3 new torch-gated tests -- `train_run` refusing an over-cap V2-GC resume,
   the shared constructor building each registered candidate's objective, and the
   V2-GC grid telemetry.
3. `stage1_runner.py smoke --candidate v2_gc` against real PhoBERT and the real
   prepared corpus -- the pre-training smoke that BLOCKER 4 made impossible and
   this repair enables.

---

FINAL AUDIT — V2-GC REPAIR

BASELINE_ANCESTRY:
c3e4cc40117ac0b3647991901971d08177117ca3

CURRENT_HEAD:
c3e4cc40117ac0b3647991901971d08177117ca3 (no commit created)

WORKTREE_DIRTY:
YES — 15 modified, 5 untracked, 0 staged, 0 commits made

SCIENTIFIC_OBJECTIVE_CHANGED:
NO — unmark/stage1/objective_grid.py untouched by this repair; Stage1Objective.forward still has zero diff lines

OBJECTIVE_FORMULA:
L_total = 1.0*L_align + 1.0*L_clean + 1.0*L_grid
d_i = 1 - cos(H_corrupt[i], stop_gradient(H_clean[i]))
valid_i = attention_mask_i == 1 AND special_tokens_mask_i == 0
L_grid(example) = sum_i(valid_i * d_i) / sum_i(valid_i); L_grid = mean over examples

ENCODER_FORWARDS:
3 historical, 3 V2-GC, 0 added by the smoke path

ADAPTER_PARAMETER_DELTA:
0 (STATICALLY_VERIFIED; RUNTIME_NOT_EXECUTED — torch absent locally)

HISTORICAL_BEHAVIOR_PRESERVED:
YES — historical objective, validation, selection and checkpoint compatibility all unchanged and re-executed

HISTORICAL_20K_TO_40K_POLICY_PRESERVED:
YES — PRECOMMITTED_CONTINUATION is unchanged for lr_pilot, r_phase1 and final_main; CASE 3 passes both before and after the repair

V2_GC_HARD_MAX_UPDATES:
20000

V2_GC_CAN_EXECUTE_UPDATE_20001:
NO — resolve_run_cap refuses any cap above 20000 before train_run's `while global_update < cap` loop is reached

V2_GC_20K_BOUNDARY_CAN_TRIGGER_40K:
NO — refused independently by resolve_budget (cap not raised), continuation_permitted (second leg not entered) and resolve_run_cap (a 40k cap is rejected)

V2_GC_RESUME_ABOVE_20K_REFUSED:
YES — refused, never clamped or normalised; a >20k payload is treated as real evidence of unauthorised work

V2_GC_REAL_MODEL_SMOKE_DISPATCH:
IMPLEMENTED — smoke_check resolves via candidates.candidate_for_stage and builds via the shared execute.build_candidate_objective; `smoke --candidate v2_gc` exercises GridConsistencyObjective with no optimizer, no backward, no selection, and asserts every loss term finite including loss_grid and mean_distance_grid

GRID_METRIC_USED_FOR_SELECTION:
NO — validation.evaluate never invokes the objective's forward, so L_grid is never computed on the selection path

LOCAL_TESTS_PASSED:
4545

LOCAL_TESTS_FAILED:
0

LOCAL_TESTS_SKIPPED:
201

TORCH_GATED_TESTS_PENDING:
29 — 26 original V2-GC tests (intact, unmodified) plus 3 added by this repair

COLAB_GPU_TESTS_REQUIRED_BEFORE_TRAINING:
- the 26 original V2-GC numeric/gradient/forward-count tests
- the 3 new torch-gated tests (over-cap resume refusal, shared constructor per candidate, grid telemetry)
- `stage1_runner.py smoke --candidate v2_gc` against real PhoBERT and the real prepared corpus

BLOCKERS_BEFORE_20K_TRAINING:
- NONE in the repository. Both Audit 065 blockers are closed and the false-assurance test is replaced by 41 execution-path tests that fail against the pre-repair code.
- REMAINING GATE (environmental, not a defect): the 29 torch-gated tests and the V2-GC real-model smoke have never executed, because torch is absent from the local ML-free venv by design and was not installed. Numeric correctness, gradient routing, the zero-parameter delta and the 3-forward count remain STATICALLY_VERIFIED only. Run them on Colab/GPU first.
- OPEN (non-blocking, unchanged from Audit 065): the adjudication protocol for comparing V2-GC against UNMARK-A/B is undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_COLAB_GATE
