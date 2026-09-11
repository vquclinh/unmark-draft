# Audit 070 - PREFLIGHT 5A: Provenance / Logging Text Repair

**Scope:** two misleading textual descriptions exposed by PREFLIGHT 5A during the
final UNMARK-v2 pre-training audit. **Display and persisted-prose only.**
**Date:** 2026-09-11
**Baseline HEAD before this repair:** `9b735958a53401607130c956d07fbf6882b60855`
**Type:** provenance/logging repair plus regression tests. No commit was created,
no model was trained, no UIT-VSFC data, official validation or official TEST was
accessed, and no ML package was installed.

Audits 065-069 are byte-unchanged (`61672fb15d1a22342848cf4e5d18b23d`,
`35076e34f9ffd8360da4fb53c15cbc00`, `e8038e6b12d442226f3904bdab3bf13c`,
`d5eeeafc0b6d5b946e76f3ec90c6fb3d`, `6046c25ae81e7b974cdaeb31ce89726f`).

---

## 0. Executive verdict

**PASS_WITH_COLAB_RERUN — provenance/logging repair only.**

Both defects were **descriptions that were true when written and became false
when a later candidate joined the path they sit on**. Neither was a behaviour
defect: enforcement, dispatch, budgets, selection and routing were all correct
throughout, and none of them was touched.

Two files changed, **57 insertions / 8 deletions**, all of it one string literal,
comments, and a pair of pure display helpers. A new 36-test regression file
pins both invariants; 22 of those 36 fail against the pre-repair tree.

---

## 1. Defect 1 - the first-screen artifact note was C3-specific

### What was wrong

`unmark/stage1/execute.py`, in the branch `elif stage in FIRST_SCREEN_STAGES:`,
persisted this note into the stage artifact:

```
"post-hoc research candidate: the historical objective plus a
 token-grid consistency term, ONE run, no selection performed here"
```

That branch is shared by **all three** first-screen candidates:

```python
FIRST_SCREEN_STAGES = ("v2_gc", "v2_scf", "v2_grd")
```

"a token-grid consistency term" is C3's loss. C1 (`v2_scf`) keeps the historical
objective and changes the *fusion*; C2 (`v2_grd`) relates FIRST_TOKEN geometry to
a clean native teacher. The note was therefore **false for two of the three
candidates that persist it**, and it was written into a durable artifact.

The adjacent explanatory comment had the same fault ("V2-GC is one run", "The
grid term is a training term").

### The repair

The note now describes **what the stage does**, and leaves **what the candidate
is** to the authoritative machine-readable field:

```
"post-hoc research candidate: ONE fixed first-screen run; checkpoint
 selection within the run uses the locked Stage-1 held-out rule; no
 between-candidate selection is performed here"
```

True for C1, C2 and C3 alike. The comment was made candidate-neutral in the same
way. Nothing else in the branch changed: `artifact["objective"]`,
`artifact["budget"]`, `selected`, `label`, `learning_rate`, `r` and
`budget_limited` are byte-identical, and a test asserts all three of the
machine-readable blocks are still written.

`FIRST_SCREEN_STAGES` membership, candidate dispatch and every training path are
untouched.

---

## 2. Defect 2 - the runner banner advertised a continuation V2 cannot take

### What was wrong

`scripts/stage1_runner.py`'s generic startup banner printed, unconditionally:

```
  budget         : 20000 updates, one continuation, then STOP
```

for **every** command, including `v2-scf`, `v2-grd` and `v2-gc`. All three are
governed by `FIRST_SCREEN_HARD_CAP`:

```
hard_max_updates                  = 20000
allows_precommitted_continuation  = False
```

so the banner promised an operator a 20k -> 40k continuation that the run is
structurally forbidden from taking. Each candidate handler already printed a
correct, contradicting line a few rows later.

### The repair

Two pure display helpers, both deriving from the candidate register rather than
restating a constant:

```python
def banner_stage(args) -> str | None:
    """The registered stage this command will execute, if any. DISPLAY ONLY."""

def budget_banner(stage: str | None) -> str:
    """The budget line for the startup banner. DISPLAY ONLY."""
```

and the single print site becomes `print(budget_banner(banner_stage(args)))`.

Observed end-to-end through the real CLI:

| command | banner line |
| --- | --- |
| `v2-scf` / `v2-grd` / `v2-gc` | `20000 updates, HARD CAP (first_screen_hard_cap); NO continuation to 40000` |
| `lr-pilot` / `r-phase1` / `final-main` | `20000 updates, one continuation, then STOP` *(unchanged)* |
| `prepare-corpus` | `20000 updates, one continuation, then STOP` *(unchanged)* |
| `smoke --candidate v2_grd` | `... HARD CAP ...; NO continuation to 40000` |
| `smoke` (default `lr_pilot`) | `20000 updates, one continuation, then STOP` *(unchanged)* |

A command that executes no registered stage (`prepare-corpus`) keeps the exact
historical line, so no historical message changed anywhere.

---

## 3. Tests

New file: `tests/test_stage1_v2_provenance_text.py` — **36 tests, torch-free**.

| Group | What it pins |
| --- | --- |
| First-screen note | the branch really is shared by all three candidates; the persisted note (read from the AST, so a comment cannot satisfy it) contains **no** candidate-specific vocabulary — `token-grid`, `grid consistency`, `l_grid`, `scale-calibrated`, `calibrat`, `relational`, `gram`, `first_token`, `distillation`, `l_grd`; it states the three facts it must; `objective` / `budget` / `selected` are still written |
| Banner | no hard-capped candidate is offered a continuation; historical stages keep their exact line verbatim; a stage-less command keeps the generic line; the line is derived from the register for every candidate; `main` holds no hard-coded continuation sentence; command→stage mapping for all six commands; `smoke --candidate` resolution; `prepare-corpus` resolves no stage |
| Real CLI | `main()` is driven for all six training commands and its printed banner is checked; each fails closed on a missing corpus **before** any model, corpus or optimizer is touched |
| Facts | C1/C2/C3 `hard_max_updates == 20000` and `allows_precommitted_continuation is False`; historical budgets unchanged; the three W&B projects exact; every candidate's `(objective, fusion)` identity unchanged |

**Mutation check.** The new file was run against a pristine checkout of the
pre-repair HEAD: **22 failed, 14 passed**, and the two headline failures name the
defects exactly:

```
AssertionError: the shared first-screen note claims 'token-grid', which is true
of only one candidate.
  'token-grid' is contained here: ...ve plus a token-grid consistency term...

AssertionError: v2-scf printed a continuation promise:
  ['  budget         : 20000 updates, one continuation, then STOP']
```

The 14 that pass on both trees are the pure non-regression assertions (registry,
budgets, W&B projects, identities) — correctly unaffected by the repair.

No existing test was weakened, deleted, xfailed or skipped. No existing test
asserted either text, which is precisely why the defects survived to PREFLIGHT.

### Results

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| new file alone | **36 passed** |
| targeted (new file + C1/C2/C3 suites + runner + artifact-identity contracts) | **377 passed, 69 skipped** |
| full local suite | **4732 passed, 242 skipped, 0 failed** (161 s) |

The 242 skips are the torch-gated candidate tests: torch is absent from the local
ML-free venv by design and was **not** installed.

---

## 4. Scientific non-regression

Only three paths changed, and the complete non-test diff is one import, two
display helpers, one print statement, one string literal and three comment
blocks. Every scientific module is **unchanged on disk**:

```
unmark/modeling/adapter.py              unchanged
unmark/stage1/objective.py              unchanged
unmark/stage1/objective_grid.py         unchanged
unmark/stage1/objective_relational.py   unchanged
unmark/stage1/trainer.py                unchanged
unmark/stage1/candidates.py             unchanged
unmark/stage1/selection.py              unchanged
unmark/stage1/validation.py             unchanged
unmark/stage1/protocol.py               unchanged
unmark/stage1/contracts.py              unchanged
unmark/stage1/initialisation.py         unchanged
unmark/stage1/data.py                   unchanged
unmark/stage1/optim.py                  unchanged
unmark/stage1/reconstruct.py            unchanged
```

| Item | Status | Evidence |
| --- | --- | --- |
| C1 SCF equation | **unchanged** | `adapter.py:132-133` `scale = e_norm / f_norm.clamp(min=FUSION_SCALE_EPSILON)`; `return scale * fused` |
| C2 GRD equation | **unchanged** | `objective_relational.py:202, 262, 266, 363` row-normalised Gram, `teacher_target.detach()`, `(difference[mask] ** 2).mean()`, 0.5 weighting |
| C3 GC equation | **unchanged** | `objective_grid.py:214, 235, 339` `hidden_clean_target.detach()`, `sum(dim=1)/counts`, `lambda_grid * loss_grid` |
| Objective dispatch | **unchanged** | `build_candidate_objective` not in the diff |
| Stage-1 corruption | **unchanged** | `CORRUPTION_SEED = 35422`, `pi_strip` untouched |
| Schedules / seeds / LR / r | **unchanged** | run 36930, init 51800, corruption 35422, LR 1e-4, r 1.0, lambdas (1.0, 1.0) |
| Initialization | **unchanged** | `initialisation.py` not in the diff |
| Optimizer | **unchanged** | `optim.py` not in the diff |
| Evaluation cadence | **unchanged** | `EVAL_EVERY_UPDATES = 500`, batch 128, max_length 256, fp32 |
| Checkpoint selection | **unchanged** | `selection.py` and `validation.py` not in the diff |
| 20k hard-cap enforcement | **unchanged** | re-executed: C1/C2/C3 refuse cap 40000 and accept 20000; historical accepts 40000 |
| Resume state | **unchanged** | `trainer.py` not in the diff |
| W&B project routing | **unchanged** | C1 `UNMARK-v2-C1-SCF-Stage1`, C2 `UNMARK-v2-C2-GRD-Stage1`, C3 `UNMARK-v2-C3-GC-Stage1`, historical `unmark-stage1` |
| Model parameters / forward count | **unchanged** | `ADAPTER_TRAINABLE_PARAMETERS = 3551232`; no objective or adapter module touched |

---

FINAL AUDIT — PREFLIGHT 5A PROVENANCE TEXT REPAIR

BASELINE_HEAD:
9b735958a53401607130c956d07fbf6882b60855

DEFECT_1:
The first-screen artifact note hard-coded C3's "token-grid consistency term" on a branch shared by v2_scf, v2_grd and v2_gc, making the persisted prose false for C1 and C2.

DEFECT_1_BEHAVIOUR_BUG:
NO — provenance prose only; `artifact["objective"]` already recorded each candidate's authoritative identity correctly

DEFECT_1_REPAIRED:
YES — the note is now candidate-neutral: "post-hoc research candidate: ONE fixed first-screen run; checkpoint selection within the run uses the locked Stage-1 held-out rule; no between-candidate selection is performed here". `objective`, `budget`, `selected`, `label`, `learning_rate`, `r` and `budget_limited` are untouched.

DEFECT_2:
The generic runner banner printed "20000 updates, one continuation, then STOP" for every command, including the three V2 candidates whose budgets set allows_precommitted_continuation = False.

DEFECT_2_BEHAVIOUR_BUG:
NO — display text only; resolve_run_cap, resolve_budget and continuation_permitted all already refused the 40k leg for a hard-capped candidate

DEFECT_2_REPAIRED:
YES — `budget_banner(banner_stage(args))` derives the line from the candidate register. V2 commands print "20000 updates, HARD CAP (first_screen_hard_cap); NO continuation to 40000"; historical commands and prepare-corpus print the exact historical line.

SCIENTIFIC_BEHAVIOUR_CHANGED:
NO

C1_EQUATION_CHANGED:
NO

C2_EQUATION_CHANGED:
NO

C3_EQUATION_CHANGED:
NO

OBJECTIVE_DISPATCH_CHANGED:
NO

CORRUPTION_SCHEDULES_SEEDS_LR_R_CHANGED:
NO

INITIALIZATION_OPTIMIZER_CADENCE_CHANGED:
NO

CHECKPOINT_SELECTION_CHANGED:
NO

HARD_CAP_ENFORCEMENT_CHANGED:
NO — re-executed: C1/C2/C3 refuse a 40000 cap, historical still accepts it

RESUME_SEMANTICS_CHANGED:
NO

WANDB_ROUTING_CHANGED:
NO

MODEL_PARAMETERS_OR_FORWARD_COUNT_CHANGED:
NO

FILES_CHANGED:
unmark/stage1/execute.py (21 lines: one note literal + comments)
scripts/stage1_runner.py (44 lines: one import, two display helpers, one print site)
tests/test_stage1_v2_provenance_text.py (new, 36 tests)

LOCAL_TESTS_PASSED:
4732

LOCAL_TESTS_FAILED:
0

LOCAL_TESTS_SKIPPED:
242

MUTATION_CHECK:
22 of 36 new tests fail against the pre-repair tree, naming both defects verbatim; the 14 that pass on both are the pure non-regression assertions

COLAB_RERUN_REQUIRED:
YES — the 69 torch-gated candidate tests and the three real-model smokes still have never executed locally (torch absent by design, not installed). A final Colab GPU rerun at the NEW committed HEAD is required before Stage-1 training.

BLOCKERS_BEFORE_STAGE1_TRAINING:
- NONE new. This repair changed no scientific behaviour.
- REMAINING GATE (unchanged): the 69 torch-gated candidate tests must pass on Colab/GPU at the new HEAD, and the three real-model smokes (`--candidate v2_scf`, `v2_grd`, `v2_gc`) must run against real PhoBERT and the real prepared corpus.
- OPEN (non-blocking, unchanged): the adjudication protocol for comparing C1, C2, C3 and UNMARK-A/B is undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_COLAB_RERUN — provenance/logging repair only
