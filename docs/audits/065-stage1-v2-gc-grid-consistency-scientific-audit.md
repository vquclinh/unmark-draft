# Audit 065 - Stage-1 V2-GC (Grid Consistency) Read-Only Scientific Audit

**Scope:** read-only scientific audit of the uncommitted working tree that
implements the first post-hoc UNMARK-v2 candidate, **V2-GC (Grid Consistency)**.
**Date:** 2026-09-11
**Type:** read-only audit. No implementation, test, spec or historical audit file
was modified, no commit was created, no training was run.

This audit does not implement, repair, or reopen anything. Findings are reported,
not fixed. No UIT-VSFC data, no official validation and no official TEST was
accessed.

**Path note:** the audit request named `/content/unmark-draft`; the audited tree
is the real working copy at `/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft`.

---

## 0. Executive verdict

**PASS_WITH_BLOCKERS.**

The objective mathematics, encoder-forward accounting, architecture/parameter
isolation, historical compatibility, provenance identity and checkpoint-selection
semantics are all clean and verified.

One material scientific defect was found: **V2-GC is not capped at 20,000
updates.** The stage-agnostic budget-continuation path automatically promotes any
run to 40,000 updates when its best held-out checkpoint lands exactly on the cap,
and no `v2_gc` guard exists anywhere. The constant that reads like a cap
(`V2_GC_MAX_UPDATES`) is decorative, and the test that appears to cover the cap is
vacuous.

---

## A. Working-tree isolation - **PASS**

```
HEAD    : c3e4cc40117ac0b3647991901971d08177117ca3   (== baseline; no commits made)
BRANCH  : main
STAGED  : (empty)
```

**Modified (12)**

```
docs/spec/stage1-adapter-finalists.json
scripts/stage1_runner.py
scripts/stage1_wandb_monitor.py
tests/test_stage1_provenance_contract.py
unmark/stage1/contracts.py
unmark/stage1/execute.py
unmark/stage1/finalists.py
unmark/stage1/fused.py
unmark/stage1/objective.py
unmark/stage1/protocol.py
unmark/stage1/selection.py
unmark/stage1/trainer.py
```

**Untracked (2)**

```
tests/test_stage1_v2_gc.py
unmark/stage1/objective_grid.py
```

**Diffstat:** 501 insertions, 17 deletions across 12 files.

Every one of the 17 deleted lines was inspected individually. All 17 are
accounted for by a necessary edit:

| Deletion | Reason |
| --- | --- |
| freeze-artifact `note` | verifier field-contract synchronization |
| monitor loss-key loop | extended with the two grid series |
| provenance test nested-identity branch | extended to cover `objective` |
| `resume=bool(resume),` | gained the objective-identity kwargs |
| `objective = Stage1Objective(...)` | moved verbatim into an `else:` branch |
| `adapted_representation` signature/docstring (5 lines) | refactored into `adapted_branch` + delegating accessor |
| six trainer telemetry lines | replaced by `loss_telemetry(...)` |

**No unrelated change, no reformatting, no drive-by cleanup.**
`unmark/modeling/` is entirely untouched.

---

## B. Objective mathematics

| # | Item | Verdict | Location | Note |
| --- | --- | --- | --- | --- |
| 1 | `L = L_align + L_clean + L_grid` | **PASS** | `unmark/stage1/objective_grid.py:336-340` | Three weighted terms, no fourth |
| 2 | `lambda_align = lambda_clean = lambda_grid = 1.0` | **PASS** | runtime-checked -> `{1.0, 1.0, 1.0}` | `lambda_a`/`lambda_c` from `lambdas_for_r(1.0)`; `lambda_g` from `LAMBDA_GRID` |
| 3 | `lambda_grid` not overridable | **PASS** | `unmark/stage1/contracts.py:596-614` | Frozen dataclass refuses any value != 1.0; `dataclasses.replace` refused; **no CLI flag, no env var** |
| 4 | FINAL contextual hidden states `[B,L,768]` | **PASS** | `unmark/stage1/objective_grid.py:295-312` | `adapted_branch` -> `_hidden_states(outputs)` = encoder `last_hidden_state`. Not `z`, not raw word embeddings, not pooled, not FIRST_TOKEN |
| 5 | `attention_mask & ~special_tokens_mask` | **PASS** | `unmark/stage1/objective_grid.py:216` | Reuses locked `unmark.modeling.pooling.content_mask`; no second definition of a content token |
| 6 | per-example token mean, then batch mean | **PASS** | `objective_grid.py:234-235`, then `:335` | `(d*m).sum(dim=1)/counts` -> `[B]`, then `.mean()`. **Not** a global token-weighted batch mean |
| 7 | clean target unconditionally detached | **PASS** | `unmark/stage1/objective_grid.py:214` | `target = hidden_clean_target.detach()` inside the primitive; callers cannot opt out |
| 8 | corrupted branch receives L_grid gradient | **PASS (static)** | `objective_grid.py:226-235` | `hidden_corrupt` is never detached; the runtime test exists but is **torch-gated / not executed** |
| 9 | L_clean still sends its historical gradient | **PASS** | `unmark/stage1/objective.py:143-186` | The detach is scoped to `token_grid_distance`; the clean adapted branch's own graph is untouched |
| 10 | fail closed on incompatible shapes / grid semantics | **PASS** | `objective_grid.py:131-169` | Four shape gates plus a zero-valid-token gate; nothing padded, trimmed or reshaped |

**Exact implemented formula**

```text
m_i  = attention_mask_i == 1  AND  special_tokens_mask_i == 0
d_i  = 1 - cos( H_corrupt[i], stop_gradient(H_clean[i]) )      # dim=-1, eps=1e-8

L_grid(example) = sum_i(m_i * d_i) / sum_i(m_i)
L_grid          = mean over examples

L_total = 1.0*L_align + 1.0*L_clean + 1.0*L_grid
```

**Cosine epsilon:** identical call and identical constant -
`torch.nn.functional.cosine_similarity(..., dim=-1, eps=COSINE_EPS)` with
`COSINE_EPS = 1e-8` imported from `unmark/stage1/objective.py:33`. Byte-identical
semantics to the existing Stage-1 `representation_distance`. **PASS.**

The expected values used in the numeric tests were independently re-derived
against a reference implementation written from the specification (not from the
implementation), confirming `0.14644661` for the hand-computed case, exclusion of
padding and special tokens, and per-example `[0.0, 2.0]` -> batch `1.0` where a
flat token mean would have given `1.5`.

---

## C. Forward-pass accounting - **PASS (static)**

```
HISTORICAL_FORWARD_COUNT = 3   (1 x reference_representation + 2 x adapted_representation)
V2_GC_FORWARD_COUNT      = 3   (1 x reference_representation + 2 x adapted_branch)
EXTRA_FORWARD_INTRODUCED = NO
```

Static call-graph evidence:

* `adapted_branch` contains exactly one `unmark_encoder(...)` call;
* `adapted_representation` now delegates to it and issues **zero** encoder calls
  of its own;
* `token_grid_distance` issues **zero** encoder / branch calls;
* no helper re-enters the encoder.

The three branches are: native clean Vanilla reference, adapted clean UNMARK,
adapted corrupted UNMARK. `L_grid` reuses the adapted-clean and adapted-corrupt
hidden states produced by those same two forwards.

The dynamic counter (`test_runtime_v2_gc_runs_exactly_three_encoder_forwards`,
which patches the frozen encoder itself and compares against the historical
objective) **exists but did not execute** - it is torch-gated.

---

## D. Architecture / parameter isolation - **STATICALLY_VERIFIED** (RUNTIME_NOT_EXECUTED)

`unmark/modeling/` is entirely untouched. `unmark/stage1/objective_grid.py`
contains **zero** parameter-creating constructs (`nn.Parameter`, `nn.Embedding`,
`nn.Linear`, `nn.LayerNorm`, `register_parameter`, `register_buffer`,
`requires_grad_`) and references no adapter/gate/tone/letter/fusion/tokenizer/
position-id internals in code - the only two textual hits are docstring prose.

`unmark/stage1/protocol.py` is **purely additive**: `ADAPTER_TRAINABLE_PARAMETERS`,
`HIDDEN_SIZE`, `ENCODER_CHECKPOINT`, `ENCODER_REVISION` and `LAMBDA_SCALE_SUM` are
unchanged.

Unchanged by V2-GC: `OrthographyInputAdapter` dimensions, gate dimensions, tone
embeddings, letter embeddings, fusion dimensions, PhoBERT, tokenizer, position-id
semantics, Stage-2 inference architecture.

```
ADAPTER_PARAMETER_DELTA       = 0
EXPECTED_ADAPTER_PARAMETERS   = 3,551,232   (unchanged)
```

This count is **enforced in production at runtime** by `verify_model_contract`
(`unmark/stage1/trainer.py:253`), which `train_run` calls for every run including
V2-GC - so it is not merely a test assertion.

**Explicit limitation:** torch, numpy and transformers are all absent from the
local venv, by the repository's own ML-free design. They were **not** installed.
No runtime parameter count was executed in this audit.

---

## E. Historical compatibility - **PASS**

**E1 - historical objective unchanged.** `Stage1Objective.forward` has **zero diff
lines**. Its body still reads
`loss = self.weights.lambda_align * loss_align + self.weights.lambda_clean * loss_clean`
over one reference branch and two adapted branches. The only change to
`objective.py` is that `adapted_representation` now returns `adapted_branch(...)[1]`
- same encoder call, same locked pooling, same operation order, identical value.

**E2-E4 - executed cross-compatibility matrix:**

| Payload | Environment | Required | Observed |
| --- | --- | --- | --- |
| legacy (no `objective` key) | historical | PASS | **PASS** |
| legacy (no `objective` key) | V2-GC | REFUSE | **REFUSE** |
| `grid-consistency-v1` | historical | REFUSE | **REFUSE** |
| `grid-consistency-v1` | V2-GC | PASS | **PASS** |
| historical | V2-GC | REFUSE | **REFUSE** |

A missing `objective` block is read as, and only as, `align-clean-pooled-v1`
(`unmark/stage1/trainer.py:173`). A V2-GC checkpoint therefore cannot masquerade
as UNMARK-A/B, and cannot resume from an UNMARK-A/B checkpoint.

**E5 - finalist verification still works.** `VERIFIED_PROVENANCE_FIELDS` gained
`"objective"` and the two-direction coverage test against `require_match` passes.
All finalist freeze tests pass. Note that `verify_finalist_checkpoint` against the
real `.pt` files cannot run here - they are git-ignored external Drive artifacts.

**E6 - the freeze-artifact edit.** A structural leaf-by-leaf comparison against the
baseline commit found **exactly 2 differing leaves**, both under
`/evidence/provenance_fields_verified/`:

```diff
-      "note": "exactly what RunProvenance.require_match compares; the finalist verifier reuses verify_checkpoint so it cannot drift weaker",
+      "note": "exactly what RunProvenance.require_match compares; the finalist verifier reuses verify_checkpoint so it cannot drift weaker. `objective` was added with the V2-GC candidate: a provenance predating the field is read as the historical objective, so both finalists verify unchanged and no digest, score or identity was touched.",
         "repository_head",
         "inventory",
+        "objective",
         "lambda_align",
         "lambda_clean"
```

**Unchanged:** both finalists' `checkpoint_sha256`
(`6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91`,
`9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2`), `run_seed`
(36930, 21230), `update` (3500, 14500), `learning_rate`, `r`, `validation_score`,
`robust_score`, `source_repository_head`, `source_stage`, `freeze_complete`,
`pending_finalist_digests`.

**This edit is purely a verifier field-contract synchronization**, mechanically
required because `finalists.py` refuses any artifact whose declared field list
differs from `VERIFIED_PROVENANCE_FIELDS`. No finalist identity, checkpoint hash,
selected seed, selected update, score or scientific result was altered.

---

## F. Provenance identity - **PASS**

`RunProvenance.objective: ObjectiveIdentity` (default `HISTORICAL_OBJECTIVE`)
serialises into every checkpoint and run artifact as:

```json
{"objective_id": "grid-consistency-v1", "lambda_grid": 1.0}
```

Recoverable from a V2-GC artifact: `repository_head`, objective identity,
`lambda_grid`, `run_seed`, `init_seed`, `corruption_seed`, `learning_rate`, `r`,
`lambda_align`, `lambda_clean`, `corpus_manifest_digest`, backbone checkpoint and
revision, `protocol_version`, `precision`, `inventory`. The stage artifact also
carries a top-level `objective` block, so the objective is recoverable without
opening a checkpoint.

Resume compatibility is fail-closed in both directions (matrix in section E).

Locked V2-GC run values, all inherited from the already-closed campaign and none
retuned: `learning_rate = 1e-4`, `r = 1.0`, `run_seed = 36930`,
`init_seed = 51800` (derived via `adapter_init_seed`), `corruption_seed = 35422`,
batch size 128, evaluation cadence 500, fp32/CUDA, pinned PhoBERT revision.

---

## G. Checkpoint selection / validation semantics - **PASS (strong)**

`unmark/stage1/validation.py` is **entirely untouched**. `unmark/stage1/selection.py`
is **purely additive** - `ValidationPoint`, `.score`, `select_checkpoint`,
`select_learning_rate`, `select_r` and `budget_decision` are unmodified.

The decisive structural fact: **`validation.evaluate` never calls
`objective(batch)`.** It calls only `reference_representation` and
`adapted_representation`. Therefore `GridConsistencyObjective.forward` is never
invoked during validation and **`L_grid` is never even computed** on the selection
path. It cannot influence checkpoint selection by construction rather than by
policy.

Selection implementation reused by V2-GC, unchanged:

```
validation.evaluate
  -> ValidationPoint.score   (max over FULL/P50/P100/STRIP_ALL of mean cosine distance to h(x))
  -> selection.select_checkpoint   (lowest score, then lower d_clean, then earliest update)
```

No downstream sentiment labels, no UIT-VSFC Macro-F1, no official validation, no
`L_grid`-driven winner selection, no candidate-vs-candidate selection, no
best-seed selection. V2-GC is one run and its artifact leg performs no selection.
`loss_grid` and `mean_distance_grid` appear only in `sink.emit` telemetry and the
W&B bridge.

---

## H. 20K screening budget - **FAIL**

```
V2_GC_FIRST_SCREEN_HARD_CAP_20000 = FAIL
```

**There is no hard cap. V2-GC can automatically train to 40,000 updates.**

The exact code path that can exceed 20,000:

1. `unmark/stage1/execute.py:473` - `leg_cap = INITIAL_MAX_UPDATES` (20,000).
2. `train_run(..., cap=20000)` finishes and calls `resolve_budget`
   (`unmark/stage1/trainer.py:714-727`).
3. `budget_decision` (`unmark/stage1/selection.py:185-217`): **if the best held-out
   checkpoint is exactly update 20,000**, it returns `continue_run=True`, so
   `result.cap = EXTENDED_MAX_UPDATES` (40,000).
4. `unmark/stage1/execute.py:511-542` -
   `if leg_cap == INITIAL_MAX_UPDATES and result.cap == EXTENDED_MAX_UPDATES:`
   fires, loads the checkpoint written at 20,000, and calls `train_run` again with
   `cap=40000`.

This block is **stage-agnostic**. A search of the entire tree found no `v2_gc`
specific cap, budget, or continuation guard anywhere.

Two aggravating details:

* **`V2_GC_MAX_UPDATES` is decorative.** Its only uses are an import, a `print()`
  at `scripts/stage1_runner.py:524`, and one test assertion. It is **never
  consulted by the execution path** - the real cap comes from
  `INITIAL_MAX_UPDATES`. A constant that reads like a cap and enforces nothing is
  worse than no constant.
* The runner prints `budget : 20000 updates, the locked precommitted rule`, which
  is misleading: the locked precommitted rule *includes* the 20k -> 40k
  continuation.

**The risk is not theoretical.** The historical finalists were selected at updates
3500 and 14500, well inside budget, so the continuation never fired historically.
But V2-GC adds a term that directly minimises corrupt-vs-clean drift - the same
quantity the selection score measures - which materially raises the chance that
the best point lands at exactly 20,000 and triggers promotion.

Additionally: after such a promotion, a later `--resume` reads
`resume_cap(carried) == 40000` and continues on the 40k leg.

Not repaired in this audit, per instruction.

---

## I. Test audit

`tests/test_stage1_v2_gc.py`: **70 tests - 44 executed and PASSED, 26 SKIPPED.**
All 26 skips are `requires_torch`, reason
*"torch is not installed (ML-free local .venv); runs on Colab"*. 0 failed, 0
not-applicable. This matches the previously reported 26 exactly.

| # | Required coverage | Status |
| --- | --- | --- |
| 1 | numeric token-grid cosine correctness | **SKIPPED** (2 tests); expectations independently confirmed against a spec-derived reference |
| 2 | padding exclusion | **SKIPPED** |
| 3 | special-token exclusion | **SKIPPED** |
| 4 | per-example reduction then batch reduction | **SKIPPED** |
| 5 | clean target detach | **SKIPPED** (runtime) + **PASSED** (static) |
| 6 | clean L_clean gradient still exists | **SKIPPED** |
| 7 | corrupt L_grid gradient exists | **SKIPPED** |
| 8 | frozen encoder gets no parameter gradients | **SKIPPED** |
| 9 | `lambda_grid` exact composition | **SKIPPED** (runtime) + **PASSED** (contract) |
| 10 | zero added parameters | **SKIPPED** (runtime) + **PASSED** (static) |
| 11 | shape mismatch fail closed | **SKIPPED** |
| 12 | grid mismatch fail closed | **SKIPPED** |
| 13 | historical objective unchanged | **PASSED** (4 tests) |
| 14 | historical checkpoint compatibility | **PASSED** |
| 15 | old checkpoint cannot resume as V2-GC | **PASSED** |
| 16 | exactly 3 encoder forwards | **SKIPPED** (runtime) + **PASSED** (static, 2 tests) |

**MISSING COVERAGE - and it is the one that matters:** there is **no test asserting
the 20,000-update hard cap for V2-GC**, and no test that the 40k continuation is
blocked for this stage. Worse,
`test_the_v2_gc_plan_is_exactly_the_locked_values` asserts
`V2_GC_MAX_UPDATES == INITIAL_MAX_UPDATES == 20_000` - a **vacuous** assertion
about a constant the execution path never reads. It creates the appearance of
budget coverage where none exists.

Torch was not installed and must not be. All 26 skipped tests remain to execute on
Colab/GPU.

---

## J. Static / regression checks

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** (no output) |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| `python -m pytest tests/ -q` | **4495 passed, 198 skipped, 0 failed** (156s) |

Baseline before the V2-GC work was 4451 passed / 171 skipped. Delta: **+44 passed**
(the new torch-free V2-GC tests) and **+27 skipped** (26 torch-gated V2-GC tests
plus one `tests/test_stage1_name_resolution.py:80` module-import skip for the new
torch-importing module, confirmed by name). No new failures; no pre-existing test
regressed.

No package was installed and nothing was downloaded.

---

## K. Additional findings (lower severity, not repaired)

* **F-3 (medium).** `smoke_check` builds `Stage1Objective` only
  (`unmark/stage1/execute.py:692-694`, via `build_objective`). There is **no
  real-model, no-update smoke path for V2-GC**, so the repository's own pre-train
  discipline cannot exercise the grid objective on real PhoBERT before a 20k run.
* **F-4 (informational).** `GridConsistencyLossResult.to_dict()` writes
  `lambda_grid` twice - once via `weights.to_dict()`, once via
  `objective.to_dict()`. Both derive from the same locked constant and cannot
  disagree; harmless but redundant.
* **F-5 (informational).** Masked positions are not NaN-isolated
  (`0 * NaN = NaN`). This is identical to the behaviour of the locked
  `masked_mean_non_special`, and `_require_finite` fails closed. Repository
  consistent, not a new defect.
* **OPEN (non-blocking).** The adjudication protocol for comparing V2-GC against
  UNMARK-A/B is still undefined, and `finalists.py` still pins exactly two
  finalists.

---

FINAL AUDIT — V2-GC

BASELINE_HEAD:
c3e4cc40117ac0b3647991901971d08177117ca3

CURRENT_HEAD:
c3e4cc40117ac0b3647991901971d08177117ca3

BRANCH:
main

WORKTREE_DIRTY:
YES — 12 modified, 2 untracked, 0 staged, 0 commits made

CHANGED_FILES:
docs/spec/stage1-adapter-finalists.json
scripts/stage1_runner.py
scripts/stage1_wandb_monitor.py
tests/test_stage1_provenance_contract.py
unmark/stage1/contracts.py
unmark/stage1/execute.py
unmark/stage1/finalists.py
unmark/stage1/fused.py
unmark/stage1/objective.py
unmark/stage1/protocol.py
unmark/stage1/selection.py
unmark/stage1/trainer.py
tests/test_stage1_v2_gc.py (untracked, new)
unmark/stage1/objective_grid.py (untracked, new)

OBJECTIVE_FORMULA:
L_total = 1.0*L_align + 1.0*L_clean + 1.0*L_grid

GRID_TARGET:
H_clean (adapted-clean final contextual hidden states) — detached unconditionally at objective_grid.py:214 inside token_grid_distance; callers cannot opt out

VALID_MASK:
attention_mask == 1 AND special_tokens_mask == 0, via the locked unmark.modeling.pooling.content_mask

REDUCTION:
per-example masked token mean: sum_i(m_i * d_i) / sum_i(m_i) -> [B]; then batch mean distance_grid.mean(). Not a global token-weighted batch mean.

ENCODER_FORWARDS_HISTORICAL:
3

ENCODER_FORWARDS_V2_GC:
3

EXTRA_ENCODER_FORWARD:
NO (statically verified; dynamic counter written but torch-gated / not executed)

ADAPTER_PARAMETER_DELTA:
0 (STATICALLY_VERIFIED; RUNTIME_NOT_EXECUTED)

EXPECTED_ADAPTER_PARAMETERS:
3,551,232 (unchanged; enforced in production by verify_model_contract at trainer.py:253)

HISTORICAL_OBJECTIVE_UNCHANGED:
YES — Stage1Objective.forward has zero diff lines

HISTORICAL_CHECKPOINT_COMPATIBILITY:
YES — legacy provenance lacking the objective field verifies as align-clean-pooled-v1 (executed)

V2_GC_IDENTITY:
grid-consistency-v1, lambda_grid = 1.0

OLD_TO_V2_RESUME_REFUSED:
YES — refused in both directions (executed, 5/5 matrix cases correct)

CHECKPOINT_SELECTION_CHANGED:
NO — validation.py untouched; selection.py purely additive

GRID_METRIC_USED_FOR_SELECTION:
NO — validation.evaluate never invokes the objective's forward, so L_grid is never computed on the selection path

V2_GC_FIRST_SCREEN_HARD_CAP_20000:
FAIL

TESTS_EXECUTED:
4693 collected (4495 executed + 198 skipped); V2-GC file: 70 collected, 44 executed

TESTS_PASSED:
4495 (V2-GC file: 44)

TESTS_FAILED:
0

TESTS_SKIPPED:
198 (V2-GC file: 26)

TORCH_GATED_TESTS_PENDING:
26 — all in tests/test_stage1_v2_gc.py; cover numeric cosine correctness, padding/special exclusion, per-example reduction, detach, both gradient directions, frozen-encoder gradients, exact composition, zero added parameters, shape/grid fail-closed, and the 3-forward counter

GIT_DIFF_CHECK:
PASS (clean, no output)

BLOCKERS_BEFORE_COLAB_GPU_TEST:
- None. The tree is internally consistent, compiles, and the full non-ML suite is green.

BLOCKERS_BEFORE_20K_TRAINING:
- BLOCKER 1 (H): No 20,000-update hard cap for V2-GC. execute.py:511-542 automatically promotes to EXTENDED_MAX_UPDATES = 40,000 whenever the best held-out checkpoint lands at exactly update 20,000. The block is stage-agnostic and there is no v2_gc guard anywhere. V2_GC_MAX_UPDATES is decorative — used only in a print statement and one test, never by the execution path. Must be resolved before the screening run.
- BLOCKER 2 (I): No test covers the cap, and test_the_v2_gc_plan_is_exactly_the_locked_values asserts a constant the execution path never reads, creating false assurance of budget coverage.
- BLOCKER 3: The 26 torch-gated tests have never executed. Numeric correctness, both gradient directions, frozen-encoder gradient absence, zero-parameter delta and the 3-forward count are currently STATICALLY_VERIFIED only. Run them on Colab/GPU first.
- BLOCKER 4 (F-3): No real-model smoke path exists for V2-GC — smoke_check builds Stage1Objective only, so the grid objective cannot be exercised on real PhoBERT before training.
- OPEN (non-blocking): the adjudication protocol for comparing V2-GC against UNMARK-A/B is still undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_BLOCKERS
