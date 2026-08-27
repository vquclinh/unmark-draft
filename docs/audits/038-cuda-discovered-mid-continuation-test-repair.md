# Audit 038 — CUDA-Discovered Mid-Continuation Test Repair

**Scope:** ONE test-design defect found by the authoritative CUDA acceptance.
`tests/test_stage1_resume_state_machine_torch.py` only.
**Date:** 2026-08-26
**Mode:** TEST REPAIR. Production byte-identical. No prior audit modified.

---

## 1. Starting HEAD

```
$ git rev-parse HEAD
48911ae7e03f6d07d098d74e380f5752475b8db7      <- exact expected HEAD

$ git status --short          (clean)
$ git diff --cached           (nothing staged)
$ git diff --check            exit 0
```

Production fingerprint recorded before any edit:

```
$ find unmark scripts configs requirements -type f | sort | xargs sha256sum | sha256sum
464df4818142fd250833293f60860e546da07a020df1678a7491c26a7dd48668
```

`docs/audits/038-*` did not exist — the previous CUDA acceptance attempt deliberately did not create
an Audit 038, so the path was free. No collision.

## 2. CUDA Failure Evidence

First torch gate on the real Colab CUDA host:

```
python -B -m pytest -q -rs tests/test_stage1_resume_state_machine_torch.py
10 passed, 2 failed
```

## 3. The Two Failing Tests

```
test_case_c_a_40k_continuation_checkpoint_reconstructs_cap_40000
test_three_candidates_resume_independently
```

## 4. Traceback Root Cause

Both resume a **legitimate mid-continuation** checkpoint — `global_update = 20500`, `cap = 40000` —
through real `train_run()` while the fixture passes `tokenizer=None`. Production evaluates the loop
guard `20500 < 40000`, which is **true**, enters the training body, and reaches:

```
train_run -> prepare_serially -> prepare_example -> prepare_with_condition
AttributeError: 'NoneType' object has no attribute 'convert_tokens_to_ids'
```

The failure is the fixture's `tokenizer=None` meeting a loop body that was never expected to run.

The 10/2 split was reproduced exactly by static derivation before any change, from each case's
`(global_update, cap)` and whether it raises before the loop:

| case | gu | cap | enters loop? |
|---|---|---|---|
| case_a initial leg | 20 000 | 20 000 | no |
| case_b boundary promotes | 20 000 | 20 000 | no |
| **case_c reconstructs 40000** | **20 500** | **40 000** | **YES — failed** |
| case_c pre-repair cap refused | 20 500 | 20 000 | no (raises) |
| case_d later continuation | 40 000 | 40 000 | no |
| case_d budget limited | 40 000 | 40 000 | no |
| case_e gu > cap | 25 000 | 20 000 | no (raises) |
| case_f invalid cap | 500 | 20 000 | no (raises) |
| case_g malformed point | 500 | 20 000 | no (raises) |
| adapter/optimizer restore | 20 000 | 20 000 | no |
| foreign run refused | 500 | 20 000 | no (raises) |
| **A/B/C — candidate B** | **20 500** | **40 000** | **YES — failed** |

Exactly two, exactly the two the CUDA host named.

## 5. Why This Is Test Design, Not Production Failure

The loop was entered **because the repair works**. Before the consolidated repair, `execute_stage`
passed `cap=INITIAL_MAX_UPDATES` unconditionally, so a 20500 checkpoint came back under a 20k cap,
`20500 < 20000` was false, and the continuation silently performed zero updates while recording
itself as a complete 20k run. The CUDA run shows the opposite: the checkpoint is now reconstructed at
`cap = 40000` and production *does* continue the 40k leg.

The `AttributeError` arises solely from the test fixture supplying `tokenizer=None` — a choice that
was valid only under the (false) assumption that no case could reach the loop. No production
contract, guard or computation is implicated. Production is byte-identical before and after this
repair (§17).

## 6. The Contradiction in Audit 035

Audit 035 §9 stated:

> "**Structural** — every case restores at `global_update == cap`, so
> `while global_update < cap` is false on entry and the loop body cannot execute."

**That statement was false**, for `case_c` (20500/40000) and for candidate B of the A/B/C scenario.
It was asserted from the design of the *completed-state* cases and generalised to the whole file
without checking the two mid-continuation cases — the very cases whose purpose is that the run is
**not** complete. Static name-resolution and import checks could not catch it, because it is a
semantic property of `(global_update, cap)` pairs, not of symbols.

Audit 035 is **not** modified. This audit is the correction of record. The false claim is also
corrected in the test file's own module docstring, where the next reader will meet it.

## 7. Exact Test Changes

One file: `tests/test_stage1_resume_state_machine_torch.py`.

| Change | Purpose |
|---|---|
| Added `_ContinuationEntered(Exception)` | A dedicated class so an intentional stop can never be confused with a real failure |
| Added `resume_expecting_continuation(...)` | Drives real `train_run` into the continuation body, then stops at the sentinel |
| Rewrote `test_case_c_...reconstructs_cap_40000` | Keeps 20500/40000 semantics; proves entry instead of completion |
| Rewrote the resume half of `test_three_candidates_resume_independently` | A returns normally; B uses the sentinel; isolation strengthened |
| Corrected the module docstring | Removes the false "every case restores at `global_update == cap`" contract |
| Imports: `BATCH_SIZE`, `require_resumable_leg` | Used by the new assertions |

No production import, no new dependency, no assertion weakened. Test count unchanged at 12.

## 8. Case C Semantics

The scientific assertion is **unchanged**: a `20500 / 40000` checkpoint resumes as cap 40000. It was
*not* rewritten to 40000/40000 to dodge the loop. The case now proves, in order:

1. `resume_cap(carried) == 40000` — reconstructed from validated persisted state;
2. `require_resumable_leg(carried, 40000)` — accepted as the leg to continue under;
3. real `train_run` restores the mid-continuation state and **enters the continuation body**, where
   the sentinel stops it;
4. the restored sampler produced a full `BATCH_SIZE` batch drawn from `CHUNKS`;
5. `evaluate_fn` was never called — update 0 was restored, not re-measured.

**Reaching the sentinel is the assertion.** The loop guard is `while global_update < cap` at
`global_update == 20500`, so the body is reachable only under cap 40000.

## 9. A/B/C Semantics

Preserved: A completed the initial leg, B crashed at 20500 during the 40k continuation, C never
started. `caps == {"A": 20000, "B": 40000, "C": 20000}` still asserted, and `carried_c is None`.

Strengthened rather than weakened:

- A and B now carry **distinguishable** histories (2 points vs 3), so `result_a.points == points_a`
  and `len(result_a.points) == 2` are real anti-leak assertions rather than a comparison of two
  identical lists;
- A is a completed leg, so its real restore returns a `RunResult` normally and does **no** new
  scientific work;
- B is stopped at the sentinel after reaching its continuation body — not driven through 19 500
  updates;
- namespace isolation asserted directly: distinct checkpoint directories, `carried_a["cap"] != carried_b["cap"]`,
  `carried_a["global_update"] != carried_b["global_update"]`, and `result_a.continued is False`;
- C is proven by the absence of a checkpoint plus the locked `INITIAL_MAX_UPDATES`, with no training
  run started.

## 10. Sentinel Placement Relative to Real Restore Code

`train_run` resolves `from unmark.stage1.preparation import prepare_serially` **inside its own body**
(`unmark/stage1/trainer.py:694`), so the binding is read from the module object on every call. The
test patches `unmark.stage1.preparation.prepare_serially` — the exact lookup site, not an unrelated
definition. The mechanism was verified independently: a function-local `from X import Y` does observe
a patched module attribute.

Order inside `train_run`, by line offset from its `def`:

```
 62  verify_checkpoint                         <- real
 66  require_resumable_leg                     <- real, the repaired cap gate
 68  execution.require_compatible              <- real
 72  adapter.load_state_dict(strict=True)      <- real
 73  optimizer.load_state_dict                 <- real
 74  require_optimizer_parameter_identity      <- real
 75  require_optimizer_state_device            <- real
 76  DeterministicSampler.from_state           <- real
 77  global_update = int(resume[...])          <- real
 81  ValidationPoint.from_dict                 <- real, the repaired reader
 82  result.continued                          <- real
 90  objective.train(True)
 91  while global_update < cap:                <- the cap decision
 96  sampler.next_batch(BATCH_SIZE)            <- restored sampler produces a batch
106  prepare_serially(...)                     <- *** SENTINEL ***
122  batch_to_device
125  objective(batch)                          <- forward: never reached
128  loss.backward()                           <- never reached
     optimizer.step()                          <- never reached
```

The sentinel fires after **all eleven** restore seams and after the cap decision, and before any
preparation, forward, backward or optimizer step. Nothing under test is mocked: not `train_run`, not
`verify_checkpoint`, not `ValidationPoint.from_dict`, not `resume_cap`, not `require_resumable_leg`;
no adapter, optimizer, sampler or provenance restore is bypassed.

## 11. Why No Tokenizer Fake and No 19 500-Update Run

Supplying a fake tokenizer would have made the test *pass* by letting production grind from 20 500 to
40 000 — 19 500 real optimizer steps on a real adapter. That is a scientific training run by another
name: it would take hours of GPU time, execute tens of thousands of updates the acceptance gate
explicitly forbids, and prove nothing the sentinel does not prove in microseconds. The interesting
property is **"did production decide to continue under cap 40000?"**, and that question is answered
the instant the body is entered.

## 12. Full No-Update Safety Argument

The file no longer claims a uniform `global_update == cap`. Its contract is now truthful and split:

- **completed-state restores** (20000/20000, 40000/40000) — loop body structurally unreachable;
- **mid-continuation restores** (20500/40000) — loop entry intentionally detected and interrupted
  before preparation, forward, backward and optimizer step.

Four independent guards remain, and they overlap:

1. the sentinel raises on the first call the loop body makes;
2. `AdamW.step` is poisoned in **both** helpers and restored in `finally`;
3. `_NoForwardObjective.forward` raises;
4. `_FrozenBackbone.forward` raises.

**Scientific optimizer.step: ZERO. Synthetic optimizer.step in this file: ZERO.** No real model
training occurs.

## 13. Mutation-Quality Argument

Demonstrated against the real production functions:

1. **Cap regression.** If production reconstructed the 20500 checkpoint as cap 20000, then
   `20500 < 20000` is false, the body is never entered, the sentinel never fires, and
   `pytest.raises(_ContinuationEntered)` fails. Independently, `require_resumable_leg(payload, 20000)`
   refuses with *"refusing to resume a checkpoint written under cap 40000 with the smaller cap
   20000"*. Either way the repaired case **FAILS** — this is the property the case exists for, and it
   is exactly the pre-repair production bug.
2. **`ValidationPoint` restore regression.** `ValidationPoint(**writer_output)` still raises
   `TypeError: ... unexpected keyword argument 'score'`, at line 81 — before the loop. Sentinel never
   fires → FAIL.
3. **Provenance / optimizer / sampler regression.** All six of those seams execute before the loop
   guard; any raise propagates in place of `_ContinuationEntered` → FAIL.
4. **Correct production.** Cap 40000 reconstructed and accepted, loop entered, sentinel raises before
   `batch_to_device`, the forward, the backward and `optimizer.step`.

## 14. Review of the Other 10 Tests

All 12 were re-derived from their `(global_update, cap)` pairs (§4). The other ten either restore a
**completed** state (`gu == cap`) or raise at a restore seam before the loop guard. **None** shares
the false assumption, and none was modified. Three of them (case_e, case_f, case_g) depend on raising
before the loop, and each asserts that raise explicitly with `pytest.raises`, so the dependency is
stated rather than incidental. Scope was not broadened.

## 15. Local Focused Results

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs \
    tests/test_stage1_resume_state_machine_torch.py \
    tests/test_stage1_resume_state_machine.py \
    tests/test_stage1_training_resume.py \
    tests/test_stage1_training_resume_state.py

55 passed, 2 skipped in 0.23s

SKIPPED tests/test_stage1_resume_state_machine_torch.py:71  the real train_run half needs torch
SKIPPED tests/test_stage1_training_resume.py:42             the tensor half needs torch
```

**The torch skip is NOT proof of CUDA correctness.** The repaired file did not execute locally; it
must run on the authoritative GPU host (§20).

## 16. Full Local Suite

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3737 passed, 105 skipped in 132.30s
```

Identical to the figures before this repair, as expected for a change confined to a file that skips
locally. All 105 skips carry torch-absence reasons.

## 17. Production Byte Identity

```
before:  464df4818142fd250833293f60860e546da07a020df1678a7491c26a7dd48668
after:   464df4818142fd250833293f60860e546da07a020df1678a7491c26a7dd48668
```

`unmark/`, `scripts/`, `configs/` and `requirements/` are byte-identical. No production, proposal,
decisions, freeze or prior-audit file was modified.

## 18. Scientific `optimizer.step` Count

**ZERO.** `lr-pilot`, `r-phase1` and `final-main` were not invoked. No training was executed. The
torch-gated file skipped locally, and by §12 it cannot step an optimizer even when it runs.

## 19. Official UIT-VSFC TEST Status

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command. No information derived from it.

## 20. Exact CUDA Rerun Command

```
python -B -m pytest -q -rs tests/test_stage1_resume_state_machine_torch.py
```

Expected: **12 passed, 0 skipped, 0 failed, 0 errors.**

Diagnostics if it does not:

- `AttributeError: 'NoneType' object has no attribute 'convert_tokens_to_ids'` → the sentinel did not
  install at the lookup site;
- `TrainerContractViolation: refusing to resume ... smaller cap` in case C → production regressed to
  cap 20000;
- `Failed: DID NOT RAISE _ContinuationEntered` → the continuation body was not entered, i.e. the cap
  reconstruction regressed;
- any `AssertionError` naming `optimizer.step()` → the no-update discipline broke.

Then resume the small authoritative CUDA acceptance from its second torch gate.

## 21. Final Verdict

The defect was in the test's assumption, not in production: the CUDA run is positive evidence that
the repaired cap reconstruction continues a 20500/40000 checkpoint under cap 40000. Case C keeps its
20500/40000 semantics and now proves continuation *entry* rather than completion, stopping at a
dedicated sentinel placed after every restore seam and before any training work. The A/B/C scenario
keeps its isolation proof and gained stronger anti-leak assertions. A cap regression would make the
repaired case fail. Production is byte-identical, and no scientific optimizer step occurred.

**CUDA-DISCOVERED MID-CONTINUATION TEST REPAIR COMPLETE — READY TO COMMIT AND RERUN THE SMALL CUDA ACCEPTANCE**

---

*End of Audit 038.*
