# Audit 041 — CUDA Telemetry-Equivalence Fixture Repair

**Scope:** ONE test-fixture defect found by the authoritative CUDA run.
`tests/test_stage1_telemetry_equivalence_torch.py` only.
**Date:** 2026-08-27
**Mode:** TEST REPAIR. Production byte-identical. Audits 001–040 untouched.

---

## 1. Starting HEAD

```
$ git rev-parse HEAD
15594d62aecd47f5bd85bccffef0de69ee365e25
$ git status --short          (clean)
```

Production fingerprint recorded before any edit:

```
$ find unmark scripts configs requirements -type f | sort | xargs sha256sum | sha256sum
66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
```

## 2. CUDA Command and First Failure

```
tests/test_stage1_telemetry_equivalence_torch.py::
  test_telemetry_on_is_scientifically_identical_to_telemetry_off

SelectionViolation: cap 4 is not one of the locked budgets (20000, 40000)
```

## 3. The Failure Occurred in the Telemetry-OFF Execution

The exception was raised inside

```python
off = run_once(... sink=None ...)
```

i.e. **before any ON-vs-OFF comparison was reached**. This matters for
interpretation: the failure says nothing whatsoever about telemetry. It was the
fixture failing on its own, with the sink disabled, on the very first of the two
executions.

## 4. The Illegal Fixture: `CAP = 4`

The fixture ran a **fresh** run with `CAP = 4`. The four training iterations
executed normally. `train_run` then reached its final line:

```
resolve_budget(result) -> budget_decision(selected_update=4, cap=4)
```

and production refused.

## 5. Why Production Was Right to Refuse

`budget_decision` opens by validating the cap against the locked pair:

```python
if cap not in (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES):
    raise SelectionViolation(...)
```

The Stage-1 budget is **precommitted**: 20 000, at most one continuation to
40 000, then `BUDGET_LIMITED`. An arbitrary small cap is not a legal scientific
budget, and a rule that quietly accepted `cap=4` because a test found it
convenient would be exactly the kind of hole the protocol exists to close.

Reproduced here directly:

```
budget_decision(4, 4)          -> SelectionViolation: cap 4 is not one of the locked budgets
budget_decision(19500, 20000)  -> "selected checkpoint is inside the budget"
```

**The defect was the fixture, not the rule.** `budget_decision`, `resolve_budget`
and every other production symbol are untouched; nothing was monkeypatched to
make the test pass, and no arbitrary cap is permitted anywhere.

## 6. Legal-Resume Fixture Design

The fixture no longer starts a fresh run under a fake budget. It **resumes** a
real checkpoint on the initial leg, under the real locked cap:

```
CAP          = INITIAL_MAX_UPDATES = 20 000     (the real locked budget)
START_UPDATE = 19 950
UPDATES      = 50                               (real training iterations)
```

`build_resume_payload` writes the payload through the **real** production
machinery — `checkpoint_payload` → `save_training_checkpoint` → 
`load_training_checkpoint` — so both executions resume from a payload in the
real schema rather than a hand-built dict. It carries a real `RunProvenance`,
real adapter `state_dict`, real optimizer `state_dict`, a real
`DeterministicSampler.state_dict()`, `global_update=19950`, `cap=20000`, and a
canonical `ValidationPoint` history (update 0 plus every 500-boundary through
19 500 — 40 points, exactly what a run at 19 950 would hold).

Nothing on the restore path is bypassed. All of `verify_checkpoint`,
`require_resumable_leg`, `adapter.load_state_dict(strict=True)`,
`optimizer.load_state_dict`, `require_optimizer_parameter_identity`,
`require_optimizer_state_device`, `DeterministicSampler.from_state`,
`ValidationPoint.from_dict` and `resolve_budget` execute for real.

**Budget behaviour, deliberately chosen and verified by torch-free arithmetic.**
`point_for` makes update 0 the best point, so `select_checkpoint` selects update
0 and `budget_decision(0, 20000)` returns *"selected checkpoint is inside the
budget"* — `continue_run=False`, `budget_limited=False`, `cap` stays 20 000.
The real budget rule therefore runs under a real cap without dragging a 40 000
update leg into a unit test.

Derived and confirmed before the CUDA run:

| property | value |
|---|---|
| loop iterations | 50 (19 951 … 20 000) |
| progress events (cadence 50) | exactly one, at 20 000 |
| validation events | exactly one, at 20 000 |
| checkpoint events | exactly one, at 20 000 |
| `select_checkpoint` | update 0, score 0.1300 |
| `is_best` at 20 000 | `False` → writes `training-checkpoint-last.pt` only |

**One honest nuance.** `global_update = 19950` is not itself a boundary
production would ever *write* a checkpoint at — production writes at 500-update
boundaries. It is a legal payload that every reader accepts, and it is used here
because the alternative that is also writer-emittable (19 500) costs 500
optimizer steps instead of 50. What this file proves is telemetry equivalence,
not checkpoint emittability; the writer-emittable state set is proven separately
and exhaustively in `tests/test_stage1_resume_state_machine.py` (Audit 038 §9).

## 7. Exact TEST-ONLY Optimizer Steps

| execution | real `AdamW.step` calls |
|---|---|
| telemetry OFF | **50** |
| telemetry ON | **50** |
| fixture construction (`build_resume_payload`) | **0** |

100 TEST-ONLY synthetic optimizer steps per equivalence comparison, on a d=8
adapter with a fixed 128-example batch. **Scientific optimizer steps: 0.** No
`lr-pilot` / `r-phase1` / `final-main`, no prepared-corpus training, and no
20 000-step run.

The persisted optimizer state in the resume payload is that of a freshly built
AdamW (no accumulated moments) — a structurally valid payload that keeps the
fixture minimal. Populated moments are still compared: the checkpoint each
execution *writes* at update 20 000 carries real moments after 50 updates, and
those are compared tensor by tensor (§8).

## 8. ON/OFF Equivalence Assertions

Both executions resume from the **same** payload, deep-copied per run so they
provably start from identical state.

| compared | assertion |
|---|---|
| final adapter tensors | `torch.equal` per parameter |
| written checkpoint adapter tensors | `torch.equal` per parameter |
| optimizer moments | `torch.equal` per tensor, per state entry; asserted non-empty |
| optimizer `param_groups` | equal |
| sampler state | checkpoint `sampler_state` equal |
| final `global_update` | both 20 000 |
| `cap` | both 20 000 |
| `continued` / `budget_limited` | both `False` |
| `ValidationPoint` history | `result.points` equal; persisted `points` equal |
| `RunResult.to_dict()` | equal |
| selected checkpoint | `selected.to_dict()` equal; `selected.update == 0` |
| torch RNG state | `torch.equal` after both runs |

Counts, required equal ON vs OFF:

| counter | value |
|---|---|
| `Stage1Objective.forward` | 50 |
| `AdamW.step` | 50 |
| `DeterministicSampler.next_batch` | 50 |
| preparation-pool calls | 50 |
| `evaluate_fn` calls | **1** — update 0 was *restored*, not re-measured |

The ON run genuinely emits, and this is asserted rather than assumed: one
`train_progress` at 20 000, one `validation` at 20 000, one `checkpoint` at
20 000 naming `training-checkpoint-last.pt`, and one `run_end`. Event ordering
`validation < checkpoint < run_end` is asserted, and a separate test cross-checks
the checkpoint event's `update` against the payload actually on disk — so a
checkpoint event emitted *before* a successful save would fail.

**Mutation quality.** The repaired test fails if telemetry consumes RNG, causes
an extra optimizer step, or changes adapter state, sampler state, validation
history or checkpoint scientific state; if a checkpoint event precedes its save;
and — via `test_the_fixture_resumes_under_the_real_locked_budget` — if an
arbitrary small cap is ever reintroduced.

## 9. Production Fingerprint — Byte-Identical

```
before:  66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
after:   66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
```

`unmark/`, `scripts/`, `configs/` and `requirements/` are unchanged. No
production file, no scientific configuration, no protocol constant, and no
budget rule was modified. Audit 040 was **not** edited: it documents the
committed observability change, and this is a separate repair.

## 10. Local Results

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs \
    tests/test_stage1_telemetry_equivalence_torch.py \
    tests/test_stage1_telemetry.py \
    tests/test_stage1_resume_state_machine.py \
    tests/test_stage1_training_resume_state.py

103 passed, 1 skipped
SKIPPED tests/test_stage1_telemetry_equivalence_torch.py:44
        the scientific-equivalence half needs torch
```

Full suite: see §11.

**The repaired file did not execute locally** — torch is absent from the
deliberate ML-free venv. Its arithmetic was verified torch-free instead (loop
count, cadence landings, selection outcome, budget decision, `is_best`), but the
assertions themselves remain unproven until the CUDA rerun. **No CUDA pass is
claimed.**

## 11. Full Local Suite

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3786 passed, 106 skipped in 134.31s (0:02:14)
```

## 12. Official UIT-VSFC TEST

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted,
or passed to any command. No information derived from it.

## 13. Exact CUDA Rerun Command

```
python -B -m pytest -q -rs tests/test_stage1_telemetry_equivalence_torch.py
```

Expected: **9 passed, 0 skipped, 0 failed, 0 errors.**

Diagnostics if it does not:

* `SelectionViolation: cap ... is not one of the locked budgets` → an illegal cap
  was reintroduced into the fixture;
* `TrainerContractViolation` on the restore path → the resume payload no longer
  satisfies a production contract;
* a `torch.equal` failure → telemetry changed scientific state, which is the
  defect this file exists to catch;
* a count mismatch → telemetry added a forward, step, batch, preparation call or
  evaluation.

Then continue the focused CUDA re-acceptance from Audit 040 §22, and the
telemetry performance comparison from §22b.

---

**Status: FIXTURE REPAIRED UNDER THE REAL LOCKED BUDGET — PRODUCTION BYTE-IDENTICAL;
CUDA RERUN REQUIRED, NOT YET CLAIMED.**

*End of Audit 041.*
