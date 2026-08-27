# Audit 042 — CUDA Telemetry-Equivalence Root Cause

**Scope:** root-cause investigation of the CUDA equivalence-gate failure, then a
test-only repair. **Production byte-identical.** Audits 001–041 untouched.
**Date:** 2026-08-27

---

## 1. Observed CUDA Fact

At HEAD `8805c2d2a7e0842e350c18f65c8d720b7c08a0b9`:

```
tests/test_stage1_telemetry_equivalence_torch.py    2 failed, 7 passed

FAILED test_telemetry_on_is_scientifically_identical_to_telemetry_off
FAILED test_the_written_checkpoint_is_scientifically_identical

adapter parameter tone_embedding.weight differs between telemetry OFF and ON
(and the checkpoint adapter_state differs as well)
```

CUDA scientific equivalence was therefore **not** established, and could not be
claimed.

## 2. Hypotheses Considered

| | hypothesis | verdict |
|---|---|---|
| A | test-fixture / start-state reproducibility defect | **CONFIRMED** |
| B | shared mutable resume/checkpoint state between OFF and ON | excluded — the payload was already deep-copied per execution, and a new test proves it is not mutated |
| C | RNG initialisation/order defect in the test | **CONFIRMED — same defect as A** |
| D | genuine telemetry-induced execution divergence | **excluded** — see §5 |
| E | pre-existing CUDA nondeterminism exposed by the new test | **structurally excluded** — see §4 |

## 3. Root Cause

**The seed was set 28 lines too late.**

`run_once` executed in this order:

```
276   carried = copy.deepcopy(carried)
277   objective = build_objective()        <-- consumes GLOBAL torch RNG
...
305   torch.manual_seed(0)                 <-- far too late
306   rng_before = torch.random.get_rng_state()
308   result = train_run(...)
```

`build_objective()` constructs `_TinyBackbone`, whose `__init__` builds

```python
self.word_embeddings = torch.nn.Embedding(4096, TINY_HIDDEN, padding_idx=1)
self.projection      = torch.nn.Linear(TINY_HIDDEN, TINY_HIDDEN)
```

Both call `reset_parameters()`, which draws from the **global** torch RNG, under
no seed control.

**Why the adapter was immune and the encoder was not** — this is the crux:

* the **adapter** is safe twice over. `fresh_adapter` wraps construction in
  `torch.random.fork_rng(devices=[])` + `torch.default_generator.manual_seed(init_seed)`,
  so it is both deterministic *and* RNG-state-restoring; and it is then fully
  overwritten by `adapter.load_state_dict(resume["adapter_state"], strict=True)`;
* the **frozen encoder is restored from nothing**. Stage-1 v2 checkpoints are
  **adapter-only by design** (`trainer.py:423` persists `adapter_state`, and
  nothing persists encoder weights). Its start state is therefore a pure
  function of the ambient RNG at construction.

So the first execution (OFF) built its encoder from one RNG state, ran 50 real
updates — advancing the RNG — and the second execution (ON) then built its
encoder from a *different* RNG state. Different frozen encoder weights →
different representations out of `projection(word_embeddings(x))` → different
loss → different gradients into the adapter → **`tone_embedding.weight`
diverges after 50 updates**, and the checkpoint written at update 20 000
diverges with it.

That is precisely the observed pair of failures, and it explains both of them
with one mechanism.

## 4. Why CUDA Nondeterminism Is Excluded

The fixture never calls `.to(...)` and never references a CUDA device; `grep`
finds no `cuda`/`device` reference outside a docstring. `train_run`'s only
device boundary is `batch_to_device(..., module_device(objective))`, and
`module_device` returns the device of the objective's own parameters — CPU here.

The whole comparison therefore runs on **CPU even on the CUDA host**. The
failure was deterministic and reproducible, not a flaky nondeterminism artifact,
and hypothesis E cannot apply.

## 5. Why Telemetry Is Not Implicated

The mechanism in §3 is entirely upstream of the sink and does not involve it:

* it acts at **object construction**, before `train_run` is entered and before
  any event can be emitted;
* it is driven by *how much RNG the previous execution consumed*, which is
  identical whether that execution had telemetry on or off — telemetry consumes
  zero RNG (asserted by `test_telemetry_consumes_no_torch_rng`);
* the decisive counterfactual: **OFF vs OFF would have failed too**. The second
  OFF run would have built its encoder from the first OFF run's leftover RNG
  state exactly as the ON run did.

Because the pre-repair file never ran OFF twice, the OFF-vs-ON assertion was
**not valid evidence against telemetry**. That missing counterfactual is the
reason a fixture defect could present as a production-blocking finding.

**Production is not implicated. No production diagnosis is owed, and none is
made.**

## 6. Diagnostic Matrix Now Implemented

The repaired file establishes all three cells, and asserts start-state equality
*before* interpreting any outcome:

| test | establishes |
|---|---|
| `test_off_vs_off_is_reproducible` | **OFF1 == OFF2** — start *and* outcome |
| `test_on_vs_on_is_reproducible` | **ON1 == ON2** — start *and* outcome |
| `test_telemetry_on_is_scientifically_identical_to_telemetry_off` | **OFF == ON**, via the same comparator |
| `test_the_frozen_encoder_starts_identical_in_every_execution` | the exact tensor that diverged |
| `test_an_execution_does_not_mutate_the_canonical_starting_payload` | hypothesis B excluded by test |
| `test_the_fixture_resumes_under_the_real_locked_budget` | cap 20 000, 19 950 → 20 000 retained |

## 7. Initial-State Fingerprint Evidence

Every execution now captures, after construction and **before** training:

```
adapter tensors · frozen ENCODER tensors · resume-payload adapter tensors
sampler state · canonical ValidationPoint history · global_update · cap
provenance · torch RNG state at train_run entry
```

`assert_same_start` compares these with `torch.equal` per tensor via
`same_tensor_state`, because **shallow dict equality is insufficient** for
tensors. The encoder comparison carries an explicit message naming the cause:
*"the encoder is restored from no checkpoint, so it must be constructed under a
controlled seed"*.

`assert_same_outcome` likewise compares final adapter tensors, the written
checkpoint's adapter tensors, sampler state, points, cap, continued,
budget_limited, `RunResult.to_dict()`, RNG state and all five call counters.
Both comparators are shared by the OFF/OFF, ON/ON and OFF/ON cells, so one
standard governs the whole matrix.

## 8. The Repair

**Test-only.** `torch.manual_seed(seed)` now runs **before** `build_objective()`
in both `run_once` and `build_resume_payload`, so every execution constructs its
frozen encoder from an identical RNG state. The late duplicate seed was removed;
`rng_before` is captured after construction, where it is deterministic.

Deliberately retained, unchanged:

* `torch.equal` throughout — **not** weakened to `allclose`; the requirement
  remains bit-identical (8 uses, 0 `allclose`);
* the real locked `CAP = INITIAL_MAX_UPDATES = 20 000`;
* the short legal resume `19 950 → 20 000`, 50 real updates;
* the real `train_run`, optimizer, sampler, checkpoint and `resolve_budget`;
* every ON/OFF scientific-equivalence assertion.

## 9. Exact Files Changed

* `tests/test_stage1_telemetry_equivalence_torch.py` — seed ordering, tensor-aware
  fingerprint/comparison helpers, four new diagnostic tests (9 → 13).
* `docs/audits/042-cuda-telemetry-equivalence-root-cause.md` — this file.

No production file. No scientific configuration. `protocol.py`, LR/r grids,
seeds, batch size, corruption, optimizer, numerical policy, checkpoint and
validation cadence, budget rule, selection, architecture, precision and TEST
sealing are all untouched.

## 10. Production Fingerprint — Byte-Identical

```
before:  66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
after:   66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
```

## 11. Local Test Results

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs \
    tests/test_stage1_telemetry_equivalence_torch.py tests/test_stage1_telemetry.py \
    tests/test_stage1_resume_state_machine.py tests/test_stage1_training_resume_state.py

103 passed, 1 skipped
SKIPPED tests/test_stage1_telemetry_equivalence_torch.py:44
        the scientific-equivalence half needs torch
```

Full suite:

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3786 passed, 106 skipped in 133.21s (0:02:13)
```

**The repaired file did not execute locally** — torch is absent from the
deliberate ML-free venv. The root cause was established by reading production
and fixture source (RNG ownership, `fork_rng` semantics, adapter-only v2
checkpoint scope, CPU-only device resolution), and the repair is correct by
construction. **No CUDA pass is claimed.**

## 12. No Training, TEST Sealed

No `lr-pilot`, `r-phase1` or `final-main`. No scientific optimizer step. Official
UIT-VSFC TEST remains **SEALED / UNUSED** — not opened, inspected, read,
screened, evaluated, mounted or passed to any command.

## 13. Exact CUDA Rerun Required

```
python -B -m pytest -q -rs tests/test_stage1_telemetry_equivalence_torch.py
```

Expected: **13 passed, 0 skipped, 0 failed, 0 errors.**

Read the result in this order — the matrix is diagnostic, not decorative:

1. **`test_off_vs_off_is_reproducible` fails** → the fixture is still not
   reproducible; an OFF-vs-ON difference would again prove nothing about
   telemetry. Look for another unseeded construction.
2. **`test_the_frozen_encoder_starts_identical_in_every_execution` fails** →
   the seed ordering regressed, or something else constructs RNG-dependent state.
3. **OFF/OFF and ON/ON pass but OFF-vs-ON fails** → *then* the divergence is a
   genuine telemetry effect and production is blocked. Bisect by comparing
   adapter, gradients, optimizer state, loss, sampler and RNG state after each of
   the 50 updates to find the first diverging update, and determine whether it
   precedes emission, coincides with `float(loss.detach())`, or follows it. Do
   not repair production before that diagnosis exists.

Then continue the focused CUDA re-acceptance from Audit 040 §22 and the
telemetry performance comparison from §22b.

---

**Status: FIXTURE DEFECT PROVEN — PRODUCTION BYTE-IDENTICAL — CUDA RERUN REQUIRED.**

*End of Audit 042.*
