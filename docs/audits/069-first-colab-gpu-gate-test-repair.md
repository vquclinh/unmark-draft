# Audit 069 - First Colab GPU Gate: Stale-Test Repair

**Scope:** repair the two Torch-runtime test failures the first Colab A100 GPU
gate found at HEAD `4b7ff945e64610edad0be71df9f7f1b6dfe00149`, and nothing else.
**Date:** 2026-09-11
**Type:** narrowly scoped test repair plus verification. No commit was created,
no model was trained, no UIT-VSFC data, official validation or official TEST was
accessed, no ML package was installed, and no production code was changed.

Audits 065-068 are byte-unchanged (`61672fb15d1a22342848cf4e5d18b23d`,
`35076e34f9ffd8360da4fb53c15cbc00`, `e8038e6b12d442226f3904bdab3bf13c`,
`d5eeeafc0b6d5b946e76f3ec90c6fb3d`).

---

## 0. Executive verdict

**PASS_WITH_COLAB_RERUN.**

Both failures are **stale tests**, not production defects. Each assertion was
written before a later candidate legitimately extended the thing it was checking,
and each was verifying a weaker or wrong invariant than its own name claimed. In
one case the failure is the *evidence that a C1 safety guard works*.

No production file was touched. Only `tests/test_stage1_v2_gc.py` changed:
**157 insertions, 11 deletions.**

---

## 1. Failure 1 - the result-dict type assertion

**The GPU failure**

```
tests/test_stage1_v2_gc.py::test_runtime_the_result_dict_carries_the_identity_and_no_raw_text
    assert all(isinstance(v, (int, float, str)) for v in payload.values())
```

**Inspection.** `GridConsistencyLossResult.to_dict()` ends with
`**self.weights.to_dict(), **self.objective.to_dict()`. Since C2,
`ObjectiveIdentity.to_dict()` durably carries four keys, not two:

```python
{"objective_id": ..., "lambda_grid": ..., "lambda_grd": ..., "relational": ...}
```

For C3 the last two are legitimately `None` -- honest absence, which is exactly
the repository's convention ("`None` is honest absence, never zero"). The payload
was reproduced faithfully offline:

```
offending values: {'lambda_grd': None, 'relational': None}
OLD assertion isinstance(v, (int, float, str)) -> False
```

`None` is valid JSON (`null`) and correct provenance. **The production output is
right; the assertion was wrong.** It also never tested what its own name claimed:
`isinstance(v, str)` would have accepted an entire corpus chunk.

**Repair.** The test now checks the real invariant, in three parts:

1. **Identity** -- `objective_id == "grid-consistency-v1"`, `lambda_grid == 1.0`,
   `lambda_grd is None`, `relational is None`, `lambda_align == lambda_clean == 1.0`.
2. **Terms present** -- all three losses, all three mean distances, `batch_size == 2`.
3. **Nothing unsafe** -- a new `assert_json_safe` walker plus the repository's
   canonical `json.dumps(..., allow_nan=False)` round trip.

No existing canonical JSON-safety helper was found (searched `unmark/`, `tests/`,
`scripts/`), so the repo's established idiom -- `json.dumps` as the
serializability oracle, used in `test_preg1_profiling.py` and
`test_preg1_head.py` -- is reused and paired with an explicit recursive type
check. Permitted: `dict`, `list`/`tuple`, `str`, `int`, `float`, `bool`, `None`.
Refused explicitly: `torch.Tensor`, arbitrary runtime objects, NaN/Inf,
non-string keys.

**The "no raw text" guarantee is strengthened, not weakened.** Strings are no
longer accepted by type; they must belong to `LOCKED_IDENTITY_TOKENS` -- the
closed vocabulary of objective ids, fusion ids and relation descriptors (8
tokens). Verified offline that the walker rejects raw Vietnamese corpus text, a
runtime object, NaN, Inf and a non-string key, and accepts C2's nested
`relational` block.

No C2 provenance field was removed from production output, and the relational
specification was not flattened.

---

## 2. Failure 2 - one adapter reused for every candidate

**The GPU failure**

```
tests/test_stage1_v2_gc.py::test_runtime_the_shared_constructor_builds_each_candidates_objective
```

**Inspection.** The test obtained one adapter from `build_grid_stack()`, which
delegates to `build_stack()` in `test_stage1.py:785`:

```python
adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=d))   # historical fusion
```

and then reused that single historical-fusion adapter for **every** candidate in
`CANDIDATES`. Once C1 existed that became scientifically invalid, and production
correctly refused it at `execute.build_candidate_objective`:

```python
built = getattr(unmark_encoder.adapter.config, "fusion_id", None)
if built != candidate.fusion.fusion_id:
    raise Stage1ContractViolation(
        f"{candidate.stage}: the adapter implements fusion {built!r} but the "
        f"candidate declares {candidate.fusion.fusion_id!r}. The architecture "
        "and the provenance must be the same statement."
    )
```

**This failure is the guard doing its job.** The adapter `state_dict` is
shape-compatible across fusions, so without the guard a `v2_scf` run would train
the historical mixture rule while its provenance claimed the calibrated one. The
guard is untouched: not weakened, bypassed, removed, monkeypatched or
special-cased.

**Repair.** Each candidate is now given an adapter built for **its own** fusion,
through the authoritative candidate-aware path C1 introduced:

```python
adapter = fresh_adapter(16, 51800, candidate.fusion.fusion_id)
```

with an explicit expectation table, checked against the live register:

| stage | fusion | objective type |
| --- | --- | --- |
| `lr_pilot` / `r_phase1` / `final_main` | `historical-fusion-v1` | `Stage1Objective` |
| `v2_scf` | `scale-calibrated-fusion-v1` | `Stage1Objective` |
| `v2_grd` | `historical-fusion-v1` | `RelationalDistillationObjective` |
| `v2_gc` | `historical-fusion-v1` | `GridConsistencyObjective` |

The test asserts the stage set equals the register exactly (so a future candidate
cannot be added without an expectation here), uses `type(built) is ...` rather
than `isinstance` (the candidate objectives subclass `Stage1Objective`, so
`isinstance` could not tell them apart), and still proves the shared dispatch and
the zero-parameter-delta property.

Verified offline against the live register: all six stages present, every
declared fusion matching the table, and the guard message containing
`"implements fusion"` so the negative test's `pytest.raises(match=...)` binds.

**Negative test added.** `test_runtime_a_mismatched_fusion_fails_closed` proves
the guard permanently, in both directions:

* historical-fusion adapter + `v2_scf` candidate -> `Stage1ContractViolation`
* scale-calibrated adapter + `lr_pilot` candidate -> `Stage1ContractViolation`

---

## 3. Scientific non-regression

`git status` shows exactly one modified path: `tests/test_stage1_v2_gc.py`. No
file under `unmark/`, `scripts/` or `docs/spec/` changed. Each formula was
re-read in the current tree:

**C1 / V2-SCF** -- `adapter.py:130-133`

```python
e_norm = base.norm(dim=-1, keepdim=True)
f_norm = fused.norm(dim=-1, keepdim=True)
scale  = e_norm / f_norm.clamp(min=FUSION_SCALE_EPSILON)
return scale * fused
```

historical Stage-1 objective, 3 encoder forwards, hard cap 20 000 -- unchanged.

**C2 / V2-GRD** -- `objective_relational.py:180, 202-203, 215, 262, 266, 363, 372`
clean-native FIRST_TOKEN teacher (`hidden[:, 0, :]`), row-normalised cosine Gram,
`~torch.eye` off-diagonal mask, `teacher_target.detach()`,
`(difference[mask] ** 2).mean()`, `0.5`/`0.5`, `lambda_grd = 1.0`, historical
fusion, 3 encoder forwards, hard cap 20 000 -- unchanged.

**C3 / V2-GC** -- `objective_grid.py:214, 227, 235, 339`
`hidden_clean_target.detach()`, `eps=COSINE_EPS`, per-example
`sum(dim=1) / counts`, `lambda_grid * loss_grid`, historical fusion, 3 encoder
forwards, hard cap 20 000 -- unchanged.

**Historical Stage-1** -- untouched; no candidate budget, W&B project semantics,
selection rule or provenance field was altered.

---

## 4. Tests

| Command | Result |
| --- | --- |
| `git diff --check` | **clean** |
| `python -m compileall -q unmark/ scripts/ tests/` | **all modules compile** |
| the exact GPU-gate command, locally | **242 passed, 69 skipped** |
| `python -m pytest tests/ -q` (full local suite) | **4696 passed, 242 skipped, 0 failed** |

Torch is absent from the local ML-free venv by design and was **not** installed,
so the two repaired tests and the new negative test skip here. All three remain
`@requires_torch` and were verified to be listed as SKIPPED rather than removed,
xfailed or converted to static-only checks -- they will execute on Colab.

Torch-gated tests across the gate command, by candidate:

```
C1 test_stage1_v2_scf        67 passed   17 torch-gated
C2 test_stage1_v2_grd        81 passed   22 torch-gated
C3 test_stage1_v2_gc         52 passed   29 torch-gated
C3 test_stage1_v2_gc_budget  42 passed    1 torch-gated
                                         69 total
```

The gate previously collected 310 (2 failed + 308 passed). It should now collect
**311** -- one more, the added negative test.

What could not be verified locally was verified by faithful offline simulation:
the C3 payload was reconstructed from the real `ObjectiveIdentity.to_dict()` and
`ObjectiveWeights.to_dict()` with the torch-derived numbers stubbed as the floats
they are converted to, confirming both that the old assertion fails exactly on
`lambda_grd`/`relational` and that every new assertion passes.

---

FINAL AUDIT — FIRST COLAB GPU GATE REPAIR

SOURCE_HEAD:
4b7ff945e64610edad0be71df9f7f1b6dfe00149

FAILURE_1_ROOT_CAUSE:
STALE TEST. `assert all(isinstance(v, (int, float, str)) ...)` was written when ObjectiveIdentity.to_dict() held two keys. C2 durably added `lambda_grd` and the nested `relational` specification, both legitimately None for a non-relational objective, and None is not int/float/str. The assertion also never tested its own name: isinstance(v, str) would have accepted raw corpus text.

FAILURE_1_PRODUCTION_BUG:
NO — the payload is correct, JSON-safe provenance; None is valid JSON null and the repository's convention for honest absence

FAILURE_1_TEST_REPAIRED:
YES — now asserts the C3 identity (objective_id "grid-consistency-v1", lambda_grid 1.0, lambda_grd None, relational None, lambda_align == lambda_clean == 1.0), the expected loss/diagnostic fields and batch_size, then a recursive assert_json_safe walker plus the repository's canonical json.dumps(allow_nan=False) round trip. Refuses torch.Tensor, arbitrary runtime objects, NaN/Inf and non-string keys; strings are constrained to a closed 8-token locked-identity vocabulary, so the "no raw text" guarantee is strengthened. No C2 provenance field was removed and the relational spec was not flattened.

FAILURE_2_ROOT_CAUSE:
STALE TEST. It reused one historical-fusion adapter from build_grid_stack() for every candidate in CANDIDATES. Once C1 existed, `v2_scf` declares scale-calibrated-fusion-v1, so production correctly refused the mismatch. The failure is the C1 fail-closed guard working.

FAILURE_2_PRODUCTION_BUG:
NO — the guard in execute.build_candidate_objective is correct and necessary: adapter state_dicts are shape-compatible across fusions, so without it a run would train one architecture while its provenance claimed another

FAILURE_2_TEST_REPAIRED:
YES — each candidate is now exercised with an adapter built from its OWN fusion identity via the authoritative path `fresh_adapter(16, 51800, candidate.fusion.fusion_id)`, against an expectation table asserted to match the live register exactly; `type(built) is ...` replaces `isinstance` so the candidate subclasses are distinguished; the shared-dispatch and zero-parameter-delta properties are retained

FUSION_MISMATCH_STILL_FAILS_CLOSED:
YES — the production guard is untouched, and a new runtime negative test (`test_runtime_a_mismatched_fusion_fails_closed`) pins it in both directions: historical adapter + v2_scf candidate, and scale-calibrated adapter + lr_pilot candidate, each expecting Stage1ContractViolation

C1_SCIENTIFIC_SEMANTICS_CHANGED:
NO

C2_SCIENTIFIC_SEMANTICS_CHANGED:
NO

C3_SCIENTIFIC_SEMANTICS_CHANGED:
NO

HISTORICAL_SEMANTICS_CHANGED:
NO

FILES_CHANGED_BY_THIS_REPAIR:
tests/test_stage1_v2_gc.py (157 insertions, 11 deletions) — the only changed file; no production, script or spec file was touched

LOCAL_TESTS_PASSED:
4696

LOCAL_TESTS_FAILED:
0

LOCAL_TESTS_SKIPPED:
242

COLAB_RERUN_REQUIRED:
YES — torch is absent locally and was not installed, so the two repaired tests and the added negative test could not execute here. Re-run the gate command; expect 311 collected (one more than the previous 310) with 0 failures.

BLOCKERS_BEFORE_STAGE1_TRAINING:
- NONE new. This repair introduced no production change.
- REMAINING GATE (unchanged): the 69 torch-gated candidate tests must pass on Colab/GPU, and the three real-model smokes (`--candidate v2_scf`, `v2_grd`, `v2_gc`) must run against real PhoBERT and the real prepared corpus.
- OPEN (non-blocking, unchanged): the adjudication protocol for comparing C1, C2, C3 and UNMARK-A/B is undefined, and finalists.py still pins exactly two finalists.

AUDIT_VERDICT:
PASS_WITH_COLAB_RERUN
