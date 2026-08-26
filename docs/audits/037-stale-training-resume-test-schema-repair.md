# Audit 037 - Stale Training Resume Test Schema Repair

## 1. Starting HEAD

`de4141afa59ffde5c6511bb3df098d3627a1f18c`

This was the expected committed HEAD for the 036-MAJ1 repair.

## 2. Starting Git State

Recorded before any edit:

- `git rev-parse HEAD` -> `de4141afa59ffde5c6511bb3df098d3627a1f18c`
- `git status --short` -> `?? docs/audits/036-final-committed-pre-cuda-repository-review.md`
- `git diff --cached --name-status` -> empty
- `git diff --check` -> empty

Audit 036 was already untracked at the start and was not modified.

## 3. 036-MAJ1 Root Cause

`tests/test_stage1_training_resume.py` still used the obsolete miniature
validation-point schema:

- `{"update": ..., "score": ...}` appended by the synthetic training loop.
- `{"update": ..., "score": ...}` passed directly into `checkpoint_payload()`
  in the validation-history persistence test.

Current production no longer treats that as checkpoint state.
`checkpoint_payload()` canonicalizes every point through the current
`ValidationPoint` schema, and `ValidationPoint.from_dict()` requires
`update`, `distances`, and `d_clean`. `score` is derived from `distances`.

## 4. Stale Schema Occurrences Found

Repository search after the repair found no remaining same-defect occurrence in
`tests/test_stage1_training_resume.py`.

Other matches were not this defect:

- `tests/test_stage1_training_resume_state.py` mentions the old schema in
  explanatory prose and has an intentional malformed-point rejection case.
- `tests/test_stage1_cuda_resume_equivalence.py` uses a complete dictionary
  with `update`, `score`, `d_clean`, and the full `distances` grid.

No additional test file required modification.

## 5. Test Changes

Only `tests/test_stage1_training_resume.py` was changed:

- imported `VALIDATION_CONDITIONS`;
- imported `ValidationPoint`;
- added `point_at(update, score)` returning a real `ValidationPoint`;
- changed the miniature training loop to append `ValidationPoint` objects;
- changed resumed miniature state to restore loaded point dictionaries through
  `ValidationPoint.from_dict()`;
- changed `test_resume_restores_the_validation_history()` to pass real
  `ValidationPoint` objects into `checkpoint_payload()`;
- asserted the writer output has exactly `update`, `distances`, `d_clean`, and
  derived `score`;
- asserted production reader round-trip with
  `[ValidationPoint.from_dict(p) for p in payload["points"]]`.

No production file was modified.

## 6. Current Canonical ValidationPoint Schema

Production state fields:

- `update`;
- `distances`, with exactly every locked `VALIDATION_CONDITIONS` member;
- `d_clean`.

Serialized writer output:

- `update`;
- `distances`;
- `d_clean`;
- derived `score`.

Reader behavior:

- requires `update`, `distances`, and `d_clean`;
- refuses unknown keys;
- refuses missing or extra validation conditions;
- recomputes `score`;
- refuses a persisted `score` that contradicts recomputed distances.

## 7. Synthetic Distances and d_clean

The torch-gated miniature does not need a scientific validation score ordering;
its purpose is exact resume equivalence for adapter tensors, optimizer state,
sampler state, update count, and validation history.

The repaired helper maps the miniature's deterministic validation signal to
all locked validation conditions:

- `distances[c] = score` for every locked condition;
- `d_clean = score / 2.0`.

For loop-generated points, `score` is `abs(loss)` so the synthetic distance is
non-negative while remaining deterministically tied to the model trajectory.
For explicit fixture points, the score values remain the intended 1.0 and 2.0.

Because all condition distances are equal, `ValidationPoint.score` derives the
same synthetic value through production's `max(distances[c])` rule.

## 8. Derived Score

The repaired test no longer stores `score` as independent state. It constructs
`ValidationPoint(update, distances, d_clean)` and lets
`ValidationPoint.to_dict()` and `checkpoint_payload()` serialize the derived
score. The reader check then reconstructs from the serialized payload and
derives the score again.

This is the production contract the test is meant to guard.

## 9. Real Checkpoint / Resume Seam

The original runtime purpose is preserved:

- the miniature still uses a real torch module and real AdamW optimizer when
  torch is present;
- interruption still writes through `checkpoint_payload()` and
  `save_training_checkpoint()`;
- resume still loads through `load_training_checkpoint()`;
- provenance is still verified with `verify_checkpoint()`;
- adapter and optimizer state are still loaded from the checkpoint;
- sampler state is still restored from checkpoint state;
- validation history from the loaded checkpoint is now restored through
  `ValidationPoint.from_dict()` before the miniature continues.

The test should now fail if the stale schema is reintroduced: the persistence
test asserts the current writer schema and the interrupted-resume path feeds
loaded checkpoint point dictionaries through the production reader.

The older file remains a miniature tensor/optimizer equivalence test, not the
real `train_run` torch seam. The real `train_run` seam remains covered by
`tests/test_stage1_resume_state_machine_torch.py`.

## 10. Static Torch-Path Precondition Review

Local torch is absent, so this file still skips locally. Static trace with the
current source:

- imports before `pytest.importorskip("torch")` are torch-free;
- `point_at()` constructs a `ValidationPoint` with non-negative update, all
  locked validation conditions, no extra conditions, and finite numeric values;
- `checkpoint_payload()` accepts `ValidationPoint` objects via `_canonical_point`
  and writes `p.to_dict()`;
- `torch.save` / `torch.load` will carry the serialized dictionaries;
- resumed miniature training canonicalizes loaded dictionaries with
  `ValidationPoint.from_dict()` before appending new points;
- `ValidationPoint.from_dict()` sees required state fields, accepts the complete
  distance grid, recomputes score, and matches the persisted derived score.

No earlier constructor/schema precondition analogous to Audit 034's fixture
mistake is violated by this repair.

## 11. Local Focused Test Results

Final narrow schema/resume subset after the repair:

Command:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider -rs tests/test_stage1_training_resume.py tests/test_stage1_training_resume_state.py tests/test_stage1_resume_state_machine.py tests/test_stage1_resume_state_machine_torch.py`

Result:

- 55 passed
- 2 skipped
- 0 failed
- Runtime: 0.29s

Skip reasons:

- `tests/test_stage1_training_resume.py:42`: the tensor half needs torch; the
  torch-free half below always runs.
- `tests/test_stage1_resume_state_machine_torch.py:45`: the real train_run half
  needs torch; the torch-free half above always runs.

Audit-036 repaired-path focused suite:

Command:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider -rs tests/test_stage1_name_resolution.py tests/test_stage1_resume_state_machine.py tests/test_stage1_resume_state_machine_torch.py tests/test_stage1_artifact_identity.py tests/test_stage1_repository_provenance.py tests/test_stage1_training_resume_state.py tests/test_stage1_training_resume.py tests/test_stage1_run_independence.py tests/test_stage1_pretrain_audit.py tests/test_stage1_checkpoint.py tests/test_stage1_runner_cli_contract.py tests/test_stage1_final_freeze.py tests/test_stage1_schedule.py tests/test_stage1_corpus_verification.py tests/test_stage1_provenance_contract.py tests/test_stage1_inventory_preflight.py tests/test_stage1_parallel_preparation.py tests/test_stage1_device_contract.py`

Result:

- 406 passed
- 4 skipped
- 0 failed
- Runtime: 21.91s

Skip reasons:

- `tests/test_stage1_resume_state_machine_torch.py:45`: real train_run half
  needs torch.
- `tests/test_stage1_training_resume.py:42`: tensor half needs torch.
- `tests/test_stage1_name_resolution.py:80`: `unmark.stage1.objective` needs
  torch, absent in the ML-free venv.
- `tests/test_stage1_final_freeze.py:270`: H0 recomputation needs torch.

## 12. Full Lightweight Suite Result

The exact full-suite command was first attempted in the sandbox:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider`

Sandbox result:

- 3730 passed
- 105 skipped
- 7 failed
- Runtime: 116.17s

All 7 failures were in `tests/test_stage1_parallel.py` and were the known
sandbox restriction:

`PermissionError: [Errno 1] Operation not permitted`

The failing call was Python's multiprocessing forkserver binding a local Unix
socket. This is not a repository failure.

Required rerun outside the sandbox:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider`

Result:

- 3737 passed
- 105 skipped
- 0 failed
- Runtime: 133.14s

Skip-summary rerun outside the sandbox:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider -rs`

Result:

- 3737 passed
- 105 skipped
- 0 failed
- Runtime: 133.96s

The 105 skips are torch/CUDA or ML-free import skips. The skip-summary command
reported these skip reasons:

- `tests/test_stage1_cuda_resume_equivalence.py:24`: needs torch.
- `tests/test_stage1_device_contract_runtime.py:22`: the runtime half needs
  torch.
- `tests/test_stage1_resume_state_machine_torch.py:45`: the real train_run half
  needs torch; the torch-free half above always runs.
- `tests/test_stage1_run_independence_runtime.py:22`: the runtime half needs
  torch.
- `tests/test_stage1_training_resume.py:42`: the tensor half needs torch; the
  torch-free half below always runs.
- `tests/test_stage1_validation_measurement.py:22`: the injected-model half
  needs torch.
- `tests/test_b4b_provenance_and_positions.py`: adapter imports torch, and
  multiple Colab-gated tests report torch is not installed in the ML-free venv.
- `tests/test_evaluation_harness.py`: multiple Colab-gated tests report torch
  is not installed in the ML-free venv.
- `tests/test_neural_adapter.py`: multiple Colab-gated tests report torch is not
  installed in the ML-free venv.
- `tests/test_preg1_head.py`: multiple tests report torch is not installed
  locally.
- `tests/test_preg1_runner.py`: torch is not installed locally.
- `tests/test_stage1.py`: multiple Colab-gated tests report torch is not
  installed in the ML-free venv.
- `tests/test_stage1_final_freeze.py:270`: H0 recomputation needs torch.
- `tests/test_stage1_name_resolution.py:80`: `unmark.stage1.objective` needs
  torch, absent in the ML-free venv.
- `tests/test_stage1_torch_contracts.py`: multiple Colab-gated tests report
  torch is not installed in the ML-free venv.

## 13. Torch-Gated Tests Still Pending

Local skips are not CUDA evidence. The following Stage-1 torch/CUDA paths remain
pending authoritative CUDA execution:

- `tests/test_stage1_training_resume.py`;
- `tests/test_stage1_resume_state_machine_torch.py`;
- `tests/test_stage1_cuda_resume_equivalence.py`;
- `tests/test_stage1_device_contract_runtime.py`;
- `tests/test_stage1_run_independence_runtime.py`;
- `tests/test_stage1_validation_measurement.py`;
- `tests/test_stage1_torch_contracts.py`;
- H0 recomputation in `tests/test_stage1_final_freeze.py`;
- objective import/name-resolution torch path in
  `tests/test_stage1_name_resolution.py`;
- broader adapter/PhoBERT/evaluation torch tests skipped by the ML-free venv.

## 14. Production Byte-Identity

Before editing, production fingerprint for `unmark/`, `scripts/`, `configs/`,
and `requirements/`:

```text
66f6bc5c98fdfc570cb3c0f841becab4f5eff12e
644f43eec713618ec3bd12369fc6d0a4e6a917b1
7caa597f3d1d494a0396a22005e4244cd15be729
4a650d25652fb50c4aaf656e09c5ed301316fc7b
```

After the test repair, the same fingerprint:

```text
66f6bc5c98fdfc570cb3c0f841becab4f5eff12e
644f43eec713618ec3bd12369fc6d0a4e6a917b1
7caa597f3d1d494a0396a22005e4244cd15be729
4a650d25652fb50c4aaf656e09c5ed301316fc7b
```

`git diff --name-status -- unmark scripts configs requirements` was empty
after the repair.

Production content is byte-identical to the starting committed HEAD.

## 15. Scientific optimizer.step Status

No Stage-1 training command was run. `lr-pilot`, `r-phase1`, and `final-main`
were not invoked.

No real scientific optimizer step was intentionally executed. The edited file
contains a TEST-ONLY synthetic AdamW step in a torch-gated miniature fixture;
locally that module skipped because torch is absent.

## 16. Official UIT-VSFC TEST Status

SEALED / UNUSED.

No official UIT-VSFC TEST path was opened, inspected, read, evaluated, mounted,
screened, or passed to any command. No information was derived from official
TEST.

## 17. CUDA Tests Required Next

After committing Audits 036-037, the small authoritative CUDA acceptance should
run:

- the repaired focused suite with `tests/test_stage1_training_resume.py`
  executing instead of skipping;
- `tests/test_stage1_resume_state_machine_torch.py` with the real `train_run`
  resume path executing instead of skipping;
- `tests/test_stage1_cuda_resume_equivalence.py`;
- Stage-1 runtime/device/H0/frozen-encoder torch tests skipped locally;
- a no-update real PhoBERT/prepared-corpus smoke at the repair HEAD, with no
  optimizer, no backward, and no scientific optimizer step;
- repository-head assertion and clean-tree provenance checks on the CUDA host;
- artifact handoff validation using current campaign identity.

Do not run `lr-pilot`, `r-phase1`, `final-main`, or Stage-1 training as part of
this repair.

## 18. Final Verdict

036-MAJ1 is CLOSED in the test source.

No production code changed. The current production checkpoint/resume seam
remains exercised by the newer real-`train_run` torch fixture, and the older
torch-gated miniature now uses the canonical `ValidationPoint` contract instead
of the obsolete `{update, score}` schema.

STALE TORCH RESUME TEST SCHEMA REPAIR COMPLETE —
READY TO COMMIT AUDITS 036-037 AND RUN SMALL AUTHORITATIVE CUDA ACCEPTANCE
