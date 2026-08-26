# Audit 036 - Final Committed-Head Pre-CUDA Repository Review

## 1. REVIEW_HEAD

`de4141afa59ffde5c6511bb3df098d3627a1f18c`

This is the committed candidate reviewed for the imminent CUDA acceptance gate.

## 2. Starting Clean Git State

The review started clean.

Commands recorded at start:

- `git rev-parse HEAD` -> `de4141afa59ffde5c6511bb3df098d3627a1f18c`
- `git status --short` -> empty
- `git diff --cached --name-status` -> empty
- `git diff --check` -> empty
- `git log --oneline --decorate -15` -> HEAD was
  `de4141a (HEAD -> main, origin/main, origin/HEAD) fix: harden Stage 1 training pipeline`
- `git show --stat --oneline HEAD` -> the consolidated repair commit modified
  19 files with 5792 insertions and 46 deletions, including Audits 031-035.

`docs/audits/036-final-committed-pre-cuda-repository-review.md` did not exist
before this review.

## 3. Environment

Only the repository-local Python was used.

- `pwd` -> `/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft`
- `test -x .venv/bin/python` -> executable
- `sys.executable` -> `/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv/bin/python`
- `sys.prefix` -> `/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv`
- `.venv/bin/python -m pip --version` ->
  `pip 26.2.1 from .../.venv/lib64/python3.14/site-packages/pip (python 3.14)`

No dependency was installed or upgraded. Torch and transformers remained absent
from the local venv where expected.

## 4. Repair Delta Reviewed

Mechanical delta from historical pre-repair documentation HEAD
`55aa4064780b37626bcae7eef83c504a96fcc51f` to REVIEW_HEAD:

- Added Audits 031-035.
- Added `unmark/stage1/artifact.py`.
- Modified `scripts/stage1_runner.py`.
- Modified `unmark/stage1/checkpoint.py`,
  `unmark/stage1/execute.py`, `unmark/stage1/selection.py`,
  and `unmark/stage1/trainer.py`.
- Added/modified focused tests for name resolution, resume state, artifact
  identity, repository provenance, run independence, pretrain audit, and
  training resume state.

No production change was made in `configs/` or `requirements/`.

The production diff was read in full context, with targeted follow-up through
the changed tests and the unchanged Stage-1 execution dependencies.

## 5. Proposal / Decisions / Freeze / Code / Tests Consistency

The frozen scientific contract remains consistent across the proposal,
decisions, final freeze JSON, protocol code, and final-freeze tests:

- Backbone: `vinai/phobert-base`
- Revision: `01daacda68afe13d83023d16ec647239e344a1e6`
- Hidden size: 768
- Precision: fp32
- Adapter trainable parameters: 3,551,232
- `MAX_LENGTH`: 256
- Overflow: FAIL
- Truncation: disabled
- Batch: 128
- Gradient accumulation: 1
- Validation/checkpoint cadence: every 500 updates
- Initial budget: 20,000
- Hard stop: 40,000
- Continuation: at most one 20k->40k continuation
- `PI_STRIP`: 0.25
- Training corruption seed: 35422
- Validation corruption seed: 19225
- Document split seed: 51733
- Selection seed: 21230
- Final seeds: 36930, 7309, 5993
- LR grid: 1e-4, 3e-4, 1e-3
- r grid: 0.25, 0.5, 1.0, 2.0, 4.0
- Nominal runs: 11
- Optimizer: AdamW, betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
  constant LR, no warmup, no clipping, decay=0.01 on weights and 0.0 on
  exempt parameters.
- Official UIT-VSFC TEST: SEALED / UNUSED.

No scientific drift was found.

## 6. Complete Pipeline

Current pipeline reconstructed from source:

Pinned UVW shards -> source verification -> fixed train/validation/test shard
concatenation -> exact/canonical contamination screen against only legitimately
opened UIT-VSFC derived train and official validation -> document split before
chunking -> chunking with overflow fail -> prepared Stage-6 corpus ->
`COMPLETE.json` manifest/hash verification -> current repository identity ->
`CampaignIdentity` -> Stage-1 CLI -> scientific inventory preflight -> CUDA and
fp32 numerical policy -> frozen PhoBERT -> fresh adapter per nominal run ->
fresh `Stage1Objective` -> fresh optimizer -> fresh deterministic sampler ->
corruption/preparation -> forward/backward -> optimizer step -> validation ->
checkpoint -> resume -> 20k budget decision -> optional 40k continuation ->
`Candidate` -> LR selection -> r selection -> final-main artifacts.

The source owns the current behavior. The audits were used only as historical
defect context.

## 7. B1 Verdict - Objective Construction

Production verdict: CLOSED.

`execute_stage()` imports `Stage1Objective` lazily, constructs it inside the
nominal-run loop, and binds it to that run's fresh `UnmarkEncoder` and adapter.
There is no unresolved `objective_cls` on the real training path.

Remaining `objective_cls` occurrences are limited to no-update smoke/diagnostic
helpers and historical test prose. They do not drive `lr-pilot`, `r-phase1`, or
`final-main`.

## 8. B2 Verdict - ValidationPoint Writer / Reader

Production verdict: CLOSED.

The writer emits:

- `update`
- `distances`
- `d_clean`
- derived `score`

The reader accepts that schema through `ValidationPoint.from_dict`, rejects
missing state fields, rejects unknown fields, rejects malformed distance grids,
and refuses a persisted score that contradicts recomputed distances.

The `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` check is not material: persisted
`score` is discarded and recomputed from `distances`, so the tolerance affects
only the corruption tripwire.

## 9. B3 Verdict - 20k / 40k State Machine

Production verdict: CLOSED.

`resume_cap()` and `require_resumable_leg()` reconstruct and validate the
persisted leg from checkpoint state. Legal states are restricted to:

- initial-leg checkpoints with cap 20,000 and `0 <= global_update <= 20,000`
- continuation checkpoints with cap 40,000 and `20,000 < global_update <= 40,000`
- the in-process 20k->40k promotion from an exact completed initial leg

The reader refuses invalid caps, non-integer caps, `global_update > cap`,
40k-cap states at or below 20k, cap lowering, and attempted 60k/80k successors.

The production writer's reachable states are accepted; impossible states are
not silently relabeled.

## 10. B4 Verdict - Artifact Handoff

Production verdict: CLOSED.

Downstream stages now validate artifacts against a `CampaignIdentity` derived
from current trusted inputs: current Git HEAD, verified corpus membership
digest, pinned backbone, pinned revision, fp32 precision, and verified
inventory identity. The incoming artifact does not define the expected values.

Selection is recomputed with production `select_learning_rate()` or `select_r()`
before a value is returned to the caller. Wrong stage, wrong protocol, wrong
HEAD, wrong corpus, wrong backbone, wrong revision, wrong precision, wrong
inventory, missing identity, missing candidate, duplicate candidate, extra
candidate, off-grid value, edited selected scalar, and evidence edits that
change the winner are refused.

## 11. B5 Verdict - Git / Provenance

Production verdict: CLOSED.

`resolve_repository_head()` derives the actual full 40-character SHA from Git.
`resolve_asserted_repository_head()` treats `--repository-head` only as an
assertion, not an override. Omitted assertion records the actual HEAD, false
assertions refuse, abbreviations/branch names refuse, Git failure refuses, and
tracked dirty execution code in `unmark/`, `scripts/`, `configs/`, or
`requirements/` refuses a scientific run.

Untracked/ignored runtime artifacts are ignored by the clean-tree check.

Tracked paths outside the execution-relevant scope are docs, tests, results
placeholders, README, proposal artifacts, pyproject test configuration, and
historical diagnostic result zip. None changes Stage-1 scientific runtime under
current source.

## 12. Audit-035 Fixture Verdict

Audit-035 fixture verdict: REPAIRED IN SOURCE, CUDA EXECUTION STILL PENDING.

`tests/test_stage1_resume_state_machine_torch.py` now constructs a tiny
Roberta-like frozen backbone satisfying the real production preconditions:
checkpoint identity, model type, class name, position profile, padding index,
hidden size, real frozen parameters, eval behavior, and input-embedding
interface.

The tiny hidden size is correct for the intended test: it exercises the real
resume chain without tripping the locked 768-dimensional parameter-count gate,
which is not the behavior under test.

The fixture still traverses the real chain:

`ValidationPoint` -> `checkpoint_payload` -> `save_training_checkpoint` ->
`load_training_checkpoint` -> real `train_run` -> `verify_checkpoint` ->
`require_resumable_leg` -> strict adapter restore -> optimizer restore ->
optimizer parameter identity -> optimizer state device -> sampler restore ->
`ValidationPoint.from_dict` -> budget resolution.

Local execution skipped this file because torch is absent.

## 13. Checkpoint Writer / Reader Verdict

Production verdict: CLOSED.

There is one production `ValidationPoint` reconstruction site in the training
resume path, and it uses `ValidationPoint.from_dict`. Candidate artifact
reconstruction also uses `ValidationPoint.from_dict`.

Checkpoint payloads include schema version, provenance, adapter state,
optimizer state, global update, sampler state, cap, validation points, and
execution fingerprint. The writer canonicalizes points through the reader's
schema before serializing.

## 14. 20k / 40k State-Machine Verdict

Production verdict: CLOSED.

Fresh state starts at cap 20,000. A checkpoint at update 500 remains initial
leg. A completed 20,000/20,000 state may resume on the same leg or legally
promote to 40,000. A 20,500/40,000 checkpoint resumes as 40,000 and cannot be
lowered to 20,000. A 40,000/40,000 state stops and may be budget-limited by
selection. Third continuations are structurally unreachable.

`continued_past_initial_budget` remains equivalent to `cap > INITIAL_MAX_UPDATES`
on the result.

## 15. Multi-Run Resume Verdict

Production verdict: CLOSED IN SOURCE; CUDA execution pending.

The A/B/C scenario is correct by construction:

- A completed run is restored from its own namespace and not retrained.
- B crashed during the 40k leg resumes under cap 40,000.
- C with no checkpoint starts fresh under cap 20,000.
- Run checkpoint directories are label-local.
- Adapters, objectives, optimizers, samplers, points, and results are run-local.
- Selection receives one `Candidate` per planned nominal run.

## 16. Artifact Identity / Handoff Verdict

Production verdict: CLOSED.

`CampaignIdentity` covers repository head, protocol, corpus digest, encoder
checkpoint, encoder revision, precision, inventory source name, inventory
revision, and inventory SHA-256. It is built from current verified state and
compared field-by-field against the artifact.

The protocol does not require hostile-file cryptographic authenticity; artifact
validation is consistency and campaign-identity validation. That boundary is
appropriate for the current experiment.

## 17. Repository Provenance Verdict

Production verdict: CLOSED.

Stage-1 scientific execution records actual committed HEAD and requires tracked
execution-relevant files to match HEAD. `--repository-head` cannot create a
false provenance claim.

Prepared-corpus `CheckpointIdentity.repository_head` remains the identity of
the Stage-6 producer, while Stage-1 run provenance records the current
scientific execution HEAD and verified corpus digest. That is not a conflict.

## 18. Frozen Encoder / Model / Objective Verdict

Production verdict: CLOSED.

PhoBERT remains frozen. `UnmarkEncoder` freezes all encoder parameters,
restores encoder eval after train-mode changes, checks encoder eval at forward
boundaries, derives authoritative position ids for `inputs_embeds`, and rejects
unmeasured position semantics. `Stage1Objective` uses the frozen encoder under
`no_grad` only for the reference branch and leaves adapted branches
differentiable to the adapter.

The full encoder state hash is monitored across nominal runs.

## 19. Optimizer / Randomness / Device Verdict

Production verdict: CLOSED IN SOURCE; CUDA numerical execution pending.

The optimizer is built from trainable adapter parameters only. Parameter object
identity is checked after construction and after resume. Optimizer state device
is checked recursively, allowing CPU scalar Adam `step` tensors as intended.

Adapter initialization is deterministic, CPU-local, domain-separated by
`adapter_init_seed(run_seed)`, isolated from ambient CPU/CUDA RNG, and recomputed
fresh per nominal run. Sampler and corruption streams are deterministic and
resume-bound.

Scientific training requires CUDA and enforces deterministic fp32 policy. Local
evidence is static because the local venv has no torch/CUDA.

## 20. Data / Stage-6 Verdict

Production verdict: CLOSED.

The repair did not alter dataset identity/revision, source shard hashes,
concatenation order, contamination policy, split seed, chunking execution,
membership digest, COMPLETE verification, prepared corpus identity, inventory
pin, or TEST sealing.

Known minor documentation issue remains: manifest `chunking.algorithm` still
uses the old label `deterministic_whitespace_boundary` even though current
chunking includes safe-offset logic. Counts, lengths, hashes, split, and
overflow guards are authoritative; this does not block CUDA or training.

## 21. TEST Sealing

Official UIT-VSFC TEST status: SEALED / UNUSED.

No command in this review opened, inspected, read, evaluated, mounted, screened,
or passed an official TEST path. No information was derived from official TEST.
Stage-1 CLI exposes no official TEST route.

## 22. Static Bug Hunt

Searches covered unresolved globals, stale names, `objective_cls`,
`Stage1Objective`, `build_objective`, `ValidationPoint` reconstruction,
checkpoint writer/reader paths, cap/resume functions, artifact handoff,
repository provenance helpers, `strict=False`, broad exceptions,
`optimizer.step`, TODO/FIXME/HACK/XXX, official TEST references, and duplicated
scientific constants.

No remaining production BLOCKER or MAJOR was found.

One MAJOR test-suite defect was found:

**036-MAJ1 - `tests/test_stage1_training_resume.py` is still written against
the pre-repair validation-point schema and will fail when torch is installed.**

Concrete evidence:

- `tests/test_stage1_training_resume.py:105` appends bare
  `{"update": update, "score": ...}` points.
- `tests/test_stage1_training_resume.py:203` calls `checkpoint_payload()` with
  `{"update", "score"}` dictionaries.
- `unmark/stage1/trainer.py:433` now canonicalizes every point through
  `_canonical_point(p).to_dict()`.
- `unmark/stage1/selection.py:109-114` requires `distances` and `d_clean`.
- A local `.venv/bin/python -B -c ...` reproduction raised:
  `SelectionViolation: validation point is missing required field(s)
  ['distances', 'd_clean']; got keys ['score', 'update']`.

This file is part of the requested focused suite but skips locally because
torch is absent. On the authoritative CUDA host, it should execute and fail.
That is a material acceptance/evidence defect even though the production
writer/reader behavior is correct.

## 23. Cross-File Conflict Review

No production/spec/freeze scientific conflict was found.

Cross-file test conflict: `tests/test_stage1_training_resume.py` still asserts
the old checkpoint point schema while the repaired writer/reader and newer
torch-free tests correctly require the canonical `ValidationPoint` schema.

Documentation-only drift remains minor: README still describes an earlier
project phase, `requirements/experiment.txt` carries a future G1 environment
split TODO, and Audit 034's operational clarification notes are still not
rolled into a decisions/change-log entry.

## 24. Test-Quality Review

B1 is now covered by real global-name resolution, not just source-string AST
checks.

B2 is covered by real writer/reader schema tests in the torch-free suite and by
the repaired real-`train_run` torch fixture when torch is available.

B3 is covered by torch-free state-machine tests and by the repaired
real-`train_run` torch fixture when torch is available.

B4 is covered by artifact identity, locked-grid, and reselection tests.

B5 is covered by real Git read-only integration and monkeypatched failure
cases.

Material quality gap: the older torch-gated `test_stage1_training_resume.py`
still tests its own obsolete miniature schema and will not run locally. This is
036-MAJ1.

## 25. Focused Test Results

Command:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider -rs tests/test_stage1_name_resolution.py tests/test_stage1_resume_state_machine.py tests/test_stage1_resume_state_machine_torch.py tests/test_stage1_artifact_identity.py tests/test_stage1_repository_provenance.py tests/test_stage1_training_resume_state.py tests/test_stage1_training_resume.py tests/test_stage1_run_independence.py tests/test_stage1_pretrain_audit.py tests/test_stage1_checkpoint.py tests/test_stage1_runner_cli_contract.py tests/test_stage1_final_freeze.py tests/test_stage1_schedule.py tests/test_stage1_corpus_verification.py tests/test_stage1_provenance_contract.py tests/test_stage1_inventory_preflight.py tests/test_stage1_parallel_preparation.py tests/test_stage1_device_contract.py`

Result:

- 406 passed
- 4 skipped
- 0 failed
- Runtime: 21.84s

Skip reasons:

- `tests/test_stage1_resume_state_machine_torch.py:45`: real train_run half
  needs torch.
- `tests/test_stage1_training_resume.py:40`: tensor half needs torch.
- `tests/test_stage1_name_resolution.py:80`: `unmark.stage1.objective` needs
  torch.
- `tests/test_stage1_final_freeze.py:270`: H0 recomputation needs torch.

The skip of `tests/test_stage1_training_resume.py` is the local mechanism that
hides 036-MAJ1.

## 26. Full-Suite Results

Initial sandboxed full-suite command:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider`

Result: 7 failed, 3730 passed, 105 skipped in 116.08s.

All 7 failures were `tests/test_stage1_parallel.py` multiprocessing cases
failing with `PermissionError: [Errno 1] Operation not permitted` while Python's
forkserver tried to bind a local Unix socket. This is a sandbox execution
restriction, not a repository failure.

Required rerun outside the sandbox:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider`

Result:

- 3737 passed
- 105 skipped
- 0 failed
- Runtime: 132.85s

Skip-summary rerun:

`.venv/bin/python -B -m pytest -q -p no:cacheprovider -rs`

Result:

- 3737 passed
- 105 skipped
- 0 failed
- Runtime: 131.01s

Skip classes were torch/CUDA-gated runtime tests, H0 torch recomputation, and
one import-resolution skip for `unmark.stage1.objective` in the ML-free venv.

## 27. Torch / CUDA Limitations

Local torch/CUDA evidence is intentionally absent. The local skips are not
passes. The repaired real resume fixture and other torch-gated runtime tests
must run on the authoritative CUDA host after 036-MAJ1 is repaired.

The local review did not install torch, transformers, or any dependency.

## 28. Historical CUDA Evidence Validity Matrix

| Evidence item | Classification | Reason |
|---|---|---|
| runtime fingerprint | PARTIALLY VALID | Mechanism unchanged, but current HEAD/provenance must be observed once on the acceptance host. |
| deterministic CUDA/fp32 | STILL VALID | Source policy unchanged; still confirm on host before training. |
| H0 | STILL VALID | Initialization machinery unchanged; torch recomputation remains pending locally. |
| frozen encoder | STILL VALID | Wrapper/objective freeze mechanics unchanged; acceptance should confirm real model no-update. |
| real model/data no-update probe | PARTIALLY VALID | Needs cheap confirmation at REVIEW_HEAD because provenance handling changed. |
| checkpoint/resume | MUST RE-RUN | Resume repairs and fixture repair require CUDA/torch execution. |
| 20k->40k continuation | MUST RE-RUN | State-machine repair must be exercised on the torch path. |
| artifact handoff | MUST RE-RUN | Handoff logic is new and must be confirmed in the final acceptance flow. |
| repository-head provenance | MUST RE-RUN | Derived-head/clean-tree behavior is new. |
| spawn preparation | STILL VALID | Production Stage-1 preparation pool remains explicit spawn and unchanged by this repair. |
| performance benchmark | STILL VALID | No preparation-performance code change requiring a new benchmark was found. |

No Stage-6 reconstruction or new performance benchmark is required unless a
later repair changes those paths.

## 29. Exact Small CUDA Acceptance Items Required Next

Because 036-MAJ1 is present, do not run CUDA acceptance or Stage-1 training
until that test defect is repaired and committed.

After repair, the small authoritative CUDA acceptance should run:

- the repaired focused suite, including `tests/test_stage1_training_resume.py`;
- `tests/test_stage1_resume_state_machine_torch.py` with all torch tests
  executing rather than skipping;
- `tests/test_stage1_cuda_resume_equivalence.py`;
- Stage-1 runtime/device/H0/frozen-encoder torch tests that are skipped locally;
- a no-update real PhoBERT/prepared-corpus smoke at REVIEW_HEAD or successor
  repair HEAD, with no optimizer, no backward, and no scientific step;
- repository-head assertion and clean-tree provenance checks on the CUDA host;
- artifact handoff validation using current campaign identity.

Do not run `lr-pilot`, `r-phase1`, or `final-main` as part of this acceptance.

## 30. BLOCKER Count

0

## 31. MAJOR Count

1

## 32. MINOR Count

3

Carried non-blocking minors:

- Audit 034 operational clarifications not yet recorded in decisions/change log:
  assertion-only `--repository-head`, clean-tree scope, and `CampaignIdentity`
  field set.
- Audit 034 carried pre-existing minor group: historical HEAD in freeze file,
  stale README, harmless `budget_limited` checkpoint-key omission, and
  `load_prepared_chunks` routing non-`train` partitions to dev.
- Audit 034 minor: unused `manifest = verified.manifest` assignments in
  `run_r_phase1` and `run_final_main`.

## 33. INFORMATIONAL Count

4

Carried informational observations:

- `ValidationPoint.from_dict` uses `math.isclose` for derived score consistency.
- Artifact validation checks selection/campaign consistency, not hostile-file
  authenticity for losing evidence.
- `pyproject.toml` is tracked but outside the clean-tree scope; currently only
  pytest configuration.
- Torch is deliberately unpinned and supplied by Colab.

## 34. Every BLOCKER / MAJOR

BLOCKER: none.

MAJOR:

**036-MAJ1 - stale torch-gated training-resume test schema.**

`tests/test_stage1_training_resume.py` still writes bare `{update, score}`
validation points in its tiny torch checkpoint fixture, while the repaired
production writer now validates every point through `ValidationPoint.from_dict`,
which requires `distances` and `d_clean`. The file skips locally without torch,
but it is in the requested focused suite and should fail when the authoritative
CUDA environment installs torch. This should block the imminent CUDA acceptance
and training until repaired.

## 35. Scientific optimizer.step Status

No Stage-1 training command was run. `lr-pilot`, `r-phase1`, and `final-main`
were not invoked.

No intentional real scientific `optimizer.step()` was executed. Any
`optimizer.step` references encountered were production code not executed or
TEST-ONLY synthetic fixtures. In this local venv, torch-gated synthetic step
tests skipped.

## 36. Official UIT-VSFC TEST Status

SEALED / UNUSED.

The official TEST split was not opened, inspected, read, evaluated, mounted,
screened, or passed to any command. No information was derived from it.

## 37. Final Verdict

FINAL COMMITTED-HEAD PRE-CUDA REVIEW BLOCKED — DO NOT RUN CUDA OR TRAIN

Reason: one MAJOR test/evidence defect remains (`036-MAJ1`). Production B1-B5
repairs are closed in source, but the imminent authoritative CUDA acceptance
would include a torch-gated file that is still inconsistent with the repaired
checkpoint schema.
